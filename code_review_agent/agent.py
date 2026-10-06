"""Agent 核心：实现「输入 → 推理 → 工具调用 → 输出」的循环。

设计要点
--------
1. **事件驱动**：``ReviewAgent.stream()`` 是一个生成器，逐步产出
   ``AgentEvent``。CLI/Web 只需消费事件即可渲染进度，Agent 本身不关心 UI。
2. **无状态工具、有状态记忆**：工具是纯函数（输入参数 → 文本结果），
   会话状态集中在 :class:`~code_review_agent.memory.ConversationMemory` 中。
3. **错误分级处理**：
   - LLM 配置类错误（Key 无效）→ 直接终止，重试无意义；
   - LLM 可恢复错误（限流/超时）→ 由 LLMClient 指数退避重试；
   - 工具执行错误 → 转成文本回喂给模型，让它自我纠正；
   - 达到步数上限 → 抛 :class:`StepLimitExceeded`，由调用方决定收尾策略。
4. **终止条件**：模型不再请求工具调用即视为完成；或它显式调用
   ``finish_review`` 主动交付结论。
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Literal

from .config import Settings
from .llm import ChatMessage, LLMClient, LLMConfigError, ToolCall
from .memory import ConversationMemory, build_llm_summarizer
from .prompts import (
    CHAT_SYSTEM_PROMPT,
    SUMMARY_PROMPT,
    build_review_prompt,
    build_system_prompt,
)
from .tools import ToolContext, ToolRegistry, build_default_registry

logger = logging.getLogger(__name__)

EventType = Literal["start", "assistant", "tool_call", "tool_result", "final", "limit", "error"]


@dataclass
class AgentEvent:
    """Agent 运行过程中的一次可观察事件。

    ``data`` 用于携带结构化附注（工具返回的 meta、退出码等），
    供 CLI / Web 等不同消费者按需使用；纯文本消费者可以忽略它。
    """

    type: EventType
    text: str = ""
    tool: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    ok: bool = True
    step: int = 0
    elapsed: float = 0.0
    data: dict[str, Any] = field(default_factory=dict)


class AgentError(RuntimeError):
    """Agent 运行失败。"""


class StepLimitExceeded(AgentError):
    """超过最大步数仍未收敛。"""

    def __init__(self, message: str, *, messages: list[ChatMessage] | None = None) -> None:
        super().__init__(message)
        self.messages = messages or []


class ReviewAgent:
    """代码审查 Agent。"""

    def __init__(
        self,
        settings: Settings,
        *,
        client: LLMClient | None = None,
        registry: ToolRegistry | None = None,
        workspace: str | Path | None = None,
        with_few_shot: bool = True,
        system_prompt: str | None = None,
    ) -> None:
        self.settings = settings
        self.workspace = Path(workspace or settings.workspace).resolve()
        if not self.workspace.is_dir():
            raise AgentError(f"工作区不存在或不是目录：{self.workspace}")

        self.client = client or LLMClient(
            model=settings.model,
            api_key=settings.api_key,
            base_url=settings.base_url,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            timeout=settings.timeout,
            max_retries=settings.max_retries,
        )

        self.tool_context = ToolContext.create(
            self.workspace, max_file_bytes=settings.max_file_bytes
        )
        self.registry = registry or build_default_registry(self.tool_context)

        prompt = system_prompt or build_system_prompt(with_example=with_few_shot)
        self.memory = ConversationMemory(
            prompt,
            window=settings.memory_window,
            summarizer=build_llm_summarizer(self.client, SUMMARY_PROMPT),
        )
        self.steps_taken = 0

    # ------------------------------------------------------------------ 入口 --
    def review(
        self,
        *,
        path: str | None = None,
        focus: str | None = None,
        language_hint: str | None = None,
        report_path: str | None = None,
        save_report: bool = False,
    ) -> Iterator[AgentEvent]:
        """执行一次代码审查，逐步产出事件。"""
        prompt = build_review_prompt(
            workspace=str(self.workspace),
            path=path,
            focus=focus,
            language_hint=language_hint,
        )
        if save_report:
            prompt += (
                f"\n4. 请调用 write_report 把最终报告保存为 `{report_path or 'review_report.md'}`。"
            )
        yield from self.run(prompt)

    def chat_turn(self, user_input: str) -> Iterator[AgentEvent]:
        """多轮问答：在已有记忆上追加一次用户输入。"""
        if len(self.memory) == 0:
            self.memory.set_system_prompt(CHAT_SYSTEM_PROMPT)
        yield from self.run(user_input, keep_history=True)

    # ------------------------------------------------------------------ 主循环 --
    def run(self, user_input: str, *, keep_history: bool = False) -> Iterator[AgentEvent]:
        if not keep_history:
            self.memory.reset()

        self.memory.add(ChatMessage(role="user", content=user_input))
        yield AgentEvent(
            type="start",
            text=f"已接收任务，工作区：{self.workspace}｜工具：{', '.join(self.registry.names())}",
            data={"tools": self.registry.names()},
        )

        self.steps_taken = 0
        for step in range(1, self.settings.max_steps + 1):
            self.steps_taken = step
            started = time.perf_counter()

            # ---------------- 推理：调用 LLM ----------------
            try:
                reply = self.client.chat(self.memory.to_messages(), tools=self.registry.specs())
            except LLMConfigError as exc:
                yield AgentEvent(type="error", text=str(exc), ok=False, step=step)
                raise
            except Exception as exc:  # noqa: BLE001 - 统一转为事件回报
                yield AgentEvent(
                    type="error",
                    text=f"LLM 调用失败：{exc}",
                    ok=False,
                    step=step,
                )
                raise AgentError(str(exc)) from exc

            elapsed = time.perf_counter() - started
            self.memory.add(reply)

            if reply.content:
                yield AgentEvent(type="assistant", text=reply.content, step=step, elapsed=elapsed)

            # ---------------- 无工具调用 → 收敛，输出最终结果 ----------------
            if not reply.tool_calls:
                yield AgentEvent(
                    type="final",
                    text=reply.content or "（模型未返回内容）",
                    step=step,
                    elapsed=elapsed,
                )
                return

            # ---------------- 工具调用 ----------------
            # 用列表而非布尔量：生成器内部无法向外部作用域赋值，
            # 只能借助可变容器把「任务已交付」的信号传回主循环
            finished: list[AgentEvent] = []
            for call in reply.tool_calls:
                yield from self._execute_tool(call, step=step, finished=finished)
                if finished:
                    break
            if finished:
                # 模型显式调用 finish_review → 主动交付结论，结束本次任务
                yield finished[0]
                return

        # ---------------- 步数耗尽 ----------------
        message = (
            f"已达到最大步数 {self.settings.max_steps}，Agent 未能收敛。"
            "可通过 AGENT_MAX_STEPS 环境变量或 --max-steps 参数调大上限。"
        )
        yield AgentEvent(type="limit", text=message, ok=False, step=self.steps_taken)
        raise StepLimitExceeded(message, messages=self.memory.to_messages())

    # ------------------------------------------------------------------ 工具执行 --
    def _execute_tool(
        self,
        call: ToolCall,
        *,
        step: int,
        finished: list[AgentEvent],
    ) -> Iterator[AgentEvent]:
        name = call.function.name
        raw_args = call.function.arguments
        try:
            arguments = json.loads(raw_args) if raw_args and raw_args.strip() else {}
        except json.JSONDecodeError:
            arguments = {}

        yield AgentEvent(
            type="tool_call", tool=name, arguments=arguments, step=step, text=raw_args
        )

        result = self.registry.execute(name, raw_args or "{}")

        self.memory.add(
            ChatMessage(
                role="tool",
                content=result.content,
                tool_call_id=call.id,
                name=name,
            )
        )

        yield AgentEvent(
            type="tool_result",
            tool=name,
            text=result.content,
            ok=result.ok,
            step=step,
            data=result.meta,
        )

        # 主动终止：模型调用 finish_review 即代表交付完成
        if name == "finish_review" and result.ok:
            finished.append(AgentEvent(type="final", text=result.content, step=step))

    # ------------------------------------------------------------------ 收尾 --
    def force_conclusion(self) -> str:
        """步数耗尽后的兜底：让模型基于已有上下文直接给结论（禁止再调工具）。"""
        self.memory.add(
            ChatMessage(
                role="user",
                content=(
                    "已达到工具调用步数上限，请立即基于**已经获取到的信息**输出最终审查报告，"
                    "不要再请求调用任何工具。对于证据不足的部分，请明确标注「未能验证」。"
                ),
            )
        )
        reply = self.client.chat(self.memory.to_messages())
        self.memory.add(reply)
        return reply.content or "（模型未能给出结论）"

    # ------------------------------------------------------------------ 统计 --
    @property
    def usage(self) -> dict[str, int]:
        usage = self.client.usage
        return {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        }

    def stats(self) -> dict[str, Any]:
        return {
            "steps": self.steps_taken,
            "tool_calls": len(self.registry.call_log),
            "tools_used": _count_tools(self.registry.call_log),
            "usage": self.usage,
        }


def _count_tools(call_log: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in call_log:
        counts[entry["tool"]] = counts.get(entry["tool"], 0) + 1
    return counts
