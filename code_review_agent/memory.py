"""上下文记忆管理。

策略（两级，够用且易于解释）：
1. **滑窗**：始终保留 system 消息 + 最近 N 条消息；
2. **摘要压缩**：当窗口内消息数超过阈值时，把「即将被丢弃」的旧消息交给 LLM
   压缩成一段摘要，作为一条 assistant 消息插在 system 之后。

这样既能提供长期上下文，又不会让 token 无界增长。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Sequence

from .llm import ChatMessage

if TYPE_CHECKING:  # pragma: no cover
    from .llm import LLMClient

logger = logging.getLogger(__name__)


def estimate_tokens(text: str) -> int:
    """粗略估算 token 数。

    不对中英文分别建模，只用一个保守系数：中文约 1 字 1 token，
    英文约 4 字符 1 token，取 2.5 字符/token 作为折中。
    """
    if not text:
        return 0
    return max(1, int(len(text) / 2.5))


class ConversationMemory:
    """维护一个可裁剪、可摘要的消息列表。"""

    def __init__(
        self,
        system_prompt: str,
        *,
        window: int = 24,
        summarizer: Callable[[Sequence[ChatMessage]], str] | None = None,
        summary_trigger: int | None = None,
    ) -> None:
        self._system = ChatMessage(role="system", content=system_prompt)
        self._window = max(4, window)
        # 默认在窗口的 1.5 倍时触发压缩，避免每轮都调用摘要模型
        self._summary_trigger = summary_trigger or int(self._window * 1.5)
        self._summarizer = summarizer
        self._messages: list[ChatMessage] = []
        self._summary: str | None = None

    # ------------------------------------------------------------- 读接口 --
    @property
    def system_prompt(self) -> str:
        return self._system.content

    def set_system_prompt(self, prompt: str) -> None:
        self._system = ChatMessage(role="system", content=prompt)

    def to_messages(self) -> list[ChatMessage]:
        """返回可直接发给 LLM 的消息序列（含 system 与摘要）。"""
        result: list[ChatMessage] = [self._system]
        if self._summary:
            result.append(
                ChatMessage(
                    role="assistant",
                    content=f"[早前对话摘要] {self._summary}",
                )
            )
        result.extend(self._messages)
        return result

    def recent(self, n: int = 6) -> list[ChatMessage]:
        return self._messages[-n:]

    def __len__(self) -> int:
        return len(self._messages)

    def token_estimate(self) -> int:
        return sum(estimate_tokens(m.content or "") for m in self.to_messages())

    # ------------------------------------------------------------- 写接口 --
    def add(self, message: ChatMessage) -> None:
        self._messages.append(message)
        self._maybe_compact()

    def extend(self, messages: Sequence[ChatMessage]) -> None:
        self._messages.extend(messages)
        self._maybe_compact()

    def reset(self, *, keep_summary: bool = False) -> None:
        self._messages.clear()
        if not keep_summary:
            self._summary = None

    # ------------------------------------------------------------- 内部 --
    def _maybe_compact(self) -> None:
        if len(self._messages) <= self._summary_trigger:
            return

        keep = self._window
        # 保证裁剪边界落在「完整的一轮」上：不要以 tool 消息开头，
        # 否则后续请求会因为缺少配对的 assistant.tool_calls 而被服务端拒绝。
        cut = len(self._messages) - keep
        while cut < len(self._messages) and self._messages[cut].role == "tool":
            cut += 1

        if cut <= 0:
            return

        dropped = self._messages[:cut]
        self._messages = self._messages[cut:]

        if self._summarizer is None:
            logger.debug("无摘要器，直接丢弃 %d 条历史消息", len(dropped))
            return

        try:
            new_summary = self._summarizer(dropped)
        except Exception as exc:  # noqa: BLE001 - 摘要失败不能影响主流程
            logger.warning("历史摘要失败，保留旧摘要：%s", exc)
            return

        self._summary = (
            f"{self._summary}\n{new_summary}".strip() if self._summary else new_summary
        )
        logger.debug("已压缩 %d 条历史消息为摘要", len(dropped))


def build_llm_summarizer(client: "LLMClient", template: str) -> Callable[[Sequence[ChatMessage]], str]:
    """构造一个基于 LLM 的摘要函数（供 ConversationMemory 使用）。"""

    def _summarize(history: Sequence[ChatMessage]) -> str:
        lines: list[str] = []
        for msg in history:
            if msg.role == "tool":
                body = (msg.content or "")[:400]
                lines.append(f"[工具结果 {msg.name or ''}] {body}")
            elif msg.role == "assistant" and msg.tool_calls:
                names = ", ".join(tc.function.name for tc in msg.tool_calls)
                lines.append(f"[助手调用工具] {names}")
            else:
                lines.append(f"[{msg.role}] {(msg.content or '')[:400]}")

        prompt = template.format(history="\n".join(lines))
        reply = client.chat([ChatMessage(role="user", content=prompt)])
        return (reply.content or "").strip()

    return _summarize
