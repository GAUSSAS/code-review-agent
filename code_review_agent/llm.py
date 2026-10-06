"""LLM 客户端：封装 OpenAI 兼容接口（DeepSeek / OpenAI / 通义 / 本地 vLLM 等）。

设计要点：
1. 只依赖 ``openai`` SDK，通过 ``base_url`` 适配任意 OpenAI 兼容服务；
2. 内置**指数退避 + 抖动**重试，只对可恢复错误（限流/超时/5xx/连接中断）重试；
3. 把 SDK 异常统一翻译成本模块的异常类型，让上层不必感知 SDK 细节；
4. 记录 token 用量，便于观察上下文开销。
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# 异常体系：让上层可以区分「值得重试」和「不该重试」
# --------------------------------------------------------------------------- #
class LLMError(RuntimeError):
    """LLM 调用相关的基类异常。"""


class LLMConfigError(LLMError):
    """配置错误（缺少 API Key、参数非法等），重试无意义。"""


class LLMRetryExhausted(LLMError):
    """可恢复错误在重试耗尽后仍然失败。"""


# --------------------------------------------------------------------------- #
# 消息数据结构：内部统一使用强类型，避免到处传 dict 导致 KeyError
# --------------------------------------------------------------------------- #
@dataclass
class FunctionCall:
    name: str
    arguments: str  # 模型返回的是 JSON 字符串，保留原样以便回传给 API


@dataclass
class ToolCall:
    id: str
    function: FunctionCall
    type: str = "function"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "function": {"name": self.function.name, "arguments": self.function.arguments},
        }


@dataclass
class ChatMessage:
    """一条对话消息，兼容 OpenAI Chat Completions 的三种角色扩展。

    role 取值：system / user / assistant / tool
    """

    role: str
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"role": self.role}

        if self.role == "tool":
            payload["content"] = self.content
            payload["tool_call_id"] = self.tool_call_id
            return payload

        if self.role == "assistant" and self.tool_calls:
            # 注意：带 tool_calls 的 assistant 消息 content 允许为空字符串
            payload["content"] = self.content or ""
            payload["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
            return payload

        # 部分服务不接受 null content，统一用字符串
        payload["content"] = self.content or ""
        if self.name:
            payload["name"] = self.name
        return payload

    @classmethod
    def from_api(cls, raw: Any) -> "ChatMessage":
        """把 SDK 返回的 message 对象转成内部的 ChatMessage。"""
        tool_calls: list[ToolCall] = []
        for tc in getattr(raw, "tool_calls", None) or []:
            fn = getattr(tc, "function", None)
            tool_calls.append(
                ToolCall(
                    id=getattr(tc, "id", "") or "",
                    type=getattr(tc, "type", "function") or "function",
                    function=FunctionCall(
                        name=getattr(fn, "name", "") or "",
                        arguments=getattr(fn, "arguments", "") or "",
                    ),
                )
            )
        return cls(
            role="assistant",
            content=getattr(raw, "content", None) or "",
            tool_calls=tool_calls,
        )


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
            self.total_tokens + other.total_tokens,
        )


# --------------------------------------------------------------------------- #
# 客户端
# --------------------------------------------------------------------------- #
class LLMClient:
    """对 OpenAI 兼容 Chat Completions 接口的薄封装。"""

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        timeout: float = 60.0,
        max_retries: int = 4,
        backoff_base: float = 1.5,
        backoff_cap: float = 30.0,
        client: Any | None = None,
    ) -> None:
        if not api_key:
            raise LLMConfigError(
                "缺少 API Key。请复制 .env.example 为 .env 并填写 LLM_API_KEY，"
                "或设置同名环境变量。"
            )
        if not model:
            raise LLMConfigError("缺少模型名（LLM_MODEL）。")

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max(0, max_retries)
        self.backoff_base = backoff_base
        self.backoff_cap = backoff_cap
        self.usage = Usage()

        if client is not None:
            # 允许注入假客户端，便于离线测试
            self._client = client
        else:
            self._client = self._build_client(api_key, base_url, timeout)

    @staticmethod
    def _build_client(api_key: str, base_url: str | None, timeout: float) -> Any:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - 依赖缺失时的友好提示
            raise LLMConfigError(
                "未安装 openai SDK，请先执行：pip install -r requirements.txt"
            ) from exc

        kwargs: dict[str, Any] = {"api_key": api_key, "timeout": timeout}
        if base_url:
            kwargs["base_url"] = base_url
        # SDK 自带重试会与我们自己的退避策略叠加，这里关掉以免放大延迟
        kwargs["max_retries"] = 0
        return OpenAI(**kwargs)

    # ---------------------------------------------------------------- 调用 --
    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        tools: Iterable[dict[str, Any]] | None = None,
    ) -> ChatMessage:
        """发起一次对话补全，返回 assistant 消息。

        失败时抛出 :class:`LLMRetryExhausted`（可恢复错误重试耗尽）
        或 :class:`LLMConfigError`（不可恢复）。
        """
        payload_messages = [m.to_dict() for m in messages]
        tool_list = list(tools) if tools else []

        request: dict[str, Any] = {
            "model": self.model,
            "messages": payload_messages,
            "temperature": self.temperature,
        }
        if self.max_tokens:
            request["max_tokens"] = self.max_tokens
        if tool_list:
            request["tools"] = tool_list
            request["tool_choice"] = "auto"

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.chat.completions.create(**request)
            except Exception as exc:  # noqa: BLE001 - 需要分类处理所有 SDK 异常
                if not self._is_retryable(exc):
                    raise self._translate(exc) from exc
                last_error = exc
                if attempt >= self.max_retries:
                    break
                delay = self._backoff_delay(attempt)
                logger.warning(
                    "LLM 调用失败（第 %d/%d 次，%.1fs 后重试）：%s",
                    attempt + 1,
                    self.max_retries + 1,
                    delay,
                    exc,
                )
                time.sleep(delay)
                continue

            return self._parse_response(response)

        raise LLMRetryExhausted(
            f"LLM 调用在 {self.max_retries + 1} 次尝试后仍然失败：{last_error}"
        ) from last_error

    # ------------------------------------------------------------ 内部工具 --
    def _parse_response(self, response: Any) -> ChatMessage:
        choices = getattr(response, "choices", None) or []
        if not choices:
            raise LLMError("接口返回了空 choices，无法解析回复。")

        raw_usage = getattr(response, "usage", None)
        if raw_usage is not None:
            self.usage = self.usage + Usage(
                prompt_tokens=getattr(raw_usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(raw_usage, "completion_tokens", 0) or 0,
                total_tokens=getattr(raw_usage, "total_tokens", 0) or 0,
            )

        message = getattr(choices[0], "message", None)
        if message is None:
            raise LLMError("接口返回的 choice 中缺少 message 字段。")

        parsed = ChatMessage.from_api(message)
        if not parsed.content and not parsed.tool_calls:
            raise LLMError("模型既没有返回文本也没有返回工具调用，回复为空。")
        return parsed

    def _backoff_delay(self, attempt: int) -> float:
        """指数退避 + 抖动，并保证结果不超过 ``backoff_cap``。

        注意先乘抖动、再封顶：若写成 ``min(cap, base*2^n) * jitter``，
        抖动会把结果推到上限之外（最大 1.3 倍），使 cap 名不副实。
        """
        raw = self.backoff_base * (2**attempt)
        jittered = raw * (0.7 + 0.6 * random.random())
        return min(self.backoff_cap, jittered)

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        name = type(exc).__name__
        # AssertionError / KeyError 属于调用方（或测试替身）的编程错误，
        # 重试只会掩盖真实问题，必须直接暴露。
        if name in {"AssertionError", "KeyError", "TypeError", "AttributeError"}:
            return False
        if name in {
            "APITimeoutError",
            "APIConnectionError",
            "InternalServerError",
            "RateLimitError",
            "APIStatusError",
        }:
            # APIStatusError 需要看状态码再决定
            if name == "APIStatusError":
                status = getattr(exc, "status_code", None)
                return status is None or status >= 500 or status == 429
            return True
        # 类型未知时保守地不重试
        return False

    @staticmethod
    def _translate(exc: Exception) -> LLMError:
        name = type(exc).__name__
        status = getattr(exc, "status_code", None)
        if name == "AuthenticationError" or status == 401:
            return LLMConfigError("鉴权失败（401）：API Key 无效或与 base_url 不匹配。")
        if name == "NotFoundError" or status == 404:
            return LLMConfigError("接口不存在（404）：请检查 base_url 与模型名是否正确。")
        if name == "PermissionDeniedError" or status == 403:
            return LLMConfigError("权限不足（403）：当前 Key 无权访问该模型。")
        if name == "BadRequestError" or status == 400:
            return LLMConfigError(f"请求被拒绝（400）：{exc}")
        return LLMError(f"LLM 调用失败：{name}: {exc}")
