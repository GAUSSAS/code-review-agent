"""LLM 客户端测试：消息序列化、响应解析、重试与异常翻译。"""

from __future__ import annotations

import pytest

from code_review_agent.llm import (
    ChatMessage,
    FunctionCall,
    LLMClient,
    LLMConfigError,
    LLMError,
    LLMRetryExhausted,
    ToolCall,
)
from tests.helpers import text_response, tool_response


# --------------------------------------------------------------------------- #
# 配置校验
# --------------------------------------------------------------------------- #
def test_missing_api_key_raises_config_error() -> None:
    with pytest.raises(LLMConfigError):
        LLMClient(model="m", api_key="")


def test_missing_model_raises_config_error() -> None:
    with pytest.raises(LLMConfigError):
        LLMClient(model="", api_key="k")


# --------------------------------------------------------------------------- #
# 消息序列化
# --------------------------------------------------------------------------- #
def test_plain_message_serialization() -> None:
    payload = ChatMessage(role="user", content="你好").to_dict()

    assert payload == {"role": "user", "content": "你好"}


def test_tool_message_serialization_includes_call_id() -> None:
    payload = ChatMessage(role="tool", content="结果", tool_call_id="call_1", name="read_file").to_dict()

    assert payload["role"] == "tool"
    assert payload["tool_call_id"] == "call_1"
    assert payload["content"] == "结果"


def test_assistant_with_tool_calls_serialization() -> None:
    message = ChatMessage(
        role="assistant",
        content="",
        tool_calls=[
            ToolCall(id="call_1", function=FunctionCall(name="read_file", arguments='{"path":"a.py"}'))
        ],
    )

    payload = message.to_dict()

    assert payload["tool_calls"][0]["function"]["name"] == "read_file"
    assert payload["content"] == ""


# --------------------------------------------------------------------------- #
# 响应解析
# --------------------------------------------------------------------------- #
def test_parses_text_response_and_usage(make_llm_client) -> None:
    client, _ = make_llm_client([text_response("审查完成", total_tokens=42)])

    reply = client.chat([ChatMessage(role="user", content="hi")])

    assert reply.content == "审查完成"
    assert reply.tool_calls == []
    assert client.usage.total_tokens == 42


def test_parses_tool_call_response(make_llm_client) -> None:
    client, _ = make_llm_client([tool_response([("read_file", {"path": "a.py"})])])

    reply = client.chat([ChatMessage(role="user", content="hi")])

    assert len(reply.tool_calls) == 1
    call = reply.tool_calls[0]
    assert call.function.name == "read_file"
    assert call.function.arguments == '{"path": "a.py"}'
    assert call.id == "call_0"


def test_tools_are_forwarded_to_api(make_llm_client) -> None:
    client, fake = make_llm_client([text_response("ok")])
    specs = [
        {
            "type": "function",
            "function": {"name": "read_file", "description": "d", "parameters": {"type": "object"}},
        }
    ]

    client.chat([ChatMessage(role="user", content="hi")], tools=specs)

    sent = fake.calls[0]
    assert sent["tools"] == specs
    assert sent["tool_choice"] == "auto"


def test_no_tools_key_when_tools_absent(make_llm_client) -> None:
    client, fake = make_llm_client([text_response("ok")])

    client.chat([ChatMessage(role="user", content="hi")])

    assert "tools" not in fake.calls[0]


def test_empty_response_raises(make_llm_client) -> None:
    from types import SimpleNamespace

    empty = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="", tool_calls=[]))],
        usage=None,
    )
    client, _ = make_llm_client([empty])

    with pytest.raises(LLMError):
        client.chat([ChatMessage(role="user", content="hi")])


def test_usage_accumulates_across_calls(make_llm_client) -> None:
    client, _ = make_llm_client([text_response("a", total_tokens=10), text_response("b", total_tokens=20)])

    client.chat([ChatMessage(role="user", content="1")])
    client.chat([ChatMessage(role="user", content="2")])

    assert client.usage.total_tokens == 30


# --------------------------------------------------------------------------- #
# 重试与异常翻译
# --------------------------------------------------------------------------- #
class RateLimitError(Exception):  # 名称与 SDK 一致，用于触发重试逻辑
    pass


class AuthenticationError(Exception):
    pass


def test_retryable_error_is_retried_then_succeeds(make_llm_client) -> None:
    client, fake = make_llm_client([RateLimitError("429"), text_response("第二次成功")])
    client.max_retries = 2
    client.backoff_base = 0.0  # 测试中不真正等待

    reply = client.chat([ChatMessage(role="user", content="hi")])

    assert reply.content == "第二次成功"
    assert len(fake.calls) == 2


def test_retry_exhausted_raises(make_llm_client) -> None:
    client, fake = make_llm_client([RateLimitError("429")])
    client.max_retries = 2
    client.backoff_base = 0.0

    with pytest.raises(LLMRetryExhausted):
        client.chat([ChatMessage(role="user", content="hi")])

    assert len(fake.calls) == 3  # 首次 + 2 次重试


def test_non_retryable_error_is_translated_not_retried(make_llm_client) -> None:
    error = AuthenticationError("invalid key")
    error.status_code = 401  # type: ignore[attr-defined]
    client, fake = make_llm_client([error])
    client.max_retries = 3
    client.backoff_base = 0.0

    with pytest.raises(LLMConfigError):
        client.chat([ChatMessage(role="user", content="hi")])

    assert len(fake.calls) == 1, "鉴权错误不应重试"


def test_backoff_is_exponential_then_capped(make_llm_client, monkeypatch) -> None:
    """退避应按指数增长，并在超过上限后被截断。

    固定随机数使断言确定化：抖动系数 = 0.7 + 0.6 * 0.5 = 1.0。
    """
    import code_review_agent.llm as llm_module

    monkeypatch.setattr(llm_module.random, "random", lambda: 0.5)

    client, _ = make_llm_client([])
    client.backoff_base = 1.0
    client.backoff_cap = 4.0

    assert client._backoff_delay(0) == pytest.approx(1.0)  # noqa: SLF001
    assert client._backoff_delay(1) == pytest.approx(2.0)  # noqa: SLF001
    assert client._backoff_delay(2) == pytest.approx(4.0)  # noqa: SLF001
    # 原始值 8.0 / 16.0 超过上限，必须被截断到 cap
    assert client._backoff_delay(3) == pytest.approx(4.0)  # noqa: SLF001
    assert client._backoff_delay(4) == pytest.approx(4.0)  # noqa: SLF001


def test_backoff_never_exceeds_cap_despite_jitter(make_llm_client) -> None:
    """回归：抖动不能被乘到上限之外（曾经写成 min(cap, raw) * jitter）。"""
    client, _ = make_llm_client([])
    client.backoff_base = 1.0
    client.backoff_cap = 4.0

    delays = [client._backoff_delay(n) for n in range(12)]  # noqa: SLF001

    assert all(delay <= 4.0 for delay in delays), delays
    assert all(delay > 0 for delay in delays)
    assert delays[0] < delays[2]  # 前期确实在增长
