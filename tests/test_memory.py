"""上下文记忆测试：滑窗裁剪与摘要压缩的正确性。"""

from __future__ import annotations

import pytest

from code_review_agent.llm import ChatMessage
from code_review_agent.memory import ConversationMemory, estimate_tokens


def test_system_prompt_always_present() -> None:
    memory = ConversationMemory("你是审查助手", window=4)
    memory.add(ChatMessage(role="user", content="你好"))

    messages = memory.to_messages()

    assert messages[0].role == "system"
    assert messages[0].content == "你是审查助手"


def test_window_trims_old_messages() -> None:
    memory = ConversationMemory("sys", window=4, summary_trigger=5)
    for index in range(20):
        memory.add(ChatMessage(role="user", content=f"消息{index}"))

    messages = memory.to_messages()

    # 1 条 system + 不超过 window 条历史
    assert len(messages) <= 1 + 4
    assert messages[-1].content == "消息19"


def test_summary_is_injected_when_summarizer_present() -> None:
    calls: list[int] = []

    def summarizer(history) -> str:
        calls.append(len(history))
        return f"压缩了{len(history)}条"

    memory = ConversationMemory("sys", window=4, summarizer=summarizer, summary_trigger=5)
    for index in range(20):
        memory.add(ChatMessage(role="user", content=f"消息{index}"))

    messages = memory.to_messages()

    assert calls, "摘要器应被调用"
    assert any("早前对话摘要" in m.content for m in messages)
    assert messages[0].role == "system"
    assert messages[1].role == "assistant"


def test_summarizer_failure_does_not_break_memory() -> None:
    def broken(_history) -> str:
        raise RuntimeError("摘要服务不可用")

    memory = ConversationMemory("sys", window=4, summarizer=broken, summary_trigger=5)
    for index in range(12):
        memory.add(ChatMessage(role="user", content=f"消息{index}"))

    messages = memory.to_messages()

    assert messages[-1].content == "消息11"
    assert len(messages) <= 1 + 4


@pytest.mark.parametrize("window", [2, 3, 4, 5, 6, 7])
def test_trim_boundary_never_starts_with_tool_message(window: int) -> None:
    """裁剪后保留区的首条不能是 tool 消息。

    这是真实存在的服务端约束：``role="tool"`` 必须紧跟在带 ``tool_calls`` 的
    assistant 消息之后，否则下一次请求会被 API 直接拒绝（400）。
    ``ConversationMemory._maybe_compact`` 在裁剪时会把边界向前推过来规避。

    用参数化覆盖多种窗口大小，确保无论窗口与消息数如何对齐都成立。
    """
    from code_review_agent.llm import FunctionCall, ToolCall

    memory = ConversationMemory("sys", window=window, summary_trigger=window + 1)

    # 造 20 轮「assistant 请求工具 → tool 返回结果」，共 40 条消息
    for index in range(20):
        call = ToolCall(id=f"c{index}", function=FunctionCall(name="read_file", arguments="{}"))
        memory.add(ChatMessage(role="assistant", tool_calls=[call]))
        memory.add(ChatMessage(role="tool", content=f"结果{index}", tool_call_id=f"c{index}"))

    # 内部保留区（不含 system 与摘要）的首条必须是 assistant
    assert memory._messages, "裁剪后不应为空"  # noqa: SLF001
    assert memory._messages[0].role != "tool", (  # noqa: SLF001
        f"window={window} 时保留了悬空的 tool 消息"
    )
    # 对外暴露的消息序列中，system 之后的第一条同样不能是 tool
    visible = memory.to_messages()
    assert visible[0].role == "system"
    assert visible[1].role != "tool"


def test_every_tool_message_has_matching_assistant_call() -> None:
    """更本质的不变量：每条 tool 消息都能找到配对的前序 assistant.tool_calls。"""
    from code_review_agent.llm import FunctionCall, ToolCall

    memory = ConversationMemory("sys", window=3, summary_trigger=4)
    for index in range(10):
        call = ToolCall(id=f"c{index}", function=FunctionCall(name="read_file", arguments="{}"))
        memory.add(ChatMessage(role="assistant", tool_calls=[call]))
        memory.add(ChatMessage(role="tool", content=f"结果{index}", tool_call_id=f"c{index}"))

    pending_ids: set[str] = set()
    for message in memory._messages:  # noqa: SLF001
        if message.role == "assistant" and message.tool_calls:
            pending_ids.update(tc.id for tc in message.tool_calls)
        elif message.role == "tool":
            assert message.tool_call_id in pending_ids, (
                f"tool 消息 {message.tool_call_id} 缺少配对的 assistant.tool_calls"
            )


def test_reset_clears_history_and_summary() -> None:
    memory = ConversationMemory("sys", window=4, summarizer=lambda h: "摘要", summary_trigger=2)
    for index in range(10):
        memory.add(ChatMessage(role="user", content=f"消息{index}"))

    memory.reset()

    assert len(memory) == 0
    assert len(memory.to_messages()) == 1  # 只剩 system


def test_token_estimate_grows_with_content() -> None:
    short = estimate_tokens("你好")
    long = estimate_tokens("你好" * 100)

    assert long > short
    assert estimate_tokens("") == 0
