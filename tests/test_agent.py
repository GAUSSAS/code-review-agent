"""Agent 循环测试：多步工具调用、终止条件、错误恢复与步数上限。"""

from __future__ import annotations

import pytest

from code_review_agent.agent import AgentEvent, ReviewAgent, StepLimitExceeded
from code_review_agent.config import Settings
from code_review_agent.tools import ToolRegistry, ToolResult
from tests.helpers import text_response, tool_response


def collect(agent: ReviewAgent, **kwargs) -> list[AgentEvent]:
    return list(agent.review(**kwargs))


# --------------------------------------------------------------------------- #
# 基本循环
# --------------------------------------------------------------------------- #
def test_agent_runs_full_reason_act_loop(settings, make_llm_client) -> None:
    client, _ = make_llm_client(
        [
            tool_response([("list_directory", {"path": "."})]),
            tool_response([("analyze_code", {"path": "app/service.py"})]),
            text_response("# 审查报告\n\n发现 3 个问题"),
        ]
    )
    agent = ReviewAgent(settings, client=client)

    events = collect(agent)

    types = [e.type for e in events]
    assert types[0] == "start"
    assert types[-1] == "final"
    assert types.count("tool_call") == 2
    assert types.count("tool_result") == 2
    assert "发现 3 个问题" in events[-1].text


def test_agent_records_tool_results_into_memory(settings, make_llm_client) -> None:
    client, _ = make_llm_client(
        [
            tool_response([("read_file", {"path": "app/service.py"})]),
            text_response("完成"),
        ]
    )
    agent = ReviewAgent(settings, client=client)

    collect(agent)

    roles = [m.role for m in agent.memory.to_messages()]
    assert "tool" in roles
    assert roles[0] == "system"


def test_finish_review_terminates_loop_immediately(settings, make_llm_client) -> None:
    """模型调用 finish_review 后，Agent 必须立即结束，不再请求下一轮。"""
    report = "# 最终报告\n\n无阻塞问题"
    client, fake = make_llm_client(
        [
            tool_response([("finish_review", {"report": report})]),
            text_response("这轮不应该被调用"),
        ]
    )
    agent = ReviewAgent(settings, client=client)

    events = collect(agent)

    assert events[-1].type == "final"
    assert report in events[-1].text
    assert len(fake.calls) == 1, "finish_review 之后不应再调用模型"


def test_stats_report_tool_usage(settings, make_llm_client) -> None:
    client, _ = make_llm_client(
        [
            tool_response([("list_directory", {})]),
            tool_response([("read_file", {"path": "README.md"})]),
            text_response("完成", total_tokens=100),
        ]
    )
    agent = ReviewAgent(settings, client=client)

    collect(agent)
    stats = agent.stats()

    assert stats["steps"] == 3
    assert stats["tool_calls"] == 2
    assert stats["tools_used"] == {"list_directory": 1, "read_file": 1}
    assert stats["usage"]["total_tokens"] == 100


# --------------------------------------------------------------------------- #
# 步数上限与兜底
# --------------------------------------------------------------------------- #
def test_step_limit_raises(settings, make_llm_client) -> None:
    settings.max_steps = 2
    client, _ = make_llm_client(
        [
            tool_response([("list_directory", {})]),
            tool_response([("list_directory", {})]),
            text_response("不该到这里"),
        ]
    )
    agent = ReviewAgent(settings, client=client)

    with pytest.raises(StepLimitExceeded):
        collect(agent)


def test_force_conclusion_uses_existing_context(settings, make_llm_client) -> None:
    settings.max_steps = 1
    client, fake = make_llm_client(
        [
            tool_response([("list_directory", {})]),
            text_response("基于已有信息：未发现阻塞问题"),
        ]
    )
    agent = ReviewAgent(settings, client=client)

    with pytest.raises(StepLimitExceeded):
        collect(agent)

    conclusion = agent.force_conclusion()

    assert "未发现阻塞问题" in conclusion
    # 兜底请求里应包含「不要再请求调用任何工具」的指令
    last_messages = fake.calls[-1]["messages"]
    assert "不要再请求调用任何工具" in last_messages[-1]["content"]
    # 兜底调用明确不带 tools
    assert "tools" not in fake.calls[-1]


# --------------------------------------------------------------------------- #
# 错误处理
# --------------------------------------------------------------------------- #
def test_tool_error_is_fed_back_without_crashing(settings, make_llm_client) -> None:
    client, _ = make_llm_client(
        [
            tool_response([("read_file", {"path": "../../../etc/passwd"})]),
            text_response("路径被拒绝，我改用 search_code"),
        ]
    )
    agent = ReviewAgent(settings, client=client)

    events = collect(agent)

    tool_results = [e for e in events if e.type == "tool_result"]
    assert tool_results and tool_results[0].ok is False
    assert events[-1].type == "final"


def test_unknown_tool_is_reported_to_model(settings, make_llm_client) -> None:
    client, _ = make_llm_client(
        [tool_response([("no_such_tool", {})]), text_response("明白")]
    )
    agent = ReviewAgent(settings, client=client)

    events = collect(agent)

    assert any(e.type == "tool_result" and not e.ok for e in events)


def test_malformed_tool_arguments_do_not_crash(settings, make_llm_client) -> None:
    from types import SimpleNamespace

    bad_call = SimpleNamespace(
        id="call_x",
        type="function",
        function=SimpleNamespace(name="read_file", arguments="{not json"),
    )
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="", tool_calls=[bad_call]))],
        usage=None,
    )
    client, _ = make_llm_client([response, text_response("已纠正")])
    agent = ReviewAgent(settings, client=client)

    events = collect(agent)

    assert any(e.type == "tool_result" and not e.ok for e in events)
    assert events[-1].type == "final"


def test_llm_error_becomes_error_event(settings, make_llm_client) -> None:
    client, _ = make_llm_client([RuntimeError("网络断了")])
    agent = ReviewAgent(settings, client=client)

    events: list[AgentEvent] = []
    with pytest.raises(Exception):
        for event in agent.review():
            events.append(event)

    assert any(e.type == "error" for e in events)


# --------------------------------------------------------------------------- #
# 组件协作
# --------------------------------------------------------------------------- #
def test_agent_accepts_custom_registry(settings, workspace, make_llm_client) -> None:
    registry = ToolRegistry()
    registry.register(
        "ping",
        "测试用工具",
        {"type": "object", "properties": {}, "required": []},
        lambda: ToolResult(content="pong"),
    )
    client, fake = make_llm_client([tool_response([("ping", {})]), text_response("完成")])
    agent = ReviewAgent(settings, client=client, registry=registry)

    collect(agent)

    assert agent.registry.names() == ["ping"]
    assert "pong" in fake.calls[-1]["messages"][-1]["content"]


def test_review_prompt_includes_workspace_and_target(settings, make_llm_client) -> None:
    client, fake = make_llm_client([text_response("完成")])
    agent = ReviewAgent(settings, client=client)

    collect(agent, path="app/service.py", focus="并发安全")

    user_prompt = fake.calls[0]["messages"][-1]["content"]
    assert "app/service.py" in user_prompt
    assert "并发安全" in user_prompt
    assert str(agent.workspace) in user_prompt


def test_few_shot_can_be_disabled(settings, make_llm_client) -> None:
    client, _ = make_llm_client([text_response("完成")])
    agent = ReviewAgent(settings, client=client, with_few_shot=False)

    system_prompt = agent.memory.system_prompt

    assert "Few-shot" not in system_prompt
