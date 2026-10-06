"""Web 层测试：事件序列化、SSE 帧编码、会话生命周期。

全部离线：用 ``FakeAgent`` 替身代替真实 Agent，不触碰网络与 LLM。
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any, Iterator

import pytest

from code_review_agent.agent import AgentEvent
from code_review_agent.web.events import MAX_TOOL_TEXT, serialize_event, summarize_tool_text
from code_review_agent.web.sessions import (
    ReviewOptions,
    Session,
    SessionStore,
    sse_frame,
    stream_session,
)


# --------------------------------------------------------------------------- #
# 替身 Agent
# --------------------------------------------------------------------------- #
class FakeAgent:
    """实现 Web 层用到的那部分 Agent 接口。"""

    def __init__(self, script: list[list[AgentEvent]] | None = None) -> None:
        self.workspace = None  # Web 层不读这个字段，占位即可
        self.script = list(script or [])
        self.chat_messages: list[str] = []
        self.review_calls: list[dict[str, Any]] = []
        self.forced = 0

    def review(self, **kwargs: Any) -> Iterator[AgentEvent]:
        self.review_calls.append(kwargs)
        events = self.script.pop(0) if self.script else [AgentEvent(type="final", text="空")]
        yield from events

    def chat_turn(self, user_input: str) -> Iterator[AgentEvent]:
        self.chat_messages.append(user_input)
        events = self.script.pop(0) if self.script else [AgentEvent(type="final", text="追问回复")]
        yield from events

    def force_conclusion(self) -> str:
        self.forced += 1
        return "兜底结论"

    def stats(self) -> dict[str, Any]:
        return {
            "steps": 3,
            "tool_calls": 2,
            "tools_used": {"analyze_code": 1, "read_file": 1},
            "usage": {"prompt_tokens": 1200, "completion_tokens": 340, "total_tokens": 1540},
        }


def basic_events() -> list[AgentEvent]:
    return [
        AgentEvent(type="start", text="已接收任务"),
        AgentEvent(type="assistant", text="先看目录", step=1),
        AgentEvent(type="tool_call", tool="list_directory", arguments={"path": "."}, step=1),
        AgentEvent(type="tool_result", tool="list_directory", text="app/", step=1, ok=True),
        AgentEvent(type="final", text="# 报告\n\n没有问题", step=2),
    ]


@pytest.fixture
def options(tmp_path) -> ReviewOptions:
    return ReviewOptions(workspace=tmp_path)


# --------------------------------------------------------------------------- #
# 事件序列化
# --------------------------------------------------------------------------- #
def test_summarize_short_text_is_untouched() -> None:
    text, truncated = summarize_tool_text("很短的结果")

    assert text == "很短的结果"
    assert truncated is False


def test_summarize_long_text_keeps_head_and_tail() -> None:
    text = "A" * 5000 + "MIDDLE" + "B" * 5000

    trimmed, truncated = summarize_tool_text(text)

    assert truncated is True
    assert len(trimmed) < len(text)
    assert "此处省略" in trimmed
    assert trimmed.startswith("A" * 100)
    assert trimmed.endswith("B" * 100)
    assert "MIDDLE" not in trimmed


def test_serialize_keeps_reasoning_text_intact() -> None:
    """推理文本不该被截断，用户就是要读它。"""
    event = AgentEvent(type="assistant", text="B" * (MAX_TOOL_TEXT + 500), step=1)

    payload = serialize_event(event).to_dict()

    assert len(payload["text"]) > MAX_TOOL_TEXT
    assert "truncated" not in payload


def test_serialize_truncates_tool_results_only() -> None:
    big = AgentEvent(type="tool_result", tool="analyze_code", text="C" * (MAX_TOOL_TEXT + 500))
    payload = serialize_event(big).to_dict()

    assert payload["truncated"] is True
    assert len(payload["text"]) < MAX_TOOL_TEXT


def test_serialize_omits_empty_optional_fields() -> None:
    payload = serialize_event(AgentEvent(type="done")).to_dict()

    assert payload == {"type": "done", "ok": True, "step": 0}
    assert "text" not in payload and "tool" not in payload


def test_serialize_includes_elapsed_and_data() -> None:
    event = AgentEvent(type="stats", elapsed=1.234, data={"steps": 3})

    payload = serialize_event(event).to_dict()

    assert payload["elapsed"] == 1.23
    assert payload["data"] == {"steps": 3}


# --------------------------------------------------------------------------- #
# SSE 帧
# --------------------------------------------------------------------------- #
def test_sse_frame_format() -> None:
    frame = sse_frame("tool_call", {"type": "tool_call", "tool": "read_file"})

    assert frame.startswith("event: tool_call\ndata: ")
    assert frame.endswith("\n\n")
    body = json.loads(frame.split("data: ", 1)[1].strip())
    assert body["tool"] == "read_file"


def test_sse_frame_preserves_chinese() -> None:
    frame = sse_frame("final", {"type": "final", "text": "审查完成"})

    assert "审查完成" in frame  # ensure_ascii=False，不应是 \uXXXX


# --------------------------------------------------------------------------- #
# 会话
# --------------------------------------------------------------------------- #
def test_review_emits_all_events_then_done(options: ReviewOptions) -> None:
    agent = FakeAgent([basic_events()])
    session = Session("s1", agent, options)

    assert session.start_review() is True
    _wait_idle(session)

    types = [e["type"] for e in session.snapshot()]
    assert types[0] == "start"
    assert "tool_call" in types and "tool_result" in types
    assert types[-2:] == ["stats", "done"]
    assert session.status == "ready"
    assert agent.review_calls and agent.review_calls[0]["path"] is None


def test_session_rejects_concurrent_tasks(options: ReviewOptions) -> None:
    """同一会话不允许并发跑两个任务，否则记忆会被交叉写入。"""
    started = threading.Event()
    release = threading.Event()

    class BlockingAgent(FakeAgent):
        def review(self, **kwargs: Any) -> Iterator[AgentEvent]:
            started.set()
            release.wait(timeout=5)
            yield AgentEvent(type="final", text="完成")

    session = Session("s2", BlockingAgent(), options)
    assert session.start_review() is True
    assert started.wait(timeout=5), "后台任务未启动"

    # 任务仍在跑，第二次启动必须被拒绝
    assert session.busy is True
    assert session.start_review() is False
    assert session.chat("不应该被接受") is False

    release.set()
    _wait_idle(session)
    assert session.status == "ready"


def test_chat_reuses_same_agent_instance(options: ReviewOptions) -> None:
    """追问必须复用同一个 Agent，否则上下文记忆会丢失。"""
    agent = FakeAgent([basic_events(), [AgentEvent(type="final", text="追问答案")]])
    session = Session("s3", agent, options)

    session.start_review()
    _wait_idle(session)
    assert session.chat("第 2 个问题给出修复代码") is True
    _wait_idle(session)

    assert agent.chat_messages == ["第 2 个问题给出修复代码"]
    finals = [e["text"] for e in session.snapshot() if e["type"] == "final"]
    assert "追问答案" in finals[-1]


def test_worker_error_is_reported_as_event(options: ReviewOptions) -> None:
    class ExplodingAgent(FakeAgent):
        def review(self, **kwargs: Any) -> Iterator[AgentEvent]:
            raise RuntimeError("模拟 LLM 崩溃")
            yield  # pragma: no cover - 仅为让本方法成为生成器函数

    session = Session("s4", ExplodingAgent(), options)
    session.start_review()
    _wait_idle(session)

    assert session.status == "error"
    errors = [e for e in session.snapshot() if e["type"] == "error"]
    assert errors and "模拟 LLM 崩溃" in errors[0]["text"]


def test_save_report_writes_file(tmp_path, options: ReviewOptions) -> None:
    options.save_report = True
    agent = FakeAgent([[AgentEvent(type="final", text="# 报告正文")]])
    session = Session("s5", agent, options)

    session.start_review()
    _wait_idle(session)

    target = tmp_path / "review_report.md"
    assert target.is_file()
    assert target.read_text(encoding="utf-8").startswith("# 报告正文")
    assert session.report_path == str(target)
    assert any(e["type"] == "report_saved" for e in session.snapshot())


def test_event_buffer_is_capped(options: ReviewOptions) -> None:
    """事件上限生效后，后续事件被替换为一次性告警，内存不会无界增长。"""
    session = Session("s6", FakeAgent(), options, max_events=5)
    for index in range(20):
        session.emit("assistant", text=f"消息{index}")

    events = session.snapshot()
    assert len(events) == 6  # 5 条正常事件 + 1 条告警
    # 告警用 error 类型上报，但仍保留 step 字段，保证前端渲染不取到 undefined
    assert events[-1]["type"] == "error"
    assert events[-1]["step"] == 0
    assert "上限" in events[-1]["text"]


def test_session_store_create_get_delete(tmp_path) -> None:
    store = SessionStore()
    session = store.create(FakeAgent(), ReviewOptions(workspace=tmp_path))

    assert store.get(session.id) is session
    assert store.count() == 1
    store.delete(session.id)
    assert store.get(session.id) is None
    assert store.count() == 0


# --------------------------------------------------------------------------- #
# SSE 流
# --------------------------------------------------------------------------- #
def test_stream_yields_frames_until_session_ends(options: ReviewOptions) -> None:
    session = Session("s7", FakeAgent([basic_events()]), options)
    session.start_review()

    frames = list(stream_session(session, poll_interval=0.02, idle_timeout=5.0))

    assert frames[0].startswith(": connected")
    joined = "".join(frames)
    assert "event: tool_call" in joined
    assert "event: final" in joined
    assert "event: done" in joined
    assert "event: closed" in joined          # 结束时主动关闭，前端据此停止重连
    assert frames[-1].startswith("event: closed")


def test_stream_replays_history_for_late_subscriber(options: ReviewOptions) -> None:
    """先跑完、后连接也不能漏事件（游标从 0 开始）。"""
    session = Session("s8", FakeAgent([basic_events()]), options)
    session.start_review()
    _wait_idle(session)

    frames = list(stream_session(session, poll_interval=0.02, idle_timeout=5.0))
    joined = "".join(frames)

    assert "event: start" in joined
    assert "event: final" in joined


def test_stream_subscriber_is_cleaned_up(options: ReviewOptions) -> None:
    session = Session("s9", FakeAgent([basic_events()]), options)
    session.start_review()

    frames = list(stream_session(session, poll_interval=0.02, idle_timeout=5.0))

    assert frames  # 流已结束
    assert session._subscribers == []  # noqa: SLF001


# --------------------------------------------------------------------------- #
# 请求参数解析：API Key 与模型覆盖
# --------------------------------------------------------------------------- #
def _web_config(tmp_path, allowed_roots=()):
    from code_review_agent.web.server import WebConfig

    return WebConfig(host="127.0.0.1", port=0, root=tmp_path, allowed_roots=tuple(allowed_roots))


def test_parse_api_key_accepts_normal_key() -> None:
    from code_review_agent.web.server import parse_api_key

    assert parse_api_key("sk-abc123") == "sk-abc123"
    assert parse_api_key("  sk-padded  ") == "sk-padded"
    assert parse_api_key("") is None
    assert parse_api_key(None) is None


def test_parse_api_key_rejects_bad_input() -> None:
    from code_review_agent.web.server import BadRequest, parse_api_key

    with pytest.raises(BadRequest):
        parse_api_key(12345)
    with pytest.raises(BadRequest):
        parse_api_key("x" * 300)
    with pytest.raises(BadRequest):
        parse_api_key("sk-line1\nsk-line2")


def test_parse_overrides_whitelists_fields(tmp_path) -> None:
    """前端只能覆盖白名单字段，尤其**不能**覆盖 base_url（防 SSRF）。"""
    from code_review_agent.web.server import parse_overrides

    overrides = parse_overrides(
        {
            "model": "deepseek-reasoner",
            "temperature": 0.5,
            "max_tokens": 2048,
            "timeout": 90,
            # 以下字段必须被忽略
            "base_url": "http://evil.example.com/v1",
            "workspace": "/etc",
            "api_key": "sk-should-not-leak-here",
        }
    )

    assert overrides == {
        "model": "deepseek-reasoner",
        "temperature": 0.5,
        "max_tokens": 2048,
        "timeout": 90,
    }
    assert "base_url" not in overrides
    assert "workspace" not in overrides


def test_parse_overrides_validates_ranges() -> None:
    from code_review_agent.web.server import BadRequest, parse_overrides

    with pytest.raises(BadRequest):
        parse_overrides({"temperature": 5})
    with pytest.raises(BadRequest):
        parse_overrides({"max_tokens": 0})
    with pytest.raises(BadRequest):
        parse_overrides({"timeout": 9999})
    with pytest.raises(BadRequest):
        parse_overrides({"temperature": "热"})


def test_parse_review_options_carries_key_and_overrides(tmp_path) -> None:
    from code_review_agent.web.server import parse_review_options

    options = parse_review_options(
        {"workspace": ".", "api_key": "sk-abc", "model": "m1", "max_steps": 5},
        _web_config(tmp_path),
    )

    assert options.api_key == "sk-abc"
    assert options.overrides["model"] == "m1"
    assert options.max_steps == 5
    assert options.root == tmp_path.resolve()


# --------------------------------------------------------------------------- #
# 目录访问策略：默认只允许默认根，--allow-root 可追加白名单
# --------------------------------------------------------------------------- #
def test_policy_allows_default_root_and_its_subdirs(tmp_path) -> None:
    from code_review_agent.web.server import DirectoryAccessPolicy

    (tmp_path / "sub").mkdir()
    policy = DirectoryAccessPolicy([tmp_path])

    resolved, matched = policy.resolve("sub")
    assert resolved == (tmp_path / "sub").resolve()
    assert matched == tmp_path.resolve()

    # 绝对路径指向默认根自身也应允许
    resolved2, _ = policy.resolve(str(tmp_path))
    assert resolved2 == tmp_path.resolve()


def test_policy_rejects_paths_outside_roots(tmp_path) -> None:
    from code_review_agent.web.server import BadRequest, DirectoryAccessPolicy

    outside = tmp_path.parent / "elsewhere"
    outside.mkdir(exist_ok=True)
    policy = DirectoryAccessPolicy([tmp_path / "workspace"])

    with pytest.raises(BadRequest) as exc_info:
        policy.resolve(str(outside))
    message = str(exc_info.value)
    assert "不在允许范围内" in message
    # 错误信息必须给出可操作的建议
    assert "--allow-root" in message


def test_policy_blocks_sibling_prefix_trap(tmp_path) -> None:
    """`/work_evil` 不能因为字符串前缀与 `/work` 相同而被放行。"""
    from code_review_agent.web.server import BadRequest, DirectoryAccessPolicy

    allowed = tmp_path / "work"
    evil = tmp_path / "work_evil"
    allowed.mkdir()
    evil.mkdir()

    policy = DirectoryAccessPolicy([allowed])
    with pytest.raises(BadRequest):
        policy.resolve(str(evil))


def test_policy_allow_root_grants_access(tmp_path) -> None:
    from code_review_agent.web.server import DirectoryAccessPolicy

    default = tmp_path / "default"
    extra = tmp_path / "extra"
    default.mkdir()
    extra.mkdir()

    policy = DirectoryAccessPolicy([default, extra])

    assert set(policy.roots) == {default.resolve(), extra.resolve()}
    resolved, matched = policy.resolve(str(extra))
    assert resolved == extra.resolve()
    assert matched == extra.resolve()
    assert str(extra.resolve()) in policy.describe()


def test_policy_rejects_nonexistent_and_files(tmp_path) -> None:
    from code_review_agent.web.server import BadRequest, DirectoryAccessPolicy

    policy = DirectoryAccessPolicy([tmp_path])
    a_file = tmp_path / "a.txt"
    a_file.write_text("x", encoding="utf-8")

    with pytest.raises(BadRequest):
        policy.resolve("no-such-dir")
    with pytest.raises(BadRequest):
        policy.resolve("a.txt")


def test_parse_review_options_accepts_allowed_root(tmp_path) -> None:
    from code_review_agent.web.server import parse_review_options

    default = tmp_path / "default"
    extra = tmp_path / "extra"
    default.mkdir()
    extra.mkdir()

    config = _web_config(default, allowed_roots=[extra])
    options = parse_review_options({"workspace": str(extra)}, config)

    assert options.workspace == extra.resolve()
    assert options.root == extra.resolve()


def test_agent_factory_prefers_request_key(tmp_path) -> None:
    """请求携带的 Key 必须覆盖服务端配置，且不污染原始 Settings。"""
    from code_review_agent.config import Settings
    from code_review_agent.llm import LLMClient
    from code_review_agent.web.server import build_agent_factory

    server_settings = Settings(
        model="server-model", api_key="server-key", base_url="http://x/v1", workspace=str(tmp_path)
    )
    factory = build_agent_factory(server_settings)

    agent = factory(
        ReviewOptions(workspace=tmp_path, api_key="client-key", overrides={"model": "client-model"})
    )

    assert isinstance(agent.client, LLMClient)
    assert agent.client.model == "client-model"
    # 服务端 Settings 不应被请求影响
    assert server_settings.api_key == "server-key"
    assert server_settings.model == "server-model"


def test_agent_factory_falls_back_to_server_key(tmp_path) -> None:
    from code_review_agent.config import Settings
    from code_review_agent.web.server import build_agent_factory

    server_settings = Settings(
        model="m", api_key="server-key", base_url="http://x/v1", workspace=str(tmp_path)
    )
    agent = build_agent_factory(server_settings)(ReviewOptions(workspace=tmp_path))

    assert agent.client.model == "m"


def test_agent_factory_rejects_base_url_override(tmp_path) -> None:
    """即使有人绕过 parse_overrides 直接构造 ReviewOptions，也不该生效。"""
    from code_review_agent.config import Settings
    from code_review_agent.web.server import build_agent_factory

    server_settings = Settings(
        model="m", api_key="k", base_url="http://safe/v1", workspace=str(tmp_path)
    )
    factory = build_agent_factory(server_settings)
    agent = factory(
        ReviewOptions(workspace=tmp_path, overrides={"base_url": "http://evil/v1"})
    )

    # base_url 不在白名单里，因此仍应指向服务端配置的地址
    assert "evil" not in str(agent.client._client.base_url)  # noqa: SLF001


def test_agent_factory_without_any_key_raises(tmp_path) -> None:
    from code_review_agent.config import Settings
    from code_review_agent.llm import LLMConfigError
    from code_review_agent.web.server import build_agent_factory

    factory = build_agent_factory(Settings(model="m", api_key="", workspace=str(tmp_path)))

    with pytest.raises(LLMConfigError):
        factory(ReviewOptions(workspace=tmp_path))


def test_stats_event_is_emitted_during_run_not_only_at_end(options: ReviewOptions) -> None:
    """回归：统计必须随工具返回增量刷新。

    早期版本只在任务结束时发一次 stats，导致界面上的步数与 token
    在整个审查过程中一直显示 0——用户看不到任何进展。
    """
    agent = FakeAgent()
    agent.script = [
        [
            AgentEvent(type="start", text="开始"),
            AgentEvent(type="assistant", text="先看目录", step=1),
            AgentEvent(type="tool_call", tool="list_directory", arguments={}, step=1),
            AgentEvent(type="tool_result", tool="list_directory", text="app/", step=1),
            AgentEvent(type="assistant", text="再分析", step=2),
            AgentEvent(type="tool_call", tool="analyze_code", arguments={}, step=2),
            AgentEvent(type="tool_result", tool="analyze_code", text="发现 3 条", step=2),
            AgentEvent(type="final", text="# 报告", step=3),
        ]
    ]
    session = Session("s_stats", agent, options)

    session.start_review()
    _wait_idle(session)

    events = session.snapshot()
    stats_events = [e for e in events if e["type"] == "stats"]

    # 两次工具调用各触发一次增量刷新，外加结束时的一次
    assert len(stats_events) >= 3, [e["type"] for e in events]

    # 第一次刷新必须发生在 final 之前，否则界面上就是「全程 0，最后跳变」
    index_of_first_stats = next(i for i, e in enumerate(events) if e["type"] == "stats")
    index_of_final = next(i for i, e in enumerate(events) if e["type"] == "final")
    assert index_of_first_stats < index_of_final

    first = stats_events[0]["data"]
    assert first["usage"]["total_tokens"] > 0, "增量刷新必须带上已消耗的 token"
    assert first["steps"] >= 1


def test_stats_is_not_emitted_twice_for_same_step(options: ReviewOptions) -> None:
    """同一 step 内的多个工具调用只刷新一次，避免刷屏。"""
    agent = FakeAgent()
    agent.script = [
        [
            AgentEvent(type="tool_call", tool="read_file", arguments={}, step=1),
            AgentEvent(type="tool_result", tool="read_file", text="a", step=1),
            AgentEvent(type="tool_call", tool="search_code", arguments={}, step=1),
            AgentEvent(type="tool_result", tool="search_code", text="b", step=1),
            AgentEvent(type="final", text="完成", step=2),
        ]
    ]
    session = Session("s_stats2", agent, options)

    session.start_review()
    _wait_idle(session)

    stats_events = [e for e in session.snapshot() if e["type"] == "stats"]
    # step=1 的两次工具返回只产生一次增量刷新 + 结束时一次 = 2
    assert len(stats_events) == 2, [e["step"] for e in stats_events]


def test_error_event_carries_usage(options: ReviewOptions) -> None:
    """出错时也应能看到已消耗的额度，便于判断成本。"""

    class ExplodingAgent(FakeAgent):
        def review(self, **kwargs: Any) -> Iterator[AgentEvent]:
            raise RuntimeError("模拟失败")
            yield  # pragma: no cover - 仅为让本方法成为生成器函数

    session = Session("s_stats3", ExplodingAgent(), options)
    session.start_review()
    _wait_idle(session)

    errors = [e for e in session.snapshot() if e["type"] == "error"]
    assert errors
    # FakeAgent.stats() 返回 total_tokens=1540，应随错误事件一并上报
    assert errors[0].get("data", {}).get("usage", {}).get("total_tokens") == 1540


# --------------------------------------------------------------------------- #
# 帮助函数
# --------------------------------------------------------------------------- #
def _wait_idle(session: Session, timeout: float = 5.0) -> None:
    """等待会话的后台任务结束。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not session.busy and session.status in {"ready", "error"}:
            return
        time.sleep(0.01)
    raise AssertionError(f"会话在 {timeout}s 内未结束（status={session.status}）")
