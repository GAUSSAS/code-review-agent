"""HTTP 层测试：真正启动 http.server 并发起请求，但 Agent 用替身，全程离线。

覆盖点：
- 路由与状态码（含各类错误输入）
- SSE 端到端：POST 创建会话 → GET 流式接收 → 服务端主动关闭
- 会话隔离与并发拒绝（409）
- 参数校验：路径穿越、不存在的目录、越界 max_steps
"""

from __future__ import annotations

import http.client
import json
import threading
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import pytest

from code_review_agent.agent import AgentEvent
from code_review_agent.config import Settings
from code_review_agent.web.server import WebConfig, create_server
from tests.test_web import FakeAgent, basic_events

#: 前端源码路径：基于本文件定位，避免依赖 pytest 的当前工作目录
INDEX_HTML = Path(__file__).resolve().parent.parent / "code_review_agent" / "web" / "static" / "index.html"


# --------------------------------------------------------------------------- #
# 服务夹具
# --------------------------------------------------------------------------- #
class Harness:
    """封装一个跑在后台线程里的测试服务器，并提供便捷请求方法。

    ``factory`` 传 None 时使用**真实的** Agent 工厂，
    这样才能覆盖「未配置 API Key」等生产分支。
    """

    def __init__(self, root, factory=None, *, api_key: str = "fake-key", allowed_roots=()) -> None:
        self.settings = Settings(
            model="fake-model",
            api_key=api_key,
            base_url="http://fake.local/v1",
            workspace=str(root),
        )
        self.config = WebConfig(
            host="127.0.0.1", port=0, root=root, allowed_roots=tuple(allowed_roots)
        )
        self.agents: list[FakeAgent] = []

        if factory is not None:
            def recording_factory(options):  # noqa: ANN001
                agent = factory(options)
                self.agents.append(agent)
                return agent

            factory = recording_factory

        self.server = create_server(self.settings, self.config, factory=factory)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    # -- 请求辅助 --
    def get(self, path: str) -> tuple[int, Any]:
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        try:
            conn.request("GET", path)
            resp = conn.getresponse()
            raw = resp.read()
            try:
                return resp.status, json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return resp.status, raw.decode("utf-8", errors="replace")
        finally:
            conn.close()

    def post(self, path: str, payload: dict[str, Any]) -> tuple[int, Any]:
        body = json.dumps(payload).encode("utf-8")
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        try:
            conn.request("POST", path, body=body, headers={"Content-Type": "application/json"})
            resp = conn.getresponse()
            raw = resp.read()
            try:
                return resp.status, json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                return resp.status, raw.decode("utf-8", errors="replace")
        finally:
            conn.close()

    def read_sse(self, sid: str, *, max_bytes: int = 200_000) -> str:
        """读取 SSE 流直到服务端主动关闭。"""
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=15)
        try:
            conn.request("GET", "/api/stream?" + urlencode({"sid": sid}))
            resp = conn.getresponse()
            assert resp.status == 200, resp.status
            assert "text/event-stream" in resp.getheader("Content-Type", "")
            chunks: list[bytes] = []
            total = 0
            while True:
                line = resp.fp.readline()
                if not line:
                    break
                chunks.append(line)
                total += len(line)
                if total > max_bytes:
                    break
            return b"".join(chunks).decode("utf-8", errors="replace")
        finally:
            conn.close()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


@pytest.fixture
def harness(tmp_path):
    created: list[Harness] = []

    def make(script_factory=_simple_agent, *, use_real_factory: bool = False,
             api_key: str = "fake-key", allowed_roots=()) -> Harness:
        factory = None if use_real_factory else script_factory
        item = Harness(tmp_path, factory, api_key=api_key, allowed_roots=allowed_roots)
        created.append(item)
        return item

    yield make

    for item in created:
        item.close()


def _simple_agent(options) -> FakeAgent:  # noqa: ANN001
    return FakeAgent([basic_events()])


# --------------------------------------------------------------------------- #
# 页面与基础接口
# --------------------------------------------------------------------------- #
def test_index_page_served(harness) -> None:
    h = harness(_simple_agent)

    status, html = h.get("/")

    assert status == 200
    assert isinstance(html, str)
    assert "代码审查 Agent" in html
    assert "__APP_CONFIG__" not in html, "配置占位符应已被替换"
    assert "api/stream" in html


def test_health_reports_configuration(harness) -> None:
    h = harness(_simple_agent)

    status, data = h.get("/api/health")

    assert status == 200
    assert data["ok"] is True
    assert data["api_key_configured"] is True
    assert data["model"] == "fake-model"
    assert len(data["tools"]) == 7
    assert "analyze_code" in data["tools"]


def test_tools_endpoint_lists_schemas(harness) -> None:
    h = harness(_simple_agent)

    status, data = h.get("/api/tools")

    assert status == 200
    names = [t["name"] for t in data["tools"]]
    assert names == sorted(names)
    assert "run_tests" in names
    for tool in data["tools"]:
        assert tool["description"]
        assert tool["parameters"]["type"] == "object"


def test_unknown_path_returns_404_json(harness) -> None:
    h = harness(_simple_agent)

    status, data = h.get("/api/nope")

    assert status == 404
    assert data["ok"] is False


# --------------------------------------------------------------------------- #
# 被审查目录白名单（--allow-root）
# --------------------------------------------------------------------------- #
def test_health_lists_permitted_roots(harness, tmp_path) -> None:
    extra = tmp_path / "extra-project"
    extra.mkdir()
    h = harness(_simple_agent, allowed_roots=[extra])

    status, data = h.get("/api/health")

    assert status == 200
    assert str(tmp_path.resolve()) in data["roots"]
    assert str(extra.resolve()) in data["roots"]
    assert data["allow_roots"] == [str(extra.resolve())]


def test_review_accepts_path_from_allowed_root(harness, tmp_path) -> None:
    """--allow-root 指定的目录应当可以被审查。"""
    extra = tmp_path / "extra-project"
    extra.mkdir()
    (extra / "main.py").write_text("print('hi')\n", encoding="utf-8")
    h = harness(_simple_agent, allowed_roots=[extra])

    status, data = h.post("/api/review", {"workspace": str(extra)})

    assert status == 202
    assert data["workspace"] == str(extra.resolve())


def test_review_rejects_dir_outside_allowed_roots(harness, tmp_path) -> None:
    """不在白名单内的目录必须被拒绝，且错误信息要给出可操作建议。"""
    outside = tmp_path.parent / "outside-project"
    outside.mkdir(exist_ok=True)
    h = harness(_simple_agent)  # 未配置任何 allow_root

    status, data = h.post("/api/review", {"workspace": str(outside)})

    assert status == 400
    assert "不在允许范围内" in data["error"]
    assert "--allow-root" in data["error"]


def test_put_method_rejected(harness) -> None:
    h = harness(_simple_agent)

    conn = http.client.HTTPConnection("127.0.0.1", h.port, timeout=10)
    try:
        conn.request("PUT", "/api/review", body=b"{}")
        resp = conn.getresponse()
        assert resp.status == 405
        assert json.loads(resp.read().decode("utf-8"))["ok"] is False
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# 创建会话与参数校验
# --------------------------------------------------------------------------- #
def test_start_review_returns_session(harness) -> None:
    h = harness(_simple_agent)

    status, data = h.post("/api/review", {"workspace": ".", "focus": "安全"})

    assert status == 202
    assert data["ok"] is True
    assert data["session_id"]
    assert data["stream"].startswith("/api/stream?sid=")
    assert len(h.agents) == 1


def test_review_rejects_missing_workspace_dir(harness) -> None:
    h = harness(_simple_agent)

    status, data = h.post("/api/review", {"workspace": "不存在的目录"})

    assert status == 400
    assert "目录不存在" in data["error"]


def test_review_rejects_path_traversal(harness) -> None:
    h = harness(_simple_agent)

    status, data = h.post("/api/review", {"workspace": ".."})

    assert status == 400
    assert "不在允许范围内" in data["error"]


def test_review_rejects_file_outside_workspace(harness) -> None:
    h = harness(_simple_agent)

    status, data = h.post("/api/review", {"workspace": ".", "path": "../../secret.txt"})

    assert status == 400
    assert "工作区" in data["error"] or "不存在" in data["error"]


def test_review_rejects_bad_max_steps(harness) -> None:
    h = harness(_simple_agent)

    status, data = h.post("/api/review", {"workspace": ".", "max_steps": 999})

    assert status == 400
    assert "1 到 50" in data["error"]


def test_review_rejects_non_json_body(harness) -> None:
    h = harness(_simple_agent)

    conn = http.client.HTTPConnection("127.0.0.1", h.port, timeout=10)
    try:
        conn.request("POST", "/api/review", body=b"not json", headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        assert resp.status == 400
        assert "合法 JSON" in json.loads(resp.read().decode("utf-8"))["error"]
    finally:
        conn.close()


def test_review_without_api_key_returns_503(harness) -> None:
    """服务端没配 Key 且页面也没填时，创建会话应返回 503 而不是 500。"""
    h = harness(use_real_factory=True, api_key="")

    status, data = h.post("/api/review", {"workspace": "."})

    assert status == 503
    assert data["ok"] is False
    assert "LLM_API_KEY" in data["error"]


def test_review_accepts_client_supplied_key(harness) -> None:
    """页面填写的 Key 应能被接受：服务端无 Key 时也能发起审查。"""
    h = harness(use_real_factory=True, api_key="")

    status, data = h.post(
        "/api/review", {"workspace": ".", "api_key": "sk-from-browser-123456"}
    )

    assert status == 202
    assert data["session_id"]


def test_review_rejects_overlong_client_key(harness) -> None:
    h = harness(_simple_agent)

    status, data = h.post("/api/review", {"workspace": ".", "api_key": "x" * 300})

    assert status == 400
    assert "过长" in data["error"]


def test_review_ignores_base_url_override(harness) -> None:
    """请求体里的 base_url 不在白名单内，必须被忽略（防 SSRF）。"""
    h = harness(use_real_factory=True, api_key="server-key")

    status, data = h.post(
        "/api/review",
        {"workspace": ".", "base_url": "http://evil.internal/v1"},
    )

    assert status == 202
    assert data["session_id"]
    # 服务端自身的 base_url 不受请求影响
    assert "evil" not in h.settings.base_url


# --------------------------------------------------------------------------- #
# 测试连接（/api/test-key）
# --------------------------------------------------------------------------- #
def test_test_key_requires_a_key(harness) -> None:
    h = harness(use_real_factory=True, api_key="")

    status, data = h.post("/api/test-key", {})

    assert status == 400
    assert "API Key" in data["error"]


def test_test_key_success_shape(harness, monkeypatch) -> None:
    """成功时返回模型名与回复摘要，且**绝不回显 Key 本身**。"""
    from code_review_agent.llm import ChatMessage, LLMClient

    def fake_chat(self, messages, tools=None):  # noqa: ANN001
        return ChatMessage(role="assistant", content="连通")

    monkeypatch.setattr(LLMClient, "chat", fake_chat)
    h = harness(_simple_agent)
    secret = "sk-super-secret-key-abcdef"

    status, data = h.post("/api/test-key", {"api_key": secret, "model": "deepseek-reasoner"})

    assert status == 200
    assert data["ok"] is True
    assert data["reply"] == "连通"
    assert data["model"] == "deepseek-reasoner"
    body = json.dumps(data, ensure_ascii=False)
    assert secret not in body, "响应体绝不能包含 API Key"


def test_test_key_reports_failure(harness, monkeypatch) -> None:
    from code_review_agent.llm import LLMClient, LLMConfigError

    def failing_chat(self, messages, tools=None):  # noqa: ANN001
        raise LLMConfigError("鉴权失败（401）：API Key 无效或与 base_url 不匹配。")

    monkeypatch.setattr(LLMClient, "chat", failing_chat)
    h = harness(_simple_agent)

    status, data = h.post("/api/test-key", {"api_key": "sk-bad"})

    # 业务失败也用 200 返回，由 ok 字段区分，便于前端统一处理
    assert status == 200
    assert data["ok"] is False
    assert "401" in data["error"]


# --------------------------------------------------------------------------- #
# SSE 端到端
# --------------------------------------------------------------------------- #
def test_sse_stream_delivers_events_and_closes(harness) -> None:
    h = harness(_simple_agent)
    status, data = h.post("/api/review", {"workspace": "."})
    assert status == 202

    payload = h.read_sse(data["session_id"])

    assert payload.startswith(": connected")
    assert "event: start" in payload
    assert "event: tool_call" in payload
    assert "event: tool_result" in payload
    assert "event: final" in payload
    assert "event: done" in payload
    assert payload.rstrip().endswith(": closed") or "event: closed" in payload


def test_sse_unknown_session_returns_404(harness) -> None:
    h = harness(_simple_agent)

    status, data = h.get("/api/stream?sid=deadbeef")

    assert status == 404
    assert "会话不存在" in data["error"]


def test_sse_missing_sid_returns_400(harness) -> None:
    h = harness(_simple_agent)

    status, data = h.get("/api/stream")

    assert status == 400
    assert "sid" in data["error"]


# --------------------------------------------------------------------------- #
# 追问
# --------------------------------------------------------------------------- #
def test_chat_requires_existing_session(harness) -> None:
    h = harness(_simple_agent)

    status, data = h.post("/api/chat", {"session_id": "nope", "message": "在吗"})

    assert status == 404


def test_chat_rejects_empty_message(harness) -> None:
    h = harness(_simple_agent)
    _, created = h.post("/api/review", {"workspace": "."})
    h.read_sse(created["session_id"])  # 等审查结束

    status, data = h.post("/api/chat", {"session_id": created["session_id"], "message": "   "})

    assert status == 400
    assert "message" in data["error"]


def test_chat_appends_to_same_agent(harness) -> None:
    def factory(options):  # noqa: ANN001
        return FakeAgent([basic_events(), [AgentEvent(type="final", text="追问答案")]])

    h = harness(factory)
    _, created = h.post("/api/review", {"workspace": "."})
    h.read_sse(created["session_id"])

    status, data = h.post(
        "/api/chat", {"session_id": created["session_id"], "message": "再详细说说"}
    )
    assert status == 202

    payload = h.read_sse(created["session_id"])
    assert "追问答案" in payload
    # 关键：仍然是同一个 Agent 实例（记忆得以延续）
    assert len(h.agents) == 1
    assert h.agents[0].chat_messages == ["再详细说说"]


# --------------------------------------------------------------------------- #
# 主动拉取统计（/api/stats）
# --------------------------------------------------------------------------- #
def test_stats_endpoint_returns_snapshot(harness) -> None:
    h = harness(_simple_agent)
    _, created = h.post("/api/review", {"workspace": "."})
    h.read_sse(created["session_id"])

    status, data = h.get("/api/stats?sid=" + created["session_id"])

    assert status == 200
    assert data["ok"] is True
    assert data["session_id"] == created["session_id"]
    assert data["status"] == "ready"
    assert data["stats"]["tool_calls"] == 2
    assert data["stats"]["usage"]["total_tokens"] == 1540


def test_stats_endpoint_unknown_session(harness) -> None:
    h = harness(_simple_agent)

    status, data = h.get("/api/stats?sid=nope")

    assert status == 404
    assert "会话不存在" in data["error"]


def test_stats_endpoint_requires_sid(harness) -> None:
    h = harness(_simple_agent)

    status, data = h.get("/api/stats")

    assert status == 400
    assert "sid" in data["error"]


def test_stats_snapshot_reports_running_state(harness) -> None:
    """任务进行中拉取统计也应成功，并反映 busy 状态。"""
    import threading

    started = threading.Event()
    release = threading.Event()

    class BlockingAgent(FakeAgent):
        def review(self, **kwargs):  # noqa: ANN001
            yield AgentEvent(type="start", text="开始")
            started.set()
            release.wait(timeout=5)
            yield AgentEvent(type="final", text="完成")

    h = harness(lambda options: BlockingAgent())
    _, created = h.post("/api/review", {"workspace": "."})
    assert started.wait(timeout=5), "后台任务未启动"

    status, data = h.get("/api/stats?sid=" + created["session_id"])

    assert status == 200
    assert data["busy"] is True
    assert data["status"] == "running"
    assert data["stats"]["usage"]["total_tokens"] == 1540

    release.set()


def test_index_page_wires_all_configurable_fields(harness) -> None:
    """回归：界面上**可配置**的字段必须真的被发送到后端。

    曾经 `max_steps` 输入框存在于表单、后端也支持，但前端组装请求体时漏了它，
    表现就是「改了最大步数却仍然跑 12 步」，且没有任何提示。
    这里做一层源码级检查，防止同类接线遗漏再次发生。
    """
    import re

    html = INDEX_HTML.read_text(encoding="utf-8")
    match = re.search(r"async function startReview\(\).*?\n\}", html, re.DOTALL)
    assert match, "未找到 startReview 函数"
    body = match.group(0)

    for field in ("workspace", "focus", "language", "save_report", "interactive", "max_steps"):
        assert field in body, f"startReview 未发送字段 {field}"

    # api_key 是条件添加，单独确认
    assert "api_key" in body


def test_index_page_has_matching_input_for_each_sent_field(harness) -> None:
    """发送的字段应当在界面上有对应的输入控件，避免「发了但用户改不了」。"""
    html = INDEX_HTML.read_text(encoding="utf-8")

    for element_id in ("workspace", "browseBtn", "rootSelect", "focus", "language",
                       "maxSteps", "apiKey", "saveReport", "interactive"):
        assert f'id="{element_id}"' in html, f"缺少输入控件 {element_id}"


def test_index_page_provides_native_folder_picker(harness) -> None:
    """目录应能通过系统原生选择框来选，而不是只能手打路径。"""
    html = INDEX_HTML.read_text(encoding="utf-8")

    assert "async function browseFolder(" in html
    assert "/api/browse" in html
    assert "function currentWorkspace(" in html
    # 允许的根目录仍由服务端下发，作为快捷选择
    assert "h.roots" in html


def test_max_steps_validation_matches_backend_range(harness) -> None:
    """前端提示的范围必须与后端校验一致，否则会出现「前端放过、后端拒绝」。"""
    from code_review_agent.web.server import BadRequest, parse_max_steps

    html = INDEX_HTML.read_text(encoding="utf-8")
    assert "1~50" in html or "1 到 50" in html

    # 后端边界：1 与 50 合法，0 与 51 非法
    assert parse_max_steps(1) == 1
    assert parse_max_steps(50) == 50
    with pytest.raises(BadRequest):
        parse_max_steps(0)
    with pytest.raises(BadRequest):
        parse_max_steps(51)


# --------------------------------------------------------------------------- #
# 系统原生目录选择框（/api/browse）
# --------------------------------------------------------------------------- #
def test_browse_returns_selected_allowed_path(harness, tmp_path, monkeypatch) -> None:
    import code_review_agent.web.server as server_module

    picked = tmp_path / "picked"
    picked.mkdir()
    monkeypatch.setattr(server_module, "run_folder_dialog", lambda initial=None, **kw: (str(picked), None))
    h = harness(_simple_agent)

    status, data = h.post("/api/browse", {})

    assert status == 200
    assert data["ok"] is True
    assert data["allowed"] is True
    assert data["path"] == str(picked.resolve())
    assert data["root"] == str(tmp_path.resolve())


def test_browse_marks_outside_path_as_not_allowed(harness, tmp_path, monkeypatch) -> None:
    """选到白名单之外的目录：返回路径但标记不允许，并给出该加的参数。"""
    import code_review_agent.web.server as server_module

    outside = tmp_path.parent / "somewhere-else"
    outside.mkdir(exist_ok=True)
    monkeypatch.setattr(server_module, "run_folder_dialog", lambda initial=None, **kw: (str(outside), None))
    h = harness(_simple_agent)

    status, data = h.post("/api/browse", {})

    assert status == 200
    assert data["ok"] is True
    assert data["allowed"] is False
    assert data["path"] == str(outside.resolve())
    assert "--allow-root" in data["error"]


def test_browse_reports_cancel(harness, monkeypatch) -> None:
    import code_review_agent.web.server as server_module

    monkeypatch.setattr(server_module, "run_folder_dialog", lambda initial=None, **kw: (None, None))
    h = harness(_simple_agent)

    status, data = h.post("/api/browse", {})

    assert status == 200
    assert data["cancelled"] is True
    assert data["path"] is None


def test_browse_reports_dialog_error(harness, monkeypatch) -> None:
    """没有图形环境时应返回可读错误，而不是 500。"""
    import code_review_agent.web.server as server_module

    monkeypatch.setattr(
        server_module, "run_folder_dialog", lambda initial=None, **kw: (None, "无法初始化图形界面")
    )
    h = harness(_simple_agent)

    status, data = h.post("/api/browse", {})

    assert status == 200
    assert data["ok"] is False
    assert "图形界面" in data["error"]


def test_folder_dialog_script_exists_and_parses() -> None:
    """对话框脚本必须存在且能被独立运行（--help 应正常退出）。"""
    import subprocess
    import sys

    from code_review_agent.web.server import FOLDER_DIALOG_SCRIPT

    assert FOLDER_DIALOG_SCRIPT.is_file()

    completed = subprocess.run(  # noqa: S603
        [sys.executable, str(FOLDER_DIALOG_SCRIPT), "--help"],
        capture_output=True,
        text=True,
        timeout=60,
        shell=False,
    )
    assert completed.returncode == 0
    assert "--initial" in completed.stdout


def test_run_folder_dialog_drives_subprocess(workspace, monkeypatch) -> None:
    """run_folder_dialog 应把子进程 stdout 的路径原样返回。

    ``subprocess`` 是在函数内部导入的，所以只能替换**共享模块对象**上的 ``run``；
    替换随即由 monkeypatch 撤销，且测试执行期间 pytest 自身不会调用 subprocess。
    """
    import subprocess

    import code_review_agent.web.server as server_module

    class _Completed:
        returncode = 0
        stdout = str(workspace) + "\n"
        stderr = ""

    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _Completed())

    selected, error = server_module.run_folder_dialog(str(workspace))

    assert error is None
    assert selected == str(workspace)


def test_run_folder_dialog_returns_stderr_on_failure(monkeypatch) -> None:
    import subprocess

    import code_review_agent.web.server as server_module

    class _Completed:
        returncode = 3
        stdout = ""
        stderr = "无法初始化图形界面（可能没有桌面环境）\n"

    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: _Completed())

    selected, error = server_module.run_folder_dialog()

    assert selected is None
    assert "图形界面" in error


# --------------------------------------------------------------------------- #
# 启动行为：默认自动打开浏览器
# --------------------------------------------------------------------------- #
def test_serve_opens_browser_by_default(workspace, monkeypatch, capsys) -> None:
    """默认应自动打开浏览器，否则用户得手动复制地址。"""
    import code_review_agent.web.server as server_module

    captured: dict[str, str] = {}

    monkeypatch.setattr(server_module, "_open_browser", lambda url: captured.setdefault("url", url))
    monkeypatch.setattr(
        server_module, "create_server", lambda settings, config, **kw: _StubServer()
    )

    settings = Settings(model="m", api_key="k", base_url="http://x/v1", workspace=str(workspace))
    code = server_module.serve(settings, root=workspace, open_browser=True)

    assert code == 0
    assert captured.get("url") == "http://127.0.0.1:8765/"
    out = capsys.readouterr().out
    assert "访问地址" in out
    # 启动指纹：便于排查「跑的是旧代码」
    assert "代码版本" in out
    assert "已启用接口" in out
    assert "stats" in out


def test_serve_can_skip_browser(workspace, monkeypatch) -> None:
    """--no-browser 时不应尝试打开浏览器。"""
    import code_review_agent.web.server as server_module

    called = {"n": 0}

    def _spy(url: str) -> None:
        called["n"] += 1

    monkeypatch.setattr(server_module, "_open_browser", _spy)
    monkeypatch.setattr(
        server_module, "create_server", lambda settings, config, **kw: _StubServer()
    )

    settings = Settings(model="m", api_key="k", base_url="http://x/v1", workspace=str(workspace))
    code = server_module.serve(settings, root=workspace, open_browser=False)

    assert code == 0
    assert called["n"] == 0


def test_cli_web_passes_no_browser_flag(workspace, monkeypatch) -> None:
    """CLI 的 --no-browser 必须真正传到 serve()。"""
    import code_review_agent.cli as cli_module

    seen: dict[str, object] = {}

    def fake_serve(settings, **kwargs):  # noqa: ANN001
        seen.update(kwargs)
        return 0

    monkeypatch.setattr("code_review_agent.web.server.serve", fake_serve)

    assert cli_module.main(["web", "-w", str(workspace), "--no-browser", "--no-color"]) == 0
    assert seen.get("open_browser") is False

    seen.clear()
    assert cli_module.main(["web", "-w", str(workspace), "--no-color"]) == 0
    assert seen.get("open_browser") is True


def test_cli_web_passes_allow_root_arguments(workspace, monkeypatch, tmp_path) -> None:
    """--allow-root 可重复指定，且必须原样传给 serve()。"""
    import code_review_agent.cli as cli_module

    seen: dict[str, object] = {}

    def fake_serve(settings, **kwargs):  # noqa: ANN001
        seen.update(kwargs)
        return 0

    monkeypatch.setattr("code_review_agent.web.server.serve", fake_serve)
    extra = tmp_path / "another-project"
    extra.mkdir()

    code = cli_module.main(
        [
            "web",
            "-w", str(workspace),
            "--allow-root", str(extra),
            "--no-browser",
            "--no-color",
        ]
    )

    assert code == 0
    assert seen.get("allow_roots") == (str(extra),)


def test_open_browser_failure_is_swallowed(monkeypatch) -> None:
    """打不开浏览器不能把服务带崩。"""
    import code_review_agent.web.server as server_module

    def boom(*args, **kwargs):
        raise RuntimeError("no browser available")

    monkeypatch.setattr("webbrowser.open", boom)
    monkeypatch.setattr(server_module.os, "startfile", boom, raising=False)

    # 不应抛异常（工作线程内部吞掉）
    server_module._open_browser("http://127.0.0.1:1/")  # noqa: SLF001
    import time as _time

    _time.sleep(0.15)


class _StubServer:
    """供 serve() 测试使用的最小服务器替身。"""

    server_address = ("127.0.0.1", 8765)

    def serve_forever(self, poll_interval: float = 0.3) -> None:
        return

    def shutdown(self) -> None:
        return

    def server_close(self) -> None:
        return
