"""Web 服务：基于标准库 ``http.server`` 的零额外依赖实现。

为什么不用 Flask / FastAPI？
---------------------------
本项目的运行时依赖刻意只有 ``openai`` 一个包（见 DESIGN.md 的取舍说明）。
Web 界面是**附加的演示入口**，引入一个 Web 框架会连带拉入 WSGI/ASGI 服务器、
模板引擎等一串依赖，反而增加评审者的安装成本。标准库的 ``ThreadingHTTPServer``
配合 SSE 已经能完整支撑「边跑边看」的流式体验。

路由
----
===============  ======  ============================================
路径              方法    说明
===============  ======  ============================================
``/``            GET     单页界面
``/api/health``  GET     健康检查（模型、工具数、就绪状态）
``/api/tools``   GET     工具清单
``/api/review``  POST    创建会话并启动审查，返回 ``session_id``
``/api/stream``  GET     SSE 事件流（``?sid=``）
``/api/chat``    POST    在同一会话上追问（复用记忆）
===============  ======  ============================================
"""

from __future__ import annotations

import json
import logging
import os
import socket
import threading
from dataclasses import dataclass, field
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse

from ..agent import ReviewAgent
from ..config import PROVIDER_PRESETS, Settings
from ..llm import LLMConfigError
from .events import serialize_event  # noqa: F401 - 供调用方复用
from .sessions import ReviewOptions, Session, SessionStore, stream_session

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
INDEX_FILE = STATIC_DIR / "index.html"

#: 请求体上限（1MB），防止超大 body 打爆内存
MAX_BODY_BYTES = 1024 * 1024
#: 焦点/技术栈等自由文本的长度上限
MAX_TEXT_FIELD = 200
MAX_PATH_FIELD = 500
#: 前端可传入的 API Key 长度上限（正常 Key 远小于此）
MAX_API_KEY_LEN = 256
#: 前端可覆盖的字段白名单——**不要**允许覆盖 base_url，
#: 否则界面会变成一个「把服务端当跳板去请求任意地址」的 SSRF 入口
OVERRIDABLE_FIELDS = ("model", "temperature", "max_tokens", "timeout")
#: 弹出系统目录选择框的等待上限（秒）；用户可能停留较久，给足时间
BROWSE_TIMEOUT = 180.0
#: 目录选择框脚本路径
FOLDER_DIALOG_SCRIPT = Path(__file__).parent / "folder_dialog.py"


@dataclass
class WebConfig:
    """Web 服务配置。

    注意：是否自动打开浏览器由 :func:`serve` 的参数控制，
    不放在这里——它属于「启动动作」，不是服务本身的配置。
    """

    host: str = "127.0.0.1"
    port: int = 8765
    root: Path = field(default_factory=Path.cwd)
    #: 额外允许被审查的目录（--allow-root）。空元组表示只允许 root 之下。
    allowed_roots: tuple[Path, ...] = ()

    @property
    def permitted_roots(self) -> list[Path]:
        """默认根 + 显式允许的根，去重且顺序稳定。"""
        roots: list[Path] = []
        for candidate in (self.root, *self.allowed_roots):
            resolved = Path(candidate).resolve()
            if resolved not in roots:
                roots.append(resolved)
        return roots

    @classmethod
    def from_settings(cls, settings: Settings, **overrides: Any) -> "WebConfig":
        config = cls(root=Path.cwd())
        for key, value in overrides.items():
            if value is not None and hasattr(config, key):
                setattr(config, key, value)
        return config


# --------------------------------------------------------------------------- #
# 目录访问策略
# --------------------------------------------------------------------------- #
class DirectoryAccessPolicy:
    """决定「哪些目录允许被审查」。

    默认只允许服务启动目录之下的路径；通过 ``--allow-root`` 可以显式追加
    若干允许的根目录。这样既支持审查其他项目，又不会把界面变成
    「读取本机任意文件」的工具。

    校验一律在 ``resolve()`` 之后进行，避免 ``..`` 与符号链接绕过——
    用字符串前缀比较会被 ``/work`` 与 ``/work_evil`` 这类同前缀路径骗过。
    """

    def __init__(self, roots: list[Path] | tuple[Path, ...]) -> None:
        self.roots: list[Path] = []
        for candidate in roots:
            resolved = Path(candidate).resolve()
            if resolved not in self.roots:
                self.roots.append(resolved)

    def describe(self) -> str:
        return "、".join(str(r) for r in self.roots)

    def resolve(self, raw: str | None) -> tuple[Path, Path]:
        """把请求中的路径解析成 (绝对路径, 命中的根)。

        路径可以是绝对路径，也可以是相对于**默认根**（roots[0]）的路径。
        """
        text = (raw or ".").strip() or "."
        if len(text) > MAX_PATH_FIELD:
            raise BadRequest("workspace 路径过长")

        base = self.roots[0]
        candidate = Path(text)
        if not candidate.is_absolute():
            candidate = base / candidate

        try:
            resolved = candidate.resolve()
        except OSError as exc:
            raise BadRequest(f"无法解析路径：{exc}") from exc

        if not resolved.exists():
            raise BadRequest(f"目录不存在：{resolved}")
        if not resolved.is_dir():
            raise BadRequest(f"不是目录：{resolved}")

        matched = self.match(resolved)
        if matched is None:
            raise BadRequest(
                f"目录不在允许范围内：{resolved}\n"
                f"当前允许：{self.describe()}\n"
                f"如需审查该目录，请重启服务并追加参数：--allow-root \"{resolved}\""
            )
        return resolved, matched

    def match(self, resolved: Path) -> Path | None:
        """返回命中的根；不在任何根之下则返回 None。"""
        for root in self.roots:
            if resolved == root or resolved.is_relative_to(root):
                return root
        return None


# --------------------------------------------------------------------------- #
# 系统目录选择框
# --------------------------------------------------------------------------- #
def run_folder_dialog(
    initial: str | None = None,
    *,
    title: str = "选择被审查目录",
    timeout: float = BROWSE_TIMEOUT,
) -> tuple[str | None, str | None]:
    """弹出系统原生目录选择框，返回 ``(选中路径, 错误信息)``。

    实现说明：对话框跑在**独立子进程**里，原因有两个——

    1. ``tkinter`` 必须在主线程创建窗口，而 HTTP 请求处理在工作线程中，
       直接调用会抛 ``RuntimeError: main thread is not in main loop``；
    2. 用户可能在对话框里停留很久，独立进程不会长期占住服务器线程。

    「用户取消」返回 ``(None, None)``，与「出错」区分开。
    """
    import subprocess
    import sys as _sys

    if not FOLDER_DIALOG_SCRIPT.is_file():
        return None, f"缺少目录选择脚本：{FOLDER_DIALOG_SCRIPT}"

    if not _sys.executable:
        return None, "无法确定 Python 解释器路径，无法启动目录选择框"

    cmd = [_sys.executable, str(FOLDER_DIALOG_SCRIPT), "--title", title]
    if initial:
        cmd += ["--initial", initial]

    try:
        completed = subprocess.run(  # noqa: S603 - 命令由本模块内部构造
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            shell=False,
        )
    except subprocess.TimeoutExpired:
        return None, f"目录选择框等待超时（{int(timeout)} 秒）"
    except FileNotFoundError as exc:
        return None, f"无法启动目录选择进程：{exc}"
    except OSError as exc:
        return None, f"启动目录选择进程失败：{exc}"

    if completed.returncode != 0:
        detail = (completed.stderr or "").strip().splitlines()
        return None, (detail[-1] if detail else f"目录选择框异常退出（{completed.returncode}）")

    selected = (completed.stdout or "").strip()
    if not selected:
        return None, None  # 用户取消
    return selected, None


# --------------------------------------------------------------------------- #
# 参数校验
# --------------------------------------------------------------------------- #
class BadRequest(ValueError):
    """请求参数不合法。"""


def _require_text(data: dict[str, Any], key: str, *, max_len: int, required: bool = False) -> str | None:
    raw = data.get(key)
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        if required:
            raise BadRequest(f"缺少必填参数 `{key}`")
        return None
    if not isinstance(raw, str):
        raise BadRequest(f"参数 `{key}` 必须是字符串")
    value = raw.strip()
    if len(value) > max_len:
        raise BadRequest(f"参数 `{key}` 过长（上限 {max_len} 字符）")
    return value


def parse_review_options(data: dict[str, Any], config: WebConfig) -> ReviewOptions:
    """把请求体解析成 ReviewOptions，并在这一步完成全部校验。"""
    policy = DirectoryAccessPolicy(config.permitted_roots)
    workspace, matched_root = policy.resolve(
        _require_text(data, "workspace", max_len=MAX_PATH_FIELD)
    )
    target = _require_text(data, "path", max_len=MAX_PATH_FIELD)

    if target:
        # 只允许所选工作区内的相对文件路径
        candidate = (workspace / target).resolve()
        if not candidate.is_relative_to(workspace):
            raise BadRequest("`path` 超出了所选工作区范围")
        if not candidate.exists():
            raise BadRequest(f"指定的文件不存在：{target}")

    return ReviewOptions(
        workspace=workspace,
        root=matched_root,
        path=target,
        focus=_require_text(data, "focus", max_len=MAX_TEXT_FIELD),
        language_hint=_require_text(data, "language", max_len=MAX_TEXT_FIELD),
        save_report=bool(data.get("save_report")),
        interactive=bool(data.get("interactive", True)),
        max_steps=parse_max_steps(data.get("max_steps")),
        api_key=parse_api_key(data.get("api_key")),
        overrides=parse_overrides(data),
    )


def parse_max_steps(raw: Any) -> int | None:
    """校验最大步数：允许字符串或数字，范围 1~50。"""
    if raw in (None, ""):
        return None
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError) as exc:
        raise BadRequest("`max_steps` 必须是整数") from exc
    if not 1 <= value <= 50:
        raise BadRequest("`max_steps` 必须在 1 到 50 之间")
    return value


def parse_api_key(raw: Any) -> str | None:
    """解析前端提交的 API Key。

    安全约束（写在这里是为了让后续维护者不要放松它）：

    1. Key 只用于**本次请求构造 Agent**，不写入磁盘、不写日志、不存进全局状态；
    2. 绝不回显给任何客户端（包括健康检查接口）；
    3. 只允许填写 Key 本身，**不允许**借此覆盖 ``base_url``——否则界面会变成
       让服务端向任意地址发起请求的 SSRF 跳板。
    """
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise BadRequest("`api_key` 必须是字符串")
    key = raw.strip()
    if not key:
        return None
    if len(key) > MAX_API_KEY_LEN:
        raise BadRequest(f"`api_key` 过长（上限 {MAX_API_KEY_LEN} 字符）")
    if any(ch in key for ch in ("\r", "\n", "\0")):
        raise BadRequest("`api_key` 含有非法字符")
    return key


def parse_overrides(data: dict[str, Any]) -> dict[str, Any]:
    """解析可选的模型参数覆盖（白名单字段）。"""
    overrides: dict[str, Any] = {}

    model = data.get("model")
    if model not in (None, ""):
        if not isinstance(model, str) or len(model.strip()) > 100:
            raise BadRequest("`model` 不合法")
        overrides["model"] = model.strip()

    for field_name in ("temperature", "max_tokens", "timeout"):
        raw = data.get(field_name)
        if raw in (None, ""):
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise BadRequest(f"`{field_name}` 必须是数字") from exc
        if field_name == "temperature" and not 0.0 <= value <= 2.0:
            raise BadRequest("`temperature` 必须在 0 到 2 之间")
        if field_name == "max_tokens" and not 1 <= value <= 32000:
            raise BadRequest("`max_tokens` 必须在 1 到 32000 之间")
        if field_name == "timeout" and not 1 <= value <= 600:
            raise BadRequest("`timeout` 必须在 1 到 600 秒之间")
        overrides[field_name] = int(value) if field_name != "temperature" else value

    return overrides


# --------------------------------------------------------------------------- #
# Agent 工厂
# --------------------------------------------------------------------------- #
def build_agent_factory(settings: Settings) -> Callable[[ReviewOptions], ReviewAgent]:
    """返回一个「按请求构造 ReviewAgent」的工厂。

    每次请求都新建 Agent（因此每次都有独立的记忆）。配置优先级：

        服务端 .env / 环境变量  ←  请求携带的 api_key 与模型参数覆盖

    也就是说**前端可以临时覆盖**，但服务端的值仍是兜底默认；
    前端**不允许**覆盖 ``base_url``（防 SSRF，见 ``parse_api_key`` 的说明）。
    """

    def factory(options: ReviewOptions) -> ReviewAgent:
        overrides: dict[str, Any] = {"workspace": str(options.workspace)}
        if options.max_steps:
            overrides["max_steps"] = options.max_steps
        # 只接受白名单字段，避免前端塞入任意 Settings 属性
        for key in OVERRIDABLE_FIELDS:
            if key in options.overrides:
                overrides[key] = options.overrides[key]
        if options.api_key:
            overrides["api_key"] = options.api_key

        session_settings = Settings(**{**settings.__dict__, **overrides})
        if not session_settings.is_configured:
            raise LLMConfigError(
                "未配置 LLM_API_KEY：请在页面左侧填写 API Key，"
                "或创建 .env 文件并填写后重启服务。"
            )
        return ReviewAgent(session_settings)

    return factory


# --------------------------------------------------------------------------- #
# 请求处理
# --------------------------------------------------------------------------- #
class ReviewWebHandler(BaseHTTPRequestHandler):
    """处理单页界面与 JSON/SSE API。"""

    server_version = "CodeReviewAgentWeb/1.0"
    protocol_version = "HTTP/1.1"

    # 由 create_server 注入
    factory: Callable[[ReviewOptions], Any]
    store: SessionStore
    web_config: WebConfig
    settings: Settings

    # ------------------------------------------------------------ 基础设施 --
    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        logger.debug("%s - %s", self.address_string(), fmt % args)

    def _send(self, status: HTTPStatus | int, body: bytes, content_type: str) -> None:
        self.send_response(int(status))
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: HTTPStatus | int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _send_error_json(self, status: HTTPStatus | int, message: str) -> None:
        self._send_json(status, {"ok": False, "error": message})

    def _read_json_body(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError as exc:
            raise BadRequest("Content-Length 不合法") from exc
        if length <= 0:
            return {}
        if length > MAX_BODY_BYTES:
            raise BadRequest(f"请求体过大（上限 {MAX_BODY_BYTES // 1024}KB）")
        raw = self.rfile.read(length)
        try:
            parsed = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise BadRequest(f"请求体不是合法 JSON：{exc}") from exc
        if not isinstance(parsed, dict):
            raise BadRequest("请求体必须是 JSON 对象")
        return parsed

    def _query_sid(self) -> str:
        query = parse_qs(urlparse(self.path).query)
        sid = (query.get("sid") or [""])[0].strip()
        if not sid:
            raise BadRequest("缺少查询参数 `sid`")
        return sid

    # ------------------------------------------------------------ 路由 --
    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler 约定
        path = urlparse(self.path).path
        try:
            if path in {"/", "/index.html"}:
                self._serve_index()
            elif path == "/api/health":
                self._handle_health()
            elif path == "/api/tools":
                self._handle_tools()
            elif path == "/api/stats":
                self._handle_stats()
            elif path == "/api/stream":
                self._handle_stream()
            else:
                self._send_error_json(HTTPStatus.NOT_FOUND, f"未知路径：{path}")
        except BadRequest as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:  # noqa: BLE001 - 兜底，避免线程崩溃
            logger.exception("GET %s 处理失败", path)
            self._send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(exc).__name__}: {exc}")

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            if path == "/api/review":
                self._handle_review()
            elif path == "/api/chat":
                self._handle_chat()
            elif path == "/api/test-key":
                self._handle_test_key()
            elif path == "/api/browse":
                self._handle_browse()
            else:
                self._send_error_json(HTTPStatus.NOT_FOUND, f"未知路径：{path}")
        except BadRequest as exc:
            self._send_error_json(HTTPStatus.BAD_REQUEST, str(exc))
        except Exception as exc:  # noqa: BLE001
            logger.exception("POST %s 处理失败", path)
            self._send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(exc).__name__}: {exc}")

    # ------------------------------------------------------------ 具体处理 --
    def _serve_index(self) -> None:
        if not INDEX_FILE.is_file():
            self._send_error_json(
                HTTPStatus.INTERNAL_SERVER_ERROR, f"缺少前端文件：{INDEX_FILE}"
            )
            return

        html = INDEX_FILE.read_text(encoding="utf-8")
        injected = {
            "root": str(self.web_config.root),
            "model": self.settings.model,
            "provider": self.settings.provider,
            "tools": self._tool_names(),
        }
        # 用 JSON 注入配置；替换 < 防止提前闭合 script 标签
        payload = json.dumps(injected, ensure_ascii=False).replace("<", "\\u003c")
        html = html.replace("__APP_CONFIG__", payload)
        self._send(HTTPStatus.OK, html.encode("utf-8"), "text/html; charset=utf-8")

    def _tool_names(self) -> list[str]:
        try:
            return [tool["function"]["name"] for tool in self._specs()]
        except Exception:  # noqa: BLE001
            return []

    def _specs(self) -> list[dict[str, Any]]:
        from ..tools import ToolContext, build_default_registry

        ctx = ToolContext.create(self.web_config.root)
        return build_default_registry(ctx).specs()

    def _handle_health(self) -> None:
        roots = self.web_config.permitted_roots
        self._send_json(
            HTTPStatus.OK,
            {
                "ok": True,
                "provider": self.settings.provider,
                "model": self.settings.model,
                "base_url": self.settings.base_url,
                "api_key_configured": self.settings.is_configured,
                # 前端据此决定是否提示「请填写 Key」
                "accepts_client_key": True,
                "root": str(self.web_config.root),
                # 允许被审查的目录清单，前端用它渲染下拉框
                "roots": [str(p) for p in roots],
                "allow_roots": [str(p) for p in self.web_config.allowed_roots],
                "tools": self._tool_names(),
                "sessions": self.store.count(),
            },
        )

    def _handle_tools(self) -> None:
        from ..tools import ToolContext, build_default_registry

        ctx = ToolContext.create(self.web_config.root)
        registry = build_default_registry(ctx)
        tools = []
        for name in registry.names():
            tool = registry.get(name)
            tools.append(
                {
                    "name": name,
                    "description": tool.description if tool else "",
                    "parameters": tool.parameters if tool else {},
                }
            )
        self._send_json(HTTPStatus.OK, {"ok": True, "tools": tools})

    def _handle_stats(self) -> None:
        """主动拉取某个会话的当前统计（供界面上的「刷新统计」按钮使用）。"""
        sid = self._query_sid()
        session = self.store.get(sid)
        if session is None:
            self._send_error_json(HTTPStatus.NOT_FOUND, "会话不存在或已过期。")
            return
        self._send_json(HTTPStatus.OK, {"ok": True, **session.stats_snapshot()})

    def _handle_review(self) -> None:
        data = self._read_json_body()
        options = parse_review_options(data, self.web_config)

        try:
            agent = self.factory(options)
        except LLMConfigError as exc:
            self._send_error_json(HTTPStatus.SERVICE_UNAVAILABLE, str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            logger.exception("构造 Agent 失败")
            self._send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(exc).__name__}: {exc}")
            return

        session = self.store.create(agent, options)
        session.start_review()
        self._send_json(
            HTTPStatus.ACCEPTED,
            {
                "ok": True,
                "session_id": session.id,
                "workspace": str(options.workspace),
                "stream": f"/api/stream?sid={session.id}",
            },
        )

    def _handle_test_key(self) -> None:
        """用前端提交的 Key 发一次最小请求，验证「Key + 网络 + 模型名」是否可用。

        这是**唯一**会直接使用前端 Key 的只读接口：不创建会话、不写日志、
        不回显 Key 本身，只回答「通不通」。
        """
        data = self._read_json_body()
        api_key = parse_api_key(data.get("api_key")) or self.settings.api_key
        if not api_key:
            self._send_error_json(
                HTTPStatus.BAD_REQUEST, "没有可用的 API Key：请填写或先在服务端配置 .env。"
            )
            return

        overrides = parse_overrides(data)
        model = str(overrides.get("model") or self.settings.model)

        from ..llm import ChatMessage, LLMClient, LLMError

        try:
            client = LLMClient(
                model=model,
                api_key=api_key,
                base_url=self.settings.base_url,
                max_tokens=8,
                max_retries=0,
                timeout=20.0,
            )
            reply = client.chat(
                [ChatMessage(role="user", content="ping，请只回复两个字：连通")]
            )
        except LLMError as exc:
            self._send_json(
                HTTPStatus.OK,
                {"ok": False, "error": str(exc), "model": model},
            )
            return
        except Exception as exc:  # noqa: BLE001 - 未预期的异常也要转成可读信息
            logger.warning("测试 Key 时发生未预期异常：%s", type(exc).__name__)
            self._send_json(
                HTTPStatus.OK,
                {"ok": False, "error": f"{type(exc).__name__}: {exc}", "model": model},
            )
            return

        self._send_json(
            HTTPStatus.OK,
            {
                "ok": True,
                "model": model,
                "base_url": self.settings.base_url,
                "reply": (reply.content or "").strip()[:50],
                "total_tokens": client.usage.total_tokens,
            },
        )

    def _handle_browse(self) -> None:
        """弹出系统原生目录选择框，返回选中路径。

        安全说明：对话框本身允许浏览整个文件系统（原生行为），但**选中后仍要过白名单**。
        不在白名单内时返回 ``allowed: false`` 与提示，由前端标红——
        这样用户能立刻知道「选不了」以及该加什么启动参数，而不用等到提交审查才被拒。
        """
        data = self._read_json_body()
        initial = _require_text(data, "initial", max_len=MAX_PATH_FIELD)
        if initial and not Path(initial).expanduser().is_dir():
            initial = None  # 初始目录无效就退化成系统默认，不报错

        selected, error = run_folder_dialog(initial)
        if error:
            self._send_json(HTTPStatus.OK, {"ok": False, "error": error})
            return
        if not selected:
            self._send_json(HTTPStatus.OK, {"ok": True, "path": None, "cancelled": True})
            return

        policy = DirectoryAccessPolicy(self.web_config.permitted_roots)
        resolved = Path(selected).expanduser()
        try:
            resolved = resolved.resolve()
        except OSError as exc:
            self._send_json(HTTPStatus.OK, {"ok": False, "error": f"无法解析所选路径：{exc}"})
            return

        matched = policy.match(resolved)
        if matched is None:
            self._send_json(
                HTTPStatus.OK,
                {
                    "ok": True,
                    "path": str(resolved),
                    "allowed": False,
                    "error": (
                        f"该目录不在允许范围内。当前允许：{policy.describe()}。"
                        f'如需审查它，请重启服务并追加参数：--allow-root "{resolved}"'
                    ),
                },
            )
            return

        self._send_json(
            HTTPStatus.OK,
            {"ok": True, "path": str(resolved), "allowed": True, "root": str(matched)},
        )

    def _handle_chat(self) -> None:
        data = self._read_json_body()
        sid = _require_text(data, "session_id", max_len=64, required=True)
        message = _require_text(data, "message", max_len=2000, required=True)

        session = self.store.get(sid or "")
        if session is None:
            self._send_error_json(HTTPStatus.NOT_FOUND, "会话不存在或已过期，请重新发起审查。")
            return
        if session.busy:
            self._send_error_json(HTTPStatus.CONFLICT, "该会话仍在处理上一个任务，请稍候。")
            return
        if session.status == "error":
            self._send_error_json(HTTPStatus.CONFLICT, f"会话已因错误终止：{session.error}")
            return

        session.chat(message or "")
        self._send_json(HTTPStatus.ACCEPTED, {"ok": True, "session_id": session.id})

    def _handle_stream(self) -> None:
        sid = self._query_sid()
        session = self.store.get(sid)
        if session is None:
            self._send_error_json(HTTPStatus.NOT_FOUND, "会话不存在或已过期。")
            return

        self.send_response(int(HTTPStatus.OK))
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()

        try:
            for frame in stream_session(session):
                self.wfile.write(frame.encode("utf-8"))
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            # 浏览器主动断开（刷新/关闭页面）属正常情况
            logger.debug("SSE 客户端断开：%s", sid)
        except Exception:  # noqa: BLE001
            logger.exception("SSE 流异常：%s", sid)
        finally:
            self.close_connection = True

    # 明确拒绝未实现的写方法，避免默认 501 输出 HTML
    def do_DELETE(self) -> None:  # noqa: N802
        self._send_error_json(HTTPStatus.METHOD_NOT_ALLOWED, "不支持的方法")

    do_PUT = do_PATCH = do_DELETE


# --------------------------------------------------------------------------- #
# 服务器装配
# --------------------------------------------------------------------------- #
def create_server(
    settings: Settings,
    config: WebConfig,
    *,
    factory: Callable[[ReviewOptions], Any] | None = None,
) -> ThreadingHTTPServer:
    """构造（但不启动）Web 服务器。

    ``factory`` 可注入，测试里用来替换掉真实 LLM。
    """
    handler = partial(
        _make_handler,
        factory=factory or build_agent_factory(settings),
        store=SessionStore(),
        web_config=config,
        settings=settings,
    )

    class _Server(ThreadingHTTPServer):
        daemon_threads = True
        allow_reuse_address = True

    server = _Server((config.host, config.port), handler)
    server.daemon_threads = True
    return server


def _make_handler(
    *args: Any,
    factory: Callable[[ReviewOptions], Any],
    store: SessionStore,
    web_config: WebConfig,
    settings: Settings,
    **kwargs: Any,
) -> ReviewWebHandler:
    """把依赖注入到 handler 实例上（handler 由 socketserver 实例化，无法传参）。"""

    class _BoundHandler(ReviewWebHandler):
        pass

    _BoundHandler.factory = staticmethod(factory)  # type: ignore[assignment]
    _BoundHandler.store = store  # type: ignore[assignment]
    _BoundHandler.web_config = web_config  # type: ignore[assignment]
    _BoundHandler.settings = settings  # type: ignore[assignment]
    return _BoundHandler(*args, **kwargs)


def serve(
    settings: Settings,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    root: Path | None = None,
    open_browser: bool = True,
    allow_roots: Sequence[Path | str] = (),
) -> int:
    """启动 Web 服务并阻塞，直到 Ctrl+C。

    - ``open_browser`` 默认为 True：这是本地交互式工具，启动后自动打开界面
      才符合预期（脚本化场景可用 ``--no-browser`` 关闭）。
    - ``allow_roots`` 用于追加允许被审查的目录（``--allow-root``）。
      任何无法解析为目录的条目都会在启动阶段直接报错，避免运行到一半才发现。
    """
    default_root = (root or Path.cwd()).resolve()
    if not default_root.is_dir():
        raise NotADirectoryError(f"工作区目录不存在：{default_root}")

    extra_roots: list[Path] = []
    for raw in allow_roots:
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = default_root / candidate
        resolved = candidate.resolve()
        if not resolved.is_dir():
            print(f"--allow-root 指定的目录不可用：{resolved}")
            print("请确认该路径存在且是一个目录。")
            return 3
        if resolved not in extra_roots and resolved != default_root:
            extra_roots.append(resolved)

    config = WebConfig(
        host=host,
        port=port,
        root=default_root,
        allowed_roots=tuple(extra_roots),
    )

    try:
        server = create_server(settings, config)
    except OSError as exc:
        print(f"无法监听 {host}:{port} —— {exc}")
        print("端口可能被占用，请用 --port 指定其他端口，或用 --port 0 自动选择。")
        return 3

    actual_port = server.server_address[1]
    display_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    url = f"http://{display_host}:{actual_port}/"

    # 启动指纹：写清楚进程加载的是哪个版本与哪些接口。
    # 排查「界面行为像旧版」这类问题时，先对比这里的版本与接口列表，
    # 能立刻判断出是不是忘记重启服务。
    from .. import __version__

    routes = "health · tools · stats · review · test-key · stream · chat"

    print("=" * 68)
    print("  代码审查 Agent —— Web 界面已启动")
    print("=" * 68)
    print(f"  访问地址 : {url}")
    print(f"  代码版本 : v{__version__}")
    print(f"  已启用接口: {routes}")
    print(f"  服务目录 : {config.root}")
    if config.allowed_roots:
        print("  额外允许 : " + "　".join(str(p) for p in config.allowed_roots))
        print("             （由 --allow-root 指定，可审查这些目录下的代码）")
    print(f"  模型     : {settings.model}（{settings.provider}）")
    print(
        "  API Key  : "
        + ("服务端已配置" if settings.is_configured else "未配置（可在网页里直接填写）")
    )
    if host in {"0.0.0.0", "::"}:
        print("  警告     : 正在监听所有网卡，本工具无鉴权，请勿暴露到公网。")
    print("  按 Ctrl+C 停止服务")
    print("=" * 68)

    if open_browser:
        print("  正在打开浏览器…（打不开就手动复制上面的地址）")
        _open_browser(url)

    try:
        server.serve_forever(poll_interval=0.3)
    except KeyboardInterrupt:
        print("\n正在停止服务…")
    finally:
        server.shutdown()
        server.server_close()
    return 0

    try:
        server.serve_forever(poll_interval=0.3)
    except KeyboardInterrupt:
        print("\n正在停止服务…")
    finally:
        server.shutdown()
        server.server_close()
    return 0


def _open_browser(url: str) -> None:
    """尝试打开默认浏览器。

    失败不影响服务运行——远程会话、精简版系统、无桌面环境下
    ``webbrowser`` 可能找不到可用浏览器，此时用户手动访问即可。
    """

    def _worker() -> None:
        try:
            import webbrowser

            if webbrowser.open(url):
                return
            # 部分 Windows 环境下 webbrowser 找不到浏览器，用 os.startfile 兜底
            if hasattr(os, "startfile"):
                os.startfile(url)  # type: ignore[attr-defined]  # noqa: S606
        except Exception as exc:  # noqa: BLE001 - 打不开浏览器不能影响服务
            logger.info("自动打开浏览器失败（不影响使用）：%s", exc)

    threading.Thread(target=_worker, daemon=True).start()


def find_free_port(host: str = "127.0.0.1") -> int:
    """探测一个可用端口（供测试与 --port 0 场景使用）。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])
