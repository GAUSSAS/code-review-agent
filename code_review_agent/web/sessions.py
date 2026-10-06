"""审查会话管理：后台线程运行 Agent，事件推入队列供 SSE 消费。

为什么需要会话对象
------------------
Agent 是**有状态**的：``ConversationMemory`` 承载了完整对话历史，
所以「审查完还能追问」必须复用同一个 ``ReviewAgent`` 实例。
HTTP 是无状态的，因此用一个带 id 的会话把状态挂在服务端。

并发模型
--------
- 每个会话一个后台 worker 线程，串行执行任务；
  不允许多个任务并发跑同一会话，避免记忆被交叉写入；
- 事件先写入内存列表，再通知所有订阅者（``Condition`` 广播）；
- 订阅者（SSE 生成器）只读不写，因此列表在锁内快照即可。
"""

from __future__ import annotations

import json
import logging
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Protocol

from ..agent import AgentError, AgentEvent, StepLimitExceeded
from ..llm import LLMConfigError
from .events import serialize_event

logger = logging.getLogger(__name__)

#: 单个会话最多保留多少条事件（防止超长任务把内存吃光）
MAX_EVENTS = 4000
#: SSE 空闲时的唤醒间隔（秒）
POLL_INTERVAL = 0.4
#: 单次 SSE 连接的空闲超时（秒）；前端会自动重连
IDLE_TIMEOUT = 180.0


class AgentLike(Protocol):
    """``ReviewAgent`` 中 Web 层实际用到的接口子集。

    只声明用到的部分，便于测试注入替身，也明确了 Web 层与 Agent 层的耦合边界。
    """

    workspace: Path

    def review(self, **kwargs: Any) -> Iterator[AgentEvent]: ...

    def chat_turn(self, user_input: str) -> Iterator[AgentEvent]: ...

    def force_conclusion(self) -> str: ...

    def stats(self) -> dict[str, Any]: ...


# --------------------------------------------------------------------------- #
# 请求参数
# --------------------------------------------------------------------------- #
@dataclass
class ReviewOptions:
    """一次审查请求的参数（已由服务端校验过）。

    ``api_key`` 与 ``overrides`` 来自前端表单，属于**易失配置**：
    只活在内存里，随会话结束而消失，绝不落盘。
    """

    workspace: Path
    #: 该工作区命中的「允许根」，用于界面回显（尤其在使用 --allow-root 时）
    root: Path | None = None
    path: str | None = None
    focus: str | None = None
    language_hint: str | None = None
    save_report: bool = False
    interactive: bool = True
    max_steps: int | None = None
    api_key: str | None = None
    overrides: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# 会话
# --------------------------------------------------------------------------- #
@dataclass
class Subscriber:
    """一个等待新事件的消费者（对应一条 SSE 连接）。"""

    condition: threading.Condition = field(default_factory=lambda: threading.Condition(threading.RLock()))
    cursor: int = 0


class Session:
    """一次审查会话：持有 Agent、事件缓冲与订阅者。"""

    def __init__(
        self,
        session_id: str,
        agent: AgentLike,
        options: ReviewOptions,
        *,
        max_events: int = MAX_EVENTS,
    ) -> None:
        self.id = session_id
        self.agent = agent
        self.options = options
        self.created_at = time.time()

        self._max_events = max_events
        self._lock = threading.RLock()
        self._events: list[dict[str, Any]] = []
        self._subscribers: list[Subscriber] = []
        self._worker: threading.Thread | None = None
        self._busy = False
        self._closed = False
        self._truncation_notified = False
        self._last_stats_step = 0
        self.status = "created"  # created | running | ready | error
        self.error: str | None = None
        self.report_path: str | None = None
        self.interactive = options.interactive

    # ------------------------------------------------------------ 事件缓冲 --
    def emit(
        self,
        event_type: str,
        *,
        text: str = "",
        tool: str = "",
        arguments: dict[str, Any] | None = None,
        ok: bool = True,
        step: int = 0,
        elapsed: float = 0.0,
        data: dict[str, Any] | None = None,
        truncated: bool = False,
    ) -> None:
        """写入一条事件并唤醒所有订阅者。

        刻意使用**显式关键字参数**而不是 ``**fields``：早期版本用 ``**fields``
        解包，一旦上游字段名对不上，Python 会报
        「emit() missing 1 required positional argument: 'event_type'」——
        这句话把真正的错误（上游少给了字段）说成了完全无关的方向，
        排查时极具误导性。显式签名能让错误直接指向出问题的字段。
        """
        payload: dict[str, Any] = {"type": event_type, "ok": ok, "step": step}
        if text:
            payload["text"] = text
        if tool:
            payload["tool"] = tool
        if arguments:
            payload["arguments"] = arguments
        if elapsed:
            payload["elapsed"] = round(elapsed, 2)
        if data:
            payload["data"] = data
        if truncated:
            payload["truncated"] = True

        with self._lock:
            if self._closed:
                return
            if len(self._events) >= self._max_events:
                # 用一次性的告警替代后续事件，避免内存无界增长
                if not self._truncation_notified:
                    self._truncation_notified = True
                    self._events.append(
                        {
                            "type": "error",
                            "ok": False,
                            "step": 0,
                            "text": f"事件数量超过上限 {self._max_events}，已停止记录后续事件。",
                        }
                    )
                    self._notify_all_locked()
                return
            self._events.append(payload)
            self._notify_all_locked()

    def emit_agent_event(self, event: AgentEvent) -> None:
        """把 AgentEvent 转成事件写入缓冲。

        这里逐个字段取值（而非 ``**payload``），是为了在字段缺失时立刻抛
        ``KeyError`` 并指明是哪个字段，而不是让错误在下游变了形。
        """
        payload = serialize_event(event).to_dict()
        self.emit(
            payload["type"],
            text=payload.get("text", ""),
            tool=payload.get("tool", ""),
            arguments=payload.get("arguments"),
            ok=payload.get("ok", True),
            step=payload.get("step", 0),
            elapsed=payload.get("elapsed", 0.0),
            data=payload.get("data"),
            truncated=payload.get("truncated", False),
        )

        # 工具返回后顺手刷新一次统计。
        # 早期版本只在任务结束时发一次 stats，导致界面上的步数/token
        # 在整个审查过程中一直是 0——用户完全看不到进展。
        if event.type == "tool_result":
            self._emit_stats_if_new_step(event.step)

    def _emit_stats_if_new_step(self, step: int) -> None:
        """同一个 step 内可能有多个工具调用，只在该 step 首次刷新，避免刷屏。"""
        if step <= self._last_stats_step:
            return
        self._last_stats_step = step
        payload = self._stats_payload()
        if payload:
            self.emit("stats", step=step, data=payload)

    def _notify_all_locked(self) -> None:
        for subscriber in self._subscribers:
            subscriber.condition.acquire()
            try:
                subscriber.condition.notify_all()
            finally:
                subscriber.condition.release()

    def subscribe(self) -> Subscriber:
        """注册一个事件订阅者（调用方持有其 condition 用于等待）。"""
        subscriber = Subscriber()
        with self._lock:
            self._subscribers.append(subscriber)
        return subscriber

    def unsubscribe(self, subscriber: Subscriber) -> None:
        with self._lock:
            try:
                self._subscribers.remove(subscriber)
            except ValueError:
                pass

    def events_since(self, cursor: int) -> tuple[list[dict[str, Any]], int]:
        with self._lock:
            return list(self._events[cursor:]), len(self._events)

    def snapshot(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._events)

    # ------------------------------------------------------------ 生命周期 --
    @property
    def busy(self) -> bool:
        with self._lock:
            return self._busy

    def close(self) -> None:
        with self._lock:
            self._closed = True
        self._notify_all_locked()

    def _start_worker(self, target: Callable[[], None]) -> bool:
        """启动后台任务；若已有任务在跑则返回 False。"""
        with self._lock:
            if self._busy:
                return False
            self._busy = True
            self.status = "running"

        def _run() -> None:
            try:
                target()
            except Exception as exc:  # noqa: BLE001 - 线程内异常必须自行兜住
                # 关键：把完整 traceback 一并报给前端。
                # 早期版本只上报 type(exc).__name__ + str(exc)，结果把
                # 「真正出错的调用点」掩盖了——比如本次的 TypeError，
                # 只看到「Session.emit() missing 1 required positional argument」，
                # 却无从知道是谁以错误的方式调用了它。
                detail = traceback.format_exc()
                logger.error("会话 %s 的后台任务异常：\n%s", self.id, detail)
                summary = f"{type(exc).__name__}: {exc}"
                self.status = "error"
                self.error = summary

                # 若存在链式异常（raise ... from ...），其根因往往才是关键
                cause = exc.__cause__ or exc.__context__
                if cause is not None:
                    summary += f"\n← 根因：{type(cause).__name__}: {cause}"

                # traceback 只保留本项目内的帧，避免被 SDK 内部帧淹没
                frames = [
                    line
                    for line in detail.strip().splitlines()
                    if "code_review_agent" in line or line.startswith(("Traceback", "  File"))
                ]
                tail = "\n".join(frames[-40:]) or detail[-2000:]

                self.emit(
                    "error",
                    ok=False,
                    text=f"服务端异常：{summary}\n\n```\n{tail}\n```",
                    data=self._stats_payload(),  # 即使失败也能看到已消耗的额度
                )
            finally:
                with self._lock:
                    self._busy = False
                self.emit("idle")
            self._notify_all_locked()

        self._worker = threading.Thread(target=_run, name=f"session-{self.id}", daemon=True)
        self._worker.start()
        return True

    # ------------------------------------------------------------ 任务 --
    def start_review(self) -> bool:
        return self._start_worker(self._review_worker)

    def chat(self, message: str) -> bool:
        return self._start_worker(lambda: self._chat_worker(message))

    def _review_worker(self) -> None:
        options = self.options
        kwargs: dict[str, Any] = {
            "path": options.path,
            "focus": options.focus,
            "language_hint": options.language_hint,
        }
        report = ""
        try:
            for event in self.agent.review(**kwargs):
                self.emit_agent_event(event)
                if event.type == "final":
                    report = event.text
        except StepLimitExceeded:
            self.emit(
                "assistant",
                text="已达到最大步数，改为基于已获取信息直接给出结论…",
                ok=False,
            )
            report = self.agent.force_conclusion()
            self.emit("final", text=report)
        except (AgentError, LLMConfigError) as exc:
            self.status = "error"
            self.error = str(exc)
            self.emit("error", ok=False, text=str(exc))
            return

        if options.save_report and report:
            self._write_report(report)

        self.status = "ready"
        self.emit("stats", data=self._stats_payload())
        self.emit("done", text="审查完成", data={"interactive": self.interactive})

    def _chat_worker(self, message: str) -> None:
        try:
            for event in self.agent.chat_turn(message):
                self.emit_agent_event(event)
        except StepLimitExceeded:
            report = self.agent.force_conclusion()
            self.emit("final", text=report)
        except (AgentError, LLMConfigError) as exc:
            self.emit("error", ok=False, text=str(exc))
            return
        self.emit("stats", data=self._stats_payload())
        self.emit("done", text="回答完成", data={"interactive": True})

    def _stats_payload(self) -> dict[str, Any]:
        try:
            return self.agent.stats()
        except Exception:  # noqa: BLE001 - 统计失败不该影响主流程
            return {}

    def stats_snapshot(self) -> dict[str, Any]:
        """供 /api/stats 主动拉取：当前权威统计 + 会话状态。

        前端手动刷新统计时用它。它读取的是同一个 Agent 实例，
        因此即使事件流（SSE）出了问题，这个值仍然是对的——
        这正是「手动刷新」作为诊断手段的价值。
        """
        return {
            "session_id": self.id,
            "status": self.status,
            "busy": self.busy,
            "elapsed": round(time.time() - self.created_at, 2),
            "error": self.error,
            "stats": self._stats_payload(),
        }

    def _write_report(self, report: str) -> None:
        target = self.options.workspace / "review_report.md"
        try:
            target.write_text(report.rstrip() + "\n", encoding="utf-8")
            self.report_path = str(target)
            self.emit("report_saved", text=f"报告已保存到 {target}", data={"path": str(target)})
        except OSError as exc:
            self.emit("error", ok=False, text=f"报告保存失败：{exc}")


# --------------------------------------------------------------------------- #
# 会话仓库
# --------------------------------------------------------------------------- #
class SessionStore:
    """进程内的会话表。单机本地工具，用字典足够。"""

    def __init__(self, *, ttl_seconds: float = 3600.0, max_sessions: int = 20) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = threading.RLock()
        self._ttl = ttl_seconds
        self._max = max_sessions

    def create(self, agent: AgentLike, options: ReviewOptions) -> Session:
        session_id = uuid.uuid4().hex[:12]
        session = Session(session_id, agent, options)
        with self._lock:
            self._prune_locked()
            # 超出上限时淘汰最旧的**空闲**会话，正在跑的保留
            if len(self._sessions) >= self._max:
                for old_id, old in sorted(self._sessions.items(), key=lambda kv: kv[1].created_at):
                    if not old.busy:
                        old.close()
                        del self._sessions[old_id]
                        break
            self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        with self._lock:
            return self._sessions.get(session_id)

    def delete(self, session_id: str) -> None:
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session:
            session.close()

    def count(self) -> int:
        with self._lock:
            return len(self._sessions)

    def _prune_locked(self) -> None:
        now = time.time()
        for session_id, session in list(self._sessions.items()):
            if session.busy:
                continue
            if now - session.created_at > self._ttl:
                session.close()
                del self._sessions[session_id]


# --------------------------------------------------------------------------- #
# SSE 流
# --------------------------------------------------------------------------- #
def sse_frame(event_type: str, data: dict[str, Any]) -> str:
    """把一条事件编码为 SSE 报文。"""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event_type}\ndata: {payload}\n\n"


def stream_session(
    session: Session,
    *,
    poll_interval: float = POLL_INTERVAL,
    idle_timeout: float = IDLE_TIMEOUT,
    keepalive_every: float = 15.0,
) -> Iterator[str]:
    """把会话事件转成 SSE 帧流。

    - 从游标 0 开始，因此**先连接、后开跑**也不会漏事件；
    - 空闲时定期发注释行（``: ping``）保活，避免代理断开长连接；
    - ``idle_timeout`` 秒内没有任何事件且任务已结束，则正常关闭连接。
    """
    subscriber = session.subscribe()

    last_activity = time.time()
    last_ping = time.time()
    first = True
    try:
        while True:
            if first:
                # SSE 约定：连接建立后先发一个注释，让前端确认通道就绪
                yield ": connected\n\n"
                first = False

            events, cursor = session.events_since(subscriber.cursor)
            if events:
                subscriber.cursor = cursor
                last_activity = time.time()
                for event in events:
                    yield sse_frame(event.get("type", "message"), event)
                continue

            # 没有任何新事件：先保活，再看是否该收尾
            now = time.time()
            if now - last_ping >= keepalive_every:
                last_ping = now
                yield ": ping\n\n"

            if not session.busy and session.status in {"ready", "error"} and cursor > 0:
                # 任务已结束且事件全部发完 → 主动关闭，前端据此停止自动重连
                yield sse_frame("closed", {"type": "closed", "text": "本次会话输出结束"})
                return

            if now - last_activity > idle_timeout and not session.busy:
                return

            with subscriber.condition:
                subscriber.condition.wait(timeout=poll_interval)
    finally:
        session.unsubscribe(subscriber)
