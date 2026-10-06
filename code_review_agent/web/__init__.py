"""Web 界面层。

刻意不在包初始化时导入 ``server``：CLI 的 ``web`` 子命令需要按需加载，
而测试常常只想导入 ``sessions`` / ``events`` 做单元测试，
避免牵连 ``http.server`` 的装配逻辑。
"""

from .events import EventPayload, serialize_event, summarize_tool_text
from .sessions import (
    AgentLike,
    ReviewOptions,
    Session,
    SessionStore,
    sse_frame,
    stream_session,
)

__all__ = [
    "AgentLike",
    "EventPayload",
    "ReviewOptions",
    "Session",
    "SessionStore",
    "serialize_event",
    "summarize_tool_text",
    "sse_frame",
    "stream_session",
]
