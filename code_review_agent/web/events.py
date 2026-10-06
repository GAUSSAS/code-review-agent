"""把 ``AgentEvent`` 序列化成适合浏览器消费的 JSON。

这里是整个 Web 层**唯一**接触 Agent 内部结构的地方，与 CLI 的
``render_event()`` 地位对等——两者都只依赖 ``AgentEvent`` 这一层公开契约。

两个工程上的取舍：

1. **工具结果必须截断**。一次 ``analyze_code`` 可能返回 20KB 文本，
   原样塞进 SSE 会让浏览器端每秒解析上百 KB。前端只需要预览，
   因此截断到 ``MAX_TOOL_TEXT``，同时用 ``truncated`` 标记提示前端。
2. **推理文本不截断**。它通常很短，而且用户就是要读它。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..agent import AgentEvent

#: 工具结果文本的展示上限（字符）。超出部分只保留首尾。
MAX_TOOL_TEXT = 4000
#: 触发截断时，首尾各保留多少
HEAD_KEEP = 2500
TAIL_KEEP = 1200


def summarize_tool_text(text: str, *, limit: int = MAX_TOOL_TEXT) -> tuple[str, bool]:
    """按上限裁剪工具输出，返回 (文本, 是否被截断)。"""
    if len(text) <= limit:
        return text, False
    head = text[:HEAD_KEEP]
    tail = text[-TAIL_KEEP:]
    removed = len(text) - len(head) - len(tail)
    return f"{head}\n\n…（此处省略 {removed} 字符）…\n\n{tail}", True


@dataclass
class EventPayload:
    """事件的可序列化形式。"""

    type: str
    text: str = ""
    tool: str = ""
    arguments: dict[str, Any] | None = None
    ok: bool = True
    step: int = 0
    elapsed: float = 0.0
    data: dict[str, Any] | None = None
    truncated: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": self.type,
            "ok": self.ok,
            "step": self.step,
        }
        if self.text:
            payload["text"] = self.text
        if self.tool:
            payload["tool"] = self.tool
        if self.arguments:
            payload["arguments"] = self.arguments
        if self.elapsed:
            payload["elapsed"] = round(self.elapsed, 2)
        if self.data:
            payload["data"] = self.data
        if self.truncated:
            payload["truncated"] = True
        return payload


def serialize_event(event: AgentEvent) -> EventPayload:
    """把 AgentEvent 转成可 JSON 化的结构。"""
    text = event.text or ""
    truncated = False

    # 只有工具结果需要截断；其余事件文本都很短
    if event.type == "tool_result":
        text, truncated = summarize_tool_text(text)

    return EventPayload(
        type=event.type,
        text=text,
        tool=event.tool,
        arguments=dict(event.arguments) if event.arguments else None,
        ok=event.ok,
        step=event.step,
        elapsed=event.elapsed,
        data=dict(event.data) if event.data else None,
        truncated=truncated,
    )
