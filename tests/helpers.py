"""测试用的假 LLM 实现（不属于被测代码）。

抽成独立模块而不是塞进 ``conftest.py``，是为了让测试可以用普通绝对导入
``from tests.helpers import ...`` 拿到构造器，不依赖 pytest 是否把 rootdir
插入 ``sys.path``，在任何运行方式（pytest / IDE / python -m pytest）下都稳定。
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Sequence


def text_response(content: str, *, total_tokens: int = 10) -> SimpleNamespace:
    """构造一个「只返回文本」的假响应。"""
    message = SimpleNamespace(content=content, tool_calls=[])
    return _wrap(message, total_tokens)


def tool_response(
    calls: Sequence[tuple[str, dict[str, Any]]],
    *,
    content: str = "",
    total_tokens: int = 10,
) -> SimpleNamespace:
    """构造一个「返回工具调用」的假响应。calls 为 (工具名, 参数字典) 序列。"""
    tool_calls = [
        SimpleNamespace(
            id=f"call_{index}",
            type="function",
            function=SimpleNamespace(
                name=name,
                arguments=json.dumps(args, ensure_ascii=False),
            ),
        )
        for index, (name, args) in enumerate(calls)
    ]
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return _wrap(message, total_tokens)


def _wrap(message: SimpleNamespace, total_tokens: int) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(
            prompt_tokens=total_tokens // 2,
            completion_tokens=total_tokens // 2,
            total_tokens=total_tokens,
        ),
    )


class FakeCompletions:
    """模拟 ``client.chat.completions``。"""

    def __init__(self, owner: "FakeOpenAI") -> None:
        self._owner = owner

    def create(self, **kwargs: Any) -> Any:
        self._owner.calls.append(kwargs)
        if not self._owner.script:
            raise AssertionError("假 LLM 的脚本已用尽，但 Agent 仍在请求模型")
        item = self._owner.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeOpenAI:
    """模拟 ``openai.OpenAI`` 的最小接口：只实现 chat.completions.create。"""

    def __init__(self, script: Sequence[Any] | None = None) -> None:
        self.script = list(script or [])
        self.calls: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=FakeCompletions(self))
