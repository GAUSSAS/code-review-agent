"""工具层：工具协议、注册表与执行框架。

设计目标：
- **单一职责**：``base`` 只负责「工具是什么、怎么注册、怎么执行」，
  具体能力放在 ``builtin`` / ``analyzer`` 中；
- **自描述**：工具用 JSON Schema 描述参数，直接转换为 OpenAI function-calling 格式；
- **失败不外抛**：工具异常统一转为 ``ToolResult(ok=False)`` 文本回喂给模型，
  让 Agent 有机会自我纠正，而不是让整个循环崩掉。
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """工具执行结果。"""

    content: str
    ok: bool = True
    meta: dict[str, Any] = field(default_factory=dict)

    def to_text(self) -> str:
        return self.content


@dataclass
class Tool:
    """一个可被 LLM 调用的工具。"""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., ToolResult]

    def to_spec(self) -> dict[str, Any]:
        """转换为 OpenAI Chat Completions 的 tools 元素。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """工具注册表：负责注册、导出 schema、分发调用。"""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self.call_log: list[dict[str, Any]] = []

    # ------------------------------------------------------------ 注册 --
    def register(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        handler: Callable[..., ToolResult],
    ) -> None:
        if name in self._tools:
            raise ValueError(f"工具名重复：{name}")
        self._tools[name] = Tool(name, description, parameters, handler)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def __len__(self) -> int:
        return len(self._tools)

    def specs(self) -> list[dict[str, Any]]:
        return [tool.to_spec() for tool in self._tools.values()]

    def describe(self) -> str:
        return "\n".join(
            f"- {t.name}: {t.description.splitlines()[0]}" for t in self._tools.values()
        )

    # ------------------------------------------------------------ 执行 --
    def execute(self, name: str, arguments: str | dict[str, Any]) -> ToolResult:
        """执行工具。任何异常都会被捕获并转成可读的失败结果。"""
        tool = self._tools.get(name)
        if tool is None:
            available = ", ".join(self.names())
            return ToolResult(
                content=f"[工具错误] 不存在名为 `{name}` 的工具。可用工具：{available}",
                ok=False,
            )

        if isinstance(arguments, str):
            try:
                params = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError as exc:
                return ToolResult(
                    content=(
                        f"[工具错误] 参数不是合法 JSON：{exc}。"
                        f"收到的内容：{arguments[:300]}"
                    ),
                    ok=False,
                )
        else:
            params = dict(arguments)

        started = time.perf_counter()
        try:
            result = tool.handler(**params)
        except TypeError as exc:
            # 参数名/必填项不匹配——模型最容易犯的错，给出可纠正的提示
            result = ToolResult(
                content=f"[工具错误] 参数不匹配：{exc}。请检查参数名与必填项。",
                ok=False,
            )
        except Exception as exc:  # noqa: BLE001 - 工具内部异常不应中断 Agent
            logger.exception("工具 %s 执行异常", name)
            result = ToolResult(
                content=f"[工具错误] {type(exc).__name__}: {exc}", ok=False
            )

        elapsed = time.perf_counter() - started
        self.call_log.append(
            {
                "tool": name,
                "arguments": params,
                "ok": result.ok,
                "elapsed": round(elapsed, 3),
            }
        )
        return result

    # ------------------------------------------------------------ 辅助 --
    def register_many(self, tools: Iterable[Tool]) -> None:
        for tool in tools:
            self.register(tool.name, tool.description, tool.parameters, tool.handler)


# --------------------------------------------------------------------------- #
# 路径安全：所有文件类工具都必须复用这一层校验
# --------------------------------------------------------------------------- #
class PathGuard:
    """把用户/模型给出的路径限制在工作区内，防止路径穿越。"""

    def __init__(self, workspace: Path | str) -> None:
        self.root = Path(workspace).resolve()
        if not self.root.is_dir():
            raise NotADirectoryError(f"工作区不存在或不是目录：{self.root}")

        # 这些目录与工作区同级时不应被扫描到
        self.skip_dirs = {
            ".git", ".hg", ".svn", "__pycache__", ".mypy_cache", ".pytest_cache",
            ".ruff_cache", "node_modules", ".venv", "venv", "env", ".idea",
            ".vscode", "dist", "build", ".tox", ".egg-info",
        }

    def resolve(self, raw: str | None, *, must_exist: bool = True) -> Path:
        """解析相对/绝对路径并校验其位于工作区内。"""
        if raw in (None, "", "."):
            candidate = self.root
        else:
            text = str(raw).strip().strip('"').strip("'")
            path = Path(text)
            candidate = path if path.is_absolute() else self.root / path

        # resolve() 会展开 .. 和符号链接，再用 is_relative_to 判断边界
        resolved = candidate.resolve()
        if resolved != self.root and not resolved.is_relative_to(self.root):
            raise PermissionError(
                f"拒绝访问工作区之外的路径：{raw}（工作区：{self.root}）"
            )
        if must_exist and not resolved.exists():
            raise FileNotFoundError(f"路径不存在：{self.rel(resolved)}")
        return resolved

    def rel(self, path: Path) -> str:
        """转成相对工作区的展示路径（统一用 / 分隔，便于阅读）。"""
        try:
            return path.resolve().relative_to(self.root).as_posix() or "."
        except ValueError:
            return str(path)

    def is_skipped(self, path: Path) -> bool:
        try:
            parts = path.resolve().relative_to(self.root).parts
        except ValueError:
            return True
        return any(part in self.skip_dirs for part in parts)

    def iter_files(
        self,
        root: Path,
        *,
        glob: str = "**/*",
        limit: int = 4000,
    ) -> list[Path]:
        """按 glob 遍历文件，自动跳过噪声目录。"""
        found: list[Path] = []
        for path in sorted(root.glob(glob)):
            if len(found) >= limit:
                break
            if path.is_file() and not self.is_skipped(path):
                found.append(path)
        return found
