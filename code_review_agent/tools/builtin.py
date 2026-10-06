"""内置工具实现（本项目演示的「工具集成」部分）。

共 7 个工具，覆盖 Agent 完成一次代码审查所需的完整能力链：

===================================== ==========================================
工具                                  作用
===================================== ==========================================
``list_directory``                    侦查项目结构与技术栈
``read_file``                         带行号读取源码（支持分段）
``search_code``                       正则/字面量搜索，跨文件定位线索
``analyze_code``                      调用静态分析器，输出结构化问题清单
``run_tests``                         实际执行 pytest，用真实结果做证据
``write_report``                      把审查报告落盘
``finish_review``                     显式结束并交付最终报告
===================================== ==========================================
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .analyzer import (
    AnalysisResult,
    analyze_file,
    detect_language,
    sort_findings,
    summarize_results,
)
from .base import PathGuard, ToolRegistry, ToolResult

logger = logging.getLogger(__name__)

# 会被当作「代码」的文件后缀
CODE_SUFFIXES = {
    ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".java", ".go", ".rb",
    ".php", ".cs", ".c", ".h", ".cpp", ".hpp", ".rs", ".kt", ".swift", ".scala",
}
CONFIG_SUFFIXES = {
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".env", ".md", ".txt",
    ".properties", ".xml",
}
MAX_OUTPUT_CHARS = 20_000
MAX_SEARCH_HITS = 120


@dataclass
class ToolContext:
    """工具运行所需的共享状态。"""

    workspace: Path
    guard: PathGuard
    max_file_bytes: int = 200_000
    test_timeout: int = 300
    output_dir: Path | None = None

    @classmethod
    def create(cls, workspace: str | Path, **kwargs: Any) -> "ToolContext":
        root = Path(workspace).resolve()
        return cls(workspace=root, guard=PathGuard(root), **kwargs)


# --------------------------------------------------------------------------- #
# 输出裁剪
# --------------------------------------------------------------------------- #
def _trim(text: str, limit: int = MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    head = text[: limit // 2]
    tail = text[-limit // 2 :]
    removed = len(text) - limit
    return f"{head}\n\n...[已省略 {removed} 字符]...\n\n{tail}"


def _format_finding_line(finding: Any, index: int) -> str:
    detail = f" → {finding.detail}" if finding.detail else ""
    return (
        f"{index}. [{finding.severity}] line {finding.line}: {finding.message}"
        f"（规则 {finding.rule}）{detail}"
    )


def _render_analysis(result: AnalysisResult, *, max_findings: int = 40) -> str:
    lines = [
        f"### {result.path}",
        f"- 语言：{result.language}｜行数：{result.lines}｜分析方式：{result.method}",
    ]
    metrics = result.metrics or {}
    if metrics:
        brief = ", ".join(f"{k}={v}" for k, v in metrics.items() if not isinstance(v, (list, dict)))
        if brief:
            lines.append(f"- 度量：{brief}")

    findings = sort_findings(result.findings)
    if not findings:
        lines.append("- 未发现明显问题")
    else:
        lines.append(f"- 发现 {len(findings)} 个问题：")
        for index, finding in enumerate(findings[:max_findings], start=1):
            lines.append("  " + _format_finding_line(finding, index))
        if len(findings) > max_findings:
            lines.append(f"  ...另有 {len(findings) - max_findings} 条已省略")

    longest = metrics.get("longest_functions")
    if isinstance(longest, list) and longest:
        top = ", ".join(f"{f['name']}({f['lines']}行/复杂度{f['complexity']})" for f in longest[:3])
        lines.append(f"- 最长函数：{top}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 工具实现
# --------------------------------------------------------------------------- #
def tool_list_directory(ctx: ToolContext, path: str = ".", max_depth: int = 2) -> ToolResult:
    root = ctx.guard.resolve(path)
    if not root.is_dir():
        return ToolResult(content=f"[工具错误] 不是目录：{path}", ok=False)

    lines: list[str] = [f"目录：{ctx.guard.rel(root)}"]
    root_depth = len(root.parts)
    file_count = 0
    dir_count = 0

    for current, dirs, files in _walk(root, max_depth):
        # 原地过滤，避免走进噪声目录
        dirs[:] = sorted(
            d for d in dirs if d not in ctx.guard.skip_dirs and not d.startswith(".")
        )
        depth = len(current.parts) - root_depth
        indent = "  " * depth
        if depth > 0:
            lines.append(f"{indent}{current.name}/")
            dir_count += 1
        for name in sorted(files):
            file_path = current / name
            size = file_path.stat().st_size
            lines.append(f"{indent}  {name}  ({_human_size(size)})")
            file_count += 1
            if file_count > 400:
                lines.append("...（文件过多，已省略）")
                break
        if file_count > 400:
            break

    lines.append(f"\n统计：{dir_count} 个子目录，{file_count} 个文件")
    return ToolResult(content=_trim("\n".join(lines)), meta={"file_count": file_count})


def _walk(root: Path, max_depth: int):
    """手写 BFS，保证 max_depth 生效且顺序稳定。"""
    stack: list[tuple[Path, int]] = [(root, 0)]
    while stack:
        current, depth = stack.pop(0)
        try:
            entries = list(current.iterdir())
        except (PermissionError, OSError):
            continue
        dirs = [e.name for e in entries if e.is_dir()]
        files = [e.name for e in entries if e.is_file()]
        yield current, dirs, files
        if depth + 1 < max_depth:
            for name in dirs:
                if name not in {".git", "__pycache__", "node_modules", ".venv", "venv"}:
                    stack.append((current / name, depth + 1))


def _human_size(size: int) -> str:
    for unit in ("B", "KB", "MB"):
        if size < 1024:
            return f"{size:.0f}{unit}"
        size /= 1024
    return f"{size:.1f}GB"


def tool_read_file(
    ctx: ToolContext,
    path: str,
    start_line: int = 1,
    end_line: int | None = None,
) -> ToolResult:
    target = ctx.guard.resolve(path)
    if target.is_dir():
        listing = ", ".join(p.name for p in sorted(target.iterdir())[:30])
        return ToolResult(
            content=f"[工具错误] `{path}` 是目录，不是文件。其内容：{listing}",
            ok=False,
        )

    size = target.stat().st_size
    if size > ctx.max_file_bytes:
        return ToolResult(
            content=(
                f"[工具错误] 文件过大（{_human_size(size)}，上限 "
                f"{_human_size(ctx.max_file_bytes)}）。请用 search_code 定位关键片段，"
                f"或指定 start_line/end_line 分段读取。"
            ),
            ok=False,
        )

    try:
        text = target.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return ToolResult(content=f"[工具错误] 读取失败：{exc}", ok=False)

    all_lines = text.splitlines()
    total = len(all_lines)
    start = max(1, int(start_line or 1))
    end = total if not end_line else min(int(end_line), total)
    if start > total:
        return ToolResult(
            content=f"[工具错误] start_line={start} 超出文件总行数 {total}。", ok=False
        )
    if end < start:
        return ToolResult(content=f"[工具错误] end_line={end} 小于 start_line={start}。", ok=False)

    shown = all_lines[start - 1 : end]
    width = len(str(end))
    body = "\n".join(f"{i:>{width}} | {line}" for i, line in enumerate(shown, start=start))
    header = (
        f"文件：{ctx.guard.rel(target)}｜语言：{detect_language(target)}"
        f"｜共 {total} 行｜本次显示 {start}-{end} 行"
    )
    return ToolResult(
        content=_trim(f"{header}\n{body}"),
        meta={"total_lines": total, "start": start, "end": end},
    )


def tool_search_code(
    ctx: ToolContext,
    pattern: str,
    path: str = ".",
    glob: str = "**/*",
    case_sensitive: bool = True,
    max_results: int = 60,
) -> ToolResult:
    root = ctx.guard.resolve(path)
    if not pattern:
        return ToolResult(content="[工具错误] pattern 不能为空。", ok=False)

    try:
        regex = re.compile(pattern if case_sensitive else pattern, 0 if case_sensitive else re.IGNORECASE)
    except re.error as exc:
        return ToolResult(
            content=f"[工具错误] 非法正则表达式：{exc}。如需搜索普通文本请转义特殊字符（如 `\\(`）。",
            ok=False,
        )

    max_results = max(1, min(int(max_results), MAX_SEARCH_HITS))
    targets = [root] if root.is_file() else ctx.guard.iter_files(root, glob=glob)

    hits: list[str] = []
    files_hit = 0
    scanned = 0
    for file_path in targets:
        if len(hits) >= max_results:
            break
        scanned += 1
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        file_hits = 0
        for line_no, line in enumerate(content.splitlines(), start=1):
            if regex.search(line):
                hits.append(f"{ctx.guard.rel(file_path)}:{line_no}: {line.strip()[:200]}")
                file_hits += 1
                if len(hits) >= max_results:
                    break
        if file_hits:
            files_hit += 1

    if not hits:
        return ToolResult(
            content=f"未找到匹配 `{pattern}` 的内容（扫描 {scanned} 个文件）。",
            meta={"scanned": scanned, "hits": 0},
        )

    header = f"匹配 `{pattern}`：{len(hits)} 处，涉及 {files_hit} 个文件，扫描 {scanned} 个文件"
    return ToolResult(
        content=_trim(header + "\n" + "\n".join(hits)),
        meta={"scanned": scanned, "hits": len(hits)},
    )


def tool_analyze_code(
    ctx: ToolContext,
    path: str = ".",
    max_files: int = 20,
) -> ToolResult:
    target = ctx.guard.resolve(path)

    if target.is_file():
        files = [target]
    else:
        files = [
            p
            for p in ctx.guard.iter_files(target)
            if p.suffix.lower() in CODE_SUFFIXES
        ][: max(1, min(int(max_files), 200))]

    if not files:
        return ToolResult(
            content=f"在 `{path}` 下没有找到可分析的代码文件（支持：{', '.join(sorted(CODE_SUFFIXES))}）。"
        )

    results: list[AnalysisResult] = []
    for file_path in files:
        results.append(analyze_file(file_path, display_path=ctx.guard.rel(file_path)))

    overview = summarize_results(results)
    header = (
        f"静态分析完成：分析 {overview['files_analyzed']} 个文件，"
        f"共 {overview['total_findings']} 条发现，"
        f"严重度分布 {overview['by_severity'] or '无'}；"
        f"按语言 {overview['by_language']}"
    )
    body = "\n\n".join(_render_analysis(r) for r in results)
    return ToolResult(
        content=_trim(header + "\n\n" + body),
        meta={"files": len(results), "findings": overview["total_findings"]},
    )


def tool_run_tests(
    ctx: ToolContext,
    path: str = ".",
    args: str = "-q",
    timeout: int = 300,
) -> ToolResult:
    target = ctx.guard.resolve(path)
    if not target.exists():
        return ToolResult(content=f"[工具错误] 路径不存在：{path}", ok=False)

    # 优先使用当前解释器的 pytest 模块，避免 PATH 里找不到 pytest.exe
    base_cmd: list[str] | None = None
    if shutil.which("pytest"):
        base_cmd = ["pytest"]
    else:
        try:
            import pytest  # noqa: F401

            base_cmd = [sys.executable, "-m", "pytest"]
        except ImportError:
            base_cmd = None

    if base_cmd is None:
        return ToolResult(
            content=(
                "[工具错误] 环境中未安装 pytest，无法执行测试。"
                "请先运行 `pip install pytest`，或改用 analyze_code 做静态分析。"
            ),
            ok=False,
        )

    extra = [a for a in (args or "").split() if a]
    cmd = base_cmd + extra
    workdir = target if target.is_dir() else target.parent
    if target.is_file():
        cmd.append(target.name)

    try:
        completed = subprocess.run(  # noqa: S603 - 命令来自本工具内部构造
            cmd,
            cwd=str(workdir),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=max(10, min(int(timeout), ctx.test_timeout)),
            shell=False,
        )
    except subprocess.TimeoutExpired:
        return ToolResult(
            content=f"[工具错误] 测试超时（超过 {timeout}s），可能存在死循环或阻塞 IO。",
            ok=False,
        )
    except FileNotFoundError as exc:
        return ToolResult(content=f"[工具错误] 无法启动测试进程：{exc}", ok=False)
    except OSError as exc:
        return ToolResult(content=f"[工具错误] 执行测试失败：{exc}", ok=False)

    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    tail = "\n".join(stdout.splitlines()[-80:])
    status = "通过" if completed.returncode == 0 else f"失败（退出码 {completed.returncode}）"
    body = [
        f"命令：`{' '.join(cmd)}`（工作目录：{ctx.guard.rel(workdir)}）",
        f"结果：{status}",
        "--- 输出（末尾 80 行）---",
        tail or "(无输出)",
    ]
    if stderr.strip():
        body += ["--- stderr ---", "\n".join(stderr.splitlines()[-20:])]
    return ToolResult(
        content=_trim("\n".join(body)),
        ok=completed.returncode == 0,
        meta={"returncode": completed.returncode},
    )


def tool_write_report(
    ctx: ToolContext,
    content: str,
    path: str = "review_report.md",
) -> ToolResult:
    if not content or not content.strip():
        return ToolResult(content="[工具错误] 报告内容为空，拒绝写入。", ok=False)

    try:
        target = ctx.guard.resolve(path, must_exist=False)
    except PermissionError as exc:
        return ToolResult(content=f"[工具错误] {exc}", ok=False)

    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        target.write_text(content, encoding="utf-8")
    except OSError as exc:
        return ToolResult(content=f"[工具错误] 写入失败：{exc}", ok=False)

    return ToolResult(
        content=(
            f"报告已写入 `{ctx.guard.rel(target)}`（{len(content)} 字符，"
            f"{len(content.splitlines())} 行）。"
        ),
        meta={"path": str(target)},
    )


def tool_finish_review(ctx: ToolContext, summary: str = "", report: str = "") -> ToolResult:
    """显式结束审查。Agent 用它来标记任务完成。"""
    body = report.strip() or summary.strip()
    if not body:
        return ToolResult(content="[工具错误] 需要提供 report 或 summary。", ok=False)
    return ToolResult(
        content=body,
        meta={"finished": True},
    )


# --------------------------------------------------------------------------- #
# 注册
# --------------------------------------------------------------------------- #
def _schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


def build_default_registry(ctx: ToolContext) -> ToolRegistry:
    """构造带全部内置工具的注册表。"""
    registry = ToolRegistry()

    registry.register(
        "list_directory",
        "列出工作区（或其子目录）的目录结构与文件大小，用于快速了解项目布局。",
        _schema(
            {
                "path": {"type": "string", "description": "相对工作区的目录路径，默认 '.'"},
                "max_depth": {"type": "integer", "description": "最大递归深度，默认 2"},
            }
        ),
        lambda path=".", max_depth=2: tool_list_directory(ctx, path, max_depth),
    )

    registry.register(
        "read_file",
        "按行号读取文本文件内容（输出带行号）。用于查看源码细节；大文件请指定行范围。",
        _schema(
            {
                "path": {"type": "string", "description": "相对工作区的文件路径"},
                "start_line": {"type": "integer", "description": "起始行（从 1 开始），默认 1"},
                "end_line": {"type": "integer", "description": "结束行（含），默认到文件末尾"},
            },
            ["path"],
        ),
        lambda path, start_line=1, end_line=None: tool_read_file(ctx, path, start_line, end_line),
    )

    registry.register(
        "search_code",
        "用正则表达式跨文件搜索代码，返回 文件:行号: 内容。用于定位函数定义、可疑调用等。",
        _schema(
            {
                "pattern": {"type": "string", "description": "正则表达式，例如 `def .*password`"},
                "path": {"type": "string", "description": "搜索范围，默认整个工作区"},
                "glob": {"type": "string", "description": "文件名过滤，如 `**/*.py`，默认 `**/*`"},
                "case_sensitive": {"type": "boolean", "description": "是否区分大小写，默认 true"},
                "max_results": {"type": "integer", "description": "最多返回多少条，默认 60"},
            },
            ["pattern"],
        ),
        lambda pattern, path=".", glob="**/*", case_sensitive=True, max_results=60: tool_search_code(
            ctx, pattern, path, glob, case_sensitive, max_results
        ),
    )

    registry.register(
        "analyze_code",
        "对文件或目录做静态分析：圈复杂度、超长函数、裸 except、可变默认参数、"
        "硬编码密钥、危险调用等，返回结构化问题清单。Python 使用 AST，其他语言为启发式扫描。",
        _schema(
            {
                "path": {"type": "string", "description": "文件或目录路径，默认 '.'"},
                "max_files": {"type": "integer", "description": "目录模式下最多分析多少文件，默认 20"},
            }
        ),
        lambda path=".", max_files=20: tool_analyze_code(ctx, path, max_files),
    )

    registry.register(
        "run_tests",
        "在指定目录执行 pytest 并返回结果。用于用真实测试结果验证审查结论。",
        _schema(
            {
                "path": {"type": "string", "description": "测试目录或测试文件，默认 '.'"},
                "args": {"type": "string", "description": "额外 pytest 参数，默认 '-q'"},
                "timeout": {"type": "integer", "description": "超时秒数，默认 300"},
            }
        ),
        lambda path=".", args="-q", timeout=300: tool_run_tests(ctx, path, args, timeout),
    )

    registry.register(
        "write_report",
        "把 Markdown 审查报告写入工作区文件。仅在用户要求保存时调用。",
        _schema(
            {
                "content": {"type": "string", "description": "完整报告内容（Markdown）"},
                "path": {"type": "string", "description": "输出路径，默认 review_report.md"},
            },
            ["content"],
        ),
        lambda content, path="review_report.md": tool_write_report(ctx, content, path),
    )

    registry.register(
        "finish_review",
        "结束审查并交付最终报告。当你已完成全部取证、准备给出结论时调用它。",
        _schema(
            {
                "report": {"type": "string", "description": "完整的 Markdown 审查报告"},
                "summary": {"type": "string", "description": "一句话结论（当 report 为空时使用）"},
            }
        ),
        lambda report="", summary="": tool_finish_review(ctx, summary, report),
    )

    return registry
