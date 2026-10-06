"""静态代码分析器：给出结构化的「代码坏味道 + 风险点」清单。

分两条路线：
- **Python**：用标准库 ``ast`` 做真正的语法树分析（圈复杂度、函数长度、
  裸 except、可变默认参数、危险调用等），结论准确；
- **其他语言**：用正则做启发式扫描（JS/TS/Java 的常见问题），
  结果里会明确标注来源为启发式，避免让模型误以为是精确结论。

刻意不引入 flake8/pylint 等外部依赖：本项目要演示的是 Agent 架构，
分析器保持零依赖可运行，评审者 clone 下来就能跑。
"""

from __future__ import annotations

import ast
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #
@dataclass
class Finding:
    rule: str
    severity: str  # blocker / major / minor / info
    message: str
    line: int
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnalysisResult:
    path: str
    language: str
    lines: int
    findings: list[Finding] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    method: str = "ast"  # ast | heuristic

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "language": self.language,
            "lines": self.lines,
            "method": self.method,
            "metrics": self.metrics,
            "findings": [f.to_dict() for f in self.findings],
        }


SEVERITY_ORDER = {"blocker": 0, "major": 1, "minor": 2, "info": 3}

LANGUAGE_BY_SUFFIX = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".go": "go",
    ".rb": "ruby",
    ".php": "php",
    ".cs": "csharp",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".rs": "rust",
}


def detect_language(path: Path) -> str:
    return LANGUAGE_BY_SUFFIX.get(path.suffix.lower(), "text")


# --------------------------------------------------------------------------- #
# Python：AST 分析
# --------------------------------------------------------------------------- #
class _PythonVisitor(ast.NodeVisitor):
    COMPLEXITY_NODES = (
        ast.If, ast.For, ast.AsyncFor, ast.While, ast.ExceptHandler,
        ast.With, ast.AsyncWith, ast.BoolOp, ast.IfExp, ast.Assert,
        ast.comprehension,
    )
    DANGEROUS_CALLS = {
        "eval": ("blocker", "使用 eval() 执行动态代码，存在代码注入风险"),
        "exec": ("blocker", "使用 exec() 执行动态代码，存在代码注入风险"),
        "os.system": ("major", "os.system 以 shell 方式执行命令，存在命令注入风险"),
        "pickle.load": ("blocker", "pickle 反序列化不可信数据可导致任意代码执行"),
        "pickle.loads": ("blocker", "pickle 反序列化不可信数据可导致任意代码执行"),
        "yaml.load": ("major", "yaml.load 未指定 SafeLoader 时可执行任意构造器"),
        "input": ("info", "input() 在 Python 2 语义下会执行代码；确认运行时版本"),
    }

    def __init__(self, source: str, path: str) -> None:
        self.source = source
        self.path = path
        self.findings: list[Finding] = []
        self.functions: list[dict[str, Any]] = []
        self._max_line = len(source.splitlines())
        self._string_constants: list[tuple[str, int]] = []
        self._assign_targets: list[tuple[str, int]] = []

    # -- 辅助 --
    def _add(self, rule: str, severity: str, message: str, line: int, detail: str = "") -> None:
        self.findings.append(Finding(rule, severity, message, line, detail))

    @staticmethod
    def _complexity(node: ast.AST) -> int:
        return 1 + sum(1 for child in ast.walk(node) if isinstance(child, _PythonVisitor.COMPLEXITY_NODES))

    @staticmethod
    def _call_name(func: ast.AST) -> str:
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            parts: list[str] = []
            cursor: ast.AST | None = func
            while isinstance(cursor, ast.Attribute):
                parts.append(cursor.attr)
                cursor = cursor.value
            if isinstance(cursor, ast.Name):
                parts.append(cursor.id)
            return ".".join(reversed(parts))
        return ""

    # -- 遍历 --
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._check_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._check_function(node)

    def _check_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        length = (node.end_lineno or node.lineno) - node.lineno + 1
        complexity = self._complexity(node)
        self.functions.append(
            {"name": node.name, "line": node.lineno, "lines": length, "complexity": complexity}
        )

        if length > 80:
            self._add(
                "long-function",
                "major" if length > 150 else "minor",
                f"函数 `{node.name}` 长达 {length} 行，职责可能过多",
                node.lineno,
                "建议按单一职责拆分为多个小函数。",
            )
        if complexity > 12:
            self._add(
                "high-complexity",
                "major" if complexity > 20 else "minor",
                f"函数 `{node.name}` 圈复杂度 {complexity}，难以测试与维护",
                node.lineno,
                "可通过提前 return、抽取条件函数、用多态替代分支来降低。",
            )

        # 可变默认参数：经典的 Python 陷阱
        for default in list(node.args.defaults) + list(node.args.kw_defaults):
            if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                self._add(
                    "mutable-default",
                    "major",
                    f"函数 `{node.name}` 使用可变对象作为默认参数，会在多次调用间共享状态",
                    node.lineno,
                    "改为默认 None，在函数体内初始化。",
                )
            elif isinstance(default, ast.Call):
                name = self._call_name(default.func)
                if name in {"list", "dict", "set"}:
                    self._add(
                        "mutable-default",
                        "major",
                        f"函数 `{node.name}` 的默认参数由 `{name}()` 构造，等价于可变默认值",
                        node.lineno,
                        "改为默认 None，在函数体内初始化。",
                    )

        # 裸 except / 静默吞异常
        for child in ast.walk(node):
            if isinstance(child, ast.ExceptHandler):
                if child.type is None:
                    self._add(
                        "bare-except",
                        "major",
                        "使用了裸 `except:`，会连 KeyboardInterrupt/SystemExit 一起吞掉",
                        child.lineno,
                        "改为捕获具体异常类型，至少用 `except Exception`。",
                    )
                elif not child.body or all(
                    isinstance(stmt, ast.Pass) for stmt in child.body
                ):
                    self._add(
                        "silent-except",
                        "major",
                        "异常被静默忽略（`except ...: pass`），故障会被隐藏",
                        child.lineno,
                        "至少记录日志，或显式说明为何可以忽略。",
                    )

        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = self._call_name(node.func)
        if name in self.DANGEROUS_CALLS:
            severity, message = self.DANGEROUS_CALLS[name]
            self._add("dangerous-call", severity, f"{message}（`{name}`）", node.lineno)

        # subprocess(..., shell=True)
        if name.startswith("subprocess."):
            for keyword in node.keywords:
                if keyword.arg == "shell" and isinstance(keyword.value, ast.Constant) and keyword.value.value is True:
                    self._add(
                        "shell-injection",
                        "major",
                        "subprocess 使用 shell=True，拼接外部输入时可导致命令注入",
                        node.lineno,
                        "改用列表形式的参数并保持 shell=False。",
                    )
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        # 避免与 _check_function 中的检查重复，这里只处理顶层（不在函数内的）情况
        self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> None:
        if isinstance(node.value, str) and len(node.value) >= 8:
            self._string_constants.append((node.value, node.lineno))
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if isinstance(target, ast.Name):
                self._assign_targets.append((target.id, node.lineno))
            elif isinstance(target, ast.Attribute):
                self._assign_targets.append((target.attr, node.lineno))
        self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert) -> None:
        # assert 用于运行时校验会在 -O 下被移除
        self._add(
            "assert-as-check",
            "info",
            "使用 assert 做运行时校验：在 `python -O` 下会被移除",
            node.lineno,
            "对外部输入的校验请显式抛异常。",
        )
        self.generic_visit(node)


SECRET_NAME_PATTERN = re.compile(
    r"(api[_-]?key|secret|passwd|password|token|access[_-]?key|private[_-]?key)",
    re.IGNORECASE,
)
SECRET_VALUE_PATTERN = re.compile(
    r"^(sk-[A-Za-z0-9_\-]{12,}|AKIA[0-9A-Z]{12,}|ghp_[A-Za-z0-9]{20,})$"
)
TODO_PATTERN = re.compile(r"#\s*(TODO|FIXME|XXX|HACK)\b", re.IGNORECASE)


def analyze_python(path: Path, source: str, display_path: str) -> AnalysisResult:
    result = AnalysisResult(
        path=display_path,
        language="python",
        lines=len(source.splitlines()),
        method="ast",
    )

    try:
        tree = ast.parse(source, filename=display_path)
    except SyntaxError as exc:
        result.findings.append(
            Finding(
                "syntax-error",
                "blocker",
                f"文件存在语法错误，无法解析：{exc.msg}",
                exc.lineno or 1,
            )
        )
        return result

    visitor = _PythonVisitor(source, display_path)
    visitor.visit(tree)
    result.findings.extend(visitor.findings)

    # 硬编码密钥：赋值目标是敏感名 + 字面量看起来像密钥
    for name, line in visitor._assign_targets:  # noqa: SLF001 - 同模块内协作
        if SECRET_NAME_PATTERN.search(name):
            for value, vline in visitor._string_constants:  # noqa: SLF001
                if vline == line and len(value) >= 8 and " " not in value:
                    severity = "blocker" if SECRET_VALUE_PATTERN.match(value) else "major"
                    result.findings.append(
                        Finding(
                            "hardcoded-secret",
                            severity,
                            f"疑似硬编码凭据：变量 `{name}` 直接赋了字符串常量",
                            line,
                            "改为从环境变量或密钥管理服务读取，并尽快轮换该凭据。",
                        )
                    )
                    break

    # 长行 / TODO
    for index, text in enumerate(source.splitlines(), start=1):
        if len(text) > 120 and not text.lstrip().startswith(("#", "//", "*")):
            result.findings.append(
                Finding("long-line", "info", f"该行长度 {len(text)} 超过 120 字符", index)
            )
        match = TODO_PATTERN.search(text)
        if match:
            result.findings.append(
                Finding("todo", "info", f"遗留标记 {match.group(1).upper()} 未处理", index)
            )

    functions = visitor.functions
    result.metrics = {
        "functions": len(functions),
        "classes": sum(isinstance(n, ast.ClassDef) for n in ast.walk(tree)),
        "imports": sum(
            isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(tree)
        ),
        "max_function_lines": max((f["lines"] for f in functions), default=0),
        "max_complexity": max((f["complexity"] for f in functions), default=0),
        "avg_complexity": round(
            sum(f["complexity"] for f in functions) / len(functions), 2
        ) if functions else 0,
        "longest_functions": sorted(functions, key=lambda f: f["lines"], reverse=True)[:5],
        "docstring_coverage": _docstring_coverage(tree, source),
    }
    return result


def _docstring_coverage(tree: ast.Module, source: str) -> str:
    """统计有 docstring 的函数占比，作为可维护性参考。"""
    defs = [
        n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if not defs:
        return "n/a"
    documented = sum(1 for n in defs if ast.get_docstring(n))
    return f"{documented}/{len(defs)}"


# --------------------------------------------------------------------------- #
# 其他语言：正则启发式
# --------------------------------------------------------------------------- #
HEURISTIC_RULES: dict[str, list[tuple[str, str, str, str]]] = {
    "javascript": [
        (r"\bvar\s+\w+", "var-declaration", "minor", "使用 var 声明变量，作用域易出错，建议 let/const"),
        (r"\bconsole\.(log|debug)\(", "console-leftover", "info", "残留的 console 调试输出"),
        (r"==[^=]", "loose-equality", "minor", "使用 == 做松散比较，建议 === "),
        (r"catch\s*\([^)]*\)\s*\{\s*\}", "empty-catch", "major", "空的 catch 块会吞掉异常"),
        (r"innerHTML\s*=", "xss-risk", "major", "直接赋值 innerHTML 存在 XSS 风险，建议 textContent"),
        (r"eval\(", "dangerous-call", "blocker", "使用 eval 执行动态代码"),
    ],
    "typescript": [
        (r":\s*any\b", "any-type", "minor", "使用 any 会绕过类型检查"),
        (r"catch\s*\([^)]*\)\s*\{\s*\}", "empty-catch", "major", "空的 catch 块会吞掉异常"),
        (r"innerHTML\s*=", "xss-risk", "major", "直接赋值 innerHTML 存在 XSS 风险"),
        (r"@ts-ignore", "ts-ignore", "minor", "使用 @ts-ignore 抑制类型错误，建议修复根因"),
    ],
    "java": [
        (r"catch\s*\([^)]*\)\s*\{\s*\}", "empty-catch", "major", "空的 catch 块会吞掉异常"),
        (r"System\.out\.println", "stdout-logging", "info", "使用 System.out 输出，建议接入日志框架"),
        (r"printStackTrace\(", "printstacktrace", "minor", "printStackTrace 无法被日志系统采集"),
        (r"==\s*\"", "string-equality", "major", "字符串用 == 比较的是引用，应使用 equals"),
    ],
    "go": [
        (r"_\s*=\s*\w+", "ignored-error", "major", "错误被显式忽略，需确认是否安全"),
        (r"fmt\.Print", "stdout-logging", "info", "使用 fmt.Print 输出，建议接入日志"),
    ],
}

UNIVERSAL_RULES: list[tuple[str, str, str, str]] = [
    (r"\bTODO\b|\bFIXME\b", "todo", "info", "遗留 TODO/FIXME 未处理"),
    (r"(?i)(api[_-]?key|secret|password|token)\s*[:=]\s*[\"'][^\"']{8,}[\"']",
     "hardcoded-secret", "blocker", "疑似硬编码凭据"),
]


def analyze_heuristic(path: Path, source: str, display_path: str, language: str) -> AnalysisResult:
    result = AnalysisResult(
        path=display_path,
        language=language,
        lines=len(source.splitlines()),
        method="heuristic",
    )
    rules = HEURISTIC_RULES.get(language, []) + UNIVERSAL_RULES

    for index, text in enumerate(source.splitlines(), start=1):
        stripped = text.strip()
        if not stripped or stripped.startswith(("//", "*", "/*", "#")):
            continue
        for pattern, rule, severity, message in rules:
            if re.search(pattern, text):
                result.findings.append(Finding(rule, severity, message, index))
        if len(text) > 140:
            result.findings.append(
                Finding("long-line", "info", f"该行长度 {len(text)} 超过 140 字符", index)
            )

    result.metrics = {"heuristic_rules": len(rules)}
    return result


# --------------------------------------------------------------------------- #
# 对外入口
# --------------------------------------------------------------------------- #
def analyze_file(path: Path, *, display_path: str | None = None) -> AnalysisResult:
    """分析单个文件，自动选择 AST 或启发式策略。"""
    name = display_path or path.as_posix()
    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return AnalysisResult(
            path=name,
            language="unknown",
            lines=0,
            findings=[Finding("read-error", "blocker", f"无法读取文件：{exc}", 1)],
        )

    language = detect_language(path)
    if language == "python":
        return analyze_python(path, source, name)
    if language == "text":
        return AnalysisResult(path=name, language="text", lines=len(source.splitlines()))
    return analyze_heuristic(path, source, name, language)


def summarize_results(results: Iterable[AnalysisResult]) -> dict[str, Any]:
    """把多个文件的分析结果汇总成概览，便于塞进 Prompt。"""
    results = list(results)
    counts: dict[str, int] = {}
    total = 0
    for result in results:
        for finding in result.findings:
            counts[finding.severity] = counts.get(finding.severity, 0) + 1
            total += 1
    return {
        "files_analyzed": len(results),
        "total_findings": total,
        "by_severity": counts,
        "by_language": _count_by(results, "language"),
    }


def _count_by(results: Iterable[AnalysisResult], attr: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        key = getattr(result, attr)
        counts[key] = counts.get(key, 0) + 1
    return counts


def sort_findings(findings: Iterable[Finding]) -> list[Finding]:
    return sorted(findings, key=lambda f: (SEVERITY_ORDER.get(f.severity, 9), f.line))
