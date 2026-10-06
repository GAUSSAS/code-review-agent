"""命令行入口。

用法示例::

    python -m code_review_agent review                 # 审查当前目录并进入交互式追问
    python -m code_review_agent review src/app.py      # 审查单个文件
    python -m code_review_agent review --save          # 审查并保存 review_report.md
    python -m code_review_agent chat                   # 多轮问答模式
    python -m code_review_agent web                    # 启动 Web 界面
    python -m code_review_agent tools                  # 查看已注册工具
    python -m code_review_agent selfcheck --probe      # 环境自检 + API 连通性测试
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from .agent import AgentError, AgentEvent, ReviewAgent, StepLimitExceeded
from .config import Settings
from .llm import ChatMessage, LLMClient, LLMConfigError, LLMError

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_CONFIG = 3
EXIT_RUNTIME = 4
EXIT_LIMIT = 5

# --------------------------------------------------------------------------- #
# 终端着色（无 colorama 依赖，靠 ANSI 转义；不支持时自动降级）
# --------------------------------------------------------------------------- #
class Pretty:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled and self._supports_color()

    @staticmethod
    def _supports_color() -> bool:
        if os.environ.get("NO_COLOR"):
            return False
        if os.environ.get("TERM") == "dumb":
            return False
        if sys.platform == "win32":
            # Windows 10+ 的现代终端（Windows Terminal / VS Code）支持 ANSI
            return bool(os.environ.get("WT_SESSION") or os.environ.get("TERM_PROGRAM")) or (
                os.environ.get("ANSICON") is not None
            )
        return sys.stdout.isatty()

    def _wrap(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.enabled else text

    def bold(self, text: str) -> str:
        return self._wrap("1", text)

    def dim(self, text: str) -> str:
        return self._wrap("2", text)

    def cyan(self, text: str) -> str:
        return self._wrap("36", text)

    def green(self, text: str) -> str:
        return self._wrap("32", text)

    def yellow(self, text: str) -> str:
        return self._wrap("33", text)

    def red(self, text: str) -> str:
        return self._wrap("31", text)


# --------------------------------------------------------------------------- #
# 事件渲染
# --------------------------------------------------------------------------- #
def render_event(event: AgentEvent, *, pretty: Pretty, verbose: bool) -> None:
    if event.type == "start":
        print(pretty.dim(f"▸ {event.text}"))
        return

    if event.type == "assistant":
        body = event.text.strip()
        if body:
            preview = body if verbose or len(body) <= 600 else body[:600] + " …"
            print(pretty.cyan("\n🤔 " + preview))
        return

    if event.type == "tool_call":
        args = json.dumps(event.arguments, ensure_ascii=False)
        if len(args) > 200:
            args = args[:200] + " …"
        print(pretty.yellow(f"\n🔧 调用工具 {event.tool} {args}"))
        return

    if event.type == "tool_result":
        mark = pretty.green("✓") if event.ok else pretty.red("✗")
        lines = event.text.splitlines()
        limit = 400 if verbose else 120
        preview = "\n".join("   " + line for line in lines[:6])
        if len(lines) > 6 or (lines and len(lines[0]) > limit):
            preview = preview[: limit * 3] + f"\n   …（完整输出 {len(event.text)} 字符）"
        print(f"   {mark} {event.tool} 返回：\n{preview}")
        return

    if event.type == "limit":
        print(pretty.yellow(f"\n⚠ {event.text}"))
        return

    if event.type == "error":
        print(pretty.red(f"\n✗ {event.text}"), file=sys.stderr)
        return

    if event.type == "final":
        print(pretty.bold("\n" + "=" * 72))
        print(pretty.bold("审查报告"))
        print(pretty.bold("=" * 72))
        print(event.text.strip())
        print(pretty.bold("=" * 72))
        return


def print_stats(agent: ReviewAgent, *, pretty: Pretty) -> None:
    stats = agent.stats()
    usage = stats["usage"]
    tool_usage = "、".join(f"{k}×{v}" for k, v in stats["tools_used"].items()) or "无"
    print(
        pretty.dim(
            f"\n统计：步数 {stats['steps']}｜工具调用 {stats['tool_calls']} 次（{tool_usage}）"
            f"｜tokens 输入 {usage['prompt_tokens']} / 输出 {usage['completion_tokens']}"
        )
    )


def strip_code_fence(text: str) -> str:
    """去掉模型可能包裹的 ```markdown 围栏。"""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return stripped


# --------------------------------------------------------------------------- #
# 子命令实现
# --------------------------------------------------------------------------- #
def cmd_tools(args: argparse.Namespace, pretty: Pretty) -> int:
    from .tools import ToolContext, build_default_registry

    ctx = ToolContext.create(getattr(args, "workspace", "."))
    registry = build_default_registry(ctx)
    print(pretty.bold(f"已注册 {len(registry)} 个工具（工作区：{ctx.workspace}）\n"))
    for name in registry.names():
        tool = registry.get(name)
        assert tool is not None
        print(pretty.cyan(f"● {name}"))
        print(f"  {tool.description}")
        required = tool.parameters.get("required") or []
        props = tool.parameters.get("properties") or {}
        for prop_name, spec in props.items():
            flag = "必填" if prop_name in required else "可选"
            print(pretty.dim(f"    - {prop_name} ({spec.get('type', 'any')}, {flag}): {spec.get('description', '')}"))
        print()
    return EXIT_OK


def cmd_selfcheck(args: argparse.Namespace, pretty: Pretty) -> int:
    problems: list[str] = []
    # 子命令可能没有定义这些全局参数，统一用 getattr 兜底
    env_file = getattr(args, "env_file", ".env")
    workspace = getattr(args, "workspace", ".")
    verbose = getattr(args, "verbose", False)
    probe = getattr(args, "probe", False)

    print(pretty.bold("1) Python 运行时"))
    version = sys.version_info
    print(f"   {sys.executable}")
    print(f"   Python {version.major}.{version.minor}.{version.micro}")
    if version < (3, 10):
        problems.append("Python 版本低于 3.10（代码使用了 `X | Y` 类型语法）")

    print(pretty.bold("\n2) 依赖"))
    for module in ("openai", "pytest"):
        try:
            mod = __import__(module)
            print(pretty.green(f"   ✓ {module} {getattr(mod, '__version__', '')}"))
        except ImportError:
            print(pretty.yellow(f"   ✗ {module} 未安装"))
            if module == "openai":
                problems.append("缺少 openai SDK：pip install -r requirements.txt")
            else:
                problems.append("缺少 pytest（run_tests 工具不可用）：pip install pytest")

    print(pretty.bold("\n3) 配置"))
    settings = Settings.from_env(
        env_file,
        workspace=workspace,
        overrides={"verbose": verbose},
    )
    print(f"   provider  : {settings.provider}")
    print(f"   model     : {settings.model}")
    print(f"   base_url  : {settings.base_url}")
    print(f"   api_key   : {'已设置（' + settings.api_key[:6] + '...）' if settings.api_key else '未设置'}")
    print(f"   workspace : {settings.workspace}")
    if not settings.is_configured:
        problems.append("未配置 LLM_API_KEY：请复制 .env.example 为 .env 并填写")

    print(pretty.bold("\n4) 工具"))
    from .tools import ToolContext, build_default_registry

    try:
        ctx = ToolContext.create(settings.workspace)
        registry = build_default_registry(ctx)
        print(pretty.green(f"   ✓ {len(registry)} 个工具注册成功：{', '.join(registry.names())}"))
    except Exception as exc:  # noqa: BLE001
        print(pretty.red(f"   ✗ 工具注册失败：{exc}"))
        problems.append(f"工具注册失败：{exc}")

    if probe and settings.is_configured:
        print(pretty.bold("\n5) API 连通性测试"))
        try:
            client = LLMClient(
                model=settings.model,
                api_key=settings.api_key,
                base_url=settings.base_url,
                max_tokens=32,
                max_retries=1,
                timeout=30,
            )
            reply = client.chat(
                [ChatMessage(role="user", content="请只回复两个字：连通")]
            )
            print(pretty.green(f"   ✓ 调用成功，模型回复：{reply.content.strip()[:50]}"))
            usage = client.usage
            print(pretty.dim(f"   tokens: {usage.total_tokens}"))
        except LLMError as exc:
            print(pretty.red(f"   ✗ 调用失败：{exc}"))
            problems.append(f"LLM 调用失败：{exc}")
        except Exception as exc:  # noqa: BLE001
            print(pretty.red(f"   ✗ 调用异常：{type(exc).__name__}: {exc}"))
            problems.append(f"LLM 调用异常：{exc}")

    print()
    if problems:
        print(pretty.yellow("自检发现问题："))
        for index, item in enumerate(problems, start=1):
            print(pretty.yellow(f"   {index}. {item}"))
        return EXIT_CONFIG
    print(pretty.green("自检通过，一切就绪。"))
    return EXIT_OK


def _build_agent(args: argparse.Namespace, pretty: Pretty) -> ReviewAgent:
    settings = Settings.from_env(
        getattr(args, "env_file", ".env"),
        workspace=getattr(args, "workspace", "."),
        overrides={
            "verbose": getattr(args, "verbose", False),
            "max_steps": getattr(args, "max_steps", None),
            "model": getattr(args, "model", None),
            "temperature": getattr(args, "temperature", None),
        },
    )
    if getattr(args, "verbose", False):
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")

    if not settings.is_configured:
        raise LLMConfigError(
            "未配置 API Key。请复制 .env.example 为 .env 并填写 LLM_API_KEY，"
            "或先运行 `python -m code_review_agent selfcheck` 查看详情。"
        )
    return ReviewAgent(settings, with_few_shot=not getattr(args, "no_few_shot", False))


def _run_review_stream(agent: ReviewAgent, pretty: Pretty, verbose: bool, **kwargs: Any) -> str:
    """消费事件流，返回最终报告文本。处理步数耗尽的兜底逻辑。"""
    report = ""
    try:
        for event in agent.review(**kwargs):
            render_event(event, pretty=pretty, verbose=verbose)
            if event.type == "final":
                report = event.text
    except StepLimitExceeded:
        print(pretty.yellow("\n改用兜底策略：让模型基于已获取信息直接给出结论…"))
        report = agent.force_conclusion()
        print(pretty.bold("\n" + "=" * 72))
        print(report.strip())
        print(pretty.bold("=" * 72))
    return report


def cmd_web(args: argparse.Namespace, pretty: Pretty) -> int:
    """启动 Web 界面（按需导入，避免 CLI 其他命令依赖 http.server）。"""
    from .web.server import serve

    env_file = getattr(args, "env_file", ".env")
    workspace = getattr(args, "workspace", ".")
    settings = Settings.from_env(
        env_file,
        workspace=workspace,
        overrides={"verbose": getattr(args, "verbose", False)},
    )

    # port=0 交给 socketserver 分配真实端口，serve() 会打印实际地址
    port = getattr(args, "port", 8765)
    host = getattr(args, "host", "127.0.0.1")
    allow_roots = tuple(getattr(args, "allow_root", None) or ())

    # 真正的 URL 由 serve() 打印；这里只提示配置来源
    print(pretty.dim(f"配置文件：{env_file}｜服务目录：{Path(workspace).resolve()}"))
    if allow_roots:
        print(pretty.dim("额外允许审查：" + "、".join(allow_roots)))
    if not settings.is_configured:
        print(
            pretty.yellow(
                "提示：服务端未检测到 LLM_API_KEY。"
                "你可以在打开的网页里直接填写 API Key（测试连接 → 开始审查），"
                "或创建 .env 后重启服务。"
            )
        )

    return serve(
        settings,
        host=host,
        port=port,
        root=Path(workspace).resolve(),
        open_browser=not getattr(args, "no_browser", False),
        allow_roots=allow_roots,
    )


def cmd_review(args: argparse.Namespace, pretty: Pretty) -> int:
    agent = _build_agent(args, pretty)
    report = _run_review_stream(
        agent,
        pretty,
        args.verbose,
        path=args.path,
        focus=args.focus,
        language_hint=args.language,
        report_path=args.output,
        save_report=args.save,
    )

    if args.save:
        target = Path(args.output)
        if not target.is_absolute():
            target = Path(agent.workspace) / target
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(strip_code_fence(report) + "\n", encoding="utf-8")
        print(pretty.green(f"\n报告已保存：{target}"))

    if args.json:
        print(json.dumps(agent.stats(), ensure_ascii=False, indent=2))

    print_stats(agent, pretty=pretty)

    if args.interactive and not args.quiet:
        return _interactive_loop(agent, pretty, args.verbose)
    return EXIT_OK


def cmd_chat(args: argparse.Namespace, pretty: Pretty) -> int:
    agent = _build_agent(args, pretty)
    if args.message:
        try:
            for event in agent.chat_turn(args.message):
                render_event(event, pretty=pretty, verbose=args.verbose)
        except (AgentError, LLMError) as exc:
            print(pretty.red(f"运行失败：{exc}"), file=sys.stderr)
            return EXIT_RUNTIME
        print_stats(agent, pretty=pretty)
        return EXIT_OK
    return _interactive_loop(agent, pretty, args.verbose)


def _interactive_loop(agent: ReviewAgent, pretty: Pretty, verbose: bool) -> int:
    print(pretty.bold("\n进入交互模式：可继续追问（例如「第 2 个问题给出修复后的完整代码」）。"))
    print(pretty.dim("输入 exit / quit 退出，输入 :report 重新打印上一份报告。\n"))
    last_report = ""
    while True:
        try:
            line = input(pretty.green("你> ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return EXIT_OK
        if not line:
            continue
        if line.lower() in {"exit", "quit", ":q"}:
            return EXIT_OK
        if line == ":report":
            if last_report:
                print(last_report)
            continue

        try:
            for event in agent.chat_turn(line):
                render_event(event, pretty=pretty, verbose=verbose)
                if event.type == "final":
                    last_report = event.text
        except StepLimitExceeded:
            last_report = agent.force_conclusion()
            print(last_report)
        except (AgentError, LLMError) as exc:
            print(pretty.red(f"运行失败：{exc}"), file=sys.stderr)
        except KeyboardInterrupt:
            print(pretty.yellow("\n已中断本次回答。"))


# --------------------------------------------------------------------------- #
# 参数解析
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    # 用 parent parser 承载通用选项，每个子命令自带一份，避免「全局选项必须写在
    # 子命令之前」这种反直觉的用法（`review -v` 也能生效）。
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--workspace", "-w", default=".", help="被审查的代码目录（默认当前目录）")
    common.add_argument("--no-color", action="store_true", help="禁用彩色输出")
    common.add_argument("--verbose", "-v", action="store_true", help="显示详细日志与完整工具输出")

    # 需要读取 .env 的子命令
    with_env = argparse.ArgumentParser(add_help=False, parents=[common])
    with_env.add_argument("--env-file", default=".env", help="配置文件路径（默认 .env）")

    parser = argparse.ArgumentParser(
        prog="code_review_agent",
        description="代码审查 Agent —— 基于 LLM + 工具调用的自动化代码评审助手",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  python -m code_review_agent selfcheck --probe\n"
            "  python -m code_review_agent review --save\n"
            "  python -m code_review_agent review src/app.py --focus '并发安全'\n"
        ),
    )

    sub = parser.add_subparsers(dest="command", required=True)

    def add_llm_options(p: argparse.ArgumentParser) -> None:
        p.add_argument("--model", default=None, help="覆盖 LLM_MODEL")
        p.add_argument("--temperature", type=float, default=None, help="覆盖采样温度")
        p.add_argument("--max-steps", type=int, default=None, help="最大 Agent 步数")
        p.add_argument("--no-few-shot", action="store_true", help="系统提示词中不附带 Few-shot 示例")

    review = sub.add_parser("review", help="执行一次代码审查", parents=[with_env])
    review.add_argument("path", nargs="?", default=None, help="要审查的文件（省略则审查整个工作区）")
    review.add_argument("--focus", default=None, help="重点关注方向，如「并发安全」「性能」")
    review.add_argument("--language", default=None, help="技术栈提示，如「Python 3.12 / FastAPI」")
    review.add_argument("--save", action="store_true", help="把报告保存到文件")
    review.add_argument("--output", "-o", default="review_report.md", help="报告保存路径")
    review.add_argument("--json", action="store_true", help="额外输出统计 JSON")
    review.add_argument("--interactive", "-i", action="store_true", help="审查后进入交互式追问")
    review.add_argument("--quiet", action="store_true", help="不进入交互式追问")
    add_llm_options(review)

    chat = sub.add_parser("chat", help="多轮问答模式（记忆跨轮共享）", parents=[with_env])
    chat.add_argument("message", nargs="?", default=None, help="单次提问；省略则进入交互模式")
    add_llm_options(chat)

    sub.add_parser("tools", help="列出已注册的工具及其参数", parents=[common])

    web = sub.add_parser("web", help="启动 Web 界面（浏览器中交互）", parents=[with_env])
    web.add_argument("--host", default="127.0.0.1", help="监听地址，默认 127.0.0.1（仅本机）")
    web.add_argument("--port", type=int, default=8765, help="监听端口，0 表示自动选择空闲端口")
    web.add_argument(
        "--allow-root",
        action="append",
        metavar="DIR",
        help="额外允许被审查的目录，可重复指定。例如 --allow-root D:\\projA --allow-root D:\\projB",
    )
    web.add_argument(
        "--no-browser",
        action="store_true",
        help="不自动打开浏览器（默认会自动打开）",
    )

    check = sub.add_parser("selfcheck", help="环境与配置自检", parents=[with_env])
    check.add_argument("--probe", action="store_true", help="额外发起一次真实 LLM 调用")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    pretty = Pretty(enabled=not getattr(args, "no_color", False))

    try:
        if args.command == "tools":
            return cmd_tools(args, pretty)
        if args.command == "selfcheck":
            return cmd_selfcheck(args, pretty)
        if args.command == "web":
            return cmd_web(args, pretty)
        if args.command == "review":
            return cmd_review(args, pretty)
        if args.command == "chat":
            return cmd_chat(args, pretty)
        parser.print_help()
        return EXIT_USAGE
    except LLMConfigError as exc:
        print(pretty.red(f"配置错误：{exc}"), file=sys.stderr)
        return EXIT_CONFIG
    except StepLimitExceeded as exc:
        print(pretty.yellow(f"{exc}"), file=sys.stderr)
        return EXIT_LIMIT
    except AgentError as exc:
        print(pretty.red(f"运行失败：{exc}"), file=sys.stderr)
        return EXIT_RUNTIME
    except KeyboardInterrupt:
        print(pretty.yellow("\n已取消。"), file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
