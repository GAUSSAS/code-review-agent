"""CLI 冒烟测试：把各子命令跑一遍，确认参数解析、渲染与退出码都正常。

这些用例不访问网络：
- review/chat 通过猴子补丁替换掉真实 LLM 客户端；
- tools/selfcheck 是纯本地逻辑。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from code_review_agent import cli
from code_review_agent.agent import ReviewAgent
from code_review_agent.config import Settings
from code_review_agent.llm import LLMClient
from tests.helpers import FakeOpenAI, text_response, tool_response


@pytest.fixture
def patch_llm(monkeypatch):
    """把 ReviewAgent 内部构造的 LLMClient 换成注入假 SDK 的版本。

    注意：替换后的 ``__init__`` **仍会调用真实实现**（保留参数校验逻辑），
    而真实实现会拒绝空 api_key。因此这里在缺失时补一个占位 Key——
    这是测试关心「CLI 流程」而非「Key 校验」时必须做的隔离。
    """

    def _apply(script):
        fake = FakeOpenAI(script)
        original = LLMClient.__init__

        def patched(self, **kwargs):  # noqa: ANN001
            kwargs["client"] = fake
            if not kwargs.get("api_key"):
                kwargs["api_key"] = "test-placeholder-key"
            original(self, **kwargs)

        monkeypatch.setattr(LLMClient, "__init__", patched)
        return fake

    return _apply


def test_tools_subcommand_lists_seven_tools(workspace: Path, capsys) -> None:
    code = cli.main(["tools", "-w", str(workspace), "--no-color"])

    out = capsys.readouterr().out
    assert code == cli.EXIT_OK
    assert "已注册 7 个工具" in out
    assert "analyze_code" in out
    assert "run_tests" in out


def test_review_subcommand_end_to_end(workspace: Path, capsys, patch_llm) -> None:
    patch_llm(
        [
            tool_response([("analyze_code", {"path": "app"})]),
            text_response("## 一、项目概览\n\n示例项目"),
        ]
    )

    code = cli.main(["review", "-w", str(workspace), "--no-color", "--quiet"])

    out = capsys.readouterr().out
    assert code == cli.EXIT_OK
    assert "审查报告" in out
    assert "示例项目" in out
    assert "统计：" in out


def test_review_save_writes_report(workspace: Path, capsys, patch_llm) -> None:
    patch_llm([text_response("```markdown\n# 报告正文\n```")])

    code = cli.main(
        [
            "review",
            "-w",
            str(workspace),
            "--no-color",
            "--quiet",
            "--save",
            "-o",
            "out/audit.md",
        ]
    )

    out = capsys.readouterr().out
    saved = workspace / "out" / "audit.md"
    assert code == cli.EXIT_OK
    assert saved.is_file()
    # 模型返回的 ```markdown 围栏应被剥掉
    assert saved.read_text(encoding="utf-8").startswith("# 报告正文")
    assert "报告已保存" in out


def test_review_json_outputs_stats(workspace: Path, capsys, patch_llm) -> None:
    patch_llm([text_response("完成")])

    code = cli.main(["review", "-w", str(workspace), "--no-color", "--quiet", "--json"])

    out = capsys.readouterr().out
    assert code == cli.EXIT_OK
    assert '"tools_used"' in out


def test_chat_single_message(workspace: Path, capsys, patch_llm) -> None:
    patch_llm([text_response("这个项目有 3 个风险点。")])

    code = cli.main(["chat", "有哪些风险？", "-w", str(workspace), "--no-color"])

    out = capsys.readouterr().out
    assert code == cli.EXIT_OK
    assert "3 个风险点" in out


def test_missing_api_key_exits_with_config_code(workspace: Path, capsys, monkeypatch) -> None:
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    empty_env = workspace / "empty.env"
    empty_env.write_text("", encoding="utf-8")

    code = cli.main(
        ["review", "-w", str(workspace), "--no-color", "--quiet", "--env-file", str(empty_env)]
    )

    err = capsys.readouterr().err
    assert code == cli.EXIT_CONFIG
    assert "配置错误" in err


def test_verbose_flag_accepted_after_subcommand(workspace: Path, capsys, patch_llm) -> None:
    """回归：通用选项写在子命令之后也要生效。"""
    patch_llm([text_response("完成")])

    code = cli.main(["review", "-w", str(workspace), "--no-color", "--quiet", "-v"])

    assert code == cli.EXIT_OK


def test_interactive_loop_exits_on_eof(workspace: Path, capsys) -> None:
    """交互模式输入 EOF（Ctrl+D / Ctrl+Z）应干净退出，返回 0。"""
    settings = Settings(workspace=str(workspace), api_key="fake-key", model="fake-model")
    # 注入不会发起任何真实请求的客户端，保证测试完全离线
    client = LLMClient(model="fake-model", api_key="fake-key", client=FakeOpenAI([]))
    agent = ReviewAgent(settings, client=client)

    def raise_eof(_prompt: str = "") -> str:
        raise EOFError

    import builtins

    original = builtins.input
    builtins.input = raise_eof
    try:
        code = cli._interactive_loop(agent, cli.Pretty(enabled=False), verbose=False)  # noqa: SLF001
    finally:
        builtins.input = original

    assert code == cli.EXIT_OK
