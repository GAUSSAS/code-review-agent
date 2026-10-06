"""工具层测试：注册表导出、各工具行为、路径安全与错误处理。"""

from __future__ import annotations

from pathlib import Path

import pytest

from code_review_agent.tools import PathGuard, ToolContext, build_default_registry


@pytest.fixture
def registry(workspace: Path):
    return build_default_registry(ToolContext.create(workspace))


# --------------------------------------------------------------------------- #
# 注册表
# --------------------------------------------------------------------------- #
def test_registry_exposes_openai_compatible_schemas(registry) -> None:
    specs = registry.specs()

    assert len(specs) == 7
    for spec in specs:
        assert spec["type"] == "function"
        fn = spec["function"]
        assert fn["name"] and fn["description"]
        assert fn["parameters"]["type"] == "object"
        assert "properties" in fn["parameters"]


def test_unknown_tool_returns_error_result_not_exception(registry) -> None:
    result = registry.execute("not_a_tool", "{}")

    assert result.ok is False
    assert "不存在名为" in result.content


def test_invalid_json_arguments_are_handled(registry) -> None:
    result = registry.execute("read_file", "{不是合法 json")

    assert result.ok is False
    assert "不是合法 JSON" in result.content


def test_missing_required_argument_is_reported(registry) -> None:
    result = registry.execute("read_file", "{}")

    assert result.ok is False
    assert "参数不匹配" in result.content


# --------------------------------------------------------------------------- #
# 各工具
# --------------------------------------------------------------------------- #
def test_read_file_returns_numbered_lines(registry) -> None:
    result = registry.execute("read_file", '{"path": "app/service.py", "start_line": 1, "end_line": 3}')

    assert result.ok
    assert "1 |" in result.content
    assert "3 |" in result.content
    assert result.meta["total_lines"] > 3


def test_read_file_rejects_directory(registry) -> None:
    result = registry.execute("read_file", '{"path": "app"}')

    assert result.ok is False
    assert "是目录" in result.content


def test_read_file_start_beyond_eof(registry) -> None:
    result = registry.execute("read_file", '{"path": "app/service.py", "start_line": 99999}')

    assert result.ok is False
    assert "超出文件总行数" in result.content


def test_list_directory_shows_tree_and_stats(registry) -> None:
    result = registry.execute("list_directory", '{"path": ".", "max_depth": 3}')

    assert result.ok
    assert "app/" in result.content
    assert "service.py" in result.content
    assert "统计：" in result.content


def test_search_code_finds_matches_with_line_numbers(registry) -> None:
    result = registry.execute("search_code", '{"pattern": "def ", "glob": "**/*.py"}')

    assert result.ok
    assert "app/service.py:" in result.content
    assert "def fetch" in result.content


def test_search_code_reports_no_hits(registry) -> None:
    result = registry.execute("search_code", '{"pattern": "zzz_never_matches"}')

    assert result.ok
    assert "未找到匹配" in result.content


def test_search_code_rejects_bad_regex(registry) -> None:
    result = registry.execute("search_code", '{"pattern": "([unclosed"}')

    assert result.ok is False
    assert "非法正则" in result.content


def test_analyze_code_on_whole_workspace(registry) -> None:
    result = registry.execute("analyze_code", '{"path": ".", "max_files": 10}')

    assert result.ok
    assert "静态分析完成" in result.content
    assert result.meta["findings"] > 0


def test_write_report_creates_file(workspace: Path, registry) -> None:
    result = registry.execute(
        "write_report", '{"content": "# 报告\\n\\n内容", "path": "out/report.md"}'
    )

    assert result.ok
    assert (workspace / "out" / "report.md").read_text(encoding="utf-8").startswith("# 报告")


def test_write_report_rejects_empty_content(registry) -> None:
    result = registry.execute("write_report", '{"content": "   "}')

    assert result.ok is False
    assert "为空" in result.content


def test_run_tests_reports_pytest_result(workspace: Path, registry) -> None:
    tests_dir = workspace / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_ok.py").write_text("def test_ok():\n    assert 1 + 1 == 2\n", encoding="utf-8")

    result = registry.execute("run_tests", '{"path": "tests", "args": "-q"}')

    # 若环境未安装 pytest，工具会返回可读的失败信息而不是抛异常
    if result.ok:
        assert "通过" in result.content
    else:
        assert "pytest" in result.content


def test_finish_review_requires_payload(registry) -> None:
    assert registry.execute("finish_review", "{}").ok is False

    ok_result = registry.execute("finish_review", '{"summary": "没有问题"}')
    assert ok_result.ok
    assert ok_result.meta["finished"] is True


# --------------------------------------------------------------------------- #
# 路径安全
# --------------------------------------------------------------------------- #
def test_path_guard_blocks_escape(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    outside = tmp_path / "secret.txt"
    outside.write_text("top secret", encoding="utf-8")

    guard = PathGuard(workspace)

    with pytest.raises(PermissionError):
        guard.resolve("../secret.txt")
    with pytest.raises(PermissionError):
        guard.resolve(str(outside))


def test_read_file_cannot_escape_workspace(workspace: Path, registry) -> None:
    result = registry.execute("read_file", '{"path": "../../etc/passwd"}')

    assert result.ok is False
    assert "工作区之外" in result.content


def test_path_guard_skips_noise_directories(workspace: Path) -> None:
    noisy = workspace / "node_modules" / "pkg"
    noisy.mkdir(parents=True)
    (noisy / "index.js").write_text("module.exports = 1;\n", encoding="utf-8")

    guard = PathGuard(workspace)
    files = guard.iter_files(workspace, glob="**/*.js")

    assert all("node_modules" not in p.as_posix() for p in files)
