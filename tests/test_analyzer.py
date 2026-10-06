"""静态分析器测试：验证各条规则确实能被检出。"""

from __future__ import annotations

from pathlib import Path

from code_review_agent.tools.analyzer import (
    analyze_file,
    detect_language,
    summarize_results,
)


def _rules(result) -> set[str]:
    return {f.rule for f in result.findings}


def test_detects_mutable_default_bare_except_and_dangerous_calls(workspace: Path) -> None:
    target = workspace / "app" / "service.py"
    result = analyze_file(target, display_path="app/service.py")

    assert result.language == "python"
    assert result.method == "ast"
    rules = _rules(result)
    assert "mutable-default" in rules, rules
    assert "bare-except" in rules, rules
    assert "dangerous-call" in rules, rules
    assert "hardcoded-secret" in rules, rules


def test_metrics_are_reported(workspace: Path) -> None:
    result = analyze_file(workspace / "app" / "service.py", display_path="app/service.py")

    assert result.metrics["functions"] >= 3
    assert result.metrics["max_complexity"] >= 7
    longest = result.metrics["longest_functions"]
    assert longest and longest[0]["name"] == "classify"


def test_syntax_error_is_reported_as_blocker(tmp_path: Path) -> None:
    broken = tmp_path / "broken.py"
    broken.write_text("def broken(:\n    pass\n", encoding="utf-8")

    result = analyze_file(broken, display_path="broken.py")

    assert result.findings[0].rule == "syntax-error"
    assert result.findings[0].severity == "blocker"


def test_javascript_uses_heuristic_rules(workspace: Path) -> None:
    result = analyze_file(workspace / "app" / "utils.js", display_path="app/utils.js")

    assert result.language == "javascript"
    assert result.method == "heuristic"
    rules = _rules(result)
    assert "var-declaration" in rules
    assert "empty-catch" in rules
    assert "loose-equality" in rules


def test_severity_sorting_puts_blockers_first(workspace: Path) -> None:
    from code_review_agent.tools.analyzer import sort_findings

    result = analyze_file(workspace / "app" / "service.py", display_path="app/service.py")
    ordered = sort_findings(result.findings)

    assert ordered[0].severity == "blocker"


def test_summarize_aggregates_counts(workspace: Path) -> None:
    results = [
        analyze_file(workspace / "app" / "service.py", display_path="app/service.py"),
        analyze_file(workspace / "app" / "utils.js", display_path="app/utils.js"),
    ]
    overview = summarize_results(results)

    assert overview["files_analyzed"] == 2
    assert overview["total_findings"] > 0
    assert set(overview["by_language"]) == {"python", "javascript"}


def test_detect_language_falls_back_to_text(tmp_path: Path) -> None:
    assert detect_language(tmp_path / "a.py") == "python"
    assert detect_language(tmp_path / "a.unknownext") == "text"
