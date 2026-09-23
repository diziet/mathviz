"""scripts/doc_facts.py and its readers: a stale fact fails, regeneration fixes it."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

import doc_facts
import pytest
from doc_facts_sources import (
    Fact,
    FactError,
    count_distinct,
    makefile_recipe,
    pyproject_value,
    regex_group,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

REGISTRY = {
    "answer": Fact(
        "the ANSWER file", lambda root: (root / "ANSWER").read_text().strip()
    ),
    "double": Fact(
        "twice the ANSWER", lambda root: 2 * int((root / "ANSWER").read_text())
    ),
}
DOC = (
    "The answer is <!-- fact:answer -->42<!-- /fact -->.\n"
    "Twice that is <!-- fact:double -->84<!-- /fact -->.\n"
)
REGISTRY_FILE = """from doc_facts_sources import Fact, read_text

FACTS = {"answer": Fact("ANSWER", lambda root: read_text(root, "ANSWER").strip())}
"""


@pytest.fixture
def repo(tmp_path: Path, isolated_git_env: None) -> Path:
    """A git repo with one marked doc whose values match ANSWER."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / "ANSWER").write_text("42\n")
    (root / "README.md").write_text(DOC)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    return root


def _run(
    root: Path, mode: str, capsys: pytest.CaptureFixture[str]
) -> tuple[int, str, str]:
    code = doc_facts.run(root, REGISTRY, mode)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_current_values_pass_the_read_only_check(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, _ = _run(repo, "check", capsys)
    assert code == 0
    assert "doc_facts: ok (2 markers in 1 docs)" in out


def test_stale_value_fails_with_line_and_diff_and_leaves_the_file(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (repo / "ANSWER").write_text("43\n")
    code, _, err = _run(repo, "check", capsys)
    assert code == 1
    assert "README.md:1: fact 'answer' states '42'; the tree says '43'" in err
    assert "README.md:2: fact 'double' states '84'; the tree says '86'" in err
    assert "-The answer is <!-- fact:answer -->42<!-- /fact -->." in err
    assert "+The answer is <!-- fact:answer -->43<!-- /fact -->." in err
    assert (repo / "README.md").read_text() == DOC


def test_write_rewrites_stale_values_and_names_each_file(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (repo / "ANSWER").write_text("7\n")
    code, out, err = _run(repo, "write", capsys)
    assert code == 0, err
    assert "doc_facts: rewrote README.md" in out
    assert "<!-- fact:answer -->7<!-- /fact -->" in (repo / "README.md").read_text()
    assert "<!-- fact:double -->14<!-- /fact -->" in (repo / "README.md").read_text()


def test_write_rewrites_nothing_when_values_are_current(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, _ = _run(repo, "write", capsys)
    assert code == 0
    assert "rewrote" not in out


def test_fix_stale_rewrites_then_fails(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (repo / "ANSWER").write_text("5\n")
    code, out, err = _run(repo, "fix-stale", capsys)
    assert code == 1
    assert "doc_facts: rewrote README.md" in out
    assert "rewrote them; review and stage" in err
    assert "<!-- fact:answer -->5<!-- /fact -->" in (repo / "README.md").read_text()


def test_markers_in_code_spans_and_fences_are_examples(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    example = (
        "Write `<!-- fact:nope -->1<!-- /fact -->`.\n\n"
        "```\n<!-- fact:nope -->2<!-- /fact -->\n```\n"
    )
    (repo / "README.md").write_text(DOC + example)
    (repo / "ANSWER").write_text("1\n")
    code, _, err = _run(repo, "write", capsys)
    assert code == 0, err
    assert (repo / "README.md").read_text().endswith(example)


def test_marker_at_the_start_of_a_line_fails(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (repo / "README.md").write_text(
        DOC + "- <!-- fact:answer -->42<!-- /fact --> items\n"
    )
    code, _, err = _run(repo, "write", capsys)
    assert code == 1
    assert "README.md:3: fact marker starts a line" in err


def test_unknown_fact_fails_by_file_and_line_without_writing(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (repo / "ANSWER").write_text("9\n")
    (repo / "README.md").write_text(
        DOC + "Also <!-- fact:missing -->0<!-- /fact -->.\n"
    )
    code, _, err = _run(repo, "write", capsys)
    assert code == 1
    assert "README.md:3: unknown fact 'missing'; registered: answer, double" in err
    assert "<!-- fact:answer -->42<!-- /fact -->" in (repo / "README.md").read_text()


def test_registered_fact_without_a_marker_fails(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (repo / "README.md").write_text(
        "The answer is <!-- fact:answer -->42<!-- /fact -->.\n"
    )
    code, _, err = _run(repo, "check", capsys)
    assert code == 1
    assert "registered fact 'double' has no marker in any tracked .md file" in err


def test_missing_source_fails_with_the_fact_name(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    registry = {
        "gone": Fact("the GONE file", lambda root: regex_group(root, "GONE", "(x)"))
    }
    (repo / "README.md").write_text("Gone: <!-- fact:gone -->x<!-- /fact -->.\n")
    code = doc_facts.run(repo, registry, "check")
    err = capsys.readouterr().err
    assert code == 1
    assert "fact 'gone' (the GONE file): GONE does not exist" in err


def test_multi_line_value_is_rejected(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    registry = {"answer": Fact("two lines", lambda root: "a\nb")}
    (repo / "README.md").write_text("Value <!-- fact:answer -->a<!-- /fact -->.\n")
    assert doc_facts.run(repo, registry, "write") == 1
    assert "multi-line or marker-breaking value" in capsys.readouterr().err


def test_untracked_and_excluded_docs_are_ignored(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    stale = "Old <!-- fact:unknown -->1<!-- /fact -->.\n"
    (repo / "tests" / "fixtures").mkdir(parents=True)
    (repo / "tests" / "fixtures" / "doc.md").write_text(stale)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    (repo / "untracked.md").write_text(stale)
    code, _, err = _run(repo, "check", capsys)
    assert code == 0, err


def test_main_loads_the_registry_under_the_root(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (repo / "scripts").mkdir()
    (repo / "scripts" / "doc_facts_registry.py").write_text(REGISTRY_FILE)
    (repo / "README.md").write_text("A <!-- fact:answer -->0<!-- /fact -->.\n")
    assert doc_facts.main(["--root", str(repo), "--write"]) == 0
    assert "<!-- fact:answer -->42<!-- /fact -->" in (repo / "README.md").read_text()
    capsys.readouterr()


def test_main_without_a_registry_fails(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert doc_facts.main(["--root", str(repo)]) == 1
    assert "scripts/doc_facts_registry.py does not exist" in capsys.readouterr().err


def test_main_outside_a_git_checkout_is_a_usage_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert doc_facts.main(["--root", str(tmp_path)]) == 2
    assert "is not a git checkout" in capsys.readouterr().err


def test_readers_return_values_from_their_sources(tmp_path: Path) -> None:
    (tmp_path / "NOTES").write_text("# v1.2\n## PR 1\n## PR 2\n## PR 2\n")
    (tmp_path / "Makefile").write_text("run:\n\tpython scripts/run.py\n")
    (tmp_path / "pyproject.toml").write_text("[tool.x]\nlimit = 7\n")
    assert regex_group(tmp_path, "NOTES", r"^# v(\S+)$") == "1.2"
    assert count_distinct(tmp_path, "NOTES", r"^## PR (\d+)$") == 2
    assert makefile_recipe(tmp_path, "run") == ["python scripts/run.py"]
    assert pyproject_value(tmp_path, "tool.x.limit") == 7


@pytest.mark.parametrize(
    ("reader", "message"),
    [
        (
            lambda root: regex_group(root, "NOTES", r"^(nothing)$"),
            "has no line matching",
        ),
        (
            lambda root: count_distinct(root, "NOTES", r"^(nothing)$"),
            "has no line matching",
        ),
        (lambda root: makefile_recipe(root, "deploy"), "has no `deploy` target"),
        (lambda root: pyproject_value(root, "tool.y"), "has no `tool.y`"),
    ],
)
def test_readers_raise_fact_error_when_the_source_changed_shape(
    tmp_path: Path, reader: Callable[[Path], object], message: str
) -> None:
    (tmp_path / "NOTES").write_text("text\n")
    (tmp_path / "Makefile").write_text("run:\n\ttrue\n")
    (tmp_path / "pyproject.toml").write_text("[tool.x]\n")
    with pytest.raises(FactError, match=message):
        reader(tmp_path)
