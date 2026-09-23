"""scripts/doc_common.py: code spans, code regions, doc listing and Makefile rules."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

import pytest
from doc_common import code_regions, code_spans, list_docs, makefile_rules

if TYPE_CHECKING:
    from pathlib import Path


def _texts(markdown: str) -> list[str]:
    return [span.text for span in code_spans(markdown)]


def test_code_spans_skip_fenced_blocks() -> None:
    markdown = "`a`\n\n```bash\nmake `b`\n```\n\n~~~\n`c`\n~~~\n`d`\n"
    assert _texts(markdown) == ["a", "d"]


def test_code_spans_skip_an_indented_fence_inside_a_list() -> None:
    markdown = "- step:\n\n    ```\n    `hidden`\n    ```\n- `shown`\n"
    assert _texts(markdown) == ["shown"]


def test_code_spans_report_the_line_of_the_opening_backtick() -> None:
    markdown = "title\n\ntext `one` and\n`two`\n"
    assert [(s.line, s.text) for s in code_spans(markdown)] == [(3, "one"), (4, "two")]


def test_double_backtick_span_keeps_an_inner_backtick_and_strips_padding() -> None:
    assert _texts("``  a ` b  ``") == [" a ` b "]
    assert _texts("`` `x` ``") == ["`x`"]


def test_backtick_run_without_a_closer_of_equal_width_is_literal() -> None:
    assert _texts("a `` b `c`") == ["c"]
    assert _texts("a ` b\n\n`c`") == ["c"]


def test_escaped_backtick_does_not_open_a_span() -> None:
    assert _texts(r"\`x and `y`") == ["y"]


def test_code_span_inside_an_html_comment_is_skipped() -> None:
    markdown = "<!-- `hidden`\nstill hidden -->`shown`"
    assert _texts(markdown) == ["shown"]


def test_comment_opener_inside_a_code_span_does_not_hide_the_rest() -> None:
    assert _texts("`<!--` then `after`") == ["<!--", "after"]


def test_struck_span_is_marked_even_across_a_line_break() -> None:
    markdown = "~~`old` and\n`older`~~ now `new`\n"
    assert [(s.text, s.is_struck) for s in code_spans(markdown)] == [
        ("old", True),
        ("older", True),
        ("new", False),
    ]


def test_strikethrough_does_not_cross_a_blank_line() -> None:
    markdown = "~~ unclosed `a`\n\n`b` ~~x~~\n"
    assert [(s.text, s.is_struck) for s in code_spans(markdown)] == [
        ("a", False),
        ("b", False),
    ]


def test_code_regions_cover_fences_and_inline_spans() -> None:
    markdown = "x `a` y\n```\nz\n```\n"
    regions = code_regions(markdown)
    covered = "".join(markdown[start:end] for start, end in regions)
    assert covered == "`a`" + "```\nz\n```\n"


@pytest.mark.usefixtures("isolated_git_env")
def test_list_docs_skips_untracked_excluded_and_symlinked_docs(tmp_path: Path) -> None:
    for rel in ("README.md", "docs/a.md", "tests/fixtures/f.md", "pkg/data/d.md"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("# x\n")
    (tmp_path / "AGENTS.md").symlink_to("README.md")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    (tmp_path / "untracked.md").write_text("# y\n")
    assert list_docs(tmp_path) == ["README.md", "docs/a.md"]


def test_makefile_rules_read_targets_recipes_and_includes(tmp_path: Path) -> None:
    (tmp_path / "Makefile").write_text(
        ".PHONY: a b\nVAR := x\nFLAG ?= y\ninclude extra.mk\n"
        "a b: dep ## two targets\n\techo one \\\n\t  continued\n"
        "%.o: %.c\n\tcc\n$(VAR)-x:\n\ttrue\n"
        "define block\nhidden: rule\nendef\n"
    )
    (tmp_path / "extra.mk").write_text("c:\n\techo c\n")
    rules = makefile_rules(tmp_path)
    assert sorted(rules) == ["a", "b", "c"]
    assert rules["a"] == ["echo one  \t  continued"]
    assert rules["c"] == ["echo c"]


def test_makefile_recipe_continues_through_conditionals(tmp_path: Path) -> None:
    (tmp_path / "Makefile").write_text(
        "check:\nifeq ($(X),)\n\trun default\nelse\n\trun other\nendif\n"
        "X = 1\n\tnot a recipe\n"
    )
    assert makefile_rules(tmp_path)["check"] == ["run default", "run other"]


def test_makefile_rules_are_empty_without_a_makefile(tmp_path: Path) -> None:
    assert makefile_rules(tmp_path) == {}
