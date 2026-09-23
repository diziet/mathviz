"""This repo's doc checks: the refs scan reaches known references, each fact computes.

Engine tests (test_check_doc_refs.py, test_doc_facts.py, test_doc_common.py) are
copied unchanged between repos. This file names references and facts that exist
only here, so a broken scanner or registry cannot pass by checking nothing.
"""

from __future__ import annotations

import re
from pathlib import Path

import check_doc_refs as refs
import pytest
from doc_facts_registry import FACTS
from doc_refs_cli import ProgramIndex

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_refs_scan_checks_a_named_minimum_set_of_references() -> None:
    checker = refs.Checker(REPO_ROOT)
    for doc in ("README.md", "CLAUDE.md"):
        checker.check_doc(doc)
    assert {
        "make gate",
        "make merge",
        "make worktree",
        "docs/generators.md",
        "tasks.md",
        "mathviz generate",
        "mathviz render-all",
    } <= set(checker.checked)


def test_console_script_defines_its_documented_subcommands_and_flags() -> None:
    program = ProgramIndex(REPO_ROOT).console("mathviz")
    assert program is not None
    assert {"generate", "render-all", "render-thumbnail", "export-demo"} <= (
        program.subcommands
    )
    assert {"--profile", "--style", "--view", "--view-mode", "--help"} <= program.flags


@pytest.mark.parametrize(
    ("name", "shape"),
    [
        ("render-all-style-default", r"[a-z_]+"),
        ("render-styles", r"[a-z_]+(, [a-z_]+)+"),
        ("thumbnail-view-mode-default", r"[a-z_]+"),
        ("thumbnail-view-modes", r"[a-z_]+(, [a-z_]+)+"),
        ("preview-view-mode-count", r"[1-9]\d*"),
        ("preview-view-modes", r"[A-Z][\w ]+(, [A-Z][\w ]+)+"),
    ],
)
def test_registered_fact_computes_a_well_formed_value(name: str, shape: str) -> None:
    assert re.fullmatch(shape, str(FACTS[name].compute(REPO_ROOT)))


def test_every_registered_fact_has_a_shape_test() -> None:
    assert set(FACTS) == {
        "render-all-style-default",
        "render-styles",
        "thumbnail-view-mode-default",
        "thumbnail-view-modes",
        "preview-view-mode-count",
        "preview-view-modes",
    }


def test_defaults_are_members_of_their_allowed_values() -> None:
    styles = str(FACTS["render-styles"].compute(REPO_ROOT)).split(", ")
    modes = str(FACTS["thumbnail-view-modes"].compute(REPO_ROOT)).split(", ")
    assert FACTS["render-all-style-default"].compute(REPO_ROOT) in styles
    assert FACTS["thumbnail-view-mode-default"].compute(REPO_ROOT) in modes


def test_view_mode_count_matches_the_label_list() -> None:
    labels = str(FACTS["preview-view-modes"].compute(REPO_ROOT)).split(", ")
    assert FACTS["preview-view-mode-count"].compute(REPO_ROOT) == len(labels)
