"""scripts/branches_gc.py: classification and the delete-only-merged rule."""

from __future__ import annotations

import os
import stat
from typing import TYPE_CHECKING

import branches_gc

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

    from tests.tooling.conftest import GitFixture


def _branch(name: str, **overrides: object) -> branches_gc.Branch:
    fields: dict[str, object] = {
        "is_merged": False,
        "upstream_gone": False,
        "worktree": None,
        "has_open_pr": False,
    }
    fields.update(overrides)
    return branches_gc.Branch(name=name, **fields)  # type: ignore[arg-type]


def test_classify_puts_each_branch_in_exactly_one_class() -> None:
    branches = [
        _branch("main", is_merged=True),
        _branch("feat/wt", is_merged=True, worktree="/tmp/wt"),
        _branch("feat/pr", has_open_pr=True),
        _branch("feat/done", is_merged=True),
        _branch("feat/gone", upstream_gone=True),
        _branch("feat/wip"),
    ]
    groups = branches_gc.classify(branches)
    names = {klass: [b.name for b in items] for klass, items in groups.items()}
    assert names == {
        "checked-out": ["feat/wt"],
        "open-pr": ["feat/pr"],
        "merged": ["feat/done"],
        "superseded": ["feat/gone"],
        "active": ["feat/wip"],
    }


def test_checked_out_wins_over_merged_and_open_pr_wins_over_merged() -> None:
    groups = branches_gc.classify(
        [
            _branch("a", is_merged=True, worktree="/x"),
            _branch("b", is_merged=True, has_open_pr=True),
        ]
    )
    assert groups["merged"] == []
    assert [b.name for b in groups["checked-out"]] == ["a"]
    assert [b.name for b in groups["open-pr"]] == ["b"]


def _fake_gh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, open_branches: list[str]
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    gh = bin_dir / "gh"
    listing = ",".join(f'{{"headRefName":"{b}"}}' for b in open_branches)
    gh.write_text(f"#!/bin/sh\necho '[{listing}]'\n")
    gh.chmod(gh.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")


def test_gather_reads_git_state(
    git_repo: GitFixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    git_repo.git("branch", "feat/merged", "origin/main")
    git_repo.git("switch", "-q", "-c", "feat/open")
    git_repo.commit_file("o.txt", "o\n", "open")
    git_repo.git("switch", "-q", "-c", "feat/active")
    git_repo.commit_file("a.txt", "a\n", "active")
    git_repo.git("push", "-q", "-u", "origin", "feat/active")
    git_repo.git("push", "-q", "origin", "--delete", "feat/active")
    git_repo.git("fetch", "-q", "--prune", "origin")
    git_repo.git("switch", "-q", "main")
    _fake_gh(tmp_path, monkeypatch, ["feat/open"])
    groups = branches_gc.classify(branches_gc.gather(git_repo.clone))
    names = {klass: sorted(b.name for b in items) for klass, items in groups.items()}
    assert names["merged"] == ["feat/merged"]
    assert names["open-pr"] == ["feat/open"]
    assert names["superseded"] == ["feat/active"]
    assert names["checked-out"] == []


def test_gather_without_gh_keeps_unmerged_branches_active(
    git_repo: GitFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    empty_bin = tmp_path / "empty-bin"
    empty_bin.mkdir()
    monkeypatch.setenv("PATH", f"{empty_bin}:/usr/bin:/bin")
    git_repo.git("switch", "-q", "-c", "feat/x")
    git_repo.commit_file("x.txt", "x\n", "x")
    git_repo.git("switch", "-q", "main")
    groups = branches_gc.classify(branches_gc.gather(git_repo.clone))
    assert [b.name for b in groups["active"]] == ["feat/x"]
    assert "gh unavailable" in capsys.readouterr().err


def test_delete_removes_only_the_merged_class(
    git_repo: GitFixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    git_repo.git("branch", "feat/merged", "origin/main")
    git_repo.git("switch", "-q", "-c", "feat/keep")
    git_repo.commit_file("k.txt", "k\n", "keep")
    git_repo.git("switch", "-q", "main")
    _fake_gh(tmp_path, monkeypatch, [])
    assert branches_gc.main(["--root", str(git_repo.clone), "--delete"]) == 0
    remaining = git_repo.git(
        "for-each-ref", "refs/heads", "--format=%(refname:short)"
    ).stdout.split()
    assert sorted(remaining) == ["feat/keep", "main"]


def test_report_only_by_default(
    git_repo: GitFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    git_repo.git("branch", "feat/merged", "origin/main")
    _fake_gh(tmp_path, monkeypatch, [])
    assert branches_gc.main(["--root", str(git_repo.clone)]) == 0
    out = capsys.readouterr().out
    assert "merged (1):" in out and "report only" in out
    assert (
        git_repo.git(
            "show-ref", "--verify", "--quiet", "refs/heads/feat/merged", check=False
        ).returncode
        == 0
    )


def test_main_rejects_non_repo(tmp_path: Path) -> None:
    assert branches_gc.main(["--root", str(tmp_path)]) == 2
