"""scripts/merge.py: refusal paths, dry run and the gh merge call (gh stubbed)."""

from __future__ import annotations

import fcntl
import json
import os
import stat
import subprocess
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import gate_lock
import merge
import merge_gate
import pytest

from tests.tooling.conftest import NO_HOOKS, GitFixture

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

FAKE_GH = """#!/usr/bin/env bash
case "$1 $2" in
  "pr view") cat "$FAKE_GH_VIEW_JSON" ;;
  "pr merge") printf '%s\\n' "$*" >> "$FAKE_GH_MERGE_LOG" ;;
  *) echo "fake gh: unsupported: $*" >&2; exit 1 ;;
esac
"""


@dataclass
class MergeFixture:
    """PR-shaped fixture: feat/x pushed to origin, gh stubbed, gate stubbed to pass."""

    repo: GitFixture
    view_json: Path
    merge_log: Path
    head_sha: str

    def set_pr(self, **overrides: str) -> None:
        data = {
            "headRefName": "feat/x",
            "headRefOid": self.head_sha,
            "baseRefName": "main",
            "state": "OPEN",
            "mergeable": "MERGEABLE",
        }
        data.update(overrides)
        self.view_json.write_text(json.dumps(data))

    def run(self, *args: str, root: Path | None = None) -> int:
        return merge.main(["--pr", "7", "--root", str(root or self.repo.clone), *args])


@pytest.fixture
def pr(
    git_repo: GitFixture, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> MergeFixture:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    gh = bin_dir / "gh"
    gh.write_text(FAKE_GH)
    gh.chmod(gh.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    view_json = tmp_path / "pr.json"
    merge_log = tmp_path / "merge.log"
    monkeypatch.setenv("FAKE_GH_VIEW_JSON", str(view_json))
    monkeypatch.setenv("FAKE_GH_MERGE_LOG", str(merge_log))
    monkeypatch.setenv(merge.GATE_CMD_ENV, "true")
    git_repo.git("switch", "-q", "-c", "feat/x")
    head_sha = git_repo.commit_file("feature.py", "X = 1\n", "feature")
    git_repo.git("push", "-q", "-u", "origin", "feat/x")
    git_repo.git("switch", "-q", "main")
    fixture = MergeFixture(git_repo, view_json, merge_log, head_sha)
    fixture.set_pr()
    return fixture


def _err(capsys: pytest.CaptureFixture[str]) -> str:
    return capsys.readouterr().err


def test_refuses_dirty_tree(
    pr: MergeFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    (pr.repo.clone / "untracked.txt").write_text("x\n")
    assert pr.run() == 1
    err = _err(capsys)
    assert "refused (closed)" in err and "untracked.txt" in err


def test_refuses_divergent_local_main(
    pr: MergeFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    (pr.repo.clone / "local.txt").write_text("x\n")
    pr.repo.git("add", "local.txt")
    pr.repo.git(*NO_HOOKS, "commit", "-q", "-m", "stranded on main")
    assert pr.run() == 1
    assert "origin/main lacks" in _err(capsys)


def test_refuses_pr_that_is_not_open(
    pr: MergeFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    pr.set_pr(state="MERGED")
    assert pr.run() == 1
    assert "not OPEN" in _err(capsys)


def test_refuses_pr_not_targeting_main(
    pr: MergeFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    pr.set_pr(baseRefName="develop")
    assert pr.run() == 1
    assert "not main" in _err(capsys)


def test_refuses_when_origin_main_moves_during_the_gate(
    pr: MergeFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    mover = tmp_path / "move_main.sh"
    mover.write_text(
        f"#!/bin/sh\ncd {pr.repo.other} && git commit -q --allow-empty -m moved "
        "&& git push -q origin main\n"
    )
    monkeypatch.setenv(merge.GATE_CMD_ENV, f"sh {mover}")
    assert pr.run() == 1
    err = _err(capsys)
    assert "origin/main moved" in err and "rerun make merge" in err
    assert not pr.merge_log.exists()


def test_base_is_resolved_after_the_lock_is_acquired(
    pr: MergeFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A merge that reached main while this run queued is gated against, not refused."""
    moved = False

    @contextmanager
    def lock_then_move_main() -> Iterator[None]:
        nonlocal moved
        pr.repo.git("commit", "-q", "--allow-empty", "-m", "moved", cwd=pr.repo.other)
        pr.repo.git("push", "-q", "origin", "main", cwd=pr.repo.other)
        moved = True
        yield

    monkeypatch.setattr(merge, "gate_lock", lock_then_move_main)
    assert pr.run("--keep") == 0
    assert moved and pr.merge_log.exists()


def test_refuses_when_pr_head_moves_during_the_gate(
    pr: MergeFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    mover = tmp_path / "move_head.sh"
    mover.write_text(
        f"#!/bin/sh\ncd {pr.repo.other} && git fetch -q origin "
        "&& git switch -q -c feat/x origin/feat/x "
        "&& git commit -q --allow-empty -m moved && git push -q origin feat/x\n"
    )
    monkeypatch.setenv(merge.GATE_CMD_ENV, f"sh {mover}")
    assert pr.run() == 1
    assert "PR head moved" in _err(capsys)
    assert not pr.merge_log.exists()


def test_gate_failure_refuses_and_removes_temp_worktree(
    pr: MergeFixture,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv(merge.GATE_CMD_ENV, "false")
    assert pr.run() == 1
    assert "gate failed (exit 1)" in _err(capsys)
    worktrees = pr.repo.git("worktree", "list", "--porcelain").stdout
    assert "mathviz-merge-" not in worktrees


def test_conflict_lists_conflicting_files(
    pr: MergeFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    pr.repo.git("switch", "-q", "feat/x")
    pr.head_sha = pr.repo.commit_file("README.md", "# feature\n", "feature readme")
    pr.repo.git("push", "-q", "origin", "feat/x")
    pr.repo.git("switch", "-q", "main")
    pr.set_pr()
    pr.repo.git("switch", "-q", "main", cwd=pr.repo.other)
    pr.repo.commit_file(
        "README.md", "# upstream\n", "upstream readme", cwd=pr.repo.other
    )
    pr.repo.git("push", "-q", "origin", "main", cwd=pr.repo.other)
    assert pr.run() == 1
    err = _err(capsys)
    assert "conflicts" in err and "README.md" in err


def test_stale_gh_head_uses_fetched_branch_tip(
    pr: MergeFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    """gh lags behind a push: the stale head conflicts, the real tip resolves it."""
    pr.repo.git("switch", "-q", "feat/x")
    stale_sha = pr.repo.commit_file("README.md", "# feature\n", "feature readme")
    pr.repo.git("push", "-q", "origin", "feat/x")
    pr.repo.git("switch", "-q", "main", cwd=pr.repo.other)
    pr.repo.commit_file(
        "README.md", "# upstream\n", "upstream readme", cwd=pr.repo.other
    )
    pr.repo.git("push", "-q", "origin", "main", cwd=pr.repo.other)
    pr.repo.git("fetch", "-q", "origin")
    pr.repo.git("merge", "-q", "--no-edit", "origin/main", check=False)
    tip_sha = pr.repo.commit_file("README.md", "# resolved\n", "resolve readme")
    pr.repo.git("push", "-q", "origin", "feat/x")
    pr.repo.git("switch", "-q", "main")
    pr.set_pr(headRefOid=stale_sha)
    assert pr.run("--dry-run") == 0
    out = capsys.readouterr().out
    assert (
        f"merge: gh reports {stale_sha}, origin/feat/x is at {tip_sha}; using origin"
        in out
    )
    assert f"feat/x@{tip_sha[:10]}" in out


def test_branch_absent_from_origin_falls_back_to_gh_head(
    pr: MergeFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    pr.set_pr(headRefName="feat/unpushed")
    assert pr.run("--dry-run") == 0
    assert "using origin" not in capsys.readouterr().out


def test_mergeable_unknown_is_not_a_refusal(pr: MergeFixture) -> None:
    pr.set_pr(mergeable="UNKNOWN")
    assert pr.run("--dry-run") == 0


def test_dry_run_stops_before_gh_pr_merge(
    pr: MergeFixture, capsys: pytest.CaptureFixture[str]
) -> None:
    assert pr.run("--dry-run") == 0
    assert "dry run" in capsys.readouterr().out
    assert not pr.merge_log.exists()


def test_merges_via_gh_with_merge_commit_subject(pr: MergeFixture) -> None:
    assert pr.run("--keep") == 0
    logged = pr.merge_log.read_text().strip()
    assert logged.startswith("pr merge 7 --merge --subject Merge pull request #7 from ")
    assert f"/feat/x --match-head-commit {pr.head_sha}" in logged


@pytest.mark.parametrize(
    "stage", ["run_preview_gate", "verify_parents_unchanged", "merge_via_gh"]
)
def test_gate_lock_covers_verification_and_merge(
    pr: MergeFixture, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    """A competing local gate cannot acquire the lock before the merge finishes."""
    original = getattr(merge, stage)
    checked = False

    def guarded(*args: object) -> object:
        nonlocal checked
        with gate_lock.lock_path().open("a") as handle, pytest.raises(BlockingIOError):
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        checked = True
        return original(*args)

    monkeypatch.setattr(merge, stage, guarded)
    assert pr.run("--keep") == 0
    assert checked


def test_server_refusal_of_changed_head_is_not_reported_as_merged(
    pr: MergeFixture,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The head can move after the fetch; the server's conditional refusal is reported."""
    original = merge_gate.run

    def server(
        args: list[str], cwd: Path, check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        if args[:3] == ["gh", "pr", "merge"]:
            assert args[-2:] == ["--match-head-commit", pr.head_sha]
            return subprocess.CompletedProcess(args, 1, "", "head changed")
        return original(args, cwd, check)

    monkeypatch.setattr(merge_gate, "run", server)
    assert pr.run("--keep") == 1
    captured = capsys.readouterr()
    assert "head changed" in captured.err
    assert "PR #7 merged" not in captured.out


def test_success_from_merged_worktree_removes_it_and_prints_cd(
    pr: MergeFixture, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    worktree = tmp_path / "wt-feat-x"
    pr.repo.git("worktree", "add", "-q", str(worktree), "feat/x")
    assert pr.run(root=worktree) == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert out[-1] == f"cd {pr.repo.clone}"
    assert not worktree.exists()
    assert (
        pr.repo.git(
            "show-ref", "--verify", "--quiet", "refs/heads/feat/x", check=False
        ).returncode
        != 0
    )


def test_success_with_keep_leaves_branch_in_place(
    pr: MergeFixture, tmp_path: Path
) -> None:
    worktree = tmp_path / "wt-keep"
    pr.repo.git("worktree", "add", "-q", str(worktree), "feat/x")
    assert pr.run("--keep", root=worktree) == 0
    assert worktree.exists()


def test_main_rejects_non_repo_root(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert merge.main(["--pr", "1", "--root", str(tmp_path)]) == 2
    assert "refused (closed)" in _err(capsys)


@pytest.mark.parametrize(
    ("url", "owner"),
    [
        ("https://github.com/diziet/mathviz.git", "diziet"),
        ("git@github.com:diziet/mathviz.git", "diziet"),
        ("/tmp/origin.git", "tmp"),
    ],
)
def test_repo_owner_from_origin_url(git_repo: GitFixture, url: str, owner: str) -> None:
    git_repo.git("remote", "set-url", "origin", url)
    assert merge_gate.repo_owner(git_repo.clone) == owner


def test_default_gate_commands_seed_venv_then_run_full_gate(tmp_path: Path) -> None:
    commands = merge.default_gate_commands(tmp_path)
    assert commands == [
        ["make", "-C", str(tmp_path), "-s", "venv"],
        ["bash", str(tmp_path / "scripts" / "gate.sh")],
    ]

