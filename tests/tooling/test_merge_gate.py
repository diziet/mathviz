"""scripts/merge_gate.preview_merge: the refusal message on a failed git merge."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

import merge_gate
import pytest

from tests.tooling.conftest import NO_HOOKS, GitFixture

if TYPE_CHECKING:
    from pathlib import Path

UNKNOWN_SHA = "0" * 40


def _conflicting_branch(git_repo: GitFixture) -> tuple[str, str]:
    """Return (base_sha, head_sha) where both sides rewrite README.md."""
    git_repo.git("switch", "-q", "-c", "feat/c")
    head_sha = git_repo.commit_file("README.md", "# feature\n", "feature readme")
    git_repo.git("switch", "-q", "main")
    (git_repo.clone / "README.md").write_text("# upstream\n")
    git_repo.git("add", "README.md")
    git_repo.git(*NO_HOOKS, "commit", "-q", "-m", "upstream readme")
    return git_repo.head(), head_sha


def _preview_refusal(repo: Path, base_sha: str, head_sha: str) -> str:
    with (
        merge_gate.temp_worktree(repo, base_sha) as tree,
        pytest.raises(merge_gate.MergeRefusalError) as refusal,
    ):
        merge_gate.preview_merge(tree, head_sha)
    return str(refusal.value)


def test_conflicting_preview_names_the_conflicting_file(git_repo: GitFixture) -> None:
    base_sha, head_sha = _conflicting_branch(git_repo)
    message = _preview_refusal(git_repo.clone, base_sha, head_sha)
    assert message.startswith("preview merge has conflicts in:")
    assert "README.md" in message
    assert "merge failed" not in message


def test_non_conflict_failure_reports_git_message(git_repo: GitFixture) -> None:
    message = _preview_refusal(git_repo.clone, git_repo.head(), UNKNOWN_SHA)
    assert message.startswith("preview merge failed:")
    assert "not something we can merge" in message
    assert "conflicts in" not in message


def test_failure_message_includes_stdout_and_stderr(git_repo: GitFixture) -> None:
    result = subprocess.CompletedProcess(
        ["git", "merge"], 1, "CONFLICT (content): Merge conflict in x\n", "hook: no\n"
    )
    message = merge_gate.describe_failed_merge(git_repo.clone, result)
    assert "CONFLICT (content): Merge conflict in x" in message
    assert "hook: no" in message


def test_failure_message_is_never_empty(git_repo: GitFixture) -> None:
    result = subprocess.CompletedProcess(["git", "merge"], 128, "", "")
    message = merge_gate.describe_failed_merge(git_repo.clone, result)
    assert message == "preview merge failed: git merge exited 128 with no output"


def test_preview_tree_is_on_a_throwaway_branch_removed_on_exit(
    git_repo: GitFixture,
) -> None:
    """buildbanner omits "branch" for a detached HEAD, and test_build_banner asserts it."""
    with merge_gate.temp_worktree(git_repo.clone, git_repo.head()) as tree:
        branch = git_repo.git("symbolic-ref", "--quiet", "--short", "HEAD", cwd=tree)
        name = branch.stdout.strip()
        assert name == f"merge-preview/{tree.name}"
    assert not tree.exists()
    listed = git_repo.git("branch", "--list", "merge-preview/*").stdout
    assert listed.strip() == ""
