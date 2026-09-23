"""scripts/sync.sh: fast-forward the current branch; refuse a divergent local main."""

from __future__ import annotations

import subprocess

from tests.tooling.conftest import NO_HOOKS, GitFixture


def _sync(repo: GitFixture) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "scripts/sync.sh"],
        cwd=repo.clone,
        env=repo.env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_sync_refuses_when_local_main_diverges(git_repo: GitFixture) -> None:
    (git_repo.clone / "stranded.txt").write_text("x\n")
    git_repo.git("add", "stranded.txt")
    git_repo.git(*NO_HOOKS, "commit", "-q", "-m", "stranded")
    result = _sync(git_repo)
    assert result.returncode == 1
    assert "refused (closed)" in result.stderr


def test_sync_fast_forwards_main(git_repo: GitFixture) -> None:
    new_sha = git_repo.push_from_other("main", "upstream")
    result = _sync(git_repo)
    assert result.returncode == 0, result.stderr
    assert git_repo.head("main") == new_sha


def test_sync_fast_forwards_feature_branch(git_repo: GitFixture) -> None:
    git_repo.git("switch", "-q", "-c", "feat/s")
    git_repo.commit_file("s.txt", "s\n", "feature")
    git_repo.git("push", "-q", "-u", "origin", "feat/s")
    new_sha = git_repo.push_from_other("feat/s", "more")
    result = _sync(git_repo)
    assert result.returncode == 0, result.stderr
    assert git_repo.head("feat/s") == new_sha


def test_sync_without_remote_branch_is_a_no_op(git_repo: GitFixture) -> None:
    git_repo.git("switch", "-q", "-c", "feat/local-only")
    result = _sync(git_repo)
    assert result.returncode == 0, result.stderr
    assert "nothing to fast-forward" in result.stdout
