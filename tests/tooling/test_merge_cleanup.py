"""Post-merge cleanup preserves work created while the gate is running."""

from pathlib import Path

import merge
import pytest

from tests.tooling import test_merge

pr = test_merge.pr


@pytest.mark.parametrize("change", ["tracked", "untracked", "commit"])
def test_cleanup_preserves_work_added_during_gate(
    pr: test_merge.MergeFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    change: str,
) -> None:
    """A completed merge does not authorize discarding later local work."""
    worktree = tmp_path / "working"
    pr.repo.git("worktree", "add", "-q", str(worktree), "feat/x")
    name = "feature.py" if change == "tracked" else "new-work.txt"

    def gate(*_args: object) -> None:
        if change == "commit":
            pr.repo.commit_file(name, "keep me\n", "later work", cwd=worktree)
        else:
            (worktree / name).write_text("keep me\n")

    monkeypatch.setattr(merge, "run_preview_gate", gate)
    assert pr.run(root=worktree) == 0
    assert (worktree / name).read_text() == "keep me\n"
    assert pr.repo.head("feat/x") == pr.repo.head(cwd=worktree)
    assert "cleanup skipped (open)" in capsys.readouterr().out


def test_cleanup_refuses_a_locked_worktree(
    pr: test_merge.MergeFixture, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Cleanup is advisory and honors git's worktree protection."""
    worktree = tmp_path / "locked"
    pr.repo.git("worktree", "add", "-q", str(worktree), "feat/x")
    pr.repo.git("worktree", "lock", str(worktree))
    assert pr.run(root=worktree) == 0
    assert worktree.exists()
    assert "cleanup skipped (open)" in capsys.readouterr().out
