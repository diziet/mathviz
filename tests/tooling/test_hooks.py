"""Read-only-main hooks: pre-commit, pre-merge-commit, pre-push, ref-transaction."""

from __future__ import annotations

import os
import shutil
import stat
import sys
from typing import TYPE_CHECKING

from tests.tooling.conftest import NO_HOOKS, REPO_ROOT, SCRIPTS_DIR, GitFixture

if TYPE_CHECKING:
    from pathlib import Path


DOC_FACTS_REGISTRY = """from doc_facts_sources import Fact, read_text

FACTS = {"version": Fact("VERSION", lambda root: read_text(root, "VERSION").strip())}
"""
FACT_DOC = "Version <!-- fact:version -->{}<!-- /fact -->.\n"


def _fake_venv_ruff(clone: Path) -> None:
    """A .venv/bin/ruff wrapper that execs the real ruff (a script, not a symlink)."""
    bin_dir = clone / ".venv" / "bin"
    bin_dir.mkdir(parents=True)
    wrapper = bin_dir / "ruff"
    wrapper.write_text(f'#!/bin/sh\nexec "{REPO_ROOT}/.venv/bin/ruff" "$@"\n')
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IXUSR)


def _install_doc_facts(git_repo: GitFixture) -> None:
    """Commit doc_facts.py and its modules, a one-fact registry and a current doc.

    `.venv/bin/python` is a symlink to the interpreter that runs the tests.
    """
    clone = git_repo.clone
    for name in ("doc_facts.py", "doc_facts_sources.py", "doc_common.py"):
        shutil.copy(SCRIPTS_DIR / name, clone / "scripts" / name)
    (clone / "scripts" / "doc_facts_registry.py").write_text(DOC_FACTS_REGISTRY)
    (clone / ".venv" / "bin").mkdir(parents=True)
    (clone / ".venv" / "bin" / "python").symlink_to(sys.executable)
    (clone / "VERSION").write_text("1.0\n")
    (clone / "README.md").write_text(FACT_DOC.format("1.0"))
    git_repo.git("switch", "-q", "-c", "feat/facts")
    git_repo.git("add", "scripts", "VERSION", "README.md")
    git_repo.git("commit", "-q", "-m", "doc facts")


def test_pre_commit_regenerates_and_stages_a_stale_doc_fact(
    git_repo: GitFixture,
) -> None:
    _install_doc_facts(git_repo)
    (git_repo.clone / "VERSION").write_text("2.0\n")
    git_repo.git("add", "VERSION")
    result = git_repo.git("commit", "-q", "-m", "bump", check=False)
    assert result.returncode == 0, result.stderr
    assert "regenerated and staged doc facts in README.md" in result.stderr
    assert git_repo.git("show", "HEAD:README.md").stdout == FACT_DOC.format("2.0")


def test_pre_commit_leaves_a_doc_with_unstaged_edits_unstaged(
    git_repo: GitFixture,
) -> None:
    _install_doc_facts(git_repo)
    (git_repo.clone / "VERSION").write_text("3.0\n")
    git_repo.git("add", "VERSION")
    readme = git_repo.clone / "README.md"
    readme.write_text(readme.read_text() + "Unrelated draft.\n")
    result = git_repo.git("commit", "-q", "-m", "bump", check=False)
    assert result.returncode == 0, result.stderr
    assert "not staged, it has unstaged edits" in result.stderr
    assert git_repo.git("show", "HEAD:README.md").stdout == FACT_DOC.format("1.0")
    assert readme.read_text() == FACT_DOC.format("3.0") + "Unrelated draft.\n"


def test_pre_commit_does_not_block_when_doc_facts_fails(git_repo: GitFixture) -> None:
    _install_doc_facts(git_repo)
    (git_repo.clone / "NOTES.md").write_text(
        "Say <!-- fact:unknown -->x<!-- /fact -->.\n"
    )
    git_repo.git("add", "NOTES.md")
    result = git_repo.git("commit", "-q", "-m", "notes", check=False)
    assert result.returncode == 0, result.stderr
    assert "doc facts not regenerated" in result.stderr
    assert "unknown fact 'unknown'" in result.stderr


def test_pre_commit_refuses_commit_on_main(git_repo: GitFixture) -> None:
    (git_repo.clone / "new.txt").write_text("x\n")
    git_repo.git("add", "new.txt")
    result = git_repo.git("commit", "-q", "-m", "on main", check=False)
    assert result.returncode != 0
    assert "make worktree" in result.stderr
    assert git_repo.head() == git_repo.head("origin/main")


def test_pre_commit_allows_commit_on_feature_branch(git_repo: GitFixture) -> None:
    git_repo.git("switch", "-q", "-c", "feat/x")
    (git_repo.clone / "new.txt").write_text("x\n")
    git_repo.git("add", "new.txt")
    result = git_repo.git("commit", "-q", "-m", "on branch", check=False)
    assert result.returncode == 0, result.stderr
    assert "skipping ruff" in result.stderr


FIXABLE_SOURCE = "import os\nimport sys\n\nprint(sys.argv)\n"


def test_pre_commit_fixes_and_restages_clean_staged_file(
    git_repo: GitFixture,
) -> None:
    _fake_venv_ruff(git_repo.clone)
    git_repo.git("switch", "-q", "-c", "feat/fix")
    (git_repo.clone / "mod.py").write_text(FIXABLE_SOURCE)
    git_repo.git("add", "mod.py")
    result = git_repo.git("commit", "-q", "-m", "fix", check=False)
    assert result.returncode == 0, result.stderr
    committed = git_repo.git("show", "HEAD:mod.py").stdout
    assert committed == "import sys\n\nprint(sys.argv)\n"


def test_pre_commit_does_not_format(git_repo: GitFixture) -> None:
    _fake_venv_ruff(git_repo.clone)
    git_repo.git("switch", "-q", "-c", "feat/nofmt")
    (git_repo.clone / "mod.py").write_text("x=1\n")
    git_repo.git("add", "mod.py")
    result = git_repo.git("commit", "-q", "-m", "nofmt", check=False)
    assert result.returncode == 0, result.stderr
    assert git_repo.git("show", "HEAD:mod.py").stdout == "x=1\n"


def test_pre_commit_skips_fixes_for_file_with_unstaged_edits(
    git_repo: GitFixture,
) -> None:
    _fake_venv_ruff(git_repo.clone)
    git_repo.git("switch", "-q", "-c", "feat/partial")
    module = git_repo.clone / "mod.py"
    module.write_text(FIXABLE_SOURCE)
    git_repo.git("add", "mod.py")
    module.write_text(FIXABLE_SOURCE + "print(sys.path)\n")
    result = git_repo.git("commit", "-q", "-m", "partial", check=False)
    assert result.returncode != 0
    assert "mod.py has unstaged edits" in result.stderr
    assert "F401" in result.stdout + result.stderr
    assert git_repo.git("show", ":mod.py").stdout == FIXABLE_SOURCE


def test_pre_commit_blocks_lint_error_in_staged_file(git_repo: GitFixture) -> None:
    _fake_venv_ruff(git_repo.clone)
    git_repo.git("switch", "-q", "-c", "feat/lint")
    (git_repo.clone / "bad.py").write_text("print(undefined_name)\n")
    git_repo.git("add", "bad.py")
    result = git_repo.git("commit", "-q", "-m", "lint", check=False)
    assert result.returncode != 0
    assert "F821" in result.stdout + result.stderr


def test_pre_merge_commit_refuses_merge_into_main(git_repo: GitFixture) -> None:
    git_repo.git("switch", "-q", "-c", "feat/m")
    git_repo.commit_file("m.txt", "m\n", "feature")
    git_repo.git("switch", "-q", "main")
    result = git_repo.git("merge", "--no-ff", "--no-edit", "feat/m", check=False)
    assert result.returncode != 0
    assert "read-only mirror" in result.stderr
    assert git_repo.head("main") == git_repo.head("origin/main")


def test_pre_push_refuses_push_to_main(git_repo: GitFixture) -> None:
    git_repo.git("switch", "-q", "-c", "feat/p")
    git_repo.commit_file("p.txt", "p\n", "feature")
    before = git_repo.head("main", cwd=git_repo.origin)
    result = git_repo.git("push", "origin", "HEAD:refs/heads/main", check=False)
    assert result.returncode != 0
    assert "refs/heads/main" in result.stderr
    assert git_repo.head("main", cwd=git_repo.origin) == before


def test_pre_push_allows_feature_branch(git_repo: GitFixture) -> None:
    git_repo.git("switch", "-q", "-c", "feat/ok")
    sha = git_repo.commit_file("ok.txt", "ok\n", "feature")
    result = git_repo.git("push", "-q", "-u", "origin", "feat/ok", check=False)
    assert result.returncode == 0, result.stderr
    assert git_repo.head("feat/ok", cwd=git_repo.origin) == sha


def test_reference_transaction_refuses_non_fast_forward_move_of_main(
    git_repo: GitFixture,
) -> None:
    git_repo.git("switch", "-q", "-c", "feat/rt")
    feature_sha = git_repo.commit_file("rt.txt", "rt\n", "feature")
    main_before = git_repo.head("main")
    result = git_repo.git("update-ref", "refs/heads/main", feature_sha, check=False)
    assert result.returncode != 0
    assert "origin/main already contains" in result.stderr
    assert git_repo.head("main") == main_before


def test_reference_transaction_refuses_fast_forward_merge_of_feature_into_main(
    git_repo: GitFixture,
) -> None:
    git_repo.git("switch", "-q", "-c", "feat/ff")
    git_repo.commit_file("ff.txt", "ff\n", "feature")
    git_repo.git("switch", "-q", "main")
    result = git_repo.git("merge", "--ff-only", "feat/ff", check=False)
    assert result.returncode != 0
    assert git_repo.head("main") == git_repo.head("origin/main")


def test_reference_transaction_allows_fast_forward_from_origin(
    git_repo: GitFixture,
) -> None:
    new_sha = git_repo.push_from_other("main", "upstream")
    git_repo.git("fetch", "-q", "origin")
    result = git_repo.git("merge", "--ff-only", "origin/main", check=False)
    assert result.returncode == 0, result.stderr
    assert git_repo.head("main") == new_sha


def test_reference_transaction_allows_fetch_into_main_from_branch(
    git_repo: GitFixture,
) -> None:
    new_sha = git_repo.push_from_other("main", "upstream2")
    git_repo.git("switch", "-q", "-c", "feat/elsewhere")
    result = git_repo.git("fetch", "-q", "origin", "main:main", check=False)
    assert result.returncode == 0, result.stderr
    assert git_repo.head("main") == new_sha


def test_hooks_skip_silently_when_guard_library_is_absent(git_repo: GitFixture) -> None:
    git_repo.git("rm", "-q", "scripts/guard_main.sh")
    result = git_repo.git("commit", "-q", "-m", "old branch shape", check=False)
    assert result.returncode == 0, result.stderr
    assert result.stderr.strip() == ""


def test_hooks_are_executable_in_the_repo() -> None:
    for hook in ("pre-commit", "pre-merge-commit", "pre-push", "reference-transaction"):
        assert os.access(REPO_ROOT / ".githooks" / hook, os.X_OK), hook


def test_hookless_commit_on_main_is_possible_for_fixture_setup(
    git_repo: GitFixture,
) -> None:
    """Sanity: the NO_HOOKS override that other tests rely on bypasses the guard."""
    (git_repo.clone / "raw.txt").write_text("raw\n")
    git_repo.git("add", "raw.txt")
    result = git_repo.git(*NO_HOOKS, "commit", "-q", "-m", "bypass", check=False)
    assert result.returncode == 0, result.stderr
