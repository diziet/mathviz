"""Fixtures for the repo-tooling tests: throwaway git repos with the hooks installed."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
HOOKS_DIR = REPO_ROOT / ".githooks"
NO_HOOKS = ["-c", "core.hooksPath=/dev/null"]

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))


@dataclass
class GitFixture:
    """A bare origin, a clone with the repo's hooks, and a hook-less second clone."""

    origin: Path
    clone: Path
    other: Path
    env: dict[str, str] = field(default_factory=dict)

    def git(
        self, *args: str, cwd: Path | None = None, check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        """Run git in the clone (or `cwd`) with the isolated environment."""
        return subprocess.run(
            ["git", *args],
            cwd=cwd or self.clone,
            env=self.env,
            capture_output=True,
            text=True,
            check=check,
        )

    def head(self, ref: str = "HEAD", cwd: Path | None = None) -> str:
        """Resolve a ref to a sha."""
        return self.git("rev-parse", ref, cwd=cwd).stdout.strip()

    def commit_file(
        self, name: str, content: str, message: str, cwd: Path | None = None
    ) -> str:
        """Write, stage and commit a file; return the new sha."""
        cwd = cwd or self.clone
        (cwd / name).write_text(content)
        self.git("add", name, cwd=cwd)
        self.git("commit", "-q", "-m", message, cwd=cwd)
        return self.head(cwd=cwd)

    def push_from_other(self, branch: str, message: str) -> str:
        """Commit on `branch` in the hook-less clone and push it to origin."""
        self.git("fetch", "-q", "origin", cwd=self.other)
        if self.git(
            "show-ref",
            "--verify",
            "--quiet",
            f"refs/heads/{branch}",
            cwd=self.other,
            check=False,
        ).returncode:
            self.git("switch", "-q", "-c", branch, f"origin/{branch}", cwd=self.other)
        else:
            self.git("switch", "-q", branch, cwd=self.other)
            self.git("merge", "-q", "--ff-only", f"origin/{branch}", cwd=self.other)
        sha = self.commit_file(f"{message}.txt", message, message, cwd=self.other)
        self.git("push", "-q", "origin", branch, cwd=self.other)
        return sha


def _isolated_git_env(tmp_path: Path) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(
        {
            "GIT_CONFIG_GLOBAL": str(tmp_path / "gitconfig"),
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_AUTHOR_NAME": "tooling-test",
            "GIT_AUTHOR_EMAIL": "tooling@example.invalid",
            "GIT_COMMITTER_NAME": "tooling-test",
            "GIT_COMMITTER_EMAIL": "tooling@example.invalid",
            "MATHVIZ_GATE_LOCK": str(tmp_path / "gate.lock"),
        }
    )
    env.pop("MATHVIZ_GATE_LOCK_HELD", None)
    return env


def install_tooling(clone: Path) -> None:
    """Copy the hooks, guard library, sync script and Makefile into a fixture clone."""
    shutil.copytree(HOOKS_DIR, clone / ".githooks")
    (clone / "scripts").mkdir(exist_ok=True)
    for name in ("guard_main.sh", "sync.sh"):
        shutil.copy(SCRIPTS_DIR / name, clone / "scripts" / name)
    shutil.copy(REPO_ROOT / "Makefile", clone / "Makefile")
    (clone / "pyproject.toml").write_text(
        '[project]\nname = "fixture"\nversion = "0"\n'
    )


@pytest.fixture
def isolated_git_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Run git in tests without the caller's GIT_* variables or user config."""
    for key in [k for k in os.environ if k.startswith("GIT_")]:
        monkeypatch.delenv(key)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(tmp_path / "gitconfig"))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")


@pytest.fixture
def git_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> GitFixture:
    """Bare origin + clone (hooks on, main pushed) + hook-less second clone."""
    env = _isolated_git_env(tmp_path)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("MATHVIZ_GATE_LOCK_HELD", raising=False)
    origin = tmp_path / "origin.git"
    clone = tmp_path / "clone"
    other = tmp_path / "other"
    fixture = GitFixture(origin=origin, clone=clone, other=other, env=env)
    subprocess.run(
        ["git", "init", "-q", "--bare", "-b", "main", str(origin)], env=env, check=True
    )
    subprocess.run(["git", "clone", "-q", str(origin), str(clone)], env=env, check=True)
    fixture.git("switch", "-q", "-c", "main", check=False)
    install_tooling(clone)
    (clone / "README.md").write_text("# fixture\n")
    fixture.git("add", "-A")
    fixture.git("commit", "-q", "-m", "init")
    fixture.git("push", "-q", "-u", "origin", "main")
    fixture.git("config", "core.hooksPath", ".githooks")
    subprocess.run(["git", "clone", "-q", str(origin), str(other)], env=env, check=True)
    return fixture
