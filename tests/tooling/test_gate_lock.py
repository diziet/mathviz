"""scripts/gate_lock.py: exit-code pass-through, queueing, nesting, fail-closed."""

from __future__ import annotations

import os
import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

from tests.tooling.conftest import SCRIPTS_DIR

if TYPE_CHECKING:
    from pathlib import Path

GATE_LOCK = str(SCRIPTS_DIR / "gate_lock.py")


def _env(lock: Path) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k != "MATHVIZ_GATE_LOCK_HELD"}
    env["MATHVIZ_GATE_LOCK"] = str(lock)
    return env


def test_wrapped_exit_code_passes_through(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            GATE_LOCK,
            "--",
            sys.executable,
            "-c",
            "import sys; sys.exit(7)",
        ],
        env=_env(tmp_path / "lock"),
        check=False,
    )
    assert result.returncode == 7


HOLD_UNTIL_STDIN = "import sys; print('held', flush=True); sys.stdin.readline()"


def _start_holder(env: dict[str, str]) -> subprocess.Popen[str]:
    """Start a process that holds the lock and block until the process holds it.

    The wrapped command runs only after gate_lock.py acquired the flock, so its
    'held' line is the handshake; without it a waiter started concurrently can
    acquire the lock first and the test reads an empty waiting line (seen under load).
    """
    holder = subprocess.Popen(
        [sys.executable, GATE_LOCK, "--", sys.executable, "-c", HOLD_UNTIL_STDIN],
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    assert holder.stdout is not None
    assert holder.stdout.readline().strip() == "held"
    return holder


def test_second_process_waits_until_first_releases(tmp_path: Path) -> None:
    env = _env(tmp_path / "lock")
    holder = _start_holder(env)
    assert holder.stdin is not None
    waiter = subprocess.Popen(
        [sys.executable, GATE_LOCK, "--", sys.executable, "-c", "print('ran')"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert waiter.stderr is not None
    waiting_line = waiter.stderr.readline()
    assert waiting_line.startswith(f"waiting for gate lock held by {holder.pid}")
    assert waiter.poll() is None
    holder.stdin.write("\n")
    holder.stdin.close()
    assert holder.wait() == 0
    stdout, _ = waiter.communicate()
    assert waiter.returncode == 0
    assert stdout.strip() == "ran"


def test_nested_invocation_runs_without_waiting(tmp_path: Path) -> None:
    env = _env(tmp_path / "lock")
    holder = _start_holder(env)
    assert holder.stdin is not None
    nested = subprocess.run(
        [sys.executable, GATE_LOCK, "--", sys.executable, "-c", "print('nested')"],
        env={**env, "MATHVIZ_GATE_LOCK_HELD": str(holder.pid)},
        capture_output=True,
        text=True,
        check=False,
    )
    holder.stdin.write("\n")
    holder.stdin.close()
    holder.wait()
    assert nested.returncode == 0
    assert nested.stdout.strip() == "nested"
    assert "waiting" not in nested.stderr


def test_fails_closed_when_lock_file_cannot_be_opened(tmp_path: Path) -> None:
    marker = tmp_path / "ran.marker"
    result = subprocess.run(
        [
            sys.executable,
            GATE_LOCK,
            "--",
            sys.executable,
            "-c",
            f"open({str(marker)!r}, 'w')",
        ],
        env=_env(tmp_path / "missing-dir" / "lock"),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 97
    assert "failed closed" in result.stderr
    assert not marker.exists()


@pytest.mark.parametrize("argv", [[], ["--"]])
def test_usage_error_without_command(argv: list[str], tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, GATE_LOCK, *argv],
        env=_env(tmp_path / "lock"),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
