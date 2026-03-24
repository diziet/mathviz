"""BuildBanner endpoint serving git metadata and server uptime."""

import logging
import subprocess
import time

from fastapi import APIRouter

from mathviz.core.generator import list_generators

logger = logging.getLogger(__name__)

router = APIRouter()

_server_start_time = time.time()


def _run_git(args: list[str]) -> str:
    """Run a git command and return stripped stdout, or empty on failure."""
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""


def _get_git_branch() -> str:
    """Return the current git branch name."""
    return _run_git(["rev-parse", "--abbrev-ref", "HEAD"])


def _get_git_commit() -> str:
    """Return the short commit SHA."""
    return _run_git(["rev-parse", "--short", "HEAD"])


# Cache git metadata at module load — branch and commit don't change while the server runs
_cached_branch = _get_git_branch()
_cached_commit = _get_git_commit()
_cached_generator_count = len(list_generators())


def _format_uptime(seconds: float) -> str:
    """Format seconds as a human-readable uptime string."""
    minutes, secs = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    if minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


@router.get("/buildbanner.json")
def get_build_banner() -> dict:
    """Return build metadata for the BuildBanner client."""
    uptime_seconds = time.time() - _server_start_time

    return {
        "branch": _cached_branch,
        "commit": _cached_commit,
        "uptime": _format_uptime(uptime_seconds),
        "generators": _cached_generator_count,
    }
