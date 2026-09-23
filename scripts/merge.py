"""The only sanctioned merge path: `make merge pr=N [keep=1] [dry_run=1]`.

Refuses on a dirty tree or a divergent local main, requires an OPEN PR against
main, runs the full gate on a preview merge in a throwaway worktree under the
machine-wide gate lock, verifies neither parent moved, then merges through
`gh pr merge` (scripts/merge_gate.merge_via_gh is the only place that command
may be called). Every refusal fails CLOSED; post-merge cleanup fails OPEN.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from dataclasses import replace
from functools import partial
from pathlib import Path

from gate_lock import gate_lock
from merge_gate import (
    MergeRefusalError,
    PullRequest,
    ensure_object,
    fetch_pull_request,
    git,
    merge_via_gh,
    preview_merge,
    refuse_dirty_tree,
    refuse_divergent_main,
    remote_branch_sha,
    repo_owner,
    run,
    temp_worktree,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
GATE_CMD_ENV = "MATHVIZ_MERGE_GATE_CMD"
# Subprocess output bypasses our buffer; flush so status lines stay in order.
say = partial(print, flush=True)


def default_gate_commands(tree: Path) -> list[list[str]]:
    """Seed the preview tree's venv, then run the shared gate stage list."""
    return [
        ["make", "-C", str(tree), "-s", "venv"],
        ["bash", str(tree / "scripts" / "gate.sh")],
    ]


def run_preview_gate(tree: Path, gate_cmd: str | None) -> None:
    """Run the gate in the preview tree; raise MergeRefusalError on a red stage."""
    commands = [shlex.split(gate_cmd)] if gate_cmd else default_gate_commands(tree)
    for command in commands:
        result = subprocess.run(command, cwd=tree, check=False)
        if result.returncode:
            raise MergeRefusalError(
                f"gate failed (exit {result.returncode}): {shlex.join(command)}"
            )


def resolve_head(repo: Path, pr: PullRequest) -> PullRequest:
    """Prefer the fetched origin/<branch> tip; gh's headRefOid lags after a push."""
    # TODO: a fork PR whose branch name also exists on origin would resolve to the
    # wrong tip; the spec only covers branches pushed to origin.
    remote_sha = remote_branch_sha(repo, pr.head_ref)
    if remote_sha is None or remote_sha == pr.head_sha:
        return pr
    say(
        f"merge: gh reports {pr.head_sha}, origin/{pr.head_ref} is at {remote_sha}; "
        "using origin"
    )
    return replace(pr, head_sha=remote_sha)


def verify_parents_unchanged(repo: Path, pr: PullRequest, base_sha: str) -> None:
    """Re-fetch and require origin/main and origin/<branch> to be where the gate saw."""
    run(["git", "fetch", "-q", "origin"], repo)
    if git(repo, "rev-parse", "origin/main") != base_sha:
        raise MergeRefusalError("origin/main moved during the gate; rerun make merge")
    head_now = remote_branch_sha(repo, pr.head_ref)
    if head_now is None:
        head_now = fetch_pull_request(pr.number, repo).head_sha
    if head_now != pr.head_sha:
        raise MergeRefusalError("the PR head moved during the gate; rerun make merge")


def cleanup_merged_worktree(repo: Path, head_ref: str, head_sha: str) -> None:
    """Remove the invoking worktree and branch when it is the merged one; fails open."""
    branch = run(
        ["git", "symbolic-ref", "--quiet", "--short", "HEAD"], repo, check=False
    ).stdout.strip()
    if branch != head_ref:
        return
    primary = Path(
        git(repo, "rev-parse", "--path-format=absolute", "--git-common-dir")
    ).parent
    if primary == repo:
        say(
            f"merge: cleanup skipped (open): {repo} is the primary checkout; "
            "switch it to main yourself"
        )
        return
    if git(repo, "rev-parse", "HEAD") != head_sha:
        say(f"merge: cleanup skipped (open): {head_ref} has new local commits")
        return
    removed = run(["git", "worktree", "remove", str(repo)], primary, check=False)
    if removed.returncode:
        say(
            f"merge: cleanup skipped (open): could not remove worktree {repo}: "
            f"{removed.stderr.strip()}"
        )
        return
    # Never force deletion: git must preserve new, unmerged commits or a branch
    # checked out elsewhere since removal. Failure here does not undo the merge.
    deleted = run(["git", "branch", "-d", head_ref], primary, check=False)
    if deleted.returncode:
        say(
            f"merge: cleanup skipped (open): could not delete branch {head_ref}: "
            f"{deleted.stderr.strip()}"
        )
    else:
        say(f"merge: removed branch {head_ref}")
    say(f"merge: removed worktree {repo}")
    say(f"cd {primary}")


def merge_pull_request(
    repo: Path, number: int, keep: bool, dry_run: bool, gate_cmd: str | None
) -> None:
    """Run the whole merge flow; raises MergeRefusalError on any guard."""
    refuse_dirty_tree(repo)
    refuse_divergent_main(repo)
    # Resolve both parents only once the lock is held: a merge that lands while
    # this run queues would otherwise invalidate the base before the gate starts.
    with gate_lock():
        run(["git", "fetch", "-q", "origin"], repo)
        pr = resolve_head(repo, fetch_pull_request(number, repo))
        base_sha = git(repo, "rev-parse", "origin/main")
        ensure_object(repo, pr.head_sha, number)
        say(
            f"merge: PR #{number} {pr.head_ref}@{pr.head_sha[:10]} "
            f"onto origin/main@{base_sha[:10]}"
        )
        with temp_worktree(repo, base_sha) as tree:
            preview_merge(tree, pr.head_sha)
            run_preview_gate(tree, gate_cmd)
        verify_parents_unchanged(repo, pr, base_sha)
        if dry_run:
            say(
                "merge: dry run; gate passed, parents unchanged; "
                "stopping before pr merge"
            )
            return
        # Serialize cooperating local mergers through the final remote write.
        # GitHub also checks the head SHA; unrelated remote base writers are not
        # covered by this machine-local lock.
        merge_via_gh(pr, repo_owner(repo), repo)
    say(f"merge: PR #{number} merged")
    if not keep:
        cleanup_merged_worktree(repo, pr.head_ref, pr.head_sha)


def main(argv: list[str] | None = None) -> int:
    """CLI entry: exit 1 with a (closed) refusal message on any guard."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pr", type=int, required=True)
    parser.add_argument(
        "--keep", action="store_true", help="keep the merged worktree and branch"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="stop after the gate, before gh pr merge"
    )
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help=argparse.SUPPRESS)
    options = parser.parse_args(argv)
    repo: Path = options.root.resolve()
    if not (repo / "pyproject.toml").is_file():
        say(f"merge: refused (closed): {repo} is not the repo root", file=sys.stderr)
        return 2
    gate_cmd = os.environ.get(GATE_CMD_ENV) or None
    try:
        merge_pull_request(repo, options.pr, options.keep, options.dry_run, gate_cmd)
    except MergeRefusalError as refusal:
        say(f"merge: refused (closed): {refusal}", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as error:
        say(
            f"merge: refused (closed): {shlex.join(error.cmd)} failed:\n{error.stderr}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
