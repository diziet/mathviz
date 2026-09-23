#!/usr/bin/env bash
# Read-only local main: the guard functions behind the .githooks/ wrappers.
#
# Local `main` is a mirror of origin/main. Work happens in sibling worktrees (`make worktree`),
# lands through `make merge pr=N`, and main only moves by fast-forwarding to origin/main.
# Sourced by the hooks; every function exits non-zero with a message when it refuses.
# All guards fail CLOSED on a refusal and OPEN only when git itself cannot answer.

guard_main_commit() {
  local branch
  branch="$(git symbolic-ref --quiet --short HEAD 2>/dev/null || true)"
  if [ "$branch" = "main" ]; then
    echo "blocked: local main is a read-only mirror of origin/main; no commits on main." >&2
    echo "  Start work in a sibling worktree:  make worktree b=feat/<name>" >&2
    echo "  Land it with a gated PR:           make merge pr=<N>" >&2
    exit 1
  fi
}

# pre-push stdin: <local ref> <local sha> <remote ref> <remote sha> per line.
guard_main_push() {
  local _local_ref _local_sha remote_ref _remote_sha
  while read -r _local_ref _local_sha remote_ref _remote_sha; do
    if [ "$remote_ref" = "refs/heads/main" ]; then
      echo "blocked: refs/heads/main is only moved by the merge gate (make merge pr=<N>)." >&2
      echo "  Push a branch and open a PR instead:  git push -u origin HEAD && gh pr create" >&2
      exit 1
    fi
  done
}

# reference-transaction: $1 is prepared|committed|aborted; stdin has <old> <new> <ref> lines.
# Refusing in the `prepared` phase is what aborts the transaction (a `committed` refusal is too
# late: the ref has already moved). refs/heads/main may only move to a commit origin/main
# contains; origin/main updated in the same transaction (git fetch origin main:main) counts.
guard_main_ff() {
  local state="$1" old new ref main_old="" main_new="" remote_new=""
  [ "$state" = "prepared" ] || return 0
  while read -r old new ref; do
    case "$ref" in
      refs/heads/main) main_old="$old"; main_new="$new" ;;
      refs/remotes/origin/main) remote_new="$new" ;;
    esac
  done
  [ -n "$main_new" ] || return 0
  case "$main_new" in *[!0]*) ;; *) return 0 ;; esac   # all-zero = deletion; allow
  [ "$main_old" = "$main_new" ] && return 0
  if [ -z "$remote_new" ]; then
    remote_new="$(git rev-parse --quiet --verify refs/remotes/origin/main 2>/dev/null || true)"
    [ -n "$remote_new" ] || return 0                  # no origin/main yet: fail open
  fi
  if [ "$main_new" != "$remote_new" ] \
     && ! git merge-base --is-ancestor "$main_new" "$remote_new" 2>/dev/null; then
    echo "blocked: local main may only move to a commit origin/main already contains." >&2
    echo "  Sync with: make sync   (or git pull --ff-only origin main)" >&2
    echo "  Land work only via: make merge pr=<N>" >&2
    exit 1
  fi
}

# Apply ruff's safe fixes to staged .py files that have no unstaged edits and re-stage them,
# then lint the staged set (blocking). A file with unstaged edits is linted but not fixed, so the
# hook never stages edits the author did not stage. The repo does not run `ruff format`, so this
# step does not format. Skips with one line when the worktree has no venv.
precommit_ruff() {
  local root ruff staged file
  root="$(git rev-parse --show-toplevel)"
  ruff="$root/.venv/bin/ruff"
  if [ ! -x "$ruff" ]; then
    echo "pre-commit: no .venv/bin/ruff in this worktree; skipping ruff check" >&2
    return 0
  fi
  staged="$(git diff --cached --name-only --diff-filter=ACMR -- '*.py')"
  [ -n "$staged" ] || return 0
  while IFS= read -r file; do
    [ -f "$root/$file" ] || continue
    if git diff --quiet -- "$file"; then
      # Exit 1 means findings remain; the check below reports them.
      (cd "$root" && "$ruff" check --fix --quiet "$file" >/dev/null 2>&1) || true
      git add -- "$file"
    else
      echo "pre-commit: $file has unstaged edits; ruff fixes not applied" >&2
    fi
  done <<< "$staged"
  # shellcheck disable=SC2086
  (cd "$root" && "$ruff" check $staged)
}
