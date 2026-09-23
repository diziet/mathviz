#!/usr/bin/env bash
# The gate stage list, in one place. `make gate` runs it under the gate lock in this tree;
# `make merge` runs it in the preview-merge worktree. Stages call the Makefile targets so the
# commands are never duplicated. Cheapest first. One line per passing stage; on failure the last
# 30 lines of that stage's log plus the full-log path are printed and the run stops (exit code
# is the stage's real exit code, never parsed output).
#
# There is no docs-only mode: tests/test_docs reads README.md and every docs/*.md, so every PR
# runs every stage.
#
# Usage: scripts/gate.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
[ -f "$ROOT/Makefile" ] && [ -f "$ROOT/pyproject.toml" ] || {
  echo "gate.sh: $ROOT is not the repo root (Makefile/pyproject.toml missing)" >&2
  exit 2
}

LOG_DIR="${GATE_LOG_DIR:-${TMPDIR:-/tmp}/mathviz-gate-logs/$(date +%Y%m%d-%H%M%S)-$$}"
mkdir -p "$LOG_DIR"

run_stage() {
  local name="$1"
  shift
  local log="$LOG_DIR/$name.log"
  local start rc
  start=$(date +%s)
  if "$@" >"$log" 2>&1; then
    echo "gate: ✓ $name ($(( $(date +%s) - start ))s)"
    return 0
  else
    rc=$?
    echo "gate: ✗ $name (exit $rc) — last 30 lines:" >&2
    tail -n 30 "$log" >&2
    echo "gate: full log: $log" >&2
    return "$rc"
  fi
}

# scripts/check_gate_wiring.py parses this unindented list; each Blocking-gate target must be in it.
stages="doc-facts-check doc-refs-check gate-wiring-check lint test"
for stage in $stages; do
  run_stage "$stage" make -C "$ROOT" -s "$stage"
done
echo "gate: all stages passed (logs: $LOG_DIR)"
