#!/usr/bin/env bash
# Preflight (seconds, no build, no side effects). Reads the pins from the environment the
# Makefile passes (PYTHON_VERSION, RUFF_VERSION, UV, VENV). Cheapest checks first. Every failure
# prints the fixing command. Fails CLOSED on any drift (exit 1). Probes never trigger an install.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
[ -f "$ROOT/pyproject.toml" ] || { echo "doctor: $ROOT is not the repo root" >&2; exit 2; }
cd "$ROOT"

PYTHON_VERSION="${PYTHON_VERSION:?set by the Makefile}"
RUFF_VERSION="${RUFF_VERSION:?set by the Makefile}"
UV="${UV:-uv}"
VENV="${VENV:-.venv}"
failures=0

ok()   { echo "doctor: ✓ $1"; }
fail() { echo "doctor: ✗ $1" >&2; echo "         fix: $2" >&2; failures=$((failures + 1)); }

# 1. Tools on PATH: uv provisions the venv; git and gh are used by the merge path.
if command -v "$UV" >/dev/null 2>&1; then
  ok "uv $("$UV" --version | awk '{print $2}') at $(command -v "$UV")"
else
  fail "uv not found on PATH" "brew install uv  (or: curl -LsSf https://astral.sh/uv/install.sh | sh)"
fi
for tool in git gh; do
  if command -v "$tool" >/dev/null 2>&1; then
    ok "$tool at $(command -v "$tool")"
  else
    fail "$tool not found on PATH" "brew install $tool"
  fi
done

# 2. The venv's python is the pinned minor version.
if [ -x "$VENV/bin/python" ]; then
  actual="$("$VENV/bin/python" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
  if [ "$actual" = "$PYTHON_VERSION" ]; then
    ok "$VENV/bin/python is $("$VENV/bin/python" --version | awk '{print $2}')"
  else
    fail "$VENV/bin/python is $actual, pin is $PYTHON_VERSION" "rm -rf $VENV && make venv"
  fi
else
  fail "$VENV/bin/python missing" "make venv"
fi

# 3. The venv's ruff is the pinned version. The version alone is not enough: the path must be
# this worktree's venv, not a ruff found elsewhere on PATH.
if [ -x "$VENV/bin/ruff" ]; then
  actual="$("$VENV/bin/ruff" --version | awk '{print $2}')"
  if [ "$actual" = "$RUFF_VERSION" ]; then
    ok "$VENV/bin/ruff is $actual"
  else
    fail "$VENV/bin/ruff is $actual, pin is $RUFF_VERSION" "make venv"
  fi
else
  fail "$VENV/bin/ruff missing" "make venv"
fi

# 4. The editable install points at this worktree. Reads package metadata only; does not
# import mathviz.
if [ -x "$VENV/bin/python" ]; then
  origin="$("$VENV/bin/python" - <<'PY' 2>/dev/null || true
import json
from importlib.metadata import PackageNotFoundError, distribution
try:
    text = distribution("mathviz").read_text("direct_url.json") or "{}"
except PackageNotFoundError:
    text = "{}"
print(json.loads(text).get("url", ""))
PY
)"
  if [ "$origin" = "file://$ROOT" ]; then
    ok "mathviz is installed editable from this worktree"
  else
    fail "mathviz install origin is '${origin:-missing}', expected file://$ROOT" "make venv"
  fi
fi

# 5. The generated docs/generators.md matches the registry. --check never writes.
if [ -x "$VENV/bin/python" ]; then
  if "$VENV/bin/python" scripts/generate_generator_docs.py --check >/dev/null 2>&1; then
    ok "docs/generators.md is up to date"
  else
    fail "docs/generators.md is stale or cannot be generated" \
      "$VENV/bin/python scripts/generate_generator_docs.py"
  fi
fi

# 6. Hooks installed.
hooks_path="$(git config core.hooksPath || true)"
if [ "$hooks_path" = ".githooks" ]; then
  ok "core.hooksPath=.githooks"
else
  fail "core.hooksPath is '${hooks_path:-unset}', expected .githooks" "make hooks-install"
fi

if [ "$failures" -gt 0 ]; then
  echo "doctor: $failures problem(s); fix commands above" >&2
  exit 1
fi
echo "doctor: healthy"
