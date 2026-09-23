# mathviz task runner. `make` (or `make help`) lists every target.
#
# Every project command lives here. Pins are the variables below; `make doctor` checks them and
# `make install` provisions them, both reading the same values. Package floors live in
# pyproject.toml.
#
# TODO: the repo has no lock file, so `make venv` installs the newest release each pyproject.toml
# floor allows. Adding uv.lock is its own reviewed change.
#
# `ruff format` is not run, as a gate or otherwise: it would rewrite 251 of 334 Python files
# (measured on 25d2f66, 2026-09-23). Adopting it is its own reviewed change.
.DEFAULT_GOAL := help
.PHONY: help install venv doctor hooks-install lint test gate gate-wiring-check worktree sync merge \
        branches-gc doc-refs-check doc-facts doc-facts-check

# ---- Pins (the single source; doctor and install both read these) -----------------------------
PYTHON_VERSION := 3.11
RUFF_VERSION   := 0.16.8
UV             := uv
VENV           := .venv
# The render extra adds pyvista; without it tests/test_generators/test_lsystem.py skips a test.
EXTRAS         := dev,render

# ---- Derived paths ----------------------------------------------------------------------------
BIN      := $(VENV)/bin
PY       := $(BIN)/python
# Machine-wide gate lock: heavy targets queue behind each other instead of competing for cores.
LOCKED   := $(PY) scripts/gate_lock.py --
# `make branches-gc args="--delete"`.
args ?=

help: ## Advisory: list targets, parsed from the double-hash comment on each rule
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | sort \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ---- Setup ------------------------------------------------------------------------------------
install: ## Sanctioned path for one-time machine setup: uv-managed Python, .venv, hooks; ends with doctor
	$(UV) python install $(PYTHON_VERSION)
	$(MAKE) venv hooks-install
	$(MAKE) doctor

venv: ## Sanctioned path: create this worktree's .venv and install the package (dev, render) and the ruff pin
	$(UV) venv --allow-existing --python $(PYTHON_VERSION) --no-python-downloads $(VENV)
	$(UV) pip install --python $(PY) -e ".[$(EXTRAS)]" "ruff==$(RUFF_VERSION)"

doctor: ## Advisory preflight (seconds, no build, no side effects): tool pins, venv, generated docs, hooks
	@PYTHON_VERSION=$(PYTHON_VERSION) RUFF_VERSION=$(RUFF_VERSION) UV=$(UV) VENV=$(VENV) \
	  bash scripts/doctor.sh

hooks-install: ## Sanctioned path for pointing core.hooksPath at .githooks in this repo
	git config core.hooksPath .githooks
	@echo "hooks: core.hooksPath=.githooks"

# ---- Gates (cheapest first) -------------------------------------------------------------------
lint: ## Blocking gate: ruff check
	$(BIN)/ruff check .

test: ## Blocking gate: pytest with the pyproject.toml defaults (xdist, not slow), under the gate lock
	$(LOCKED) $(BIN)/pytest

gate: ## Blocking gate: doc-facts-check, doc-refs-check, gate-wiring-check, lint, test under the gate lock; make merge runs the same stages
	$(LOCKED) bash scripts/gate.sh

gate-wiring-check: ## Blocking gate: every test collected, no orphan script, every Blocking-gate target run by gate
	$(PY) scripts/check_gate_wiring.py

doc-refs-check: ## Blocking gate, fails closed: paths, make targets and --flags in tracked .md code spans must resolve; stale exemptions in docs/doc-refs-allow.txt fail
	$(PY) scripts/check_doc_refs.py

doc-facts: ## Sanctioned path: regenerate <!-- fact:NAME --> values in tracked .md files from scripts/doc_facts_registry.py
	$(PY) scripts/doc_facts.py --write

doc-facts-check: ## Blocking gate: print the diff and fail when a doc fact is stale; rewrites the value first, so the re-run needs only a re-stage
	$(PY) scripts/doc_facts.py --fix-stale

# ---- Workflow ---------------------------------------------------------------------------------
worktree: ## Sanctioned path for starting work: make worktree b=feat/name (sibling tree, own venv, hooks)
	@bash scripts/worktree.sh "$(b)"

sync: ## Sanctioned path for updating the current branch: fetch + fast-forward; refuses a divergent local main
	@bash scripts/sync.sh

# TODO: build a report-only local watcher that re-runs the gate on each new origin/main commit
# and notifies on failure and recovery. It would catch two PRs that each pass alone and together
# merge into a failing main.
merge: ## Sanctioned path for merging a PR: make merge pr=N [keep=1] [dry_run=1]; the only merge path
	$(PY) scripts/merge.py --pr "$(pr)" $(if $(keep),--keep,) $(if $(dry_run),--dry-run,)

branches-gc: ## Advisory: report local branches/worktrees as merged, superseded, open-PR or checked-out; args="--delete" deletes only merged branches
	$(PY) scripts/branches_gc.py $(args)
