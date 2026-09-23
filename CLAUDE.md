## Tasks

Most tasks in `tasks.md` have a **Suggested path:** section. It is one way to reach the
**Objective:**, not a specification to implement literally. Deviate from it when you have a good
reason. The **Objective:** defines success.

The tests listed under **Tests:** are the minimum. Add tests for edge cases you find during
implementation.

## Workflow

- One worktree per task: `make worktree b=<type>/<name>` creates `~/projects/mathviz/<type>/<name>`
  from `origin/main`, with its own `.venv` and the hooks installed.
- Local `main` is a read-only mirror of `origin/main`. The hooks in `.githooks/` refuse a commit
  on `main`, a push to `main`, and moving `main` to a commit that `origin/main` does not contain.
- Never run `git stash`. Every worktree of the repo shares one stash stack.
- `make merge pr=N` is the only merge path. It runs the gate on the preview merge of the PR into
  `origin/main` and merges only if neither side moved during the gate.
- `make gate` runs the same stages in the current worktree: `gate-wiring-check`, `lint`
  (`ruff check .`) and `test` (`pytest`).
- `make sync` fetches and fast-forwards the current branch.
- `make doctor` checks the pins, the venv, `docs/generators.md` and the hooks in about a second.
- `make branches-gc` reports branches and worktrees. It deletes only with `args=--delete`, and
  then only merged branches.
- Prose follows `docs/writing-style.md`.
