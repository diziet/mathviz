"""Checks that check themselves: assert the repo's gates are all wired.

(a) every test module under tests/ is collected by pytest, and a named minimum
    set of known modules is present so an empty glob cannot pass;
(b) every script under scripts/ is referenced from the Makefile, a hook, another
    script, or a test module (no orphan tooling), except the listed operator
    scripts, and each of those must still exist and still be unreferenced;
(c) the Makefile `gate` recipe runs scripts/gate.sh, and every Makefile target
    whose `## ` comment says `Blocking gate` is in that script's `stages="..."`
    list. The list is parsed rather than searched as text, because a target name
    inside a message string is not a stage.
The verdict is the exit code, never parsed output.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_DEF_PATTERN = re.compile(r"^\s*(async\s+)?def\s+test_", re.MULTILINE)
BLOCKING_TARGET_PATTERN = re.compile(
    r"^([A-Za-z0-9_-]+):.*## Blocking gate", re.MULTILINE
)
# Only an unindented list matches, not an indented subset inside a branch.
STAGES_PATTERN = re.compile(r'^stages="([^"]*)"', re.MULTILINE)
GATE_SCRIPT = "scripts/gate.sh"
KNOWN_TEST_MODULES: frozenset[str] = frozenset(
    {
        "tests/test_build_demo.py",
        "tests/test_cli/test_render_all.py",
        "tests/test_core/test_pipeline.py",
        "tests/test_docs/test_generate_generator_docs.py",
        "tests/test_docs/test_generator_docs.py",
        "tests/test_fixtures.py",
        "tests/test_generators/test_attractors.py",
        "tests/test_pipeline/test_export_routing.py",
        "tests/test_preview/test_build_banner.py",
        "tests/test_shared/test_tube_thickening.py",
        "tests/tooling/test_check_gate_wiring.py",
        "tests/tooling/test_hooks.py",
    }
)
# Scripts an operator runs by hand. No target, hook, script or test calls them.
OPERATOR_SCRIPTS: frozenset[str] = frozenset({"generate_thumbnails.py"})
# Targets that are the roots of the wiring and therefore need no caller.
WIRING_ROOTS = frozenset({"gate"})


def collected_test_files(root: Path, python: str) -> set[str]:
    """Return the repo-relative test files pytest collects; raise on collect error."""
    result = subprocess.run(
        [
            python,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-o",
            "addopts=",
            "-p",
            "no:cacheprovider",
            "tests",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode not in (0, 5):
        raise RuntimeError(f"pytest collection failed:\n{result.stdout[-2000:]}")
    return {
        line.split("::", 1)[0] for line in result.stdout.splitlines() if "::" in line
    }


def test_files_on_disk(root: Path) -> set[str]:
    """Return every tests/**/*.py (except conftest) that defines a test function."""
    found: set[str] = set()
    for path in sorted((root / "tests").rglob("*.py")):
        if path.name == "conftest.py" or "__pycache__" in path.parts:
            continue
        if TEST_DEF_PATTERN.search(path.read_text(errors="replace")):
            found.add(path.relative_to(root).as_posix())
    return found


def _is_imported_by_tests(root: Path, module_file: str, collected: set[str]) -> bool:
    """True when a collected test module imports `module_file` (a shared mixin)."""
    dotted = module_file.removesuffix(".py").replace("/", ".")
    return any(
        dotted in (root / test).read_text(errors="replace") for test in collected
    )


def check_tests_collected(
    root: Path, python: str, known: frozenset[str] = KNOWN_TEST_MODULES
) -> list[str]:
    """(a) Every on-disk test module is collected and the known minimum set present."""
    try:
        collected = collected_test_files(root, python)
    except RuntimeError as error:
        return [str(error)]
    return check_collected_tests(root, collected, known)


def check_collected_tests(
    root: Path, collected: set[str], known: frozenset[str] = KNOWN_TEST_MODULES
) -> list[str]:
    """Audit an existing collection; fail closed on missing modules or an empty set."""
    problems = [
        f"test module not collected by pytest (nor imported by a collected one): {f}"
        for f in sorted(test_files_on_disk(root) - collected)
        if not _is_imported_by_tests(root, f, collected)
    ]
    problems += [
        f"known test module missing from collection: {f}"
        for f in sorted(known - collected)
    ]
    return problems


def _reference_corpus(root: Path) -> dict[Path, str]:
    """Return {file: text} for Makefile, hooks, scripts and tests, comment lines stripped."""
    files = [
        root / "Makefile",
        *sorted((root / ".githooks").glob("*")),
        *sorted((root / "scripts").glob("*")),
        *sorted((root / "tests").rglob("*.py")),
    ]
    corpus: dict[Path, str] = {}
    for path in files:
        # This file names the operator scripts in OPERATOR_SCRIPTS; that is not a use.
        if not path.is_file() or path.resolve() == Path(__file__).resolve():
            continue
        lines = path.read_text(errors="replace").splitlines()
        corpus[path] = "\n".join(
            line for line in lines if not line.lstrip().startswith("#")
        )
    return corpus


def check_scripts_referenced(
    root: Path, operator_scripts: frozenset[str] = OPERATOR_SCRIPTS
) -> list[str]:
    """(b) Every scripts/* file is referenced, or is a still-valid operator exemption."""
    corpus = _reference_corpus(root)
    problems: list[str] = []
    for script in sorted((root / "scripts").glob("*")):
        if not script.is_file() or script.name.startswith("__"):
            continue
        import_pattern = re.compile(rf"\b(from|import)\s+{re.escape(script.stem)}\b")
        referenced = any(
            script.name in text or bool(import_pattern.search(text))
            for path, text in corpus.items()
            if path != script
        )
        if script.name in operator_scripts:
            if referenced:
                problems.append(
                    f"stale operator exemption: scripts/{script.name} is now referenced; "
                    "remove it from OPERATOR_SCRIPTS"
                )
        elif not referenced:
            problems.append(
                "orphan script (not referenced by Makefile, hooks, scripts or tests): "
                f"scripts/{script.name}"
            )
    problems += [
        f"stale operator exemption: scripts/{name} does not exist"
        for name in sorted(operator_scripts)
        if not (root / "scripts" / name).is_file()
    ]
    return problems


def blocking_targets(makefile_text: str) -> list[str]:
    """Return Makefile targets whose help comment says `Blocking gate`."""
    return BLOCKING_TARGET_PATTERN.findall(makefile_text)


def recipe_text(makefile_text: str, target: str) -> str:
    """Return the tab-indented recipe lines of `target`, or '' when it has no rule."""
    lines = makefile_text.splitlines()
    for index, line in enumerate(lines):
        if re.match(rf"^{re.escape(target)}:", line):
            recipe: list[str] = []
            for body in lines[index + 1 :]:
                if not body.startswith("\t"):
                    break
                recipe.append(body)
            return "\n".join(recipe)
    return ""


def gate_stages(gate_text: str) -> list[str] | None:
    """Return the stage list from gate.sh, or None without a `stages=` line."""
    match = STAGES_PATTERN.search(gate_text)
    return match.group(1).split() if match else None


def check_blocking_targets_wired(root: Path) -> list[str]:
    """(c) `gate` runs gate.sh and every Blocking-gate target is in its stage list."""
    makefile = root / "Makefile"
    if not makefile.is_file():
        return ["Makefile missing"]
    makefile_text = makefile.read_text()
    if GATE_SCRIPT not in recipe_text(makefile_text, "gate"):
        return [f"Makefile `gate` recipe does not run {GATE_SCRIPT}"]
    gate = root / GATE_SCRIPT
    stages = gate_stages(gate.read_text()) if gate.is_file() else None
    if stages is None:
        return [f'{GATE_SCRIPT} has no stages="..." list']
    return [
        f"blocking target '{target}' is not a stage in {GATE_SCRIPT}"
        for target in blocking_targets(makefile_text)
        if target not in WIRING_ROOTS and target not in stages
    ]


def run_all(
    root: Path, python: str, known: frozenset[str], operator_scripts: frozenset[str]
) -> list[str]:
    """Run every check and return the combined list of problems."""
    return [
        *check_tests_collected(root, python, known),
        *check_scripts_referenced(root, operator_scripts),
        *check_blocking_targets_wired(root),
    ]


def main(argv: list[str] | None = None) -> int:
    """CLI: exit 1 on any problem, 2 when not run from the repo root."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT, help=argparse.SUPPRESS)
    options = parser.parse_args(argv)
    root: Path = options.root.resolve()
    if not (root / "pyproject.toml").is_file():
        print(f"check_gate_wiring: {root} is not the repo root", file=sys.stderr)
        return 2
    problems = run_all(root, sys.executable, KNOWN_TEST_MODULES, OPERATOR_SCRIPTS)
    for problem in problems:
        print(f"check_gate_wiring: FAIL {problem}", file=sys.stderr)
    if problems:
        return 1
    print("check_gate_wiring: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
