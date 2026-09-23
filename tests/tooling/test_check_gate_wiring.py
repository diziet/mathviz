"""scripts/check_gate_wiring.py: inject each defect, see it fail by name, fix, pass."""

from __future__ import annotations

import sys
from pathlib import Path

import check_gate_wiring as wiring
import pytest

KNOWN = frozenset({"tests/test_alpha.py"})
OPERATORS = frozenset({"manual.py"})
MAKEFILE = (
    "lint: ## Blocking gate: ruff\n\truff check .\n"
    "gate: ## Blocking gate: everything\n\tbash scripts/gate.sh\n"
    "helper: ## Advisory: uses tool\n\tpython scripts/tool.py\n"
)


@pytest.fixture
def mini_repo(tmp_path: Path) -> Path:
    """A repo shaped like ours: Makefile, scripts, one test."""
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname = 'mini'\nversion = '0'\n"
    )
    (tmp_path / "Makefile").write_text(MAKEFILE)
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "gate.sh").write_text("#!/bin/sh\nfor s in lint; do make -s $s; done\n")
    (scripts / "tool.py").write_text("print('tool')\n")
    (scripts / "manual.py").write_text("print('run by hand')\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_alpha.py").write_text(
        "def test_alpha() -> None:\n    assert True\n"
    )
    return tmp_path


def _problems(root: Path) -> list[str]:
    return wiring.run_all(root, sys.executable, KNOWN, OPERATORS)


def _script_problems(root: Path) -> list[str]:
    return wiring.check_scripts_referenced(root, OPERATORS)


def test_clean_fixture_has_no_problems(mini_repo: Path) -> None:
    assert _problems(mini_repo) == []


def test_unwired_test_file_fails_by_name_then_passes_when_renamed(
    mini_repo: Path,
) -> None:
    stray = mini_repo / "tests" / "helpers_tests.py"
    stray.write_text("def test_stray() -> None:\n    assert True\n")
    problems = wiring.check_tests_collected(mini_repo, sys.executable, KNOWN)
    assert any("tests/helpers_tests.py" in p for p in problems)
    stray.rename(mini_repo / "tests" / "test_beta.py")
    assert wiring.check_tests_collected(mini_repo, sys.executable, KNOWN) == []


def test_test_module_imported_by_a_collected_test_counts_as_wired(
    mini_repo: Path,
) -> None:
    (mini_repo / "tests" / "shared_cases.py").write_text(
        "class SharedCases:\n    def test_shared(self) -> None:\n        assert True\n"
    )
    (mini_repo / "tests" / "test_alpha.py").write_text(
        "from tests.shared_cases import SharedCases\n\n"
        "class TestAlpha(SharedCases):\n    pass\n"
    )
    assert wiring.check_tests_collected(mini_repo, sys.executable, KNOWN) == []


def test_missing_known_module_fails(mini_repo: Path) -> None:
    problems = wiring.check_tests_collected(
        mini_repo, sys.executable, frozenset({"tests/test_missing.py"})
    )
    assert problems == [
        "known test module missing from collection: tests/test_missing.py"
    ]


def test_collection_error_is_reported(mini_repo: Path) -> None:
    (mini_repo / "tests" / "test_broken.py").write_text("def broken(:\n")
    problems = wiring.check_tests_collected(mini_repo, sys.executable, KNOWN)
    assert problems and "collection failed" in problems[0]


def test_orphan_script_fails_by_name_then_passes_when_referenced(
    mini_repo: Path,
) -> None:
    (mini_repo / "scripts" / "orphan.py").write_text("print('x')\n")
    problems = _script_problems(mini_repo)
    assert problems == [
        "orphan script (not referenced by Makefile, hooks, scripts or tests): "
        "scripts/orphan.py"
    ]
    (mini_repo / "Makefile").write_text(
        MAKEFILE + "orphan: ## Advisory\n\tpython scripts/orphan.py\n"
    )
    assert _script_problems(mini_repo) == []


def test_script_referenced_only_by_python_import_is_wired(mini_repo: Path) -> None:
    (mini_repo / "scripts" / "helpers_mod.py").write_text("X = 1\n")
    (mini_repo / "scripts" / "tool.py").write_text(
        "from helpers_mod import X\nprint(X)\n"
    )
    assert _script_problems(mini_repo) == []


def test_script_referenced_by_a_test_module_is_wired(mini_repo: Path) -> None:
    (mini_repo / "scripts" / "export_tool.py").write_text("print('x')\n")
    assert len(_script_problems(mini_repo)) == 1
    (mini_repo / "tests" / "test_alpha.py").write_text(
        "SCRIPT = 'scripts/export_tool.py'\n\n"
        "def test_alpha() -> None:\n    assert SCRIPT\n"
    )
    assert _script_problems(mini_repo) == []


def test_operator_exemption_fails_when_the_script_is_removed(mini_repo: Path) -> None:
    (mini_repo / "scripts" / "manual.py").unlink()
    assert _script_problems(mini_repo) == [
        "stale operator exemption: scripts/manual.py does not exist"
    ]


def test_operator_exemption_fails_when_the_script_becomes_referenced(
    mini_repo: Path,
) -> None:
    (mini_repo / "Makefile").write_text(
        MAKEFILE + "manual: ## Advisory\n\tpython scripts/manual.py\n"
    )
    problems = _script_problems(mini_repo)
    assert len(problems) == 1
    assert problems[0].startswith("stale operator exemption: scripts/manual.py is now")


def test_unlisted_operator_script_is_an_orphan(mini_repo: Path) -> None:
    problems = wiring.check_scripts_referenced(mini_repo, frozenset())
    assert problems == [
        "orphan script (not referenced by Makefile, hooks, scripts or tests): "
        "scripts/manual.py"
    ]


def test_reference_inside_a_comment_does_not_count(mini_repo: Path) -> None:
    (mini_repo / "scripts" / "ghost.py").write_text("print('x')\n")
    (mini_repo / "Makefile").write_text(MAKEFILE + "# see scripts/ghost.py\n")
    assert len(_script_problems(mini_repo)) == 1


def test_unwired_blocking_target_fails_then_passes_when_added_to_gate(
    mini_repo: Path,
) -> None:
    (mini_repo / "Makefile").write_text(
        MAKEFILE + "typecheck: ## Blocking gate: mypy\n\tmypy .\n"
    )
    problems = wiring.check_blocking_targets_wired(mini_repo)
    assert len(problems) == 1 and "'typecheck'" in problems[0]
    (mini_repo / "scripts" / "gate.sh").write_text(
        "#!/bin/sh\nfor s in lint typecheck; do make -s $s; done\n"
    )
    assert wiring.check_blocking_targets_wired(mini_repo) == []


def test_blocking_targets_are_parsed_from_help_comments() -> None:
    assert wiring.blocking_targets(MAKEFILE) == ["lint", "gate"]


def test_cli_exit_codes(
    mini_repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(wiring, "KNOWN_TEST_MODULES", KNOWN)
    monkeypatch.setattr(wiring, "OPERATOR_SCRIPTS", OPERATORS)
    assert wiring.main(["--root", str(mini_repo)]) == 0
    assert "check_gate_wiring: ok" in capsys.readouterr().out
    (mini_repo / "scripts" / "orphan.py").write_text("")
    assert wiring.main(["--root", str(mini_repo)]) == 1
    assert "FAIL orphan script" in capsys.readouterr().err
    assert wiring.main(["--root", str(mini_repo / "nowhere")]) == 2


def test_real_repo_blocking_targets_are_all_wired() -> None:
    root = Path(__file__).resolve().parents[2]
    assert wiring.check_blocking_targets_wired(root) == []
    assert wiring.check_scripts_referenced(root) == []


@pytest.mark.parametrize("collected", [set(), {"tests/test_alpha.py"}])
def test_existing_collection_can_be_audited_without_spawning_pytest(
    mini_repo: Path, monkeypatch: pytest.MonkeyPatch, collected: set[str]
) -> None:
    """Reuse collection without weakening the named minimum or orphan checks."""
    expected = wiring.check_tests_collected(mini_repo, sys.executable, KNOWN)
    if not collected:
        expected = [
            "test module not collected by pytest (nor imported by a collected one): "
            "tests/test_alpha.py",
            "known test module missing from collection: tests/test_alpha.py",
        ]

    def unexpected(*_args: object) -> None:
        pytest.fail("collection must not be repeated")

    monkeypatch.setattr(wiring, "collected_test_files", unexpected)
    assert wiring.check_collected_tests(mini_repo, collected, KNOWN) == expected

