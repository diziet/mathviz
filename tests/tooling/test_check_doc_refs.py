"""scripts/check_doc_refs.py: inject each broken reference, see it fail by name."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

import check_doc_refs as refs
import pytest
from doc_refs_tree import path_candidate

if TYPE_CHECKING:
    from pathlib import Path

FILES = {
    ".gitignore": "runs/\n",
    "Makefile": "lint: ## Blocking gate: lint\n\ttrue\ngate: lint\n\ttrue\n",
    "pyproject.toml": (
        '[project]\nname = "mini"\nversion = "0"\n'
        '[project.scripts]\nmini = "mini.cli:main"\n'
    ),
    "mini/__init__.py": "",
    "mini/options.py": 'import typer\nCONFIG = typer.Option(..., "--config")\n',
    "mini/cli.py": (
        "import typer\nfrom mini.options import CONFIG\napp = typer.Typer()\n\n"
        "@app.command()\ndef run(config: str = CONFIG) -> None: ...\n\n"
        '@app.command("score-all")\n'
        'def score(run_dir: str = typer.Option(..., "--run-dir")) -> None: ...\n'
    ),
    "scripts/tool.py": (
        "import argparse\nparser = argparse.ArgumentParser()\n"
        'parser.add_argument("--static-only", action="store_true")\n'
    ),
    "scripts/gate.sh": '[ "${1:-}" = "--docs-only" ] && echo docs\n',
    "docs/guide.md": "See `../README.md` and `guide.md`.\n",
}
CLEAN_README = (
    "Run `make gate`, `scripts/tool.py --static-only`, `bash scripts/gate.sh "
    "--docs-only`, `mini run --config x.yaml` and `score-all --run-dir DIR`.\n"
    "Files: `mini/cli.py`, `cli.py:3`, `options.py`, `runs/<id>/out.json`.\n"
    "Outside: `uv sync --locked`, `origin/main`, `/api/results`.\n"
)


def _write(root: Path, relative: str, text: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _track(root: Path) -> None:
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)


@pytest.fixture
def repo(tmp_path: Path, isolated_git_env: None) -> Path:
    """A git repo with a Makefile, a Typer console script, scripts and clean docs."""
    root = tmp_path / "repo"
    for relative, text in FILES.items():
        _write(root, relative, text)
    _write(root, "README.md", CLEAN_README)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    _track(root)
    return root


def _check(root: Path, capsys: pytest.CaptureFixture[str]) -> tuple[int, str]:
    code = refs.main(["--root", str(root)])
    captured = capsys.readouterr()
    return code, captured.out + captured.err


def _with_readme(root: Path, extra: str) -> Path:
    _write(root, "README.md", CLEAN_README + extra)
    return root


def test_clean_repo_passes(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, output = _check(repo, capsys)
    assert code == 0, output
    assert "check_doc_refs: ok (2 docs" in output


def test_missing_path_fails_with_file_and_line(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, output = _check(_with_readme(repo, "\nGone: `scripts/removed.py`.\n"), capsys)
    assert code == 1
    assert "README.md:5: path `scripts/removed.py` does not exist" in output


def test_missing_bare_file_name_with_known_extension_fails(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, output = _check(_with_readme(repo, "`nowhere.toml`\n"), capsys)
    assert code == 1
    assert "path `nowhere.toml`" in output


def test_gitignored_missing_path_is_not_reported(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, output = _check(_with_readme(repo, "`runs/latest/summary.txt`\n"), capsys)
    assert code == 0, output


def test_struck_through_reference_is_not_checked(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, output = _check(_with_readme(repo, "~~`scripts/old.py`~~ replaced\n"), capsys)
    assert code == 0, output


def test_unknown_make_target_fails_by_name(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, output = _check(_with_readme(repo, "`make lint deploy p=x`\n"), capsys)
    assert code == 1
    assert "make-target `make deploy` is not a target in the Makefile" in output
    assert "make lint`" not in output


def test_make_span_for_another_makefile_is_not_checked(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, output = _check(
        _with_readme(repo, "`make -C sub deploy` `make -o x lint`\n"), capsys
    )
    assert code == 0, output


def test_flag_missing_from_the_console_script_fails(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, output = _check(_with_readme(repo, "`mini run --force`\n"), capsys)
    assert code == 1
    assert "flag `--force` is not defined by mini" in output


def test_unknown_subcommand_fails_by_name(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, output = _check(_with_readme(repo, "`mini report --run-dir x`\n"), capsys)
    assert code == 1
    assert "subcommand `mini report` is not a subcommand of mini" in output


def test_flag_of_one_script_is_not_accepted_for_another(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, output = _check(
        _with_readme(repo, "`python scripts/tool.py --run-dir x`\n"), capsys
    )
    assert code == 1
    assert "flag `--run-dir` is not defined by scripts/tool.py" in output


def test_missing_script_in_a_command_fails_as_a_path(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, output = _check(_with_readme(repo, "`python3 scripts/gone.py --x`\n"), capsys)
    assert code == 1
    assert "path `scripts/gone.py` does not exist" in output


def test_bare_flag_is_checked_against_every_repo_flag(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, output = _check(
        _with_readme(repo, "`--static-only` `--run-dir DIR` `--gone`\n"), capsys
    )
    assert code == 1
    assert "flag `--gone` is not defined by any command in the repo" in output
    assert "--static-only`" not in output


def test_exemption_silences_its_finding(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write(
        repo, refs.ALLOWLIST, "# note\nREADME.md | --gone | an external tool's flag\n"
    )
    code, output = _check(_with_readme(repo, "`--gone`\n"), capsys)
    assert code == 0, output
    assert "1 exempted" in output


def test_stale_exemption_fails_with_its_line(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write(repo, refs.ALLOWLIST, "\n* | scripts/old.py | removed in 2026\n")
    code, output = _check(repo, capsys)
    assert code == 1
    assert "docs/doc-refs-allow.txt:2: stale exemption `* | scripts/old.py`" in output


def test_exemption_without_a_reason_is_a_usage_error(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write(repo, refs.ALLOWLIST, "README.md | --gone |\n")
    code, output = _check(repo, capsys)
    assert code == 2
    assert "docs/doc-refs-allow.txt:1: expected" in output


def test_repo_without_docs_fails_closed(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    subprocess.run(
        ["git", "rm", "-q", "--cached", "README.md", "docs/guide.md"],
        cwd=repo,
        check=True,
    )
    code, output = _check(repo, capsys)
    assert code == 1
    assert "read 0 docs" in output


def test_non_git_directory_is_a_usage_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, output = _check(tmp_path, capsys)
    assert code == 2
    assert "is not a git checkout" in output


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("scripts/x.py", "scripts/x.py"),
        ("./tests/x.py::test_y", "tests/x.py"),
        ("core/db.py:59-61", "core/db.py"),
        ("runs/<run_id>/", "runs/*/"),
        ("uv.lock", "uv.lock"),
        ("summary.json", "summary.json"),
        ("origin/main", None),
        ("../feat/name", None),
        ("/api/results", None),
        ("https://example.com/a", None),
        ("a.b.c", None),
        ("*.md", None),
        ("10**(-d)/100", None),
        ("'{...}'", None),
    ],
)
def test_path_candidate_shapes(token: str, expected: str | None) -> None:
    assert path_candidate(token) == expected
