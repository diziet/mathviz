"""Tests that generate --dry-run resolves config like a real generate and runs no pipeline."""

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from mathviz.cli import app
from mathviz.core.generator import get_generator

runner = CliRunner()

SEED_AND_PARAMS_TOML = "seed = 11\n\n[params]\nmajor_radius = 2.5\n"
TORUS_DEFAULT_MINOR_RADIUS = 0.4


@pytest.fixture(autouse=True)
def isolated_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run each test in an empty directory so no stray mathviz.toml is discovered."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _write(path: Path, text: str) -> Path:
    """Write text to path and return the path."""
    path.write_text(text)
    return path


def _dry_run_json(*args: str) -> dict[str, Any]:
    """Run `mathviz generate torus --dry-run --json` with extra args; return the payload."""
    result = runner.invoke(app, ["generate", "torus", "--dry-run", "--json", *args])
    assert result.exit_code == 0, result.output
    return json.loads(result.output)


def _dry_run_error(*args: str) -> str:
    """Run a torus dry-run that must fail; return the JSON error message."""
    result = runner.invoke(app, ["generate", "torus", "--dry-run", "--json", *args])
    assert result.exit_code == 2, result.output
    return json.loads(result.output)["error"]


def _fail_if_called(*args: Any, **kwargs: Any) -> None:
    """Stand in for a pipeline call that --dry-run must not make."""
    raise AssertionError("--dry-run called the pipeline")


class TestDryRunConfigValues:
    """Dry-run reports the seed and parameters from every config layer."""

    def test_object_config_sets_seed_and_params(self, tmp_path: Path) -> None:
        """--config with seed = 11 and [params] major_radius = 2.5 appears in the payload."""
        config = _write(tmp_path / "torus.toml", SEED_AND_PARAMS_TOML)
        data = _dry_run_json("--config", str(config))
        assert data["seed"] == 11
        assert data["parameters"] == {
            "major_radius": 2.5,
            "minor_radius": TORUS_DEFAULT_MINOR_RADIUS,
        }

    def test_project_config_sets_seed_and_params(self, isolated_cwd: Path) -> None:
        """mathviz.toml in the working directory sets the dry-run seed and parameters."""
        _write(isolated_cwd / "mathviz.toml", SEED_AND_PARAMS_TOML)
        data = _dry_run_json()
        assert data["seed"] == 11
        assert data["parameters"]["major_radius"] == 2.5

    def test_cli_seed_and_param_override_config(self, tmp_path: Path) -> None:
        """--seed and --param override the same keys from --config."""
        config = _write(tmp_path / "torus.toml", SEED_AND_PARAMS_TOML)
        data = _dry_run_json(
            "--config", str(config), "--seed", "7", "--param", "major_radius=3.0"
        )
        assert data["seed"] == 7
        assert data["parameters"]["major_radius"] == 3.0

    def test_cli_param_merges_with_config_params(self, tmp_path: Path) -> None:
        """A --param key that the config does not set keeps the config's other params."""
        config = _write(tmp_path / "torus.toml", SEED_AND_PARAMS_TOML)
        data = _dry_run_json("--config", str(config), "--param", "minor_radius=0.3")
        assert data["parameters"] == {"major_radius": 2.5, "minor_radius": 0.3}

    def test_rich_output_shows_config_seed(self, tmp_path: Path) -> None:
        """The non-JSON dry-run output prints the seed from --config."""
        config = _write(tmp_path / "torus.toml", SEED_AND_PARAMS_TOML)
        result = runner.invoke(app, ["generate", "torus", "--dry-run", "--config", str(config)])
        assert result.exit_code == 0, result.output
        assert "Seed: 11" in result.output

    def test_seed_and_params_match_real_generate(self, tmp_path: Path) -> None:
        """Dry-run reports the seed and parameters that a real generate reports."""
        config = _write(tmp_path / "torus.toml", SEED_AND_PARAMS_TOML)
        args = ["--config", str(config), "--param", "minor_radius=0.3"]
        dry = _dry_run_json(*args)
        real = runner.invoke(app, ["generate", "torus", "--json", *args])
        assert real.exit_code in (0, 1), real.output
        real_data = json.loads(real.output)
        assert (dry["seed"], dry["parameters"]) == (real_data["seed"], real_data["parameters"])


class TestDryRunConfigErrors:
    """Dry-run stops with exit code 2 on the config errors that stop a real generate."""

    def test_missing_config_file_exits_2(self, tmp_path: Path) -> None:
        """A --config path that does not exist stops the dry-run."""
        missing = tmp_path / "missing.toml"
        assert _dry_run_error("--config", str(missing)) == f"Config file not found: {missing}"

    def test_missing_profile_exits_2(self) -> None:
        """An unknown --profile name stops the dry-run."""
        assert _dry_run_error("--profile", "no_such_profile").startswith(
            "Sampling profile not found:"
        )

    def test_invalid_representation_config_exits_2(self, tmp_path: Path) -> None:
        """An invalid [representation] section in --config stops the dry-run."""
        config = _write(tmp_path / "bad.toml", '[representation]\ntype = "tube"\ntube_sides = 0\n')
        assert _dry_run_error("--config", str(config)).startswith(
            "Invalid representation config:"
        )

    def test_invalid_container_flag_exits_2(self) -> None:
        """--width 0 fails Container validation and stops the dry-run."""
        assert "width_mm" in _dry_run_error("--width", "0")

    def test_unknown_cli_param_exits_2(self) -> None:
        """A --param key that torus does not define stops the dry-run and names the key."""
        assert _dry_run_error("--param", "count=10").startswith(
            "Unknown parameter(s) 'count' for torus."
        )

    def test_unknown_config_param_exits_2(self, tmp_path: Path) -> None:
        """A [params] key in --config that torus does not define stops the dry-run."""
        config = _write(tmp_path / "bad.toml", "[params]\nradius = 2.0\n")
        assert "'radius'" in _dry_run_error("--config", str(config))


class TestDryRunRunsNoPipeline:
    """Dry-run resolves config without generating geometry or writing files."""

    def test_config_dry_run_does_not_generate_or_export(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A dry-run with --config and --output never calls run() or generate()."""
        monkeypatch.setattr("mathviz.cli.run", _fail_if_called)
        monkeypatch.setattr(get_generator("torus"), "generate", _fail_if_called)
        config = _write(tmp_path / "torus.toml", SEED_AND_PARAMS_TOML)
        output = tmp_path / "torus.ply"
        data = _dry_run_json("--config", str(config), "--output", str(output))
        assert data["output"] == str(output)
        assert "export" in data["stages"]
        assert not output.exists()
