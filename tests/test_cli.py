"""Tests for the MathViz CLI."""

import json
from pathlib import Path

from typer.testing import CliRunner

from mathviz.cli import app

runner = CliRunner()


class TestGenerateJson:
    """Test generate command with --json output."""

    def test_generate_torus_json_valid(self) -> None:
        """mathviz generate torus --json outputs valid JSON with timing and metadata."""
        result = runner.invoke(app, ["generate", "torus", "--json"])
        assert result.exit_code == 0 or result.exit_code == 1
        data = json.loads(result.output)
        assert data["generator"] == "torus"
        assert "timings" in data
        assert "parameters" in data
        assert "seed" in data
        assert "validation" in data
        assert isinstance(data["timings"], dict)
        assert len(data["timings"]) > 0

    def test_generate_torus_json_has_mesh_info(self) -> None:
        """JSON output includes mesh vertex and face counts."""
        result = runner.invoke(app, ["generate", "torus", "--json"])
        data = json.loads(result.output)
        assert data["mesh_vertices"] is not None
        assert data["mesh_vertices"] > 0
        assert data["mesh_faces"] is not None
        assert data["mesh_faces"] > 0


class TestGenerateDryRun:
    """Test generate command with --dry-run."""

    def test_dry_run_writes_no_files(self, tmp_path: Path) -> None:
        """--dry-run writes no files but prints expected output info."""
        output_file = tmp_path / "should_not_exist.ply"
        result = runner.invoke(
            app,
            ["generate", "torus", "--dry-run", "--output", str(output_file)],
        )
        assert result.exit_code == 0
        assert not output_file.exists()

    def test_dry_run_json_output(self) -> None:
        """--dry-run --json produces valid JSON with dry_run flag."""
        result = runner.invoke(app, ["generate", "torus", "--dry-run", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["dry_run"] is True
        assert data["generator"] == "torus"
        assert "parameters" in data
        assert "stages" in data

    def test_dry_run_shows_parameters(self) -> None:
        """--dry-run shows the resolved parameters."""
        result = runner.invoke(
            app,
            ["generate", "torus", "--dry-run", "--json", "--param", "major_radius=3.0"],
        )
        data = json.loads(result.output)
        assert data["parameters"]["major_radius"] == 3.0

    def test_dry_run_includes_sample_stage(self) -> None:
        """--dry-run stages list includes the sample stage."""
        result = runner.invoke(app, ["generate", "torus", "--dry-run", "--json"])
        data = json.loads(result.output)
        assert "sample" in data["stages"]


class TestGenerateReport:
    """Test generate command with --report."""

    def test_report_writes_valid_json(self, tmp_path: Path) -> None:
        """--report path.json writes valid JSON report."""
        report_path = tmp_path / "report.json"
        result = runner.invoke(
            app,
            ["generate", "torus", "--report", str(report_path)],
        )
        assert result.exit_code == 0 or result.exit_code == 1
        assert report_path.exists()
        data = json.loads(report_path.read_text())
        assert data["generator"] == "torus"
        assert "timings" in data
        assert "validation" in data

    def test_report_and_json_both_work(self, tmp_path: Path) -> None:
        """--report and --json can be used together."""
        report_path = tmp_path / "report.json"
        result = runner.invoke(
            app,
            ["generate", "torus", "--report", str(report_path), "--json"],
        )
        assert result.exit_code == 0 or result.exit_code == 1
        stdout_data = json.loads(result.output)
        report_data = json.loads(report_path.read_text())
        assert stdout_data["generator"] == report_data["generator"]


class TestListCommand:
    """Test list command."""

    def test_list_json_valid_array(self) -> None:
        """mathviz list --json produces valid JSON array of generator info."""
        result = runner.invoke(app, ["list", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)
        assert len(data) > 0
        entry = data[0]
        assert "name" in entry
        assert "category" in entry
        assert "description" in entry
        assert "aliases" in entry

    def test_list_includes_torus(self) -> None:
        """List output includes the torus generator."""
        result = runner.invoke(app, ["list", "--json"])
        data = json.loads(result.output)
        names = [g["name"] for g in data]
        assert "torus" in names

    def test_list_rich_output(self) -> None:
        """List without --json produces table output."""
        result = runner.invoke(app, ["list"])
        assert result.exit_code == 0
        assert "torus" in result.output


class TestInfoCommand:
    """Test info command."""

    def test_info_torus_json_valid(self) -> None:
        """mathviz info torus --json produces valid JSON param schema."""
        result = runner.invoke(app, ["info", "torus", "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["name"] == "torus"
        assert data["category"] == "parametric"
        assert "default_params" in data
        assert "major_radius" in data["default_params"]
        assert "minor_radius" in data["default_params"]

    def test_info_unknown_generator(self) -> None:
        """Info for unknown generator exits with code 2."""
        result = runner.invoke(app, ["info", "nonexistent", "--json"])
        assert result.exit_code == 2
        data = json.loads(result.output)
        assert "error" in data


class TestUnknownGenerator:
    """Test error handling for unknown generators."""

    def test_generate_unknown_exits_2(self) -> None:
        """Unknown generator name exits with code 2 and a clear error."""
        result = runner.invoke(app, ["generate", "nonexistent_gen", "--json"])
        assert result.exit_code == 2
        data = json.loads(result.output)
        assert "error" in data
        assert "nonexistent_gen" in data["error"]

    def test_generate_unknown_rich_output(self) -> None:
        """Unknown generator with rich output shows error."""
        result = runner.invoke(app, ["generate", "nonexistent_gen"])
        assert result.exit_code == 2
        assert "nonexistent_gen" in result.output


class TestParamParsing:
    """Test --param key=value parsing."""

    def test_multiple_params_parsed(self) -> None:
        """--param R=2.0 --param r=0.5 parses multiple key=value params."""
        result = runner.invoke(
            app,
            [
                "generate", "torus", "--json",
                "--param", "major_radius=2.0",
                "--param", "minor_radius=0.5",
            ],
        )
        assert result.exit_code == 0 or result.exit_code == 1
        data = json.loads(result.output)
        assert data["parameters"]["major_radius"] == 2.0
        assert data["parameters"]["minor_radius"] == 0.5

    def test_param_integer_coercion(self) -> None:
        """Integer values are coerced from strings."""
        result = runner.invoke(
            app,
            ["generate", "torus", "--dry-run", "--json", "--param", "count=10"],
        )
        data = json.loads(result.output)
        assert data["parameters"]["count"] == 10
        assert isinstance(data["parameters"]["count"], int)

    def test_param_float_coercion(self) -> None:
        """Float values are coerced from strings."""
        result = runner.invoke(
            app,
            ["generate", "torus", "--dry-run", "--json", "--param", "scale=1.5"],
        )
        data = json.loads(result.output)
        assert data["parameters"]["scale"] == 1.5

    def test_param_string_passthrough(self) -> None:
        """Non-numeric values stay as strings."""
        result = runner.invoke(
            app,
            ["generate", "torus", "--dry-run", "--json", "--param", "mode=fast"],
        )
        data = json.loads(result.output)
        assert data["parameters"]["mode"] == "fast"

    def test_invalid_param_format(self) -> None:
        """Invalid param format (no =) exits with error."""
        result = runner.invoke(app, ["generate", "torus", "--param", "bad_param"])
        assert result.exit_code == 2


class TestValidateCommand:
    """Test validate command."""

    def test_validate_torus_json(self) -> None:
        """Validate command outputs JSON with validation results."""
        result = runner.invoke(app, ["validate", "torus", "--json"])
        assert result.exit_code in (0, 1)
        data = json.loads(result.output)
        assert "passed" in data
        assert "checks" in data
        assert isinstance(data["checks"], list)

    def test_validate_unknown_generator(self) -> None:
        """Validate unknown generator exits with code 2."""
        result = runner.invoke(app, ["validate", "nonexistent", "--json"])
        assert result.exit_code == 2


class TestExitCodes:
    """Test exit code semantics."""

    def test_success_exit_code(self) -> None:
        """Successful dry-run returns exit code 0."""
        result = runner.invoke(app, ["generate", "torus", "--dry-run"])
        assert result.exit_code == 0

    def test_error_exit_code_unknown_generator(self) -> None:
        """Unknown generator returns exit code 2."""
        result = runner.invoke(app, ["generate", "nonexistent"])
        assert result.exit_code == 2
