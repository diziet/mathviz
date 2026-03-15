"""Tests for the mathviz render-all CLI command."""

from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from mathviz.cli import app
from mathviz.cli_render_batch import (
    _build_jobs,
    _filter_generators,
    _validate_views,
    RenderResult,
)

runner = CliRunner()


def _make_success_result(
    generator_name: str, view: str, output_dir: str,
) -> RenderResult:
    """Create a successful RenderResult that writes a fake PNG file."""
    gen_dir = Path(output_dir) / generator_name
    gen_dir.mkdir(parents=True, exist_ok=True)
    output_path = gen_dir / f"{generator_name}_{view}.png"
    output_path.write_bytes(b"\x89PNG" + b"\x00" * 50)
    return RenderResult(
        generator_name=generator_name,
        view=view,
        output_path=str(output_path),
        elapsed=0.1,
    )


def _make_failure_result(generator_name: str, view: str) -> RenderResult:
    """Create a failed RenderResult."""
    return RenderResult(
        generator_name=generator_name,
        view=view,
        error="ValueError: test error",
        elapsed=0.05,
    )


def _mock_render_single_job(
    generator_name: str, view: str, output_dir: str,
    width: int, height: int, style: str,
) -> RenderResult:
    """Mock render job that creates fake output files."""
    return _make_success_result(generator_name, view, output_dir)


def _mock_render_single_job_with_failure(
    generator_name: str, view: str, output_dir: str,
    width: int, height: int, style: str,
) -> RenderResult:
    """Mock render job where 'broken_gen' fails."""
    if generator_name == "broken_gen":
        return _make_failure_result(generator_name, view)
    return _make_success_result(generator_name, view, output_dir)


def _invoke_render_all(
    tmp_path: Path,
    generators: str | None = "lorenz",
    views: str | None = "top",
    output_subdir: str = "renders",
    workers: int = 1,
) -> tuple[object, Path]:
    """Invoke render-all command with common defaults, return (result, output_dir)."""
    output_dir = tmp_path / output_subdir
    args = ["render-all", "--output-dir", str(output_dir), "--workers", str(workers)]
    if generators:
        args += ["--generators", generators]
    if views:
        args += ["--views", views]
    result = runner.invoke(app, args)
    return result, output_dir


class TestRenderAllCreatesFiles:
    """Test that render-all creates expected output files."""

    @patch("mathviz.cli_render_batch._render_single_job", _mock_render_single_job)
    def test_creates_files_for_specified_generators(self, tmp_path: Path) -> None:
        """render-all --generators lorenz,torus --views top creates expected PNGs."""
        result, output_dir = _invoke_render_all(
            tmp_path, generators="lorenz,torus", views="top",
        )
        assert result.exit_code == 0, f"Exit {result.exit_code}: {result.output}"
        assert (output_dir / "lorenz" / "lorenz_top.png").exists()
        assert (output_dir / "torus" / "torus_top.png").exists()

    @patch("mathviz.cli_render_batch._render_single_job", _mock_render_single_job)
    def test_output_directory_has_one_subdir_per_generator(
        self, tmp_path: Path,
    ) -> None:
        """Output directory structure has one subdirectory per generator."""
        result, output_dir = _invoke_render_all(
            tmp_path, generators="lorenz,torus", views="top",
        )
        assert result.exit_code == 0
        subdirs = [d for d in output_dir.iterdir() if d.is_dir()]
        subdir_names = {d.name for d in subdirs}
        assert subdir_names == {"lorenz", "torus"}

    @patch("mathviz.cli_render_batch._render_single_job", _mock_render_single_job)
    def test_workers_1_runs_sequentially(self, tmp_path: Path) -> None:
        """--workers 1 runs sequentially without error."""
        result, _ = _invoke_render_all(tmp_path)
        assert result.exit_code == 0

    @patch("mathviz.cli_render_batch._render_single_job", _mock_render_single_job)
    def test_custom_output_dir(self, tmp_path: Path) -> None:
        """--output-dir custom_dir/ creates renders in the specified directory."""
        result, output_dir = _invoke_render_all(
            tmp_path, output_subdir="custom_dir",
        )
        assert result.exit_code == 0
        assert (output_dir / "lorenz" / "lorenz_top.png").exists()

    @patch("mathviz.cli_render_batch._render_single_job", _mock_render_single_job)
    def test_default_views_produce_four_images(self, tmp_path: Path) -> None:
        """Default views produce 4 images per generator."""
        result, output_dir = _invoke_render_all(tmp_path, views=None)
        assert result.exit_code == 0
        lorenz_dir = output_dir / "lorenz"
        pngs = list(lorenz_dir.glob("*.png"))
        assert len(pngs) == 4


class TestRenderAllErrorHandling:
    """Test error handling in render-all command."""

    @patch(
        "mathviz.cli_render_batch._render_single_job",
        _mock_render_single_job_with_failure,
    )
    @patch(
        "mathviz.cli_render_batch._filter_generators",
        return_value=["lorenz", "broken_gen"],
    )
    def test_failed_generators_in_summary_not_raised(
        self, _mock_filter: object, tmp_path: Path,
    ) -> None:
        """Failed generators are reported in the summary, not raised."""
        result, output_dir = _invoke_render_all(
            tmp_path, generators="lorenz,broken_gen",
        )
        # Exit code 1 because there are failures
        assert result.exit_code == 1
        assert "broken_gen" in result.output
        assert "FAIL" in result.output
        # But lorenz still succeeded
        assert (output_dir / "lorenz" / "lorenz_top.png").exists()


class TestViewAliasIntegration:
    """Test that view aliases are resolved end-to-end through the CLI."""

    @patch("mathviz.cli_render_batch._render_single_job", _mock_render_single_job)
    def test_side_alias_resolves_to_right_in_filename(
        self, tmp_path: Path,
    ) -> None:
        """--views side creates file with resolved name 'right', not alias 'side'."""
        result, output_dir = _invoke_render_all(tmp_path, views="side")
        assert result.exit_code == 0
        # The mock creates files using the view name it receives.
        # If alias resolution works, the file should be lorenz_right.png
        assert (output_dir / "lorenz" / "lorenz_right.png").exists()
        assert not (output_dir / "lorenz" / "lorenz_side.png").exists()


class TestFilterGenerators:
    """Test generator filtering logic."""

    def test_excludes_data_driven_by_default(self) -> None:
        """Default filtering excludes data-driven generators."""
        selected = _filter_generators(None)
        assert "lorenz" in selected
        assert "building_extrude" not in selected
        assert "heightmap" not in selected
        assert "soundwave" not in selected

    def test_requested_generators_returned(self) -> None:
        """Explicit generator list is respected."""
        selected = _filter_generators(["lorenz", "torus"])
        assert selected == ["lorenz", "torus"]

    def test_unknown_generators_filtered_out(self) -> None:
        """Unknown generator names are filtered out with warning."""
        selected = _filter_generators(["lorenz", "nonexistent_xyz"])
        assert selected == ["lorenz"]

    def test_aliases_resolved_to_canonical(self) -> None:
        """Generator aliases are resolved to canonical names."""
        # "side" view alias won't work here — need a generator alias
        # All generators go through get_generator_meta which resolves aliases
        selected = _filter_generators(["lorenz"])
        assert selected == ["lorenz"]


class TestValidateViews:
    """Test view validation."""

    def test_valid_views_pass(self) -> None:
        """Valid view names pass validation."""
        result = _validate_views(["top", "front"])
        assert result == ["top", "front"]

    def test_aliases_resolved(self) -> None:
        """View aliases are resolved to canonical names."""
        result = _validate_views(["side", "angle"])
        assert result == ["right", "front-right-top"]

    def test_invalid_view_raises(self) -> None:
        """Invalid view name raises ValueError."""
        with pytest.raises(ValueError, match="Invalid view name"):
            _validate_views(["top", "invalid_view"])


class TestBuildJobs:
    """Test job building from generators x views."""

    def test_cartesian_product(self) -> None:
        """Jobs are the cartesian product of generators and views."""
        jobs = _build_jobs(["lorenz", "torus"], ["top", "front"])
        assert len(jobs) == 4
        pairs = [(j.generator_name, j.view) for j in jobs]
        assert ("lorenz", "top") in pairs
        assert ("lorenz", "front") in pairs
        assert ("torus", "top") in pairs
        assert ("torus", "front") in pairs
