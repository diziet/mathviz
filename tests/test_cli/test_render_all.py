"""Tests for the mathviz render-all CLI command."""

from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from mathviz.cli import app
from mathviz.cli_render_batch import (
    BatchSummary,
    RenderResult,
    _build_jobs,
    _filter_generators,
    _validate_views,
    run_batch_render,
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


class TestRenderAllCreatesFiles:
    """Test that render-all creates expected output files."""

    @patch("mathviz.cli_render_batch._render_single_job", _mock_render_single_job)
    def test_creates_files_for_specified_generators(self, tmp_path: Path) -> None:
        """render-all --generators lorenz,torus --views top creates expected PNGs."""
        output_dir = tmp_path / "renders"
        result = runner.invoke(
            app,
            [
                "render-all",
                "--generators", "lorenz,torus",
                "--views", "top",
                "--output-dir", str(output_dir),
                "--workers", "1",
            ],
        )
        assert result.exit_code == 0, f"Exit {result.exit_code}: {result.output}"
        assert (output_dir / "lorenz" / "lorenz_top.png").exists()
        assert (output_dir / "torus" / "torus_top.png").exists()

    @patch("mathviz.cli_render_batch._render_single_job", _mock_render_single_job)
    def test_output_directory_has_one_subdir_per_generator(
        self, tmp_path: Path,
    ) -> None:
        """Output directory structure has one subdirectory per generator."""
        output_dir = tmp_path / "renders"
        result = runner.invoke(
            app,
            [
                "render-all",
                "--generators", "lorenz,torus",
                "--views", "top",
                "--output-dir", str(output_dir),
                "--workers", "1",
            ],
        )
        assert result.exit_code == 0
        subdirs = [d for d in output_dir.iterdir() if d.is_dir()]
        subdir_names = {d.name for d in subdirs}
        assert subdir_names == {"lorenz", "torus"}

    @patch("mathviz.cli_render_batch._render_single_job", _mock_render_single_job)
    def test_workers_1_runs_sequentially(self, tmp_path: Path) -> None:
        """--workers 1 runs sequentially without error."""
        output_dir = tmp_path / "renders"
        result = runner.invoke(
            app,
            [
                "render-all",
                "--generators", "lorenz",
                "--views", "top",
                "--output-dir", str(output_dir),
                "--workers", "1",
            ],
        )
        assert result.exit_code == 0

    @patch("mathviz.cli_render_batch._render_single_job", _mock_render_single_job)
    def test_custom_output_dir(self, tmp_path: Path) -> None:
        """--output-dir custom_dir/ creates renders in the specified directory."""
        custom_dir = tmp_path / "custom_dir"
        result = runner.invoke(
            app,
            [
                "render-all",
                "--generators", "lorenz",
                "--views", "top",
                "--output-dir", str(custom_dir),
                "--workers", "1",
            ],
        )
        assert result.exit_code == 0
        assert (custom_dir / "lorenz" / "lorenz_top.png").exists()

    @patch("mathviz.cli_render_batch._render_single_job", _mock_render_single_job)
    def test_default_views_produce_four_images(self, tmp_path: Path) -> None:
        """Default views produce 4 images per generator."""
        output_dir = tmp_path / "renders"
        result = runner.invoke(
            app,
            [
                "render-all",
                "--generators", "lorenz",
                "--output-dir", str(output_dir),
                "--workers", "1",
            ],
        )
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
        output_dir = tmp_path / "renders"
        result = runner.invoke(
            app,
            [
                "render-all",
                "--generators", "lorenz,broken_gen",
                "--views", "top",
                "--output-dir", str(output_dir),
                "--workers", "1",
            ],
        )
        # Exit code 1 because there are failures
        assert result.exit_code == 1
        assert "broken_gen" in result.output
        assert "FAIL" in result.output
        # But lorenz still succeeded
        assert (output_dir / "lorenz" / "lorenz_top.png").exists()


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


class TestValidateViews:
    """Test view validation."""

    def test_valid_views_pass(self) -> None:
        """Valid view names pass validation."""
        result = _validate_views(["top", "front", "side", "angle"])
        assert result == ["top", "front", "side", "angle"]

    def test_invalid_view_raises(self) -> None:
        """Invalid view name raises ValueError."""
        import pytest

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
