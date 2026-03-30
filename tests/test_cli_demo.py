"""Tests for the export-demo CLI subcommand (Task 168)."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from mathviz.cli import app
from mathviz.cli_demo import DEFAULT_GENERATORS

runner = CliRunner()


class TestExportDemoHelp:
    """Verify --help output for export-demo."""

    def test_help_shows_all_flags(self) -> None:
        """mathviz export-demo --help shows --generators, --output, --profile."""
        result = runner.invoke(app, ["export-demo", "--help"])
        assert result.exit_code == 0
        assert "--generators" in result.output
        assert "--output" in result.output
        assert "--profile" in result.output

    def test_help_shows_verbose_and_quiet(self) -> None:
        """mathviz export-demo --help shows --verbose and --quiet."""
        result = runner.invoke(app, ["export-demo", "--help"])
        assert "--verbose" in result.output
        assert "--quiet" in result.output

    def test_help_shows_description(self) -> None:
        """Help text includes a meaningful description."""
        result = runner.invoke(app, ["export-demo", "--help"])
        assert "demo" in result.output.lower()


class TestExportDemoRun:
    """Verify export-demo runs the build and produces expected output."""

    @patch("mathviz.cli_demo.build_demo", return_value=1)
    def test_runs_with_single_generator(
        self, mock_build: object, tmp_path: Path
    ) -> None:
        """mathviz export-demo --generators lorenz --output <dir> succeeds."""
        out = tmp_path / "demo-out"
        result = runner.invoke(
            app, ["export-demo", "--generators", "lorenz", "--output", str(out)]
        )
        assert result.exit_code == 0

    @patch("mathviz.cli_demo.build_demo", return_value=1)
    def test_passes_generators_to_build(
        self, mock_build: object, tmp_path: Path
    ) -> None:
        """Generator spec is forwarded to build_demo."""
        out = tmp_path / "demo-out"
        runner.invoke(
            app,
            ["export-demo", "--generators", "lorenz,gyroid", "--output", str(out)],
        )
        mock_build.assert_called_once_with("lorenz,gyroid", out, "preview")

    @patch("mathviz.cli_demo.build_demo", return_value=1)
    def test_passes_profile_to_build(
        self, mock_build: object, tmp_path: Path
    ) -> None:
        """Profile flag is forwarded to build_demo."""
        out = tmp_path / "demo-out"
        runner.invoke(
            app,
            [
                "export-demo",
                "--generators", "lorenz",
                "--output", str(out),
                "--profile", "production",
            ],
        )
        mock_build.assert_called_once_with("lorenz", out, "production")

    @patch("mathviz.cli_demo.build_demo", return_value=1)
    def test_default_generators(self, mock_build: object, tmp_path: Path) -> None:
        """Default generators list is the curated set of ~15."""
        out = tmp_path / "demo-out"
        runner.invoke(app, ["export-demo", "--output", str(out)])
        call_args = mock_build.call_args
        generator_spec = call_args[0][0]
        names = [n.strip() for n in generator_spec.split(",")]
        assert len(names) == 15
        assert "lorenz" in names

    @patch("mathviz.cli_demo.build_demo", return_value=0)
    def test_exit_code_1_on_zero_generators(
        self, mock_build: object, tmp_path: Path
    ) -> None:
        """Exit code 1 when no generators succeed."""
        out = tmp_path / "demo-out"
        result = runner.invoke(
            app, ["export-demo", "--generators", "lorenz", "--output", str(out)]
        )
        assert result.exit_code == 1

    @patch("mathviz.cli_demo.build_demo", return_value=3)
    def test_quiet_suppresses_output(
        self, mock_build: object, tmp_path: Path
    ) -> None:
        """--quiet suppresses progress messages."""
        out = tmp_path / "demo-out"
        result = runner.invoke(
            app,
            [
                "export-demo",
                "--generators", "lorenz",
                "--output", str(out),
                "--quiet",
            ],
        )
        assert result.exit_code == 0
        assert "Building demo site" not in result.output


class TestExportDemoOutputStructure:
    """Verify end-to-end output directory structure with mocked pipeline."""

    @patch("mathviz.demo_builder.generate_thumbnail")
    @patch("mathviz.demo_builder.cloud_to_binary_ply", return_value=b"fake-ply")
    @patch("mathviz.demo_builder.mesh_to_glb", return_value=b"fake-glb")
    @patch("mathviz.demo_builder.run_pipeline")
    @patch("mathviz.demo_builder._build_resolved_config")
    @patch("mathviz.demo_builder.get_generator_meta")
    def test_output_directory_structure(
        self,
        mock_meta: object,
        mock_config: object,
        mock_run: object,
        mock_glb: object,
        mock_ply: object,
        mock_thumb: object,
        tmp_path: Path,
    ) -> None:
        """Output directory has data/<name>/ with expected files."""
        from unittest.mock import MagicMock

        from mathviz.core.generator import GeneratorMeta
        from mathviz.core.math_object import Mesh, PointCloud
        import numpy as np

        meta = GeneratorMeta(
            name="lorenz",
            category="attractors",
            aliases=[],
            description="Test",
            resolution_params={},
            generator_class=MagicMock(),
        )
        mock_meta.return_value = meta

        resolved = MagicMock()
        mock_config.return_value = resolved

        obj = MagicMock()
        obj.mesh = Mesh(
            vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64),
            faces=np.array([[0, 1, 2]], dtype=np.int64),
        )
        obj.point_cloud = PointCloud(
            points=np.array([[0, 0, 0]], dtype=np.float64),
        )
        pipeline_result = MagicMock()
        pipeline_result.math_object = obj
        mock_run.return_value = pipeline_result

        # generate_thumbnail returns a path; we create a fake webp
        fake_webp = tmp_path / "fake.webp"
        from PIL import Image
        img = Image.new("RGB", (64, 64), "red")
        img.save(fake_webp, "WEBP")
        mock_thumb.return_value = fake_webp

        out = tmp_path / "demo-out"
        result = runner.invoke(
            app,
            ["export-demo", "--generators", "lorenz", "--output", str(out)],
        )
        assert result.exit_code == 0

        # Check structure
        assert (out / "data" / "lorenz" / "mesh.glb").is_file()
        assert (out / "data" / "lorenz" / "cloud.ply").is_file()
        assert (out / "data" / "lorenz" / "thumbnail.png").is_file()
        assert (out / "manifest.json").is_file()

        manifest = json.loads((out / "manifest.json").read_text())
        assert len(manifest) == 1
        assert manifest[0]["name"] == "lorenz"
