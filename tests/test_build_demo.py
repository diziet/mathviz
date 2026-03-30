"""Tests for scripts/build_demo.py — static demo site builder (Task 167)."""

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mathviz.core.generator import GeneratorMeta
from mathviz.core.math_object import Mesh, PointCloud

# Import the script module from the scripts/ directory
_SCRIPTS_DIR = str(Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
import build_demo


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_math_object(name: str = "lorenz") -> MagicMock:
    """Create a mock MathObject with mesh and point cloud."""
    obj = MagicMock()
    obj.generator_name = name
    obj.mesh = Mesh(
        vertices=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
    )
    obj.point_cloud = PointCloud(
        points=np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float64),
    )
    return obj


def _make_pipeline_result(name: str = "lorenz") -> MagicMock:
    """Create a mock PipelineResult."""
    result = MagicMock()
    result.math_object = _make_math_object(name)
    return result


def _make_generator_meta(name: str, category: str = "attractors") -> GeneratorMeta:
    """Create a GeneratorMeta for testing."""
    return GeneratorMeta(
        name=name,
        category=category,
        aliases=[],
        description=f"Test {name} generator",
        resolution_params={},
        generator_class=MagicMock(),
    )


def _make_resolved_config() -> MagicMock:
    """Create a mock resolved config matching pipeline expectations."""
    return MagicMock(
        container=MagicMock(), placement=MagicMock(), sampler_config=None
    )


def _fake_export_thumb(name: str, data_dir: Path) -> None:
    """Write a fake thumbnail PNG into data_dir."""
    (data_dir / "thumbnail.png").write_bytes(b"fake-png")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def output_dir(tmp_path: Path) -> Path:
    """Return a temporary output directory."""
    return tmp_path / "dist"


@pytest.fixture()
def _mock_pipeline():
    """Patch pipeline and export functions for all tests."""
    with (
        patch.object(build_demo, "run_pipeline") as mock_run,
        patch.object(build_demo, "mesh_to_glb", return_value=b"fake-glb"),
        patch.object(build_demo, "cloud_to_binary_ply", return_value=b"fake-ply"),
        patch("build_demo._export_thumbnail") as mock_export_thumb,
    ):
        mock_run.side_effect = lambda name, **kw: _make_pipeline_result(name)
        mock_export_thumb.side_effect = _fake_export_thumb
        yield {
            "run_pipeline": mock_run,
            "export_thumbnail": mock_export_thumb,
        }


def _mock_generators_fixture(metas: dict[str, GeneratorMeta]):
    """Create patch context for list_generators, get_generator_meta, and config."""
    return (
        patch.object(
            build_demo, "list_generators", return_value=list(metas.values())
        ),
        patch.object(
            build_demo, "get_generator_meta", side_effect=lambda n: metas[n]
        ),
        patch.object(build_demo, "_build_resolved_config", return_value=_make_resolved_config()),
    )


@pytest.fixture()
def _mock_generators_two():
    """Mock list_generators and get_generator_meta for lorenz + gyroid."""
    metas = {
        "lorenz": _make_generator_meta("lorenz", "attractors"),
        "gyroid": _make_generator_meta("gyroid", "geometry"),
    }
    patches = _mock_generators_fixture(metas)
    with patches[0], patches[1], patches[2]:
        yield metas


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBuildDemoOutput:
    """Verify the output structure of a build."""

    @pytest.mark.usefixtures("_mock_pipeline", "_mock_generators_two")
    def test_produces_index_html_and_manifest(self, output_dir: Path) -> None:
        """Script produces dist/index.html and dist/manifest.json."""
        build_demo.build_demo("all", output_dir, "preview")

        assert (output_dir / "index.html").is_file()
        assert (output_dir / "manifest.json").is_file()

    @pytest.mark.usefixtures("_mock_pipeline", "_mock_generators_two")
    def test_produces_data_directories(self, output_dir: Path) -> None:
        """Script produces at least one dist/data/{name}/ directory."""
        build_demo.build_demo("all", output_dir, "preview")

        data_dir = output_dir / "data"
        assert data_dir.is_dir()
        subdirs = [d.name for d in data_dir.iterdir() if d.is_dir()]
        assert len(subdirs) >= 1

    @pytest.mark.usefixtures("_mock_pipeline", "_mock_generators_two")
    def test_generator_directory_contents(self, output_dir: Path) -> None:
        """Each generator directory contains mesh.glb, cloud.ply, thumbnail.png."""
        build_demo.build_demo("all", output_dir, "preview")

        for name in ("lorenz", "gyroid"):
            gen_dir = output_dir / "data" / name
            assert gen_dir.is_dir(), f"Missing data dir for {name}"
            assert (gen_dir / "mesh.glb").is_file(), f"Missing mesh.glb for {name}"
            assert (gen_dir / "cloud.ply").is_file(), f"Missing cloud.ply for {name}"
            assert (gen_dir / "thumbnail.png").is_file(), f"Missing thumbnail for {name}"


class TestManifestSchema:
    """Verify manifest.json matches the expected schema."""

    @pytest.mark.usefixtures("_mock_pipeline", "_mock_generators_two")
    def test_manifest_valid_json(self, output_dir: Path) -> None:
        """manifest.json contains valid JSON."""
        build_demo.build_demo("all", output_dir, "preview")
        manifest = json.loads((output_dir / "manifest.json").read_text())
        assert isinstance(manifest, list)

    @pytest.mark.usefixtures("_mock_pipeline", "_mock_generators_two")
    def test_manifest_entry_fields(self, output_dir: Path) -> None:
        """Each manifest entry has required fields matching Task 166 schema."""
        build_demo.build_demo("all", output_dir, "preview")
        manifest = json.loads((output_dir / "manifest.json").read_text())

        required_fields = {
            "name", "category", "display_name", "description",
            "thumbnail", "mesh", "cloud",
        }
        for entry in manifest:
            missing = required_fields - set(entry.keys())
            assert not missing, f"Entry {entry.get('name')} missing fields: {missing}"
            assert entry["name"]
            assert entry["category"]


class TestGeneratorFiltering:
    """Verify --generators flag filters correctly."""

    @pytest.mark.usefixtures("_mock_pipeline", "_mock_generators_two")
    def test_specific_generators_produces_exact_dirs(self, output_dir: Path) -> None:
        """--generators lorenz,gyroid produces exactly 2 data directories."""
        build_demo.build_demo("lorenz,gyroid", output_dir, "preview")

        data_dir = output_dir / "data"
        subdirs = sorted(d.name for d in data_dir.iterdir() if d.is_dir())
        assert subdirs == ["gyroid", "lorenz"]

    @pytest.mark.usefixtures("_mock_pipeline", "_mock_generators_two")
    def test_manifest_matches_generator_count(self, output_dir: Path) -> None:
        """Manifest entry count matches requested generators."""
        build_demo.build_demo("lorenz,gyroid", output_dir, "preview")

        manifest = json.loads((output_dir / "manifest.json").read_text())
        assert len(manifest) == 2


class TestFailureHandling:
    """Verify that failing generators are skipped gracefully."""

    def test_failing_generator_skipped_with_warning(
        self, output_dir: Path
    ) -> None:
        """A failing generator is skipped; other generators still export."""
        good_meta = _make_generator_meta("lorenz", "attractors")
        bad_meta = _make_generator_meta("broken", "attractors")
        metas = {"lorenz": good_meta, "broken": bad_meta}

        def _fake_run(name: str, **kwargs: Any) -> MagicMock:
            if name == "broken":
                raise RuntimeError("Simulated failure")
            return _make_pipeline_result(name)

        gen_patches = _mock_generators_fixture(metas)
        with (
            gen_patches[0],
            gen_patches[1],
            gen_patches[2],
            patch.object(build_demo, "run_pipeline", side_effect=_fake_run),
            patch.object(build_demo, "mesh_to_glb", return_value=b"fake-glb"),
            patch.object(build_demo, "cloud_to_binary_ply", return_value=b"fake-ply"),
            patch("build_demo._export_thumbnail") as mock_thumb,
        ):
            mock_thumb.side_effect = _fake_export_thumb

            count = build_demo.build_demo("lorenz,broken", output_dir, "preview")

        assert count == 1
        assert (output_dir / "data" / "lorenz").is_dir()
        assert not (output_dir / "data" / "broken").is_dir()

        manifest = json.loads((output_dir / "manifest.json").read_text())
        names = [e["name"] for e in manifest]
        assert "lorenz" in names
        assert "broken" not in names

    def test_partial_directory_cleaned_on_export_failure(
        self, output_dir: Path
    ) -> None:
        """If export fails after data_dir is created, the directory is cleaned up."""
        meta = _make_generator_meta("failing", "attractors")
        metas = {"failing": meta}

        def _failing_thumb(name: str, data_dir: Path) -> None:
            raise OSError("Thumbnail generation failed")

        gen_patches = _mock_generators_fixture(metas)
        with (
            gen_patches[0],
            gen_patches[1],
            gen_patches[2],
            patch.object(
                build_demo, "run_pipeline",
                side_effect=lambda name, **kw: _make_pipeline_result(name),
            ),
            patch.object(build_demo, "mesh_to_glb", return_value=b"fake-glb"),
            patch.object(build_demo, "cloud_to_binary_ply", return_value=b"fake-ply"),
            patch("build_demo._export_thumbnail", side_effect=_failing_thumb),
        ):
            count = build_demo.build_demo("failing", output_dir, "preview")

        assert count == 0
        assert not (output_dir / "data" / "failing").is_dir()

    def test_all_generators_fail_no_manifest(self, output_dir: Path) -> None:
        """When all generators fail, no manifest.json is written."""
        meta = _make_generator_meta("broken", "attractors")
        metas = {"broken": meta}

        def _fail_run(name: str, **kwargs: Any) -> MagicMock:
            raise RuntimeError("Simulated failure")

        gen_patches = _mock_generators_fixture(metas)
        with (
            gen_patches[0],
            gen_patches[1],
            gen_patches[2],
            patch.object(build_demo, "run_pipeline", side_effect=_fail_run),
        ):
            count = build_demo.build_demo("broken", output_dir, "preview")

        assert count == 0
        assert not (output_dir / "manifest.json").is_file()


class TestParseArgs:
    """Verify CLI argument parsing."""

    def test_defaults(self) -> None:
        """Default args produce expected values."""
        args = build_demo.parse_args([])
        assert args.generators == "all"
        assert args.output == Path("dist")
        assert args.profile == "preview"

    def test_custom_args(self) -> None:
        """Custom args are parsed correctly."""
        args = build_demo.parse_args(
            ["--generators", "lorenz,gyroid", "--output", "build", "--profile", "production"]
        )
        assert args.generators == "lorenz,gyroid"
        assert args.output == Path("build")
        assert args.profile == "production"


class TestResolveGeneratorNames:
    """Verify generator name resolution."""

    def test_all_returns_sorted_names(self) -> None:
        """'all' returns sorted list of all generator names."""
        metas = [
            _make_generator_meta("zeta"),
            _make_generator_meta("alpha"),
        ]
        with patch.object(build_demo, "list_generators", return_value=metas):
            names = build_demo.resolve_generator_names("all")
        assert names == ["alpha", "zeta"]

    def test_comma_separated(self) -> None:
        """Comma-separated spec returns exact names in order."""
        names = build_demo.resolve_generator_names("lorenz,gyroid")
        assert names == ["lorenz", "gyroid"]

    def test_whitespace_handling(self) -> None:
        """Whitespace around names is stripped."""
        names = build_demo.resolve_generator_names(" lorenz , gyroid ")
        assert names == ["lorenz", "gyroid"]
