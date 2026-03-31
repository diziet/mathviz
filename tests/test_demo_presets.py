"""Tests for demo preset resolution and export (Task 169)."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from mathviz.demo_builder import (
    DEFAULT_SEED,
    DemoBuildResult,
    load_presets_file,
    resolve_demo_preset,
)

# ---------------------------------------------------------------------------
# load_presets_file
# ---------------------------------------------------------------------------


class TestLoadPresetsFile:
    """Test loading demo_presets.json."""

    def test_loads_valid_json(self, tmp_path: Path) -> None:
        """Valid JSON dict is returned as-is."""
        presets = {"lorenz": {"params": {"sigma": 12}, "seed": 7}}
        path = tmp_path / "presets.json"
        path.write_text(json.dumps(presets), encoding="utf-8")
        result = load_presets_file(path)
        assert result == presets

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        """Missing file returns empty dict."""
        result = load_presets_file(tmp_path / "nonexistent.json")
        assert result == {}

    def test_invalid_json_returns_empty(self, tmp_path: Path) -> None:
        """Malformed JSON returns empty dict."""
        path = tmp_path / "bad.json"
        path.write_text("not json", encoding="utf-8")
        result = load_presets_file(path)
        assert result == {}

    def test_non_dict_json_returns_empty(self, tmp_path: Path) -> None:
        """JSON that is not a dict returns empty dict."""
        path = tmp_path / "list.json"
        path.write_text("[1, 2, 3]", encoding="utf-8")
        result = load_presets_file(path)
        assert result == {}


# ---------------------------------------------------------------------------
# resolve_demo_preset
# ---------------------------------------------------------------------------


class TestResolveDemoPreset:
    """Test 3-tier fallback for preset resolution."""

    def test_presets_file_used_for_known_generator(self) -> None:
        """With presets containing lorenz, uses those params and seed."""
        presets = {"lorenz": {"params": {"sigma": 12}, "seed": 7}}
        with patch("mathviz.demo_builder._newest_snapshot_for", return_value=None):
            params, seed = resolve_demo_preset("lorenz", presets)
        assert params == {"sigma": 12}
        assert seed == 7

    def test_falls_back_to_defaults_when_no_preset(self) -> None:
        """With no presets and no snapshots, falls back to defaults."""
        with patch("mathviz.demo_builder._newest_snapshot_for", return_value=None):
            params, seed = resolve_demo_preset("lorenz", {})
        assert params is None
        assert seed == DEFAULT_SEED

    def test_snapshot_takes_precedence_over_presets(self) -> None:
        """Snapshot on disk overrides demo_presets.json entry."""
        presets = {"lorenz": {"params": {"sigma": 12}, "seed": 7}}
        snapshot_data = {"params": {"sigma": 20}, "seed": 99}
        with patch(
            "mathviz.demo_builder._newest_snapshot_for", return_value=snapshot_data
        ):
            params, seed = resolve_demo_preset("lorenz", presets)
        assert params == {"sigma": 20}
        assert seed == 99

    def test_no_presets_flag_uses_defaults(self) -> None:
        """--no-presets (use_presets=False) ignores both snapshots and file."""
        presets = {"lorenz": {"params": {"sigma": 12}, "seed": 7}}
        params, seed = resolve_demo_preset("lorenz", presets, use_presets=False)
        assert params is None
        assert seed == DEFAULT_SEED

    def test_preset_missing_seed_uses_default(self) -> None:
        """Preset entry without seed falls back to DEFAULT_SEED."""
        presets = {"lorenz": {"params": {"sigma": 12}}}
        with patch("mathviz.demo_builder._newest_snapshot_for", return_value=None):
            params, seed = resolve_demo_preset("lorenz", presets)
        assert params == {"sigma": 12}
        assert seed == DEFAULT_SEED


# ---------------------------------------------------------------------------
# Manifest includes params and seed
# ---------------------------------------------------------------------------


class TestManifestParamsAndSeed:
    """Test that manifest entries contain params and seed fields."""

    @patch("mathviz.demo_builder._copy_static_assets")
    @patch("mathviz.demo_builder.validate_generator_names", return_value=[])
    @patch("mathviz.demo_builder.generate_thumbnail", return_value=None)
    @patch("mathviz.demo_builder.cloud_to_binary_ply", return_value=b"fake-ply")
    @patch("mathviz.demo_builder.mesh_to_glb", return_value=b"fake-glb")
    @patch("mathviz.demo_builder.run_pipeline")
    @patch("mathviz.demo_builder._build_resolved_config")
    @patch("mathviz.demo_builder.get_generator_meta")
    @patch("mathviz.demo_builder.load_presets_file")
    @patch("mathviz.demo_builder._newest_snapshot_for", return_value=None)
    def test_manifest_entries_include_params_and_seed(
        self,
        mock_snap: object,
        mock_presets: object,
        mock_meta: object,
        mock_config: object,
        mock_run: object,
        mock_glb: object,
        mock_ply: object,
        mock_thumb: object,
        mock_validate: object,
        mock_static: object,
        tmp_path: Path,
    ) -> None:
        """Manifest entries have params and seed fields."""

        from mathviz.core.generator import GeneratorMeta
        from mathviz.demo_builder import build_demo

        meta = GeneratorMeta(
            name="lorenz",
            category="attractors",
            aliases=[],
            description="Test",
            resolution_params={},
            generator_class=MagicMock(),
        )
        mock_meta.return_value = meta
        mock_config.return_value = MagicMock()
        mock_presets.return_value = {"lorenz": {"params": {"sigma": 12}, "seed": 7}}

        obj = MagicMock()
        obj.mesh = MagicMock()
        obj.point_cloud = MagicMock()
        pipeline_result = MagicMock()
        pipeline_result.math_object = obj
        mock_run.return_value = pipeline_result

        out = tmp_path / "demo-out"
        build_demo("lorenz", out, "preview")

        manifest = json.loads((out / "manifest.json").read_text())
        assert len(manifest) == 1
        assert manifest[0]["params"] == {"sigma": 12}
        assert manifest[0]["seed"] == 7

    @patch("mathviz.demo_builder._copy_static_assets")
    @patch("mathviz.demo_builder.validate_generator_names", return_value=[])
    @patch("mathviz.demo_builder.generate_thumbnail", return_value=None)
    @patch("mathviz.demo_builder.cloud_to_binary_ply", return_value=b"fake-ply")
    @patch("mathviz.demo_builder.mesh_to_glb", return_value=b"fake-glb")
    @patch("mathviz.demo_builder.run_pipeline")
    @patch("mathviz.demo_builder._build_resolved_config")
    @patch("mathviz.demo_builder.get_generator_meta")
    @patch("mathviz.demo_builder.load_presets_file")
    @patch("mathviz.demo_builder._newest_snapshot_for", return_value=None)
    def test_manifest_default_params_when_no_presets(
        self,
        mock_snap: object,
        mock_presets: object,
        mock_meta: object,
        mock_config: object,
        mock_run: object,
        mock_glb: object,
        mock_ply: object,
        mock_thumb: object,
        mock_validate: object,
        mock_static: object,
        tmp_path: Path,
    ) -> None:
        """With no presets, manifest shows empty params and seed 42."""
        from mathviz.core.generator import GeneratorMeta
        from mathviz.demo_builder import build_demo

        meta = GeneratorMeta(
            name="lorenz",
            category="attractors",
            aliases=[],
            description="Test",
            resolution_params={},
            generator_class=MagicMock(),
        )
        mock_meta.return_value = meta
        mock_config.return_value = MagicMock()
        mock_presets.return_value = {}

        obj = MagicMock()
        obj.mesh = MagicMock()
        obj.point_cloud = MagicMock()
        pipeline_result = MagicMock()
        pipeline_result.math_object = obj
        mock_run.return_value = pipeline_result

        out = tmp_path / "demo-out"
        build_demo("lorenz", out, "preview")

        manifest = json.loads((out / "manifest.json").read_text())
        assert manifest[0]["params"] == {}
        assert manifest[0]["seed"] == DEFAULT_SEED


# ---------------------------------------------------------------------------
# --no-presets CLI flag
# ---------------------------------------------------------------------------


class TestNoPresetsFlag:
    """Test --no-presets CLI flag."""

    def test_help_shows_no_presets_flag(self) -> None:
        """mathviz export-demo --help shows --no-presets."""
        from typer.testing import CliRunner

        from mathviz.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["export-demo", "--help"])
        assert result.exit_code == 0
        assert "--no-presets" in result.output

    @patch("mathviz.cli_demo.build_demo")
    def test_no_presets_passes_use_presets_false(
        self, mock_build: MagicMock, tmp_path: Path
    ) -> None:
        """--no-presets passes use_presets=False to build_demo."""
        from typer.testing import CliRunner

        from mathviz.cli import app

        mock_build.return_value = DemoBuildResult(
            succeeded=1, skipped=[], is_manifest_written=True
        )
        runner = CliRunner()
        out = tmp_path / "demo-out"
        runner.invoke(
            app,
            [
                "export-demo",
                "--generators", "lorenz",
                "--output", str(out),
                "--no-presets",
            ],
        )
        mock_build.assert_called_once_with(
            "lorenz", out, "preview", use_presets=False
        )

    @patch("mathviz.cli_demo.build_demo")
    def test_default_uses_presets(
        self, mock_build: MagicMock, tmp_path: Path
    ) -> None:
        """Without --no-presets, use_presets=True by default."""
        from typer.testing import CliRunner

        from mathviz.cli import app

        mock_build.return_value = DemoBuildResult(
            succeeded=1, skipped=[], is_manifest_written=True
        )
        runner = CliRunner()
        out = tmp_path / "demo-out"
        runner.invoke(
            app,
            [
                "export-demo",
                "--generators", "lorenz",
                "--output", str(out),
            ],
        )
        mock_build.assert_called_once_with(
            "lorenz", out, "preview", use_presets=True
        )


# ---------------------------------------------------------------------------
# export_presets.py script
# ---------------------------------------------------------------------------


class TestExportPresetsScript:
    """Test scripts/export_presets.py reads snapshots and writes valid JSON."""

    def test_writes_valid_presets_from_snapshots(self, tmp_path: Path) -> None:
        """export_presets reads snapshots and writes demo_presets.json."""
        _scripts_dir = str(Path(__file__).resolve().parent.parent / "scripts")
        if _scripts_dir not in sys.path:
            sys.path.insert(0, _scripts_dir)
        import export_presets

        snap1 = MagicMock()
        snap1.generator = "lorenz"
        snap1.params = {"sigma": 10}
        snap1.seed = 7
        snap1.created_at = "2026-03-30T12:00:00"

        snap2 = MagicMock()
        snap2.generator = "gyroid"
        snap2.params = {"scale": 2.0}
        snap2.seed = 42
        snap2.created_at = "2026-03-29T12:00:00"

        output_file = tmp_path / "presets.json"
        with patch.object(
            export_presets, "list_snapshots", return_value=[snap1, snap2]
        ):
            presets = export_presets.build_presets_from_snapshots()
            export_presets.write_presets(presets, output_file)

        data = json.loads(output_file.read_text())
        assert "lorenz" in data
        assert data["lorenz"]["params"] == {"sigma": 10}
        assert data["lorenz"]["seed"] == 7
        assert "gyroid" in data

    def test_picks_newest_snapshot_per_generator(self, tmp_path: Path) -> None:
        """When multiple snapshots exist for a generator, picks the newest."""
        _scripts_dir = str(Path(__file__).resolve().parent.parent / "scripts")
        if _scripts_dir not in sys.path:
            sys.path.insert(0, _scripts_dir)
        import export_presets

        # list_snapshots returns newest first
        newer = MagicMock()
        newer.generator = "lorenz"
        newer.params = {"sigma": 20}
        newer.seed = 99

        older = MagicMock()
        older.generator = "lorenz"
        older.params = {"sigma": 10}
        older.seed = 7

        with patch.object(
            export_presets, "list_snapshots", return_value=[newer, older]
        ):
            presets = export_presets.build_presets_from_snapshots()

        assert presets["lorenz"]["params"] == {"sigma": 20}
        assert presets["lorenz"]["seed"] == 99

    def test_empty_snapshots_writes_empty_file(self, tmp_path: Path) -> None:
        """No snapshots produces an empty presets dict."""
        _scripts_dir = str(Path(__file__).resolve().parent.parent / "scripts")
        if _scripts_dir not in sys.path:
            sys.path.insert(0, _scripts_dir)
        import export_presets

        output_file = tmp_path / "presets.json"
        with patch.object(export_presets, "list_snapshots", return_value=[]):
            presets = export_presets.build_presets_from_snapshots()
            export_presets.write_presets(presets, output_file)

        data = json.loads(output_file.read_text())
        assert data == {}
