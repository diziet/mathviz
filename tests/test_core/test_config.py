"""Tests for config file support and sampling profiles."""

from pathlib import Path
from typing import Any

import pytest

from mathviz.core.config import (
    discover_project_config,
    load_object_config,
    load_sampling_profile,
    load_toml,
    merge_configs,
    resolve_config,
)
from mathviz.pipeline.sampler import SamplingMethod

# --- Fixtures ---


@pytest.fixture()
def tmp_project_dir(tmp_path: Path) -> Path:
    """Create a temp directory with a mathviz.toml project config."""
    config = tmp_path / "mathviz.toml"
    config.write_text("[container]\nwidth_mm = 120.0\nheight_mm = 80.0\ndepth_mm = 50.0\n")
    return tmp_path


@pytest.fixture()
def tmp_object_config(tmp_path: Path) -> Path:
    """Create a per-object TOML config."""
    obj_config = tmp_path / "my_object.toml"
    obj_config.write_text(
        'generator = "lorenz"\nseed = 99\n\n'
        "[params]\nsigma = 12.0\nrho = 30.0\n\n"
        "[container]\nwidth_mm = 150.0\n"
    )
    return obj_config


# --- Tests: Project config ---


class TestProjectConfig:
    """Test project config auto-discovery and loading."""

    def test_discover_in_directory(self, tmp_project_dir: Path) -> None:
        """Auto-discover mathviz.toml in the given directory."""
        found = discover_project_config(tmp_project_dir)
        assert found is not None
        assert found.name == "mathviz.toml"

    def test_discover_returns_none_when_absent(self, tmp_path: Path) -> None:
        """Returns None when no mathviz.toml exists."""
        found = discover_project_config(tmp_path)
        assert found is None

    def test_project_config_sets_container(self, tmp_project_dir: Path) -> None:
        """Project config sets container dimensions."""
        cfg = load_toml(tmp_project_dir / "mathviz.toml")
        resolved = resolve_config(project=cfg)
        assert resolved.container.width_mm == 120.0
        assert resolved.container.height_mm == 80.0
        assert resolved.container.depth_mm == 50.0

    def test_cli_overrides_project_config(self, tmp_project_dir: Path) -> None:
        """CLI flag overrides project config container dimensions."""
        project_cfg = load_toml(tmp_project_dir / "mathviz.toml")
        cli_overrides: dict[str, Any] = {"container": {"width_mm": 200.0}}
        resolved = resolve_config(project=project_cfg, cli_overrides=cli_overrides)
        assert resolved.container.width_mm == 200.0
        # Other dimensions still from project config
        assert resolved.container.height_mm == 80.0


# --- Tests: Per-object config ---


class TestObjectConfig:
    """Test per-object TOML config loading."""

    def test_object_config_specifies_generator_and_params(self, tmp_object_config: Path) -> None:
        """Per-object config specifies generator and params."""
        cfg = load_object_config(tmp_object_config)
        resolved = resolve_config(object_config=cfg)
        assert resolved.generator == "lorenz"
        assert resolved.seed == 99
        assert resolved.params["sigma"] == 12.0
        assert resolved.params["rho"] == 30.0

    def test_object_config_container_override(self, tmp_object_config: Path) -> None:
        """Per-object config overrides container width."""
        cfg = load_object_config(tmp_object_config)
        resolved = resolve_config(object_config=cfg)
        assert resolved.container.width_mm == 150.0

    def test_missing_config_file_error(self, tmp_path: Path) -> None:
        """Missing config file produces clear error with the path it tried."""
        missing = tmp_path / "nonexistent.toml"
        with pytest.raises(FileNotFoundError, match=str(missing)):
            load_object_config(missing)


# --- Tests: Sampling profiles ---


class TestSamplingProfiles:
    """Test predefined sampling profile loading."""

    def test_preview_profile_loads(self) -> None:
        """Sampling profile 'preview' loads correct point budget and method."""
        profile = load_sampling_profile("preview")
        sampling = profile["sampling"]
        assert sampling["method"] == "random_surface"
        assert sampling["num_points"] == 10000

    def test_preview_profile_builds_sampler_config(self) -> None:
        """Preview profile resolves to correct SamplerConfig."""
        profile = load_sampling_profile("preview")
        resolved = resolve_config(object_config=profile)
        assert resolved.sampler_config is not None
        assert resolved.sampler_config.method == SamplingMethod.RANDOM_SURFACE
        assert resolved.sampler_config.num_points == 10000

    def test_production_profile_loads(self) -> None:
        """Production profile loads high-density settings."""
        profile = load_sampling_profile("production")
        sampling = profile["sampling"]
        assert sampling["method"] == "uniform_surface"
        assert sampling["density"] == 8.0

    def test_missing_profile_error(self) -> None:
        """Missing profile produces error listing available profiles."""
        with pytest.raises(FileNotFoundError, match="available:"):
            load_sampling_profile("nonexistent")


# --- Tests: Merge order ---


class TestMergeOrder:
    """Test config merge order: built-in -> project -> per-object -> CLI."""

    def test_builtin_defaults(self) -> None:
        """With no config layers, built-in defaults are used."""
        resolved = resolve_config()
        assert resolved.container.width_mm == 100.0
        assert resolved.container.depth_mm == 100.0

    def test_project_overrides_builtin(self, tmp_project_dir: Path) -> None:
        """Project config overrides built-in defaults."""
        project_cfg = load_toml(tmp_project_dir / "mathviz.toml")
        resolved = resolve_config(project=project_cfg)
        assert resolved.container.width_mm == 120.0

    def test_object_overrides_project(
        self,
        tmp_project_dir: Path,
        tmp_object_config: Path,
    ) -> None:
        """Per-object config overrides project config."""
        project_cfg = load_toml(tmp_project_dir / "mathviz.toml")
        object_cfg = load_object_config(tmp_object_config)
        resolved = resolve_config(project=project_cfg, object_config=object_cfg)
        # Object config sets width=150, project sets width=120
        assert resolved.container.width_mm == 150.0
        # Object config doesn't set height, project's 80 wins
        assert resolved.container.height_mm == 80.0

    def test_cli_overrides_all(
        self,
        tmp_project_dir: Path,
        tmp_object_config: Path,
    ) -> None:
        """CLI overrides beat project and per-object configs."""
        project_cfg = load_toml(tmp_project_dir / "mathviz.toml")
        object_cfg = load_object_config(tmp_object_config)
        cli: dict[str, Any] = {"container": {"width_mm": 300.0}, "seed": 7}
        resolved = resolve_config(
            project=project_cfg,
            object_config=object_cfg,
            cli_overrides=cli,
        )
        assert resolved.container.width_mm == 300.0
        assert resolved.seed == 7

    def test_merge_preserves_nested_keys(self) -> None:
        """Deep merge preserves non-overridden nested keys."""
        merged = merge_configs(
            project={"container": {"width_mm": 120.0, "height_mm": 80.0}},
            cli_overrides={"container": {"width_mm": 200.0}},
        )
        assert merged["container"]["width_mm"] == 200.0
        assert merged["container"]["height_mm"] == 80.0

    def test_seed_from_config_not_overridden_by_absent_cli(self) -> None:
        """Seed from object config is preserved when CLI doesn't specify seed."""
        resolved = resolve_config(object_config={"seed": 99})
        assert resolved.seed == 99

    def test_seed_zero_is_valid(self) -> None:
        """Seed of 0 is a valid value and not treated as falsy."""
        resolved = resolve_config(object_config={"seed": 0})
        assert resolved.seed == 0

    def test_no_sampler_config_when_no_sampling_section(self) -> None:
        """Without sampling config, sampler_config should be None."""
        resolved = resolve_config()
        assert resolved.sampler_config is None
