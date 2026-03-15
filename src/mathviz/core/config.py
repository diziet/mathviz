"""Config file support: project TOML, per-object TOML, and sampling profiles."""

import logging
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mathviz.core.container import Container, PlacementPolicy
from mathviz.pipeline.sampler import SamplerConfig, SamplingMethod

logger = logging.getLogger(__name__)

PROJECT_CONFIG_NAME = "mathviz.toml"

# Built-in defaults for the full config hierarchy
_BUILTIN_DEFAULTS: dict[str, Any] = {
    "container": {
        "width_mm": 100.0,
        "height_mm": 100.0,
        "depth_mm": 100.0,
        "margin_x_mm": 5.0,
        "margin_y_mm": 5.0,
        "margin_z_mm": 5.0,
    },
    "placement": {},
}


@dataclass
class ResolvedConfig:
    """Final merged configuration ready for pipeline use."""

    generator: str | None = None
    seed: int | None = None
    params: dict[str, Any] = field(default_factory=dict)
    container: Container = field(default_factory=Container)
    placement: PlacementPolicy = field(default_factory=PlacementPolicy)
    sampler_config: SamplerConfig | None = None
    representation: dict[str, Any] = field(default_factory=dict)


def load_toml(path: Path) -> dict[str, Any]:
    """Load and parse a TOML file, raising FileNotFoundError with clear message."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "rb") as f:
        return tomllib.load(f)


def discover_project_config(search_dir: Path | None = None) -> Path | None:
    """Find mathviz.toml in the given directory (default: cwd)."""
    base = search_dir or Path.cwd()
    candidate = base / PROJECT_CONFIG_NAME
    if candidate.is_file():
        logger.info("Found project config: %s", candidate)
        return candidate
    return None


def load_project_config(path: Path | None = None) -> dict[str, Any]:
    """Load project config from path, or auto-discover in cwd."""
    if path is not None:
        return load_toml(path)
    discovered = discover_project_config()
    if discovered is not None:
        return load_toml(discovered)
    return {}


def load_object_config(path: Path) -> dict[str, Any]:
    """Load a per-object TOML config file."""
    return load_toml(path)


def load_sampling_profile(name: str) -> dict[str, Any]:
    """Load a named sampling profile from the profiles directory."""
    profiles_dir = Path(__file__).parent.parent / "profiles"
    profile_path = profiles_dir / f"{name}.toml"
    if not profile_path.exists():
        available = [p.stem for p in profiles_dir.glob("*.toml")]
        raise FileNotFoundError(
            f"Sampling profile not found: {profile_path} "
            f"(available: {', '.join(sorted(available)) or 'none'})"
        )
    return load_toml(profile_path)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base, returning a new dict."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def merge_configs(
    *,
    project: dict[str, Any] | None = None,
    object_config: dict[str, Any] | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge config layers: built-in defaults -> project -> per-object -> CLI."""
    merged = dict(_BUILTIN_DEFAULTS)
    for layer in [project, object_config, cli_overrides]:
        if layer:
            merged = deep_merge(merged, layer)
    return merged


def _build_container(cfg: dict[str, Any]) -> Container:
    """Build a Container from a config dict section."""
    container_data = cfg.get("container", {})
    return Container(**container_data)


def _build_placement(cfg: dict[str, Any]) -> PlacementPolicy:
    """Build a PlacementPolicy from a config dict section."""
    placement_data = cfg.get("placement", {})
    return PlacementPolicy(**placement_data)


def _build_sampler_config(cfg: dict[str, Any]) -> SamplerConfig | None:
    """Build a SamplerConfig from a config dict section, if present."""
    sampling_data = cfg.get("sampling", {})
    if not sampling_data:
        return None
    method_str = sampling_data.get("method")
    kwargs: dict[str, Any] = {}
    if method_str:
        kwargs["method"] = SamplingMethod(method_str)
    for key in ("density", "num_points", "seed", "resample"):
        if key in sampling_data:
            kwargs[key] = sampling_data[key]
    if "point_budget" in sampling_data and "num_points" not in kwargs:
        kwargs["num_points"] = sampling_data["point_budget"]
    return SamplerConfig(**kwargs)


def resolve_config(
    *,
    project: dict[str, Any] | None = None,
    object_config: dict[str, Any] | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> ResolvedConfig:
    """Merge all config layers and build typed config objects."""
    merged = merge_configs(
        project=project,
        object_config=object_config,
        cli_overrides=cli_overrides,
    )
    return ResolvedConfig(
        generator=merged.get("generator"),
        seed=merged.get("seed"),
        params=merged.get("params", {}),
        container=_build_container(merged),
        placement=_build_placement(merged),
        sampler_config=_build_sampler_config(merged),
        representation=merged.get("representation", {}),
    )
