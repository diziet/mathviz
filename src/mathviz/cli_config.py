"""CLI config resolution: merge mathviz.toml, --config, --profile and CLI flags."""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mathviz.cli_output import error_exit
from mathviz.core.config import (
    ResolvedConfig,
    deep_merge,
    load_object_config,
    load_project_config,
    load_sampling_profile,
    resolve_config,
)

DEFAULT_SEED = 42


@dataclass
class ConfigFlags:
    """CLI flag values that override the project and per-object config layers."""

    params: dict[str, Any] = field(default_factory=dict)
    seed: int | None = None
    config_path: Path | None = None
    profile_name: str | None = None
    container_width: float | None = None
    container_height: float | None = None
    container_depth: float | None = None


def resolve_cli_config(flags: ConfigFlags, json_output: bool) -> ResolvedConfig:
    """Resolve all config layers for a CLI command, exiting with code 2 on a config error.

    The returned seed is DEFAULT_SEED when no layer sets one.
    """
    project_cfg = load_project_config()
    object_cfg = (
        _load_safe(load_object_config, json_output, flags.config_path)
        if flags.config_path
        else None
    )
    if flags.profile_name:
        profile_cfg = _load_safe(load_sampling_profile, json_output, flags.profile_name)
        object_cfg = deep_merge(object_cfg, profile_cfg) if object_cfg else profile_cfg

    try:
        resolved = resolve_config(
            project=project_cfg,
            object_config=object_cfg,
            cli_overrides=_build_cli_overrides(flags),
        )
    except ValueError as exc:
        error_exit(str(exc), json_output)
        raise  # unreachable

    if resolved.seed is None:
        resolved.seed = DEFAULT_SEED
    return resolved


def _load_safe(
    loader: Callable[..., dict[str, Any]], json_output: bool, *args: Any
) -> dict[str, Any]:
    """Load a config/profile, calling error_exit on FileNotFoundError."""
    try:
        return loader(*args)
    except FileNotFoundError as exc:
        error_exit(str(exc), json_output)
        raise  # unreachable


def _build_cli_overrides(flags: ConfigFlags) -> dict[str, Any]:
    """Build CLI override dict from explicitly provided flag values."""
    overrides: dict[str, Any] = {}
    if flags.params:
        overrides["params"] = flags.params
    if flags.seed is not None:
        overrides["seed"] = flags.seed
    dims = {
        "width_mm": flags.container_width,
        "height_mm": flags.container_height,
        "depth_mm": flags.container_depth,
    }
    container_overrides = {k: v for k, v in dims.items() if v is not None}
    if container_overrides:
        overrides["container"] = container_overrides
    return overrides
