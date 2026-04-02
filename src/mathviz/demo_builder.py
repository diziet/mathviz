"""Demo site builder — package-importable entry point for export-demo CLI.

Provides :func:`build_demo` which orchestrates the full static demo build:
resolve generators, run pipeline, export assets, write manifest.
"""

import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mathviz.core.config import load_sampling_profile, resolve_config
from mathviz.core.generator import GeneratorMeta, get_generator_meta, list_generators
from mathviz.pipeline.runner import run as run_pipeline
from mathviz.preview.lod import cloud_to_binary_ply, mesh_to_glb
from mathviz.preview.renderer import RenderConfig, render_to_png

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
PRESETS_PATH = Path(__file__).resolve().parent / "data" / "demo_presets.json"
DEFAULT_SEED = 42
THUMBNAIL_VIEW_MODE = "vertex"
THUMBNAIL_BG_COLOR = "#1a1a2e"
THUMBNAIL_OBJ_COLOR = "#44ccff"
_ROOT_ASSETS = ("buildbanner.js", "favicon.ico")


@dataclass
class DemoBuildResult:
    """Result of a demo site build."""

    succeeded: int = 0
    skipped: list[str] = field(default_factory=list)
    is_manifest_written: bool = False


def resolve_generator_names(spec: str) -> list[str]:
    """Resolve generator spec to a sorted list of canonical names."""
    if spec.strip().lower() == "all":
        return sorted(m.name for m in list_generators())
    return [name.strip() for name in spec.split(",") if name.strip()]


def validate_generator_names(names: list[str]) -> list[str]:
    """Validate generator names against the registry, returning unknown names."""
    known = {m.name for m in list_generators()}
    return [n for n in names if n not in known]


def load_presets_file(path: Path | None = None) -> dict[str, Any]:
    """Load demo_presets.json, returning empty dict if missing or invalid."""
    presets_path = path or PRESETS_PATH
    if not presets_path.is_file():
        return {}
    try:
        data = json.loads(presets_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            logger.warning("demo_presets.json is not a dict, ignoring")
            return {}
        return data
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read demo_presets.json: %s", exc)
        return {}


def _newest_snapshot_for(name: str) -> dict[str, Any] | None:
    """Return params/seed from the newest snapshot for a generator, or None."""
    from mathviz.preview.snapshots import list_snapshots

    for snap in list_snapshots():
        if snap.generator == name:
            return {"params": snap.params, "seed": snap.seed}
    return None


def resolve_demo_preset(
    name: str,
    presets: dict[str, Any],
    use_presets: bool = True,
) -> tuple[dict[str, Any] | None, int]:
    """Resolve params and seed for a generator with 3-tier fallback.

    Precedence (when use_presets=True):
    1. Snapshot on disk (newest for the generator)
    2. demo_presets.json entry
    3. Generator defaults (params=None, seed=42)

    When use_presets=False, always returns generator defaults.
    Returns (params, seed) where params=None means use generator defaults.
    """
    if not use_presets:
        return None, DEFAULT_SEED

    snapshot = _newest_snapshot_for(name)
    if snapshot is not None:
        logger.info("Using snapshot preset for %s", name)
        return snapshot["params"], snapshot["seed"]

    if name in presets:
        entry = presets[name]
        params = entry.get("params")
        seed = entry.get("seed", DEFAULT_SEED)
        logger.info("Using demo_presets.json for %s", name)
        return params, seed

    logger.info("Using default params for %s", name)
    return None, DEFAULT_SEED


def _build_resolved_config(profile_name: str) -> Any:
    """Load a sampling profile and resolve into pipeline config."""
    profile_cfg = load_sampling_profile(profile_name)
    return resolve_config(object_config=profile_cfg)


def _export_generator(
    name: str,
    output_dir: Path,
    resolved: Any,
    params: dict[str, Any] | None,
    seed: int,
) -> dict[str, Any]:
    """Run pipeline and export mesh.glb, cloud.ply, thumbnail.png for one generator."""
    meta: GeneratorMeta = get_generator_meta(name)
    result = run_pipeline(
        meta.name,
        params=params,
        seed=seed,
        container=resolved.container,
        placement=resolved.placement,
        sampler_config=resolved.sampler_config,
    )
    obj = result.math_object

    data_dir = output_dir / "data" / meta.name
    data_dir.mkdir(parents=True, exist_ok=True)

    if obj.mesh is not None:
        (data_dir / "mesh.glb").write_bytes(mesh_to_glb(obj.mesh))
    if obj.point_cloud is not None:
        (data_dir / "cloud.ply").write_bytes(cloud_to_binary_ply(obj.point_cloud))

    thumb_path = data_dir / "thumbnail.png"
    try:
        thumb_config = RenderConfig(
            width=472, height=472,
            style=THUMBNAIL_VIEW_MODE,
            background_color=THUMBNAIL_BG_COLOR,
            object_color=THUMBNAIL_OBJ_COLOR,
            point_size=2.0,
        )
        render_to_png(obj, thumb_path, config=thumb_config, view="front-right-top")
        logger.info("Generated thumbnail for %s", name)
    except Exception as exc:
        logger.warning("Thumbnail not generated for %s: %s", name, exc)

    return _build_manifest_entry(meta, data_dir, params, seed)


def _build_manifest_entry(
    meta: GeneratorMeta,
    data_dir: Path,
    params: dict[str, Any] | None,
    seed: int,
) -> dict[str, Any]:
    """Build a manifest entry dict for a generator, omitting missing assets."""
    entry: dict[str, Any] = {
        "name": meta.name,
        "category": meta.category,
        "display_name": meta.name.replace("_", " ").title(),
        "description": meta.description,
        "params": params if params is not None else {},
        "seed": seed,
    }
    if (data_dir / "thumbnail.png").is_file():
        entry["thumbnail"] = f"./data/{meta.name}/thumbnail.png"
    else:
        logger.warning("No thumbnail.png produced for %s", meta.name)
    if (data_dir / "mesh.glb").is_file():
        entry["mesh"] = f"./data/{meta.name}/mesh.glb"
    else:
        logger.warning("No mesh.glb produced for %s", meta.name)
    if (data_dir / "cloud.ply").is_file():
        entry["cloud"] = f"./data/{meta.name}/cloud.ply"
    else:
        logger.warning("No cloud.ply produced for %s", meta.name)
    return entry


def _cleanup_partial_data(data_dir: Path) -> None:
    """Remove a partially-created generator data directory."""
    if data_dir.is_dir():
        shutil.rmtree(data_dir)
        logger.info("Cleaned up partial data directory: %s", data_dir)


def _copy_static_assets(output_dir: Path) -> None:
    """Copy demo.html (as index.html), buildbanner.js, and favicon into output."""
    demo_src = STATIC_DIR / "demo.html"
    if demo_src.is_file():
        shutil.copy2(demo_src, output_dir / "index.html")
    else:
        raise FileNotFoundError(
            f"Required demo.html not found at {demo_src}; cannot build demo site"
        )

    for asset_name in _ROOT_ASSETS:
        src = STATIC_DIR / asset_name
        if src.is_file():
            shutil.copy2(src, output_dir / asset_name)
        else:
            logger.warning("Asset not found: %s", src)

    for js_file in sorted(STATIC_DIR.glob("demo-*.js")):
        shutil.copy2(js_file, output_dir / js_file.name)


def _write_manifest(entries: list[dict[str, Any]], output_dir: Path) -> None:
    """Write manifest.json to the output directory."""
    (output_dir / "manifest.json").write_text(
        json.dumps(entries, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def build_demo(
    generator_spec: str,
    output_dir: Path,
    profile: str,
    use_presets: bool = True,
) -> DemoBuildResult:
    """Build the full demo site. Returns a DemoBuildResult with outcome details."""
    output_dir.mkdir(parents=True, exist_ok=True)
    names = resolve_generator_names(generator_spec)

    unknown = validate_generator_names(names)
    if unknown:
        raise ValueError(
            f"Unknown generator(s): {', '.join(unknown)}. "
            f"Use 'all' or check available generators."
        )

    resolved = _build_resolved_config(profile)
    presets = load_presets_file() if use_presets else {}

    logger.info("Building demo for %d generators into %s", len(names), output_dir)

    entries: list[dict[str, Any]] = []
    result = DemoBuildResult()

    for name in names:
        try:
            logger.info("Processing generator: %s", name)
            params, seed = resolve_demo_preset(name, presets, use_presets)
            entry = _export_generator(name, output_dir, resolved, params, seed)
            entries.append(entry)
            result.succeeded += 1
            logger.info("Completed: %s", name)
        except Exception:
            logger.warning("Skipping generator %s due to error", name, exc_info=True)
            result.skipped.append(name)
            _cleanup_partial_data(output_dir / "data" / name)

    if entries:
        _write_manifest(entries, output_dir)
        _copy_static_assets(output_dir)
        result.is_manifest_written = True
    else:
        logger.warning("No generators succeeded; skipping manifest and assets")

    if result.skipped:
        logger.warning(
            "Skipped %d generator(s): %s",
            len(result.skipped),
            ", ".join(result.skipped),
        )

    logger.info(
        "Demo build complete: %d/%d generators exported to %s",
        result.succeeded,
        len(names),
        output_dir,
    )
    return result
