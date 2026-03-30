"""Demo site builder — package-importable entry point for export-demo CLI.

Provides :func:`build_demo` which orchestrates the full static demo build:
resolve generators, run pipeline, export assets, write manifest.
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Any

from mathviz.core.config import load_sampling_profile, resolve_config
from mathviz.core.generator import GeneratorMeta, get_generator_meta, list_generators
from mathviz.pipeline.runner import run as run_pipeline
from mathviz.preview.lod import cloud_to_binary_ply, mesh_to_glb
from mathviz.preview.thumbnails import generate_thumbnail
from PIL import Image

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
DEFAULT_SEED = 42
THUMBNAIL_VIEW_MODE = "vertex"
_ROOT_ASSETS = ("buildbanner.js", "favicon.ico")


def resolve_generator_names(spec: str) -> list[str]:
    """Resolve generator spec to a sorted list of canonical names."""
    if spec.strip().lower() == "all":
        return sorted(m.name for m in list_generators())
    return [name.strip() for name in spec.split(",") if name.strip()]


def _build_resolved_config(profile_name: str) -> Any:
    """Load a sampling profile and resolve into pipeline config."""
    profile_cfg = load_sampling_profile(profile_name)
    return resolve_config(object_config=profile_cfg)


def _export_generator(
    name: str,
    output_dir: Path,
    resolved: Any,
) -> dict[str, Any]:
    """Run pipeline and export mesh.glb, cloud.ply, thumbnail.png for one generator."""
    meta: GeneratorMeta = get_generator_meta(name)
    result = run_pipeline(
        meta.name,
        params=None,
        seed=DEFAULT_SEED,
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

    webp_path = generate_thumbnail(name, THUMBNAIL_VIEW_MODE)
    Image.open(webp_path).save(data_dir / "thumbnail.png", "PNG")

    return _build_manifest_entry(meta, data_dir)


def _build_manifest_entry(meta: GeneratorMeta, data_dir: Path) -> dict[str, Any]:
    """Build a manifest entry dict for a generator, omitting missing assets."""
    entry: dict[str, Any] = {
        "name": meta.name,
        "category": meta.category,
        "display_name": meta.name.replace("_", " ").title(),
        "description": meta.description,
        "thumbnail": f"./data/{meta.name}/thumbnail.png",
    }
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
        logger.warning("demo.html not found at %s", demo_src)

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


def build_demo(generator_spec: str, output_dir: Path, profile: str) -> int:
    """Build the full demo site. Returns count of successful generators."""
    output_dir.mkdir(parents=True, exist_ok=True)
    names = resolve_generator_names(generator_spec)
    resolved = _build_resolved_config(profile)

    logger.info("Building demo for %d generators into %s", len(names), output_dir)

    entries: list[dict[str, Any]] = []
    succeeded = 0

    for name in names:
        try:
            logger.info("Processing generator: %s", name)
            entry = _export_generator(name, output_dir, resolved)
            entries.append(entry)
            succeeded += 1
            logger.info("Completed: %s", name)
        except (RuntimeError, ValueError, OSError, IOError, ImportError):
            logger.warning("Skipping generator %s due to error", name, exc_info=True)
            _cleanup_partial_data(output_dir / "data" / name)

    if entries:
        _write_manifest(entries, output_dir)
        _copy_static_assets(output_dir)
    else:
        logger.warning("No generators succeeded; skipping manifest and assets")

    logger.info(
        "Demo build complete: %d/%d generators exported to %s",
        succeeded,
        len(names),
        output_dir,
    )
    return succeeded
