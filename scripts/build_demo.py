#!/usr/bin/env python3
"""Build a self-contained static demo site from generator pipeline output.

Produces a deployable directory with index.html, geometry files, thumbnails,
and manifest.json suitable for Cloudflare Pages or any static host.

Usage:
    python scripts/build_demo.py                         # all generators
    python scripts/build_demo.py --generators lorenz,gyroid
    python scripts/build_demo.py --output build/ --profile preview
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

from mathviz.core.config import load_sampling_profile, resolve_config
from mathviz.core.container import Container, PlacementPolicy
from mathviz.core.generator import GeneratorMeta, get_generator_meta, list_generators
from mathviz.pipeline.runner import run as run_pipeline
from mathviz.preview.lod import cloud_to_binary_ply, mesh_to_glb
from mathviz.preview.thumbnails import generate_thumbnail

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent.parent / "src" / "mathviz" / "static"
DEFAULT_OUTPUT = Path("dist")
DEFAULT_PROFILE = "preview"
DEFAULT_SEED = 42
THUMBNAIL_VIEW_MODE = "vertex"

# Files to copy into the output root
_ROOT_ASSETS = ("buildbanner.js", "favicon.ico")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for build_demo."""
    parser = argparse.ArgumentParser(
        description="Build a self-contained static demo site"
    )
    parser.add_argument(
        "--generators",
        default="all",
        help="Comma-separated generator names, or 'all' (default: all)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--profile",
        default=DEFAULT_PROFILE,
        help=f"Sampling profile name (default: {DEFAULT_PROFILE})",
    )
    return parser.parse_args(argv)


def resolve_generator_names(spec: str) -> list[str]:
    """Resolve generator spec to a sorted list of canonical names."""
    if spec.strip().lower() == "all":
        return sorted(m.name for m in list_generators())
    return [name.strip() for name in spec.split(",") if name.strip()]


def _build_resolved_config(profile_name: str) -> Any:
    """Load a sampling profile and resolve into pipeline config."""
    profile_cfg = load_sampling_profile(profile_name)
    return resolve_config(object_config=profile_cfg)


def export_generator(
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

    _export_glb(obj, data_dir)
    _export_ply(obj, data_dir)
    _export_thumbnail(meta.name, data_dir)

    return _build_manifest_entry(meta, data_dir)


def _export_glb(obj: Any, data_dir: Path) -> None:
    """Export mesh as GLB if available."""
    if obj.mesh is not None:
        glb_bytes = mesh_to_glb(obj.mesh)
        (data_dir / "mesh.glb").write_bytes(glb_bytes)


def _export_ply(obj: Any, data_dir: Path) -> None:
    """Export point cloud as binary PLY if available."""
    if obj.point_cloud is not None:
        ply_bytes = cloud_to_binary_ply(obj.point_cloud)
        (data_dir / "cloud.ply").write_bytes(ply_bytes)


def _export_thumbnail(name: str, data_dir: Path) -> None:
    """Generate and copy thumbnail PNG into data directory."""
    from PIL import Image

    webp_path = generate_thumbnail(name, THUMBNAIL_VIEW_MODE)
    png_path = data_dir / "thumbnail.png"
    Image.open(webp_path).save(png_path, "PNG")


def _build_manifest_entry(meta: GeneratorMeta, data_dir: Path) -> dict[str, Any]:
    """Build a manifest entry dict for a generator."""
    return {
        "name": meta.name,
        "category": meta.category,
        "display_name": meta.name.replace("_", " ").title(),
        "description": meta.description,
        "thumbnail": f"./data/{meta.name}/thumbnail.png",
        "mesh": f"./data/{meta.name}/mesh.glb",
        "cloud": f"./data/{meta.name}/cloud.ply",
    }


def copy_static_assets(output_dir: Path) -> None:
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

    _copy_js_modules(output_dir)


def _copy_js_modules(output_dir: Path) -> None:
    """Copy demo-*.js modules referenced by demo.html."""
    for js_file in sorted(STATIC_DIR.glob("demo-*.js")):
        shutil.copy2(js_file, output_dir / js_file.name)


def write_manifest(entries: list[dict[str, Any]], output_dir: Path) -> None:
    """Write manifest.json to the output directory."""
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(entries, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )


def build_demo(
    generator_spec: str,
    output_dir: Path,
    profile: str,
) -> int:
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
            entry = export_generator(name, output_dir, resolved)
            entries.append(entry)
            succeeded += 1
            logger.info("Completed: %s", name)
        except Exception:
            logger.warning("Skipping generator %s due to error", name, exc_info=True)

    write_manifest(entries, output_dir)
    copy_static_assets(output_dir)

    logger.info(
        "Demo build complete: %d/%d generators exported to %s",
        succeeded,
        len(names),
        output_dir,
    )
    return succeeded


def main(argv: list[str] | None = None) -> None:
    """Entry point for the build_demo script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    args = parse_args(argv)
    succeeded = build_demo(args.generators, args.output, args.profile)
    if succeeded == 0:
        logger.error("No generators exported successfully")
        sys.exit(1)


if __name__ == "__main__":
    main()
