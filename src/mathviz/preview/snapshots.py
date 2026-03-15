"""Snapshot persistence for the preview server.

Saves geometry, metadata, and optional thumbnail to a local directory
for later comparison or re-use.
"""

import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mathviz.core.math_object import MathObject
from mathviz.preview.lod import cloud_to_binary_ply, mesh_to_glb

logger = logging.getLogger(__name__)

DEFAULT_SNAPSHOTS_DIR = Path.home() / ".mathviz" / "snapshots"
SNAPSHOTS_DIR_ENV_VAR = "MATHVIZ_SNAPSHOTS_DIR"
THUMBNAIL_SIZE = 256


def get_snapshots_dir() -> Path:
    """Return the configured snapshots directory."""
    env_val = os.environ.get(SNAPSHOTS_DIR_ENV_VAR)
    if env_val:
        return Path(env_val)
    return DEFAULT_SNAPSHOTS_DIR


def _generate_snapshot_id(now: datetime) -> str:
    """Generate a timestamp-based snapshot ID with microsecond precision."""
    return now.strftime("%Y%m%d-%H%M%S-%f")


def _write_metadata(
    snapshot_dir: Path,
    generator: str,
    params: dict[str, Any],
    seed: int,
    container: dict[str, Any],
    geometry_id: str,
    created_at: datetime,
) -> None:
    """Write metadata.json to the snapshot directory."""
    metadata = {
        "generator": generator,
        "params": params,
        "seed": seed,
        "container": container,
        "created_at": created_at.isoformat(),
        "geometry_id": geometry_id,
    }
    meta_path = snapshot_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Wrote snapshot metadata to %s", meta_path)


def _save_geometry_files(snapshot_dir: Path, math_object: MathObject) -> None:
    """Save mesh (GLB) and/or point cloud (PLY) to the snapshot directory."""
    if math_object.mesh is not None:
        glb_data = mesh_to_glb(math_object.mesh)
        mesh_path = snapshot_dir / "mesh.glb"
        mesh_path.write_bytes(glb_data)
        logger.info("Saved mesh to %s", mesh_path)

    if math_object.point_cloud is not None:
        ply_data = cloud_to_binary_ply(math_object.point_cloud)
        cloud_path = snapshot_dir / "cloud.ply"
        cloud_path.write_bytes(ply_data)
        logger.info("Saved point cloud to %s", cloud_path)


def _save_thumbnail(snapshot_dir: Path, math_object: MathObject) -> None:
    """Render and save a thumbnail PNG. Skips gracefully if PyVista unavailable."""
    try:
        from mathviz.preview.renderer import RenderConfig, render_to_png
    except ImportError:
        logger.debug("PyVista not available, skipping thumbnail")
        return

    try:
        config = RenderConfig(width=THUMBNAIL_SIZE, height=THUMBNAIL_SIZE)
        thumbnail_path = snapshot_dir / "thumbnail.png"
        render_to_png(math_object, thumbnail_path, config=config)
        logger.info("Saved thumbnail to %s", thumbnail_path)
    except (ImportError, ValueError, OSError, RuntimeError) as exc:
        logger.warning("Could not render thumbnail: %s", exc)


def save_snapshot(
    math_object: MathObject,
    generator: str,
    params: dict[str, Any],
    seed: int,
    container: dict[str, Any],
    geometry_id: str,
) -> tuple[str, Path]:
    """Save a snapshot and return (snapshot_id, snapshot_dir)."""
    now = datetime.now(timezone.utc)
    snapshot_id = _generate_snapshot_id(now)
    snapshots_dir = get_snapshots_dir()
    snapshot_dir = snapshots_dir / snapshot_id

    # Handle rare collision by appending a suffix
    if snapshot_dir.exists():
        suffix = 1
        while (snapshots_dir / f"{snapshot_id}-{suffix}").exists():
            suffix += 1
        snapshot_id = f"{snapshot_id}-{suffix}"
        snapshot_dir = snapshots_dir / snapshot_id

    snapshot_dir.mkdir(parents=True, exist_ok=True)

    try:
        _write_metadata(
            snapshot_dir, generator, params, seed, container, geometry_id, now
        )
        _save_geometry_files(snapshot_dir, math_object)
        _save_thumbnail(snapshot_dir, math_object)
    except Exception:
        shutil.rmtree(snapshot_dir, ignore_errors=True)
        raise

    logger.info("Saved snapshot %s to %s", snapshot_id, snapshot_dir)
    return snapshot_id, snapshot_dir
