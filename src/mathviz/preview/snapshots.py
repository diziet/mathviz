"""Snapshot persistence for the preview server.

Saves geometry, metadata, and optional thumbnail to a local directory
for later comparison or re-use. Supports listing, loading, and deleting
saved snapshots.
"""

import json
import logging
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mathviz.core.math_object import MathObject
from mathviz.preview.lod import cloud_to_binary_ply, mesh_to_glb

logger = logging.getLogger(__name__)

DEFAULT_SNAPSHOTS_DIR = Path.home() / ".mathviz" / "snapshots"
SNAPSHOTS_DIR_ENV_VAR = "MATHVIZ_SNAPSHOTS_DIR"
THUMBNAIL_SIZE = 256
GEOMETRY_FILES = {"mesh.glb", "cloud.ply"}


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
    ui_state: dict[str, Any] | None = None,
) -> None:
    """Write metadata.json to the snapshot directory."""
    metadata: dict[str, Any] = {
        "generator": generator,
        "params": params,
        "seed": seed,
        "container": container,
        "created_at": created_at.isoformat(),
        "geometry_id": geometry_id,
    }
    if ui_state is not None:
        metadata["ui_state"] = ui_state
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


def _save_thumbnail(snapshot_dir: Path, png_data: bytes) -> None:
    """Save pre-validated PNG bytes as thumbnail.png."""
    thumbnail_path = snapshot_dir / "thumbnail.png"
    thumbnail_path.write_bytes(png_data)
    logger.info("Saved thumbnail to %s", thumbnail_path)


def save_snapshot(
    math_object: MathObject,
    generator: str,
    params: dict[str, Any],
    seed: int,
    container: dict[str, Any],
    geometry_id: str,
    thumbnail_png: bytes | None = None,
    ui_state: dict[str, Any] | None = None,
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
            snapshot_dir, generator, params, seed, container, geometry_id, now,
            ui_state=ui_state,
        )
        _save_geometry_files(snapshot_dir, math_object)
        if thumbnail_png is not None:
            _save_thumbnail(snapshot_dir, thumbnail_png)
    except Exception:
        shutil.rmtree(snapshot_dir, ignore_errors=True)
        raise

    logger.info("Saved snapshot %s to %s", snapshot_id, snapshot_dir)
    return snapshot_id, snapshot_dir


@dataclass
class SnapshotInfo:
    """Summary of a saved snapshot for gallery display."""

    snapshot_id: str
    generator: str
    params: dict[str, Any]
    seed: int
    container: dict[str, Any]
    created_at: str
    has_thumbnail: bool
    thumbnail_url: str | None
    geometry_files: list[str]
    geometry_id: str
    ui_state: dict[str, Any] | None = None


def _read_snapshot_metadata(snapshot_dir: Path) -> dict[str, Any] | None:
    """Read and parse metadata.json from a snapshot directory."""
    meta_path = snapshot_dir / "metadata.json"
    if not meta_path.is_file():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read metadata from %s: %s", meta_path, exc)
        return None


_REQUIRED_METADATA_KEYS = {"generator", "params", "seed", "container", "created_at", "geometry_id"}


def _snapshot_info_from_dir(snapshot_dir: Path) -> SnapshotInfo | None:
    """Build a SnapshotInfo from a snapshot directory, or None if invalid."""
    metadata = _read_snapshot_metadata(snapshot_dir)
    if metadata is None:
        return None

    missing = _REQUIRED_METADATA_KEYS - metadata.keys()
    if missing:
        logger.warning("Snapshot %s missing metadata keys: %s", snapshot_dir.name, missing)
        return None

    snapshot_id = snapshot_dir.name
    has_thumb = (snapshot_dir / "thumbnail.png").is_file()
    thumb_url = f"/api/snapshots/{snapshot_id}/thumbnail" if has_thumb else None
    geo_files = [f.name for f in snapshot_dir.iterdir() if f.name in GEOMETRY_FILES]

    return SnapshotInfo(
        snapshot_id=snapshot_id,
        generator=metadata["generator"],
        params=metadata["params"],
        seed=metadata["seed"],
        container=metadata["container"],
        created_at=metadata["created_at"],
        has_thumbnail=has_thumb,
        thumbnail_url=thumb_url,
        geometry_files=geo_files,
        geometry_id=metadata["geometry_id"],
        ui_state=metadata.get("ui_state"),
    )


def list_snapshots() -> list[SnapshotInfo]:
    """Return all saved snapshots, sorted newest first."""
    snapshots_dir = get_snapshots_dir()
    if not snapshots_dir.is_dir():
        return []

    results: list[SnapshotInfo] = []
    for entry in snapshots_dir.iterdir():
        if not entry.is_dir():
            continue
        info = _snapshot_info_from_dir(entry)
        if info is not None:
            results.append(info)

    results.sort(key=lambda s: s.created_at, reverse=True)
    return results


def get_snapshot_dir(snapshot_id: str) -> Path | None:
    """Return the path for a snapshot, or None if it doesn't exist or is invalid."""
    snapshots_dir = get_snapshots_dir()
    snapshot_dir = (snapshots_dir / snapshot_id).resolve()
    if not snapshot_dir.is_relative_to(snapshots_dir.resolve()):
        return None
    if not snapshot_dir.is_dir():
        return None
    return snapshot_dir


def delete_snapshot(snapshot_id: str) -> bool:
    """Delete a snapshot directory. Returns True if deleted, False if not found.

    Raises OSError if the directory cannot be removed (e.g. permission error).
    """
    snapshot_dir = get_snapshot_dir(snapshot_id)
    if snapshot_dir is None:
        return False
    shutil.rmtree(snapshot_dir)
    logger.info("Deleted snapshot %s", snapshot_id)
    return True
