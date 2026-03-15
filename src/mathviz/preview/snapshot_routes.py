"""Snapshot browse, serve, and delete API routes."""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response

from mathviz.preview.snapshots import (
    GEOMETRY_FILES,
    delete_snapshot,
    get_snapshot_dir,
    list_snapshots,
    save_snapshot,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/snapshots", tags=["snapshots"])


def _snapshot_to_dict(s: Any) -> dict[str, Any]:
    """Convert a SnapshotInfo to a JSON-serializable dict."""
    return {
        "snapshot_id": s.snapshot_id,
        "generator": s.generator,
        "params": s.params,
        "seed": s.seed,
        "container": s.container,
        "created_at": s.created_at,
        "has_thumbnail": s.has_thumbnail,
        "thumbnail_url": s.thumbnail_url,
        "geometry_files": s.geometry_files,
    }


@router.get("")
def list_all_snapshots() -> list[dict]:
    """Return all saved snapshots, sorted newest first."""
    return [_snapshot_to_dict(s) for s in list_snapshots()]


@router.get("/{snapshot_id}/thumbnail")
def get_snapshot_thumbnail(snapshot_id: str) -> Response:
    """Serve the thumbnail PNG for a snapshot."""
    snap_dir = get_snapshot_dir(snapshot_id)
    if snap_dir is None:
        raise HTTPException(status_code=404, detail="Snapshot not found.")
    thumb = snap_dir / "thumbnail.png"
    if not thumb.is_file():
        raise HTTPException(status_code=404, detail="No thumbnail for this snapshot.")
    return FileResponse(str(thumb), media_type="image/png")


@router.get("/{snapshot_id}/geometry/{filename}")
def get_snapshot_geometry(snapshot_id: str, filename: str) -> Response:
    """Serve a saved geometry file from a snapshot."""
    if filename not in GEOMETRY_FILES:
        raise HTTPException(status_code=400, detail=f"Invalid geometry file: {filename}")
    snap_dir = get_snapshot_dir(snapshot_id)
    if snap_dir is None:
        raise HTTPException(status_code=404, detail="Snapshot not found.")
    geo_path = snap_dir / filename
    if not geo_path.is_file():
        raise HTTPException(status_code=404, detail=f"Geometry file {filename} not found.")
    media = "model/gltf-binary" if filename.endswith(".glb") else "application/x-ply"
    return FileResponse(str(geo_path), media_type=media)


@router.delete("/{snapshot_id}")
def remove_snapshot(snapshot_id: str) -> dict:
    """Delete a snapshot directory."""
    if not delete_snapshot(snapshot_id):
        raise HTTPException(status_code=404, detail="Snapshot not found.")
    return {"deleted": snapshot_id}
