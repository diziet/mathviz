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
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/snapshots", tags=["snapshots"])

_GEOMETRY_MEDIA_TYPES: dict[str, str] = {
    "mesh.glb": "model/gltf-binary",
    "cloud.ply": "application/x-ply",
}


def _snapshot_to_dict(s: Any) -> dict[str, Any]:
    """Convert a SnapshotInfo to a JSON-serializable dict."""
    result: dict[str, Any] = {
        "snapshot_id": s.snapshot_id,
        "generator": s.generator,
        "params": s.params,
        "seed": s.seed,
        "container": s.container,
        "created_at": s.created_at,
        "has_thumbnail": s.has_thumbnail,
        "thumbnail_url": s.thumbnail_url,
        "geometry_files": s.geometry_files,
        "geometry_id": s.geometry_id,
    }
    if s.ui_state is not None:
        result["ui_state"] = s.ui_state
    return result


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
    media = _GEOMETRY_MEDIA_TYPES.get(filename, "application/octet-stream")
    return FileResponse(str(geo_path), media_type=media)


@router.delete("/{snapshot_id}")
def remove_snapshot(snapshot_id: str) -> dict:
    """Delete a snapshot directory."""
    try:
        deleted = delete_snapshot(snapshot_id)
    except OSError as exc:
        logger.error("Failed to delete snapshot %s: %s", snapshot_id, exc)
        raise HTTPException(status_code=500, detail="Failed to delete snapshot.") from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="Snapshot not found.")
    return {"deleted": snapshot_id}
