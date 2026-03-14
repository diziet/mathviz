"""Geometry loader: reads STL, OBJ, and PLY files into MathObject."""

import logging
from pathlib import Path

import numpy as np
import trimesh

from mathviz.core.math_object import CoordSpace, MathObject, Mesh, PointCloud
from mathviz.pipeline.metadata import resolve_format

logger = logging.getLogger(__name__)

SUPPORTED_MESH_FORMATS = {"stl", "obj", "ply"}
SUPPORTED_CLOUD_FORMATS = {"ply", "xyz"}


class GeometryLoadError(ValueError):
    """Raised when geometry cannot be loaded from a file."""


def load_geometry(path: Path, fmt: str | None = None) -> MathObject:
    """Load geometry from a file into a MathObject.

    Attempts mesh loading first via trimesh. If the file is a PLY point cloud
    (no faces), falls back to point cloud loading.

    Returns a MathObject in PHYSICAL coordinate space (file units assumed physical).
    """
    resolved_path = Path(path)
    if not resolved_path.exists():
        raise GeometryLoadError(f"File not found: {resolved_path}")

    resolved_fmt = resolve_format(resolved_path, fmt)

    if resolved_fmt in SUPPORTED_MESH_FORMATS:
        return _try_load_mesh_or_cloud(resolved_path, resolved_fmt)

    if resolved_fmt in SUPPORTED_CLOUD_FORMATS:
        return _load_cloud(resolved_path, resolved_fmt)

    raise GeometryLoadError(
        f"Unsupported format '{resolved_fmt}'. "
        f"Supported: {sorted(SUPPORTED_MESH_FORMATS | SUPPORTED_CLOUD_FORMATS)}"
    )


def _try_load_mesh_or_cloud(path: Path, fmt: str) -> MathObject:
    """Load a file as mesh; if it has no faces, treat as point cloud."""
    loaded = trimesh.load(str(path), file_type=fmt, process=False)

    if isinstance(loaded, trimesh.PointCloud):
        points = np.asarray(loaded.vertices, dtype=np.float64)
        return _build_cloud_object(points, path)

    if isinstance(loaded, trimesh.Trimesh):
        if len(loaded.faces) == 0:
            points = np.asarray(loaded.vertices, dtype=np.float64)
            return _build_cloud_object(points, path)

        return _build_mesh_object(loaded, path)

    raise GeometryLoadError(
        f"Unexpected trimesh type {type(loaded).__name__} from '{path}'"
    )


def _load_cloud(path: Path, fmt: str) -> MathObject:
    """Load a point cloud file (PLY or XYZ)."""
    if fmt == "xyz":
        points = np.loadtxt(str(path), dtype=np.float64)
        if points.size == 0:
            raise GeometryLoadError(f"XYZ file is empty: {path}")
        if points.ndim == 1:
            points = points.reshape(1, -1)
        if points.shape[1] < 3:
            raise GeometryLoadError(f"XYZ file has {points.shape[1]} columns, expected >= 3")
        points = points[:, :3]
        return _build_cloud_object(points, path)

    # PLY cloud — use trimesh
    return _try_load_mesh_or_cloud(path, fmt)


def _build_mesh_object(tm: trimesh.Trimesh, path: Path) -> MathObject:
    """Build a MathObject from a trimesh.Trimesh."""
    vertices = np.asarray(tm.vertices, dtype=np.float64)
    faces = np.asarray(tm.faces, dtype=np.int64)
    logger.info("Loaded mesh from %s: %d vertices, %d faces", path, len(vertices), len(faces))
    return MathObject(
        mesh=Mesh(vertices=vertices, faces=faces),
        generator_name=path.stem,
        coord_space=CoordSpace.PHYSICAL,
    )


def _build_cloud_object(points: np.ndarray, path: Path) -> MathObject:
    """Build a MathObject from a point array."""
    points = np.asarray(points, dtype=np.float64)
    logger.info("Loaded point cloud from %s: %d points", path, len(points))
    return MathObject(
        point_cloud=PointCloud(points=points),
        generator_name=path.stem,
        coord_space=CoordSpace.PHYSICAL,
    )


def has_mesh(obj: MathObject) -> bool:
    """Check if a MathObject has mesh geometry."""
    return obj.mesh is not None


def has_point_cloud(obj: MathObject) -> bool:
    """Check if a MathObject has point cloud geometry."""
    return obj.point_cloud is not None
