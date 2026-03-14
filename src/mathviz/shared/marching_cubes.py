"""Marching cubes wrapper with coordinate rescaling, normals, smoothing, and decimation.

Wraps skimage.measure.marching_cubes with post-processing steps used by
implicit surface generators and 3D fractals.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import trimesh
from skimage.measure import marching_cubes

from mathviz.core.math_object import Mesh

logger = logging.getLogger(__name__)

DEFAULT_ISOLEVEL = 0.0
MIN_FIELD_SIZE = 2


class NoSurfaceError(ValueError):
    """Raised when the isolevel does not intersect the scalar field."""


@dataclass
class SpatialBounds:
    """Axis-aligned bounds for a 3D scalar field."""

    min_corner: tuple[float, float, float]
    max_corner: tuple[float, float, float]

    def extent(self) -> np.ndarray:
        """Return the spatial extent as a (3,) array."""
        return np.array(self.max_corner) - np.array(self.min_corner)


def extract_mesh(
    field: np.ndarray,
    bounds: SpatialBounds,
    isolevel: float = DEFAULT_ISOLEVEL,
    smoothing_iterations: int = 0,
    target_face_count: Optional[int] = None,
) -> Mesh:
    """Extract a triangle mesh from a 3D scalar field via marching cubes.

    Vertices are returned in the coordinate space defined by *bounds*,
    not in voxel indices.

    Raises ``NoSurfaceError`` if the isolevel does not intersect the field.
    """
    _validate_inputs(field, bounds, smoothing_iterations, target_face_count)

    try:
        voxel_verts, faces, _, _ = marching_cubes(field, level=isolevel)
    except ValueError as exc:
        if "surface" in str(exc).lower():
            raise NoSurfaceError(
                f"No isosurface found at isolevel {isolevel}"
            ) from exc
        raise
    vertices = _rescale_to_bounds(voxel_verts, field.shape, bounds)
    faces = faces.astype(np.int64)

    logger.info(
        "Marching cubes: %d vertices, %d faces (isolevel=%.4f)",
        len(vertices),
        len(faces),
        isolevel,
    )

    tm = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    if smoothing_iterations > 0:
        tm = _apply_smoothing(tm, smoothing_iterations)

    if target_face_count is not None:
        tm = _apply_decimation(tm, target_face_count)

    normals = _compute_vertex_normals(tm)

    return Mesh(
        vertices=tm.vertices.astype(np.float64),
        faces=tm.faces.astype(np.int64),
        normals=normals,
    )


def _validate_inputs(
    field: np.ndarray,
    bounds: SpatialBounds,
    smoothing_iterations: int,
    target_face_count: Optional[int],
) -> None:
    """Validate marching cubes inputs."""
    if field.ndim != 3:
        raise ValueError(f"field must be 3D, got {field.ndim}D")

    for dim_size in field.shape:
        if dim_size < MIN_FIELD_SIZE:
            raise ValueError(
                f"Each field dimension must be >= {MIN_FIELD_SIZE}, got shape {field.shape}"
            )

    extent = bounds.extent()
    if np.any(extent <= 0):
        raise ValueError(f"bounds extent must be positive in all axes, got {extent}")

    if smoothing_iterations < 0:
        raise ValueError(
            f"smoothing_iterations must be >= 0, got {smoothing_iterations}"
        )

    if target_face_count is not None and target_face_count < 4:
        raise ValueError(f"target_face_count must be >= 4, got {target_face_count}")


def _rescale_to_bounds(
    voxel_verts: np.ndarray,
    field_shape: tuple[int, ...],
    bounds: SpatialBounds,
) -> np.ndarray:
    """Convert vertices from voxel indices to the spatial coordinate system."""
    grid_max = np.array(field_shape, dtype=np.float64) - 1.0
    normalized = voxel_verts / grid_max
    extent = bounds.extent()
    min_corner = np.array(bounds.min_corner, dtype=np.float64)
    return (normalized * extent + min_corner).astype(np.float64)


def _compute_vertex_normals(tm: trimesh.Trimesh) -> np.ndarray:
    """Compute per-vertex normals from the trimesh object."""
    return np.array(tm.vertex_normals, dtype=np.float64)


def _apply_smoothing(
    tm: trimesh.Trimesh, iterations: int
) -> trimesh.Trimesh:
    """Apply Laplacian smoothing via trimesh (returns a new mesh)."""
    tm = tm.copy()
    trimesh.smoothing.filter_laplacian(tm, iterations=iterations)
    logger.debug("Applied %d Laplacian smoothing iterations", iterations)
    return tm


def _apply_decimation(
    tm: trimesh.Trimesh, target_face_count: int
) -> trimesh.Trimesh:
    """Decimate mesh to approximately the target face count."""
    original_count = len(tm.faces)
    if target_face_count >= original_count:
        logger.debug(
            "Skipping decimation: target %d >= current %d",
            target_face_count,
            original_count,
        )
        return tm

    decimated = tm.simplify_quadric_decimation(face_count=target_face_count)
    logger.info(
        "Decimated %d -> %d faces (target %d)",
        original_count,
        len(decimated.faces),
        target_face_count,
    )
    return decimated
