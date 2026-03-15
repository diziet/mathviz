"""Shared mesh face-building utilities for parametric surface generators."""

import numpy as np
from scipy.spatial import cKDTree

from mathviz.core.math_object import BoundingBox

_BBOX_RELATIVE_PADDING = 0.02
_BBOX_ABSOLUTE_PADDING = 1e-6
DEFAULT_SEPARATION_EPSILON = 0.005
COINCIDENCE_THRESHOLD = 1e-8
_MAX_SEPARATION_EPSILON = 1.0


def compute_padded_bounding_box(vertices: np.ndarray) -> BoundingBox:
    """Compute axis-aligned bounding box from vertices with padding."""
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    padding = (vmax - vmin) * _BBOX_RELATIVE_PADDING + _BBOX_ABSOLUTE_PADDING
    return BoundingBox(
        min_corner=tuple(vmin - padding),
        max_corner=tuple(vmax + padding),
    )


def _compute_vertex_normals(
    vertices: np.ndarray, faces: np.ndarray,
) -> np.ndarray:
    """Compute per-vertex normals by averaging adjacent face normals."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)

    vertex_normals = np.zeros_like(vertices)
    for col in range(3):
        np.add.at(vertex_normals, faces[:, col], face_normals)

    lengths = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    lengths = np.where(lengths < 1e-12, 1.0, lengths)
    return vertex_normals / lengths


def validate_separation_epsilon(epsilon: float) -> None:
    """Validate separation_epsilon parameter."""
    if epsilon < 0:
        raise ValueError(f"separation_epsilon must be >= 0, got {epsilon}")
    if epsilon > _MAX_SEPARATION_EPSILON:
        raise ValueError(
            f"separation_epsilon must be <= {_MAX_SEPARATION_EPSILON}, "
            f"got {epsilon}"
        )


def separate_coincident_vertices(
    vertices: np.ndarray,
    faces: np.ndarray,
    epsilon: float = DEFAULT_SEPARATION_EPSILON,
) -> np.ndarray:
    """Offset one vertex in each coincident pair along its face normal.

    Finds truly coincident vertex pairs (distance < COINCIDENCE_THRESHOLD)
    and displaces the higher-indexed vertex by ``epsilon`` along its
    averaged face normal. Returns a copy of *vertices*; *faces* is unchanged.
    """
    if epsilon == 0:
        return vertices.copy()

    tree = cKDTree(vertices)
    pairs = tree.query_pairs(COINCIDENCE_THRESHOLD, output_type="ndarray")
    if len(pairs) == 0:
        return vertices.copy()

    normals = _compute_vertex_normals(vertices, faces)
    result = vertices.copy()

    displaced: set[int] = set()
    for idx_a, idx_b in pairs:
        target = max(idx_a, idx_b)
        if target in displaced:
            continue
        displaced.add(target)
        normal = normals[target]
        if np.linalg.norm(normal) < 1e-10:
            normal = np.array([0.0, 0.0, 1.0])
        result[target] += epsilon * normal

    return result


def build_wrapped_grid_faces(n_u: int, n_v: int) -> np.ndarray:
    """Build triangle faces for a grid periodic in both u and v."""
    rows = np.arange(n_u)
    cols = np.arange(n_v)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    rr, cc = rr.ravel(), cc.ravel()

    i00 = rr * n_v + cc
    i10 = ((rr + 1) % n_u) * n_v + cc
    i01 = rr * n_v + ((cc + 1) % n_v)
    i11 = ((rr + 1) % n_u) * n_v + ((cc + 1) % n_v)

    tri1 = np.stack([i00, i10, i11], axis=-1)
    tri2 = np.stack([i00, i11, i01], axis=-1)
    return np.concatenate([tri1, tri2], axis=0).astype(np.int64)


def _build_v_wrapped_interior_faces(
    n_u_rows: int, n_v: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build interior triangle faces for rows 0..n_u_rows-1 with v-wrapping."""
    rows = np.arange(n_u_rows)
    cols = np.arange(n_v)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    rr, cc = rr.ravel(), cc.ravel()

    i00 = rr * n_v + cc
    i10 = (rr + 1) * n_v + cc
    i01 = rr * n_v + ((cc + 1) % n_v)
    i11 = (rr + 1) * n_v + ((cc + 1) % n_v)

    tri1 = np.stack([i00, i10, i11], axis=-1)
    tri2 = np.stack([i00, i11, i01], axis=-1)
    return tri1, tri2


def build_klein_wrapped_faces(n_u: int, n_v: int) -> np.ndarray:
    """Build triangle faces for a Klein bottle grid with shifted u-seam.

    The figure-8 Klein bottle cross-section is reflected (v → -v) after a
    full u-period. At the u-seam (row n_u-1 wrapping to row 0), vertex
    (n_u-1, v) connects to (0, (n_v - v) % n_v) instead of (0, v).
    """
    interior_tri1, interior_tri2 = _build_v_wrapped_interior_faces(
        n_u - 1, n_v,
    )

    # Seam faces: row n_u-1 wrapping to row 0 with v-reflection
    # The figure-8 immersion satisfies f(0, v) = f(2π, -v), so vertex
    # (n_u-1, v) is spatially adjacent to (0, (n_v - v) % n_v).
    sc = np.arange(n_v)
    s00 = (n_u - 1) * n_v + sc
    s01 = (n_u - 1) * n_v + ((sc + 1) % n_v)
    # Row 0 reflected: v maps to -v, and v+1 maps to -(v+1).
    # Since v increases clockwise at the seam but counter-clockwise in
    # row 0 after reflection, we swap the winding to keep faces consistent.
    s10 = (n_v - sc) % n_v  # row 0, reflected v
    s11 = (n_v - sc - 1) % n_v  # row 0, reflected v+1

    seam_tri1 = np.stack([s00, s10, s11], axis=-1)
    seam_tri2 = np.stack([s00, s11, s01], axis=-1)

    return np.concatenate(
        [interior_tri1, interior_tri2, seam_tri1, seam_tri2], axis=0,
    ).astype(np.int64)


def build_open_grid_faces(n_u: int, n_v: int) -> np.ndarray:
    """Build triangle faces for an open grid (no wrapping)."""
    rows = np.arange(n_u - 1)
    cols = np.arange(n_v - 1)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    rr, cc = rr.ravel(), cc.ravel()

    i00 = rr * n_v + cc
    i10 = (rr + 1) * n_v + cc
    i01 = rr * n_v + (cc + 1)
    i11 = (rr + 1) * n_v + (cc + 1)

    tri1 = np.stack([i00, i10, i11], axis=-1)
    tri2 = np.stack([i00, i11, i01], axis=-1)
    return np.concatenate([tri1, tri2], axis=0).astype(np.int64)


def build_mixed_grid_faces(
    n_u: int, n_v: int, wrap_u: bool, wrap_v: bool,
) -> np.ndarray:
    """Build triangle faces for a grid with independent per-axis wrapping."""
    u_range = np.arange(n_u) if wrap_u else np.arange(n_u - 1)
    v_range = np.arange(n_v) if wrap_v else np.arange(n_v - 1)
    rr, cc = np.meshgrid(u_range, v_range, indexing="ij")
    rr, cc = rr.ravel(), cc.ravel()

    next_r = ((rr + 1) % n_u) if wrap_u else (rr + 1)
    next_c = ((cc + 1) % n_v) if wrap_v else (cc + 1)

    i00 = rr * n_v + cc
    i10 = next_r * n_v + cc
    i01 = rr * n_v + next_c
    i11 = next_r * n_v + next_c

    tri1 = np.stack([i00, i10, i11], axis=-1)
    tri2 = np.stack([i00, i11, i01], axis=-1)
    return np.concatenate([tri1, tri2], axis=0).astype(np.int64)


def build_sphere_faces(n_lat: int, n_lon: int) -> np.ndarray:
    """Build triangle faces for a sphere-like grid with poles.

    The grid has n_lat latitude rows (excluding poles) and n_lon longitude
    columns. Two extra vertices are appended: south pole at index n_lat*n_lon,
    north pole at n_lat*n_lon + 1.
    """
    n_body = n_lat * n_lon
    south_pole = n_body
    north_pole = n_body + 1

    # South pole fan
    j_range = np.arange(n_lon)
    j_next = (j_range + 1) % n_lon
    south_fan = np.column_stack([
        np.full(n_lon, south_pole), j_range, j_next,
    ])

    # Body quads (vectorized)
    rows = np.arange(n_lat - 1)
    cols = np.arange(n_lon)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    rr, cc = rr.ravel(), cc.ravel()
    cc_next = (cc + 1) % n_lon

    i00 = rr * n_lon + cc
    i10 = (rr + 1) * n_lon + cc
    i01 = rr * n_lon + cc_next
    i11 = (rr + 1) * n_lon + cc_next

    body_tri1 = np.column_stack([i00, i10, i11])
    body_tri2 = np.column_stack([i00, i11, i01])

    # North pole fan
    last_row = (n_lat - 1) * n_lon
    north_fan = np.column_stack([
        last_row + j_range, np.full(n_lon, north_pole), last_row + j_next,
    ])

    return np.concatenate([
        south_fan, body_tri1, body_tri2, north_fan,
    ], axis=0).astype(np.int64)
