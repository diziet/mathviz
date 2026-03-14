"""Tube thickening: extrude a circular cross-section along a curve.

Uses the parallel transport (Bishop) frame to avoid the twisting
artifacts of Frenet-Serret frames. Handles closed curves by blending
the last frame back into the first to eliminate seam gaps.
"""

import logging
import warnings

import numpy as np

from mathviz.core.math_object import Curve, Mesh

logger = logging.getLogger(__name__)

DEFAULT_SIDES = 16
MIN_CURVE_POINTS = 2
MIN_CLOSED_CURVE_POINTS = 3
EPSILON = 1e-12


def thicken_curve(
    curve: Curve,
    radius: float,
    sides: int = DEFAULT_SIDES,
) -> Mesh:
    """Convert a curve into a tube mesh via parallel transport frames.

    For closed curves the tube is seamless; for open curves, flat caps
    are added at both ends.
    """
    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}")
    if sides < 3:
        raise ValueError(f"sides must be >= 3, got {sides}")

    min_required = MIN_CLOSED_CURVE_POINTS if curve.closed else MIN_CURVE_POINTS
    if len(curve.points) < min_required:
        raise ValueError(
            f"Curve needs >= {min_required} points, got {len(curve.points)}"
        )

    points = curve.points
    closed = curve.closed
    _warn_self_intersection(points, radius)

    tangents = _compute_tangents(points, closed)
    normals, binormals = _compute_bishop_frames(tangents, closed)

    vertices = _build_ring_vertices(points, normals, binormals, radius, sides)
    faces = _build_tube_faces(len(points), sides, closed)

    if not closed:
        cap_verts, cap_faces = _build_caps(
            points, sides, len(vertices)
        )
        vertices = np.concatenate([vertices, cap_verts], axis=0)
        faces = np.concatenate([faces, cap_faces], axis=0)

    return Mesh(
        vertices=vertices.astype(np.float64),
        faces=faces.astype(np.int64),
    )


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    """Compute angle in radians between two unit vectors."""
    return float(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0)))


def _compute_tangents(points: np.ndarray, closed: bool) -> np.ndarray:
    """Compute unit tangent vectors at each curve point."""
    tangents = np.empty_like(points)

    if closed:
        tangents = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
    else:
        tangents[0] = points[1] - points[0]
        tangents[-1] = points[-1] - points[-2]
        tangents[1:-1] = points[2:] - points[:-2]

    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms = np.maximum(norms, EPSILON)
    return tangents / norms


def _initial_normal(tangent: np.ndarray) -> np.ndarray:
    """Find an initial normal vector perpendicular to the tangent."""
    candidates = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    dots = np.abs(candidates @ tangent)
    least_parallel = candidates[np.argmin(dots)]
    normal = np.cross(tangent, least_parallel)
    return normal / np.linalg.norm(normal)


def _compute_bishop_frames(
    tangents: np.ndarray, closed: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Build parallel-transport frames along the curve."""
    n = len(tangents)
    normals = np.empty_like(tangents)
    binormals = np.empty_like(tangents)

    normals[0] = _initial_normal(tangents[0])
    binormals[0] = np.cross(tangents[0], normals[0])

    for i in range(1, n):
        normals[i] = _parallel_transport(
            normals[i - 1], tangents[i - 1], tangents[i]
        )
        binormals[i] = np.cross(tangents[i], normals[i])

    if closed:
        normals, binormals = _close_frames(
            normals, binormals, tangents
        )

    return normals, binormals


def _parallel_transport(
    prev_normal: np.ndarray,
    prev_tangent: np.ndarray,
    curr_tangent: np.ndarray,
) -> np.ndarray:
    """Transport a normal from one tangent to the next via rotation."""
    axis = np.cross(prev_tangent, curr_tangent)
    axis_len = np.linalg.norm(axis)
    cos_angle = np.clip(np.dot(prev_tangent, curr_tangent), -1.0, 1.0)

    if axis_len < EPSILON:
        if cos_angle < -0.99:
            # Anti-parallel (hairpin): rotate 180° around any perpendicular axis
            perp = _initial_normal(prev_tangent)
            rotated = _rotate_around_axis(prev_normal, perp, np.pi)
        else:
            # Nearly parallel: no rotation needed
            return prev_normal.copy()
    else:
        axis /= axis_len
        angle = np.arccos(cos_angle)
        rotated = _rotate_around_axis(prev_normal, axis, angle)

    # Re-orthogonalize against current tangent
    rotated -= np.dot(rotated, curr_tangent) * curr_tangent
    norm = np.linalg.norm(rotated)
    if norm < EPSILON:
        # Fallback: find a fresh perpendicular to current tangent
        return _initial_normal(curr_tangent)
    return rotated / norm


def _rotate_around_axis(
    vec: np.ndarray, axis: np.ndarray, angle: float
) -> np.ndarray:
    """Rodrigues' rotation formula."""
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    return (
        vec * cos_a
        + np.cross(axis, vec) * sin_a
        + axis * np.dot(axis, vec) * (1.0 - cos_a)
    )


def _close_frames(
    normals: np.ndarray,
    binormals: np.ndarray,
    tangents: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Blend frames for closed curves so last frame matches first."""
    n = len(normals)
    # Transport last normal forward to see the mismatch angle
    transported = _parallel_transport(normals[-1], tangents[-1], tangents[0])
    correction = _angle_between(transported, normals[0])

    # Determine sign of correction
    cross = np.cross(transported, normals[0])
    if np.dot(cross, tangents[0]) < 0:
        correction = -correction

    # Distribute correction evenly (vectorized Rodrigues')
    fracs = np.arange(n, dtype=np.float64) / n
    angles = fracs * correction
    cos_a = np.cos(angles)[:, np.newaxis]
    sin_a = np.sin(angles)[:, np.newaxis]
    dot_at = np.sum(tangents * normals, axis=1, keepdims=True)
    cross_tn = np.cross(tangents, normals)

    normals = normals * cos_a + cross_tn * sin_a + tangents * dot_at * (1.0 - cos_a)
    binormals = np.cross(tangents, normals)

    return normals, binormals


def _build_ring_vertices(
    points: np.ndarray,
    normals: np.ndarray,
    binormals: np.ndarray,
    radius: float,
    sides: int,
) -> np.ndarray:
    """Place a ring of vertices at each curve point."""
    angles = np.linspace(0, 2.0 * np.pi, sides, endpoint=False)
    cos_a = np.cos(angles)  # (sides,)
    sin_a = np.sin(angles)  # (sides,)

    # Broadcast: (n, 1, 3) + (n, sides, 3)
    offset_n = normals[:, np.newaxis, :] * (radius * cos_a)[np.newaxis, :, np.newaxis]
    offset_b = binormals[:, np.newaxis, :] * (radius * sin_a)[np.newaxis, :, np.newaxis]
    ring_verts = points[:, np.newaxis, :] + offset_n + offset_b

    return ring_verts.reshape(-1, 3)


def _build_tube_faces(
    num_points: int, sides: int, closed: bool
) -> np.ndarray:
    """Connect adjacent rings with triangle pairs (vectorized)."""
    num_segments = num_points if closed else num_points - 1
    i = np.arange(num_segments)[:, np.newaxis]
    j = np.arange(sides)[np.newaxis, :]
    i_next = (i + 1) % num_points
    j_next = (j + 1) % sides

    a = i * sides + j
    b = i * sides + j_next
    c = i_next * sides + j_next
    d = i_next * sides + j

    tri1 = np.stack([a, d, c], axis=-1)
    tri2 = np.stack([a, c, b], axis=-1)
    faces = np.concatenate([tri1, tri2], axis=1)
    return faces.reshape(-1, 3).astype(np.int64)


def _build_caps(
    points: np.ndarray,
    sides: int,
    vertex_offset: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build flat disk caps for open curve ends."""
    cap_verts = []
    cap_faces = []

    for end_idx, flip in [(0, True), (-1, False)]:
        center = points[end_idx]
        center_vidx = vertex_offset + len(cap_verts)
        cap_verts.append(center)

        # Use existing ring vertices — reference them by their tube index
        if end_idx == 0:
            ring_base = 0
        else:
            ring_base = (len(points) - 1) * sides

        for j in range(sides):
            j_next = (j + 1) % sides
            a = ring_base + j
            b = ring_base + j_next
            if flip:
                cap_faces.append([center_vidx, b, a])
            else:
                cap_faces.append([center_vidx, a, b])

    cap_verts_arr = np.array(cap_verts, dtype=np.float64)
    cap_faces_arr = np.array(cap_faces, dtype=np.int64)
    return cap_verts_arr, cap_faces_arr


def _warn_self_intersection(points: np.ndarray, radius: float) -> None:
    """Warn if the tube radius is large relative to minimum segment length."""
    if len(points) < 3:
        return

    segments = np.diff(points, axis=0)
    seg_lengths = np.linalg.norm(segments, axis=1)
    min_seg = np.min(seg_lengths[seg_lengths > 0]) if np.any(seg_lengths > 0) else 0.0

    if min_seg > 0 and radius > min_seg:
        warnings.warn(
            f"Tube radius {radius:.4f} exceeds minimum segment length "
            f"{min_seg:.4f}; self-intersection likely",
            stacklevel=2,
        )
