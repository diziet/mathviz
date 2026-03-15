"""Möbius trefoil parametric surface generator.

Generates a Möbius-like strip whose centerline follows a trefoil knot path.
The strip cross-section makes a half-twist as it traverses the knot, producing
a non-orientable (single-sided) surface that combines Möbius topology with
knot geometry.
"""

import logging
from typing import Any

import numpy as np

from mathviz.core.generator import GeneratorBase, register
from mathviz.core.math_object import BoundingBox, MathObject, Mesh
from mathviz.core.representation import RepresentationConfig, RepresentationType
from mathviz.generators.parametric._mesh_utils import compute_padded_bounding_box

logger = logging.getLogger(__name__)

_DEFAULT_WIDTH = 0.3
_DEFAULT_CURVE_POINTS = 1024
_DEFAULT_GRID_RESOLUTION = 32
_MIN_CURVE_POINTS = 16
_MIN_GRID_RESOLUTION = 4


def _evaluate_trefoil_curve(t: np.ndarray) -> np.ndarray:
    """Evaluate the trefoil knot centerline at parameter values t in [0, 2pi)."""
    x = np.sin(t) + 2.0 * np.sin(2.0 * t)
    y = np.cos(t) - 2.0 * np.cos(2.0 * t)
    z = -np.sin(3.0 * t)
    return np.column_stack([x, y, z])


def _compute_bishop_frames(
    curve: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute parallel-transport (Bishop) frames along a closed curve.

    Returns normal and binormal arrays each of shape (N, 3).
    """
    n = len(curve)
    tangents = np.roll(curve, -1, axis=0) - np.roll(curve, 1, axis=0)
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-10, None)
    tangents = tangents / norms

    t0 = tangents[0]
    arbitrary = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(t0, arbitrary)) > 0.9:
        arbitrary = np.array([1.0, 0.0, 0.0])
    normal = np.cross(t0, arbitrary)
    normal = normal / np.linalg.norm(normal)

    normals = np.empty((n, 3), dtype=np.float64)
    normals[0] = normal

    for i in range(1, n):
        b = np.cross(tangents[i - 1], tangents[i])
        b_norm = np.linalg.norm(b)
        if b_norm < 1e-10:
            normals[i] = normals[i - 1]
        else:
            b = b / b_norm
            angle = np.arccos(np.clip(
                np.dot(tangents[i - 1], tangents[i]), -1.0, 1.0,
            ))
            normals[i] = _rotate_around_axis(normals[i - 1], b, angle)

    binormals = np.cross(tangents, normals)
    return normals, binormals


def _rotate_around_axis(
    vec: np.ndarray, axis: np.ndarray, angle: float,
) -> np.ndarray:
    """Rodrigues' rotation of vec around axis by angle."""
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    return (
        vec * cos_a
        + np.cross(axis, vec) * sin_a
        + axis * np.dot(axis, vec) * (1 - cos_a)
    )


def _build_mobius_strip_mesh(
    curve: np.ndarray,
    normals: np.ndarray,
    binormals: np.ndarray,
    width: float,
    n_cross: int,
) -> Mesh:
    """Sweep a line segment along the curve with a half-twist to form a Möbius strip."""
    n_along = len(curve)
    v_vals = np.linspace(-width / 2.0, width / 2.0, n_cross)

    # The half-twist angle goes from 0 to pi over the full curve
    twist_angles = np.linspace(0, np.pi, n_along, endpoint=False)

    cos_tw = np.cos(twist_angles)
    sin_tw = np.sin(twist_angles)

    # Twisted frame: rotate normal/binormal by twist angle
    twisted_n = (
        cos_tw[:, np.newaxis] * normals
        + sin_tw[:, np.newaxis] * binormals
    )

    # Offset along twisted normal for each cross-section point
    offsets = v_vals[np.newaxis, :, np.newaxis] * twisted_n[:, np.newaxis, :]
    vertices = (curve[:, np.newaxis, :] + offsets).reshape(-1, 3)
    vertices = vertices.astype(np.float64)

    faces = _build_mobius_seam_faces(n_along, n_cross)
    return Mesh(vertices=vertices, faces=faces)


def _build_mobius_seam_faces(n_u: int, n_v: int) -> np.ndarray:
    """Build triangle faces for a grid wrapping in u with v-reversal at the seam.

    Interior rows connect normally (u wraps, v open). At the u-seam
    (row n_u-1 → row 0), column v connects to column (n_v-1-v) to
    implement the half-twist that makes the surface non-orientable.
    """
    # Interior faces: rows 0..n_u-2, columns 0..n_v-2
    rows = np.arange(n_u - 1)
    cols = np.arange(n_v - 1)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    rr, cc = rr.ravel(), cc.ravel()

    i00 = rr * n_v + cc
    i10 = (rr + 1) * n_v + cc
    i01 = rr * n_v + (cc + 1)
    i11 = (rr + 1) * n_v + (cc + 1)

    interior_t1 = np.stack([i00, i10, i11], axis=-1)
    interior_t2 = np.stack([i00, i11, i01], axis=-1)

    # Seam faces: row n_u-1 connects to row 0 with v-reversal
    sc = np.arange(n_v - 1)
    s00 = (n_u - 1) * n_v + sc
    s01 = (n_u - 1) * n_v + (sc + 1)
    # Row 0, reversed: v maps to (n_v-1-v)
    s10 = (n_v - 1 - sc)
    s11 = (n_v - 1 - sc - 1)

    seam_t1 = np.stack([s00, s10, s11], axis=-1)
    seam_t2 = np.stack([s00, s11, s01], axis=-1)

    return np.concatenate(
        [interior_t1, interior_t2, seam_t1, seam_t2], axis=0,
    ).astype(np.int64)


def _validate_params(
    width: float, curve_points: int, grid_resolution: int,
) -> None:
    """Validate Möbius trefoil parameters."""
    if width <= 0:
        raise ValueError(f"width must be positive, got {width}")
    if curve_points < _MIN_CURVE_POINTS:
        raise ValueError(
            f"curve_points must be >= {_MIN_CURVE_POINTS}, got {curve_points}"
        )
    if grid_resolution < _MIN_GRID_RESOLUTION:
        raise ValueError(
            f"grid_resolution must be >= {_MIN_GRID_RESOLUTION}, "
            f"got {grid_resolution}"
        )


@register
class MobiusTrefoilGenerator(GeneratorBase):
    """Möbius trefoil surface generator."""

    name = "mobius_trefoil"
    category = "parametric"
    aliases = ()
    description = (
        "Möbius strip twisted into a trefoil knot shape, "
        "combining non-orientability with knot topology"
    )
    resolution_params = {
        "curve_points": "Number of points along the trefoil centerline",
        "grid_resolution": "Cross-section resolution of the strip",
    }
    _resolution_defaults = {
        "curve_points": _DEFAULT_CURVE_POINTS,
        "grid_resolution": _DEFAULT_GRID_RESOLUTION,
    }

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters for the Möbius trefoil."""
        return {"width": _DEFAULT_WIDTH}

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate a Möbius trefoil mesh."""
        merged = self.get_default_params()
        if params:
            merged.update(params)

        width = float(merged["width"])
        curve_points = int(
            resolution_kwargs.get("curve_points", _DEFAULT_CURVE_POINTS)
        )
        grid_resolution = int(
            resolution_kwargs.get("grid_resolution", _DEFAULT_GRID_RESOLUTION)
        )

        _validate_params(width, curve_points, grid_resolution)

        t = np.linspace(0, 2 * np.pi, curve_points, endpoint=False)
        curve = _evaluate_trefoil_curve(t)
        normals, binormals = _compute_bishop_frames(curve)

        mesh = _build_mobius_strip_mesh(
            curve, normals, binormals, width, grid_resolution,
        )
        bbox = compute_padded_bounding_box(mesh.vertices)

        merged["curve_points"] = curve_points
        merged["grid_resolution"] = grid_resolution

        logger.info(
            "Generated mobius_trefoil: width=%.3f, curve_pts=%d, "
            "grid=%d, vertices=%d, faces=%d",
            width, curve_points, grid_resolution,
            len(mesh.vertices), len(mesh.faces),
        )

        return MathObject(
            mesh=mesh,
            generator_name=self.name,
            category=self.category,
            parameters=merged,
            seed=seed,
            bounding_box=bbox,
        )

    def get_default_representation(self) -> RepresentationConfig:
        """Return the recommended representation."""
        return RepresentationConfig(type=RepresentationType.SURFACE_SHELL)
