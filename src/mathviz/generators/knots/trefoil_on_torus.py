"""Trefoil on torus generator — a (2,3) torus knot rendered with its host torus.

Produces the knot as a TUBE curve alongside the torus surface as a WIREFRAME
mesh, showing how the knot sits on the torus.
"""

import logging
from typing import Any

import numpy as np

from mathviz.core.generator import GeneratorBase, register
from mathviz.core.math_object import BoundingBox, Curve, MathObject, Mesh
from mathviz.core.representation import RepresentationConfig, RepresentationType
from mathviz.generators.knots._knot_utils import (
    extract_curve_points,
    validate_curve_points,
)

logger = logging.getLogger(__name__)

_DEFAULT_TORUS_R = 1.0
_DEFAULT_TORUS_R_SMALL = 0.4
_DEFAULT_CURVE_POINTS = 1024
_DEFAULT_TORUS_RESOLUTION = 32
_DEFAULT_TUBE_RADIUS = 0.08
_MIN_TORUS_RESOLUTION = 8


def _compute_trefoil_on_torus(
    torus_r: float, torus_r_small: float, num_points: int,
) -> np.ndarray:
    """Compute the (2,3) torus knot sitting on a torus surface."""
    t = np.linspace(0.0, 2.0 * np.pi, num_points, endpoint=False)

    r = torus_r + torus_r_small * np.cos(3.0 * t)
    x = r * np.cos(2.0 * t)
    y = r * np.sin(2.0 * t)
    z = torus_r_small * np.sin(3.0 * t)

    return np.column_stack([x, y, z]).astype(np.float64)


def _build_torus_mesh(
    torus_r: float, torus_r_small: float, resolution: int,
) -> Mesh:
    """Build a triangle mesh for the torus surface."""
    n = resolution
    u_vals = np.linspace(0, 2 * np.pi, n, endpoint=False)
    v_vals = np.linspace(0, 2 * np.pi, n, endpoint=False)
    uu, vv = np.meshgrid(u_vals, v_vals, indexing="ij")

    x = (torus_r + torus_r_small * np.cos(vv)) * np.cos(uu)
    y = (torus_r + torus_r_small * np.cos(vv)) * np.sin(uu)
    z = torus_r_small * np.sin(vv)

    vertices = np.column_stack(
        [x.ravel(), y.ravel(), z.ravel()]
    ).astype(np.float64)

    faces = _build_wrapped_grid_faces(n)
    return Mesh(vertices=vertices, faces=faces)


def _build_wrapped_grid_faces(n: int) -> np.ndarray:
    """Build triangle faces for a periodic n*n grid wrapping in both axes."""
    row = np.arange(n)
    col = np.arange(n)
    rr, cc = np.meshgrid(row, col, indexing="ij")
    rr = rr.ravel()
    cc = cc.ravel()

    i00 = rr * n + cc
    i10 = ((rr + 1) % n) * n + cc
    i01 = rr * n + ((cc + 1) % n)
    i11 = ((rr + 1) % n) * n + ((cc + 1) % n)

    tri1 = np.stack([i00, i10, i11], axis=-1)
    tri2 = np.stack([i00, i11, i01], axis=-1)
    faces = np.concatenate([tri1, tri2], axis=0)
    return faces.astype(np.int64)


@register
class TrefoilOnTorusGenerator(GeneratorBase):
    """Trefoil knot rendered alongside its host torus surface."""

    name = "trefoil_on_torus"
    category = "knots"
    aliases = ()
    description = (
        "(2,3) torus knot with wireframe torus surface showing "
        "how the knot sits on the torus"
    )
    resolution_params = {
        "curve_points": "Number of sample points along the knot curve",
        "torus_resolution": "Grid divisions per axis for the torus mesh",
    }
    _resolution_defaults = {
        "curve_points": _DEFAULT_CURVE_POINTS,
        "torus_resolution": _DEFAULT_TORUS_RESOLUTION,
    }

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters for trefoil on torus."""
        return {
            "torus_R": _DEFAULT_TORUS_R,
            "torus_r": _DEFAULT_TORUS_R_SMALL,
        }

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate trefoil knot curve and torus surface mesh."""
        merged = self.get_default_params()
        if params:
            merged.update(params)

        merged, curve_points = extract_curve_points(
            merged, resolution_kwargs, _DEFAULT_CURVE_POINTS,
        )

        torus_r = float(merged["torus_R"])
        torus_r_small = float(merged["torus_r"])
        torus_resolution = int(
            resolution_kwargs.get("torus_resolution", _DEFAULT_TORUS_RESOLUTION)
        )

        if torus_r <= 0:
            raise ValueError(f"torus_R must be positive, got {torus_r}")
        if torus_r_small <= 0:
            raise ValueError(f"torus_r must be positive, got {torus_r_small}")
        if torus_r_small >= torus_r:
            raise ValueError(
                f"torus_r must be less than torus_R, "
                f"got torus_R={torus_r}, torus_r={torus_r_small}"
            )
        validate_curve_points(curve_points)
        if torus_resolution < _MIN_TORUS_RESOLUTION:
            raise ValueError(
                f"torus_resolution must be >= {_MIN_TORUS_RESOLUTION}, "
                f"got {torus_resolution}"
            )

        merged["curve_points"] = curve_points
        merged["torus_resolution"] = torus_resolution

        knot_points = _compute_trefoil_on_torus(
            torus_r, torus_r_small, curve_points,
        )
        knot_curve = Curve(points=knot_points, closed=True)

        torus_mesh = _build_torus_mesh(
            torus_r, torus_r_small, torus_resolution,
        )

        xy_extent = torus_r + torus_r_small
        z_extent = torus_r_small
        bbox = BoundingBox(
            min_corner=(-xy_extent, -xy_extent, -z_extent),
            max_corner=(xy_extent, xy_extent, z_extent),
        )

        logger.info(
            "Generated trefoil on torus: R=%.2f, r=%.2f, points=%d",
            torus_r, torus_r_small, curve_points,
        )

        return MathObject(
            mesh=torus_mesh,
            curves=[knot_curve],
            generator_name=self.resolved_name or self.name,
            category=self.category,
            parameters=merged,
            seed=seed,
            bounding_box=bbox,
        )

    def get_default_representation(self) -> RepresentationConfig:
        """Return TUBE for the knot curve (mesh uses WIREFRAME separately)."""
        return RepresentationConfig(
            type=RepresentationType.TUBE, tube_radius=_DEFAULT_TUBE_RADIUS,
        )
