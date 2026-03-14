"""3D Voronoi cell boundaries as wireframe geometry.

Generates N random seed points via default_rng(seed), computes the 3D
Voronoi tessellation with scipy.spatial.Voronoi, and extracts finite
cell ridges as Curve edges for wireframe rendering.
"""

import logging
from typing import Any

import numpy as np
from numpy.random import default_rng
from scipy.spatial import Voronoi

from mathviz.core.generator import GeneratorBase, register
from mathviz.core.math_object import BoundingBox, Curve, MathObject
from mathviz.core.representation import RepresentationConfig, RepresentationType

logger = logging.getLogger(__name__)

_DEFAULT_NUM_POINTS = 20
_DEFAULT_SCALE = 1.0
_MIN_NUM_POINTS = 4
_DEFAULT_CURVE_POINTS = 2


def _validate_params(num_points: int, scale: float) -> None:
    """Validate voronoi parameters."""
    if num_points < _MIN_NUM_POINTS:
        raise ValueError(
            f"num_points must be >= {_MIN_NUM_POINTS}, got {num_points}"
        )
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")


def _generate_seed_points(
    num_points: int, scale: float, seed: int,
) -> np.ndarray:
    """Generate random 3D seed points for Voronoi tessellation."""
    rng = default_rng(seed)
    return rng.uniform(-scale, scale, size=(num_points, 3))


def _extract_finite_ridges(vor: Voronoi) -> list[np.ndarray]:
    """Extract finite ridge edges from a Voronoi tessellation.

    Returns a list of (2, 3) float64 arrays, one per finite edge.
    """
    edges: list[np.ndarray] = []
    for ridge_vertices in vor.ridge_vertices:
        if -1 in ridge_vertices:
            continue
        pts = vor.vertices[ridge_vertices]
        if np.any(np.isnan(pts)) or np.any(np.isinf(pts)):
            continue
        edges.append(pts.astype(np.float64))
    return edges


def _build_curves_from_ridges(ridges: list[np.ndarray]) -> list[Curve]:
    """Convert ridge vertex arrays to Curve objects."""
    curves: list[Curve] = []
    for ridge_pts in ridges:
        curves.append(Curve(points=ridge_pts.astype(np.float64), closed=False))
    return curves


def _collect_all_points(curves: list[Curve]) -> np.ndarray:
    """Gather all points from curves into a single array for bounding box."""
    if not curves:
        return np.zeros((1, 3), dtype=np.float64)
    return np.concatenate([c.points for c in curves], axis=0)


@register
class Voronoi3DGenerator(GeneratorBase):
    """3D Voronoi cell boundaries as wireframe geometry."""

    name = "voronoi_3d"
    category = "geometry"
    aliases = ()
    description = "3D Voronoi cell boundaries as wireframe edges"
    resolution_params = {}

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters for voronoi_3d."""
        return {
            "num_points": _DEFAULT_NUM_POINTS,
            "scale": _DEFAULT_SCALE,
        }

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate 3D Voronoi wireframe geometry."""
        merged = self.get_default_params()
        if params:
            merged.update(params)

        num_points = int(merged["num_points"])
        scale = float(merged["scale"])
        _validate_params(num_points, scale)

        seed_points = _generate_seed_points(num_points, scale, seed)
        vor = Voronoi(seed_points)

        ridges = _extract_finite_ridges(vor)
        curves = _build_curves_from_ridges(ridges)

        if not curves:
            raise ValueError("Voronoi produced no finite ridges")

        all_pts = _collect_all_points(curves)
        bbox = BoundingBox.from_points(all_pts)

        logger.info(
            "Generated voronoi_3d: points=%d, edges=%d, seed=%d",
            num_points, len(curves), seed,
        )

        return MathObject(
            curves=curves,
            generator_name=self.name,
            category=self.category,
            parameters=merged,
            seed=seed,
            bounding_box=bbox,
        )

    def get_default_representation(self) -> RepresentationConfig:
        """Return wireframe as the default representation."""
        return RepresentationConfig(type=RepresentationType.WIREFRAME)
