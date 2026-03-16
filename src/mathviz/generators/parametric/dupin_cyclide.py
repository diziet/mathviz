"""Dupin cyclide parametric surface generator.

The Dupin cyclide is a smooth inversive geometry shape that generalizes
the torus, cylinder, and cone. It is parameterized as an inversion of
a torus via a Möbius transformation with parameters a, b, c, d.

The surface is periodic in both u and v, so the grid wraps at boundaries.
"""

import logging
from typing import Any

import numpy as np

from mathviz.core.generator import GeneratorBase, register
from mathviz.core.math_object import BoundingBox, MathObject, Mesh
from mathviz.core.representation import RepresentationConfig, RepresentationType
from mathviz.generators.parametric._mesh_utils import (
    build_wrapped_grid_faces,
    compute_padded_bounding_box,
)

logger = logging.getLogger(__name__)

_DEFAULT_A = 1.0
_DEFAULT_B = 0.8
_DEFAULT_C = 0.5
_DEFAULT_D = 0.6
_DEFAULT_GRID_RESOLUTION = 128
_MIN_GRID_RESOLUTION = 3


def _evaluate_cyclide(
    u: np.ndarray,
    v: np.ndarray,
    a: float,
    b: float,
    c: float,
    d: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the Dupin cyclide parametric surface f(u, v) -> (x, y, z).

    u, v are 2D arrays from meshgrid, both in [0, 2*pi).
    """
    cos_u = np.cos(u)
    sin_u = np.sin(u)
    cos_v = np.cos(v)
    sin_v = np.sin(v)

    denom = a - c * cos_u * cos_v

    x = (d * (c - a * cos_u * cos_v) + b * b * cos_u) / denom
    y = (b * sin_u * (a - d * cos_v)) / denom
    z = (b * sin_v * (c * cos_u - d)) / denom

    return x, y, z


def _validate_params(
    a: float, b: float, c: float, d: float, grid_resolution: int,
) -> None:
    """Validate Dupin cyclide parameters, raising ValueError if invalid."""
    if a <= 0:
        raise ValueError(f"a must be positive, got {a}")
    if b <= 0:
        raise ValueError(f"b must be positive, got {b}")
    if c <= 0:
        raise ValueError(f"c must be positive, got {c}")
    if c >= a:
        raise ValueError(
            f"c must be less than a to avoid singularities, got c={c}, a={a}"
        )
    if d <= 0:
        raise ValueError(f"d must be positive, got {d}")
    if grid_resolution < _MIN_GRID_RESOLUTION:
        raise ValueError(
            f"grid_resolution must be >= {_MIN_GRID_RESOLUTION}, "
            f"got {grid_resolution}"
        )


def _generate_cyclide_mesh(
    a: float,
    b: float,
    c: float,
    d: float,
    grid_resolution: int,
) -> Mesh:
    """Build a triangle mesh for a Dupin cyclide from parametric evaluation."""
    n = grid_resolution
    u_vals = np.linspace(0, 2 * np.pi, n, endpoint=False)
    v_vals = np.linspace(0, 2 * np.pi, n, endpoint=False)
    uu, vv = np.meshgrid(u_vals, v_vals, indexing="ij")

    x, y, z = _evaluate_cyclide(uu, vv, a, b, c, d)

    vertices = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    vertices = vertices.astype(np.float64)

    faces = build_wrapped_grid_faces(n, n)

    return Mesh(vertices=vertices, faces=faces)


@register
class DupinCyclideGenerator(GeneratorBase):
    """Parametric Dupin cyclide surface generator."""

    name = "dupin_cyclide"
    category = "parametric"
    aliases = ("cyclide",)
    description = (
        "Dupin cyclide — inversive geometry shape generalizing "
        "torus, cylinder, and cone"
    )
    resolution_params = {"grid_resolution": "Number of grid divisions per axis"}
    _resolution_defaults = {"grid_resolution": _DEFAULT_GRID_RESOLUTION}

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters for the Dupin cyclide generator."""
        return {
            "a": _DEFAULT_A,
            "b": _DEFAULT_B,
            "c": _DEFAULT_C,
            "d": _DEFAULT_D,
        }

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate a Dupin cyclide mesh."""
        merged = self.get_default_params()
        if params:
            merged.update(params)

        a = float(merged["a"])
        b = float(merged["b"])
        c = float(merged["c"])
        d = float(merged["d"])
        grid_resolution = int(
            resolution_kwargs.get("grid_resolution", _DEFAULT_GRID_RESOLUTION)
        )

        _validate_params(a, b, c, d, grid_resolution)

        mesh = _generate_cyclide_mesh(a, b, c, d, grid_resolution)
        bbox = compute_padded_bounding_box(mesh.vertices)

        merged["grid_resolution"] = grid_resolution

        logger.info(
            "Generated dupin_cyclide: a=%.3f, b=%.3f, c=%.3f, d=%.3f, "
            "grid=%d, vertices=%d, faces=%d",
            a, b, c, d, grid_resolution,
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

    def get_param_ranges(self) -> dict[str, dict[str, float]]:
        """Return exploration ranges for cyclide parameters.

        Note: c must be strictly less than a. The range for c is capped
        below a's minimum to avoid invalid combinations in parameter sweeps.
        """
        return {
            "a": {"min": 0.5, "max": 3.0, "step": 0.1},
            "b": {"min": 0.1, "max": 2.0, "step": 0.1},
            "c": {"min": 0.1, "max": 0.8, "step": 0.1},
            "d": {"min": 0.1, "max": 2.0, "step": 0.1},
        }

    def get_default_representation(self) -> RepresentationConfig:
        """Return the recommended representation for the Dupin cyclide."""
        return RepresentationConfig(type=RepresentationType.SURFACE_SHELL)
