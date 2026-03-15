"""Dini's surface parametric generator.

Dini's surface is a twisted pseudospherical surface that looks like a
seashell or spiral horn. It has constant negative Gaussian curvature
and is parameterized by a helical twist around a tractrix.
"""

import logging
from typing import Any

import numpy as np

from mathviz.core.generator import GeneratorBase, register
from mathviz.core.math_object import MathObject, Mesh
from mathviz.core.representation import RepresentationConfig, RepresentationType
from mathviz.generators.parametric._mesh_utils import (
    build_open_grid_faces,
    compute_padded_bounding_box,
)

logger = logging.getLogger(__name__)

_DEFAULT_A = 1.0
_DEFAULT_B = 0.2
_DEFAULT_TURNS = 2
_DEFAULT_GRID_RESOLUTION = 128
_MIN_GRID_RESOLUTION = 8
_MIN_TURNS = 1
_V_EPSILON = 0.01


def _evaluate_dini_surface(
    u: np.ndarray, v: np.ndarray, a: float, b: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate Dini's surface immersion f(u, v) -> (x, y, z).

    u, v are 2D arrays from meshgrid.
    u in [0, turns*2*pi], v in (epsilon, pi-epsilon).
    """
    x = a * np.cos(u) * np.sin(v)
    y = a * np.sin(u) * np.sin(v)
    z = a * (np.cos(v) + np.log(np.tan(v / 2.0))) + b * u
    return x, y, z


def _validate_params(
    a: float, b: float, turns: int, grid_resolution: int,
) -> None:
    """Validate Dini surface parameters."""
    if a <= 0:
        raise ValueError(f"a must be positive, got {a}")
    if b <= 0:
        raise ValueError(f"b must be positive, got {b}")
    if turns < _MIN_TURNS:
        raise ValueError(f"turns must be >= {_MIN_TURNS}, got {turns}")
    if grid_resolution < _MIN_GRID_RESOLUTION:
        raise ValueError(
            f"grid_resolution must be >= {_MIN_GRID_RESOLUTION}, "
            f"got {grid_resolution}"
        )


def _generate_dini_mesh(
    a: float, b: float, turns: int, grid_resolution: int,
) -> Mesh:
    """Build triangle mesh for Dini's surface."""
    n = grid_resolution
    u_max = turns * 2.0 * np.pi
    u_vals = np.linspace(0, u_max, n, endpoint=True)
    v_vals = np.linspace(_V_EPSILON, np.pi - _V_EPSILON, n, endpoint=True)
    uu, vv = np.meshgrid(u_vals, v_vals, indexing="ij")

    x, y, z = _evaluate_dini_surface(uu, vv, a, b)
    vertices = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    vertices = vertices.astype(np.float64)
    faces = build_open_grid_faces(n, n)
    return Mesh(vertices=vertices, faces=faces)


@register
class DiniSurfaceGenerator(GeneratorBase):
    """Parametric Dini's surface — twisted pseudospherical spiral."""

    name = "dini_surface"
    category = "parametric"
    aliases = ("dini",)
    description = "Twisted pseudospherical surface resembling a seashell"
    resolution_params = {"grid_resolution": "Number of grid divisions per axis"}
    _resolution_defaults = {"grid_resolution": _DEFAULT_GRID_RESOLUTION}

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters for Dini's surface."""
        return {
            "a": _DEFAULT_A,
            "b": _DEFAULT_B,
            "turns": _DEFAULT_TURNS,
        }

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate a Dini's surface mesh.

        Surface is analytically deterministic; seed is stored for
        metadata provenance only (no RNG used).
        """
        merged = self.get_default_params()
        if params:
            merged.update(params)

        a = float(merged["a"])
        b = float(merged["b"])
        turns = int(merged["turns"])
        grid_resolution = int(
            resolution_kwargs.get("grid_resolution", _DEFAULT_GRID_RESOLUTION)
        )

        _validate_params(a, b, turns, grid_resolution)

        mesh = _generate_dini_mesh(a, b, turns, grid_resolution)
        bbox = compute_padded_bounding_box(mesh.vertices)

        merged["grid_resolution"] = grid_resolution

        logger.info(
            "Generated dini_surface: a=%.3f, b=%.3f, turns=%d, grid=%d, "
            "vertices=%d, faces=%d",
            a, b, turns, grid_resolution,
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
        """Return exploration ranges for parameters."""
        return {
            "a": {"min": 0.1, "max": 5.0, "step": 0.1},
            "b": {"min": 0.01, "max": 1.0, "step": 0.01},
            "turns": {"min": 1, "max": 10, "step": 1},
        }

    def get_default_representation(self) -> RepresentationConfig:
        """Return the recommended representation for Dini's surface."""
        return RepresentationConfig(type=RepresentationType.SURFACE_SHELL)
