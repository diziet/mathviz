"""Superellipsoid parametric surface generator.

The superellipsoid generalizes the ellipsoid using signed-power trigonometric
functions. With exponents e1=e2=1 it approximates a sphere; other values
produce rounded cubes, octahedra, cylinders, and other forms.
"""

import logging
from typing import Any

import numpy as np

from mathviz.core.generator import GeneratorBase, register
from mathviz.core.math_object import BoundingBox, MathObject, Mesh
from mathviz.core.representation import RepresentationConfig, RepresentationType
from mathviz.generators.parametric._mesh_utils import build_sphere_faces

logger = logging.getLogger(__name__)

_DEFAULT_A1 = 1.0
_DEFAULT_A2 = 1.0
_DEFAULT_A3 = 1.0
_DEFAULT_E1 = 1.0
_DEFAULT_E2 = 1.0
_DEFAULT_GRID_RESOLUTION = 128
_MIN_GRID_RESOLUTION = 4


def _sgn_pow(base: np.ndarray, exponent: float) -> np.ndarray:
    """Signed-power function: sign(x) * |x|^p."""
    return np.sign(base) * np.abs(base) ** exponent


def _evaluate_superellipsoid(
    eta: np.ndarray, omega: np.ndarray,
    a1: float, a2: float, a3: float,
    e1: float, e2: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate superellipsoid surface.

    eta in [-pi/2, pi/2] (latitude), omega in [-pi, pi) (longitude).
    """
    cos_eta = np.cos(eta)
    sin_eta = np.sin(eta)
    cos_omega = np.cos(omega)
    sin_omega = np.sin(omega)

    x = a1 * _sgn_pow(cos_eta, e1) * _sgn_pow(cos_omega, e2)
    y = a2 * _sgn_pow(cos_eta, e1) * _sgn_pow(sin_omega, e2)
    z = a3 * _sgn_pow(sin_eta, e1)
    return x, y, z


def _validate_params(
    a1: float, a2: float, a3: float,
    e1: float, e2: float, grid_resolution: int,
) -> None:
    """Validate superellipsoid parameters."""
    for name, val in [("a1", a1), ("a2", a2), ("a3", a3)]:
        if val <= 0:
            raise ValueError(f"{name} must be positive, got {val}")
    for name, val in [("e1", e1), ("e2", e2)]:
        if val <= 0:
            raise ValueError(f"{name} must be positive, got {val}")
    if grid_resolution < _MIN_GRID_RESOLUTION:
        raise ValueError(
            f"grid_resolution must be >= {_MIN_GRID_RESOLUTION}, "
            f"got {grid_resolution}"
        )


def _generate_superellipsoid_mesh(
    a1: float, a2: float, a3: float,
    e1: float, e2: float, grid_resolution: int,
) -> Mesh:
    """Build triangle mesh for the superellipsoid with pole vertices."""
    n = grid_resolution
    # Latitude: exclude exact poles to avoid degenerate triangles
    eta_vals = np.linspace(-np.pi / 2, np.pi / 2, n + 2)[1:-1]
    omega_vals = np.linspace(-np.pi, np.pi, n, endpoint=False)
    eta_grid, omega_grid = np.meshgrid(eta_vals, omega_vals, indexing="ij")

    x, y, z = _evaluate_superellipsoid(
        eta_grid, omega_grid, a1, a2, a3, e1, e2,
    )
    body_verts = np.column_stack([x.ravel(), y.ravel(), z.ravel()])

    # Add pole vertices
    south_pole = np.array([[0.0, 0.0, -a3]])
    north_pole = np.array([[0.0, 0.0, a3]])
    vertices = np.concatenate([body_verts, south_pole, north_pole], axis=0)
    vertices = vertices.astype(np.float64)

    faces = build_sphere_faces(n, n)
    return Mesh(vertices=vertices, faces=faces)


@register
class SuperellipsoidGenerator(GeneratorBase):
    """Parametric superellipsoid surface generator."""

    name = "superellipsoid"
    category = "parametric"
    aliases = ()
    description = "Superellipsoid with configurable exponents and radii"
    resolution_params = {"grid_resolution": "Number of grid divisions per axis"}

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters for the superellipsoid."""
        return {
            "a1": _DEFAULT_A1, "a2": _DEFAULT_A2, "a3": _DEFAULT_A3,
            "e1": _DEFAULT_E1, "e2": _DEFAULT_E2,
        }

    def get_default_resolution(self) -> dict[str, Any]:
        """Return default values for resolution parameters."""
        return {"grid_resolution": _DEFAULT_GRID_RESOLUTION}

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate a superellipsoid mesh."""
        merged = self.get_default_params()
        if params:
            merged.update(params)

        a1 = float(merged["a1"])
        a2 = float(merged["a2"])
        a3 = float(merged["a3"])
        e1 = float(merged["e1"])
        e2 = float(merged["e2"])
        grid_resolution = int(
            resolution_kwargs.get("grid_resolution", _DEFAULT_GRID_RESOLUTION)
        )

        _validate_params(a1, a2, a3, e1, e2, grid_resolution)

        mesh = _generate_superellipsoid_mesh(a1, a2, a3, e1, e2, grid_resolution)
        bbox = BoundingBox(
            min_corner=(-a1, -a2, -a3), max_corner=(a1, a2, a3),
        )

        merged["grid_resolution"] = grid_resolution

        logger.info(
            "Generated superellipsoid: a=(%.3f,%.3f,%.3f), e=(%.3f,%.3f), "
            "grid=%d, vertices=%d, faces=%d",
            a1, a2, a3, e1, e2, grid_resolution,
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
        """Return the recommended representation for the superellipsoid."""
        return RepresentationConfig(type=RepresentationType.SURFACE_SHELL)
