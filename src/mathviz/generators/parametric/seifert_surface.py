"""Seifert surface generator using Milnor fiber parameterization.

Generates orientable surfaces bounded by knots. For the trefoil (a torus
knot), uses the Milnor fiber of z1^p + z2^q on S³ with stereographic
projection. For the figure-eight knot, uses a spanning surface construction.
"""

import logging
from typing import Any

import numpy as np

from mathviz.core.generator import GeneratorBase, register
from mathviz.core.math_object import BoundingBox, MathObject, Mesh
from mathviz.core.representation import RepresentationConfig, RepresentationType
from mathviz.generators.parametric._mesh_utils import build_mixed_grid_faces

logger = logging.getLogger(__name__)

_DEFAULT_KNOT_TYPE = "trefoil"
_DEFAULT_THETA = 0.0
_DEFAULT_GRID_RESOLUTION = 128
_MIN_GRID_RESOLUTION = 8
_VALID_KNOT_TYPES = ("trefoil", "figure_eight")

_TREFOIL_P = 2
_TREFOIL_Q = 3
_ALPHA_MIN = 0.4
_ALPHA_MAX = np.pi / 2 - 0.05


def _validate_params(knot_type: str, theta: float, grid_resolution: int) -> None:
    """Validate Seifert surface parameters."""
    if knot_type not in _VALID_KNOT_TYPES:
        raise ValueError(
            f"knot_type must be one of {_VALID_KNOT_TYPES}, got {knot_type!r}"
        )
    if grid_resolution < _MIN_GRID_RESOLUTION:
        raise ValueError(
            f"grid_resolution must be >= {_MIN_GRID_RESOLUTION}, "
            f"got {grid_resolution}"
        )


def _stereographic_project(
    s3w: np.ndarray, s3x: np.ndarray, s3y: np.ndarray, s3z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stereographic projection from S³ south pole to R³."""
    denom = np.maximum(1.0 + s3w, 1e-8)
    return s3x / denom, s3y / denom, s3z / denom


def _generate_trefoil_fiber(theta: float, n: int) -> Mesh:
    """Generate trefoil Seifert surface via Milnor fiber of z1² + z2³."""
    p, q = _TREFOIL_P, _TREFOIL_Q

    alpha = np.linspace(_ALPHA_MIN, _ALPHA_MAX, n)
    beta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    aa, bb = np.meshgrid(alpha, beta, indexing="ij")

    ca, sa = np.cos(aa), np.sin(aa)
    big_a = ca ** p
    big_b = sa ** q

    # Fiber condition: arg(big_a·e^{ip·β} + big_b·e^{iq·γ}) = θ
    # Solve: sin(q·γ − θ) = −big_a·sin(p·β − θ) / big_b
    phase = p * bb - theta
    ratio = np.clip(-big_a * np.sin(phase) / big_b, -1.0, 1.0)

    psi = np.arcsin(ratio)
    re_check = big_a * np.cos(phase) + big_b * np.cos(psi)

    # Use complementary arcsin branch where real part is non-positive
    psi_alt = np.pi - psi
    use_alt = re_check <= 0
    psi_final = np.where(use_alt, psi_alt, psi)

    gamma = (psi_final + theta) / q

    # S³ coordinates → stereographic projection
    s3w = ca * np.cos(bb)
    s3x = ca * np.sin(bb)
    s3y = sa * np.cos(gamma)
    s3z = sa * np.sin(gamma)

    x, y, z = _stereographic_project(s3w, s3x, s3y, s3z)

    vertices = np.column_stack(
        [x.ravel(), y.ravel(), z.ravel()]
    ).astype(np.float64)
    faces = build_mixed_grid_faces(n, n, wrap_u=False, wrap_v=True)
    return Mesh(vertices=vertices, faces=faces)


def _figure_eight_knot(
    t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parameterize the figure-eight knot in R³."""
    x = np.sin(t) + 2.0 * np.sin(2.0 * t)
    y = np.cos(t) - 2.0 * np.cos(2.0 * t)
    z = -np.sin(3.0 * t)
    return x, y, z


def _generate_figure_eight_surface(theta: float, n: int) -> Mesh:
    """Generate figure-eight Seifert surface via spanning construction."""
    r_vals = np.linspace(0, 1, n)
    t_vals = np.linspace(0, 2 * np.pi, n, endpoint=False)
    rr, tt = np.meshgrid(r_vals, t_vals, indexing="ij")

    kx, ky, kz = _figure_eight_knot(tt)

    # Center curve: small circle with theta-dependent phase
    center_radius = 0.3
    cx = center_radius * np.cos(tt + theta)
    cy = center_radius * np.sin(tt + theta)
    cz = np.zeros_like(tt)

    # Quadratic easing for smooth convergence toward center
    s = rr ** 2
    x = (1.0 - s) * cx + s * kx
    y = (1.0 - s) * cy + s * ky
    z = (1.0 - s) * cz + s * kz

    vertices = np.column_stack(
        [x.ravel(), y.ravel(), z.ravel()]
    ).astype(np.float64)
    faces = build_mixed_grid_faces(n, n, wrap_u=False, wrap_v=True)
    return Mesh(vertices=vertices, faces=faces)


def _compute_bounding_box(vertices: np.ndarray) -> BoundingBox:
    """Compute axis-aligned bounding box from vertices with padding."""
    vmin = vertices.min(axis=0)
    vmax = vertices.max(axis=0)
    padding = (vmax - vmin) * 0.02 + 1e-6
    return BoundingBox(
        min_corner=tuple(vmin - padding),
        max_corner=tuple(vmax + padding),
    )


def _trefoil_reference_knot(n_points: int) -> np.ndarray:
    """Reference trefoil knot via stereographic projection from S³."""
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    z1 = np.exp(2j * t) / np.sqrt(2)
    z2 = np.exp(3j * t) / np.sqrt(2)
    denom = np.maximum(1.0 + z1.real, 1e-8)
    return np.column_stack([z1.imag / denom, z2.real / denom, z2.imag / denom])


def _figure_eight_reference_knot(n_points: int) -> np.ndarray:
    """Reference figure-eight knot in R³."""
    t = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    x, y, z = _figure_eight_knot(t)
    return np.column_stack([x, y, z])


@register
class SeifertSurfaceGenerator(GeneratorBase):
    """Seifert surface bounded by a knot via Milnor fiber parameterization."""

    name = "seifert_surface"
    category = "parametric"
    aliases = ("seifert",)
    description = "Orientable surface bounded by a knot (Seifert surface)"
    resolution_params = {"grid_resolution": "Number of grid divisions per axis"}
    _resolution_defaults = {"grid_resolution": _DEFAULT_GRID_RESOLUTION}

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters for the Seifert surface."""
        return {
            "knot_type": _DEFAULT_KNOT_TYPE,
            "theta": _DEFAULT_THETA,
        }

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate a Seifert surface mesh."""
        merged = self.get_default_params()
        if params:
            merged.update(params)

        knot_type = str(merged["knot_type"])
        theta = float(merged["theta"])
        grid_resolution = int(
            resolution_kwargs.get("grid_resolution", _DEFAULT_GRID_RESOLUTION)
        )

        _validate_params(knot_type, theta, grid_resolution)

        if knot_type == "trefoil":
            mesh = _generate_trefoil_fiber(theta, grid_resolution)
        else:
            mesh = _generate_figure_eight_surface(theta, grid_resolution)

        bbox = _compute_bounding_box(mesh.vertices)
        merged["grid_resolution"] = grid_resolution

        logger.info(
            "Generated seifert_surface: knot_type=%s, theta=%.3f, grid=%d, "
            "vertices=%d, faces=%d",
            knot_type, theta, grid_resolution,
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
        """Return parameter ranges for Seifert surface."""
        return {
            "theta": {"min": 0.0, "max": 2 * np.pi, "step": 0.1},
        }

    def get_default_representation(self) -> RepresentationConfig:
        """Return the recommended representation for the Seifert surface."""
        return RepresentationConfig(type=RepresentationType.SURFACE_SHELL)
