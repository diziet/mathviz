"""Spherical harmonics surface generator.

Modulates a unit sphere's radius by a combination of real spherical harmonics.
Accepts either a single (l, m) pair or a coefficient vector. When all
coefficients are zero except Y₀₀, the result approximates a sphere.
"""

import logging
from typing import Any

import numpy as np
from scipy.special import sph_harm_y  # type: ignore[import-untyped]

from mathviz.core.generator import GeneratorBase, register
from mathviz.core.math_object import BoundingBox, MathObject, Mesh
from mathviz.core.representation import RepresentationConfig, RepresentationType
from mathviz.generators.parametric._mesh_utils import build_sphere_faces

logger = logging.getLogger(__name__)

_DEFAULT_L = 0
_DEFAULT_M = 0
_DEFAULT_BASE_RADIUS = 1.0
_DEFAULT_AMPLITUDE = 0.3
_DEFAULT_GRID_RESOLUTION = 128
_MIN_GRID_RESOLUTION = 4
_MAX_L = 10


def _real_spherical_harmonic(
    l_deg: int, m_ord: int, theta: np.ndarray, phi: np.ndarray,
) -> np.ndarray:
    """Compute real-valued spherical harmonic Y_l^m(theta, phi).

    theta is polar angle [0, pi], phi is azimuthal [0, 2*pi).
    Uses scipy.special.sph_harm_y(l, m, theta, phi).
    """
    y_complex = sph_harm_y(l_deg, abs(m_ord), theta, phi)
    if m_ord > 0:
        return np.real(y_complex) * np.sqrt(2)
    if m_ord < 0:
        return np.imag(y_complex) * np.sqrt(2)
    return np.real(y_complex)


def _compute_radius_field(
    theta: np.ndarray, phi: np.ndarray,
    coefficients: list[tuple[int, int, float]],
    base_radius: float,
) -> np.ndarray:
    """Compute radius r(theta, phi) = base + sum(c_lm * Y_lm)."""
    radius = np.full_like(theta, base_radius, dtype=np.float64)
    for l_deg, m_ord, coeff in coefficients:
        if abs(coeff) < 1e-15:
            continue
        radius = radius + coeff * _real_spherical_harmonic(
            l_deg, m_ord, theta, phi,
        )
    return radius


def _parse_coefficients(
    params: dict[str, Any],
) -> list[tuple[int, int, float]]:
    """Parse coefficient specification from parameters.

    Supports two modes:
    - Single harmonic: (l, m) with amplitude
    - Coefficient vector: list of (l, m, coefficient) tuples
    """
    if "coefficients" in params:
        return [(int(l), int(m), float(c))
                for l, m, c in params["coefficients"]]
    l_deg = int(params.get("l", _DEFAULT_L))
    m_ord = int(params.get("m", _DEFAULT_M))
    amplitude = float(params.get("amplitude", _DEFAULT_AMPLITUDE))
    coeffs: list[tuple[int, int, float]] = []
    if l_deg == 0 and m_ord == 0:
        return coeffs
    coeffs.append((l_deg, m_ord, amplitude))
    return coeffs


def _validate_params(
    coefficients: list[tuple[int, int, float]],
    base_radius: float, grid_resolution: int,
) -> None:
    """Validate spherical harmonics parameters."""
    if base_radius <= 0:
        raise ValueError(f"base_radius must be positive, got {base_radius}")
    for l_deg, m_ord, _ in coefficients:
        if l_deg < 0:
            raise ValueError(f"l must be non-negative, got {l_deg}")
        if l_deg > _MAX_L:
            raise ValueError(f"l must be <= {_MAX_L}, got {l_deg}")
        if abs(m_ord) > l_deg:
            raise ValueError(
                f"|m| must be <= l, got l={l_deg}, m={m_ord}"
            )
    if grid_resolution < _MIN_GRID_RESOLUTION:
        raise ValueError(
            f"grid_resolution must be >= {_MIN_GRID_RESOLUTION}, "
            f"got {grid_resolution}"
        )


def _generate_sh_mesh(
    coefficients: list[tuple[int, int, float]],
    base_radius: float, grid_resolution: int,
) -> Mesh:
    """Build triangle mesh for spherical harmonics surface."""
    n = grid_resolution
    theta_vals = np.linspace(0, np.pi, n + 2)[1:-1]
    phi_vals = np.linspace(0, 2 * np.pi, n, endpoint=False)
    theta_grid, phi_grid = np.meshgrid(theta_vals, phi_vals, indexing="ij")

    r = _compute_radius_field(theta_grid, phi_grid, coefficients, base_radius)

    x = r * np.sin(theta_grid) * np.cos(phi_grid)
    y = r * np.sin(theta_grid) * np.sin(phi_grid)
    z = r * np.cos(theta_grid)

    body_verts = np.column_stack([x.ravel(), y.ravel(), z.ravel()])

    # Compute pole radii using the actual harmonic modulation
    r_south = _compute_radius_field(
        np.array([np.pi]), np.array([0.0]), coefficients, base_radius,
    )[0]
    r_north = _compute_radius_field(
        np.array([0.0]), np.array([0.0]), coefficients, base_radius,
    )[0]
    south_pole = np.array([[0.0, 0.0, -r_south]])
    north_pole = np.array([[0.0, 0.0, r_north]])
    vertices = np.concatenate([body_verts, south_pole, north_pole], axis=0)
    vertices = vertices.astype(np.float64)

    faces = build_sphere_faces(n, n)
    return Mesh(vertices=vertices, faces=faces)


def _compute_bounding_box(
    coefficients: list[tuple[int, int, float]], base_radius: float,
) -> BoundingBox:
    """Compute conservative bounding box."""
    max_amplitude = sum(abs(c) for _, _, c in coefficients)
    extent = base_radius + max_amplitude
    return BoundingBox(
        min_corner=(-extent, -extent, -extent),
        max_corner=(extent, extent, extent),
    )


@register
class SphericalHarmonicsGenerator(GeneratorBase):
    """Spherical harmonics modulated surface generator."""

    name = "spherical_harmonics"
    category = "parametric"
    aliases = ()
    description = "Sphere modulated by spherical harmonics"
    resolution_params = {"grid_resolution": "Number of grid divisions per axis"}
    _resolution_defaults = {"grid_resolution": _DEFAULT_GRID_RESOLUTION}

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters for spherical harmonics."""
        return {
            "l": _DEFAULT_L,
            "m": _DEFAULT_M,
            "base_radius": _DEFAULT_BASE_RADIUS,
            "amplitude": _DEFAULT_AMPLITUDE,
        }

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate a spherical harmonics surface mesh."""
        merged = self.get_default_params()
        if params:
            merged.update(params)

        base_radius = float(merged["base_radius"])
        coefficients = _parse_coefficients(merged)
        grid_resolution = int(
            resolution_kwargs.get("grid_resolution", _DEFAULT_GRID_RESOLUTION)
        )

        _validate_params(coefficients, base_radius, grid_resolution)

        mesh = _generate_sh_mesh(coefficients, base_radius, grid_resolution)
        bbox = _compute_bounding_box(coefficients, base_radius)

        merged["grid_resolution"] = grid_resolution

        logger.info(
            "Generated spherical_harmonics: base_r=%.3f, n_coeffs=%d, "
            "grid=%d, vertices=%d, faces=%d",
            base_radius, len(coefficients), grid_resolution,
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
