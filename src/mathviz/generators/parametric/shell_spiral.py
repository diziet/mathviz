"""Shell spiral parametric generator.

A logarithmic spiral with exponentially expanding cross-section, producing
a nautilus-like seashell form. The centerline follows r = exp(growth_rate * θ)
while the tube radius grows as R₀ · exp(opening_rate · θ).
"""

import logging
from typing import Any

import numpy as np

from mathviz.core.generator import GeneratorBase, register
from mathviz.core.math_object import MathObject, Mesh
from mathviz.core.representation import RepresentationConfig, RepresentationType
from mathviz.generators.parametric._mesh_utils import (
    build_mixed_grid_faces,
    compute_padded_bounding_box,
)

logger = logging.getLogger(__name__)

_DEFAULT_GROWTH_RATE = 0.1
_DEFAULT_TURNS = 3
_DEFAULT_OPENING_RATE = 0.08
_DEFAULT_ELLIPTICITY = 1.0
_DEFAULT_CURVE_POINTS = 1024
_DEFAULT_RADIAL_SEGMENTS = 32
_MIN_GROWTH_RATE = 0.001
_MIN_TURNS = 0.5
_MIN_OPENING_RATE = 0.0
_MIN_ELLIPTICITY = 0.1
_MAX_ELLIPTICITY = 5.0
_MIN_CURVE_POINTS = 8
_MIN_RADIAL_SEGMENTS = 4
_INITIAL_TUBE_RADIUS = 0.15


def _compute_spiral_centerline(
    growth_rate: float,
    turns: float,
    curve_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the logarithmic spiral centerline and parameter values."""
    theta = np.linspace(0, turns * 2 * np.pi, curve_points)
    r = np.exp(growth_rate * theta)
    cx = r * np.cos(theta)
    cy = r * np.sin(theta)
    cz = np.zeros_like(theta)
    centerline = np.column_stack([cx, cy, cz])
    return centerline, theta


def _compute_frame_vectors(
    centerline: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute tangent, normal, and binormal vectors along the centerline."""
    tangent = np.gradient(centerline, axis=0)
    t_len = np.linalg.norm(tangent, axis=1, keepdims=True)
    t_len = np.where(t_len < 1e-12, 1.0, t_len)
    tangent = tangent / t_len

    # Planar spiral in xy — binormal is always +z
    binormal = np.zeros_like(tangent)
    binormal[:, 2] = 1.0

    normal = np.cross(binormal, tangent)
    n_len = np.linalg.norm(normal, axis=1, keepdims=True)
    n_len = np.where(n_len < 1e-12, 1.0, n_len)
    normal = normal / n_len

    return tangent, normal, binormal


def _build_shell_spiral_mesh(
    growth_rate: float,
    turns: float,
    opening_rate: float,
    ellipticity: float,
    curve_points: int,
    radial_segments: int,
) -> Mesh:
    """Build triangle mesh for the shell spiral."""
    centerline, theta = _compute_spiral_centerline(
        growth_rate, turns, curve_points,
    )
    _tangent, normal, binormal = _compute_frame_vectors(centerline)

    tube_radius = _INITIAL_TUBE_RADIUS * np.exp(opening_rate * theta)
    phi = np.linspace(0, 2 * np.pi, radial_segments, endpoint=False)

    # Vectorized vertex computation: (curve_points, radial_segments, 3)
    cos_phi = np.cos(phi)  # (radial_segments,)
    sin_phi = np.sin(phi)  # (radial_segments,)

    # Radial offsets scaled by ellipticity on the binormal axis
    r_normal = tube_radius[:, np.newaxis] * cos_phi[np.newaxis, :]
    r_binormal = (
        tube_radius[:, np.newaxis] * ellipticity * sin_phi[np.newaxis, :]
    )

    vertices = (
        centerline[:, np.newaxis, :]
        + r_normal[:, :, np.newaxis] * normal[:, np.newaxis, :]
        + r_binormal[:, :, np.newaxis] * binormal[:, np.newaxis, :]
    )

    vertices = vertices.reshape(-1, 3).astype(np.float64)

    # Open along the curve, wrapped around the cross-section
    faces = build_mixed_grid_faces(
        curve_points, radial_segments, wrap_u=False, wrap_v=True,
    )

    return Mesh(vertices=vertices, faces=faces)


def _validate_params(
    growth_rate: float,
    turns: float,
    opening_rate: float,
    ellipticity: float,
    curve_points: int,
    radial_segments: int,
) -> None:
    """Validate shell spiral parameters."""
    if growth_rate < _MIN_GROWTH_RATE:
        raise ValueError(
            f"growth_rate must be >= {_MIN_GROWTH_RATE}, got {growth_rate}"
        )
    if turns < _MIN_TURNS:
        raise ValueError(f"turns must be >= {_MIN_TURNS}, got {turns}")
    if opening_rate < _MIN_OPENING_RATE:
        raise ValueError(
            f"opening_rate must be >= {_MIN_OPENING_RATE}, got {opening_rate}"
        )
    if ellipticity < _MIN_ELLIPTICITY or ellipticity > _MAX_ELLIPTICITY:
        raise ValueError(
            f"ellipticity must be in [{_MIN_ELLIPTICITY}, {_MAX_ELLIPTICITY}], "
            f"got {ellipticity}"
        )
    if curve_points < _MIN_CURVE_POINTS:
        raise ValueError(
            f"curve_points must be >= {_MIN_CURVE_POINTS}, got {curve_points}"
        )
    if radial_segments < _MIN_RADIAL_SEGMENTS:
        raise ValueError(
            f"radial_segments must be >= {_MIN_RADIAL_SEGMENTS}, "
            f"got {radial_segments}"
        )


@register
class ShellSpiralGenerator(GeneratorBase):
    """Parametric shell spiral generator."""

    name = "shell_spiral"
    category = "parametric"
    aliases = ()
    description = (
        "Logarithmic spiral with exponentially expanding cross-section, "
        "producing a nautilus-like seashell form"
    )
    resolution_params = {
        "curve_points": "Number of points along the spiral centerline",
        "radial_segments": "Number of segments around the cross-section",
    }
    _resolution_defaults = {
        "curve_points": _DEFAULT_CURVE_POINTS,
        "radial_segments": _DEFAULT_RADIAL_SEGMENTS,
    }

    def get_param_ranges(self) -> dict[str, dict[str, float]]:
        """Return exploration ranges for shell spiral parameters."""
        return {
            "growth_rate": {"min": 0.01, "max": 0.5, "step": 0.01},
            "turns": {"min": 0.5, "max": 8.0, "step": 0.5},
            "opening_rate": {"min": 0.0, "max": 0.3, "step": 0.01},
            "ellipticity": {"min": 0.1, "max": 5.0, "step": 0.1},
        }

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters for the shell spiral."""
        return {
            "growth_rate": _DEFAULT_GROWTH_RATE,
            "turns": _DEFAULT_TURNS,
            "opening_rate": _DEFAULT_OPENING_RATE,
            "ellipticity": _DEFAULT_ELLIPTICITY,
        }

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate a shell spiral mesh."""
        merged = self.get_default_params()
        if params:
            merged.update(params)

        growth_rate = float(merged["growth_rate"])
        turns = float(merged["turns"])
        opening_rate = float(merged["opening_rate"])
        ellipticity = float(merged["ellipticity"])
        defaults = self._resolution_defaults
        curve_points = int(
            resolution_kwargs.get("curve_points", defaults["curve_points"])
        )
        radial_segments = int(
            resolution_kwargs.get(
                "radial_segments", defaults["radial_segments"],
            )
        )

        _validate_params(
            growth_rate, turns, opening_rate, ellipticity,
            curve_points, radial_segments,
        )

        mesh = _build_shell_spiral_mesh(
            growth_rate, turns, opening_rate, ellipticity,
            curve_points, radial_segments,
        )
        bbox = compute_padded_bounding_box(mesh.vertices)

        merged["curve_points"] = curve_points
        merged["radial_segments"] = radial_segments

        logger.info(
            "Generated shell_spiral: growth=%.3f, turns=%.1f, "
            "opening=%.3f, ellipticity=%.2f, "
            "vertices=%d, faces=%d",
            growth_rate, turns, opening_rate, ellipticity,
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
        """Return the recommended representation for the shell spiral."""
        return RepresentationConfig(type=RepresentationType.SURFACE_SHELL)
