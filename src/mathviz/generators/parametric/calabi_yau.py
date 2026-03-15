"""Calabi-Yau manifold cross-section generator.

Generates the iconic string-theory shape by projecting a complex algebraic
surface satisfying z1^n + z2^n = 1 in C² down to R³. Multiple patches
(one per value of k in 0..n-1) are combined to form the crystalline
flower-like geometry.
"""

import logging
from typing import Any

import numpy as np

from mathviz.core.generator import GeneratorBase, register
from mathviz.core.math_object import BoundingBox, MathObject, Mesh
from mathviz.core.representation import RepresentationConfig, RepresentationType
from mathviz.generators.parametric._mesh_utils import build_open_grid_faces

logger = logging.getLogger(__name__)

_DEFAULT_N = 5
_DEFAULT_ALPHA = np.pi / 4.0
_DEFAULT_GRID_RESOLUTION = 128
_MIN_GRID_RESOLUTION = 4
_MIN_N = 2
_MAX_N = 20


def _complex_power(z: np.ndarray, exponent: float) -> np.ndarray:
    """Raise complex array to a real power, preserving branch structure."""
    r = np.abs(z)
    theta = np.angle(z)
    return (r ** exponent) * np.exp(1.0j * exponent * theta)


def _evaluate_patch(
    u: np.ndarray,
    v: np.ndarray,
    k: int,
    n: int,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate one patch of the Calabi-Yau surface.

    The surface satisfies z1^n + z2^n = 1 in C². We parameterize as:
        z1 = exp(2*pi*i*k/n) * (cos(u) * exp(i*v))^(2/n)
        z2 = exp(2*pi*i*k/n) * (sin(u) * exp(i*v))^(2/n)
    then project to 3D via Re/Im mixing controlled by alpha.
    """
    phase = np.exp(2.0j * np.pi * k / n)
    exponent = 2.0 / n

    cos_u = np.cos(u)
    sin_u = np.sin(u)

    # Complex bases for z1 and z2
    z1_base = cos_u * np.exp(1.0j * v)
    z2_base = sin_u * np.exp(1.0j * v)

    # Raise to power 2/n (handle sign for real parts via abs + sign)
    z1 = phase * _complex_power(z1_base, exponent)
    z2 = phase * _complex_power(z2_base, exponent)

    # Project to 3D
    x1 = z1.real
    x2 = z1.imag
    y1 = z2.real
    y2 = z2.imag

    # Compute z_val from the algebraic relation
    z_val = x2 + 1.0j * y2

    x_out = x1
    y_out = y1
    z_out = np.cos(alpha) * z_val.real + np.sin(alpha) * z_val.imag

    return x_out, y_out, z_out


def _validate_params(n: int, alpha: float, grid_resolution: int) -> None:
    """Validate Calabi-Yau parameters."""
    if n < _MIN_N:
        raise ValueError(f"n must be >= {_MIN_N}, got {n}")
    if n > _MAX_N:
        raise ValueError(f"n must be <= {_MAX_N}, got {n}")
    if grid_resolution < _MIN_GRID_RESOLUTION:
        raise ValueError(
            f"grid_resolution must be >= {_MIN_GRID_RESOLUTION}, "
            f"got {grid_resolution}"
        )
    if not np.isfinite(alpha):
        raise ValueError(f"alpha must be finite, got {alpha}")


def _generate_calabi_yau_mesh(
    n: int, alpha: float, grid_resolution: int,
) -> Mesh:
    """Build combined triangle mesh for all n patches."""
    res = grid_resolution
    u_vals = np.linspace(0, np.pi / 2, res)
    v_vals = np.linspace(0, 2.0 * np.pi, res)
    uu, vv = np.meshgrid(u_vals, v_vals, indexing="ij")

    all_vertices = []
    all_faces = []
    vertex_offset = 0
    single_patch_faces = build_open_grid_faces(res, res)

    for k in range(n):
        x, y, z = _evaluate_patch(uu, vv, k, n, alpha)
        patch_verts = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
        all_vertices.append(patch_verts)
        all_faces.append(single_patch_faces + vertex_offset)
        vertex_offset += len(patch_verts)

    vertices = np.concatenate(all_vertices, axis=0).astype(np.float64)
    faces = np.concatenate(all_faces, axis=0).astype(np.int64)
    return Mesh(vertices=vertices, faces=faces)


def _compute_bounding_box() -> BoundingBox:
    """Compute conservative bounding box for Calabi-Yau surface."""
    # Conservative envelope: surface fits within ±2.0 for any valid n
    extent = 2.0
    return BoundingBox(
        min_corner=(-extent, -extent, -extent),
        max_corner=(extent, extent, extent),
    )


@register
class CalabiYauGenerator(GeneratorBase):
    """Calabi-Yau manifold cross-section generator."""

    name = "calabi_yau"
    category = "parametric"
    aliases = ("calabi_yau_manifold",)
    description = (
        "Calabi-Yau manifold cross-section — crystalline flower-like "
        "string theory shape from z1^n + z2^n = 1 in C²"
    )
    resolution_params = {"grid_resolution": "Number of grid divisions per axis"}
    _resolution_defaults = {"grid_resolution": _DEFAULT_GRID_RESOLUTION}

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters for Calabi-Yau surface."""
        return {
            "n": _DEFAULT_N,
            "alpha": _DEFAULT_ALPHA,
        }

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate a Calabi-Yau manifold cross-section mesh."""
        merged = self.get_default_params()
        if params:
            merged.update(params)

        n = int(merged["n"])
        alpha = float(merged["alpha"])
        grid_resolution = int(
            resolution_kwargs.get("grid_resolution", _DEFAULT_GRID_RESOLUTION)
        )

        _validate_params(n, alpha, grid_resolution)

        mesh = _generate_calabi_yau_mesh(n, alpha, grid_resolution)
        bbox = _compute_bounding_box()

        merged["grid_resolution"] = grid_resolution

        logger.info(
            "Generated calabi_yau: n=%d, alpha=%.3f, grid=%d, "
            "vertices=%d, faces=%d",
            n, alpha, grid_resolution,
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
        """Return the recommended representation for Calabi-Yau surface."""
        return RepresentationConfig(type=RepresentationType.SURFACE_SHELL)
