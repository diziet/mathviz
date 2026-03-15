"""Quaternion Julia set generator.

Iterates q -> q² + c in quaternion space (4D) and extracts a 3D isosurface
by fixing the 4th quaternion component (slice_w). Produces smoother, more
organic shapes than the Mandelbulb. Uses marching cubes on a 3D voxel grid
to extract the boundary surface.

Default representation: SURFACE_SHELL.

Seed is stored for metadata/provenance only — the computation is fully
deterministic for given parameters.
"""

import logging
from typing import Any

from mathviz.core.generator import GeneratorBase, register
from mathviz.core.math_object import BoundingBox, MathObject
from mathviz.core.representation import RepresentationConfig, RepresentationType
from mathviz.generators.fractals._common import (
    DEFAULT_EXTENT,
    DEFAULT_VOXEL_RESOLUTION,
    ESCAPE_ISOLEVEL,
    build_voxel_grid,
    validate_escape_radius,
    validate_fractal_params,
    validate_quaternion_c_params,
)
from mathviz.generators.fractals._escape_kernel import quaternion_julia_escape_field
from mathviz.shared.marching_cubes import extract_mesh

logger = logging.getLogger(__name__)

_DEFAULT_C_REAL = -0.2
_DEFAULT_C_I = 0.8
_DEFAULT_C_J = 0.0
_DEFAULT_C_K = 0.0
_DEFAULT_MAX_ITER = 10
_DEFAULT_ESCAPE_RADIUS = 2.0
_DEFAULT_SLICE_W = 0.0

# Quaternion Julia doesn't use the spherical "power" parameter,
# but validate_fractal_params requires one. Use a dummy value.
_DUMMY_POWER = 2.0


@register
class QuaternionJuliaGenerator(GeneratorBase):
    """Quaternion Julia set via escape-time iteration and marching cubes."""

    name = "quaternion_julia"
    category = "fractals"
    aliases = ("qjulia",)
    description = "Quaternion Julia set — 4D fractal sliced to 3D"
    resolution_params = {
        "voxel_resolution": "Voxels per axis (N³ cost)",
    }
    _resolution_defaults = {"voxel_resolution": DEFAULT_VOXEL_RESOLUTION}

    def get_param_ranges(self) -> dict[str, dict[str, float]]:
        """Return exploration ranges for quaternion Julia parameters."""
        return {
            "c_real": {"min": -2.0, "max": 2.0, "step": 0.05},
            "c_i": {"min": -2.0, "max": 2.0, "step": 0.05},
            "c_j": {"min": -2.0, "max": 2.0, "step": 0.05},
            "c_k": {"min": -2.0, "max": 2.0, "step": 0.05},
            "max_iter": {"min": 3, "max": 30, "step": 1},
            "escape_radius": {"min": 1.0, "max": 10.0, "step": 0.5},
            "slice_w": {"min": -1.0, "max": 1.0, "step": 0.05},
        }

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters for the quaternion Julia set."""
        return {
            "c_real": _DEFAULT_C_REAL,
            "c_i": _DEFAULT_C_I,
            "c_j": _DEFAULT_C_J,
            "c_k": _DEFAULT_C_K,
            "max_iter": _DEFAULT_MAX_ITER,
            "escape_radius": _DEFAULT_ESCAPE_RADIUS,
            "extent": DEFAULT_EXTENT,
            "slice_w": _DEFAULT_SLICE_W,
        }

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate a quaternion Julia mesh via marching cubes."""
        merged = self.get_default_params()
        if params:
            merged.update(params)

        c_real = float(merged["c_real"])
        c_i = float(merged["c_i"])
        c_j = float(merged["c_j"])
        c_k = float(merged["c_k"])
        max_iter = int(merged["max_iter"])
        escape_radius = float(merged["escape_radius"])
        extent = float(merged["extent"])
        slice_w = float(merged["slice_w"])
        voxel_resolution = int(
            resolution_kwargs.get("voxel_resolution", DEFAULT_VOXEL_RESOLUTION)
        )

        validate_fractal_params(
            _DUMMY_POWER, max_iter, voxel_resolution, extent,
        )
        validate_quaternion_c_params(c_real, c_i, c_j, c_k)
        validate_escape_radius(escape_radius)
        merged["voxel_resolution"] = voxel_resolution

        xs, ys, zs, bounds = build_voxel_grid(voxel_resolution, extent)
        field = quaternion_julia_escape_field(
            xs, ys, zs, slice_w,
            max_iter, escape_radius,
            c_real, c_i, c_j, c_k,
        )
        mesh = extract_mesh(field, bounds, isolevel=ESCAPE_ISOLEVEL)

        bbox = BoundingBox(
            min_corner=bounds.min_corner,
            max_corner=bounds.max_corner,
        )

        logger.info(
            "Generated quaternion_julia: c=(%.3f, %.3f, %.3f, %.3f), "
            "max_iter=%d, escape_r=%.1f, slice_w=%.3f, "
            "voxel_res=%d, vertices=%d, faces=%d",
            c_real, c_i, c_j, c_k, max_iter, escape_radius, slice_w,
            voxel_resolution, len(mesh.vertices), len(mesh.faces),
        )

        return MathObject(
            mesh=mesh,
            generator_name=self.resolved_name or self.name,
            category=self.category,
            parameters=merged,
            seed=seed,
            bounding_box=bbox,
        )

    def get_default_representation(self) -> RepresentationConfig:
        """Return SURFACE_SHELL as default representation."""
        return RepresentationConfig(type=RepresentationType.SURFACE_SHELL)
