"""Mandelbulb 3D fractal generator.

Evaluates the Mandelbulb escape-time field on a voxel grid and extracts the
boundary surface via marching cubes. The inner iteration kernel is
numba-JIT-compiled for acceptable performance at voxel_resolution >= 128.
Default representation: SPARSE_SHELL.

Seed is stored for metadata/provenance only — the Mandelbulb computation
is fully deterministic for given parameters.
"""

import logging
from typing import Any

from mathviz.core.generator import GeneratorBase, register
from mathviz.core.math_object import BoundingBox, MathObject
from mathviz.core.representation import RepresentationConfig, RepresentationType
from mathviz.generators.fractals._common import (
    DEFAULT_EXTENT,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_POWER,
    DEFAULT_VOXEL_RESOLUTION,
    ESCAPE_ISOLEVEL,
    build_voxel_grid,
    validate_fractal_params,
)
from mathviz.generators.fractals._escape_kernel import mandelbulb_escape_field
from mathviz.shared.marching_cubes import extract_mesh

logger = logging.getLogger(__name__)


@register
class MandelbulbGenerator(GeneratorBase):
    """Mandelbulb 3D fractal via escape-time iteration and marching cubes."""

    name = "mandelbulb"
    category = "fractals"
    aliases = ()
    description = "Mandelbulb 3D fractal with numba-JIT escape-time kernel"
    resolution_params = {
        "voxel_resolution": "Voxels per axis (N³ cost)",
    }
    _resolution_defaults = {"voxel_resolution": DEFAULT_VOXEL_RESOLUTION}

    def get_param_ranges(self) -> dict[str, dict[str, float]]:
        """Return exploration ranges for the Mandelbulb parameters."""
        return {
            "power": {"min": 2.0, "max": 16.0, "step": 0.5},
            "max_iterations": {"min": 3, "max": 30, "step": 1},
            "extent": {"min": 0.8, "max": 2.5, "step": 0.1},
        }

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters for the Mandelbulb."""
        return {
            "power": DEFAULT_POWER,
            "max_iterations": DEFAULT_MAX_ITERATIONS,
            "extent": DEFAULT_EXTENT,
        }

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate a Mandelbulb mesh via marching cubes.

        Seed is stored for metadata only — output is fully deterministic.
        """
        merged = self.get_default_params()
        if params:
            merged.update(params)

        power = float(merged["power"])
        max_iterations = int(merged["max_iterations"])
        extent = float(merged["extent"])
        voxel_resolution = int(
            resolution_kwargs.get("voxel_resolution", DEFAULT_VOXEL_RESOLUTION)
        )

        validate_fractal_params(power, max_iterations, voxel_resolution, extent)
        merged["voxel_resolution"] = voxel_resolution

        xs, ys, zs, bounds = build_voxel_grid(voxel_resolution, extent)
        field = mandelbulb_escape_field(xs, ys, zs, max_iterations, power)
        mesh = extract_mesh(field, bounds, isolevel=ESCAPE_ISOLEVEL)

        bbox = BoundingBox(
            min_corner=bounds.min_corner,
            max_corner=bounds.max_corner,
        )

        logger.info(
            "Generated mandelbulb: power=%.1f, max_iter=%d, "
            "voxel_res=%d, vertices=%d, faces=%d",
            power, max_iterations, voxel_resolution,
            len(mesh.vertices), len(mesh.faces),
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
        """Return SPARSE_SHELL as default representation."""
        return RepresentationConfig(type=RepresentationType.SPARSE_SHELL)
