"""Julia 3D fractal generator.

Same iteration as the Mandelbulb but with a fixed c parameter instead of using
the voxel coordinate as c. The inner kernel is numba-JIT-compiled.
Default representation: SPARSE_SHELL.

Seed is stored for metadata/provenance only — the Julia 3D computation
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
    validate_c_params,
    validate_fractal_params,
)
from mathviz.generators.fractals._escape_kernel import julia3d_escape_field
from mathviz.shared.marching_cubes import extract_mesh

logger = logging.getLogger(__name__)

_DEFAULT_C_RE = -0.2
_DEFAULT_C_IM = 0.6
_DEFAULT_C_Z = 0.2


@register
class Julia3DGenerator(GeneratorBase):
    """Julia 3D fractal via escape-time iteration and marching cubes."""

    name = "julia3d"
    category = "fractals"
    aliases = ("julia_3d",)
    description = "3D Julia set fractal with numba-JIT escape-time kernel"
    resolution_params = {
        "voxel_resolution": "Voxels per axis (N³ cost)",
    }

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters for Julia 3D."""
        return {
            "power": DEFAULT_POWER,
            "max_iterations": DEFAULT_MAX_ITERATIONS,
            "extent": DEFAULT_EXTENT,
            "c_re": _DEFAULT_C_RE,
            "c_im": _DEFAULT_C_IM,
            "c_z": _DEFAULT_C_Z,
        }

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate a Julia 3D mesh via marching cubes.

        Seed is stored for metadata only — output is fully deterministic.
        """
        merged = self.get_default_params()
        if params:
            merged.update(params)

        power = float(merged["power"])
        max_iterations = int(merged["max_iterations"])
        extent = float(merged["extent"])
        c_re = float(merged["c_re"])
        c_im = float(merged["c_im"])
        c_z = float(merged["c_z"])
        voxel_resolution = int(
            resolution_kwargs.get("voxel_resolution", DEFAULT_VOXEL_RESOLUTION)
        )

        validate_fractal_params(power, max_iterations, voxel_resolution, extent)
        validate_c_params(c_re, c_im, c_z)
        merged["voxel_resolution"] = voxel_resolution

        xs, ys, zs, bounds = build_voxel_grid(voxel_resolution, extent)
        field = julia3d_escape_field(
            xs, ys, zs, max_iterations, power, c_re, c_im, c_z,
        )
        mesh = extract_mesh(field, bounds, isolevel=ESCAPE_ISOLEVEL)

        bbox = BoundingBox(
            min_corner=bounds.min_corner,
            max_corner=bounds.max_corner,
        )

        logger.info(
            "Generated julia3d: power=%.1f, c=(%.3f, %.3f, %.3f), "
            "max_iter=%d, voxel_res=%d, vertices=%d, faces=%d",
            power, c_re, c_im, c_z, max_iterations, voxel_resolution,
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
