"""Shared base class for triply periodic minimal surface (TPMS) generators.

All TPMS generators share the same parameters (cell_size, periods,
voxel_resolution), bounds computation, validation, and generate() orchestration.
Each subclass only needs to supply its unique scalar field evaluation function.
"""

import logging
from abc import abstractmethod
from typing import Any

import numpy as np

from mathviz.core.generator import GeneratorBase
from mathviz.core.math_object import MathObject
from mathviz.core.representation import RepresentationConfig, RepresentationType
from mathviz.shared.marching_cubes import SpatialBounds, bounds_to_bbox, extract_mesh

logger = logging.getLogger(__name__)

DEFAULT_CELL_SIZE = 1.0
DEFAULT_PERIODS = 2
DEFAULT_VOXEL_RESOLUTION = 128
MIN_VOXEL_RESOLUTION = 4
MIN_PERIODS = 1


def compute_tpms_bounds(cell_size: float, periods: int) -> SpatialBounds:
    """Compute cubic spatial bounds for the given number of TPMS periods."""
    extent = cell_size * periods * 2 * np.pi
    half = extent / 2.0
    return SpatialBounds(
        min_corner=(-half, -half, -half),
        max_corner=(half, half, half),
    )


def validate_tpms_params(
    cell_size: float, periods: int, voxel_resolution: int,
) -> None:
    """Validate TPMS parameters, raising ValueError for invalid inputs."""
    if cell_size <= 0:
        raise ValueError(f"cell_size must be positive, got {cell_size}")
    if periods < MIN_PERIODS:
        raise ValueError(
            f"periods must be >= {MIN_PERIODS}, got {periods}"
        )
    if voxel_resolution < MIN_VOXEL_RESOLUTION:
        raise ValueError(
            f"voxel_resolution must be >= {MIN_VOXEL_RESOLUTION}, "
            f"got {voxel_resolution}"
        )


class TPMSGeneratorBase(GeneratorBase):
    """Base class for TPMS generators sharing cell_size/periods/voxel_resolution.

    Subclasses must implement ``evaluate_field`` to provide the scalar field
    for their specific TPMS equation. All other logic (parameter handling,
    bounds, validation, marching cubes, metadata) is handled here.
    """

    category = "implicit"
    resolution_params = {
        "voxel_resolution": "Number of voxels per axis (N³ cost)",
    }

    def get_default_resolution(self) -> dict[str, Any]:
        """Return default values for resolution parameters."""
        return {"voxel_resolution": DEFAULT_VOXEL_RESOLUTION}

    @abstractmethod
    def evaluate_field(
        self, voxel_resolution: int, bounds: SpatialBounds,
    ) -> np.ndarray:
        """Evaluate the TPMS scalar field on an N³ voxel grid."""
        ...

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters for this TPMS generator."""
        return {
            "cell_size": DEFAULT_CELL_SIZE,
            "periods": DEFAULT_PERIODS,
        }

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate a TPMS mesh via marching cubes on the implicit field."""
        merged = self.get_default_params()
        if params:
            merged.update(params)

        cell_size = float(merged["cell_size"])
        periods = int(merged["periods"])
        voxel_resolution = int(
            resolution_kwargs.get("voxel_resolution", DEFAULT_VOXEL_RESOLUTION)
        )

        validate_tpms_params(cell_size, periods, voxel_resolution)
        merged["voxel_resolution"] = voxel_resolution

        bounds = compute_tpms_bounds(cell_size, periods)
        field = self.evaluate_field(voxel_resolution, bounds)
        mesh = extract_mesh(field, bounds, isolevel=0.0)
        bbox = bounds_to_bbox(bounds)

        logger.info(
            "Generated %s: cell_size=%.3f, periods=%d, "
            "voxel_resolution=%d, vertices=%d, faces=%d",
            self.name, cell_size, periods, voxel_resolution,
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
        """Return the recommended representation for TPMS surfaces."""
        return RepresentationConfig(type=RepresentationType.SURFACE_SHELL)
