"""Representation type enum and configuration model."""

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


class RepresentationType(str, Enum):
    """How raw geometry should be realized for engraving."""

    SURFACE_SHELL = "surface_shell"
    TUBE = "tube"
    RAW_POINT_CLOUD = "raw_point_cloud"
    VOLUME_FILL = "volume_fill"
    SPARSE_SHELL = "sparse_shell"
    SLICE_STACK = "slice_stack"
    WIREFRAME = "wireframe"
    WEIGHTED_CLOUD = "weighted_cloud"
    HEIGHTMAP_RELIEF = "heightmap_relief"


class RepresentationConfig(BaseModel):
    """Configuration for how geometry is represented for engraving."""

    type: RepresentationType
    tube_radius: Optional[float] = Field(default=None, gt=0)
    tube_sides: Optional[int] = Field(default=16, gt=0)
    shell_thickness: Optional[float] = Field(default=None, gt=0)
    volume_density: Optional[float] = Field(default=None, gt=0)
    slice_count: Optional[int] = Field(default=None, gt=0)
    slice_axis: Literal["x", "y", "z"] = "z"
    wireframe_thickness: Optional[float] = Field(default=None, gt=0)
    density_weight_function: Optional[str] = None
