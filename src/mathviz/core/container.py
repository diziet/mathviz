"""Physical glass block dimensions and placement policy models."""

from typing import Literal, Optional, Self

from pydantic import BaseModel, Field, model_validator


class Container(BaseModel):
    """Physical glass block dimensions with per-axis margins."""

    width_mm: float = Field(default=100.0, gt=0, description="X-axis dimension in mm")
    height_mm: float = Field(default=100.0, gt=0, description="Y-axis dimension in mm")
    depth_mm: float = Field(default=100.0, gt=0, description="Z-axis dimension in mm")
    margin_x_mm: float = Field(default=5.0, ge=0, description="X-axis margin in mm")
    margin_y_mm: float = Field(default=5.0, ge=0, description="Y-axis margin in mm")
    margin_z_mm: float = Field(default=5.0, ge=0, description="Z-axis margin in mm")

    @model_validator(mode="after")
    def _check_margins_within_dimensions(self) -> Self:
        """Ensure margins don't exceed half the dimension on any axis."""
        for dim, margin, axis in [
            (self.width_mm, self.margin_x_mm, "x"),
            (self.height_mm, self.margin_y_mm, "y"),
            (self.depth_mm, self.margin_z_mm, "z"),
        ]:
            if 2 * margin >= dim:
                msg = f"margin_{axis}_mm ({margin}) must be less than half of dimension ({dim})"
                raise ValueError(msg)
        return self

    @property
    def usable_volume(self) -> tuple[float, float, float]:
        """Return usable dimensions after subtracting margins on each side."""
        return (
            self.width_mm - 2 * self.margin_x_mm,
            self.height_mm - 2 * self.margin_y_mm,
            self.depth_mm - 2 * self.margin_z_mm,
        )

    @classmethod
    def with_uniform_margin(
        cls,
        w: float = 100,
        h: float = 100,
        d: float = 100,
        margin: float = 5,
    ) -> "Container":
        """Create a container with uniform margins on all axes."""
        return cls(
            width_mm=w,
            height_mm=h,
            depth_mm=d,
            margin_x_mm=margin,
            margin_y_mm=margin,
            margin_z_mm=margin,
        )


class PlacementPolicy(BaseModel):
    """Controls how geometry is positioned and scaled within the container."""

    anchor: Literal["center", "front", "back", "top", "bottom", "left", "right"] = "center"
    viewing_axis: Literal["+z", "-z", "+x", "-x", "+y", "-y"] = "+z"
    preserve_aspect_ratio: bool = True
    depth_bias: float = Field(default=1.0, gt=0, description="Depth scaling factor")
    offset_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    scale_override: Optional[float] = Field(default=None, gt=0)
    rotation_degrees: tuple[float, float, float] = (0.0, 0.0, 0.0)
