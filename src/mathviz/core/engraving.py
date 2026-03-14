"""Engraving profile configuration model."""

from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator


class EngravingProfile(BaseModel):
    """Fabrication constraints and optical corrections for laser engraving."""

    point_budget: int = Field(default=2_000_000, gt=0, description="Max points for this block")
    min_point_spacing_mm: float = Field(default=0.05, gt=0, description="Minimum point distance")
    max_point_spacing_mm: float = Field(default=2.0, gt=0, description="Maximum point gap")

    occlusion_mode: Literal["none", "shell_fade", "radial_gradient", "custom"] = "none"
    occlusion_shell_layers: int = Field(default=3, gt=0, description="Outer layers to thin")
    occlusion_density_falloff: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="1=maximum thinning of outer layers, 0=uniform density (no falloff)",
    )

    depth_compensation: bool = False
    depth_compensation_factor: float = Field(
        default=1.5,
        gt=0,
        description="Density multiplier at max depth; ignored unless depth_compensation=True",
    )

    @model_validator(mode="after")
    def _check_spacing_order(self) -> Self:
        """Ensure min spacing does not exceed max spacing."""
        if self.min_point_spacing_mm > self.max_point_spacing_mm:
            msg = (
                f"min_point_spacing_mm ({self.min_point_spacing_mm}) "
                f"must not exceed max_point_spacing_mm ({self.max_point_spacing_mm})"
            )
            raise ValueError(msg)
        return self
