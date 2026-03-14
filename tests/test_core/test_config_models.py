"""Tests for RepresentationConfig and EngravingProfile models."""

import pytest
from pydantic import ValidationError

from mathviz.core.engraving import EngravingProfile
from mathviz.core.representation import RepresentationConfig, RepresentationType


class TestRepresentationType:
    """Test RepresentationType enum values."""

    def test_all_types_exist(self) -> None:
        """All expected representation types are defined."""
        expected = {
            "surface_shell",
            "tube",
            "raw_point_cloud",
            "volume_fill",
            "sparse_shell",
            "slice_stack",
            "wireframe",
            "weighted_cloud",
            "heightmap_relief",
        }
        actual = {t.value for t in RepresentationType}
        assert actual == expected

    def test_string_enum(self) -> None:
        """RepresentationType is a string enum."""
        assert RepresentationType.TUBE == "tube"


class TestRepresentationConfig:
    """Test RepresentationConfig model."""

    def test_minimal_config(self) -> None:
        """Config with only type is valid."""
        cfg = RepresentationConfig(type=RepresentationType.SURFACE_SHELL)
        assert cfg.type == RepresentationType.SURFACE_SHELL
        assert cfg.tube_radius is None
        assert cfg.tube_sides == 16

    def test_tube_config(self) -> None:
        """Tube config with radius and sides."""
        cfg = RepresentationConfig(
            type=RepresentationType.TUBE,
            tube_radius=0.3,
            tube_sides=8,
        )
        assert cfg.tube_radius == 0.3
        assert cfg.tube_sides == 8

    def test_negative_tube_radius_rejected(self) -> None:
        """Negative tube radius is rejected."""
        with pytest.raises(ValidationError):
            RepresentationConfig(type=RepresentationType.TUBE, tube_radius=-1.0)

    def test_from_string_type(self) -> None:
        """Type can be specified as string."""
        cfg = RepresentationConfig(type="wireframe")
        assert cfg.type == RepresentationType.WIREFRAME


class TestEngravingProfile:
    """Test EngravingProfile model."""

    def test_defaults(self) -> None:
        """Default EngravingProfile values."""
        ep = EngravingProfile()
        assert ep.point_budget == 2_000_000
        assert ep.min_point_spacing_mm == 0.05
        assert ep.max_point_spacing_mm == 2.0
        assert ep.occlusion_mode == "none"
        assert ep.depth_compensation is False

    def test_point_budget_must_be_positive(self) -> None:
        """point_budget must be positive."""
        with pytest.raises(ValidationError):
            EngravingProfile(point_budget=0)

    def test_negative_point_budget_rejected(self) -> None:
        """Negative point_budget is rejected."""
        with pytest.raises(ValidationError):
            EngravingProfile(point_budget=-100)

    def test_custom_occlusion_mode(self) -> None:
        """Custom occlusion mode is accepted."""
        ep = EngravingProfile(occlusion_mode="shell_fade", occlusion_shell_layers=5)
        assert ep.occlusion_mode == "shell_fade"
        assert ep.occlusion_shell_layers == 5

    def test_invalid_occlusion_mode_rejected(self) -> None:
        """Invalid occlusion mode is rejected."""
        with pytest.raises(ValidationError):
            EngravingProfile(occlusion_mode="invalid")

    def test_density_falloff_bounds(self) -> None:
        """Density falloff must be between 0 and 1."""
        EngravingProfile(occlusion_density_falloff=0.0)
        EngravingProfile(occlusion_density_falloff=1.0)
        with pytest.raises(ValidationError):
            EngravingProfile(occlusion_density_falloff=1.5)
        with pytest.raises(ValidationError):
            EngravingProfile(occlusion_density_falloff=-0.1)
