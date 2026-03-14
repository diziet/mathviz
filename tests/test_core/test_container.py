"""Tests for Container and PlacementPolicy models."""

import pytest
from pydantic import ValidationError

from mathviz.core.container import Container, PlacementPolicy


class TestContainerUsableVolume:
    """Test usable_volume property with various margin configurations."""

    def test_non_uniform_margins(self) -> None:
        """Usable volume with non-uniform margins computes correctly."""
        c = Container(
            width_mm=100,
            height_mm=80,
            depth_mm=40,
            margin_x_mm=5,
            margin_y_mm=10,
            margin_z_mm=3,
        )
        assert c.usable_volume == (90.0, 60.0, 34.0)

    def test_cube_container(self) -> None:
        """Container with equal width/height/depth computes correct usable volume."""
        c = Container(
            width_mm=50,
            height_mm=50,
            depth_mm=50,
            margin_x_mm=5,
            margin_y_mm=5,
            margin_z_mm=5,
        )
        assert c.usable_volume == (40.0, 40.0, 40.0)

    def test_zero_margin_on_one_axis(self) -> None:
        """Container with zero margin on z-axis works."""
        c = Container(
            width_mm=100,
            height_mm=100,
            depth_mm=40,
            margin_x_mm=5,
            margin_y_mm=5,
            margin_z_mm=0,
        )
        assert c.usable_volume == (90.0, 90.0, 40.0)

    def test_default_container(self) -> None:
        """Default container has expected usable volume."""
        c = Container()
        assert c.usable_volume == (90.0, 90.0, 30.0)


class TestContainerWithUniformMargin:
    """Test the with_uniform_margin classmethod."""

    def test_default_args(self) -> None:
        """Default arguments produce standard container."""
        c = Container.with_uniform_margin()
        assert c.width_mm == 100
        assert c.height_mm == 100
        assert c.depth_mm == 40
        assert c.margin_x_mm == 5
        assert c.margin_y_mm == 5
        assert c.margin_z_mm == 5

    def test_custom_dimensions(self) -> None:
        """Custom dimensions with uniform margin."""
        c = Container.with_uniform_margin(w=200, h=150, d=80, margin=10)
        assert c.usable_volume == (180.0, 130.0, 60.0)


class TestContainerValidation:
    """Test Pydantic validation on Container fields."""

    def test_negative_dimension_rejected(self) -> None:
        """Negative dimensions are rejected."""
        with pytest.raises(ValidationError):
            Container(width_mm=-10)

    def test_zero_dimension_rejected(self) -> None:
        """Zero dimensions are rejected."""
        with pytest.raises(ValidationError):
            Container(depth_mm=0)

    def test_negative_margin_rejected(self) -> None:
        """Negative margins are rejected."""
        with pytest.raises(ValidationError):
            Container(margin_x_mm=-1)

    def test_margin_exceeding_half_dimension_rejected(self) -> None:
        """Margins >= half the dimension are rejected."""
        with pytest.raises(ValidationError):
            Container(width_mm=10, margin_x_mm=20)

    def test_margin_equal_to_half_dimension_rejected(self) -> None:
        """Margin exactly half the dimension is rejected (zero usable volume)."""
        with pytest.raises(ValidationError):
            Container(width_mm=10, margin_x_mm=5)


class TestPlacementPolicyDefaults:
    """Test PlacementPolicy default values."""

    def test_defaults(self) -> None:
        """Default PlacementPolicy has center anchor, +z viewing, aspect ratio preserved."""
        p = PlacementPolicy()
        assert p.anchor == "center"
        assert p.viewing_axis == "+z"
        assert p.preserve_aspect_ratio is True
        assert p.depth_bias == 1.0
        assert p.offset_mm == (0.0, 0.0, 0.0)
        assert p.scale_override is None
        assert p.rotation_degrees == (0.0, 0.0, 0.0)

    def test_custom_anchor(self) -> None:
        """Custom anchor value is accepted."""
        p = PlacementPolicy(anchor="top")
        assert p.anchor == "top"

    def test_invalid_anchor_rejected(self) -> None:
        """Invalid anchor literal is rejected."""
        with pytest.raises(ValidationError):
            PlacementPolicy(anchor="middle")

    def test_invalid_viewing_axis_rejected(self) -> None:
        """Invalid viewing axis is rejected."""
        with pytest.raises(ValidationError):
            PlacementPolicy(viewing_axis="+w")
