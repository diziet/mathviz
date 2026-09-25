"""Tests that ring_thickness and link_thickness set the tube radius of linked structures."""

import pytest

from mathviz.core.generator import get_generator

_THICKNESS_PARAMS = [
    ("borromean_rings", "ring_thickness"),
    ("chain_links", "link_thickness"),
]


@pytest.mark.parametrize(("generator_name", "param"), _THICKNESS_PARAMS)
@pytest.mark.parametrize("thickness", [0.0, -0.1])
def test_non_positive_thickness_raises(
    generator_name: str, param: str, thickness: float,
) -> None:
    """A thickness of zero or less raises ValueError before any geometry is built."""
    gen = get_generator(generator_name).create(resolved_name=generator_name)
    with pytest.raises(ValueError, match=f"{param} must be positive"):
        gen.generate(params={param: thickness}, curve_points=64)
