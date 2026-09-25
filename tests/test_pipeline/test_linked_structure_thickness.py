"""Tests that ring_thickness and link_thickness set the tube radius of linked structures."""

import numpy as np
import pytest

from mathviz.core.container import Container, PlacementPolicy
from mathviz.core.generator import (
    GeneratorBase,
    clear_registry,
    get_generator,
    register,
)
from mathviz.core.representation import RepresentationConfig, RepresentationType
from mathviz.generators.knots.linked_structures import BorromeanRingsGenerator
from mathviz.pipeline.representation_defaults import GENERATOR_DEFAULTS
from mathviz.pipeline.representation_strategy import get_default
from mathviz.pipeline.runner import run

_THICKNESS_PARAMS = [
    ("borromean_rings", "ring_thickness"),
    ("chain_links", "link_thickness"),
]
_CURVE_POINTS = 64
_CUSTOM_THICKNESS = 0.2


def _create(generator_name: str) -> GeneratorBase:
    """Create the generator registered under generator_name."""
    return get_generator(generator_name).create(resolved_name=generator_name)


def _run_mesh_vertices(
    generator_name: str,
    params: dict[str, float] | None = None,
    tube_radius: float | None = None,
) -> np.ndarray:
    """Run the pipeline and return the mesh vertices, with an explicit TUBE config if given."""
    representation_config = None
    if tube_radius is not None:
        representation_config = RepresentationConfig(
            type=RepresentationType.TUBE, tube_radius=tube_radius,
        )
    result = run(
        generator_name,
        params=params,
        resolution_kwargs={"curve_points": _CURVE_POINTS},
        container=Container.with_uniform_margin(),
        placement=PlacementPolicy(),
        representation_config=representation_config,
    )
    return result.math_object.mesh.vertices


@pytest.mark.parametrize(("generator_name", "param"), _THICKNESS_PARAMS)
@pytest.mark.parametrize("thickness", [0.0, -0.1])
def test_non_positive_thickness_raises(
    generator_name: str, param: str, thickness: float,
) -> None:
    """A thickness of zero or less raises ValueError before any geometry is built."""
    with pytest.raises(ValueError, match=f"{param} must be positive"):
        _create(generator_name).generate(
            params={param: thickness}, curve_points=_CURVE_POINTS,
        )


@pytest.mark.parametrize(("generator_name", "param"), _THICKNESS_PARAMS)
def test_default_tube_radius_equals_thickness_param(
    generator_name: str, param: str,
) -> None:
    """get_default sets tube_radius to the thickness parameter of the generated object."""
    obj = _create(generator_name).generate(
        params={param: _CUSTOM_THICKNESS}, curve_points=_CURVE_POINTS,
    )
    config = get_default(obj.generator_name, obj=obj)
    assert config.type == RepresentationType.TUBE
    assert config.tube_radius == _CUSTOM_THICKNESS


@pytest.mark.parametrize(("generator_name", "param"), _THICKNESS_PARAMS)
def test_default_tube_radius_matches_advertised_thickness_default(
    generator_name: str, param: str,
) -> None:
    """Without a thickness param, get_default uses the default from get_default_params."""
    gen = _create(generator_name)
    advertised = gen.get_default_params()[param]
    obj = gen.generate(curve_points=_CURVE_POINTS)
    assert get_default(obj.generator_name, obj=obj).tube_radius == advertised
    assert get_default(generator_name).tube_radius == advertised


def test_default_tube_radius_reads_thickness_through_alias() -> None:
    """An object generated under an alias still takes its tube radius from ring_thickness."""
    clear_registry(suppress_discovery=True)
    register(aliases=["borromean_alias"])(BorromeanRingsGenerator)
    obj = _create("borromean_alias").generate(
        params={"ring_thickness": _CUSTOM_THICKNESS}, curve_points=_CURVE_POINTS,
    )
    assert obj.generator_name == "borromean_alias"
    assert get_default(obj.generator_name, obj=obj).tube_radius == _CUSTOM_THICKNESS


def test_thickness_param_does_not_change_shared_default_config() -> None:
    """A custom link_thickness leaves the config chain_links shares with torus_knot unchanged."""
    shared_radius = GENERATOR_DEFAULTS["torus_knot"].tube_radius
    obj = _create("chain_links").generate(
        params={"link_thickness": _CUSTOM_THICKNESS}, curve_points=_CURVE_POINTS,
    )
    get_default(obj.generator_name, obj=obj)
    assert GENERATOR_DEFAULTS["chain_links"].tube_radius == shared_radius
    assert GENERATOR_DEFAULTS["torus_knot"].tube_radius == shared_radius


@pytest.mark.parametrize(("generator_name", "param"), _THICKNESS_PARAMS)
def test_run_thickness_param_matches_explicit_tube_radius(
    generator_name: str, param: str,
) -> None:
    """run() with a thickness param builds the mesh that an explicit tube_radius builds."""
    from_param = _run_mesh_vertices(generator_name, params={param: _CUSTOM_THICKNESS})
    from_config = _run_mesh_vertices(generator_name, tube_radius=_CUSTOM_THICKNESS)
    np.testing.assert_array_equal(from_param, from_config)


@pytest.mark.parametrize(("generator_name", "param"), _THICKNESS_PARAMS)
def test_run_explicit_tube_radius_overrides_thickness_param(
    generator_name: str, param: str,
) -> None:
    """An explicit representation_config tube_radius takes precedence over the thickness param."""
    explicit_radius = 0.05
    with_param = _run_mesh_vertices(
        generator_name, params={param: _CUSTOM_THICKNESS}, tube_radius=explicit_radius,
    )
    without_param = _run_mesh_vertices(generator_name, tube_radius=explicit_radius)
    np.testing.assert_array_equal(with_param, without_param)
