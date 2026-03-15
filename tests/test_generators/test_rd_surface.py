"""Tests for the reaction-diffusion surface generator (rd_surface)."""

import numpy as np
import pytest

from mathviz.core.generator import clear_registry, list_generators, register
from mathviz.core.representation import RepresentationType
from mathviz.generators.procedural.rd_surface import ReactionDiffusionSurface

_TEST_RESOLUTION = 16
_TEST_ITERATIONS = 500


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset registry and register rd_surface for each test."""
    clear_registry(suppress_discovery=True)
    register(ReactionDiffusionSurface)
    yield
    clear_registry(suppress_discovery=True)


def _make_generator() -> ReactionDiffusionSurface:
    """Create a generator instance."""
    return ReactionDiffusionSurface()


def _default_params(**overrides: object) -> dict:
    """Return test params with small resolution and iteration count."""
    base = {"iterations": _TEST_ITERATIONS}
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Valid mesh output with expected vertex count
# ---------------------------------------------------------------------------


def test_produces_valid_mesh() -> None:
    """Generator produces a valid mesh with expected vertex count."""
    gen = _make_generator()
    obj = gen.generate(
        params=_default_params(),
        seed=42,
        grid_resolution=_TEST_RESOLUTION,
    )
    obj.validate_or_raise()

    assert obj.mesh is not None
    n = _TEST_RESOLUTION
    expected_verts = n * n  # torus default: n*n vertices
    assert len(obj.mesh.vertices) == expected_verts
    assert obj.mesh.faces.shape[1] == 3
    assert obj.mesh.normals is not None
    assert len(obj.mesh.normals) == expected_verts


def test_sphere_produces_valid_mesh() -> None:
    """Sphere base surface produces valid mesh with pole vertices."""
    gen = _make_generator()
    obj = gen.generate(
        params=_default_params(base_surface="sphere"),
        seed=42,
        grid_resolution=_TEST_RESOLUTION,
    )
    obj.validate_or_raise()

    assert obj.mesh is not None
    n = _TEST_RESOLUTION
    # Sphere: n_lat*n_lon body + 2 poles
    expected_verts = n * n + 2
    assert len(obj.mesh.vertices) == expected_verts


# ---------------------------------------------------------------------------
# Different feed/kill rates produce different displacements
# ---------------------------------------------------------------------------


def test_different_fk_rates_produce_different_displacements() -> None:
    """Different feed/kill rates produce different vertex displacements."""
    gen = _make_generator()
    # Spots preset
    obj_spots = gen.generate(
        params=_default_params(feed_rate=0.035, kill_rate=0.065),
        seed=42,
        grid_resolution=_TEST_RESOLUTION,
    )
    # Stripes preset
    obj_stripes = gen.generate(
        params=_default_params(feed_rate=0.055, kill_rate=0.062),
        seed=42,
        grid_resolution=_TEST_RESOLUTION,
    )

    assert obj_spots.mesh is not None
    assert obj_stripes.mesh is not None
    assert not np.allclose(
        obj_spots.mesh.vertices, obj_stripes.mesh.vertices, atol=1e-6,
    )


# ---------------------------------------------------------------------------
# Seed-dependent output
# ---------------------------------------------------------------------------


def test_same_seed_identical() -> None:
    """Same seed produces identical output."""
    gen = _make_generator()
    kwargs = {
        "params": _default_params(),
        "seed": 77,
        "grid_resolution": _TEST_RESOLUTION,
    }
    obj1 = gen.generate(**kwargs)
    obj2 = gen.generate(**kwargs)

    assert obj1.mesh is not None
    assert obj2.mesh is not None
    np.testing.assert_array_equal(obj1.mesh.vertices, obj2.mesh.vertices)


def test_different_seeds_differ() -> None:
    """Different seeds produce different meshes."""
    gen = _make_generator()
    obj1 = gen.generate(
        params=_default_params(), seed=1, grid_resolution=_TEST_RESOLUTION,
    )
    obj2 = gen.generate(
        params=_default_params(), seed=2, grid_resolution=_TEST_RESOLUTION,
    )

    assert obj1.mesh is not None
    assert obj2.mesh is not None
    assert not np.array_equal(obj1.mesh.vertices, obj2.mesh.vertices)


# ---------------------------------------------------------------------------
# Base surface switching
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("surface", ["torus", "sphere", "klein_bottle"])
def test_base_surface_switch(surface: str) -> None:
    """Each base_surface value produces a valid mesh."""
    gen = _make_generator()
    obj = gen.generate(
        params=_default_params(base_surface=surface),
        seed=42,
        grid_resolution=_TEST_RESOLUTION,
    )
    obj.validate_or_raise()
    assert obj.mesh is not None
    assert len(obj.mesh.vertices) > 0
    assert obj.parameters["base_surface"] == surface


# ---------------------------------------------------------------------------
# Displacement scale of 0 gives unmodified base surface
# ---------------------------------------------------------------------------


def test_zero_displacement_matches_base() -> None:
    """Displacement scale of 0 produces the unmodified base surface."""
    gen = _make_generator()
    obj = gen.generate(
        params=_default_params(displacement_scale=0.0),
        seed=42,
        grid_resolution=_TEST_RESOLUTION,
    )
    obj.validate_or_raise()

    # Generate with large displacement to verify difference
    obj_displaced = gen.generate(
        params=_default_params(
            displacement_scale=5.0,
            feed_rate=0.035,
            kill_rate=0.065,
            iterations=2000,
        ),
        seed=42,
        grid_resolution=_TEST_RESOLUTION,
    )

    assert obj.mesh is not None
    assert obj_displaced.mesh is not None
    # Zero displacement should differ from a heavily displaced version
    assert not np.allclose(
        obj.mesh.vertices, obj_displaced.mesh.vertices, atol=1e-3,
    )

    # Also verify the zero-displaced mesh has valid normals
    assert obj.mesh.normals is not None


def test_zero_displacement_preserves_base_geometry() -> None:
    """With displacement_scale=0, vertices match the base surface exactly."""
    from mathviz.generators.procedural._rd_surface_mesh import generate_base_mesh

    base_mesh = generate_base_mesh("torus", _TEST_RESOLUTION)
    gen = _make_generator()
    obj = gen.generate(
        params=_default_params(displacement_scale=0.0),
        seed=42,
        grid_resolution=_TEST_RESOLUTION,
    )

    assert obj.mesh is not None
    np.testing.assert_allclose(
        obj.mesh.vertices, base_mesh.vertices, atol=1e-12,
    )


# ---------------------------------------------------------------------------
# Registry and metadata
# ---------------------------------------------------------------------------


def test_registers_in_list() -> None:
    """Generator registers and appears in mathviz list."""
    names = [m.name for m in list_generators()]
    assert "rd_surface" in names


def test_metadata() -> None:
    """Generator records correct metadata."""
    gen = _make_generator()
    obj = gen.generate(
        params=_default_params(), seed=99, grid_resolution=_TEST_RESOLUTION,
    )

    assert obj.generator_name == "rd_surface"
    assert obj.category == "procedural"
    assert obj.seed == 99
    assert "feed_rate" in obj.parameters
    assert "kill_rate" in obj.parameters
    assert "base_surface" in obj.parameters


def test_default_representation() -> None:
    """Default representation is SURFACE_SHELL."""
    gen = _make_generator()
    rep = gen.get_default_representation()
    assert rep.type == RepresentationType.SURFACE_SHELL


def test_bounding_box_set() -> None:
    """Output has a bounding box computed from displaced vertices."""
    gen = _make_generator()
    obj = gen.generate(
        params=_default_params(), seed=42, grid_resolution=_TEST_RESOLUTION,
    )

    assert obj.bounding_box is not None
    assert all(
        lo < hi
        for lo, hi in zip(obj.bounding_box.min_corner, obj.bounding_box.max_corner)
    )
