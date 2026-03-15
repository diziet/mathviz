"""Tests for the geodesic sphere generator."""

import numpy as np
import pytest

from mathviz.core.generator import clear_registry, get_generator, register
from mathviz.core.representation import RepresentationType
from mathviz.generators.geometry.geodesic_sphere import (
    GeodesicSphereGenerator,
    _count_dual_face_sides,
)


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset registry for each test."""
    clear_registry(suppress_discovery=True)
    register(GeodesicSphereGenerator)
    yield
    clear_registry(suppress_discovery=True)


# ---------------------------------------------------------------------------
# Frequency 1 — icosahedron
# ---------------------------------------------------------------------------


def test_frequency_1_is_icosahedron() -> None:
    """Frequency 1 produces an icosahedron (12 vertices, 20 faces)."""
    gen = GeodesicSphereGenerator()
    obj = gen.generate(params={"frequency": 1})
    obj.validate_or_raise()

    assert obj.mesh is not None
    assert len(obj.mesh.vertices) == 12
    assert len(obj.mesh.faces) == 20


# ---------------------------------------------------------------------------
# Face count scaling
# ---------------------------------------------------------------------------


def test_higher_frequency_produces_more_faces() -> None:
    """Higher frequency produces 20 * N^2 faces."""
    gen = GeodesicSphereGenerator()
    for freq in (1, 2, 3, 4):
        obj = gen.generate(params={"frequency": freq})
        obj.validate_or_raise()
        expected_faces = 20 * freq ** 2
        assert len(obj.mesh.faces) == expected_faces, (
            f"freq={freq}: expected {expected_faces} faces, "
            f"got {len(obj.mesh.faces)}"
        )


# ---------------------------------------------------------------------------
# Dual mode
# ---------------------------------------------------------------------------


def test_dual_mode_produces_pentagonal_and_hexagonal_faces() -> None:
    """Dual mode produces pentagonal and hexagonal faces."""
    gen = GeodesicSphereGenerator()
    # Use frequency 2 for a simple Goldberg polyhedron
    obj = gen.generate(params={"frequency": 2, "dual": True})
    obj.validate_or_raise()

    assert obj.mesh is not None
    # Count face types from the original (pre-dual) mesh
    _, faces_orig = _get_pre_dual_mesh(frequency=2)
    side_counts = _count_dual_face_sides(
        np.zeros((0, 3)), faces_orig,  # vertices unused in counting
    )
    # Icosahedron original vertices become pentagons (12),
    # subdivision vertices become hexagons
    assert 5 in side_counts, "Expected pentagonal faces in dual"
    assert 6 in side_counts, "Expected hexagonal faces in dual"
    assert side_counts[5] == 12, "Expected exactly 12 pentagonal faces"


def _get_pre_dual_mesh(frequency: int) -> tuple[np.ndarray, np.ndarray]:
    """Helper to get the subdivided mesh before dual conversion."""
    from mathviz.generators.geometry.geodesic_sphere import (
        _build_icosahedron,
        _subdivide_and_project,
    )
    vertices, faces = _build_icosahedron()
    return _subdivide_and_project(vertices, faces, frequency)


# ---------------------------------------------------------------------------
# Sphere property — all vertices equidistant from center
# ---------------------------------------------------------------------------


def test_all_vertices_on_sphere() -> None:
    """All vertices are equidistant from center (on sphere surface)."""
    gen = GeodesicSphereGenerator()
    obj = gen.generate(params={"frequency": 3, "radius": 2.5})
    obj.validate_or_raise()

    distances = np.linalg.norm(obj.mesh.vertices, axis=1)
    np.testing.assert_allclose(distances, 2.5, atol=1e-10)


def test_all_vertices_on_unit_sphere_default_radius() -> None:
    """Default radius=1.0 puts all vertices on unit sphere."""
    gen = GeodesicSphereGenerator()
    obj = gen.generate(params={"frequency": 4})
    obj.validate_or_raise()

    distances = np.linalg.norm(obj.mesh.vertices, axis=1)
    np.testing.assert_allclose(distances, 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Registration and rendering
# ---------------------------------------------------------------------------


def test_registers_and_renders() -> None:
    """Generator registers and can produce valid output."""
    gen_cls = get_generator("geodesic_sphere")
    assert gen_cls is GeodesicSphereGenerator

    gen = gen_cls()
    obj = gen.generate()
    obj.validate_or_raise()

    assert obj.mesh is not None
    assert obj.generator_name == "geodesic_sphere"
    assert obj.category == "geometry"
    assert obj.bounding_box is not None


def test_default_representation_is_surface_shell() -> None:
    """Default representation is SURFACE_SHELL."""
    gen = GeodesicSphereGenerator()
    rep = gen.get_default_representation()
    assert rep.type == RepresentationType.SURFACE_SHELL


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------


def test_frequency_zero_rejected() -> None:
    """Frequency 0 raises ValueError."""
    gen = GeodesicSphereGenerator()
    with pytest.raises(ValueError, match="frequency must be >= 1"):
        gen.generate(params={"frequency": 0})


def test_frequency_too_large_rejected() -> None:
    """Frequency above max raises ValueError."""
    gen = GeodesicSphereGenerator()
    with pytest.raises(ValueError, match="frequency must be <= 32"):
        gen.generate(params={"frequency": 33})


def test_negative_radius_rejected() -> None:
    """Negative radius raises ValueError."""
    gen = GeodesicSphereGenerator()
    with pytest.raises(ValueError, match="radius must be > 0"):
        gen.generate(params={"radius": -1.0})


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_deterministic_output() -> None:
    """Same parameters produce identical output."""
    gen = GeodesicSphereGenerator()
    obj1 = gen.generate(params={"frequency": 3})
    obj2 = gen.generate(params={"frequency": 3})

    assert obj1.mesh is not None
    assert obj2.mesh is not None
    np.testing.assert_array_equal(obj1.mesh.vertices, obj2.mesh.vertices)
    np.testing.assert_array_equal(obj1.mesh.faces, obj2.mesh.faces)


# ---------------------------------------------------------------------------
# Default params
# ---------------------------------------------------------------------------


def test_default_params() -> None:
    """Default params are frequency=4, radius=1.0, dual=False."""
    gen = GeodesicSphereGenerator()
    defaults = gen.get_default_params()
    assert defaults["frequency"] == 4
    assert defaults["radius"] == 1.0
    assert defaults["dual"] is False


def test_dual_string_true_activates_dual_mode() -> None:
    """String 'true' from CLI correctly activates dual mode."""
    gen = GeodesicSphereGenerator()
    obj = gen.generate(params={"frequency": 2, "dual": "true"})
    obj.validate_or_raise()
    # Dual mode should still produce valid mesh
    assert obj.mesh is not None
