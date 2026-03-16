"""Tests for the Möbius strip generator seam fix.

Covers seam edge quality, face areas, vertex normals, and overall mesh quality.
"""

import numpy as np
import pytest

from mathviz.core.generator import clear_registry, get_generator, register
from mathviz.core.representation import RepresentationType
from mathviz.generators.parametric.mobius_strip import MobiusStripGenerator


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset registry and register the Möbius strip for each test."""
    clear_registry(suppress_discovery=True)
    register(MobiusStripGenerator)
    yield
    clear_registry(suppress_discovery=True)


@pytest.fixture()
def mesh_obj():
    """Generate a Möbius strip mesh at default resolution."""
    gen = MobiusStripGenerator()
    return gen.generate(grid_resolution=128)


@pytest.fixture()
def edge_lengths(mesh_obj):
    """Compute all edge lengths from the mesh."""
    verts = mesh_obj.mesh.vertices
    faces = mesh_obj.mesh.faces
    edges = set()
    for f in faces:
        edges.add((min(f[0], f[1]), max(f[0], f[1])))
        edges.add((min(f[1], f[2]), max(f[1], f[2])))
        edges.add((min(f[0], f[2]), max(f[0], f[2])))
    edge_list = np.array(list(edges))
    v0 = verts[edge_list[:, 0]]
    v1 = verts[edge_list[:, 1]]
    return np.linalg.norm(v1 - v0, axis=1)


def _compute_face_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute unnormalized face normals via cross product."""
    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]
    return np.cross(v1 - v0, v2 - v0)


def _compute_face_areas(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute triangle areas for all faces."""
    return 0.5 * np.linalg.norm(_compute_face_normals(verts, faces), axis=1)


def _compute_vertex_normals(
    verts: np.ndarray, faces: np.ndarray,
) -> np.ndarray:
    """Compute per-vertex normals by averaging adjacent face normals."""
    face_normals = _compute_face_normals(verts, faces)

    vertex_normals = np.zeros_like(verts)
    for col in range(3):
        np.add.at(vertex_normals, faces[:, col], face_normals)

    return vertex_normals


# ---------------------------------------------------------------------------
# Seam quality tests
# ---------------------------------------------------------------------------


def test_max_edge_under_3x_median(edge_lengths) -> None:
    """No edge exceeds 3x median length (eliminates >< artifact)."""
    median_len = np.median(edge_lengths)
    max_len = edge_lengths.max()
    assert max_len < 3.0 * median_len, (
        f"Max edge {max_len:.4f} >= 3x median {median_len:.4f}"
    )


def test_seam_face_areas_within_5x_interior(mesh_obj) -> None:
    """Seam face areas within 5x of interior face areas."""
    verts = mesh_obj.mesh.vertices
    faces = mesh_obj.mesh.faces
    areas = _compute_face_areas(verts, faces)

    median_area = np.median(areas)
    assert np.all(areas <= 5.0 * median_area), (
        f"Max area {areas.max():.6f} > 5x median {median_area:.6f}"
    )


def test_vertex_normals_at_u0_have_magnitude(mesh_obj) -> None:
    """Vertex normals at u=0 have magnitude > 50% of interior mean."""
    n = mesh_obj.parameters["grid_resolution"]
    verts = mesh_obj.mesh.vertices
    faces = mesh_obj.mesh.faces

    normals = _compute_vertex_normals(verts, faces)
    magnitudes = np.linalg.norm(normals, axis=1)

    # Row 0 vertices are indices 0..n-1 (exclude boundary corners
    # at v=0 and v=n-1 which naturally have fewer contributing faces)
    row0_mags = magnitudes[1:n - 1]
    # Interior vertices exclude row 0, last row, and the duplicate row
    interior_mags = magnitudes[n:n * (n - 1)]
    interior_mean = interior_mags.mean()

    assert np.median(row0_mags) > 0.5 * interior_mean, (
        f"Median row0 normal mag {np.median(row0_mags):.4f} "
        f"<= 50% of interior mean {interior_mean:.4f}"
    )


# ---------------------------------------------------------------------------
# Basic mesh validity
# ---------------------------------------------------------------------------


def test_produces_valid_mesh(mesh_obj) -> None:
    """Default parameters produce a valid mesh."""
    mesh_obj.validate_or_raise()
    assert mesh_obj.mesh is not None
    assert len(mesh_obj.mesh.vertices) > 0
    assert len(mesh_obj.mesh.faces) > 0


def test_no_nan_or_inf(mesh_obj) -> None:
    """No NaN or infinite values in vertices."""
    assert np.all(np.isfinite(mesh_obj.mesh.vertices))


def test_face_indices_valid(mesh_obj) -> None:
    """All face indices reference valid vertices."""
    assert np.all(mesh_obj.mesh.faces >= 0)
    assert np.all(mesh_obj.mesh.faces < len(mesh_obj.mesh.vertices))


def test_determinism() -> None:
    """Same seed produces identical geometry."""
    gen = MobiusStripGenerator()
    obj1 = gen.generate(seed=42, grid_resolution=32)
    obj2 = gen.generate(seed=42, grid_resolution=32)
    np.testing.assert_array_equal(obj1.mesh.vertices, obj2.mesh.vertices)
    np.testing.assert_array_equal(obj1.mesh.faces, obj2.mesh.faces)


def test_registered_and_discoverable() -> None:
    """Generator is discoverable via the registry."""
    found = get_generator("mobius_strip")
    assert found is MobiusStripGenerator


def test_default_representation() -> None:
    """Default representation is SURFACE_SHELL."""
    gen = MobiusStripGenerator()
    rep = gen.get_default_representation()
    assert rep.type == RepresentationType.SURFACE_SHELL
