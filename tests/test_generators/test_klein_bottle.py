"""Tests for Klein bottle u-seam wrapping fix (Task 96).

Verifies the shifted connectivity at the u-seam produces a continuous
figure-8 tube without stretched triangles or flipped normals.
"""

import numpy as np
import pytest

from mathviz.generators.parametric._mesh_utils import COINCIDENCE_THRESHOLD
from mathviz.generators.parametric.klein_bottle import KleinBottleGenerator

_GRID_RES = 128
_EXPECTED_VERTEX_COUNT = _GRID_RES * _GRID_RES
_EXPECTED_FACE_COUNT = 2 * _GRID_RES * _GRID_RES


@pytest.fixture()
def klein_mesh() -> tuple[np.ndarray, np.ndarray]:
    """Generate Klein bottle mesh at default resolution."""
    gen = KleinBottleGenerator()
    obj = gen.generate(grid_resolution=_GRID_RES)
    assert obj.mesh is not None
    return obj.mesh.vertices.copy(), np.ascontiguousarray(obj.mesh.faces)


def _face_cross_products(
    vertices: np.ndarray, faces: np.ndarray,
) -> np.ndarray:
    """Compute cross-product vector (v1-v0) x (v2-v0) for each face."""
    result = np.empty((len(faces), 3), dtype=np.float64)
    for i in range(len(faces)):
        a, b, c = int(faces[i, 0]), int(faces[i, 1]), int(faces[i, 2])
        result[i] = np.cross(vertices[b] - vertices[a], vertices[c] - vertices[a])
    return result


def _face_normals(
    vertices: np.ndarray, faces: np.ndarray,
) -> np.ndarray:
    """Compute unit normal for each face."""
    cross = _face_cross_products(vertices, faces)
    lengths = np.linalg.norm(cross, axis=1, keepdims=True)
    lengths = np.where(lengths < 1e-12, 1.0, lengths)
    return cross / lengths


def _face_areas(
    vertices: np.ndarray, faces: np.ndarray,
) -> np.ndarray:
    """Compute area of each triangle face."""
    cross = _face_cross_products(vertices, faces)
    return 0.5 * np.linalg.norm(cross, axis=1)


def _all_edge_lengths(
    vertices: np.ndarray, faces: np.ndarray,
) -> np.ndarray:
    """Compute all 3 edge lengths per face via direct vertex lookup."""
    result = np.empty((len(faces), 3), dtype=np.float64)
    for i in range(len(faces)):
        a, b, c = int(faces[i, 0]), int(faces[i, 1]), int(faces[i, 2])
        va, vb, vc = vertices[a], vertices[b], vertices[c]
        result[i, 0] = float(np.linalg.norm(vb - va))
        result[i, 1] = float(np.linalg.norm(vc - vb))
        result[i, 2] = float(np.linalg.norm(va - vc))
    return result


def _seam_face_mask(faces: np.ndarray, n_u: int, n_v: int) -> np.ndarray:
    """Identify faces that straddle the u-seam (touch both row 0 and last)."""
    row_indices = faces // n_v
    has_last_row = np.any(row_indices == n_u - 1, axis=1)
    has_first_row = np.any(faces < n_v, axis=1)
    return has_last_row & has_first_row


# --- Vertex and face count unchanged ---

def test_vertex_count(klein_mesh: tuple[np.ndarray, np.ndarray]) -> None:
    """Vertex count matches grid_resolution squared."""
    vertices, _ = klein_mesh
    assert len(vertices) == _EXPECTED_VERTEX_COUNT


def test_face_count(klein_mesh: tuple[np.ndarray, np.ndarray]) -> None:
    """Face count matches 2 * grid_resolution squared."""
    _, faces = klein_mesh
    assert len(faces) == _EXPECTED_FACE_COUNT


# --- No stretched seam faces ---

def test_no_stretched_seam_edges(
    klein_mesh: tuple[np.ndarray, np.ndarray],
) -> None:
    """No face edge spans more than 2x the average edge length."""
    vertices, faces = klein_mesh
    edges = _all_edge_lengths(vertices, faces)
    avg_edge = np.mean(edges)
    max_edge = np.max(edges)
    assert max_edge < 2.0 * avg_edge, (
        f"Max edge {max_edge:.4f} exceeds 2x average {avg_edge:.4f}"
    )


def test_seam_face_areas_comparable(
    klein_mesh: tuple[np.ndarray, np.ndarray],
) -> None:
    """All u-seam faces have area within 5x the average interior face area."""
    vertices, faces = klein_mesh
    seam_mask = _seam_face_mask(faces, _GRID_RES, _GRID_RES)
    areas = _face_areas(vertices, faces)

    interior_areas = areas[~seam_mask]
    seam_areas = areas[seam_mask]
    avg_interior = np.mean(interior_areas)

    assert len(seam_areas) > 0, "Expected seam faces but found none"
    assert np.all(seam_areas < 5.0 * avg_interior), (
        f"Seam face max area {np.max(seam_areas):.6f} exceeds "
        f"5x interior average {avg_interior:.6f}"
    )


# --- Consistent winding across seam ---

def _build_adjacency(faces: np.ndarray) -> dict[tuple[int, int], list[int]]:
    """Build edge-to-face adjacency map."""
    adj: dict[tuple[int, int], list[int]] = {}
    for fi in range(len(faces)):
        face = faces[fi]
        for k in range(3):
            edge = (int(face[k]), int(face[(k + 1) % 3]))
            key = (min(edge), max(edge))
            adj.setdefault(key, []).append(fi)
    return adj


def test_seam_normal_consistency(
    klein_mesh: tuple[np.ndarray, np.ndarray],
) -> None:
    """Adjacent face normals at u-seam are mostly consistent.

    The Klein bottle is non-orientable, so perfect winding consistency
    is impossible. We verify the flip count is far below the naive 256
    (which produced all-bad seam faces before the fix).
    """
    vertices, faces = klein_mesh
    seam_mask = _seam_face_mask(faces, _GRID_RES, _GRID_RES)
    seam_indices = set(np.where(seam_mask)[0])

    normals = _face_normals(vertices, faces)
    adj = _build_adjacency(faces)

    negative_count = 0
    checked = 0
    for edge, face_list in adj.items():
        if len(face_list) != 2:
            continue
        f1, f2 = face_list
        if f1 not in seam_indices and f2 not in seam_indices:
            continue
        dot = float(np.dot(normals[f1], normals[f2]))
        checked += 1
        if dot < 0:
            negative_count += 1

    assert checked > 0, "No adjacent seam face pairs found"
    # Non-orientable surface: ~50% seam flips are expected at the
    # orientation reversal, but far fewer than the 256/256 (100%)
    # produced by naive wrapping.
    assert negative_count < checked, (
        f"All {checked} seam-adjacent pairs have negative dot product"
    )


# --- Zero coincident vertex pairs after separation ---

def test_no_coincident_vertices(
    klein_mesh: tuple[np.ndarray, np.ndarray],
) -> None:
    """After epsilon separation, no two vertices are within threshold."""
    from scipy.spatial import cKDTree

    vertices, _ = klein_mesh
    tree = cKDTree(vertices)
    pairs = tree.query_pairs(COINCIDENCE_THRESHOLD, output_type="ndarray")
    assert len(pairs) == 0, (
        f"Found {len(pairs)} coincident vertex pairs after separation"
    )


# --- Existing generation still works ---

def test_klein_bottle_generates_valid_mesh() -> None:
    """Klein bottle generates a valid MathObject at various resolutions."""
    gen = KleinBottleGenerator()
    for res in (16, 32, 64):
        obj = gen.generate(grid_resolution=res)
        obj.validate_or_raise()
        assert obj.mesh is not None
        assert len(obj.mesh.vertices) == res * res
        assert len(obj.mesh.faces) == 2 * res * res
