"""Tests for the boundary-edge count that the validator's manifold check reports."""

import numpy as np

from mathviz.core.math_object import Mesh
from mathviz.core.validator import CheckResult, validate_mesh

# --- Helpers ---


def _mesh_from_faces(faces: list[list[int]] | np.ndarray, vertex_count: int) -> Mesh:
    """Build a mesh with seeded random vertex positions and the given faces."""
    rng = np.random.default_rng(0)
    return Mesh(
        vertices=rng.random((vertex_count, 3)),
        faces=np.asarray(faces, dtype=np.int64),
    )


def _manifold_check(mesh: Mesh) -> CheckResult:
    """Return the single manifold check from validate_mesh."""
    checks = [c for c in validate_mesh(mesh).checks if c.name == "manifold"]
    assert len(checks) == 1
    return checks[0]


def _reference_boundary_count(faces: np.ndarray) -> int:
    """Count sorted edge rows that np.unique(axis=0) reports exactly once."""
    edges = np.sort(faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return int(np.sum(counts == 1))


# --- Boundary-edge count tests ---


class TestBoundaryEdgeCount:
    """The manifold check counts edges that belong to exactly one face."""

    def test_closed_tetrahedron_reports_no_boundary_edges(self) -> None:
        """A closed tetrahedron passes the manifold check with 'No boundary edges detected'."""
        faces = [[0, 1, 2], [0, 3, 1], [1, 3, 2], [2, 3, 0]]
        check = _manifold_check(_mesh_from_faces(faces, 4))
        assert check.passed is True
        assert check.message == "No boundary edges detected"

    def test_single_triangle_reports_three_boundary_edges(self) -> None:
        """One triangle fails the manifold check with 3 boundary edges."""
        check = _manifold_check(_mesh_from_faces([[0, 1, 2]], 3))
        assert check.passed is False
        assert check.message == "Mesh has 3 boundary edges (open surface)"

    def test_quad_of_two_triangles_reports_four_boundary_edges(self) -> None:
        """Two triangles that share one diagonal report 4 boundary edges."""
        check = _manifold_check(_mesh_from_faces([[0, 1, 2], [0, 2, 3]], 4))
        assert check.message == "Mesh has 4 boundary edges (open surface)"

    def test_edge_shared_by_three_faces_is_not_counted(self) -> None:
        """Three faces on edge (0, 1) report only their 6 other edges as boundary edges."""
        faces = [[0, 1, 2], [1, 0, 3], [0, 1, 4]]
        check = _manifold_check(_mesh_from_faces(faces, 5))
        assert check.message == "Mesh has 6 boundary edges (open surface)"

    def test_face_repeated_with_opposite_winding_reports_no_boundary_edges(self) -> None:
        """Faces (0, 1, 2) and (0, 2, 1) share every edge, so the count is 0."""
        check = _manifold_check(_mesh_from_faces([[0, 1, 2], [0, 2, 1]], 3))
        assert check.passed is True
        assert check.message == "No boundary edges detected"

    def test_seeded_random_faces_match_row_unique_reference(self) -> None:
        """For 2,000 seeded random faces on 300 vertices, the count equals the row-unique count."""
        rng = np.random.default_rng(7)
        vertex_count = 300
        faces = np.array(
            [rng.choice(vertex_count, size=3, replace=False) for _ in range(2000)],
            dtype=np.int64,
        )
        expected = _reference_boundary_count(faces)
        assert expected > 0
        check = _manifold_check(_mesh_from_faces(faces, vertex_count))
        assert check.message == f"Mesh has {expected} boundary edges (open surface)"
