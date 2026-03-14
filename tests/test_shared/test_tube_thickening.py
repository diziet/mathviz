"""Tests for tube thickening (Bishop frame extrusion)."""

import warnings

import numpy as np
import pytest
import trimesh

from mathviz.core.math_object import Curve
from mathviz.shared.tube_thickening import thicken_curve


def _circle_curve(n: int = 64, r: float = 1.0) -> Curve:
    """Create a closed circular curve in the XY plane."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    points = np.column_stack([r * np.cos(t), r * np.sin(t), np.zeros(n)])
    return Curve(points=points.astype(np.float64), closed=True)


def _helix_curve(
    n: int = 100, radius: float = 1.0, pitch: float = 0.5, turns: int = 3
) -> Curve:
    """Create an open helix curve."""
    t = np.linspace(0, turns * 2 * np.pi, n)
    points = np.column_stack([
        radius * np.cos(t),
        radius * np.sin(t),
        pitch * t,
    ])
    return Curve(points=points.astype(np.float64), closed=False)


def _line_curve(n: int = 10) -> Curve:
    """Create a straight open line along the X axis."""
    points = np.column_stack([
        np.linspace(0, 1, n),
        np.zeros(n),
        np.zeros(n),
    ])
    return Curve(points=points.astype(np.float64), closed=False)


class TestClosedCircleTorus:
    """Closed circular curve produces a torus-like mesh with no seam gap."""

    def test_torus_mesh_has_no_seam_gap(self) -> None:
        """All vertices at the seam should be shared, not duplicated."""
        curve = _circle_curve(n=64)
        mesh = thicken_curve(curve, radius=0.1, sides=16)

        tm = trimesh.Trimesh(
            vertices=mesh.vertices, faces=mesh.faces, process=False
        )

        # A proper torus is watertight — no boundary edges
        assert tm.is_watertight, "Torus mesh has seam gap (not watertight)"

    def test_torus_vertex_positions_are_smooth(self) -> None:
        """Vertices near the seam should be equally spaced, not bunched."""
        curve = _circle_curve(n=64)
        sides = 16
        mesh = thicken_curve(curve, radius=0.1, sides=sides)

        # Compare first ring and last ring distances
        first_ring = mesh.vertices[:sides]
        last_ring = mesh.vertices[-sides:]
        ring_second = mesh.vertices[sides : 2 * sides]

        gap_last_first = np.linalg.norm(first_ring - last_ring, axis=1).mean()
        gap_first_second = np.linalg.norm(ring_second - first_ring, axis=1).mean()

        # Gap across seam should be comparable to gap between adjacent rings
        assert gap_last_first < gap_first_second * 2.0


class TestWatertightClosedCurves:
    """Output mesh is watertight for closed curves."""

    def test_closed_circle_is_watertight(self) -> None:
        curve = _circle_curve(n=32)
        mesh = thicken_curve(curve, radius=0.1, sides=8)
        tm = trimesh.Trimesh(
            vertices=mesh.vertices, faces=mesh.faces, process=False
        )
        assert tm.is_watertight

    def test_closed_trefoil_is_watertight(self) -> None:
        """A trefoil knot (closed, non-planar) should also be watertight."""
        t = np.linspace(0, 2 * np.pi, 100, endpoint=False)
        points = np.column_stack([
            np.sin(t) + 2 * np.sin(2 * t),
            np.cos(t) - 2 * np.cos(2 * t),
            -np.sin(3 * t),
        ])
        curve = Curve(points=points.astype(np.float64), closed=True)
        mesh = thicken_curve(curve, radius=0.05, sides=8)
        tm = trimesh.Trimesh(
            vertices=mesh.vertices, faces=mesh.faces, process=False
        )
        assert tm.is_watertight


class TestVertexCount:
    """Vertex count equals curve_points x sides (plus caps for open)."""

    def test_closed_curve_vertex_count(self) -> None:
        n_points = 32
        sides = 12
        curve = _circle_curve(n=n_points)
        mesh = thicken_curve(curve, radius=0.1, sides=sides)
        assert len(mesh.vertices) == n_points * sides

    def test_open_curve_vertex_count_with_caps(self) -> None:
        n_points = 10
        sides = 8
        curve = _line_curve(n=n_points)
        mesh = thicken_curve(curve, radius=0.05, sides=sides)
        # Tube ring verts + 2 center verts for caps
        expected = n_points * sides + 2
        assert len(mesh.vertices) == expected


class TestHelixNoTwist:
    """A helix (high curvature) produces no twisting artifacts."""

    def test_helix_ring_normals_are_consistent(self) -> None:
        """Adjacent rings should have smoothly varying orientations."""
        curve = _helix_curve(n=200, radius=1.0, pitch=0.3, turns=5)
        sides = 16
        mesh = thicken_curve(curve, radius=0.05, sides=sides)

        # For each pair of adjacent rings, the ring vertices should be
        # rotated by roughly the same small angle, not a large jump
        max_angular_jump = 0.0
        for i in range(len(curve.points) - 1):
            ring_a = mesh.vertices[i * sides : (i + 1) * sides]
            ring_b = mesh.vertices[(i + 1) * sides : (i + 2) * sides]

            center_a = ring_a.mean(axis=0)
            center_b = ring_b.mean(axis=0)

            # Vectors from center to first vertex of each ring
            va = ring_a[0] - center_a
            vb = ring_b[0] - center_b

            va /= np.linalg.norm(va)
            vb /= np.linalg.norm(vb)

            cos_angle = np.clip(np.dot(va, vb), -1.0, 1.0)
            angle = np.arccos(cos_angle)
            max_angular_jump = max(max_angular_jump, angle)

        # With Bishop frames, angular change between adjacent rings
        # should be small (< 30 degrees for a smooth helix)
        assert max_angular_jump < np.radians(30), (
            f"Max angular jump {np.degrees(max_angular_jump):.1f}° "
            "indicates twisting artifact"
        )


class TestSelfIntersectionWarning:
    """Self-intersection warning fires for large radius relative to curvature."""

    def test_large_radius_warns(self) -> None:
        curve = _circle_curve(n=20, r=1.0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            thicken_curve(curve, radius=2.0, sides=8)
            user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
            assert len(user_warnings) >= 1
            assert "self-intersection" in str(user_warnings[0].message).lower()

    def test_small_radius_no_warning(self) -> None:
        curve = _circle_curve(n=64, r=1.0)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            thicken_curve(curve, radius=0.01, sides=8)
            user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
            assert len(user_warnings) == 0


class TestValidation:
    """Input validation edge cases."""

    def test_negative_radius_raises(self) -> None:
        curve = _line_curve()
        with pytest.raises(ValueError, match="radius must be positive"):
            thicken_curve(curve, radius=-1.0)

    def test_too_few_sides_raises(self) -> None:
        curve = _line_curve()
        with pytest.raises(ValueError, match="sides must be >= 3"):
            thicken_curve(curve, radius=0.1, sides=2)

    def test_single_point_raises(self) -> None:
        curve = Curve(
            points=np.array([[0.0, 0.0, 0.0]]),
            closed=False,
        )
        with pytest.raises(ValueError, match="Curve needs >= 2 points"):
            thicken_curve(curve, radius=0.1)

    def test_output_mesh_validates(self) -> None:
        """Output mesh passes its own validate()."""
        curve = _circle_curve(n=32)
        mesh = thicken_curve(curve, radius=0.1, sides=8)
        errors = mesh.validate()
        assert errors == []
