"""Tests for the marching cubes wrapper."""

import numpy as np
import pytest
import trimesh

from mathviz.shared.marching_cubes import NoSurfaceError, SpatialBounds, extract_mesh


def _sphere_field(resolution: int = 30) -> np.ndarray:
    """Create a scalar field for a unit sphere: x² + y² + z² - 1."""
    lin = np.linspace(-2.0, 2.0, resolution)
    x, y, z = np.meshgrid(lin, lin, lin, indexing="ij")
    return x**2 + y**2 + z**2 - 1.0


UNIT_BOUNDS = SpatialBounds(min_corner=(-2.0, -2.0, -2.0), max_corner=(2.0, 2.0, 2.0))


class TestSphereImplicitField:
    """Sphere implicit field x²+y²+z²-1 produces a roughly spherical mesh."""

    def test_sphere_mesh_is_roughly_spherical(self) -> None:
        """Vertices should lie approximately on a unit sphere."""
        field = _sphere_field(resolution=40)
        mesh = extract_mesh(field, UNIT_BOUNDS)

        distances = np.linalg.norm(mesh.vertices, axis=1)
        assert np.allclose(distances, 1.0, atol=0.15), (
            f"Vertex distances from origin: mean={distances.mean():.3f}, "
            f"std={distances.std():.3f}"
        )

    def test_sphere_mesh_has_normals(self) -> None:
        """Output mesh should include vertex normals."""
        field = _sphere_field(resolution=30)
        mesh = extract_mesh(field, UNIT_BOUNDS)

        assert mesh.normals is not None
        assert mesh.normals.shape == mesh.vertices.shape

    def test_sphere_mesh_validates(self) -> None:
        """Output mesh passes its own validate()."""
        field = _sphere_field(resolution=30)
        mesh = extract_mesh(field, UNIT_BOUNDS)
        assert mesh.validate() == []


class TestCoordinateSpace:
    """Output vertices are in the coordinate space of the provided bounds."""

    def test_vertices_within_bounds(self) -> None:
        """All vertices should be within the spatial bounds."""
        bounds = SpatialBounds(min_corner=(-3.0, -3.0, -3.0), max_corner=(3.0, 3.0, 3.0))
        field = _sphere_field(resolution=30)
        mesh = extract_mesh(field, bounds)

        min_c = np.array(bounds.min_corner)
        max_c = np.array(bounds.max_corner)
        assert np.all(mesh.vertices >= min_c - 0.01)
        assert np.all(mesh.vertices <= max_c + 0.01)

    def test_asymmetric_bounds_shift_vertices(self) -> None:
        """Asymmetric bounds should shift vertex positions accordingly."""
        # Sphere centered at origin in a shifted bounding box
        lin = np.linspace(-1.0, 1.0, 30)
        x, y, z = np.meshgrid(lin, lin, lin, indexing="ij")
        field = x**2 + y**2 + z**2 - 0.5**2

        bounds_centered = SpatialBounds(
            min_corner=(-1.0, -1.0, -1.0), max_corner=(1.0, 1.0, 1.0)
        )
        bounds_shifted = SpatialBounds(
            min_corner=(9.0, 9.0, 9.0), max_corner=(11.0, 11.0, 11.0)
        )

        mesh_centered = extract_mesh(field, bounds_centered)
        mesh_shifted = extract_mesh(field, bounds_shifted)

        center_centered = mesh_centered.vertices.mean(axis=0)
        center_shifted = mesh_shifted.vertices.mean(axis=0)

        # Centered mesh should be near origin, shifted near (10, 10, 10)
        assert np.allclose(center_centered, [0.0, 0.0, 0.0], atol=0.1)
        assert np.allclose(center_shifted, [10.0, 10.0, 10.0], atol=0.1)

    def test_vertices_not_in_voxel_space(self) -> None:
        """Vertices should not be in raw voxel index space."""
        field = _sphere_field(resolution=30)
        mesh = extract_mesh(field, UNIT_BOUNDS)

        # If in voxel space, vertices would range from 0 to ~29
        # In coordinate space they should range from ~-1 to ~1 (sphere surface)
        assert mesh.vertices.max() < 5.0
        assert mesh.vertices.min() > -5.0


class TestWatertight:
    """Mesh is watertight for a closed implicit surface."""

    def test_sphere_is_watertight(self) -> None:
        field = _sphere_field(resolution=40)
        mesh = extract_mesh(field, UNIT_BOUNDS)

        tm = trimesh.Trimesh(
            vertices=mesh.vertices, faces=mesh.faces, process=False
        )
        assert tm.is_watertight


class TestDecimation:
    """Decimation reduces face count to approximately the target."""

    def test_decimation_reduces_faces(self) -> None:
        field = _sphere_field(resolution=40)
        mesh_full = extract_mesh(field, UNIT_BOUNDS)
        target = len(mesh_full.faces) // 4

        mesh_decimated = extract_mesh(field, UNIT_BOUNDS, target_face_count=target)

        # Should be within 50% of target (decimation is approximate)
        assert len(mesh_decimated.faces) < len(mesh_full.faces)
        assert len(mesh_decimated.faces) <= target * 1.5

    def test_decimation_preserves_shape(self) -> None:
        """Decimated sphere should still be roughly spherical."""
        field = _sphere_field(resolution=40)
        mesh = extract_mesh(field, UNIT_BOUNDS, target_face_count=200)

        distances = np.linalg.norm(mesh.vertices, axis=1)
        assert np.allclose(distances, 1.0, atol=0.3)


class TestResolutionScaling:
    """Higher voxel_resolution produces more faces."""

    def test_higher_resolution_more_faces(self) -> None:
        bounds = UNIT_BOUNDS
        field_low = _sphere_field(resolution=20)
        field_high = _sphere_field(resolution=40)

        mesh_low = extract_mesh(field_low, bounds)
        mesh_high = extract_mesh(field_high, bounds)

        assert len(mesh_high.faces) > len(mesh_low.faces)


class TestValidation:
    """Input validation edge cases."""

    def test_non_3d_field_raises(self) -> None:
        field_2d = np.zeros((10, 10))
        with pytest.raises(ValueError, match="field must be 3D"):
            extract_mesh(field_2d, UNIT_BOUNDS)

    def test_too_small_field_raises(self) -> None:
        field_tiny = np.zeros((1, 10, 10))
        with pytest.raises(ValueError, match="must be >= 2"):
            extract_mesh(field_tiny, UNIT_BOUNDS)

    def test_invalid_bounds_raises(self) -> None:
        field = _sphere_field(resolution=10)
        bad_bounds = SpatialBounds(min_corner=(1.0, 1.0, 1.0), max_corner=(0.0, 0.0, 0.0))
        with pytest.raises(ValueError, match="positive"):
            extract_mesh(field, bad_bounds)

    def test_negative_smoothing_raises(self) -> None:
        field = _sphere_field(resolution=10)
        with pytest.raises(ValueError, match="smoothing_iterations"):
            extract_mesh(field, UNIT_BOUNDS, smoothing_iterations=-1)

    def test_too_small_target_faces_raises(self) -> None:
        field = _sphere_field(resolution=10)
        with pytest.raises(ValueError, match="target_face_count"):
            extract_mesh(field, UNIT_BOUNDS, target_face_count=2)

    def test_no_isosurface_raises_no_surface_error(self) -> None:
        """Isolevel outside the field range raises NoSurfaceError, not generic ValueError."""
        field = _sphere_field(resolution=10)
        with pytest.raises(NoSurfaceError, match="No isosurface found"):
            extract_mesh(field, UNIT_BOUNDS, isolevel=9999.0)
