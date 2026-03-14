"""Tests for the representation strategy layer."""

import numpy as np
import pytest
import trimesh

from mathviz.core.math_object import Curve, MathObject, Mesh, PointCloud
from mathviz.core.representation import RepresentationConfig, RepresentationType
from mathviz.pipeline.representation_strategy import apply, get_default


def _make_cube_mesh() -> Mesh:
    """Create a simple cube mesh for testing."""
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=np.float64)
    faces = np.array([
        [0, 1, 2], [0, 2, 3],
        [4, 6, 5], [4, 7, 6],
        [0, 4, 5], [0, 5, 1],
        [2, 6, 7], [2, 7, 3],
        [0, 3, 7], [0, 7, 4],
        [1, 5, 6], [1, 6, 2],
    ], dtype=np.int64)
    return Mesh(vertices=vertices, faces=faces)


def _make_helix_curve() -> Curve:
    """Create a helix curve for testing."""
    t = np.linspace(0, 4 * np.pi, 100)
    points = np.column_stack([np.cos(t), np.sin(t), t / (4 * np.pi)])
    return Curve(points=points.astype(np.float64), closed=False)


def _make_circle_curve() -> Curve:
    """Create a closed circle curve for testing."""
    t = np.linspace(0, 2 * np.pi, 50, endpoint=False)
    points = np.column_stack([np.cos(t), np.sin(t), np.zeros_like(t)])
    return Curve(points=points.astype(np.float64), closed=True)


class TestSurfaceShell:
    """SURFACE_SHELL representation tests."""

    def test_mesh_passthrough(self) -> None:
        """SURFACE_SHELL on a mesh input returns MathObject with mesh intact."""
        mesh = _make_cube_mesh()
        obj = MathObject(mesh=mesh, generator_name="torus")
        config = RepresentationConfig(type=RepresentationType.SURFACE_SHELL)

        result = apply(obj, config)

        assert result.mesh is not None
        np.testing.assert_array_equal(result.mesh.vertices, mesh.vertices)
        np.testing.assert_array_equal(result.mesh.faces, mesh.faces)

    def test_sets_representation_field(self) -> None:
        """Applying a representation sets the representation field on MathObject."""
        mesh = _make_cube_mesh()
        obj = MathObject(mesh=mesh, generator_name="torus")
        config = RepresentationConfig(type=RepresentationType.SURFACE_SHELL)

        result = apply(obj, config)

        assert result.representation == "surface_shell"

    def test_no_mesh_raises(self) -> None:
        """SURFACE_SHELL on input without mesh raises ValueError."""
        curve = _make_helix_curve()
        obj = MathObject(curves=[curve], generator_name="test")
        config = RepresentationConfig(type=RepresentationType.SURFACE_SHELL)

        with pytest.raises(ValueError, match="requires a mesh"):
            apply(obj, config)


class TestRawPointCloud:
    """RAW_POINT_CLOUD representation tests."""

    def test_curve_to_point_cloud(self) -> None:
        """RAW_POINT_CLOUD on a curve input produces a PointCloud from curve points."""
        curve = _make_helix_curve()
        obj = MathObject(curves=[curve], generator_name="lorenz")
        config = RepresentationConfig(type=RepresentationType.RAW_POINT_CLOUD)

        result = apply(obj, config)

        assert result.point_cloud is not None
        assert isinstance(result.point_cloud, PointCloud)
        assert len(result.point_cloud.points) == len(curve.points)
        np.testing.assert_array_equal(result.point_cloud.points, curve.points)

    def test_multiple_curves_concatenated(self) -> None:
        """RAW_POINT_CLOUD concatenates points from multiple curves."""
        c1 = _make_helix_curve()
        c2 = _make_circle_curve()
        obj = MathObject(curves=[c1, c2], generator_name="test")
        config = RepresentationConfig(type=RepresentationType.RAW_POINT_CLOUD)

        result = apply(obj, config)

        expected_count = len(c1.points) + len(c2.points)
        assert len(result.point_cloud.points) == expected_count

    def test_no_curves_raises(self) -> None:
        """RAW_POINT_CLOUD on input without curves raises ValueError."""
        mesh = _make_cube_mesh()
        obj = MathObject(mesh=mesh, generator_name="test")
        config = RepresentationConfig(type=RepresentationType.RAW_POINT_CLOUD)

        with pytest.raises(ValueError, match="requires curve input"):
            apply(obj, config)

    def test_sets_representation_field(self) -> None:
        """Applying RAW_POINT_CLOUD sets representation field."""
        obj = MathObject(curves=[_make_helix_curve()], generator_name="test")
        config = RepresentationConfig(type=RepresentationType.RAW_POINT_CLOUD)
        result = apply(obj, config)
        assert result.representation == "raw_point_cloud"


class TestTube:
    """TUBE representation tests."""

    def test_curve_to_watertight_mesh(self) -> None:
        """TUBE on a curve input produces a watertight mesh via tube thickening."""
        curve = _make_circle_curve()
        obj = MathObject(curves=[curve], generator_name="trefoil_knot")
        config = RepresentationConfig(
            type=RepresentationType.TUBE,
            tube_radius=0.1,
            tube_sides=8,
        )

        result = apply(obj, config)

        assert result.mesh is not None
        assert result.mesh.vertices.shape[1] == 3
        assert result.mesh.faces.shape[1] == 3
        assert len(result.mesh.vertices) > 0
        assert len(result.mesh.faces) > 0

    def test_mesh_only_input_raises(self) -> None:
        """TUBE on a mesh-only input (no curves) raises a clear error."""
        mesh = _make_cube_mesh()
        obj = MathObject(mesh=mesh, generator_name="test")
        config = RepresentationConfig(
            type=RepresentationType.TUBE,
            tube_radius=0.1,
        )

        with pytest.raises(ValueError, match="requires curve input"):
            apply(obj, config)

    def test_sets_representation_field(self) -> None:
        """TUBE sets the representation field."""
        curve = _make_helix_curve()
        obj = MathObject(curves=[curve], generator_name="lorenz")
        config = RepresentationConfig(
            type=RepresentationType.TUBE,
            tube_radius=0.05,
        )

        result = apply(obj, config)

        assert result.representation == "tube"


class TestHeightmapRelief:
    """HEIGHTMAP_RELIEF representation tests."""

    def test_scalar_field_to_mesh(self) -> None:
        """HEIGHTMAP_RELIEF on a scalar field produces a mesh with correct z-range."""
        rows, cols = 10, 10
        # Simple gradient: z ranges from 0 to 1
        field = np.linspace(0, 1, rows * cols).reshape(rows, cols)

        obj = MathObject(
            scalar_field=field,
            generator_name="mandelbrot",
        )
        config = RepresentationConfig(type=RepresentationType.HEIGHTMAP_RELIEF)

        result = apply(obj, config)

        assert result.mesh is not None
        z_values = result.mesh.vertices[:, 2]
        assert z_values.min() == pytest.approx(0.0, abs=1e-10)
        assert z_values.max() == pytest.approx(1.0, abs=1e-10)

    def test_no_scalar_field_raises(self) -> None:
        """HEIGHTMAP_RELIEF without scalar_field raises ValueError."""
        mesh = _make_cube_mesh()
        obj = MathObject(mesh=mesh, generator_name="test")
        config = RepresentationConfig(type=RepresentationType.HEIGHTMAP_RELIEF)

        with pytest.raises(ValueError, match="requires a scalar_field"):
            apply(obj, config)

    def test_correct_face_count(self) -> None:
        """HEIGHTMAP_RELIEF produces correct number of faces for grid."""
        rows, cols = 5, 6
        field = np.ones((rows, cols))
        obj = MathObject(
            scalar_field=field,
            generator_name="test",
        )
        config = RepresentationConfig(type=RepresentationType.HEIGHTMAP_RELIEF)

        result = apply(obj, config)

        expected_faces = (rows - 1) * (cols - 1) * 2
        assert len(result.mesh.faces) == expected_faces


def _make_watertight_cube_mesh(size: float = 1.0) -> Mesh:
    """Create a watertight cube mesh using trimesh for correct winding."""
    box = trimesh.creation.box(extents=[size, size, size])
    return Mesh(
        vertices=np.asarray(box.vertices, dtype=np.float64),
        faces=np.asarray(box.faces, dtype=np.int64),
    )


def _make_open_mesh() -> Mesh:
    """Create a non-watertight mesh (single triangle)."""
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0],
    ], dtype=np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    return Mesh(vertices=vertices, faces=faces)


class TestVolumeFill:
    """VOLUME_FILL representation tests."""

    def test_fills_interior_of_watertight_cube(self) -> None:
        """VOLUME_FILL on a watertight cube fills interior (points inside, none outside)."""
        mesh = _make_watertight_cube_mesh(size=2.0)
        obj = MathObject(mesh=mesh, generator_name="test")
        config = RepresentationConfig(
            type=RepresentationType.VOLUME_FILL,
            volume_density=100.0,
        )

        result = apply(obj, config)

        assert result.point_cloud is not None
        points = result.point_cloud.points
        assert len(points) > 0

        # All points should be inside the cube [-1, 1]^3
        assert np.all(points >= -1.0 - 0.5)
        assert np.all(points <= 1.0 + 0.5)

        # Verify using trimesh that points are actually inside
        box = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
        inside = box.contains(points)
        assert np.all(inside), f"{np.sum(~inside)} points outside mesh"

    def test_non_watertight_mesh_raises(self) -> None:
        """VOLUME_FILL on a non-watertight mesh raises ValueError."""
        mesh = _make_open_mesh()
        obj = MathObject(mesh=mesh, generator_name="test")
        config = RepresentationConfig(type=RepresentationType.VOLUME_FILL)

        with pytest.raises(ValueError, match="watertight"):
            apply(obj, config)

    def test_sets_representation_field(self) -> None:
        """VOLUME_FILL sets the representation field."""
        mesh = _make_watertight_cube_mesh()
        obj = MathObject(mesh=mesh, generator_name="test")
        config = RepresentationConfig(type=RepresentationType.VOLUME_FILL)

        result = apply(obj, config)
        assert result.representation == "volume_fill"


class TestSliceStack:
    """SLICE_STACK representation tests."""

    def test_produces_discrete_layers(self) -> None:
        """SLICE_STACK with slice_count=5 produces roughly 5 discrete z-layers."""
        mesh = _make_watertight_cube_mesh(size=2.0)
        obj = MathObject(mesh=mesh, generator_name="test")
        config = RepresentationConfig(
            type=RepresentationType.SLICE_STACK,
            slice_count=5,
            slice_axis="z",
        )

        result = apply(obj, config)

        assert result.point_cloud is not None
        points = result.point_cloud.points
        assert len(points) > 0

        # Check that z-values cluster into roughly 5 distinct layers
        z_values = points[:, 2]
        unique_z = np.unique(np.round(z_values, decimals=4))
        assert len(unique_z) >= 4, f"Expected ~5 layers, got {len(unique_z)}"

    def test_no_mesh_raises(self) -> None:
        """SLICE_STACK without a mesh raises ValueError."""
        curve = _make_helix_curve()
        obj = MathObject(curves=[curve], generator_name="test")
        config = RepresentationConfig(type=RepresentationType.SLICE_STACK)

        with pytest.raises(ValueError, match="requires a mesh"):
            apply(obj, config)

    def test_sets_representation_field(self) -> None:
        """SLICE_STACK sets the representation field."""
        mesh = _make_watertight_cube_mesh()
        obj = MathObject(mesh=mesh, generator_name="test")
        config = RepresentationConfig(
            type=RepresentationType.SLICE_STACK,
            slice_count=3,
        )

        result = apply(obj, config)
        assert result.representation == "slice_stack"


class TestWireframe:
    """WIREFRAME representation tests."""

    def test_produces_tube_mesh_along_edges(self) -> None:
        """WIREFRAME produces thin tubes along mesh edges."""
        mesh = _make_cube_mesh()
        obj = MathObject(mesh=mesh, generator_name="test")
        config = RepresentationConfig(
            type=RepresentationType.WIREFRAME,
            wireframe_thickness=0.05,
        )

        result = apply(obj, config)

        assert result.mesh is not None
        assert len(result.mesh.vertices) > 0
        assert len(result.mesh.faces) > 0

    def test_vertex_count_scales_with_edges_and_sides(self) -> None:
        """WIREFRAME vertex count scales with edge count * tube_sides."""
        mesh = _make_cube_mesh()
        obj = MathObject(mesh=mesh, generator_name="test")
        tube_sides = 8
        config = RepresentationConfig(
            type=RepresentationType.WIREFRAME,
            wireframe_thickness=0.05,
            tube_sides=tube_sides,
        )

        result = apply(obj, config)

        # A cube has 12 faces -> 18 unique edges
        # Each edge (2-point open curve) gets tube_sides ring verts per point
        # plus 2 cap center verts = 2*sides + 2 verts per edge
        tm_cube = trimesh.Trimesh(
            vertices=mesh.vertices, faces=mesh.faces, process=False
        )
        edge_count = len(tm_cube.edges_unique)
        expected_verts_per_edge = 2 * tube_sides + 2
        expected_total = edge_count * expected_verts_per_edge
        assert result.mesh is not None
        assert len(result.mesh.vertices) == expected_total

    def test_no_mesh_raises(self) -> None:
        """WIREFRAME without a mesh raises ValueError."""
        curve = _make_helix_curve()
        obj = MathObject(curves=[curve], generator_name="test")
        config = RepresentationConfig(type=RepresentationType.WIREFRAME)

        with pytest.raises(ValueError, match="requires a mesh"):
            apply(obj, config)

    def test_sets_representation_field(self) -> None:
        """WIREFRAME sets the representation field."""
        mesh = _make_cube_mesh()
        obj = MathObject(mesh=mesh, generator_name="test")
        config = RepresentationConfig(type=RepresentationType.WIREFRAME)

        result = apply(obj, config)
        assert result.representation == "wireframe"


class TestGetDefault:
    """Test get_default() generator name mapping."""

    def test_known_generator(self) -> None:
        """Known generator returns mapped config."""
        config = get_default("torus")
        assert config.type == RepresentationType.SURFACE_SHELL

    def test_attractor_generator(self) -> None:
        """Attractor generator defaults to TUBE."""
        config = get_default("lorenz")
        assert config.type == RepresentationType.TUBE
        assert config.tube_radius is not None

    def test_unknown_generator_fallback(self) -> None:
        """Unknown generator returns fallback config."""
        config = get_default("some_unknown_generator")
        assert config.type == RepresentationType.SURFACE_SHELL


class TestCandidatesMode:
    """Test candidates mode stub."""

    def test_candidates_raises_not_implemented(self) -> None:
        """candidates=True raises NotImplementedError."""
        obj = MathObject(mesh=_make_cube_mesh(), generator_name="test")
        config = RepresentationConfig(type=RepresentationType.SURFACE_SHELL)

        with pytest.raises(NotImplementedError, match="candidates"):
            apply(obj, config, candidates=True)
