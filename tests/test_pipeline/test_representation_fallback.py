"""Tests for representation fallback logic (Task 39)."""

import numpy as np
import pytest

from mathviz.core.generator import clear_registry, list_generators
from mathviz.core.math_object import Curve, MathObject, Mesh, PointCloud
from mathviz.core.representation import RepresentationType
from mathviz.pipeline.representation_defaults import GENERATOR_DEFAULTS
from mathviz.pipeline.representation_strategy import (
    _get_fallback,
    apply,
    get_default,
)


# --- Helpers ---


def _simple_curve() -> Curve:
    """Create a simple 3D curve."""
    t = np.linspace(0, 2 * np.pi, 50)
    points = np.column_stack([np.cos(t), np.sin(t), np.zeros_like(t)])
    return Curve(points=points.astype(np.float64))


def _simple_mesh() -> Mesh:
    """Create a minimal triangle mesh."""
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
    ], dtype=np.float64)
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int64)
    return Mesh(vertices=vertices, faces=faces)


def _simple_cloud() -> PointCloud:
    """Create a minimal point cloud."""
    points = np.array([
        [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1],
    ], dtype=np.float64)
    return PointCloud(points=points)


# --- Named generator defaults ---


class TestNamedGeneratorDefaults:
    """Verify specific generators get the correct representation type."""

    def test_kepler_orbit_gets_tube(self) -> None:
        """kepler_orbit (curve) gets TUBE representation by default."""
        config = get_default("kepler_orbit")
        assert config.type == RepresentationType.TUBE

    def test_cardioid_gets_tube(self) -> None:
        """cardioid (curve) gets TUBE representation by default."""
        config = get_default("cardioid")
        assert config.type == RepresentationType.TUBE

    def test_sacks_spiral_gets_sparse_shell(self) -> None:
        """sacks_spiral (point cloud) gets SPARSE_SHELL by default."""
        config = get_default("sacks_spiral")
        assert config.type == RepresentationType.SPARSE_SHELL

    def test_torus_gets_surface_shell(self) -> None:
        """torus (mesh) still gets SURFACE_SHELL representation."""
        config = get_default("torus")
        assert config.type == RepresentationType.SURFACE_SHELL


# --- Geometry-based fallback ---


class TestGetFallback:
    """_get_fallback inspects geometry and picks a compatible representation."""

    def test_curve_only_gets_tube(self) -> None:
        """MathObject with only curves gets TUBE fallback."""
        obj = MathObject(curves=[_simple_curve()], generator_name="unknown_curve")
        config = _get_fallback(obj)
        assert config.type == RepresentationType.TUBE
        assert config.tube_radius is not None
        assert config.tube_radius > 0

    def test_point_cloud_only_gets_sparse_shell(self) -> None:
        """MathObject with only point_cloud gets SPARSE_SHELL fallback."""
        obj = MathObject(point_cloud=_simple_cloud(), generator_name="unknown_cloud")
        config = _get_fallback(obj)
        assert config.type == RepresentationType.SPARSE_SHELL

    def test_mesh_only_gets_surface_shell(self) -> None:
        """MathObject with mesh gets SURFACE_SHELL fallback."""
        obj = MathObject(mesh=_simple_mesh(), generator_name="unknown_mesh")
        config = _get_fallback(obj)
        assert config.type == RepresentationType.SURFACE_SHELL

    def test_scalar_field_only_gets_heightmap(self) -> None:
        """MathObject with only scalar_field gets HEIGHTMAP_RELIEF fallback."""
        field = np.ones((10, 10), dtype=np.float64)
        obj = MathObject(scalar_field=field, generator_name="unknown_heightmap")
        config = _get_fallback(obj)
        assert config.type == RepresentationType.HEIGHTMAP_RELIEF

    def test_no_geometry_raises(self) -> None:
        """MathObject with no geometry raises a clear error."""
        obj = MathObject(generator_name="empty")
        with pytest.raises(ValueError, match="no mesh, curves, point_cloud, or scalar_field"):
            _get_fallback(obj)

    def test_tube_radius_based_on_bbox(self) -> None:
        """Fallback tube_radius is ~1% of the bounding-box diagonal."""
        obj = MathObject(curves=[_simple_curve()], generator_name="test")
        config = _get_fallback(obj)
        # Curve spans [-1, 1] in x and y → diagonal ≈ 2.83
        assert 0.01 < config.tube_radius < 0.1


# --- get_default with obj fallback ---


class TestGetDefaultWithObj:
    """get_default falls back to geometry inspection when no entry exists."""

    def test_unknown_curve_generator_gets_tube(self) -> None:
        """Unknown generator with curves gets TUBE via fallback."""
        obj = MathObject(
            curves=[_simple_curve()], generator_name="brand_new_curve_gen"
        )
        config = get_default("brand_new_curve_gen", obj=obj)
        assert config.type == RepresentationType.TUBE

    def test_unknown_cloud_generator_gets_sparse(self) -> None:
        """Unknown generator with point_cloud gets SPARSE_SHELL via fallback."""
        obj = MathObject(
            point_cloud=_simple_cloud(), generator_name="brand_new_cloud_gen"
        )
        config = get_default("brand_new_cloud_gen", obj=obj)
        assert config.type == RepresentationType.SPARSE_SHELL

    def test_known_generator_ignores_obj(self) -> None:
        """Known generator uses its default, not geometry inspection."""
        obj = MathObject(
            point_cloud=_simple_cloud(), generator_name="torus"
        )
        config = get_default("torus", obj=obj)
        assert config.type == RepresentationType.SURFACE_SHELL


# --- Apply with point-cloud SPARSE_SHELL ---


class TestApplySparseShellPointCloud:
    """SPARSE_SHELL works with point-cloud-only inputs (passthrough)."""

    def test_sparse_shell_passes_through_cloud(self) -> None:
        """SPARSE_SHELL with point-cloud-only input passes through."""
        obj = MathObject(
            point_cloud=_simple_cloud(), generator_name="sacks_spiral"
        )
        config = get_default("sacks_spiral")
        result = apply(obj, config)
        assert result.point_cloud is not None
        assert result.representation == "sparse_shell"


# --- All registered generators have compatible defaults ---


class TestAllRegisteredGenerators:
    """Every registered generator can get a representation without error."""

    @pytest.fixture(autouse=True)
    def _ensure_discovery(self) -> None:
        """Reset registry with discovery enabled so auto-discovery triggers."""
        clear_registry(suppress_discovery=False)
        yield
        clear_registry(suppress_discovery=False)

    def test_all_generators_have_representation(self) -> None:
        """All generators in the registry can resolve a representation config."""
        generators = list_generators()
        assert len(generators) > 0, "No generators found in registry"

        missing = []
        for meta in generators:
            config = get_default(meta.name)
            if config is None:
                missing.append(meta.name)

        assert missing == [], f"Generators without representation: {missing}"

    def test_all_generators_in_defaults(self) -> None:
        """All registered generators have an entry in GENERATOR_DEFAULTS."""
        generators = list_generators()
        missing = [
            meta.name
            for meta in generators
            if meta.name not in GENERATOR_DEFAULTS
        ]
        assert missing == [], (
            f"Generators missing from GENERATOR_DEFAULTS: {missing}"
        )
