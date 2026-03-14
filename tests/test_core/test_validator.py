"""Tests for the two-tier validator: mesh checks and engraving checks."""

import numpy as np

from mathviz.core.container import Container
from mathviz.core.engraving import EngravingProfile
from mathviz.core.math_object import Mesh, PointCloud
from mathviz.core.validator import (
    CheckResult,
    Severity,
    ValidationResult,
    validate_engraving,
    validate_mesh,
)

# --- Helpers ---


def _make_cube_mesh() -> Mesh:
    """Create a simple watertight cube mesh centered at origin."""
    vertices = np.array(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            [0, 1, 2], [0, 2, 3],  # bottom
            [4, 6, 5], [4, 7, 6],  # top
            [0, 4, 5], [0, 5, 1],  # front
            [2, 6, 7], [2, 7, 3],  # back
            [0, 3, 7], [0, 7, 4],  # left
            [1, 5, 6], [1, 6, 2],  # right
        ],
        dtype=np.int64,
    )
    return Mesh(vertices=vertices, faces=faces)


def _make_mobius_strip() -> Mesh:
    """Create a Möbius strip mesh — open surface with boundary edges."""
    n_segments = 20
    width = 0.4
    vertices = []
    faces = []
    for i in range(n_segments):
        t = 2 * np.pi * i / n_segments
        half_t = t / 2
        cx, cy = np.cos(t), np.sin(t)
        dx = np.cos(half_t) * cx
        dy = np.cos(half_t) * cy
        dz = np.sin(half_t)
        v_inner = np.array([cx - width * dx, cy - width * dy, -width * dz])
        v_outer = np.array([cx + width * dx, cy + width * dy, width * dz])
        vertices.extend([v_inner, v_outer])

    for i in range(n_segments):
        j = (i + 1) % n_segments
        a, b = 2 * i, 2 * i + 1
        c, d = 2 * j, 2 * j + 1
        faces.append([a, c, b])
        faces.append([b, c, d])

    return Mesh(
        vertices=np.array(vertices, dtype=np.float64),
        faces=np.array(faces, dtype=np.int64),
    )


def _make_grid_cloud(count: int, extent: float = 1.0) -> PointCloud:
    """Create a uniform grid point cloud centered at origin."""
    n = max(1, int(round(count ** (1 / 3))))
    coords = np.linspace(-extent, extent, n)
    xx, yy, zz = np.meshgrid(coords, coords, coords)
    points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]).astype(np.float64)
    return PointCloud(points=points)


def _default_container() -> Container:
    """Create a default container (100×100×40, 5mm margin)."""
    return Container.with_uniform_margin()


def _default_profile() -> EngravingProfile:
    """Create a default engraving profile."""
    return EngravingProfile()


# --- ValidationResult tests ---


class TestValidationResult:
    """Test ValidationResult dataclass behavior."""

    def test_empty_result_passes(self) -> None:
        """Empty result with no checks should pass."""
        result = ValidationResult()
        assert result.passed is True

    def test_passed_with_only_warnings(self) -> None:
        """Result passes when there are only warnings, no errors."""
        result = ValidationResult(
            checks=[
                CheckResult("a", False, Severity.WARNING, "warn"),
                CheckResult("b", True, Severity.INFO, "ok"),
            ]
        )
        assert result.passed is True

    def test_fails_with_error(self) -> None:
        """Result fails when there is at least one error."""
        result = ValidationResult(
            checks=[
                CheckResult("a", False, Severity.ERROR, "bad"),
                CheckResult("b", True, Severity.INFO, "ok"),
            ]
        )
        assert result.passed is False


# --- Mesh validation tests ---


class TestMeshValidation:
    """Test mesh validation checks."""

    def test_watertight_cube_passes(self) -> None:
        """A watertight cube should pass all mesh checks."""
        mesh = _make_cube_mesh()
        result = validate_mesh(mesh)
        assert result.passed is True

    def test_boundary_edges_trigger_warning_not_error(self) -> None:
        """Mesh with boundary edges (Möbius strip) triggers warning, not error."""
        mesh = _make_mobius_strip()
        result = validate_mesh(mesh)
        # Should pass (warnings allowed)
        assert result.passed is True
        # Should have a warning about manifold/boundary edges
        warning_names = {c.name for c in result.warnings}
        assert "manifold" in warning_names or "watertight" in warning_names

    def test_mesh_outside_container_fails(self) -> None:
        """Mesh extending outside container triggers error."""
        mesh = _make_cube_mesh()  # extends from -1 to 1
        tiny_container = Container(
            width_mm=1.0, height_mm=1.0, depth_mm=1.0,
            margin_x_mm=0.1, margin_y_mm=0.1, margin_z_mm=0.1,
        )
        result = validate_mesh(mesh, container=tiny_container)
        assert result.passed is False
        error_names = {c.name for c in result.errors}
        assert "container_bounds" in error_names

    def test_mesh_inside_container_passes(self) -> None:
        """Mesh fitting inside container passes bounds check."""
        mesh = _make_cube_mesh()  # extends from -1 to 1
        big_container = Container(
            width_mm=100.0, height_mm=100.0, depth_mm=100.0,
            margin_x_mm=5.0, margin_y_mm=5.0, margin_z_mm=5.0,
        )
        result = validate_mesh(mesh, container=big_container)
        bounds_checks = [c for c in result.checks if c.name == "container_bounds"]
        assert len(bounds_checks) == 1
        assert bounds_checks[0].passed is True

    def test_normals_with_nan_fails(self) -> None:
        """Mesh with NaN normals triggers error."""
        mesh = _make_cube_mesh()
        normals = np.zeros((len(mesh.faces), 3), dtype=np.float64)
        normals[0, 0] = float("nan")
        mesh.normals = normals
        result = validate_mesh(mesh)
        error_names = {c.name for c in result.errors}
        assert "normals" in error_names


# --- Engraving validation tests ---


class TestEngravingValidation:
    """Test point cloud / engraving validation checks."""

    def test_point_cloud_exceeding_budget_fails(self) -> None:
        """Point cloud exceeding point budget fails engraving validation."""
        cloud = _make_grid_cloud(1000)  # ~1000 points
        profile = EngravingProfile(point_budget=10)
        result = validate_engraving(cloud, profile)
        assert result.passed is False
        error_names = {c.name for c in result.errors}
        assert "point_budget" in error_names

    def test_point_cloud_within_budget_passes(self) -> None:
        """Point cloud within budget passes."""
        cloud = _make_grid_cloud(27)  # 27 points (3^3)
        profile = EngravingProfile(point_budget=100)
        result = validate_engraving(cloud, profile)
        budget_checks = [c for c in result.checks if c.name == "point_budget"]
        assert len(budget_checks) == 1
        assert budget_checks[0].passed is True

    def test_points_outside_container_fails(self) -> None:
        """Point cloud with points outside container fails."""
        # Points at ±10mm, container usable volume is 0.8mm per axis
        cloud = PointCloud(
            points=np.array([[10.0, 0.0, 0.0], [-10.0, 0.0, 0.0]], dtype=np.float64)
        )
        tiny_container = Container(
            width_mm=1.0, height_mm=1.0, depth_mm=1.0,
            margin_x_mm=0.1, margin_y_mm=0.1, margin_z_mm=0.1,
        )
        profile = _default_profile()
        result = validate_engraving(cloud, profile, container=tiny_container)
        assert result.passed is False
        error_names = {c.name for c in result.errors}
        assert "container_bounds" in error_names

    def test_points_inside_container_passes(self) -> None:
        """Point cloud inside container passes bounds check."""
        cloud = PointCloud(
            points=np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float64)
        )
        big_container = Container(
            width_mm=100.0, height_mm=100.0, depth_mm=100.0,
            margin_x_mm=5.0, margin_y_mm=5.0, margin_z_mm=5.0,
        )
        profile = _default_profile()
        result = validate_engraving(cloud, profile, container=big_container)
        bounds_checks = [c for c in result.checks if c.name == "container_bounds"]
        assert len(bounds_checks) == 1
        assert bounds_checks[0].passed is True

    def test_opacity_warning_fires_when_dense(self) -> None:
        """Opacity warning fires when >70% of voxels in a projection are occupied."""
        # Create a dense filled cube — every grid cell occupied in all projections
        n = 20
        coords = np.linspace(-1, 1, n)
        xx, yy, zz = np.meshgrid(coords, coords, coords)
        points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]).astype(np.float64)
        cloud = PointCloud(points=points)
        profile = EngravingProfile(point_budget=100_000)
        result = validate_engraving(cloud, profile)
        opacity_checks = [c for c in result.checks if c.name == "opacity"]
        assert len(opacity_checks) == 1
        assert opacity_checks[0].passed is False
        assert opacity_checks[0].severity == Severity.WARNING

    def test_opacity_passes_for_sparse_cloud(self) -> None:
        """Sparse point cloud should not trigger opacity warning."""
        # A few widely spaced points
        points = np.array(
            [[0.0, 0.0, 0.0], [10.0, 10.0, 10.0], [20.0, 0.0, 20.0]],
            dtype=np.float64,
        )
        cloud = PointCloud(points=points)
        profile = EngravingProfile(point_budget=100_000)
        result = validate_engraving(cloud, profile)
        opacity_checks = [c for c in result.checks if c.name == "opacity"]
        assert len(opacity_checks) == 1
        assert opacity_checks[0].passed is True

    def test_min_spacing_warning(self) -> None:
        """Points closer than min spacing should trigger warning."""
        points = np.array(
            [[0.0, 0.0, 0.0], [0.001, 0.0, 0.0], [10.0, 10.0, 10.0]],
            dtype=np.float64,
        )
        cloud = PointCloud(points=points)
        profile = EngravingProfile(min_point_spacing_mm=0.05)
        result = validate_engraving(cloud, profile)
        spacing_checks = [c for c in result.checks if c.name == "min_spacing"]
        assert len(spacing_checks) == 1
        assert spacing_checks[0].passed is False

    def test_max_gap_warning(self) -> None:
        """Points with large gaps should trigger warning."""
        points = np.array(
            [[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        cloud = PointCloud(points=points)
        profile = EngravingProfile(max_point_spacing_mm=2.0)
        result = validate_engraving(cloud, profile)
        gap_checks = [c for c in result.checks if c.name == "max_gap"]
        assert len(gap_checks) == 1
        assert gap_checks[0].passed is False
