"""Tests for geometry dataclasses and validation."""

import numpy as np
import pytest

from mathviz.core.math_object import (
    BoundingBox,
    CoordSpace,
    Curve,
    MathObject,
    Mesh,
    PointCloud,
)


def _valid_mesh() -> Mesh:
    """Create a minimal valid mesh (single triangle)."""
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
    )
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    return Mesh(vertices=vertices, faces=faces)


def _valid_point_cloud() -> PointCloud:
    """Create a minimal valid point cloud."""
    points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float64)
    return PointCloud(points=points)


def _valid_curve() -> Curve:
    """Create a minimal valid curve."""
    points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float64)
    return Curve(points=points)


class TestMathObjectNoGeometry:
    """MathObject with no geometry (all None) fails validation."""

    def test_no_geometry_fails_validation(self) -> None:
        obj = MathObject()
        errors = obj.validate()
        assert len(errors) == 1
        assert "no geometry" in errors[0]

    def test_no_geometry_validate_or_raise(self) -> None:
        obj = MathObject()
        with pytest.raises(ValueError, match="no geometry"):
            obj.validate_or_raise()

    def test_empty_curves_list_fails_validation(self) -> None:
        obj = MathObject(curves=[])
        errors = obj.validate()
        assert any("no geometry" in e for e in errors)


class TestMeshValidation:
    """Mesh validation tests."""

    def test_valid_mesh_passes(self) -> None:
        mesh = _valid_mesh()
        assert mesh.validate() == []

    def test_out_of_bounds_face_indices(self) -> None:
        vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
        faces = np.array([[0, 1, 5]], dtype=np.int64)  # index 5 out of bounds
        mesh = Mesh(vertices=vertices, faces=faces)
        errors = mesh.validate()
        assert any("out of bounds" in e for e in errors)

    def test_negative_face_indices(self) -> None:
        vertices = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
        )
        faces = np.array([[0, 1, -1]], dtype=np.int64)
        mesh = Mesh(vertices=vertices, faces=faces)
        errors = mesh.validate()
        assert any("negative" in e for e in errors)

    def test_nan_in_vertices(self) -> None:
        vertices = np.array(
            [[np.nan, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64
        )
        faces = np.array([[0, 1, 2]], dtype=np.int64)
        mesh = Mesh(vertices=vertices, faces=faces)
        errors = mesh.validate()
        assert any("NaN" in e for e in errors)

    def test_wrong_vertices_shape(self) -> None:
        vertices = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
        faces = np.array([[0, 1, 0]], dtype=np.int64)
        mesh = Mesh(vertices=vertices, faces=faces)
        errors = mesh.validate()
        assert any("vertices shape" in e for e in errors)

    def test_wrong_vertices_dtype(self) -> None:
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.int32)
        faces = np.array([[0, 1, 2]], dtype=np.int64)
        mesh = Mesh(vertices=vertices, faces=faces)
        errors = mesh.validate()
        assert any("vertices dtype" in e for e in errors)


class TestPointCloudValidation:
    """PointCloud validation tests."""

    def test_valid_point_cloud_passes(self) -> None:
        cloud = _valid_point_cloud()
        assert cloud.validate() == []

    def test_mismatched_intensities_length(self) -> None:
        points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float64)
        intensities = np.array([0.5], dtype=np.float64)  # length 1, but 2 points
        cloud = PointCloud(points=points, intensities=intensities)
        errors = cloud.validate()
        assert any("intensities length" in e for e in errors)

    def test_mismatched_normals_length(self) -> None:
        points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float64)
        normals = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)  # 1 normal, 2 points
        cloud = PointCloud(points=points, normals=normals)
        errors = cloud.validate()
        assert any("normals length" in e for e in errors)

    def test_nan_in_points(self) -> None:
        points = np.array([[0.0, np.nan, 0.0]], dtype=np.float64)
        cloud = PointCloud(points=points)
        errors = cloud.validate()
        assert any("NaN" in e for e in errors)


class TestCurveValidation:
    """Curve validation tests."""

    def test_valid_curve_passes(self) -> None:
        curve = _valid_curve()
        assert curve.validate() == []

    def test_nan_in_curve_points(self) -> None:
        points = np.array([[np.nan, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float64)
        curve = Curve(points=points)
        errors = curve.validate()
        assert any("NaN" in e for e in errors)


class TestMathObjectValidation:
    """MathObject-level validation tests."""

    def test_valid_mesh_object_passes(self) -> None:
        obj = MathObject(mesh=_valid_mesh())
        assert obj.validate() == []

    def test_valid_point_cloud_object_passes(self) -> None:
        obj = MathObject(point_cloud=_valid_point_cloud())
        assert obj.validate() == []

    def test_valid_curves_object_passes(self) -> None:
        obj = MathObject(curves=[_valid_curve()])
        assert obj.validate() == []

    def test_mesh_errors_prefixed(self) -> None:
        vertices = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        faces = np.array([[0, 1, 2]], dtype=np.int64)
        obj = MathObject(mesh=Mesh(vertices=vertices, faces=faces))
        errors = obj.validate()
        assert any(e.startswith("mesh:") for e in errors)

    def test_point_cloud_errors_prefixed(self) -> None:
        points = np.array([[0.0, np.nan, 0.0]], dtype=np.float64)
        obj = MathObject(point_cloud=PointCloud(points=points))
        errors = obj.validate()
        assert any(e.startswith("point_cloud:") for e in errors)

    def test_validate_or_raise_joins_all_errors(self) -> None:
        """validate_or_raise raises ValueError with all errors joined by semicolons."""
        obj = MathObject()  # no geometry
        with pytest.raises(ValueError) as exc_info:
            obj.validate_or_raise()
        assert "Invalid MathObject:" in str(exc_info.value)
        assert "no geometry" in str(exc_info.value)

    def test_validate_or_raise_multiple_errors_joined(self) -> None:
        """Multiple errors are joined with semicolons."""
        # Mesh with NaN vertices AND out-of-bounds faces
        vertices = np.array([[np.nan, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
        faces = np.array([[0, 1, 5]], dtype=np.int64)
        obj = MathObject(mesh=Mesh(vertices=vertices, faces=faces))
        with pytest.raises(ValueError) as exc_info:
            obj.validate_or_raise()
        msg = str(exc_info.value)
        assert "; " in msg  # errors joined by semicolons
        assert "NaN" in msg
        assert "out of bounds" in msg


class TestCoordSpace:
    """CoordSpace enum tests."""

    def test_default_is_abstract(self) -> None:
        obj = MathObject(mesh=_valid_mesh())
        assert obj.coord_space == CoordSpace.ABSTRACT

    def test_can_set_physical(self) -> None:
        obj = MathObject(mesh=_valid_mesh(), coord_space=CoordSpace.PHYSICAL)
        assert obj.coord_space == CoordSpace.PHYSICAL

    def test_enum_values(self) -> None:
        assert CoordSpace.ABSTRACT.value == "abstract"
        assert CoordSpace.PHYSICAL.value == "physical"


class TestBoundingBox:
    """BoundingBox tests."""

    def test_default_bounding_box_is_none(self) -> None:
        obj = MathObject(mesh=_valid_mesh())
        assert obj.bounding_box is None

    def test_custom_bounding_box(self) -> None:
        bb = BoundingBox(min_corner=(-1.0, -2.0, -3.0), max_corner=(1.0, 2.0, 3.0))
        obj = MathObject(mesh=_valid_mesh(), bounding_box=bb)
        assert obj.bounding_box.min_corner == (-1.0, -2.0, -3.0)
