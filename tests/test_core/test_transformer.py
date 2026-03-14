"""Tests for the transformer pipeline stage."""

import numpy as np
import pytest

from mathviz.core.container import Container, PlacementPolicy
from mathviz.core.math_object import (
    BoundingBox,
    CoordSpace,
    Curve,
    MathObject,
    Mesh,
    PointCloud,
)
from mathviz.pipeline.transformer import fit


def _make_cube_mesh(size: float = 2.0, center: tuple = (0, 0, 0)) -> Mesh:
    """Create a simple cube mesh centered at the given point."""
    half = size / 2.0
    cx, cy, cz = center
    vertices = np.array([
        [cx - half, cy - half, cz - half],
        [cx + half, cy - half, cz - half],
        [cx + half, cy + half, cz - half],
        [cx - half, cy + half, cz - half],
        [cx - half, cy - half, cz + half],
        [cx + half, cy - half, cz + half],
        [cx + half, cy + half, cz + half],
        [cx - half, cy + half, cz + half],
    ], dtype=np.float64)
    faces = np.array([
        [0, 1, 2], [0, 2, 3],
        [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4],
        [2, 3, 7], [2, 7, 6],
        [0, 3, 7], [0, 7, 4],
        [1, 2, 6], [1, 6, 5],
    ], dtype=np.int64)
    return Mesh(vertices=vertices, faces=faces)


def _make_abstract_obj(
    mesh: Mesh | None = None,
    point_cloud: PointCloud | None = None,
    curves: list[Curve] | None = None,
) -> MathObject:
    """Create a MathObject in ABSTRACT space."""
    return MathObject(
        mesh=mesh,
        point_cloud=point_cloud,
        curves=curves,
        coord_space=CoordSpace.ABSTRACT,
        generator_name="test",
    )


def _fit_with_depth_bias(depth_bias: float) -> BoundingBox:
    """Run transformer with given depth_bias and return result bounding box."""
    mesh = _make_cube_mesh(size=2.0)
    obj = _make_abstract_obj(mesh=mesh)
    container = Container.with_uniform_margin(w=100, h=100, d=100, margin=5)
    policy = PlacementPolicy(depth_bias=depth_bias)
    result = fit(obj, container, policy)
    return result.bounding_box


class TestAbstractToPhysical:
    """Test that coord_space transitions correctly."""

    def test_output_is_physical(self) -> None:
        """Abstract->Physical sets coord_space to PHYSICAL."""
        mesh = _make_cube_mesh()
        obj = _make_abstract_obj(mesh=mesh)
        container = Container()
        policy = PlacementPolicy()

        result = fit(obj, container, policy)

        assert result.coord_space == CoordSpace.PHYSICAL

    def test_already_physical_raises(self) -> None:
        """Already-PHYSICAL input raises an error."""
        mesh = _make_cube_mesh()
        obj = MathObject(
            mesh=mesh,
            coord_space=CoordSpace.PHYSICAL,
        )
        container = Container()
        policy = PlacementPolicy()

        with pytest.raises(ValueError, match="already in PHYSICAL"):
            fit(obj, container, policy)


class TestBoundingBoxFits:
    """Test that output fits within the container usable volume."""

    def test_output_fits_in_container(self) -> None:
        """Output bounding box fits within container usable volume."""
        mesh = _make_cube_mesh(size=10.0)
        obj = _make_abstract_obj(mesh=mesh)
        container = Container()
        policy = PlacementPolicy()

        result = fit(obj, container, policy)

        usable = container.usable_volume
        bbox = result.bounding_box
        size = bbox.size

        for i in range(3):
            assert size[i] <= usable[i] + 1e-9, (
                f"Axis {i}: size {size[i]} exceeds usable {usable[i]}"
            )

    def test_non_cube_container_fits(self) -> None:
        """Output fits even with asymmetric container."""
        mesh = _make_cube_mesh(size=5.0)
        obj = _make_abstract_obj(mesh=mesh)
        container = Container(width_mm=200, height_mm=50, depth_mm=80)
        policy = PlacementPolicy()

        result = fit(obj, container, policy)

        usable = container.usable_volume
        bbox = result.bounding_box
        size = bbox.size

        for i in range(3):
            assert size[i] <= usable[i] + 1e-9


class TestAspectRatioPreservation:
    """Test aspect ratio preservation during scaling."""

    def test_aspect_ratio_preserved(self) -> None:
        """A 2:1:1 object in a cube container stays 2:1:1."""
        vertices = np.array([
            [0, 0, 0],
            [4, 0, 0],
            [4, 2, 0],
            [0, 2, 0],
            [0, 0, 2],
            [4, 0, 2],
            [4, 2, 2],
            [0, 2, 2],
        ], dtype=np.float64)
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        mesh = Mesh(vertices=vertices, faces=faces)
        obj = _make_abstract_obj(mesh=mesh)

        container = Container.with_uniform_margin(w=100, h=100, d=100, margin=5)
        policy = PlacementPolicy(preserve_aspect_ratio=True)

        result = fit(obj, container, policy)
        bbox = result.bounding_box
        sx, sy, sz = bbox.size

        ratio_xy = sx / sy
        ratio_xz = sx / sz
        assert abs(ratio_xy - 2.0) < 1e-6, f"x:y ratio {ratio_xy}, expected 2.0"
        assert abs(ratio_xz - 2.0) < 1e-6, f"x:z ratio {ratio_xz}, expected 2.0"

    def test_non_uniform_scaling_when_disabled(self) -> None:
        """preserve_aspect_ratio=False allows per-axis scaling."""
        vertices = np.array([
            [0, 0, 0],
            [4, 0, 0],
            [4, 2, 0],
            [0, 2, 0],
            [0, 0, 1],
            [4, 0, 1],
            [4, 2, 1],
            [0, 2, 1],
        ], dtype=np.float64)
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        mesh = Mesh(vertices=vertices, faces=faces)
        obj = _make_abstract_obj(mesh=mesh)

        container = Container.with_uniform_margin(w=100, h=100, d=100, margin=5)
        policy = PlacementPolicy(preserve_aspect_ratio=False)

        result = fit(obj, container, policy)
        bbox = result.bounding_box
        sx, sy, sz = bbox.size

        usable = container.usable_volume
        for i, s in enumerate([sx, sy, sz]):
            assert abs(s - usable[i]) < 1e-6, (
                f"Axis {i}: size {s} should fill usable {usable[i]}"
            )


class TestDepthBias:
    """Test depth_bias z-axis scaling."""

    def test_depth_bias_gt_1_stretches_z(self) -> None:
        """depth_bias > 1.0 stretches z relative to x/y."""
        bbox_normal = _fit_with_depth_bias(1.0)
        bbox_stretched = _fit_with_depth_bias(1.5)

        assert bbox_stretched.size[2] > bbox_normal.size[2] * 1.4, (
            f"z with bias=1.5 ({bbox_stretched.size[2]}) should be ~1.5x "
            f"z with bias=1.0 ({bbox_normal.size[2]})"
        )

        assert abs(bbox_normal.size[0] - bbox_stretched.size[0]) < 1e-6, (
            "x should be unchanged by depth_bias"
        )
        assert abs(bbox_normal.size[1] - bbox_stretched.size[1]) < 1e-6, (
            "y should be unchanged by depth_bias"
        )

    def test_depth_bias_lt_1_compresses_z(self) -> None:
        """depth_bias < 1.0 compresses z relative to x/y."""
        bbox_normal = _fit_with_depth_bias(1.0)
        bbox_compressed = _fit_with_depth_bias(0.7)

        assert bbox_compressed.size[2] < bbox_normal.size[2] * 0.8, (
            f"z with bias=0.7 ({bbox_compressed.size[2]}) should be ~0.7x "
            f"z with bias=1.0 ({bbox_normal.size[2]})"
        )

        assert abs(bbox_normal.size[0] - bbox_compressed.size[0]) < 1e-6, (
            "x should be unchanged by depth_bias"
        )
        assert abs(bbox_normal.size[1] - bbox_compressed.size[1]) < 1e-6, (
            "y should be unchanged by depth_bias"
        )


class TestAnchor:
    """Test anchor positioning."""

    def test_anchor_bottom(self) -> None:
        """Anchor 'bottom' places geometry against the bottom face."""
        mesh = _make_cube_mesh(size=2.0)
        obj = _make_abstract_obj(mesh=mesh)
        container = Container.with_uniform_margin(w=100, h=100, d=100, margin=5)
        policy = PlacementPolicy(anchor="bottom")

        result = fit(obj, container, policy)
        bbox = result.bounding_box

        margin_y = container.margin_y_mm
        assert abs(bbox.min_corner[1] - margin_y) < 1e-6, (
            f"Bottom face at {bbox.min_corner[1]}, expected {margin_y}"
        )

    def test_anchor_center(self) -> None:
        """Anchor 'center' places geometry at container center."""
        mesh = _make_cube_mesh(size=2.0)
        obj = _make_abstract_obj(mesh=mesh)
        container = Container()
        policy = PlacementPolicy(anchor="center")

        result = fit(obj, container, policy)
        bbox = result.bounding_box
        obj_center_x = (bbox.min_corner[0] + bbox.max_corner[0]) / 2.0
        obj_center_y = (bbox.min_corner[1] + bbox.max_corner[1]) / 2.0

        assert abs(obj_center_x - container.width_mm / 2.0) < 1e-6
        assert abs(obj_center_y - container.height_mm / 2.0) < 1e-6


class TestRotation:
    """Test rotation before scaling."""

    def test_rotation_applies_before_scaling(self) -> None:
        """A rotated object still fits the container."""
        vertices = np.array([
            [0, 0, 0],
            [10, 0, 0],
            [10, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [10, 0, 1],
            [10, 1, 1],
            [0, 1, 1],
        ], dtype=np.float64)
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        mesh = Mesh(vertices=vertices, faces=faces)
        obj = _make_abstract_obj(mesh=mesh)

        container = Container.with_uniform_margin(w=100, h=100, d=100, margin=5)
        policy = PlacementPolicy(rotation_degrees=(0, 0, 45))

        result = fit(obj, container, policy)

        usable = container.usable_volume
        bbox = result.bounding_box
        for i in range(3):
            assert bbox.size[i] <= usable[i] + 1e-6, (
                f"Axis {i}: size {bbox.size[i]} > usable {usable[i]}"
            )


class TestAllGeometryTypes:
    """Test that all three geometry types are transformed when present."""

    def test_all_types_transformed(self) -> None:
        """Mesh, cloud, and curves are all transformed."""
        mesh = _make_cube_mesh(size=2.0)
        cloud = PointCloud(
            points=np.array([
                [0, 0, 0], [1, 1, 1], [-1, -1, -1],
            ], dtype=np.float64)
        )
        curve = Curve(
            points=np.array([
                [0, 0, 0], [0.5, 0.5, 0.5], [1, 0, 0],
            ], dtype=np.float64)
        )
        obj = _make_abstract_obj(mesh=mesh, point_cloud=cloud, curves=[curve])

        container = Container()
        policy = PlacementPolicy()

        result = fit(obj, container, policy)

        assert result.coord_space == CoordSpace.PHYSICAL
        assert result.mesh is not None
        assert result.point_cloud is not None
        assert result.curves is not None

        assert not np.array_equal(obj.mesh.vertices, result.mesh.vertices)
        assert not np.array_equal(obj.point_cloud.points, result.point_cloud.points)
        assert not np.array_equal(obj.curves[0].points, result.curves[0].points)

    def test_original_not_mutated(self) -> None:
        """Original MathObject is not modified."""
        mesh = _make_cube_mesh(size=2.0)
        original_verts = mesh.vertices.copy()
        obj = _make_abstract_obj(mesh=mesh)

        container = Container()
        policy = PlacementPolicy()

        fit(obj, container, policy)

        assert np.array_equal(obj.mesh.vertices, original_verts)
        assert obj.coord_space == CoordSpace.ABSTRACT


class TestOffset:
    """Test offset application."""

    def test_offset_shifts_geometry(self) -> None:
        """Offset shifts geometry by the specified amount."""
        mesh = _make_cube_mesh(size=2.0)
        obj = _make_abstract_obj(mesh=mesh)
        container = Container()

        policy_no_offset = PlacementPolicy()
        result_no_offset = fit(obj, container, policy_no_offset)

        policy_with_offset = PlacementPolicy(offset_mm=(5.0, -3.0, 2.0))
        result_with_offset = fit(obj, container, policy_with_offset)

        diff = result_with_offset.mesh.vertices - result_no_offset.mesh.vertices
        expected_offset = np.array([5.0, -3.0, 2.0])

        assert np.allclose(diff, expected_offset, atol=1e-9)


class TestBoundingBoxUpdated:
    """Test that the output bounding box is correctly computed."""

    def test_bounding_box_matches_geometry(self) -> None:
        """Output bounding box matches actual geometry extents."""
        mesh = _make_cube_mesh(size=3.0)
        obj = _make_abstract_obj(mesh=mesh)
        container = Container()
        policy = PlacementPolicy()

        result = fit(obj, container, policy)

        actual_min = result.mesh.vertices.min(axis=0)
        actual_max = result.mesh.vertices.max(axis=0)

        assert np.allclose(result.bounding_box.min_corner, actual_min, atol=1e-9)
        assert np.allclose(result.bounding_box.max_corner, actual_max, atol=1e-9)
