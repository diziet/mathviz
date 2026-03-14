"""Tests for the sampler pipeline stage."""

import numpy as np
import pytest
import trimesh

from mathviz.core.math_object import MathObject, Mesh, PointCloud
from mathviz.pipeline.sampler import SamplerConfig, SamplingMethod, sample


def _make_cube_mesh(size: float = 10.0) -> Mesh:
    """Create a watertight cube mesh centered at origin."""
    tm = trimesh.primitives.Box(extents=(size, size, size))
    return Mesh(
        vertices=np.array(tm.vertices, dtype=np.float64),
        faces=np.array(tm.faces, dtype=np.int64),
    )


def _make_sphere_mesh(radius: float = 5.0) -> Mesh:
    """Create a watertight sphere mesh centered at origin."""
    tm = trimesh.primitives.Sphere(radius=radius, subdivisions=3)
    return Mesh(
        vertices=np.array(tm.vertices, dtype=np.float64),
        faces=np.array(tm.faces, dtype=np.int64),
    )


def _make_math_object(mesh: Mesh) -> MathObject:
    """Create a MathObject with a mesh."""
    return MathObject(mesh=mesh, generator_name="test")


class TestVolumeFillCube:
    """Volume fill of a cube produces points only inside the cube."""

    def test_all_points_inside_cube(self) -> None:
        size = 10.0
        obj = _make_math_object(_make_cube_mesh(size))
        config = SamplerConfig(
            method=SamplingMethod.VOLUME_FILL,
            num_points=500,
            seed=42,
        )
        result = sample(obj, config)

        assert result.point_cloud is not None
        points = result.point_cloud.points
        half = size / 2.0
        # All points must be inside the cube (with small tolerance for jitter)
        assert np.all(points >= -half - 0.01), "Points found below cube min"
        assert np.all(points <= half + 0.01), "Points found above cube max"

    def test_volume_fill_produces_points(self) -> None:
        obj = _make_math_object(_make_cube_mesh(10.0))
        config = SamplerConfig(
            method=SamplingMethod.VOLUME_FILL,
            num_points=200,
            seed=99,
        )
        result = sample(obj, config)
        assert result.point_cloud is not None
        assert len(result.point_cloud.points) > 0


class TestSkipExisting:
    """Sampler skips when point_cloud already exists and no resampling flag."""

    def test_skip_when_cloud_exists(self) -> None:
        existing_cloud = PointCloud(
            points=np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
        )
        obj = MathObject(
            mesh=_make_cube_mesh(),
            point_cloud=existing_cloud,
            generator_name="test",
        )
        config = SamplerConfig(
            method=SamplingMethod.UNIFORM_SURFACE,
            num_points=100,
        )
        result = sample(obj, config)
        # Should return the same point cloud unchanged
        assert result.point_cloud is existing_cloud
        assert len(result.point_cloud.points) == 1

    def test_resample_when_flag_set(self) -> None:
        existing_cloud = PointCloud(
            points=np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
        )
        obj = MathObject(
            mesh=_make_cube_mesh(),
            point_cloud=existing_cloud,
            generator_name="test",
        )
        config = SamplerConfig(
            method=SamplingMethod.RANDOM_SURFACE,
            num_points=100,
            resample=True,
        )
        result = sample(obj, config)
        # Should have been resampled with many more points
        assert result.point_cloud is not existing_cloud
        assert len(result.point_cloud.points) > 1


class TestDensityScaling:
    """Sampler with density parameter scales point count with surface area."""

    def test_density_scales_with_area(self) -> None:
        small_mesh = _make_cube_mesh(5.0)  # surface area = 6 * 5^2 = 150
        large_mesh = _make_cube_mesh(10.0)  # surface area = 6 * 10^2 = 600

        config = SamplerConfig(
            method=SamplingMethod.RANDOM_SURFACE,
            density=0.5,
            seed=42,
        )

        result_small = sample(_make_math_object(small_mesh), config)
        result_large = sample(_make_math_object(large_mesh), config)

        count_small = len(result_small.point_cloud.points)
        count_large = len(result_large.point_cloud.points)

        # Large cube has 4x the surface area, so should have ~4x the points
        ratio = count_large / count_small
        assert 3.0 < ratio < 5.0, f"Expected ratio ~4.0, got {ratio:.2f}"


class TestNumPointsAccuracy:
    """Requesting N points produces approximately N points (within 10%)."""

    @pytest.mark.parametrize("target", [100, 500, 1000])
    def test_random_surface_num_points(self, target: int) -> None:
        obj = _make_math_object(_make_sphere_mesh(10.0))
        config = SamplerConfig(
            method=SamplingMethod.RANDOM_SURFACE,
            num_points=target,
            seed=42,
        )
        result = sample(obj, config)
        actual = len(result.point_cloud.points)
        assert abs(actual - target) / target < 0.10, (
            f"Expected ~{target}, got {actual}"
        )

    def test_uniform_surface_num_points(self) -> None:
        target = 500
        obj = _make_math_object(_make_sphere_mesh(10.0))
        config = SamplerConfig(
            method=SamplingMethod.UNIFORM_SURFACE,
            num_points=target,
            seed=42,
        )
        result = sample(obj, config)
        actual = len(result.point_cloud.points)
        # sample_surface_even may return fewer points due to rejection
        assert actual <= target
        assert actual > target * 0.5, f"Expected >50% of {target}, got {actual}"


class TestEdgeCases:
    """Edge cases and error handling."""

    def test_no_mesh_raises(self) -> None:
        obj = MathObject(generator_name="test", point_cloud=PointCloud(
            points=np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        ))
        obj.point_cloud = None
        config = SamplerConfig(method=SamplingMethod.UNIFORM_SURFACE, num_points=10)
        with pytest.raises(ValueError, match="requires a mesh"):
            sample(obj, config)

    def test_density_and_num_points_mutually_exclusive(self) -> None:
        with pytest.raises(ValueError, match="mutually exclusive"):
            SamplerConfig(density=1.0, num_points=100)

    def test_output_dtype_is_float64(self) -> None:
        obj = _make_math_object(_make_cube_mesh())
        config = SamplerConfig(
            method=SamplingMethod.RANDOM_SURFACE,
            num_points=50,
        )
        result = sample(obj, config)
        assert result.point_cloud.points.dtype == np.float64

    def test_output_shape_is_nx3(self) -> None:
        obj = _make_math_object(_make_cube_mesh())
        config = SamplerConfig(
            method=SamplingMethod.RANDOM_SURFACE,
            num_points=50,
        )
        result = sample(obj, config)
        assert result.point_cloud.points.ndim == 2
        assert result.point_cloud.points.shape[1] == 3
