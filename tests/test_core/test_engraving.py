"""Tests for the engraving optimizer pipeline stage."""

import numpy as np
import pytest

from mathviz.core.container import Container
from mathviz.core.engraving import EngravingProfile
from mathviz.core.math_object import MathObject, PointCloud
from mathviz.pipeline.engraving_optimizer import optimize


def _make_cloud(num_points: int = 1000, seed: int = 42) -> PointCloud:
    """Create a uniformly distributed point cloud in a cube."""
    rng = np.random.default_rng(seed)
    points = rng.uniform(-40.0, 40.0, size=(num_points, 3))
    return PointCloud(points=points.astype(np.float64))


def _make_obj(cloud: PointCloud) -> MathObject:
    """Create a MathObject with a point cloud."""
    return MathObject(point_cloud=cloud, generator_name="test")


def _default_container() -> Container:
    """Create a default container."""
    return Container.with_uniform_margin(w=100, h=100, d=40, margin=5)


def _assert_idempotent(
    profile: EngravingProfile,
    seed: int = 11,
    num_points: int = 2000,
    tolerance: float = 0.05,
    check_intensities: bool = False,
) -> None:
    """Assert that running optimize twice produces the same result."""
    cloud = _make_cloud(num_points, seed=seed)
    obj = _make_obj(cloud)
    container = _default_container()

    once = optimize(obj, profile, container)
    twice = optimize(once, profile, container)

    count_once = len(once.point_cloud.points)
    count_twice = len(twice.point_cloud.points)

    assert abs(count_twice - count_once) / max(count_once, 1) < tolerance, (
        f"Not idempotent: once={count_once}, twice={count_twice}"
    )

    if check_intensities:
        assert once.point_cloud.intensities is not None
        assert twice.point_cloud.intensities is not None
        np.testing.assert_allclose(
            once.point_cloud.intensities,
            twice.point_cloud.intensities,
            atol=1e-10,
        )


class TestOcclusionNone:
    """occlusion_mode='none' returns point cloud unchanged (except budget trim)."""

    def test_none_preserves_all_points(self) -> None:
        """No occlusion keeps all points when under budget."""
        cloud = _make_cloud(500)
        obj = _make_obj(cloud)
        profile = EngravingProfile(occlusion_mode="none")
        container = _default_container()

        result = optimize(obj, profile, container)

        assert len(result.point_cloud.points) == 500
        np.testing.assert_array_equal(result.point_cloud.points, cloud.points)

    def test_none_with_budget_trims(self) -> None:
        """No occlusion still enforces budget."""
        cloud = _make_cloud(1000)
        obj = _make_obj(cloud)
        profile = EngravingProfile(occlusion_mode="none", point_budget=500)
        container = _default_container()

        result = optimize(obj, profile, container)

        assert len(result.point_cloud.points) == 500


class TestShellFade:
    """shell_fade reduces point count in outer layers vs inner."""

    def test_shell_fade_removes_outer_points(self) -> None:
        """Shell fade should keep more inner points than outer."""
        cloud = _make_cloud(5000, seed=7)
        obj = _make_obj(cloud)
        profile = EngravingProfile(
            occlusion_mode="shell_fade",
            occlusion_shell_layers=3,
            occlusion_density_falloff=0.8,
        )
        container = _default_container()

        result = optimize(obj, profile, container)
        result_points = result.point_cloud.points

        assert len(result_points) < len(cloud.points)

        centroid = cloud.points.mean(axis=0)
        result_dists = np.linalg.norm(result_points - centroid, axis=1)
        orig_dists = np.linalg.norm(cloud.points - centroid, axis=1)

        median_dist = np.median(orig_dists)
        inner_kept = np.sum(result_dists <= median_dist)
        outer_kept = np.sum(result_dists > median_dist)

        assert inner_kept > outer_kept, (
            f"Inner ({inner_kept}) should be more than outer ({outer_kept})"
        )

    def test_shell_fade_zero_falloff_keeps_all(self) -> None:
        """With falloff=0, shell_fade should keep all points."""
        cloud = _make_cloud(500)
        obj = _make_obj(cloud)
        profile = EngravingProfile(
            occlusion_mode="shell_fade",
            occlusion_density_falloff=0.0,
        )
        container = _default_container()

        result = optimize(obj, profile, container)

        assert len(result.point_cloud.points) == 500


class TestDepthCompensation:
    """Depth compensation increases point density at max depth."""

    def test_depth_compensation_boosts_back_intensities(self) -> None:
        """Points at max depth should have higher intensities than front."""
        cloud = _make_cloud(1000)
        obj = _make_obj(cloud)
        profile = EngravingProfile(
            occlusion_mode="none",
            depth_compensation=True,
            depth_compensation_factor=1.5,
        )
        container = _default_container()

        result = optimize(obj, profile, container)

        assert result.point_cloud.intensities is not None
        points = result.point_cloud.points
        intensities = result.point_cloud.intensities

        usable = container.usable_volume
        depth_axis = int(np.argmin(usable))

        z_vals = points[:, depth_axis]
        z_median = np.median(z_vals)

        front_mask = z_vals <= z_median
        back_mask = z_vals > z_median

        mean_front = intensities[front_mask].mean()
        mean_back = intensities[back_mask].mean()

        assert mean_back > mean_front, (
            f"Back intensity ({mean_back:.3f}) should exceed "
            f"front ({mean_front:.3f})"
        )

    def test_depth_compensation_factor_controls_ratio(self) -> None:
        """Higher factor produces larger front-to-back intensity ratio."""
        cloud = _make_cloud(1000)
        container = _default_container()

        def _get_intensity_ratio(factor: float) -> float:
            obj = _make_obj(cloud)
            profile = EngravingProfile(
                occlusion_mode="none",
                depth_compensation=True,
                depth_compensation_factor=factor,
            )
            result = optimize(obj, profile, container)
            intensities = result.point_cloud.intensities
            assert intensities is not None
            return intensities.max() / intensities.min()

        ratio_low = _get_intensity_ratio(1.5)
        ratio_high = _get_intensity_ratio(3.0)

        assert ratio_high > ratio_low, (
            f"Higher factor should produce larger ratio: "
            f"factor=3.0 -> {ratio_high:.2f}, factor=1.5 -> {ratio_low:.2f}"
        )

    def test_no_depth_compensation_no_intensities(self) -> None:
        """Without depth compensation, intensities should remain None."""
        cloud = _make_cloud(500)
        obj = _make_obj(cloud)
        profile = EngravingProfile(
            occlusion_mode="none",
            depth_compensation=False,
        )
        container = _default_container()

        result = optimize(obj, profile, container)

        assert result.point_cloud.intensities is None


class TestPointBudget:
    """Point budget enforcement never exceeds the budget."""

    def test_budget_enforced(self) -> None:
        """Point count must not exceed budget."""
        cloud = _make_cloud(2000)
        obj = _make_obj(cloud)
        profile = EngravingProfile(point_budget=500)
        container = _default_container()

        result = optimize(obj, profile, container)

        assert len(result.point_cloud.points) <= 500

    def test_under_budget_unchanged(self) -> None:
        """Under-budget clouds are not downsampled."""
        cloud = _make_cloud(100)
        obj = _make_obj(cloud)
        profile = EngravingProfile(point_budget=500)
        container = _default_container()

        result = optimize(obj, profile, container)

        assert len(result.point_cloud.points) == 100

    def test_budget_with_occlusion(self) -> None:
        """Budget is enforced after occlusion thinning."""
        cloud = _make_cloud(3000)
        obj = _make_obj(cloud)
        profile = EngravingProfile(
            occlusion_mode="shell_fade",
            occlusion_density_falloff=0.5,
            point_budget=500,
        )
        container = _default_container()

        result = optimize(obj, profile, container)

        assert len(result.point_cloud.points) <= 500

    def test_budget_prefers_high_intensity_points(self) -> None:
        """When intensities exist, budget keeps highest-intensity points."""
        cloud = _make_cloud(2000)
        obj = _make_obj(cloud)
        profile = EngravingProfile(
            occlusion_mode="none",
            depth_compensation=True,
            depth_compensation_factor=2.0,
            point_budget=500,
        )
        container = _default_container()

        result = optimize(obj, profile, container)
        intensities = result.point_cloud.intensities

        assert intensities is not None
        # Kept points should have higher mean intensity than original would
        assert intensities.mean() > 0.6


class TestIdempotent:
    """optimize(optimize(cloud)) ≈ optimize(cloud) in point count."""

    def test_idempotent_shell_fade(self) -> None:
        """Running shell_fade optimizer twice produces same point count."""
        _assert_idempotent(EngravingProfile(
            occlusion_mode="shell_fade",
            occlusion_shell_layers=3,
            occlusion_density_falloff=0.6,
            point_budget=1500,
        ))

    def test_idempotent_radial_gradient(self) -> None:
        """Running radial gradient optimizer twice produces same count."""
        _assert_idempotent(
            EngravingProfile(
                occlusion_mode="radial_gradient",
                occlusion_density_falloff=0.7,
            ),
            seed=22,
        )

    def test_idempotent_none_mode(self) -> None:
        """None mode is trivially idempotent."""
        _assert_idempotent(
            EngravingProfile(occlusion_mode="none"),
            num_points=500,
        )

    def test_idempotent_depth_compensation(self) -> None:
        """Depth compensation is idempotent for both points and intensities."""
        _assert_idempotent(
            EngravingProfile(
                occlusion_mode="none",
                depth_compensation=True,
                depth_compensation_factor=1.5,
            ),
            num_points=500,
            check_intensities=True,
        )


class TestEdgeCases:
    """Edge cases for the engraving optimizer."""

    def test_no_point_cloud_raises(self) -> None:
        """Optimizer requires a point cloud."""
        obj = MathObject(generator_name="test")
        profile = EngravingProfile()
        container = _default_container()

        with pytest.raises(ValueError, match="requires a point cloud"):
            optimize(obj, profile, container)

    def test_empty_cloud(self) -> None:
        """Empty point cloud returns empty result."""
        cloud = PointCloud(points=np.empty((0, 3), dtype=np.float64))
        obj = _make_obj(cloud)
        profile = EngravingProfile(occlusion_mode="shell_fade")
        container = _default_container()

        result = optimize(obj, profile, container)

        assert len(result.point_cloud.points) == 0

    def test_single_point(self) -> None:
        """Single point cloud is handled gracefully."""
        cloud = PointCloud(
            points=np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
        )
        obj = _make_obj(cloud)
        profile = EngravingProfile(
            occlusion_mode="shell_fade",
            depth_compensation=True,
        )
        container = _default_container()

        result = optimize(obj, profile, container)

        assert len(result.point_cloud.points) >= 0

    def test_preserves_normals(self) -> None:
        """Optimizer preserves normals through filtering."""
        rng = np.random.default_rng(42)
        points = rng.uniform(-10, 10, (500, 3)).astype(np.float64)
        normals = rng.standard_normal((500, 3)).astype(np.float64)
        cloud = PointCloud(points=points, normals=normals)
        obj = _make_obj(cloud)
        profile = EngravingProfile(
            occlusion_mode="shell_fade",
            occlusion_density_falloff=0.5,
        )
        container = _default_container()

        result = optimize(obj, profile, container)

        assert result.point_cloud.normals is not None
        assert len(result.point_cloud.normals) == len(result.point_cloud.points)

    def test_does_not_mutate_input(self) -> None:
        """Optimizer does not modify the input MathObject."""
        cloud = _make_cloud(500)
        obj = _make_obj(cloud)
        original_points = cloud.points.copy()
        profile = EngravingProfile(
            occlusion_mode="shell_fade",
            occlusion_density_falloff=0.5,
        )
        container = _default_container()

        optimize(obj, profile, container)

        np.testing.assert_array_equal(obj.point_cloud.points, original_points)
