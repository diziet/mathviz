"""Engraving optimizer: post-sampling adjustments for laser engraving medium."""

import copy
import logging
from typing import Callable

import numpy as np

from mathviz.core.container import Container
from mathviz.core.engraving import EngravingProfile
from mathviz.core.math_object import MathObject, PointCloud

logger = logging.getLogger(__name__)


def optimize(
    obj: MathObject,
    profile: EngravingProfile,
    container: Container,
) -> MathObject:
    """Apply engraving optimizations to a MathObject's point cloud.

    Returns a new MathObject (original is not mutated). Idempotent — running
    twice produces the same result.
    """
    if obj.point_cloud is None:
        raise ValueError("Engraving optimizer requires a point cloud")

    cloud = obj.point_cloud
    original_count = len(cloud.points)

    if original_count == 0:
        logger.info("Engraving optimizer: empty point cloud, nothing to do")
        return _shallow_copy_with_cloud(obj, cloud)

    # Step 1: Occlusion thinning
    if profile.occlusion_mode == "shell_fade":
        mask = _compute_occlusion_mask(cloud.points, profile, _shell_fade_retention)
        cloud = _subset_cloud(cloud, mask)
    elif profile.occlusion_mode == "radial_gradient":
        mask = _compute_occlusion_mask(cloud.points, profile, _radial_retention)
        cloud = _subset_cloud(cloud, mask)

    # Step 2: Depth compensation
    if profile.depth_compensation:
        cloud = _apply_depth_compensation(cloud, profile, container)

    # Step 3: Point budget enforcement
    if len(cloud.points) > profile.point_budget:
        cloud = _enforce_budget(cloud, profile.point_budget)

    logger.info(
        "Engraving optimizer: %d -> %d points (mode=%s, depth_comp=%s, budget=%d)",
        original_count,
        len(cloud.points),
        profile.occlusion_mode,
        profile.depth_compensation,
        profile.point_budget,
    )
    return _shallow_copy_with_cloud(obj, cloud)


def _shallow_copy_with_cloud(obj: MathObject, cloud: PointCloud) -> MathObject:
    """Create a shallow copy of obj with a new point cloud."""
    result = copy.copy(obj)
    result.point_cloud = cloud
    return result


# --- Occlusion ---

RetentionFn = Callable[[np.ndarray, EngravingProfile], np.ndarray]


def _compute_occlusion_mask(
    points: np.ndarray,
    profile: EngravingProfile,
    retention_fn: RetentionFn,
) -> np.ndarray:
    """Compute a boolean keep-mask using a retention function."""
    centroid = points.mean(axis=0)
    distances = np.linalg.norm(points - centroid, axis=1)
    max_dist = distances.max()

    if max_dist < 1e-12:
        return np.ones(len(points), dtype=bool)

    normalized = distances / max_dist
    retention = retention_fn(normalized, profile)
    threshold = _deterministic_threshold(points)
    return threshold <= retention


def _shell_fade_retention(normalized: np.ndarray, profile: EngravingProfile) -> np.ndarray:
    """Retention curve for shell_fade: progressive thinning of outer layers."""
    exponent = 1.0 / max(profile.occlusion_shell_layers, 1)
    return 1.0 - profile.occlusion_density_falloff * (normalized ** exponent)


def _radial_retention(normalized: np.ndarray, profile: EngravingProfile) -> np.ndarray:
    """Retention curve for radial_gradient: linear density decrease outward."""
    return 1.0 - profile.occlusion_density_falloff * normalized


def _deterministic_threshold(points: np.ndarray) -> np.ndarray:
    """Generate a deterministic threshold from point coordinates for idempotency.

    Uses a spatial hash so the same point always gets the same threshold,
    regardless of its index in the array.
    """
    golden = (1.0 + np.sqrt(5.0)) / 2.0
    sqrt2 = np.sqrt(2.0)
    sqrt3 = np.sqrt(3.0)
    hashed = points[:, 0] * golden + points[:, 1] * sqrt2 + points[:, 2] * sqrt3
    return hashed % 1.0


# --- Depth compensation ---


def _apply_depth_compensation(
    cloud: PointCloud,
    profile: EngravingProfile,
    container: Container,
) -> PointCloud:
    """Apply depth-dependent intensity weighting.

    Linearly interpolates intensity from 1.0 at front to a value derived from
    depth_compensation_factor at back. Idempotent: intensities are written
    directly from depth position, not accumulated.
    """
    points = cloud.points
    usable = container.usable_volume

    # Depth axis = axis with smallest usable dimension
    depth_axis = int(np.argmin(usable))

    z_vals = points[:, depth_axis]
    z_min, z_max = z_vals.min(), z_vals.max()
    z_range = z_max - z_min

    if z_range < 1e-12:
        return cloud

    # Normalized depth: 0=front (min), 1=back (max)
    normalized_depth = (z_vals - z_min) / z_range

    # Intensity: linearly interpolate from 1/factor at front to 1.0 at back
    # This ensures back points are weighted more heavily. The factor controls
    # the ratio between back and front intensity.
    factor = profile.depth_compensation_factor
    front_weight = 1.0 / factor  # e.g. factor=1.5 -> front=0.667
    intensities = front_weight + normalized_depth * (1.0 - front_weight)

    return PointCloud(
        points=points,
        normals=cloud.normals,
        intensities=intensities,
    )


# --- Shared subsetting ---


def _subset_cloud(cloud: PointCloud, idx: np.ndarray) -> PointCloud:
    """Subset a point cloud by boolean mask or integer indices."""
    return PointCloud(
        points=cloud.points[idx],
        normals=cloud.normals[idx] if cloud.normals is not None else None,
        intensities=cloud.intensities[idx] if cloud.intensities is not None else None,
    )


def _enforce_budget(cloud: PointCloud, budget: int) -> PointCloud:
    """Downsample to budget, preferring points with higher intensities."""
    count = len(cloud.points)
    if count <= budget:
        return cloud

    if cloud.intensities is not None:
        # Keep the most important points (highest intensity)
        indices = np.argsort(cloud.intensities)[::-1][:budget]
        indices.sort()  # preserve spatial ordering
    else:
        # Uniform downsampling when no intensities
        indices = np.linspace(0, count - 1, budget, dtype=int)

    return _subset_cloud(cloud, indices)
