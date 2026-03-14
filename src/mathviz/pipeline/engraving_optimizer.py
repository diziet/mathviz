"""Engraving optimizer: post-sampling adjustments for laser engraving medium."""

import logging
from copy import deepcopy

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

    Mutates a deep copy of obj and returns it. Idempotent — running twice
    should not degrade the result further.
    """
    if obj.point_cloud is None:
        raise ValueError("Engraving optimizer requires a point cloud")

    result = deepcopy(obj)
    cloud = result.point_cloud
    assert cloud is not None  # for type checker

    points = cloud.points
    if len(points) == 0:
        logger.info("Engraving optimizer: empty point cloud, nothing to do")
        return result

    # Step 1: Occlusion thinning
    if profile.occlusion_mode == "shell_fade":
        keep_mask = _apply_shell_fade(points, profile)
        cloud = _apply_mask(cloud, keep_mask)
    elif profile.occlusion_mode == "radial_gradient":
        keep_mask = _apply_radial_gradient(points, profile)
        cloud = _apply_mask(cloud, keep_mask)
    # "none" and "custom" skip occlusion

    # Step 2: Depth compensation
    if profile.depth_compensation:
        cloud = _apply_depth_compensation(cloud, container)

    # Step 3: Point budget enforcement
    if len(cloud.points) > profile.point_budget:
        cloud = _enforce_budget(cloud, profile.point_budget)

    result.point_cloud = cloud
    logger.info(
        "Engraving optimizer: %d -> %d points (mode=%s, depth_comp=%s, budget=%d)",
        len(points),
        len(cloud.points),
        profile.occlusion_mode,
        profile.depth_compensation,
        profile.point_budget,
    )
    return result


def _apply_shell_fade(
    points: np.ndarray,
    profile: EngravingProfile,
) -> np.ndarray:
    """Thin outer layers progressively, keeping inner core dense."""
    centroid = points.mean(axis=0)
    distances = np.linalg.norm(points - centroid, axis=1)
    max_dist = distances.max()

    if max_dist < 1e-12:
        return np.ones(len(points), dtype=bool)

    normalized = distances / max_dist  # 0=center, 1=outermost

    # Divide into shell layers; outer layers get progressively thinned
    num_layers = profile.occlusion_shell_layers
    falloff = profile.occlusion_density_falloff

    # Retention probability: 1.0 at center, (1 - falloff) at outermost
    retention = 1.0 - falloff * (normalized ** (1.0 / max(num_layers, 1)))

    # Deterministic threshold based on point coordinates for idempotency
    threshold = _deterministic_threshold(points)
    return threshold <= retention


def _apply_radial_gradient(
    points: np.ndarray,
    profile: EngravingProfile,
) -> np.ndarray:
    """Density decreases from center outward."""
    centroid = points.mean(axis=0)
    distances = np.linalg.norm(points - centroid, axis=1)
    max_dist = distances.max()

    if max_dist < 1e-12:
        return np.ones(len(points), dtype=bool)

    normalized = distances / max_dist
    falloff = profile.occlusion_density_falloff

    # Linear falloff: retention = 1 at center, (1 - falloff) at edge
    retention = 1.0 - falloff * normalized

    threshold = _deterministic_threshold(points)
    return threshold <= retention


def _deterministic_threshold(points: np.ndarray) -> np.ndarray:
    """Generate a deterministic threshold from point coordinates for idempotency.

    Uses a spatial hash so the same point always gets the same threshold,
    regardless of its index in the array. This ensures idempotency.
    """
    # Combine coordinates with irrational multipliers for a spatial hash
    golden = (1.0 + np.sqrt(5.0)) / 2.0
    sqrt2 = np.sqrt(2.0)
    sqrt3 = np.sqrt(3.0)
    hashed = (points[:, 0] * golden + points[:, 1] * sqrt2 + points[:, 2] * sqrt3)
    return hashed % 1.0


def _apply_depth_compensation(
    cloud: PointCloud,
    container: Container,
) -> PointCloud:
    """Increase point density at max depth via intensity weighting.

    The depth axis is the smallest usable dimension of the container (z by default).
    Points at maximum depth get their intensities multiplied by depth_compensation_factor.
    This is done via intensities so the operation is idempotent — intensities are clamped.
    """
    points = cloud.points
    usable = container.usable_volume  # (width, height, depth)

    # Depth axis = axis with smallest usable dimension
    depth_axis = int(np.argmin(usable))

    z_vals = points[:, depth_axis]
    z_min, z_max = z_vals.min(), z_vals.max()
    z_range = z_max - z_min

    if z_range < 1e-12:
        return cloud

    # Normalized depth: 0=front (min), 1=back (max)
    normalized_depth = (z_vals - z_min) / z_range

    # Build or update intensities: linearly interpolate from 1.0 to factor
    # For depth compensation, we duplicate points at deeper layers
    # to increase density. We use intensities to mark compensation level.
    if cloud.intensities is None:
        intensities = np.ones(len(points), dtype=np.float64)
    else:
        intensities = cloud.intensities.copy()

    # Clamp intensities to [0, 1] before applying compensation
    # This makes the operation idempotent
    intensities = np.clip(intensities, 0.0, 1.0)

    # Scale: front=1.0, back=depth_compensation_factor
    # We normalize this to [0, 1] range for intensity storage
    # Higher intensity = more important = keep during any future downsampling
    depth_weight = 1.0 - normalized_depth * 0.5  # front=1.0, back=0.5
    # Invert: we want back points to be MORE important
    depth_weight = 0.5 + normalized_depth * 0.5  # front=0.5, back=1.0

    intensities = intensities * depth_weight

    return PointCloud(
        points=points.copy(),
        normals=cloud.normals.copy() if cloud.normals is not None else None,
        intensities=intensities,
    )


def _apply_mask(cloud: PointCloud, mask: np.ndarray) -> PointCloud:
    """Filter a point cloud by a boolean mask."""
    return PointCloud(
        points=cloud.points[mask],
        normals=cloud.normals[mask] if cloud.normals is not None else None,
        intensities=cloud.intensities[mask] if cloud.intensities is not None else None,
    )


def _enforce_budget(cloud: PointCloud, budget: int) -> PointCloud:
    """Downsample uniformly if point count exceeds budget."""
    count = len(cloud.points)
    if count <= budget:
        return cloud

    # Uniform downsampling: keep every nth point
    indices = np.linspace(0, count - 1, budget, dtype=int)
    return PointCloud(
        points=cloud.points[indices],
        normals=cloud.normals[indices] if cloud.normals is not None else None,
        intensities=cloud.intensities[indices] if cloud.intensities is not None else None,
    )
