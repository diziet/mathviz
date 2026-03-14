"""Transformer: fits abstract-space geometry into a physical container."""

import logging
from dataclasses import replace

import numpy as np
from scipy.spatial.transform import Rotation

from mathviz.core.container import Container, PlacementPolicy
from mathviz.core.math_object import (
    BoundingBox,
    CoordSpace,
    Curve,
    MathObject,
    Mesh,
    PointCloud,
)

logger = logging.getLogger(__name__)


def fit(
    obj: MathObject,
    container: Container,
    policy: PlacementPolicy,
) -> MathObject:
    """Fit abstract-space geometry into a physical container.

    Args:
        obj: MathObject in ABSTRACT coordinate space.
        container: Physical glass block dimensions.
        policy: Placement policy (anchor, rotation, depth_bias, etc.).

    Returns:
        New MathObject in PHYSICAL coordinate space.

    Raises:
        ValueError: If the input is already in PHYSICAL space.
    """
    if obj.coord_space == CoordSpace.PHYSICAL:
        raise ValueError(
            "Cannot transform: MathObject is already in PHYSICAL coordinate space"
        )

    point_arrays = obj.all_point_arrays()
    if not point_arrays:
        raise ValueError("MathObject has no geometry points to transform")

    rotation = _build_rotation(policy.rotation_degrees)
    has_rotation = not np.allclose(policy.rotation_degrees, (0.0, 0.0, 0.0))

    result = _deep_copy_math_object(obj)

    if has_rotation:
        _apply_rotation(result, rotation)

    bbox_min, bbox_max = _compute_min_max(result)
    bbox_size = bbox_max - bbox_min
    bbox_center = (bbox_min + bbox_max) / 2.0

    usable = np.array(container.usable_volume)
    scale_vec = _compute_scale_vec(bbox_size, usable, policy)

    container_center = np.array([
        container.width_mm / 2.0,
        container.height_mm / 2.0,
        container.depth_mm / 2.0,
    ])
    anchor_point = _compute_anchor_point(
        container_center, usable, bbox_size * scale_vec, policy.anchor
    )

    _apply_scale_and_translate(result, bbox_center, scale_vec, anchor_point)

    offset = np.array(policy.offset_mm)
    if not np.allclose(offset, 0.0):
        _apply_offset(result, offset)

    result.coord_space = CoordSpace.PHYSICAL
    result.bounding_box = _compute_bounding_box(result)

    _warn_if_outside_container(result.bounding_box, container, policy)
    result.validate_or_raise()

    logger.info(
        "Transformed %s: scale_xy=%.4f, scale_z=%.4f, depth_bias=%.2f, anchor=%s",
        obj.generator_name or "object",
        scale_vec[0],
        scale_vec[2],
        policy.depth_bias,
        policy.anchor,
    )
    return result


def _build_rotation(rotation_degrees: tuple[float, float, float]) -> Rotation:
    """Build a scipy Rotation from Euler angles in degrees."""
    return Rotation.from_euler("xyz", rotation_degrees, degrees=True)


def _compute_scale_vec(
    bbox_size: np.ndarray,
    usable: np.ndarray,
    policy: PlacementPolicy,
) -> np.ndarray:
    """Compute the scale vector to fit geometry into usable volume."""
    safe_size = np.where(bbox_size > 1e-12, bbox_size, 1e-12)
    ratios = usable / safe_size

    if policy.scale_override is not None:
        scale_vec = np.full(3, policy.scale_override)
    elif policy.preserve_aspect_ratio:
        scale_vec = np.full(3, float(ratios.min()))
    else:
        scale_vec = ratios.copy()

    scale_vec[2] *= policy.depth_bias
    return scale_vec


def _compute_anchor_point(
    container_center: np.ndarray,
    usable: np.ndarray,
    scaled_size: np.ndarray,
    anchor: str,
) -> np.ndarray:
    """Compute the target center position based on anchor mode.

    Note: when depth_bias > 1.0 with z-axis anchors (front/back), the biased
    size may exceed the usable volume, potentially positioning geometry outside
    the container. This is the intentional trade-off documented in the task spec.
    """
    point = container_center.copy()

    half_usable = usable / 2.0
    half_scaled = scaled_size / 2.0

    if anchor == "bottom":
        point[1] = container_center[1] - half_usable[1] + half_scaled[1]
    elif anchor == "top":
        point[1] = container_center[1] + half_usable[1] - half_scaled[1]
    elif anchor == "left":
        point[0] = container_center[0] - half_usable[0] + half_scaled[0]
    elif anchor == "right":
        point[0] = container_center[0] + half_usable[0] - half_scaled[0]
    elif anchor == "front":
        point[2] = container_center[2] - half_usable[2] + half_scaled[2]
    elif anchor == "back":
        point[2] = container_center[2] + half_usable[2] - half_scaled[2]

    return point


def _deep_copy_math_object(obj: MathObject) -> MathObject:
    """Create a deep copy of a MathObject with copied arrays."""
    result = replace(obj)
    if result.mesh is not None:
        result.mesh = Mesh(
            vertices=result.mesh.vertices.copy(),
            faces=result.mesh.faces.copy(),
            normals=result.mesh.normals.copy() if result.mesh.normals is not None else None,
        )
    if result.point_cloud is not None:
        result.point_cloud = PointCloud(
            points=result.point_cloud.points.copy(),
            normals=(
                result.point_cloud.normals.copy()
                if result.point_cloud.normals is not None
                else None
            ),
            intensities=(
                result.point_cloud.intensities.copy()
                if result.point_cloud.intensities is not None
                else None
            ),
        )
    if result.curves is not None:
        result.curves = [
            Curve(points=c.points.copy(), closed=c.closed) for c in result.curves
        ]
    return result


def _apply_rotation(obj: MathObject, rotation: Rotation) -> None:
    """Apply rotation to all geometry in place."""
    for arr in obj.all_point_arrays():
        arr[:] = rotation.apply(arr)
    if obj.mesh is not None and obj.mesh.normals is not None:
        obj.mesh.normals = rotation.apply(obj.mesh.normals)
    if obj.point_cloud is not None and obj.point_cloud.normals is not None:
        obj.point_cloud.normals = rotation.apply(obj.point_cloud.normals)


def _apply_scale_and_translate(
    obj: MathObject,
    bbox_center: np.ndarray,
    scale_vec: np.ndarray,
    anchor_point: np.ndarray,
) -> None:
    """Apply scale-and-translate to all point arrays in place."""
    for arr in obj.all_point_arrays():
        arr[:] = (arr - bbox_center) * scale_vec + anchor_point


def _apply_offset(obj: MathObject, offset: np.ndarray) -> None:
    """Apply an offset translation to all geometry in place."""
    for arr in obj.all_point_arrays():
        arr += offset


def _compute_min_max(obj: MathObject) -> tuple[np.ndarray, np.ndarray]:
    """Compute global min/max across all geometry without concatenation."""
    arrays = obj.all_point_arrays()
    global_min = arrays[0].min(axis=0).copy()
    global_max = arrays[0].max(axis=0).copy()
    for arr in arrays[1:]:
        np.minimum(global_min, arr.min(axis=0), out=global_min)
        np.maximum(global_max, arr.max(axis=0), out=global_max)
    return global_min, global_max


def _compute_bounding_box(obj: MathObject) -> BoundingBox:
    """Compute the AABB of all geometry on the MathObject."""
    global_min, global_max = _compute_min_max(obj)
    min_corner = tuple(float(v) for v in global_min)
    max_corner = tuple(float(v) for v in global_max)
    return BoundingBox(min_corner=min_corner, max_corner=max_corner)


def _warn_if_outside_container(
    bbox: BoundingBox, container: Container, policy: PlacementPolicy
) -> None:
    """Log a warning if the bounding box exceeds the usable volume."""
    usable = container.usable_volume
    size = bbox.size
    axis_names = ("x", "y", "z")
    for i in range(3):
        if size[i] > usable[i] + 1e-9:
            logger.warning(
                "Geometry exceeds usable %s-axis: %.2fmm > %.2fmm "
                "(depth_bias=%.2f, offset=%s)",
                axis_names[i],
                size[i],
                usable[i],
                policy.depth_bias,
                policy.offset_mm,
            )
