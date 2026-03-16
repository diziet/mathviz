"""Post-transform dense sampling: sample mesh surface after physical-space scaling."""

import logging
import threading
from dataclasses import replace
from typing import Any

import numpy as np
import trimesh

from mathviz.core.math_object import MathObject, PointCloud

logger = logging.getLogger(__name__)

MAX_DENSE_SAMPLES = 200_000
MAX_RESOLUTION_SCALED_SAMPLES = 500_000
_DENSE_SURFACE_DENSITY = 100.0
_DENSE_SEED = 42
_MIN_SAMPLES = 10

# trimesh.sample relies on numpy's legacy global RNG.  Protect the
# seed-then-sample sequence so concurrent threads don't interleave.
_rng_lock = threading.Lock()


def apply_post_transform_sampling(
    obj: MathObject,
    *,
    max_samples: int = MAX_DENSE_SAMPLES,
    surface_density: float = _DENSE_SURFACE_DENSITY,
) -> MathObject:
    """Sample the mesh surface in physical space, capping at max_samples.

    This produces a much denser cloud than pre-transform sampling because
    the mesh area in physical space (mm^2) is typically much larger than
    in abstract space (unit^2).
    """
    if obj.mesh is None:
        raise ValueError("Post-transform sampling requires a mesh")

    tm = trimesh.Trimesh(
        vertices=obj.mesh.vertices,
        faces=obj.mesh.faces,
        process=False,
    )

    total_area = float(tm.area)
    sample_count = max(_MIN_SAMPLES, int(total_area * surface_density))
    sample_count = min(sample_count, max_samples)

    with _rng_lock:
        np.random.seed(_DENSE_SEED)
        points, face_indices = tm.sample(sample_count, return_index=True)
    normals = tm.face_normals[face_indices]

    cloud = PointCloud(
        points=np.asarray(points, dtype=np.float64),
        normals=np.asarray(normals, dtype=np.float64),
    )

    logger.info(
        "Dense sampling: %d points from %.1f mm² surface (cap=%d)",
        len(cloud.points),
        total_area,
        max_samples,
    )

    return replace(obj, point_cloud=cloud)


def _compute_resolution_scale(
    resolution_kwargs: dict[str, Any],
    default_resolution: dict[str, Any],
) -> float:
    """Compute the squared ratio of requested to default resolution.

    Finds the first matching numeric resolution parameter between the two
    dicts and returns (requested / default)².  Returns 1.0 when no match
    is found or when the default is zero.
    """
    for key, default_val in default_resolution.items():
        if key not in resolution_kwargs:
            continue
        requested_val = resolution_kwargs[key]
        if not isinstance(default_val, (int, float)) or default_val <= 0:
            continue
        if not isinstance(requested_val, (int, float)) or requested_val <= 0:
            continue
        ratio = float(requested_val) / float(default_val)
        return ratio * ratio
    return 1.0


def apply_resolution_scaled_sampling(
    obj: MathObject,
    *,
    resolution_kwargs: dict[str, Any],
    default_resolution: dict[str, Any],
    max_samples: int = MAX_RESOLUTION_SCALED_SAMPLES,
    base_density: float = _DENSE_SURFACE_DENSITY,
) -> MathObject:
    """Sample the mesh surface with density scaled by resolution ratio.

    Multiplies the base surface density by (resolution / default)² so that
    higher-resolution meshes produce proportionally denser point clouds.
    """
    if obj.mesh is None:
        raise ValueError("Resolution-scaled sampling requires a mesh")

    scale = _compute_resolution_scale(resolution_kwargs, default_resolution)
    surface_density = base_density * scale

    tm = trimesh.Trimesh(
        vertices=obj.mesh.vertices,
        faces=obj.mesh.faces,
        process=False,
    )

    total_area = float(tm.area)
    sample_count = max(_MIN_SAMPLES, int(total_area * surface_density))
    sample_count = min(sample_count, max_samples)

    with _rng_lock:
        np.random.seed(_DENSE_SEED)
        points, face_indices = tm.sample(sample_count, return_index=True)
    normals = tm.face_normals[face_indices]

    cloud = PointCloud(
        points=np.asarray(points, dtype=np.float64),
        normals=np.asarray(normals, dtype=np.float64),
    )

    logger.info(
        "Resolution-scaled sampling: %d points from %.1f mm² surface "
        "(density=%.1f, scale=%.2fx, cap=%d)",
        len(cloud.points),
        total_area,
        surface_density,
        scale,
        max_samples,
    )

    return replace(obj, point_cloud=cloud)
