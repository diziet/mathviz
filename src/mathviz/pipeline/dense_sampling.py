"""Post-transform dense sampling: sample mesh surface after physical-space scaling."""

import logging
from dataclasses import replace

import numpy as np
import trimesh

from mathviz.core.math_object import MathObject, PointCloud

logger = logging.getLogger(__name__)

MAX_DENSE_SAMPLES = 200_000
_DENSE_SURFACE_DENSITY = 100.0
_DENSE_SEED = 42
_MIN_SAMPLES = 10


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
