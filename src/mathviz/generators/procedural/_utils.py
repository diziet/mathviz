"""Shared utilities for procedural generators."""

import numpy as np

from mathviz.core.math_object import BoundingBox


def compute_heightmap_bounding_box(field: np.ndarray) -> BoundingBox:
    """Compute bounding box for a 2D heightmap scalar field.

    Assumes x,y span [0, 1] and z spans the field value range.
    Returns the bounding box and (z_min, z_max) for reuse in logging.
    """
    z_min = float(np.min(field))
    z_max = float(np.max(field))
    bbox = BoundingBox(
        min_corner=(0.0, 0.0, z_min),
        max_corner=(1.0, 1.0, z_max),
    )
    return bbox
