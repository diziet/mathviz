"""Tests for dense sampling caps and defaults.

Verifies that MAX_DENSE_SAMPLES and MAX_RESOLUTION_SCALED_SAMPLES are
5,000,000 and that client-specified max_samples below the cap is respected.
"""

import inspect

import numpy as np
import pytest

from mathviz.core.math_object import CoordSpace, MathObject, Mesh
from mathviz.pipeline.dense_sampling import (
    MAX_DENSE_SAMPLES,
    MAX_RESOLUTION_SCALED_SAMPLES,
    apply_edge_sampling,
    apply_post_transform_sampling,
    apply_resolution_scaled_sampling,
)


def _make_cube_obj() -> MathObject:
    """Create a cube MathObject for testing."""
    vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ], dtype=np.float64)
    faces = np.array([
        [0, 1, 2], [0, 2, 3],
        [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4],
        [2, 3, 7], [2, 7, 6],
        [0, 3, 7], [0, 7, 4],
        [1, 2, 6], [1, 6, 5],
    ], dtype=np.int64)
    mesh = Mesh(vertices=vertices, faces=faces)
    return MathObject(
        mesh=mesh,
        coord_space=CoordSpace.PHYSICAL,
        generator_name="test",
    )


def test_max_dense_samples_is_5_million() -> None:
    """Default cap is 5,000,000 when no max_samples override is provided."""
    assert MAX_DENSE_SAMPLES == 5_000_000


def test_max_resolution_scaled_samples_is_5_million() -> None:
    """Resolution-scaled cap is also 5,000,000."""
    assert MAX_RESOLUTION_SCALED_SAMPLES == 5_000_000


def test_default_parameter_values_match_constants() -> None:
    """Function default kwargs use the module-level constants."""
    post_sig = inspect.signature(apply_post_transform_sampling)
    assert post_sig.parameters["max_samples"].default == MAX_DENSE_SAMPLES

    edge_sig = inspect.signature(apply_edge_sampling)
    assert edge_sig.parameters["max_samples"].default == MAX_DENSE_SAMPLES

    res_sig = inspect.signature(apply_resolution_scaled_sampling)
    assert res_sig.parameters["max_samples"].default == MAX_RESOLUTION_SCALED_SAMPLES


def test_client_specified_max_samples_below_cap_is_respected() -> None:
    """Client-specified max_samples below 5M is respected."""
    obj = _make_cube_obj()
    client_cap = 500
    result = apply_post_transform_sampling(obj, max_samples=client_cap)
    assert len(result.point_cloud.points) <= client_cap


def test_client_edge_sampling_respects_cap() -> None:
    """Edge sampling also respects a client-specified cap."""
    obj = _make_cube_obj()
    client_cap = 100
    result = apply_edge_sampling(obj, max_samples=client_cap)
    assert len(result.point_cloud.points) <= client_cap
