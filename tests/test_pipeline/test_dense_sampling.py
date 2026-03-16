"""Tests for dense sampling caps and defaults.

Verifies that MAX_DENSE_SAMPLES and MAX_RESOLUTION_SCALED_SAMPLES are
5,000,000 hard ceilings, DEFAULT_DENSE_SAMPLES is 500,000, function
defaults use the default constant, and enforcement clamps values above
the ceiling.
"""

import inspect

import numpy as np

from mathviz.core.math_object import CoordSpace, MathObject, Mesh
from mathviz.pipeline.dense_sampling import (
    DEFAULT_DENSE_SAMPLES,
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


def test_hard_caps_are_5_million() -> None:
    """Hard ceiling constants are 5,000,000."""
    assert MAX_DENSE_SAMPLES == 5_000_000
    assert MAX_RESOLUTION_SCALED_SAMPLES == 5_000_000


def test_default_is_500k() -> None:
    """Comfortable default is 500,000."""
    assert DEFAULT_DENSE_SAMPLES == 500_000


def test_function_defaults_use_default_constant() -> None:
    """Function default kwargs use DEFAULT_DENSE_SAMPLES, not the hard cap."""
    post_sig = inspect.signature(apply_post_transform_sampling)
    assert post_sig.parameters["max_samples"].default == DEFAULT_DENSE_SAMPLES

    edge_sig = inspect.signature(apply_edge_sampling)
    assert edge_sig.parameters["max_samples"].default == DEFAULT_DENSE_SAMPLES

    res_sig = inspect.signature(apply_resolution_scaled_sampling)
    assert res_sig.parameters["max_samples"].default == DEFAULT_DENSE_SAMPLES


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


def test_hard_cap_enforced_on_post_transform() -> None:
    """Passing max_samples above the hard cap gets clamped."""
    obj = _make_cube_obj()
    over_cap = MAX_DENSE_SAMPLES + 1_000_000
    result = apply_post_transform_sampling(obj, max_samples=over_cap)
    assert len(result.point_cloud.points) <= MAX_DENSE_SAMPLES


def test_hard_cap_enforced_on_edge_sampling() -> None:
    """Edge sampling clamps values above the hard cap."""
    obj = _make_cube_obj()
    over_cap = MAX_DENSE_SAMPLES + 1_000_000
    result = apply_edge_sampling(obj, max_samples=over_cap)
    assert len(result.point_cloud.points) <= MAX_DENSE_SAMPLES


def test_hard_cap_enforced_on_resolution_scaled() -> None:
    """Resolution-scaled sampling clamps values above the hard cap."""
    obj = _make_cube_obj()
    over_cap = MAX_RESOLUTION_SCALED_SAMPLES + 1_000_000
    result = apply_resolution_scaled_sampling(
        obj,
        resolution_kwargs={"res": 10},
        default_resolution={"res": 10},
        max_samples=over_cap,
    )
    assert len(result.point_cloud.points) <= MAX_RESOLUTION_SCALED_SAMPLES
