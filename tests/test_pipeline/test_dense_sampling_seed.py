"""Tests for post_transform and resolution_scaled surface sampling: seeding, counts and normals."""

import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pytest
import trimesh

from mathviz.core.container import Container, PlacementPolicy
from mathviz.core.math_object import CoordSpace, MathObject, Mesh, PointCloud
from mathviz.pipeline.dense_sampling import (
    apply_post_transform_sampling,
    apply_resolution_scaled_sampling,
)
from mathviz.pipeline.runner import run

_MODES = ("post_transform", "resolution_scaled")
# A cube of extent 2 has surface area 24. At surface_density 10 it gets 240 surface points.
_CUBE_SURFACE_POINTS = 240
_MAX_SAMPLES = 1000
# post_transform gives 0.3 of max_samples to edge sampling.
_EDGE_POINTS = 300
# Seed for NumPy's global RNG. It differs from 42, the seed the old code wrote into that RNG.
_GLOBAL_RNG_SEED = 2024
_CONCURRENT_SEEDS = tuple(range(8))
_CONCURRENT_ROUNDS = 4
# At surface_density 1000 each call samples 24,000 points, so the threads overlap.
_CONCURRENT_SURFACE_DENSITY = 1000.0
_PIPELINE_MAX_SAMPLES = 5000


def _cube_object(seed: int) -> MathObject:
    """Return an axis-aligned cube mesh of extent 2 centred on the origin, with the given seed."""
    box = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    mesh = Mesh(
        vertices=np.asarray(box.vertices, dtype=np.float64),
        faces=np.asarray(box.faces, dtype=np.int64),
    )
    return MathObject(
        mesh=mesh, coord_space=CoordSpace.PHYSICAL, generator_name="box", seed=seed
    )


def _sample_object(
    obj: MathObject,
    mode: str,
    surface_density: float = 10.0,
    max_samples: int = _MAX_SAMPLES,
) -> MathObject:
    """Apply one sampling mode to obj with a fixed density and cap."""
    if mode == "post_transform":
        return apply_post_transform_sampling(
            obj, max_samples=max_samples, surface_density=surface_density
        )
    return apply_resolution_scaled_sampling(
        obj,
        resolution_kwargs={"voxel_resolution": 128},
        default_resolution={"voxel_resolution": 128},
        max_samples=max_samples,
        base_density=surface_density,
    )


def _sample_cube(seed: int, mode: str, **kwargs: Any) -> PointCloud:
    """Sample a cube mesh with the given seed and mode, and return the cloud."""
    result = _sample_object(_cube_object(seed), mode, **kwargs)
    assert result.point_cloud is not None, f"{mode} sampling must set point_cloud"
    return result.point_cloud


@pytest.fixture
def restore_global_rng() -> Iterator[None]:
    """Restore NumPy's global RNG state after a test that seeds it."""
    state = np.random.get_state()
    yield
    np.random.set_state(state)


@pytest.mark.parametrize("mode", _MODES)
def test_same_seed_gives_identical_points_and_normals(mode: str) -> None:
    """Two samplings of one mesh with the same seed return equal points and normals."""
    first = _sample_cube(7, mode)
    second = _sample_cube(7, mode)
    np.testing.assert_array_equal(first.points, second.points)
    np.testing.assert_array_equal(first.normals, second.normals)


@pytest.mark.parametrize("mode", _MODES)
def test_different_seeds_give_different_surface_points_on_same_mesh(mode: str) -> None:
    """Seeds 1 and 2 on one mesh return the same point count at different surface positions."""
    first = _sample_cube(1, mode)
    second = _sample_cube(2, mode)
    assert first.points.shape == second.points.shape
    assert not np.array_equal(
        first.points[:_CUBE_SURFACE_POINTS], second.points[:_CUBE_SURFACE_POINTS]
    )


def test_post_transform_edge_points_do_not_depend_on_seed() -> None:
    """post_transform places the same edge points for seeds 1 and 2."""
    first = _sample_cube(1, "post_transform")
    second = _sample_cube(2, "post_transform")
    np.testing.assert_array_equal(
        first.points[_CUBE_SURFACE_POINTS:], second.points[_CUBE_SURFACE_POINTS:]
    )


@pytest.mark.parametrize("mode", _MODES)
def test_negative_seed_samples_deterministically(mode: str) -> None:
    """A negative MathObject.seed samples without error and repeats for the same seed."""
    first = _sample_cube(-3, mode)
    second = _sample_cube(-3, mode)
    assert len(first.points) > 0
    np.testing.assert_array_equal(first.points, second.points)


@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.usefixtures("restore_global_rng")
def test_sampling_leaves_global_numpy_rng_unchanged(mode: str) -> None:
    """Surface sampling neither seeds nor draws from NumPy's global RNG."""
    np.random.seed(_GLOBAL_RNG_SEED)
    expected = np.random.RandomState(_GLOBAL_RNG_SEED).random_sample(4)
    _sample_cube(7, mode)
    np.testing.assert_array_equal(np.random.random_sample(4), expected)


@pytest.mark.parametrize("mode", _MODES)
def test_concurrent_calls_match_serial_results_per_seed(mode: str) -> None:
    """Threads sampling with different seeds at once each get the serial cloud for their seed."""
    kwargs = {"surface_density": _CONCURRENT_SURFACE_DENSITY, "max_samples": 50_000}
    expected = {seed: _sample_cube(seed, mode, **kwargs).points for seed in _CONCURRENT_SEEDS}
    tasks = [seed for _ in range(_CONCURRENT_ROUNDS) for seed in _CONCURRENT_SEEDS]
    barrier = threading.Barrier(len(_CONCURRENT_SEEDS))

    def sample_after_barrier(seed: int) -> np.ndarray:
        barrier.wait()
        return _sample_cube(seed, mode, **kwargs).points

    with ThreadPoolExecutor(max_workers=len(_CONCURRENT_SEEDS)) as pool:
        results = list(pool.map(sample_after_barrier, tasks))

    for seed, points in zip(tasks, results, strict=True):
        np.testing.assert_array_equal(points, expected[seed], err_msg=f"seed {seed}")


def test_post_transform_counts_surface_then_edge_points() -> None:
    """A cube of area 24 at density 10 with cap 1000 gives 240 surface and 300 edge points."""
    cloud = _sample_cube(7, "post_transform")
    assert cloud.points.shape == (_CUBE_SURFACE_POINTS + _EDGE_POINTS, 3)
    assert cloud.normals.shape == cloud.points.shape
    assert cloud.points.dtype == np.float64
    assert np.isfinite(cloud.normals[:_CUBE_SURFACE_POINTS]).all()
    assert np.isnan(cloud.normals[_CUBE_SURFACE_POINTS:]).all()
    np.testing.assert_allclose(
        np.max(np.abs(cloud.points[_CUBE_SURFACE_POINTS:]), axis=1), 1.0, atol=1e-12
    )


def test_post_transform_caps_surface_points_at_surface_budget() -> None:
    """At density 100 a cube of area 24 asks for 2400 points; cap 1000 keeps 700 surface + 300."""
    cloud = _sample_cube(7, "post_transform", surface_density=100.0)
    assert cloud.points.shape == (_MAX_SAMPLES, 3)
    assert np.isfinite(cloud.normals[: _MAX_SAMPLES - _EDGE_POINTS]).all()
    assert np.isnan(cloud.normals[_MAX_SAMPLES - _EDGE_POINTS :]).all()


def test_resolution_scaled_count_is_area_times_density() -> None:
    """resolution_scaled at the default resolution gives area 24 × density 10 = 240 points."""
    cloud = _sample_cube(7, "resolution_scaled")
    assert cloud.points.shape == (_CUBE_SURFACE_POINTS, 3)
    assert cloud.normals.shape == (_CUBE_SURFACE_POINTS, 3)
    assert cloud.points.dtype == np.float64


def test_resolution_scaled_caps_count_at_max_samples() -> None:
    """At density 100 a cube of area 24 asks for 2400 points; cap 1000 keeps 1000."""
    cloud = _sample_cube(7, "resolution_scaled", surface_density=100.0)
    assert cloud.points.shape == (_MAX_SAMPLES, 3)


@pytest.mark.parametrize("mode", _MODES)
def test_surface_normals_are_outward_face_normals(mode: str) -> None:
    """Each surface point on the cube gets the unit normal of the cube face it lies on."""
    cloud = _sample_cube(7, mode)
    points = cloud.points[:_CUBE_SURFACE_POINTS]
    axis = np.argmax(np.abs(points), axis=1)
    rows = np.arange(len(points))
    expected = np.zeros_like(points)
    expected[rows, axis] = np.sign(points[rows, axis])
    assert cloud.normals.dtype == np.float64
    np.testing.assert_allclose(cloud.normals[:_CUBE_SURFACE_POINTS], expected, atol=1e-12)


@pytest.mark.parametrize("mode", _MODES)
def test_sampling_keeps_mesh_and_seed(mode: str) -> None:
    """Sampling returns an object with the input mesh and seed."""
    obj = _cube_object(7)
    result = _sample_object(obj, mode)
    assert result.mesh is obj.mesh
    assert result.seed == 7


def _run_torus(seed: int, mode: str) -> PointCloud:
    """Run the torus pipeline with the given seed and sampling mode, and return the cloud."""
    result = run(
        "torus",
        seed=seed,
        container=Container.with_uniform_margin(),
        placement=PlacementPolicy(),
        sampling_mode=mode,
        max_samples=_PIPELINE_MAX_SAMPLES,
    )
    assert result.math_object.point_cloud is not None, f"{mode} run must set point_cloud"
    return result.math_object.point_cloud


@pytest.mark.parametrize("mode", _MODES)
def test_pipeline_cloud_depends_only_on_seed(mode: str) -> None:
    """Two torus runs with seed 7 give equal clouds; seed 8 gives the same count elsewhere."""
    first = _run_torus(7, mode)
    second = _run_torus(7, mode)
    other = _run_torus(8, mode)
    assert first.points.shape == (_PIPELINE_MAX_SAMPLES, 3)
    np.testing.assert_array_equal(first.points, second.points)
    np.testing.assert_array_equal(first.normals, second.normals)
    assert other.points.shape == first.points.shape
    assert not np.array_equal(first.points, other.points)
