"""Tests for SPARSE_SHELL surface sampling: seeding, sample count, normals and passthrough."""

from collections.abc import Iterator

import numpy as np
import pytest
import trimesh

from mathviz.core.container import Container, PlacementPolicy
from mathviz.core.math_object import MathObject, Mesh, PointCloud
from mathviz.core.representation import RepresentationConfig, RepresentationType
from mathviz.pipeline.representation_strategy import apply
from mathviz.pipeline.runner import run

_SPARSE_SHELL = RepresentationConfig(type=RepresentationType.SPARSE_SHELL)
# Seed for NumPy's global RNG. It differs from 42, the seed the old code wrote into that RNG.
_GLOBAL_RNG_SEED = 2024
_PIPELINE_VOXEL_RESOLUTION = 32


def _box_object(seed: int, extent: float = 2.0) -> MathObject:
    """Return an axis-aligned cube mesh centred on the origin, with the given seed."""
    box = trimesh.creation.box(extents=(extent, extent, extent))
    mesh = Mesh(
        vertices=np.asarray(box.vertices, dtype=np.float64),
        faces=np.asarray(box.faces, dtype=np.int64),
    )
    return MathObject(mesh=mesh, generator_name="box", seed=seed)


def _sample_box(
    seed: int, config: RepresentationConfig = _SPARSE_SHELL, extent: float = 2.0
) -> PointCloud:
    """Apply SPARSE_SHELL to a cube mesh and return the sampled cloud."""
    result = apply(_box_object(seed, extent), config)
    assert result.point_cloud is not None, "SPARSE_SHELL on a mesh must set point_cloud"
    return result.point_cloud


@pytest.fixture
def restore_global_rng() -> Iterator[None]:
    """Restore NumPy's global RNG state after a test that seeds it."""
    state = np.random.get_state()
    yield
    np.random.set_state(state)


def test_same_seed_gives_identical_points_and_normals() -> None:
    """Two samplings of one mesh with the same seed return equal points and normals."""
    first = _sample_box(seed=7)
    second = _sample_box(seed=7)
    np.testing.assert_array_equal(first.points, second.points)
    np.testing.assert_array_equal(first.normals, second.normals)


def test_different_seeds_give_different_points_on_same_mesh() -> None:
    """Seeds 1 and 2 on one mesh return the same point count at different positions."""
    first = _sample_box(seed=1)
    second = _sample_box(seed=2)
    assert len(first.points) == len(second.points)
    assert not np.array_equal(first.points, second.points)


def test_negative_seed_samples_deterministically() -> None:
    """A negative MathObject.seed samples without error and repeats for the same seed."""
    first = _sample_box(seed=-3)
    second = _sample_box(seed=-3)
    np.testing.assert_array_equal(first.points, second.points)


@pytest.mark.usefixtures("restore_global_rng")
def test_sampling_leaves_global_numpy_rng_unchanged() -> None:
    """SPARSE_SHELL neither seeds nor draws from NumPy's global RNG."""
    np.random.seed(_GLOBAL_RNG_SEED)
    expected = np.random.RandomState(_GLOBAL_RNG_SEED).random_sample(4)
    _sample_box(seed=7)
    np.testing.assert_array_equal(np.random.random_sample(4), expected)


def test_sample_count_is_surface_area_times_density() -> None:
    """A cube of area 24 at surface_density 10 gives 240 points."""
    config = RepresentationConfig(
        type=RepresentationType.SPARSE_SHELL, surface_density=10.0
    )
    cloud = _sample_box(seed=7, config=config)
    assert cloud.points.shape == (240, 3)
    assert cloud.normals.shape == (240, 3)


def test_sample_count_has_minimum_of_ten() -> None:
    """A mesh whose area times density is below 10 still gives 10 points."""
    cloud = _sample_box(seed=7, extent=0.01)
    assert cloud.points.shape == (10, 3)


def test_normals_are_outward_face_normals_of_sampled_points() -> None:
    """Each point on the cube gets the unit normal of the cube face it lies on."""
    cloud = _sample_box(seed=7)
    axis = np.argmax(np.abs(cloud.points), axis=1)
    rows = np.arange(len(cloud.points))
    expected = np.zeros_like(cloud.points)
    expected[rows, axis] = np.sign(cloud.points[rows, axis])
    assert cloud.normals.dtype == np.float64
    np.testing.assert_allclose(cloud.normals, expected, atol=1e-12)


def test_mesh_input_keeps_mesh_and_sets_sparse_shell_representation() -> None:
    """SPARSE_SHELL on a mesh keeps the mesh and sets representation to sparse_shell."""
    obj = _box_object(seed=7)
    result = apply(obj, _SPARSE_SHELL)
    assert result.mesh is obj.mesh
    assert result.representation == "sparse_shell"


def test_point_cloud_only_input_passes_through_unchanged() -> None:
    """SPARSE_SHELL returns a point-cloud-only input's points and normals unchanged."""
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    normals = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float64)
    obj = MathObject(point_cloud=PointCloud(points=points, normals=normals), seed=7)
    result = apply(obj, _SPARSE_SHELL)
    assert result.mesh is None
    assert result.representation == "sparse_shell"
    np.testing.assert_array_equal(result.point_cloud.points, points)
    np.testing.assert_array_equal(result.point_cloud.normals, normals)


@pytest.mark.parametrize("generator_name", ["julia3d", "mandelbulb"])
def test_pipeline_same_seed_gives_identical_cloud(generator_name: str) -> None:
    """Two pipeline runs with the same seed give equal sparse_shell clouds."""
    clouds = [
        run(
            generator_name,
            seed=42,
            resolution_kwargs={"voxel_resolution": _PIPELINE_VOXEL_RESOLUTION},
            container=Container.with_uniform_margin(),
            placement=PlacementPolicy(),
        ).math_object.point_cloud
        for _ in range(2)
    ]
    np.testing.assert_array_equal(clouds[0].points, clouds[1].points)
    np.testing.assert_array_equal(clouds[0].normals, clouds[1].normals)
