"""Tests that tube_thickening's Bishop frames match the per-step NumPy reference.

The reference below is the frame code tube_thickening used before its transport loop moved to
Python floats. Both do the same float operations in the same order, so on the development
machine they agree bit for bit. FRAME_ATOL leaves room for NumPy's dot or trigonometric
functions to round one ULP differently from the math module on another platform. A semantic
change, such as another rotation axis, sign, branch or initial normal, moves a component of a
unit frame vector by far more than FRAME_ATOL.
"""

import warnings

import numpy as np
import pytest

from mathviz.core.container import Container, PlacementPolicy
from mathviz.core.math_object import Curve
from mathviz.pipeline.runner import PipelineResult, run
from mathviz.shared import tube_thickening
from mathviz.shared.tube_thickening import EPSILON, thicken_curve

FRAME_ATOL = 1e-10
LORENZ_STEPS = 1050
LORENZ_TRANSIENT = 200


def _reference_initial_normal(tangent: np.ndarray) -> np.ndarray:
    """Cross the tangent with the least parallel coordinate axis."""
    candidates = np.eye(3)
    least_parallel = candidates[np.argmin(np.abs(candidates @ tangent))]
    normal = np.cross(tangent, least_parallel)
    return normal / np.linalg.norm(normal)


def _reference_rotate(vec: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues' rotation formula on NumPy 3-vectors."""
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    return vec * cos_a + np.cross(axis, vec) * sin_a + axis * np.dot(axis, vec) * (1.0 - cos_a)


def _reference_transport(
    prev_normal: np.ndarray, prev_tangent: np.ndarray, curr_tangent: np.ndarray
) -> np.ndarray:
    """Transport one normal with per-step NumPy calls."""
    axis = np.cross(prev_tangent, curr_tangent)
    axis_len = np.linalg.norm(axis)
    cos_angle = np.clip(np.dot(prev_tangent, curr_tangent), -1.0, 1.0)
    if axis_len < EPSILON:
        if cos_angle >= -0.99:
            return prev_normal.copy()
        perp = _reference_initial_normal(prev_tangent)
        rotated = _reference_rotate(prev_normal, perp, np.pi)
    else:
        rotated = _reference_rotate(prev_normal, axis / axis_len, np.arccos(cos_angle))
    rotated -= np.dot(rotated, curr_tangent) * curr_tangent
    norm = np.linalg.norm(rotated)
    if norm < EPSILON:
        return _reference_initial_normal(curr_tangent)
    return rotated / norm


def _reference_close(normals: np.ndarray, tangents: np.ndarray) -> np.ndarray:
    """Rotate each normal by its share of the closing mismatch angle."""
    transported = _reference_transport(normals[-1], tangents[-1], tangents[0])
    correction = np.arccos(np.clip(np.dot(transported, normals[0]), -1.0, 1.0))
    if np.dot(np.cross(transported, normals[0]), tangents[0]) < 0:
        correction = -correction
    angles = np.arange(len(normals), dtype=np.float64) / len(normals) * correction
    cos_a = np.cos(angles)[:, np.newaxis]
    sin_a = np.sin(angles)[:, np.newaxis]
    dot_at = np.sum(tangents * normals, axis=1, keepdims=True)
    return normals * cos_a + np.cross(tangents, normals) * sin_a + tangents * dot_at * (1.0 - cos_a)


def _reference_frames(tangents: np.ndarray, closed: bool) -> tuple[np.ndarray, np.ndarray]:
    """Build Bishop frames with the per-step NumPy loop."""
    normals = np.empty_like(tangents)
    normals[0] = _reference_initial_normal(tangents[0])
    for i in range(1, len(tangents)):
        normals[i] = _reference_transport(normals[i - 1], tangents[i - 1], tangents[i])
    if closed:
        normals = _reference_close(normals, tangents)
    return normals, np.cross(tangents, normals)


def _tangents(points: list[list[float]] | np.ndarray, closed: bool) -> np.ndarray:
    """Return the unit tangents thicken_curve computes for these points."""
    return tube_thickening._compute_tangents(np.asarray(points, dtype=np.float64), closed)


def _helix_points(n: int = 400) -> np.ndarray:
    """Return points on an open helix with five turns."""
    t = np.linspace(0, 10 * np.pi, n)
    return np.column_stack([np.cos(t), np.sin(t), 0.3 * t])


def _trefoil_points(n: int = 300) -> np.ndarray:
    """Return points on a closed trefoil knot."""
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack(
        [np.sin(t) + 2 * np.sin(2 * t), np.cos(t) - 2 * np.cos(2 * t), -np.sin(3 * t)]
    )


def _nearly_parallel_tangents() -> np.ndarray:
    """Return unit tangents whose turn angles run from 0 to 1e-6, crossing EPSILON."""
    turn_angles = np.repeat([0.0, 1e-14, 5e-13, 1e-12, 2e-12, 1e-11, 1e-9, 1e-6], 4)
    angles = np.cumsum(turn_angles)
    tangents = np.column_stack([np.cos(angles), np.sin(angles), np.full_like(angles, 0.3)])
    return tangents / np.linalg.norm(tangents, axis=1, keepdims=True)


def _random_walk_points(seed: int, n: int = 2000) -> np.ndarray:
    """Return a seeded random walk, a stand-in for an attractor trajectory."""
    return np.cumsum(np.random.default_rng(seed).normal(size=(n, 3)), axis=0)


FRAME_CASES = {
    "straight_line": (_tangents(np.column_stack([np.linspace(0, 1, 20), np.zeros(20),
                                                 np.zeros(20)]), False), False),
    "helix": (_tangents(_helix_points(), False), False),
    "nearly_parallel": (_nearly_parallel_tangents(), False),
    # Tangents +x, +x, -x, -x, -x: the second step takes the anti-parallel branch.
    "hairpin": (_tangents([[0, 0, 0], [2, 0, 0], [1, 0, 0], [-1, 0, 0], [-3, 0, 0]], False),
                False),
    # A zero tangent carries the +z normal onto a +z tangent, so the next step falls back to
    # a fresh perpendicular.
    "zero_tangent_fallback": (_tangents([[0, 0, 0], [1, 0, 0], [0, 0, 0], [1, 0, 1],
                                         [0.1, 0, 2]], False), False),
    "random_walk_open": (_tangents(_random_walk_points(seed=7), False), False),
    "circle_closed": (_tangents(np.column_stack([np.cos(np.linspace(0, 2 * np.pi, 64, False)),
                                                 np.sin(np.linspace(0, 2 * np.pi, 64, False)),
                                                 np.zeros(64)]), True), True),
    "trefoil_closed": (_tangents(_trefoil_points(), True), True),
    "random_walk_closed": (_tangents(_random_walk_points(seed=11), True), True),
}


@pytest.mark.parametrize("case", sorted(FRAME_CASES))
def test_bishop_frames_match_per_step_numpy_reference(case: str) -> None:
    """Normals and binormals equal the reference within FRAME_ATOL."""
    tangents, closed = FRAME_CASES[case]
    normals, binormals = tube_thickening._compute_bishop_frames(tangents, closed)
    ref_normals, ref_binormals = _reference_frames(tangents, closed)
    assert normals.shape == ref_normals.shape == tangents.shape
    np.testing.assert_allclose(normals, ref_normals, rtol=0, atol=FRAME_ATOL)
    np.testing.assert_allclose(binormals, ref_binormals, rtol=0, atol=FRAME_ATOL)


@pytest.mark.parametrize("case", ["helix", "hairpin", "random_walk_open", "trefoil_closed"])
def test_bishop_frames_are_orthonormal(case: str) -> None:
    """Each normal is a unit vector perpendicular to its tangent, and binormal = t x n."""
    tangents, closed = FRAME_CASES[case]
    normals, binormals = tube_thickening._compute_bishop_frames(tangents, closed)
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(np.sum(normals * tangents, axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(binormals, np.cross(tangents, normals), atol=0)


def test_thicken_curve_repeats_bit_for_bit() -> None:
    """Two calls on the same random-walk curve return identical vertices and faces."""
    curve = Curve(points=_random_walk_points(seed=3), closed=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        first = thicken_curve(curve, radius=0.05, sides=8)
        second = thicken_curve(curve, radius=0.05, sides=8)
    np.testing.assert_array_equal(first.vertices, second.vertices)
    np.testing.assert_array_equal(first.faces, second.faces)


def _run_lorenz_tube(seed: int) -> PipelineResult:
    """Run lorenz through the pipeline with its default tube representation."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return run(
            "lorenz",
            seed=seed,
            resolution_kwargs={"integration_steps": LORENZ_STEPS},
            params={"transient_steps": LORENZ_TRANSIENT},
            container=Container(),
            placement=PlacementPolicy(),
        )


def test_lorenz_pipeline_mesh_matches_reference_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    """The lorenz tube mesh and validation report match a run with the reference frames."""
    result = _run_lorenz_tube(seed=42)
    monkeypatch.setattr(tube_thickening, "_compute_bishop_frames", _reference_frames)
    reference = _run_lorenz_tube(seed=42)

    mesh, ref_mesh = result.math_object.mesh, reference.math_object.mesh
    assert mesh.vertices.shape == ref_mesh.vertices.shape
    np.testing.assert_array_equal(mesh.faces, ref_mesh.faces)
    # Coordinates are in container millimetres; FRAME_ATOL scales with the tube radius there.
    np.testing.assert_allclose(mesh.vertices, ref_mesh.vertices, rtol=0, atol=FRAME_ATOL)
    assert result.validation == reference.validation


def test_lorenz_pipeline_mesh_depends_only_on_seed() -> None:
    """The same seed repeats the tube mesh exactly; another seed changes it."""
    first = _run_lorenz_tube(seed=42).math_object.mesh
    repeat = _run_lorenz_tube(seed=42).math_object.mesh
    other = _run_lorenz_tube(seed=43).math_object.mesh
    np.testing.assert_array_equal(first.vertices, repeat.vertices)
    assert not np.array_equal(first.vertices, other.vertices)
