"""Tests for the L-system / fractal tree generator."""

import numpy as np
import pytest
from typer.testing import CliRunner

from mathviz.cli import app
from mathviz.core.generator import clear_registry, get_generator, list_generators, register
from mathviz.core.representation import RepresentationType
from mathviz.generators.procedural._lsystem_engine import PRESETS, rewrite
from mathviz.generators.procedural.lsystem import LSystemGenerator

_runner = CliRunner()


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset registry and re-register L-system generator for each test."""
    clear_registry(suppress_discovery=True)
    register(LSystemGenerator)
    yield
    clear_registry(suppress_discovery=True)


_LOW_ITERATIONS = 3


# ---------------------------------------------------------------------------
# Tree preset produces a branching curve structure
# ---------------------------------------------------------------------------


def test_tree_preset_produces_branching_curves() -> None:
    """Tree preset produces multiple branch curves."""
    gen = LSystemGenerator()
    obj = gen.generate(params={"preset": "tree", "iterations": _LOW_ITERATIONS}, seed=42)
    obj.validate_or_raise()

    assert obj.curves is not None
    assert len(obj.curves) > 1, "Tree should produce multiple branches"
    for curve in obj.curves:
        assert curve.points.shape[0] >= 2
        assert curve.points.shape[1] == 3
        assert curve.points.dtype == np.float64


# ---------------------------------------------------------------------------
# Increasing iterations increases segments
# ---------------------------------------------------------------------------


def test_more_iterations_more_segments() -> None:
    """More iterations produce more curve points."""
    gen = LSystemGenerator()
    obj_low = gen.generate(
        params={"preset": "tree", "iterations": 2, "jitter": 0.0}, seed=42,
    )
    obj_high = gen.generate(
        params={"preset": "tree", "iterations": 4, "jitter": 0.0}, seed=42,
    )

    total_low = sum(c.points.shape[0] for c in obj_low.curves)
    total_high = sum(c.points.shape[0] for c in obj_high.curves)
    assert total_high > total_low


# ---------------------------------------------------------------------------
# Different presets produce distinct geometries
# ---------------------------------------------------------------------------


def test_different_presets_produce_distinct_geometries() -> None:
    """Each preset produces a different bounding box extent."""
    gen = LSystemGenerator()
    bboxes = {}
    for preset_name in ("tree", "bush", "fern"):
        obj = gen.generate(
            params={"preset": preset_name, "iterations": _LOW_ITERATIONS},
            seed=42,
        )
        obj.validate_or_raise()
        bboxes[preset_name] = obj.bounding_box

    # At least two presets should differ in bounding box size
    sizes = {
        name: tuple(
            round(b.max_corner[i] - b.min_corner[i], 4) for i in range(3)
        )
        for name, b in bboxes.items()
    }
    unique_sizes = set(sizes.values())
    assert len(unique_sizes) >= 2, f"Presets too similar: {sizes}"


# ---------------------------------------------------------------------------
# Angle parameter affects branching spread
# ---------------------------------------------------------------------------


def test_angle_affects_spread() -> None:
    """Different angles produce different bounding box extents."""
    gen = LSystemGenerator()
    obj_narrow = gen.generate(
        params={"preset": "tree", "iterations": _LOW_ITERATIONS, "angle": 10.0, "jitter": 0.0},
        seed=42,
    )
    obj_wide = gen.generate(
        params={"preset": "tree", "iterations": _LOW_ITERATIONS, "angle": 45.0, "jitter": 0.0},
        seed=42,
    )

    def x_spread(obj: object) -> float:
        bb = obj.bounding_box
        return bb.max_corner[0] - bb.min_corner[0]

    assert x_spread(obj_wide) > x_spread(obj_narrow)


# ---------------------------------------------------------------------------
# Seed-dependent random jitter
# ---------------------------------------------------------------------------


def test_output_is_seed_dependent() -> None:
    """Different seeds produce different outputs due to random jitter."""
    gen = LSystemGenerator()
    obj1 = gen.generate(
        params={"preset": "tree", "iterations": _LOW_ITERATIONS}, seed=1,
    )
    obj2 = gen.generate(
        params={"preset": "tree", "iterations": _LOW_ITERATIONS}, seed=2,
    )

    pts1 = np.vstack([c.points for c in obj1.curves])
    pts2 = np.vstack([c.points for c in obj2.curves])

    # With jitter, different seeds should give different points
    assert not np.allclose(pts1, pts2, atol=1e-6)


def test_same_seed_identical() -> None:
    """Same seed produces identical output."""
    gen = LSystemGenerator()
    obj1 = gen.generate(
        params={"preset": "tree", "iterations": _LOW_ITERATIONS}, seed=42,
    )
    obj2 = gen.generate(
        params={"preset": "tree", "iterations": _LOW_ITERATIONS}, seed=42,
    )

    for c1, c2 in zip(obj1.curves, obj2.curves):
        np.testing.assert_array_equal(c1.points, c2.points)


# ---------------------------------------------------------------------------
# Hilbert 3D preset produces a space-filling curve
# ---------------------------------------------------------------------------


def test_hilbert3d_space_filling() -> None:
    """Hilbert 3D preset fills space (points span all three axes)."""
    gen = LSystemGenerator()
    obj = gen.generate(
        params={"preset": "hilbert3d", "iterations": 2, "jitter": 0.0},
        seed=42,
    )
    obj.validate_or_raise()

    assert obj.curves is not None
    all_pts = np.vstack([c.points for c in obj.curves])

    # Space-filling: should have extent in all three axes
    for axis in range(3):
        extent = all_pts[:, axis].max() - all_pts[:, axis].min()
        assert extent > 0.1, f"Axis {axis} extent too small: {extent}"


# ---------------------------------------------------------------------------
# Sierpinski preset
# ---------------------------------------------------------------------------


def test_sierpinski_preset() -> None:
    """Sierpinski preset produces valid curves."""
    gen = LSystemGenerator()
    obj = gen.generate(
        params={"preset": "sierpinski", "iterations": 4, "jitter": 0.0},
        seed=42,
    )
    obj.validate_or_raise()
    assert obj.curves is not None
    assert len(obj.curves) >= 1


# ---------------------------------------------------------------------------
# Generator registers and appears in list
# ---------------------------------------------------------------------------


def test_generator_registers() -> None:
    """L-system generator appears in the registry."""
    gen_cls = get_generator("lsystem")
    assert gen_cls is not None

    names = [m.name for m in list_generators()]
    assert "lsystem" in names


# ---------------------------------------------------------------------------
# Representation
# ---------------------------------------------------------------------------


def test_default_representation_is_tube() -> None:
    """Default representation is TUBE."""
    gen = LSystemGenerator()
    rep = gen.get_default_representation()
    assert rep.type == RepresentationType.TUBE
    assert rep.tube_radius is not None
    assert rep.tube_radius > 0


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def test_metadata() -> None:
    """Generator records correct metadata."""
    gen = LSystemGenerator()
    obj = gen.generate(
        params={"preset": "bush", "iterations": _LOW_ITERATIONS}, seed=7,
    )
    assert obj.generator_name == "lsystem"
    assert obj.category == "procedural"
    assert obj.seed == 7
    assert obj.parameters["preset"] == "bush"
    assert obj.bounding_box is not None


# ---------------------------------------------------------------------------
# Rewrite engine
# ---------------------------------------------------------------------------


def test_rewrite_zero_iterations() -> None:
    """Zero iterations returns the axiom unchanged."""
    result = rewrite("F", {"F": "FF"}, 0)
    assert result == "F"


def test_rewrite_one_iteration() -> None:
    """One iteration applies rules once."""
    result = rewrite("F", {"F": "F+F"}, 1)
    assert result == "F+F"


# ---------------------------------------------------------------------------
# CLI: render-2d succeeds
# ---------------------------------------------------------------------------


def test_render_2d_cli(tmp_path: object) -> None:
    """mathviz render-2d lsystem -o test.png succeeds (or skips if no PyVista)."""
    out_file = str(tmp_path / "lsystem_test.png")  # type: ignore[operator]
    result = _runner.invoke(app, [
        "render-2d", "lsystem",
        "-o", out_file,
        "--param", "iterations=3",
    ])
    if result.exit_code == 2 and "PyVista" in result.output:
        pytest.skip("PyVista not installed")
    assert result.exit_code == 0, (
        f"render-2d failed (exit={result.exit_code}): {result.output}"
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_invalid_preset_raises() -> None:
    """Unknown preset raises ValueError."""
    gen = LSystemGenerator()
    with pytest.raises(ValueError, match="Unknown preset"):
        gen.generate(params={"preset": "nonexistent"})


def test_invalid_iterations_raises() -> None:
    """Too many iterations raises ValueError."""
    gen = LSystemGenerator()
    with pytest.raises(ValueError, match="iterations"):
        gen.generate(params={"iterations": 99})
