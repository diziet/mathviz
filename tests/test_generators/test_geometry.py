"""Tests for geometry generators: voronoi_3d and generic_parametric."""

import numpy as np
import pytest

from mathviz.core.generator import clear_registry, register
from mathviz.core.representation import RepresentationType
from mathviz.generators.geometry.generic_parametric import (
    GenericParametricGenerator,
    validate_expression,
)
from mathviz.generators.geometry.voronoi_3d import Voronoi3DGenerator
from mathviz.generators.parametric.torus import TorusGenerator


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset registry for each test."""
    clear_registry(suppress_discovery=True)
    register(Voronoi3DGenerator)
    register(GenericParametricGenerator)
    register(TorusGenerator)
    yield
    clear_registry(suppress_discovery=True)


# ---------------------------------------------------------------------------
# Voronoi 3D: determinism with seed=42
# ---------------------------------------------------------------------------


def test_voronoi_seed_determinism() -> None:
    """Voronoi with seed=42 produces consistent geometry across runs."""
    gen = Voronoi3DGenerator()

    obj1 = gen.generate(seed=42)
    obj2 = gen.generate(seed=42)

    obj1.validate_or_raise()
    obj2.validate_or_raise()

    assert obj1.curves is not None
    assert obj2.curves is not None
    assert len(obj1.curves) == len(obj2.curves)

    for c1, c2 in zip(obj1.curves, obj2.curves):
        np.testing.assert_array_equal(c1.points, c2.points)


def test_voronoi_different_seeds_differ() -> None:
    """Different seeds produce different geometry."""
    gen = Voronoi3DGenerator()
    obj1 = gen.generate(seed=42)
    obj2 = gen.generate(seed=99)

    assert obj1.curves is not None
    assert obj2.curves is not None
    assert len(obj1.curves) != len(obj2.curves) or not all(
        np.array_equal(c1.points, c2.points)
        for c1, c2 in zip(obj1.curves, obj2.curves)
    )


# ---------------------------------------------------------------------------
# Voronoi 3D: cell count scales with seed points
# ---------------------------------------------------------------------------


def test_voronoi_edge_count_scales_with_points() -> None:
    """More seed points produce more Voronoi edges."""
    gen = Voronoi3DGenerator()

    obj_small = gen.generate(params={"num_points": 10}, seed=42)
    obj_large = gen.generate(params={"num_points": 50}, seed=42)

    assert obj_small.curves is not None
    assert obj_large.curves is not None
    assert len(obj_large.curves) > len(obj_small.curves)


def test_voronoi_output_structure() -> None:
    """Voronoi output has curves, correct metadata, and valid geometry."""
    gen = Voronoi3DGenerator()
    obj = gen.generate(params={"num_points": 15}, seed=7)
    obj.validate_or_raise()

    assert obj.curves is not None
    assert len(obj.curves) > 0
    assert obj.generator_name == "voronoi_3d"
    assert obj.category == "geometry"
    assert obj.bounding_box is not None

    for curve in obj.curves:
        assert curve.points.shape[1] == 3
        assert curve.points.shape[0] >= 3
        assert curve.closed


def test_voronoi_default_representation() -> None:
    """Voronoi default representation is WIREFRAME."""
    gen = Voronoi3DGenerator()
    rep = gen.get_default_representation()
    assert rep.type == RepresentationType.WIREFRAME


def test_voronoi_min_points_validation() -> None:
    """Voronoi rejects too few seed points."""
    gen = Voronoi3DGenerator()
    with pytest.raises(ValueError, match="num_points must be"):
        gen.generate(params={"num_points": 2})


# ---------------------------------------------------------------------------
# Generic parametric: torus formula matches dedicated torus generator
# ---------------------------------------------------------------------------


def test_generic_parametric_torus_matches_dedicated() -> None:
    """Generic parametric with torus formula matches the dedicated torus."""
    torus_gen = TorusGenerator()
    resolution = 32
    torus_obj = torus_gen.generate(
        params={"major_radius": 1.0, "minor_radius": 0.4},
        grid_resolution=resolution,
    )

    generic_gen = GenericParametricGenerator()
    generic_obj = generic_gen.generate(
        params={
            "x_expr": "(1.0 + 0.4 * cos(v)) * cos(u)",
            "y_expr": "(1.0 + 0.4 * cos(v)) * sin(u)",
            "z_expr": "0.4 * sin(v)",
            "wrap_u": True,
            "wrap_v": True,
        },
        grid_resolution=resolution,
    )

    torus_obj.validate_or_raise()
    generic_obj.validate_or_raise()

    assert torus_obj.mesh is not None
    assert generic_obj.mesh is not None

    np.testing.assert_allclose(
        generic_obj.mesh.vertices,
        torus_obj.mesh.vertices,
        atol=1e-10,
    )
    np.testing.assert_array_equal(
        generic_obj.mesh.faces,
        torus_obj.mesh.faces,
    )


def test_generic_parametric_output_structure() -> None:
    """Generic parametric output has mesh, metadata, and valid geometry."""
    gen = GenericParametricGenerator()
    obj = gen.generate(
        params={
            "x_expr": "cos(u) * cos(v)",
            "y_expr": "sin(u) * cos(v)",
            "z_expr": "sin(v)",
            "u_range": [0, 6.283185307179586],
            "v_range": [-1.5707963267948966, 1.5707963267948966],
            "wrap_u": True,
            "wrap_v": False,
        },
        grid_resolution=16,
    )
    obj.validate_or_raise()

    assert obj.mesh is not None
    assert obj.generator_name == "generic_parametric"
    assert obj.category == "geometry"
    assert obj.bounding_box is not None


def test_generic_parametric_default_representation() -> None:
    """Generic parametric default representation is SURFACE_SHELL."""
    gen = GenericParametricGenerator()
    rep = gen.get_default_representation()
    assert rep.type == RepresentationType.SURFACE_SHELL


# ---------------------------------------------------------------------------
# Generic parametric: malicious expressions are rejected
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("expr", [
    "__import__('os')",
    "exec('print(1)')",
    "eval('1+1')",
    "open('/etc/passwd')",
    "os.system('ls')",
    "__builtins__['eval']('1')",
    "compile('x', 'f', 'exec')",
    "getattr(u, '__class__')",
    "subprocess.run(['ls'])",
    "u.__class__.__subclasses__()",
    "[x for x in range(10)]",
    "lambda: 1",
])
def test_malicious_expressions_rejected(expr: str) -> None:
    """Malicious expressions are rejected by AST validation."""
    with pytest.raises(ValueError, match="Disallowed"):
        validate_expression(expr)


def test_malicious_expression_in_generator() -> None:
    """Malicious expressions in generator params raise ValueError."""
    gen = GenericParametricGenerator()
    with pytest.raises(ValueError, match="Disallowed"):
        gen.generate(
            params={
                "x_expr": "__import__('os').system('rm -rf /')",
                "y_expr": "v",
                "z_expr": "u",
            },
        )


def test_safe_expressions_allowed() -> None:
    """Valid math expressions pass validation."""
    safe_exprs = [
        "sin(u) * cos(v)",
        "sqrt(u**2 + v**2)",
        "exp(-u) * log(v + 1)",
        "arctan2(v, u) / pi",
        "clip(u, -1, 1)",
    ]
    for expr in safe_exprs:
        validate_expression(expr)


def test_unknown_name_in_expression() -> None:
    """Expressions with unknown names raise ValueError."""
    gen = GenericParametricGenerator()
    with pytest.raises(ValueError, match="Disallowed name"):
        gen.generate(
            params={
                "x_expr": "nonexistent_func(u)",
                "y_expr": "v",
                "z_expr": "u",
            },
        )


def test_generic_parametric_grid_resolution_validation() -> None:
    """Grid resolution below minimum raises ValueError."""
    gen = GenericParametricGenerator()
    with pytest.raises(ValueError, match="grid_resolution must be"):
        gen.generate(grid_resolution=1)
