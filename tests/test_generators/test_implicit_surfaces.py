"""Tests for Schwarz P, Schwarz D, Costa surface, and genus-2 surface generators."""

import numpy as np
import pytest
import trimesh

from mathviz.core.generator import GeneratorBase, clear_registry, register
from mathviz.generators.implicit.costa_surface import CostaSurfaceGenerator
from mathviz.generators.implicit.genus2_surface import Genus2SurfaceGenerator
from mathviz.generators.implicit.schwarz_d import SchwarzDGenerator
from mathviz.generators.implicit.schwarz_p import SchwarzPGenerator

_ALL_GENERATORS = [
    SchwarzPGenerator, SchwarzDGenerator,
    CostaSurfaceGenerator, Genus2SurfaceGenerator,
]


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset registry and register generators for each test."""
    clear_registry(suppress_discovery=True)
    for gen_cls in _ALL_GENERATORS:
        register(gen_cls)
    yield
    clear_registry(suppress_discovery=True)


_TEST_VOXEL_RESOLUTION = 32


# ===========================================================================
# TPMS generators (Schwarz P & D) — parameterized
# ===========================================================================

_TPMS_GENERATORS = [SchwarzPGenerator, SchwarzDGenerator]
_TPMS_IDS = ["schwarz_p", "schwarz_d"]


@pytest.mark.parametrize("gen_cls", _TPMS_GENERATORS, ids=_TPMS_IDS)
def test_tpms_produces_manifold_mesh(gen_cls: type[GeneratorBase]) -> None:
    """TPMS generator produces a non-empty mesh with positive area."""
    gen = gen_cls()
    obj = gen.generate(voxel_resolution=_TEST_VOXEL_RESOLUTION)
    obj.validate_or_raise()
    assert obj.mesh is not None
    assert len(obj.mesh.vertices) > 0
    assert len(obj.mesh.faces) > 0

    tm = trimesh.Trimesh(
        vertices=obj.mesh.vertices, faces=obj.mesh.faces, process=False,
    )
    assert tm.area > 0


@pytest.mark.parametrize("gen_cls", _TPMS_GENERATORS, ids=_TPMS_IDS)
def test_tpms_periods_scale_face_count(gen_cls: type[GeneratorBase]) -> None:
    """TPMS with periods=3 has more faces than periods=1."""
    gen = gen_cls()
    obj_1 = gen.generate(
        params={"periods": 1}, voxel_resolution=_TEST_VOXEL_RESOLUTION,
    )
    obj_3 = gen.generate(
        params={"periods": 3}, voxel_resolution=_TEST_VOXEL_RESOLUTION,
    )
    assert obj_1.mesh is not None and obj_3.mesh is not None
    assert len(obj_3.mesh.faces) > len(obj_1.mesh.faces)


# ===========================================================================
# Costa surface
# ===========================================================================


def test_costa_surface_produces_geometry() -> None:
    """Costa surface produces non-empty valid geometry."""
    gen = CostaSurfaceGenerator()
    obj = gen.generate(grid_resolution=32)
    obj.validate_or_raise()
    assert obj.mesh is not None
    assert len(obj.mesh.vertices) > 0
    assert len(obj.mesh.faces) > 0
    assert not np.any(np.isnan(obj.mesh.vertices))


# ===========================================================================
# Genus-2 surface
# ===========================================================================

_GENUS2_VOXEL_RESOLUTION = 48


def test_genus2_surface_produces_mesh() -> None:
    """Genus-2 surface produces a non-empty mesh."""
    gen = Genus2SurfaceGenerator()
    obj = gen.generate(voxel_resolution=_GENUS2_VOXEL_RESOLUTION)
    obj.validate_or_raise()
    assert obj.mesh is not None
    assert len(obj.mesh.vertices) > 0
    assert len(obj.mesh.faces) > 0


def test_genus2_surface_has_genus_greater_than_one() -> None:
    """Genus-2 surface has Euler characteristic consistent with genus > 1."""
    gen = Genus2SurfaceGenerator()
    obj = gen.generate(voxel_resolution=_GENUS2_VOXEL_RESOLUTION)
    assert obj.mesh is not None

    tm = trimesh.Trimesh(
        vertices=obj.mesh.vertices, faces=obj.mesh.faces, process=False,
    )
    # For a closed genus-g surface: chi = 2 - 2g
    # genus > 1 means chi < 0
    assert tm.euler_number < 0, (
        f"Expected negative Euler characteristic for genus > 1, "
        f"got {tm.euler_number}"
    )
