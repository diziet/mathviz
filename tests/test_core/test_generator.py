"""Tests for GeneratorBase, @register decorator, and generator registry."""

from typing import Any

import numpy as np
import pytest

from mathviz.core.generator import (
    DuplicateRegistrationError,
    GeneratorBase,
    clear_registry,
    get_generator,
    get_generator_meta,
    list_generators,
    register,
)
from mathviz.core.math_object import BoundingBox, CoordSpace, MathObject, PointCloud

# ---------------------------------------------------------------------------
# Placeholder generators for testing registry mechanics
# ---------------------------------------------------------------------------


class _BasePlaceholder(GeneratorBase):
    """Minimal concrete generator for tests."""

    category = "test"
    description = "A test generator"
    resolution_params: dict[str, str] = {}

    def get_default_params(self) -> dict[str, Any]:
        return {"scale": 1.0}

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        rng = np.random.default_rng(seed)
        points = rng.standard_normal((100, 3))
        return MathObject(
            generator_name=self.name,
            parameters=params or self.get_default_params(),
            seed=seed,
            coord_space=CoordSpace.ABSTRACT,
            point_cloud=PointCloud(points=points.astype(np.float64)),
            bounding_box=BoundingBox(
                min_corner=(-3.0, -3.0, -3.0),
                max_corner=(3.0, 3.0, 3.0),
            ),
        )


@pytest.fixture(autouse=True)
def _clean_registry():
    """Clear the registry before and after each test."""
    clear_registry()
    yield
    clear_registry()


# ---------------------------------------------------------------------------
# Test: canonical name registration
# ---------------------------------------------------------------------------


def test_register_by_canonical_name() -> None:
    """Registering a generator makes it discoverable by canonical name."""

    @register
    class SimpleGen(_BasePlaceholder):
        name = "simple_gen"

    found = get_generator("simple_gen")
    assert found is SimpleGen


# ---------------------------------------------------------------------------
# Test: alias registration
# ---------------------------------------------------------------------------


def test_register_with_aliases() -> None:
    """Registering with aliases makes it discoverable by each alias."""

    @register(aliases=["alias_a", "alias_b"])
    class AliasedGen(_BasePlaceholder):
        name = "aliased_gen"

    assert get_generator("alias_a") is AliasedGen
    assert get_generator("alias_b") is AliasedGen


# ---------------------------------------------------------------------------
# Test: alias and canonical resolve to the same class
# ---------------------------------------------------------------------------


def test_alias_and_canonical_resolve_same_class() -> None:
    """Alias and canonical name resolve to the same class."""

    @register(aliases=["knot_alias"])
    class KnotGen(_BasePlaceholder):
        name = "knot_gen"

    assert get_generator("knot_gen") is get_generator("knot_alias")


# ---------------------------------------------------------------------------
# Test: duplicate registration raises error
# ---------------------------------------------------------------------------


def test_duplicate_name_raises() -> None:
    """Duplicate name registration raises an error."""

    @register
    class FirstGen(_BasePlaceholder):
        name = "duplicate_name"

    with pytest.raises(DuplicateRegistrationError, match="duplicate_name"):

        @register
        class SecondGen(_BasePlaceholder):
            name = "duplicate_name"


def test_duplicate_alias_raises() -> None:
    """Duplicate alias registration raises an error."""

    @register(aliases=["shared_alias"])
    class GenA(_BasePlaceholder):
        name = "gen_a"

    with pytest.raises(DuplicateRegistrationError, match="shared_alias"):

        @register(aliases=["shared_alias"])
        class GenB(_BasePlaceholder):
            name = "gen_b"


# ---------------------------------------------------------------------------
# Test: listing returns all generators with metadata
# ---------------------------------------------------------------------------


def test_list_generators_returns_metadata() -> None:
    """Registry listing returns all registered generators with metadata."""

    @register(aliases=["alt_sphere"])
    class SphereGen(_BasePlaceholder):
        name = "sphere"
        category = "parametric"
        description = "Generates a sphere"
        resolution_params = {"grid_resolution": "UV grid density"}

    @register
    class CubeGen(_BasePlaceholder):
        name = "cube"
        category = "parametric"
        description = "Generates a cube"

    generators = list_generators()
    assert len(generators) == 2

    names = {g.name for g in generators}
    assert names == {"sphere", "cube"}

    sphere_meta = next(g for g in generators if g.name == "sphere")
    assert sphere_meta.category == "parametric"
    assert sphere_meta.description == "Generates a sphere"
    assert "alt_sphere" in sphere_meta.aliases
    assert sphere_meta.resolution_params == {"grid_resolution": "UV grid density"}
    assert sphere_meta.generator_class is SphereGen


# ---------------------------------------------------------------------------
# Test: get_generator_meta
# ---------------------------------------------------------------------------


def test_get_generator_meta() -> None:
    """get_generator_meta returns full metadata for a registered name."""

    @register
    class MetaGen(_BasePlaceholder):
        name = "meta_gen"
        description = "Meta test"

    meta = get_generator_meta("meta_gen")
    assert meta.name == "meta_gen"
    assert meta.generator_class is MetaGen


# ---------------------------------------------------------------------------
# Test: unknown name raises KeyError
# ---------------------------------------------------------------------------


def test_unknown_name_raises_key_error() -> None:
    """Looking up an unregistered name raises KeyError."""
    with pytest.raises(KeyError, match="nonexistent"):
        get_generator("nonexistent")


# ---------------------------------------------------------------------------
# Test: register decorator without arguments uses class aliases
# ---------------------------------------------------------------------------


def test_register_uses_class_aliases() -> None:
    """@register without arguments uses aliases from the class attribute."""

    @register
    class ClassAliasGen(_BasePlaceholder):
        name = "class_alias_gen"
        aliases = ["from_class"]

    assert get_generator("from_class") is ClassAliasGen


# ---------------------------------------------------------------------------
# Test: generator produces valid MathObject
# ---------------------------------------------------------------------------


def test_placeholder_generates_valid_math_object() -> None:
    """A registered generator produces a valid MathObject."""

    @register
    class ValidGen(_BasePlaceholder):
        name = "valid_gen"

    gen = ValidGen()
    obj = gen.generate()
    errors = obj.validate()
    assert errors == []
    assert obj.generator_name == "valid_gen"
