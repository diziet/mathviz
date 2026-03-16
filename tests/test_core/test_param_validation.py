"""Tests for generator parameter key validation."""

from typing import Any

import pytest

from mathviz.core.generator import GeneratorBase, clear_registry, list_generators, register
from mathviz.core.math_object import MathObject


# ---------------------------------------------------------------------------
# Minimal test generator
# ---------------------------------------------------------------------------


class _DummyGenerator(GeneratorBase):
    """Minimal generator for param validation tests."""

    name = "dummy_validator"
    category = "test"
    resolution_params = {"grid_resolution": "Grid cells per axis"}
    _resolution_defaults = {"grid_resolution": 64}

    def get_default_params(self) -> dict[str, Any]:
        """Return default params."""
        return {
            "power": 2.0,
            "max_iterations": 10,
            "extent": 1.5,
        }

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Stub generate — not called in these tests."""
        raise NotImplementedError


@pytest.fixture(autouse=True)
def _clean_registry() -> None:
    """Isolate registry for each test."""
    clear_registry(suppress_discovery=True)
    register(_DummyGenerator)
    yield
    clear_registry(suppress_discovery=True)


@pytest.fixture
def gen() -> _DummyGenerator:
    """Return a _DummyGenerator instance."""
    return _DummyGenerator()


# ---------------------------------------------------------------------------
# Unknown param raises ValueError listing valid params
# ---------------------------------------------------------------------------


def test_unknown_param_raises_valueerror(gen: _DummyGenerator) -> None:
    """Unknown param key raises ValueError with valid param names."""
    with pytest.raises(ValueError, match="Unknown parameter.*'bad_key'"):
        gen.validate_param_keys({"bad_key": 42})


def test_unknown_param_lists_valid_params(gen: _DummyGenerator) -> None:
    """Error message includes all valid param names."""
    with pytest.raises(ValueError, match="Valid params:.*extent.*max_iterations.*power"):
        gen.validate_param_keys({"nope": 1})


def test_multiple_unknown_params_listed(gen: _DummyGenerator) -> None:
    """All unknown param names appear in the error message."""
    with pytest.raises(ValueError, match="alpha.*beta"):
        gen.validate_param_keys({"alpha": 1, "beta": 2, "power": 3.0})


# ---------------------------------------------------------------------------
# Valid params pass without error
# ---------------------------------------------------------------------------


def test_valid_params_pass(gen: _DummyGenerator) -> None:
    """Recognized params do not raise."""
    gen.validate_param_keys({"power": 3.0, "max_iterations": 20})


def test_none_params_pass(gen: _DummyGenerator) -> None:
    """None params do not raise."""
    gen.validate_param_keys(None)


def test_empty_params_pass(gen: _DummyGenerator) -> None:
    """Empty dict does not raise."""
    gen.validate_param_keys({})


def test_all_default_params_pass(gen: _DummyGenerator) -> None:
    """All default param keys are accepted."""
    gen.validate_param_keys(gen.get_default_params())


# ---------------------------------------------------------------------------
# quaternion_julia accepts max_iterations (not max_iter)
# ---------------------------------------------------------------------------


def test_quaternion_julia_accepts_max_iterations() -> None:
    """quaternion_julia uses max_iterations, not max_iter."""
    clear_registry(suppress_discovery=True)
    from mathviz.generators.fractals.quaternion_julia import QuaternionJuliaGenerator

    register(QuaternionJuliaGenerator)
    qj = QuaternionJuliaGenerator()

    # max_iterations is valid
    qj.validate_param_keys({"max_iterations": 20})

    # max_iter is NOT valid
    with pytest.raises(ValueError, match="Unknown parameter.*max_iter"):
        qj.validate_param_keys({"max_iter": 20})


# ---------------------------------------------------------------------------
# Resolution param passed as regular param gets helpful error
# ---------------------------------------------------------------------------


def test_resolution_param_as_regular_param_gives_helpful_error(
    gen: _DummyGenerator,
) -> None:
    """Resolution param passed as a regular param produces a specific error."""
    with pytest.raises(ValueError, match="Resolution parameter.*grid_resolution"):
        gen.validate_param_keys({"grid_resolution": 128})


def test_resolution_param_error_mentions_correct_usage(
    gen: _DummyGenerator,
) -> None:
    """Resolution param error mentions --resolution or API field."""
    with pytest.raises(ValueError, match="--resolution.*resolution field"):
        gen.validate_param_keys({"grid_resolution": 128})


# ---------------------------------------------------------------------------
# All existing generators pass validation with their documented defaults
# ---------------------------------------------------------------------------


def test_all_generators_pass_validation_with_defaults() -> None:
    """Every registered generator's default params pass validation."""
    clear_registry(suppress_discovery=False)
    generators = list_generators()
    assert len(generators) > 0, "No generators discovered"

    for meta in generators:
        gen_instance = meta.generator_class()
        defaults = gen_instance.get_default_params()
        # Should not raise
        gen_instance.validate_param_keys(defaults)
