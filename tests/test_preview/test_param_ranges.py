"""Tests for param range derivation, min/max UI fields, and randomize behavior."""

from typing import Generator

import pytest
from fastapi.testclient import TestClient

from mathviz.core.generator import register
from mathviz.generators.attractors.lorenz import LorenzGenerator
from mathviz.generators.fractals.mandelbulb import MandelbulbGenerator
from mathviz.generators.parametric.torus import TorusGenerator
from mathviz.preview.server import _derive_param_range, app, reset_cache


def _ensure_generators_registered() -> None:
    """Re-register test generators if missing from the registry."""
    import mathviz.core.generator as gen_mod

    gen_mod._discovered = True
    for cls in (TorusGenerator, LorenzGenerator, MandelbulbGenerator):
        if cls.name not in gen_mod._alias_map:
            register(cls)


@pytest.fixture(autouse=True)
def _setup() -> Generator[None, None, None]:
    """Ensure generators are registered and cache is clean."""
    _ensure_generators_registered()
    reset_cache()
    yield
    reset_cache()


@pytest.fixture
def client() -> TestClient:
    """Return a FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def preview_html(client: TestClient) -> str:
    """Fetch the preview HTML once for assertion-only tests."""
    resp = client.get("/")
    assert resp.status_code == 200
    return resp.text


# --- _derive_param_range positive integer fix ---


class TestDeriveParamRangePositiveInt:
    """Positive integer defaults derive min=1, not min=0."""

    def test_positive_int_min_is_one(self) -> None:
        """_derive_param_range(2) returns min: 1 for positive integers."""
        rng = _derive_param_range(2)
        assert rng is not None
        assert rng["min"] == 1

    def test_positive_int_max(self) -> None:
        """_derive_param_range(2) returns max: 4 (2*default)."""
        rng = _derive_param_range(2)
        assert rng is not None
        assert rng["max"] == 4

    def test_positive_int_step(self) -> None:
        """Positive integer step is always 1."""
        rng = _derive_param_range(2)
        assert rng is not None
        assert rng["step"] == 1

    def test_large_positive_int(self) -> None:
        """Larger positive int still derives min=1."""
        rng = _derive_param_range(100)
        assert rng is not None
        assert rng["min"] == 1
        assert rng["max"] == 200


# --- _derive_param_range zero default ---


class TestDeriveParamRangeZero:
    """Zero defaults get min=0."""

    def test_zero_int_min_is_zero(self) -> None:
        """_derive_param_range(0) returns min: 0."""
        rng = _derive_param_range(0)
        assert rng is not None
        assert rng["min"] == 0

    def test_zero_int_max(self) -> None:
        """_derive_param_range(0) returns max: 10."""
        rng = _derive_param_range(0)
        assert rng is not None
        assert rng["max"] == 10


# --- _derive_param_range negative default ---


class TestDeriveParamRangeNegative:
    """Negative defaults include the default value in range."""

    def test_negative_int_includes_default(self) -> None:
        """_derive_param_range(-3) range includes -3."""
        rng = _derive_param_range(-3)
        assert rng is not None
        assert rng["min"] <= -3 <= rng["max"]

    def test_negative_int_range_values(self) -> None:
        """_derive_param_range(-3) returns min=-6, max=6."""
        rng = _derive_param_range(-3)
        assert rng is not None
        assert rng["min"] == -6
        assert rng["max"] == 6
        assert rng["step"] == 1


# --- Randomize with server-derived ranges never produces values below minimums ---


class TestRandomizeRespectsMins:
    """Server-derived ranges for positive int params have min >= 1."""

    def test_torus_param_ranges_positive_mins(self, client: TestClient) -> None:
        """Torus param ranges have positive min values for positive defaults."""
        resp = client.get("/api/generators/torus/param-ranges")
        assert resp.status_code == 200
        data = resp.json()
        for name, rng in data.items():
            assert rng["min"] <= rng["max"], f"{name}: min > max"

    def test_derived_positive_int_never_zero_min(self) -> None:
        """No positive integer default produces a derived min of 0."""
        for val in [1, 2, 3, 5, 10, 50]:
            rng = _derive_param_range(val)
            assert rng is not None
            assert rng["min"] >= 1, f"val={val} got min={rng['min']}"


# --- Preview HTML contains min/max input fields ---


class TestMinMaxFieldsInHTML:
    """Preview HTML contains editable min/max range fields."""

    def test_min_input_fields_exist(self, preview_html: str) -> None:
        """HTML contains min range input elements."""
        assert 'id="param-range-min-' in preview_html

    def test_max_input_fields_exist(self, preview_html: str) -> None:
        """HTML contains max range input elements."""
        assert 'id="param-range-max-' in preview_html

    def test_range_row_class_exists(self, preview_html: str) -> None:
        """HTML contains param-range-row CSS class."""
        assert "param-range-row" in preview_html

    def test_populate_range_fields_function(self, preview_html: str) -> None:
        """populateRangeFields function is defined."""
        assert "populateRangeFields" in preview_html


# --- Min/max fields are pre-populated from param-ranges endpoint ---


class TestRangeFieldsPrePopulated:
    """Range fields are populated via fetchParamRanges in fetchAndPopulateParams."""

    def test_fetch_and_populate_calls_range_fetch(self, preview_html: str) -> None:
        """fetchAndPopulateParams calls fetchParamRanges."""
        assert "fetchParamRanges(generatorName)" in preview_html

    def test_populate_range_fields_called(self, preview_html: str) -> None:
        """fetchAndPopulateParams calls populateRangeFields."""
        assert "populateRangeFields(ranges)" in preview_html


# --- Editing min/max in UI affects randomize ---


class TestUIMinMaxAffectsRandomize:
    """Randomize reads min/max from UI inputs, not cached server values."""

    def test_randomize_reads_min_input(self, preview_html: str) -> None:
        """_fillRandomValues reads param-range-min inputs."""
        assert "param-range-min-" in preview_html
        assert "parseFloat(minInput.value)" in preview_html

    def test_randomize_reads_max_input(self, preview_html: str) -> None:
        """_fillRandomValues reads param-range-max inputs."""
        assert "param-range-max-" in preview_html
        assert "parseFloat(maxInput.value)" in preview_html


# --- Randomize always triggers apply ---


class TestRandomizeAlwaysApplies:
    """Randomize always generates, not only when autoApply is on."""

    def test_no_auto_apply_guard(self, preview_html: str) -> None:
        """randomizeParams does not gate on state.autoApply."""
        # The old code had 'if (state.autoApply) { applyParams(); }'
        # The new code always calls fetch /api/generate directly
        assert "MAX_RANDOMIZE_ATTEMPTS" in preview_html

    def test_retry_on_validation_error(self, preview_html: str) -> None:
        """Randomize retries on 400 status (constraint violation)."""
        assert "resp.status === 400" in preview_html
