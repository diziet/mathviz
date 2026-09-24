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
    """Register the test generators if missing; reset the cache before and after the test."""
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
        """_derive_param_range(2) returns max: 10 (floor of 10 for small values)."""
        rng = _derive_param_range(2)
        assert rng is not None
        assert rng["max"] == 10

    def test_positive_int_step(self) -> None:
        """_derive_param_range(2) returns step: 1."""
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


# --- Server-derived ranges: min <= max for torus, min >= 1 for positive ints ---


class TestRandomizeRespectsMins:
    """Server-derived ranges: torus min <= max, positive int min >= 1."""

    def test_torus_param_ranges_positive_mins(self, client: TestClient) -> None:
        """Each torus param range has min <= max."""
        resp = client.get("/api/generators/torus/param-ranges")
        assert resp.status_code == 200
        data = resp.json()
        for name, rng in data.items():
            assert rng["min"] <= rng["max"], f"{name}: min > max"

    def test_derived_positive_int_never_zero_min(self) -> None:
        """Positive integer defaults 1, 2, 3, 5, 10 and 50 derive a min of at least 1."""
        for val in [1, 2, 3, 5, 10, 50]:
            rng = _derive_param_range(val)
            assert rng is not None
            assert rng["min"] >= 1, f"val={val} got min={rng['min']}"


# --- Preview HTML contains min/max input fields ---


class TestMinMaxFieldsInHTML:
    """Preview HTML contains editable min/max range fields."""

    def test_min_input_fields_created(self, preview_html: str) -> None:
        """Preview HTML contains the param-range-min- ID prefix."""
        assert "param-range-min-" in preview_html

    def test_max_input_fields_created(self, preview_html: str) -> None:
        """Preview HTML contains the param-range-max- ID prefix."""
        assert "param-range-max-" in preview_html

    def test_range_row_class_exists(self, preview_html: str) -> None:
        """HTML contains param-range-row CSS class."""
        assert "param-range-row" in preview_html

    def test_populate_range_fields_function(self, preview_html: str) -> None:
        """populateRangeFields function is defined."""
        assert "populateRangeFields" in preview_html


# --- Range fetch and populate calls in the preview HTML ---


class TestRangeFieldsPrePopulated:
    """Preview HTML calls fetchParamRanges and populateRangeFields."""

    def test_fetch_and_populate_calls_range_fetch(self, preview_html: str) -> None:
        """Preview HTML calls fetchParamRanges(generatorName)."""
        assert "fetchParamRanges(generatorName)" in preview_html

    def test_populate_range_fields_called(self, preview_html: str) -> None:
        """Preview HTML calls populateRangeFields(ranges)."""
        assert "populateRangeFields(ranges)" in preview_html


# --- Range input parsing in the preview HTML ---


class TestUIMinMaxAffectsRandomize:
    """Preview HTML parses the min and max range inputs."""

    def test_randomize_reads_min_input(self, preview_html: str) -> None:
        """Preview HTML contains param-range-min- and parseFloat(minInput.value)."""
        assert "param-range-min-" in preview_html
        assert "parseFloat(minInput.value)" in preview_html

    def test_randomize_reads_max_input(self, preview_html: str) -> None:
        """Preview HTML contains param-range-max- and parseFloat(maxInput.value)."""
        assert "param-range-max-" in preview_html
        assert "parseFloat(maxInput.value)" in preview_html


# --- Randomize retry code in the preview HTML ---


class TestRandomizeAlwaysApplies:
    """Tests for the randomize retry and generate code in the preview HTML."""

    def test_no_auto_apply_guard(self, preview_html: str) -> None:
        """Preview HTML contains MAX_RANDOMIZE_ATTEMPTS."""
        # randomizeParams used to call applyParams() only inside
        # 'if (state.autoApply)'. It now calls fetch /api/generate itself.
        assert "MAX_RANDOMIZE_ATTEMPTS" in preview_html

    def test_retry_on_validation_error(self, preview_html: str) -> None:
        """Preview HTML checks result.status === 400."""
        assert "result.status === 400" in preview_html

    def test_exhausted_retries_show_error(self, preview_html: str) -> None:
        """Preview HTML contains the message text "Randomize failed after"."""
        assert "Randomize failed after" in preview_html

    def test_shared_generate_helper(self, preview_html: str) -> None:
        """Generate logic is extracted into _doGenerate helper."""
        assert "_doGenerate" in preview_html
