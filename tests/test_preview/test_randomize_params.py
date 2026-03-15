"""Tests for the randomize parameters feature in preview UI."""

from typing import Any, Generator

import pytest
from fastapi.testclient import TestClient

from mathviz.core.generator import GeneratorBase, register
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


# --- Fixtures ---


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


# --- Randomize button exists in HTML ---


class TestRandomizeButtonPresent:
    """Preview HTML contains a Randomize button."""

    def test_randomize_button_exists(self, preview_html: str) -> None:
        """Randomize button element is present in the HTML."""
        assert 'id="param-randomize-btn"' in preview_html

    def test_randomize_button_in_param_buttons(self, preview_html: str) -> None:
        """Randomize button is inside the param-buttons container."""
        idx_buttons = preview_html.find("param-buttons")
        idx_randomize = preview_html.find("param-randomize-btn")
        assert idx_buttons < idx_randomize

    def test_randomize_function_defined(self, preview_html: str) -> None:
        """randomizeParams function is defined in the JavaScript."""
        assert "randomizeParams" in preview_html

    def test_randomize_fetches_ranges(self, preview_html: str) -> None:
        """Randomize logic fetches param-ranges from the API."""
        assert "param-ranges" in preview_html


# --- Clicking Randomize changes parameter input values ---


class TestRandomizeChangesValues:
    """Randomize logic uses param-ranges to set random values in inputs."""

    def test_randomize_in_range_function(self, preview_html: str) -> None:
        """randomizeInRange helper function exists for uniform random selection."""
        assert "randomizeInRange" in preview_html

    def test_randomize_iterates_param_inputs(self, preview_html: str) -> None:
        """Randomize iterates over all param inputs in paramFields."""
        assert "paramFields.querySelectorAll" in preview_html


# --- Randomized values fall within exploration ranges ---


class TestParamRangesEndpoint:
    """GET /api/generators/{name}/param-ranges returns valid ranges."""

    def test_lorenz_param_ranges(self, client: TestClient) -> None:
        """Lorenz param-ranges endpoint returns ranges for all params."""
        resp = client.get("/api/generators/lorenz/param-ranges")
        assert resp.status_code == 200
        data = resp.json()
        assert "sigma" in data
        assert "rho" in data
        assert "beta" in data
        for name, rng in data.items():
            assert "min" in rng
            assert "max" in rng
            assert "step" in rng
            assert rng["min"] <= rng["max"]

    def test_torus_param_ranges(self, client: TestClient) -> None:
        """Torus param-ranges endpoint returns ranges for radii."""
        resp = client.get("/api/generators/torus/param-ranges")
        assert resp.status_code == 200
        data = resp.json()
        assert "major_radius" in data
        assert "minor_radius" in data

    def test_lorenz_sigma_range_is_explicit(self, client: TestClient) -> None:
        """Lorenz sigma range uses the explicit exploration range, not derived."""
        resp = client.get("/api/generators/lorenz/param-ranges")
        data = resp.json()
        assert data["sigma"]["min"] == 5.0
        assert data["sigma"]["max"] == 20.0

    def test_unknown_generator_returns_404(self, client: TestClient) -> None:
        """Param-ranges for unknown generator returns 404."""
        resp = client.get("/api/generators/nonexistent/param-ranges")
        assert resp.status_code == 404


# --- Seed is also randomized ---


class TestSeedRandomization:
    """Randomize also sets a random seed value."""

    def test_randomize_sets_seed(self, preview_html: str) -> None:
        """Randomize logic sets seed-input to a random value."""
        assert "seed-input" in preview_html
        assert "Math.floor(Math.random() * 1000000)" in preview_html


# --- Default exploration ranges derived from defaults ---


class TestDerivedRanges:
    """Default exploration ranges are derived from defaults when not specified."""

    def test_positive_float_range(self) -> None:
        """Positive float derives to [0.25x, 2x]."""
        rng = _derive_param_range(10.0)
        assert rng is not None
        assert rng["min"] == pytest.approx(2.5)
        assert rng["max"] == pytest.approx(20.0)

    def test_positive_int_range(self) -> None:
        """Positive int derives to [0, 2x]."""
        rng = _derive_param_range(5)
        assert rng is not None
        assert rng["min"] == 0
        assert rng["max"] == 10
        assert rng["step"] == 1

    def test_zero_float_range(self) -> None:
        """Zero float derives to [-1, 1]."""
        rng = _derive_param_range(0.0)
        assert rng is not None
        assert rng["min"] == -1.0
        assert rng["max"] == 1.0

    def test_zero_int_range(self) -> None:
        """Zero int derives to [0, 10]."""
        rng = _derive_param_range(0)
        assert rng is not None
        assert rng["min"] == 0
        assert rng["max"] == 10

    def test_boolean_returns_none(self) -> None:
        """Boolean params return None (no numeric range)."""
        rng = _derive_param_range(True)
        assert rng is None

    def test_string_returns_none(self) -> None:
        """String params return None."""
        rng = _derive_param_range("hello")
        assert rng is None

    def test_negative_int_range(self) -> None:
        """Negative int derives to [-2x, 2x] with step 1."""
        rng = _derive_param_range(-5)
        assert rng is not None
        assert rng["min"] == -10
        assert rng["max"] == 10
        assert rng["step"] == 1

    def test_negative_float_range(self) -> None:
        """Negative float derives to [2x, 2*abs(x)]."""
        rng = _derive_param_range(-5.0)
        assert rng is not None
        assert rng["min"] == pytest.approx(-10.0)
        assert rng["max"] == pytest.approx(10.0)
        assert rng["step"] == pytest.approx(0.5)

    def test_tiny_float_step_clamped(self) -> None:
        """Very small float doesn't produce zero step."""
        rng = _derive_param_range(0.0000001)
        assert rng is not None
        assert rng["step"] >= 1e-6


# --- Generators can define explicit exploration ranges ---


class TestExplicitRanges:
    """Generators can define explicit exploration ranges via get_param_ranges()."""

    def test_lorenz_has_explicit_ranges(self) -> None:
        """LorenzGenerator.get_param_ranges() returns non-empty dict."""
        gen = LorenzGenerator()
        ranges = gen.get_param_ranges()
        assert len(ranges) > 0
        assert "sigma" in ranges

    def test_torus_has_explicit_ranges(self) -> None:
        """TorusGenerator.get_param_ranges() returns non-empty dict."""
        gen = TorusGenerator()
        ranges = gen.get_param_ranges()
        assert len(ranges) > 0
        assert "major_radius" in ranges

    def test_mandelbulb_has_explicit_ranges(self) -> None:
        """MandelbulbGenerator.get_param_ranges() returns non-empty dict."""
        gen = MandelbulbGenerator()
        ranges = gen.get_param_ranges()
        assert len(ranges) > 0
        assert "power" in ranges

    def test_base_class_default_is_empty(self) -> None:
        """GeneratorBase.get_param_ranges() returns empty dict by default."""
        # Verify the base class implementation returns empty dict
        # Use a concrete generator and call the base method explicitly
        gen = TorusGenerator()
        assert GeneratorBase.get_param_ranges(gen) == {}

    def test_explicit_ranges_override_derived(self, client: TestClient) -> None:
        """Explicit ranges from generator take precedence over derived."""
        resp = client.get("/api/generators/lorenz/param-ranges")
        data = resp.json()
        # Lorenz sigma default is 10.0, derived would be [2.5, 20.0]
        # Explicit is [5.0, 20.0]
        assert data["sigma"]["min"] == 5.0


# --- Keyboard shortcut triggers randomization ---


class TestKeyboardShortcut:
    """R key triggers randomization when no input is focused."""

    def test_keydown_listener_exists(self, preview_html: str) -> None:
        """Keyboard event listener for R key is present."""
        assert "keydown" in preview_html
        assert "e.key === 'r'" in preview_html or "e.key === 'R'" in preview_html

    def test_shortcut_checks_active_element(self, preview_html: str) -> None:
        """Shortcut skips when an input/textarea/select is focused."""
        assert "activeElement" in preview_html

    def test_shortcut_calls_randomize(self, preview_html: str) -> None:
        """Shortcut calls randomizeParams()."""
        assert "randomizeParams()" in preview_html
