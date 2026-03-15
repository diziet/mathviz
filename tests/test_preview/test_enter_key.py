"""Tests for Enter key triggering regeneration in preview UI input fields."""

from typing import Generator

import pytest
from fastapi.testclient import TestClient

from mathviz.core.generator import register
from mathviz.generators.parametric.torus import TorusGenerator
from mathviz.preview.server import app, reset_cache


def _ensure_torus_registered() -> None:
    """Re-register the torus generator if missing from the registry."""
    import mathviz.core.generator as gen_mod

    if "torus" not in gen_mod._alias_map:
        gen_mod._discovered = True
        register(TorusGenerator)


# --- Fixtures ---


@pytest.fixture(autouse=True)
def _ensure_generators() -> Generator[None, None, None]:
    """Ensure generators are registered and cache is clean."""
    _ensure_torus_registered()
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


# --- Enter key event handling tests ---


def test_html_includes_keydown_handling(preview_html: str) -> None:
    """Preview HTML includes keydown event handling on input fields."""
    assert "handleEnterKey" in preview_html
    assert "e.key !== 'Enter'" in preview_html


def test_enter_on_seed_triggers_generation(preview_html: str) -> None:
    """Pressing Enter in seed input triggers generation via applyGenerator."""
    assert "handleEnterKey(document.getElementById('controls')" in preview_html
    assert "applyGenerator(currentGen, parseSeed())" in preview_html


def test_enter_on_container_input_triggers_regeneration(
    preview_html: str,
) -> None:
    """Pressing Enter in a container dimension input triggers regeneration."""
    assert (
        "handleEnterKey(document.getElementById('container-panel')"
        in preview_html
    )


def test_event_delegation_covers_dynamic_inputs(preview_html: str) -> None:
    """Event delegation covers dynamically-added parameter inputs."""
    assert (
        "handleEnterKey(document.getElementById('param-panel')"
        in preview_html
    )
    # Uses event delegation on the panel, not individual inputs
    assert "panel.addEventListener('keydown'" in preview_html


def test_non_enter_keys_do_not_trigger(preview_html: str) -> None:
    """Non-Enter keys do not trigger regeneration."""
    assert "if (e.key !== 'Enter') return" in preview_html


def test_checkbox_inputs_excluded(preview_html: str) -> None:
    """Checkbox inputs are excluded from Enter key handling."""
    assert "e.target.type === 'checkbox'" in preview_html


def test_input_blurred_after_enter(preview_html: str) -> None:
    """Input is blurred after Enter key to show result immediately."""
    assert "e.target.blur()" in preview_html
