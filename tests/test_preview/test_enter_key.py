"""Tests for Enter key triggering regeneration in preview UI input fields."""

import re
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


def _extract_script(html: str) -> str:
    """Extract the main script block from the HTML."""
    match = re.search(
        r'<script type="module">(.*?)</script>', html, re.DOTALL
    )
    assert match, "No module script block found in HTML"
    return match.group(1)


# --- Enter key event handling tests ---


def test_html_includes_keydown_handling(preview_html: str) -> None:
    """Preview HTML defines a handleEnterKey function with keydown listener."""
    script = _extract_script(preview_html)
    assert re.search(
        r"function\s+handleEnterKey\s*\(", script
    ), "handleEnterKey function not defined"
    assert re.search(
        r"addEventListener\s*\(\s*['\"]keydown['\"]", script
    ), "No keydown event listener found"


def test_enter_on_seed_triggers_generation(preview_html: str) -> None:
    """Controls panel Enter handler calls applyGenerator with current seed."""
    script = _extract_script(preview_html)
    # handleEnterKey is called with the controls panel element
    assert re.search(
        r"handleEnterKey\s*\(\s*document\.getElementById\s*\(\s*['\"]controls['\"]\s*\)",
        script,
    ), "handleEnterKey not wired to controls panel"
    # The controls handler invokes applyGenerator
    assert re.search(
        r"applyGenerator\s*\(\s*currentGen\s*,\s*parseSeed\s*\(\s*\)\s*\)",
        script,
    ), "Controls Enter handler does not call applyGenerator"


def test_enter_on_container_input_triggers_regeneration(
    preview_html: str,
) -> None:
    """Container panel Enter handler triggers regeneration via applyParams."""
    script = _extract_script(preview_html)
    assert re.search(
        r"handleEnterKey\s*\(\s*document\.getElementById\s*\(\s*['\"]container-panel['\"]\s*\)",
        script,
    ), "handleEnterKey not wired to container-panel"
    # Verify container handler uses applyParams (preserves user parameters)
    # Find the container-panel handler block and check it calls applyParams
    container_match = re.search(
        r"handleEnterKey\s*\(\s*document\.getElementById\s*\(\s*['\"]container-panel['\"]\s*\)\s*,\s*\(\)\s*=>\s*\{([^}]+)\}",
        script,
    )
    assert container_match, "Could not parse container-panel handler body"
    assert "applyParams" in container_match.group(1), (
        "Container Enter handler should call applyParams to preserve parameters"
    )


def test_event_delegation_covers_dynamic_inputs(preview_html: str) -> None:
    """Param panel uses event delegation via handleEnterKey for dynamic inputs."""
    script = _extract_script(preview_html)
    assert re.search(
        r"handleEnterKey\s*\(\s*document\.getElementById\s*\(\s*['\"]param-panel['\"]\s*\)",
        script,
    ), "handleEnterKey not wired to param-panel"


def test_non_enter_keys_do_not_trigger(preview_html: str) -> None:
    """handleEnterKey guards on Enter key only."""
    script = _extract_script(preview_html)
    # The function checks e.key against 'Enter' and returns early otherwise
    assert re.search(
        r"""e\.key\s*!==?\s*['"]Enter['"]""", script
    ), "No Enter key guard found in handleEnterKey"


def test_checkbox_inputs_excluded(preview_html: str) -> None:
    """handleEnterKey skips checkbox inputs."""
    script = _extract_script(preview_html)
    assert re.search(
        r"""e\.target\.type\s*===?\s*['"]checkbox['"]""", script
    ), "No checkbox exclusion in handleEnterKey"


def test_input_blurred_after_enter(preview_html: str) -> None:
    """handleEnterKey blurs the input after triggering."""
    script = _extract_script(preview_html)
    assert re.search(
        r"e\.target\.blur\s*\(\s*\)", script
    ), "Input not blurred after Enter"


def test_all_three_panels_have_enter_handlers(preview_html: str) -> None:
    """All three panels (controls, container, param) have Enter key handlers."""
    script = _extract_script(preview_html)
    panel_ids = ["controls", "container-panel", "param-panel"]
    for panel_id in panel_ids:
        pattern = (
            r"handleEnterKey\s*\(\s*document\.getElementById\s*\(\s*['\"]"
            + re.escape(panel_id)
            + r"['\"]\s*\)"
        )
        assert re.search(pattern, script), (
            f"Missing handleEnterKey for '{panel_id}'"
        )
