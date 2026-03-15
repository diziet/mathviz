"""Tests for Auto-Apply toggle and controls panel layout in preview UI."""

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


# --- Auto-Apply checkbox presence ---


def test_html_contains_auto_apply_checkbox(preview_html: str) -> None:
    """Preview HTML contains an Auto-Apply checkbox with id auto-apply."""
    assert 'id="auto-apply"' in preview_html
    assert 'type="checkbox"' in preview_html
    # Verify it's labelled Auto-Apply
    assert "Auto-Apply" in preview_html


# --- JS state includes autoApply ---


def test_state_includes_auto_apply(preview_html: str) -> None:
    """JS state object includes autoApply initialized to false."""
    script = _extract_script(preview_html)
    assert re.search(
        r"autoApply\s*:\s*false", script
    ), "state.autoApply not initialized to false"


# --- Auto-apply triggers regeneration on input ---


def test_auto_apply_input_listener(preview_html: str) -> None:
    """When auto-apply is enabled, input events trigger regeneration."""
    script = _extract_script(preview_html)
    # The param panel has an input event listener for auto-apply
    assert re.search(
        r"addEventListener\s*\(\s*['\"]input['\"]", script
    ), "No input event listener found for auto-apply"
    # The handler checks state.autoApply
    assert re.search(
        r"state\.autoApply", script
    ), "Handler does not check state.autoApply"
    # The handler calls applyParams
    assert re.search(
        r"applyParams\s*\(\s*\)", script
    ), "Auto-apply handler does not call applyParams"


# --- Debounce behavior ---


def test_auto_apply_uses_debounce(preview_html: str) -> None:
    """Regeneration is debounced (not fired on every input immediately)."""
    script = _extract_script(preview_html)
    # Uses setTimeout for debounce
    assert re.search(
        r"setTimeout\s*\(", script
    ), "No setTimeout found for debounce"
    # Uses clearTimeout to reset on each event
    assert re.search(
        r"clearTimeout\s*\(", script
    ), "No clearTimeout found — debounce does not reset"
    # Debounce delay is defined
    assert re.search(
        r"AUTO_APPLY_DEBOUNCE_MS\s*=\s*\d+", script
    ), "Debounce delay constant not defined"


# --- Auto-apply disabled does not trigger ---


def test_auto_apply_disabled_no_trigger(preview_html: str) -> None:
    """When auto-apply is disabled, changing inputs does not trigger."""
    script = _extract_script(preview_html)
    # The handler returns early when autoApply is false
    assert re.search(
        r"if\s*\(\s*!state\.autoApply\s*\)\s*return", script
    ), "No early return when autoApply is disabled"


# --- Panel layout: param-panel below container-panel ---


def test_param_panel_below_container_in_dom(preview_html: str) -> None:
    """Parameter section appears below container dimensions in DOM order."""
    container_pos = preview_html.find('id="container-panel"')
    param_pos = preview_html.find('id="param-panel"')
    assert container_pos > 0, "container-panel not found"
    assert param_pos > 0, "param-panel not found"
    assert container_pos < param_pos, (
        "param-panel should appear after container-panel in DOM"
    )


def test_panels_wrapped_in_left_column(preview_html: str) -> None:
    """Both panels are wrapped in a scrollable left-column container."""
    assert 'id="left-column"' in preview_html
    # left-column should contain both panels
    left_col_start = preview_html.find('id="left-column"')
    # Find the closing div for left-column by checking nesting
    assert left_col_start > 0, "left-column not found"
    # container-panel and param-panel should be inside left-column
    container_pos = preview_html.find('id="container-panel"')
    param_pos = preview_html.find('id="param-panel"')
    assert container_pos > left_col_start, (
        "container-panel not inside left-column"
    )
    assert param_pos > left_col_start, (
        "param-panel not inside left-column"
    )


# --- Controls panel scrollable ---


def test_left_column_is_scrollable(preview_html: str) -> None:
    """Controls panel is scrollable when content overflows."""
    assert re.search(
        r"#left-column\s*\{[^}]*overflow-y\s*:\s*auto", preview_html
    ), "left-column should have overflow-y: auto for scrolling"
