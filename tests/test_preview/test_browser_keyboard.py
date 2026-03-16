"""Tests for generator browser keyboard navigation (Task 125)."""

from collections.abc import Generator

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
    """Fetch preview HTML for assertion tests."""
    resp = client.get("/")
    assert resp.status_code == 200
    return resp.text


class TestCmdKOpens:
    """Cmd+K opens the browser modal."""

    def test_cmd_k_handler_present(self, preview_html: str) -> None:
        """Cmd+K keyboard shortcut handler exists."""
        assert "metaKey" in preview_html or "ctrlKey" in preview_html
        assert "openBrowser" in preview_html

    def test_cmd_k_toggles_browser(self, preview_html: str) -> None:
        """Cmd+K opens when closed and closes when open."""
        assert "key === 'k'" in preview_html or "key==='k'" in preview_html
        assert "closeBrowser" in preview_html


class TestNumberKeyCategories:
    """Number keys select categories by position."""

    def test_number_key_handler(self, preview_html: str) -> None:
        """Number key handler calls handleBrowserDigit."""
        assert "handleBrowserDigit" in preview_html
        assert "parseInt(e.key" in preview_html

    def test_select_by_number_function(self, preview_html: str) -> None:
        """selectBrowserItemByNumber converts number to index."""
        assert "selectBrowserItemByNumber" in preview_html
        # 0 maps to index 9 (10th item)
        assert "num === 0 ? 9 : num - 1" in preview_html


class TestNumberKeyGenerators:
    """Number keys select generators within a category."""

    def test_number_keys_work_in_generator_view(
        self, preview_html: str
    ) -> None:
        """Number keys activate items in both category and generator views."""
        assert "activateBrowserItem" in preview_html
        assert "selectBrowserItemByNumber" in preview_html

    def test_number_keys_skip_when_search_focused(
        self, preview_html: str
    ) -> None:
        """Number key shortcuts are skipped when search input is focused."""
        assert "document.activeElement === browserSearch" in preview_html


class TestTwoDigitInput:
    """Two-digit input (e.g. 1 2 within 500ms) selects item 12."""

    def test_digit_buffer_exists(self, preview_html: str) -> None:
        """Digit buffer accumulates keystrokes."""
        assert "browserDigitBuffer" in preview_html

    def test_digit_timeout_configured(self, preview_html: str) -> None:
        """500ms timeout for two-digit input."""
        assert "DIGIT_TIMEOUT_MS" in preview_html
        assert "500" in preview_html

    def test_two_digit_combination(self, preview_html: str) -> None:
        """Two digits combine into a single number."""
        assert "browserDigitBuffer.length >= 2" in preview_html
        assert "parseInt(browserDigitBuffer, 10)" in preview_html

    def test_single_digit_timeout_fires(self, preview_html: str) -> None:
        """Single digit fires after timeout elapses."""
        assert "setTimeout" in preview_html
        assert "DIGIT_TIMEOUT_MS" in preview_html


class TestArrowKeyNavigation:
    """Arrow keys move focus through the grid."""

    def test_arrow_key_handler(self, preview_html: str) -> None:
        """Arrow keys call handleBrowserArrowKey."""
        assert "handleBrowserArrowKey" in preview_html
        assert "ArrowUp" in preview_html
        assert "ArrowDown" in preview_html
        assert "ArrowLeft" in preview_html
        assert "ArrowRight" in preview_html

    def test_grid_column_detection(self, preview_html: str) -> None:
        """Grid navigation detects number of columns."""
        assert "getBrowserGridColumns" in preview_html

    def test_arrow_wrapping(self, preview_html: str) -> None:
        """Arrow navigation wraps at grid edges."""
        # ArrowRight wraps: (idx + 1) % cards.length
        assert "% cards.length" in preview_html


class TestEnterActivation:
    """Enter opens focused category or loads focused generator."""

    def test_enter_activates_focused(self, preview_html: str) -> None:
        """Enter key activates the focused browser item."""
        assert "e.key === 'Enter'" in preview_html or "key==='Enter'" in preview_html
        assert "activateBrowserItem" in preview_html
        assert "browserFocusedIndex" in preview_html

    def test_enter_requires_focus(self, preview_html: str) -> None:
        """Enter only activates when an item is focused (index >= 0)."""
        assert "browserFocusedIndex >= 0" in preview_html


class TestEscapeNavigation:
    """Escape goes back from category detail, closes from category grid."""

    def test_escape_closes_from_categories(self, preview_html: str) -> None:
        """Escape closes modal from category grid."""
        assert "Escape" in preview_html
        assert "closeBrowser" in preview_html

    def test_escape_goes_back_from_generators(
        self, preview_html: str
    ) -> None:
        """Escape goes back from generator detail to categories."""
        assert "browserGoBack" in preview_html
        assert "browsing_generators_in_category" in preview_html


class TestBackspaceNavigation:
    """Backspace goes back from category detail to category grid."""

    def test_backspace_goes_back(self, preview_html: str) -> None:
        """Backspace triggers browserGoBack from generator view."""
        assert "Backspace" in preview_html
        assert "browserGoBack" in preview_html

    def test_backspace_only_from_generators(self, preview_html: str) -> None:
        """Backspace only navigates back when in generator detail."""
        assert "browsing_generators_in_category" in preview_html


class TestSearchInputBypass:
    """Keyboard navigation does not trigger while search input is focused."""

    def test_search_focus_check(self, preview_html: str) -> None:
        """Shortcuts check if search input is active element."""
        assert "document.activeElement === browserSearch" in preview_html

    def test_backspace_skips_when_search_focused(
        self, preview_html: str
    ) -> None:
        """Backspace handler checks activeElement !== browserSearch."""
        assert "document.activeElement !== browserSearch" in preview_html


class TestFocusIndicator:
    """Focus indicator is visible on the currently highlighted card."""

    def test_focus_class_in_css(self, preview_html: str) -> None:
        """CSS defines browser-focused class with visible styling."""
        assert "browser-focused" in preview_html
        assert "box-shadow" in preview_html

    def test_focus_class_applied_by_js(self, preview_html: str) -> None:
        """JS adds/removes browser-focused class on cards."""
        assert "classList.add('browser-focused')" in preview_html
        assert "classList.remove('browser-focused')" in preview_html

    def test_focus_state_tracked(self, preview_html: str) -> None:
        """browserFocusedIndex tracks the focused card."""
        assert "browserFocusedIndex" in preview_html

    def test_focus_scrolls_into_view(self, preview_html: str) -> None:
        """Focused card scrolls into view."""
        assert "scrollIntoView" in preview_html
