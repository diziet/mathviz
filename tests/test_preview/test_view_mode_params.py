"""Tests for view mode switching preserving param-editor params.

Verifies that _doGenerate updates state.generatedWith so that switching to
Dense Cloud or HD Cloud after generating with non-default params via the
editor does not reset to generator defaults. These are JS source verification
tests — the bug is client-side state management, not server behavior.
"""

import re

import pytest
from fastapi.testclient import TestClient

from mathviz.preview.server import app


_DO_GENERATE_PATTERN = re.compile(
    r"async function _doGenerate\b.*?^  \}",
    re.DOTALL | re.MULTILINE,
)


@pytest.fixture
def client() -> TestClient:
    """Return a FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def preview_html(client: TestClient) -> str:
    """Fetch the preview HTML once."""
    resp = client.get("/")
    assert resp.status_code == 200
    return resp.text


@pytest.fixture
def do_generate_body(preview_html: str) -> str:
    """Extract the _doGenerate function body from preview HTML."""
    match = _DO_GENERATE_PATTERN.search(preview_html)
    assert match is not None, "_doGenerate function not found in HTML"
    return match.group(0)


@pytest.fixture
def apply_sampling_body(preview_html: str) -> str:
    """Extract the applySamplingMode function body from preview HTML."""
    fn_match = re.search(
        r"function applySamplingMode\(body\)\s*\{([^}]+)\}",
        preview_html,
    )
    assert fn_match is not None, "applySamplingMode function not found in HTML"
    return fn_match.group(1)


class TestApplySamplingMode:
    """Verify applySamplingMode helper exists and is used by both code paths."""

    def test_helper_exists(self, preview_html: str) -> None:
        """applySamplingMode helper function is defined."""
        assert "function applySamplingMode(body)" in preview_html

    def test_helper_handles_dense(self, preview_html: str) -> None:
        """VIEW_MODE_TO_SAMPLING maps dense to post_transform."""
        assert "dense: 'post_transform'" in preview_html

    def test_helper_handles_hd_cloud(self, preview_html: str) -> None:
        """VIEW_MODE_TO_SAMPLING maps hd_cloud to resolution_scaled."""
        assert "hd_cloud: 'resolution_scaled'" in preview_html

    def test_load_from_api_uses_helper(self, preview_html: str) -> None:
        """loadFromAPI calls applySamplingMode instead of inline checks."""
        load_match = re.search(
            r"async function loadFromAPI\b.*?^\}",
            preview_html,
            re.DOTALL | re.MULTILINE,
        )
        assert load_match is not None
        assert "applySamplingMode(body)" in load_match.group(0)

    def test_do_generate_uses_helper(self, do_generate_body: str) -> None:
        """_doGenerate calls applySamplingMode to respect current view mode."""
        assert "applySamplingMode(body)" in do_generate_body


class TestDoGenerateUpdatesState:
    """Verify _doGenerate sets state.generatedWith after success."""

    def test_sets_generated_with(self, do_generate_body: str) -> None:
        """_doGenerate updates state.generatedWith after displayGenerateResult."""
        assert "state.generatedWith" in do_generate_body, (
            "_doGenerate must update state.generatedWith"
        )

    def test_sets_geometry_id(self, do_generate_body: str) -> None:
        """_doGenerate updates state.geometryId after displayGenerateResult."""
        assert "state.geometryId" in do_generate_body, (
            "_doGenerate must update state.geometryId"
        )

    def test_generated_with_includes_resolution(self, do_generate_body: str) -> None:
        """state.generatedWith in _doGenerate includes resolution field."""
        gw_match = re.search(
            r"state\.generatedWith\s*=\s*\{([^}]+)\}", do_generate_body,
        )
        assert gw_match is not None
        assert "resolution" in gw_match.group(1), (
            "state.generatedWith must include resolution"
        )

    def test_generated_with_set_after_display(self, do_generate_body: str) -> None:
        """state.generatedWith is set after displayGenerateResult, not before."""
        display_pos = do_generate_body.find("displayGenerateResult")
        gw_pos = do_generate_body.find("state.generatedWith")
        assert display_pos < gw_pos, (
            "state.generatedWith must be set after displayGenerateResult"
        )

    def test_generated_with_includes_params(self, do_generate_body: str) -> None:
        """state.generatedWith includes params from the editor."""
        gw_match = re.search(
            r"state\.generatedWith\s*=\s*\{([^}]+)\}", do_generate_body,
        )
        assert gw_match is not None
        fields = gw_match.group(1)
        assert "params" in fields
        assert "generator" in fields
        assert "seed" in fields
        assert "container" in fields

    def test_save_btn_guarded(self, do_generate_body: str) -> None:
        """Save button enable is null-guarded to avoid masking success."""
        assert "getElementById('save-btn')" in do_generate_body
        # Should not directly chain .disabled on getElementById result
        assert "getElementById('save-btn').disabled" not in do_generate_body

    def test_view_mode_handler_reads_generated_with(
        self, preview_html: str,
    ) -> None:
        """View mode change handler reads state.generatedWith for re-generate."""
        assert "state.generatedWith" in preview_html
        assert "const gw = state.generatedWith" in preview_html
        assert "loadFromAPI(gw.generator, gw.params, gw.seed" in preview_html

    def test_view_mode_handler_checks_sampling_change(
        self, preview_html: str,
    ) -> None:
        """View mode handler only re-generates when sampling pipeline changes."""
        assert "samplingChanged" in preview_html
        assert "samplingFor" in preview_html
