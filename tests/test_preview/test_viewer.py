"""Tests for the Three.js viewer and preview CLI command."""

from pathlib import Path
from typing import Generator
from unittest.mock import patch

import pytest
import trimesh
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from mathviz.cli import app as cli_app
from mathviz.core.generator import register
from mathviz.generators.parametric.torus import TorusGenerator
from mathviz.preview.server import app, reset_cache, set_served_file


def _ensure_torus_registered() -> None:
    """Re-register the torus generator if missing from the registry."""
    import mathviz.core.generator as gen_mod

    if "torus" not in gen_mod._alias_map:
        gen_mod._discovered = True
        register(TorusGenerator)


# --- Fixtures ---


@pytest.fixture(autouse=True)
def _ensure_generators() -> Generator[None, None, None]:
    """Ensure generators are registered and state is clean."""
    _ensure_torus_registered()
    reset_cache()
    set_served_file(None)
    yield
    reset_cache()
    set_served_file(None)


@pytest.fixture
def client() -> TestClient:
    """Return a FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def runner() -> CliRunner:
    """Return a Typer CLI test runner."""
    return CliRunner()


@pytest.fixture
def stl_file(tmp_path: Path) -> Path:
    """Create a temporary STL file for testing."""
    mesh = trimesh.creation.box(extents=(1, 1, 1))
    path = tmp_path / "test_cube.stl"
    mesh.export(str(path), file_type="stl")
    return path


# --- Viewer HTML serving ---


class TestViewerHTML:
    """Tests for the / endpoint serving the Three.js viewer."""

    def test_root_serves_html(self, client: TestClient) -> None:
        """GET / returns HTML containing Three.js setup."""
        resp = client.get("/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "three" in resp.text.lower()

    def test_html_contains_orbit_controls(self, client: TestClient) -> None:
        """Viewer HTML sets up OrbitControls."""
        resp = client.get("/")
        assert "OrbitControls" in resp.text

    def test_html_contains_view_mode_toggles(self, client: TestClient) -> None:
        """Viewer HTML has shaded/wireframe/point cloud view mode options."""
        resp = client.get("/")
        html = resp.text
        assert 'value="shaded"' in html
        assert 'value="wireframe"' in html
        assert 'value="vertex"' in html

    def test_html_contains_background_toggle(self, client: TestClient) -> None:
        """Viewer HTML has a background toggle control."""
        resp = client.get("/")
        assert "toggle-bg" in resp.text

    def test_html_contains_point_size_slider(self, client: TestClient) -> None:
        """Viewer HTML has a point size slider."""
        resp = client.get("/")
        assert "point-size" in resp.text

    def test_html_contains_screenshot_button(self, client: TestClient) -> None:
        """Viewer HTML has a screenshot button."""
        resp = client.get("/")
        assert "screenshot" in resp.text.lower()

    def test_html_contains_info_display(self, client: TestClient) -> None:
        """Viewer HTML has an info panel with FPS, gen time, and geometry stats."""
        resp = client.get("/")
        html = resp.text
        assert "info-panel" in html
        assert "info-fps" in html
        assert "info-gen-time" in html

    def test_html_contains_bounding_box_toggle(self, client: TestClient) -> None:
        """Viewer HTML has a bounding box toggle."""
        resp = client.get("/")
        assert "show-bbox" in resp.text

    def test_html_parses_query_params(self, client: TestClient) -> None:
        """Viewer HTML contains code to parse URL query params."""
        resp = client.get("/")
        assert "URLSearchParams" in resp.text

    def test_html_calls_generate_endpoint(self, client: TestClient) -> None:
        """Viewer HTML contains fetch call to /api/generate."""
        resp = client.get("/")
        assert "/api/generate" in resp.text


# --- URL query param forwarding ---


class TestQueryParamForwarding:
    """Tests that URL query params are used to drive generation."""

    def test_generator_param_in_html(self, client: TestClient) -> None:
        """Viewer parses generator query param from URL."""
        resp = client.get("/?generator=torus&major_radius=2.0")
        assert resp.status_code == 200
        assert "parseQueryParams" in resp.text


# --- File serving ---


class TestFileServing:
    """Tests for serving local geometry files."""

    def test_stl_file_served_as_glb(self, client: TestClient, stl_file: Path) -> None:
        """STL files are converted to GLB for the viewer."""
        set_served_file(str(stl_file))
        resp = client.get("/api/file")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "model/gltf-binary"
        assert resp.content[:4] == b"glTF"

    def test_no_file_configured_returns_404(self, client: TestClient) -> None:
        """When no file is configured, /api/file returns 404."""
        resp = client.get("/api/file")
        assert resp.status_code == 404

    def test_ply_file_served_directly(self, client: TestClient, tmp_path: Path) -> None:
        """PLY files are served directly with correct content type."""
        ply_content = b"ply\nformat ascii 1.0\nelement vertex 1\n"
        ply_content += b"property float x\nproperty float y\nproperty float z\n"
        ply_content += b"end_header\n0.0 0.0 0.0\n"
        ply_path = tmp_path / "test.ply"
        ply_path.write_bytes(ply_content)
        set_served_file(str(ply_path))
        resp = client.get("/api/file")
        assert resp.status_code == 200
        assert "ply" in resp.headers["content-type"]

    def test_gltf_file_served_with_json_content_type(
        self, client: TestClient, tmp_path: Path,
    ) -> None:
        """GLTF files are served with model/gltf+json content type."""
        gltf_path = tmp_path / "test.gltf"
        gltf_path.write_text('{"asset":{"version":"2.0"}}', encoding="utf-8")
        set_served_file(str(gltf_path))
        resp = client.get("/api/file")
        assert resp.status_code == 200
        assert "gltf+json" in resp.headers["content-type"]


# --- Preview CLI command ---


class TestPreviewCLI:
    """Tests for the mathviz preview CLI command."""

    def test_preview_generator_no_open(self, runner: CliRunner) -> None:
        """mathviz preview torus --no-open starts server."""
        with patch("uvicorn.run") as mock_run:
            result = runner.invoke(cli_app, ["preview", "torus", "--no-open"])
            assert result.exit_code == 0
            mock_run.assert_called_once()

    def test_preview_file_no_open(self, runner: CliRunner, stl_file: Path) -> None:
        """mathviz preview <stl_file> --no-open serves the file."""
        with patch("uvicorn.run"):
            result = runner.invoke(cli_app, ["preview", str(stl_file), "--no-open"])
            assert result.exit_code == 0

    def test_preview_with_params(self, runner: CliRunner) -> None:
        """mathviz preview torus --param major_radius=2.0 passes params."""
        with patch("uvicorn.run"):
            result = runner.invoke(
                cli_app,
                ["preview", "torus", "--param", "major_radius=2.0", "--no-open"],
            )
            assert result.exit_code == 0

    def test_preview_prints_url(self, runner: CliRunner) -> None:
        """Preview command prints the URL to the console."""
        with patch("uvicorn.run"):
            result = runner.invoke(cli_app, ["preview", "torus", "--no-open"])
            assert "http://127.0.0.1:8000/" in result.output

    def test_preview_custom_port(self, runner: CliRunner) -> None:
        """mathviz preview --port 9000 uses the custom port."""
        with patch("uvicorn.run") as mock_run:
            result = runner.invoke(
                cli_app, ["preview", "torus", "--no-open", "--port", "9000"]
            )
            assert result.exit_code == 0
            _, kwargs = mock_run.call_args
            assert kwargs["port"] == 9000


# --- Server shutdown ---


class TestServerShutdown:
    """Tests for clean server shutdown behavior."""

    def test_uvicorn_run_is_called_with_correct_app(self, runner: CliRunner) -> None:
        """Preview command calls uvicorn.run with the correct app string."""
        with patch("uvicorn.run") as mock_run:
            runner.invoke(cli_app, ["preview", "torus", "--no-open"])
            args, _ = mock_run.call_args
            assert args[0] == "mathviz.preview.server:app"
