"""Tests for static file packaging and serving."""

import importlib.resources

import pytest
from fastapi.testclient import TestClient

from mathviz.preview.server import app


@pytest.fixture()
def client() -> TestClient:
    """Create a test client for the preview server."""
    return TestClient(app)


def test_index_html_accessible_via_importlib_resources() -> None:
    """index.html is accessible via importlib.resources from the installed package."""
    static = importlib.resources.files("mathviz").joinpath("static")
    index = static.joinpath("index.html")
    content = index.read_text(encoding="utf-8")
    assert len(content) > 0
    assert "<!DOCTYPE html>" in content


def test_preview_root_returns_200(client: TestClient) -> None:
    """Preview server GET / returns 200 with HTML content."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_html_contains_threejs_import(client: TestClient) -> None:
    """The HTML content contains a Three.js script import."""
    response = client.get("/")
    assert response.status_code == 200
    assert "three" in response.text.lower()
    assert "THREE" in response.text


def test_html_contains_canvas_container(client: TestClient) -> None:
    """The HTML content contains the app container div."""
    response = client.get("/")
    assert response.status_code == 200
    assert "canvas-container" in response.text


def test_pyproject_includes_static_html() -> None:
    """Package data glob in pyproject.toml includes static/*.html."""
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent.parent
    pyproject = (project_root / "pyproject.toml").read_text(encoding="utf-8")
    assert "static/*.html" in pyproject
    assert "static/**/*" in pyproject
