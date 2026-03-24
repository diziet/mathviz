"""Tests for the buildbanner middleware integration."""

import importlib

import pytest
from fastapi.testclient import TestClient

from mathviz.preview.server import app


@pytest.fixture
def client() -> TestClient:
    """Return a FastAPI test client."""
    return TestClient(app)


class TestBuildBannerMiddleware:
    """Tests for GET /buildbanner.json via BuildBannerMiddleware."""

    def test_returns_200(self, client: TestClient) -> None:
        """Endpoint returns 200 OK."""
        resp = client.get("/buildbanner.json")
        assert resp.status_code == 200

    def test_returns_json_with_required_fields(self, client: TestClient) -> None:
        """Response contains sha, branch, server_started, repo_url."""
        data = client.get("/buildbanner.json").json()
        assert "sha" in data
        assert "branch" in data
        assert "server_started" in data
        assert "repo_url" in data

    def test_extras_include_app_name(self, client: TestClient) -> None:
        """Extras callback adds app_name field."""
        data = client.get("/buildbanner.json").json()
        assert data.get("app_name") == "MathViz"

    def test_extras_include_generators(self, client: TestClient) -> None:
        """Extras callback adds generators count."""
        data = client.get("/buildbanner.json").json()
        assert isinstance(data.get("generators"), int)
        assert data["generators"] >= 0

    def test_content_type_is_json(self, client: TestClient) -> None:
        """Response Content-Type is application/json."""
        resp = client.get("/buildbanner.json")
        assert "application/json" in resp.headers["content-type"]


class TestCustomBuildBannerRemoved:
    """Verify the custom build_banner.py module no longer exists."""

    def test_custom_module_not_importable(self) -> None:
        """Custom build_banner.py cannot be imported."""
        with pytest.raises(ImportError):
            importlib.import_module("mathviz.preview.build_banner")


class TestBannerRendersInBrowser:
    """Verify the HTML page includes the buildbanner script tag."""

    def test_index_includes_buildbanner_script(self, client: TestClient) -> None:
        """The viewer page includes a script tag for buildbanner.js with the JSON endpoint."""
        resp = client.get("/")
        assert resp.status_code == 200
        html = resp.text
        assert 'src="/static/buildbanner.js"' in html
        assert 'data-endpoint="/buildbanner.json"' in html

    def test_buildbanner_json_has_repo_url_for_link(self, client: TestClient) -> None:
        """The JSON endpoint returns a repo_url that the banner uses as the GitHub link."""
        data = client.get("/buildbanner.json").json()
        assert "repo_url" in data
        assert isinstance(data["repo_url"], str)
