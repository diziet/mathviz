"""Tests for the buildbanner middleware integration."""

from pathlib import Path

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

    def test_custom_module_deleted(self) -> None:
        """Custom build_banner.py is deleted."""
        custom_path = Path(__file__).parent.parent.parent / "src" / "mathviz" / "preview" / "build_banner.py"
        assert not custom_path.exists(), f"Custom build_banner.py still exists at {custom_path}"
