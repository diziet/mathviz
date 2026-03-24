"""Tests for the /buildbanner.json endpoint."""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from mathviz.preview.server import app


@pytest.fixture
def client() -> TestClient:
    """Return a FastAPI test client."""
    return TestClient(app)


class TestBuildBannerEndpoint:
    """Tests for GET /buildbanner.json."""

    def test_returns_200(self, client: TestClient) -> None:
        """Endpoint returns 200 OK."""
        resp = client.get("/buildbanner.json")
        assert resp.status_code == 200

    def test_returns_json_with_required_fields(self, client: TestClient) -> None:
        """Response contains branch, commit, uptime, and generators."""
        data = client.get("/buildbanner.json").json()
        assert "branch" in data
        assert "commit" in data
        assert "uptime" in data
        assert "generators" in data

    def test_generators_is_integer(self, client: TestClient) -> None:
        """Generator count is a non-negative integer."""
        data = client.get("/buildbanner.json").json()
        assert isinstance(data["generators"], int)
        assert data["generators"] >= 0

    def test_uptime_is_string(self, client: TestClient) -> None:
        """Uptime is a human-readable string."""
        data = client.get("/buildbanner.json").json()
        assert isinstance(data["uptime"], str)
        assert len(data["uptime"]) > 0

    @patch("mathviz.preview.build_banner._cached_commit", "")
    @patch("mathviz.preview.build_banner._cached_branch", "")
    def test_handles_missing_git(self, client: TestClient) -> None:
        """Returns empty strings when git is unavailable."""
        data = client.get("/buildbanner.json").json()
        assert data["branch"] == ""
        assert data["commit"] == ""

    @patch("mathviz.preview.build_banner._cached_branch", "main")
    @patch("mathviz.preview.build_banner._cached_commit", "abc1234")
    def test_returns_mocked_git_values(
        self,
        client: TestClient,
    ) -> None:
        """Returns git values from the cached module-level constants."""
        data = client.get("/buildbanner.json").json()
        assert data["branch"] == "main"
        assert data["commit"] == "abc1234"


class TestFormatUptime:
    """Tests for the _format_uptime helper."""

    def test_seconds_only(self) -> None:
        """Formats durations under a minute."""
        from mathviz.preview.build_banner import _format_uptime

        assert _format_uptime(45) == "45s"

    def test_minutes_and_seconds(self) -> None:
        """Formats durations under an hour."""
        from mathviz.preview.build_banner import _format_uptime

        assert _format_uptime(125) == "2m 5s"

    def test_hours_minutes_seconds(self) -> None:
        """Formats durations with hours."""
        from mathviz.preview.build_banner import _format_uptime

        assert _format_uptime(3661) == "1h 1m 1s"
