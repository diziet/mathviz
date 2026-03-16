"""Tests for subprocess-based thumbnail generation and render-thumbnail CLI."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from mathviz.preview.thumbnails import (
    THUMBNAIL_SIZE,
    ThumbnailSubprocessError,
    ThumbnailTimeoutError,
    generate_thumbnail_subprocess,
    get_or_generate_thumbnail,
)
from tests.test_preview.conftest import create_fake_thumbnail


@pytest.fixture(autouse=True)
def _setup(thumbnail_setup: None) -> None:
    """Use shared thumbnail setup fixture."""


def _mock_subprocess_success(generator_name: str, view_mode: str) -> MagicMock:
    """Create a mock subprocess.run that writes a fake WebP and returns rc=0."""
    def side_effect(cmd, **kwargs):
        create_fake_thumbnail(generator_name, view_mode)
        result = MagicMock()
        result.returncode = 0
        result.stdout = ""
        result.stderr = ""
        return result
    return side_effect


class TestThumbnailSubprocessGeneratesWebp:
    """Test that subprocess generation produces a valid WebP file."""

    def test_subprocess_generates_webp(self) -> None:
        """Subprocess produces a valid WebP file in the cache directory."""
        with patch(
            "mathviz.preview.thumbnails.subprocess.run",
            side_effect=_mock_subprocess_success("torus", "vertex"),
        ):
            path = generate_thumbnail_subprocess("torus", "vertex")

        assert path.is_file()
        img = Image.open(path)
        assert img.format == "WEBP"
        assert img.size == (THUMBNAIL_SIZE, THUMBNAIL_SIZE)


class TestThumbnailCachedSkipsSubprocess:
    """Test that cached thumbnails skip subprocess spawning."""

    def test_cached_skips_subprocess(self) -> None:
        """Cached thumbnail returns immediately without spawning a subprocess."""
        create_fake_thumbnail("torus", "vertex")

        with patch("mathviz.preview.thumbnails.subprocess.run") as mock_run:
            path = get_or_generate_thumbnail("torus", "vertex")

        mock_run.assert_not_called()
        assert path.is_file()


class TestThumbnailSubprocessFailure:
    """Test that subprocess failures raise ThumbnailSubprocessError."""

    def test_subprocess_failure_raises(self) -> None:
        """Non-zero exit raises ThumbnailSubprocessError, not server crash."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Error: some rendering failure"

        with patch(
            "mathviz.preview.thumbnails.subprocess.run",
            return_value=mock_result,
        ):
            with pytest.raises(ThumbnailSubprocessError, match="rendering failure"):
                generate_thumbnail_subprocess("torus", "vertex")


class TestThumbnailSubprocessTimeout:
    """Test that subprocess timeout is handled gracefully."""

    def test_subprocess_timeout_raises(self) -> None:
        """Long-running generation is killed after timeout."""
        with patch(
            "mathviz.preview.thumbnails.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd=["test"], timeout=5),
        ):
            with pytest.raises(ThumbnailTimeoutError, match="timed out"):
                generate_thumbnail_subprocess("torus", "vertex", timeout=5)


class TestThumbnailSubprocessEndpoint:
    """Test that the HTTP endpoint returns 503 on subprocess failures."""

    @pytest.fixture
    def client(self):
        """Return a FastAPI test client."""
        from fastapi.testclient import TestClient
        from mathviz.preview.server import app
        return TestClient(app)

    def test_subprocess_failure_returns_503(self, client) -> None:
        """Non-zero subprocess exit returns 503, not server crash."""
        with patch(
            "mathviz.preview.thumbnails.generate_thumbnail_subprocess",
            side_effect=ThumbnailSubprocessError("render failed"),
        ):
            resp = client.get("/api/generators/torus/thumbnail")
        assert resp.status_code == 503
        assert "failed" in resp.json()["detail"].lower()

    def test_subprocess_timeout_returns_503(self, client) -> None:
        """Subprocess timeout returns 503."""
        with patch(
            "mathviz.preview.thumbnails.generate_thumbnail_subprocess",
            side_effect=ThumbnailTimeoutError("timed out"),
        ):
            resp = client.get("/api/generators/torus/thumbnail")
        assert resp.status_code == 503
        assert "timed out" in resp.json()["detail"].lower()
