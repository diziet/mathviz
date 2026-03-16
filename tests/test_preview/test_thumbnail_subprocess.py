"""Tests for subprocess-based thumbnail generation and render-thumbnail CLI."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from mathviz.core.generator import register
from mathviz.generators.parametric.torus import TorusGenerator
from mathviz.preview.thumbnails import (
    THUMBNAIL_SIZE,
    THUMBNAILS_DIR_ENV_VAR,
    ThumbnailSubprocessError,
    ThumbnailTimeoutError,
    generate_thumbnail_subprocess,
    get_or_generate_thumbnail,
    get_thumbnail_path,
    get_thumbnails_dir,
)


def _ensure_torus_registered() -> None:
    """Re-register the torus generator if missing."""
    import mathviz.core.generator as gen_mod

    if "torus" not in gen_mod._alias_map:
        gen_mod._discovered = True
        register(TorusGenerator)


@pytest.fixture(autouse=True)
def _setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Register generators and redirect thumbnail dir to temp."""
    _ensure_torus_registered()
    monkeypatch.setenv(THUMBNAILS_DIR_ENV_VAR, str(tmp_path / "thumbnails"))


def _place_fake_webp(generator_name: str, view_mode: str) -> Path:
    """Write a valid WebP file at the expected cache path."""
    path = get_thumbnail_path(generator_name, view_mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (THUMBNAIL_SIZE, THUMBNAIL_SIZE), color=(64, 64, 64))
    img.save(path, "webp")
    return path


def _mock_subprocess_success(generator_name: str, view_mode: str) -> MagicMock:
    """Create a mock subprocess.run that writes a fake WebP and returns rc=0."""
    def side_effect(cmd, **kwargs):
        _place_fake_webp(generator_name, view_mode)
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
            side_effect=_mock_subprocess_success("torus", "points"),
        ):
            path = generate_thumbnail_subprocess("torus", "points")

        assert path.is_file()
        img = Image.open(path)
        assert img.format == "WEBP"
        assert img.size == (THUMBNAIL_SIZE, THUMBNAIL_SIZE)


class TestThumbnailCachedSkipsSubprocess:
    """Test that cached thumbnails skip subprocess spawning."""

    def test_cached_skips_subprocess(self) -> None:
        """Cached thumbnail returns immediately without spawning a subprocess."""
        _place_fake_webp("torus", "points")

        with patch("mathviz.preview.thumbnails.subprocess.run") as mock_run:
            path = get_or_generate_thumbnail("torus", "points")

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
                generate_thumbnail_subprocess("torus", "points")


class TestThumbnailSubprocessTimeout:
    """Test that subprocess timeout is handled gracefully."""

    def test_subprocess_timeout_raises(self) -> None:
        """Long-running generation is killed after timeout."""
        with patch(
            "mathviz.preview.thumbnails.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd=["test"], timeout=5),
        ):
            with pytest.raises(ThumbnailTimeoutError, match="timed out"):
                generate_thumbnail_subprocess("torus", "points", timeout=5)


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
