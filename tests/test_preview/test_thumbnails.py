"""Tests for generator thumbnail endpoint and persistent disk cache."""

import io
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from mathviz.core.generator import register
from mathviz.core.math_object import MathObject, Mesh
from mathviz.generators.parametric.torus import TorusGenerator
from mathviz.preview.server import app
from mathviz.preview.thumbnails import (
    THUMBNAIL_SIZE,
    THUMBNAILS_DIR_ENV_VAR,
    get_thumbnail_path,
    get_thumbnails_dir,
)


def _ensure_torus_registered() -> None:
    """Re-register the torus generator if missing."""
    import mathviz.core.generator as gen_mod

    if "torus" not in gen_mod._alias_map:
        gen_mod._discovered = True
        register(TorusGenerator)


def _make_test_math_object() -> MathObject:
    """Create a simple mesh MathObject for testing."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array(
        [[-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0]],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
    return MathObject(mesh=Mesh(vertices=verts, faces=faces), generator_name="torus")


def _fake_render_to_png(obj: Any, path: Path, **kwargs: Any) -> Path:
    """Write a valid PNG file that Pillow can open."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (THUMBNAIL_SIZE, THUMBNAIL_SIZE), color=(128, 128, 128))
    img.save(path, "PNG")
    return path


def _mock_pipeline_result() -> MagicMock:
    """Create a mock PipelineResult with a MathObject."""
    result = MagicMock()
    result.math_object = _make_test_math_object()
    return result


@pytest.fixture(autouse=True)
def _setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Register generators and redirect thumbnail dir to temp."""
    _ensure_torus_registered()
    monkeypatch.setenv(THUMBNAILS_DIR_ENV_VAR, str(tmp_path / "thumbnails"))


@pytest.fixture
def client() -> TestClient:
    """Return a FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def _mock_rendering() -> Generator[MagicMock, None, None]:
    """Stub out pipeline + renderer for thumbnail tests."""
    with (
        patch(
            "mathviz.preview.thumbnails.run_pipeline",
            return_value=_mock_pipeline_result(),
        ) as mock_run,
        patch(
            "mathviz.preview.thumbnails.render_to_png",
            side_effect=_fake_render_to_png,
        ),
    ):
        yield mock_run


class TestThumbnailEndpoint:
    """Tests for GET /api/generators/{name}/thumbnail."""

    def test_thumbnail_route_returns_webp(
        self, _mock_rendering: MagicMock, client: TestClient,
    ) -> None:
        """Thumbnail endpoint returns WebP content type."""
        resp = client.get("/api/generators/torus/thumbnail")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/webp"

    def test_thumbnail_generates_webp(
        self, _mock_rendering: MagicMock, client: TestClient,
    ) -> None:
        """Generated thumbnail is a valid WebP file."""
        resp = client.get("/api/generators/torus/thumbnail")
        assert resp.status_code == 200
        img = Image.open(io.BytesIO(resp.content))
        assert img.format == "WEBP"

    def test_thumbnail_dimensions(
        self, _mock_rendering: MagicMock, client: TestClient,
    ) -> None:
        """Output image is 472x472 pixels."""
        resp = client.get("/api/generators/torus/thumbnail")
        assert resp.status_code == 200
        img = Image.open(io.BytesIO(resp.content))
        assert img.size == (THUMBNAIL_SIZE, THUMBNAIL_SIZE)

    def test_unknown_generator_returns_404(self, client: TestClient) -> None:
        """Thumbnail endpoint returns 404 for unknown generators."""
        resp = client.get("/api/generators/nonexistent_xyz_999/thumbnail")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_thumbnail_cached_on_disk(
        self, _mock_rendering: MagicMock, client: TestClient,
    ) -> None:
        """Second call returns cached file without re-rendering."""
        resp1 = client.get("/api/generators/torus/thumbnail")
        assert resp1.status_code == 200
        assert _mock_rendering.call_count == 1

        resp2 = client.get("/api/generators/torus/thumbnail")
        assert resp2.status_code == 200
        # Pipeline should NOT be called again — served from disk cache
        assert _mock_rendering.call_count == 1

    def test_different_view_modes_produce_different_files(
        self, _mock_rendering: MagicMock, client: TestClient,
    ) -> None:
        """Different view_mode values produce different cached files."""
        resp_points = client.get("/api/generators/torus/thumbnail?view_mode=points")
        resp_shaded = client.get("/api/generators/torus/thumbnail?view_mode=shaded")

        assert resp_points.status_code == 200
        assert resp_shaded.status_code == 200

        points_path = get_thumbnail_path("torus", "points")
        shaded_path = get_thumbnail_path("torus", "shaded")

        assert points_path != shaded_path
        assert points_path.is_file()
        assert shaded_path.is_file()

    def test_invalid_view_mode_returns_400(self, client: TestClient) -> None:
        """Invalid view_mode returns 400."""
        resp = client.get("/api/generators/torus/thumbnail?view_mode=invalid")
        assert resp.status_code == 400
        assert "Invalid view_mode" in resp.json()["detail"]

    def test_thumbnail_missing_pyvista_returns_501(self, client: TestClient) -> None:
        """When render unavailable, endpoint returns 501 not 500."""
        with patch(
            "mathviz.preview.thumbnails.run_pipeline",
            side_effect=ImportError("No module named 'pyvista'"),
        ):
            resp = client.get("/api/generators/torus/thumbnail")
        assert resp.status_code == 501
        assert "unavailable" in resp.json()["detail"].lower()


class TestDeleteThumbnails:
    """Tests for DELETE /api/thumbnails."""

    def test_clears_all_cached_thumbnails(
        self, _mock_rendering: MagicMock, client: TestClient,
    ) -> None:
        """DELETE /api/thumbnails removes all cached thumbnail files."""
        client.get("/api/generators/torus/thumbnail?view_mode=points")
        client.get("/api/generators/torus/thumbnail?view_mode=shaded")

        assert get_thumbnail_path("torus", "points").is_file()
        assert get_thumbnail_path("torus", "shaded").is_file()

        resp = client.delete("/api/thumbnails")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["thumbnails_removed"] == 2

        assert not get_thumbnail_path("torus", "points").is_file()
        assert not get_thumbnail_path("torus", "shaded").is_file()


class TestBatchThumbnails:
    """Tests for GET /api/generators/thumbnails."""

    def test_returns_urls_for_all_generators(self, client: TestClient) -> None:
        """Batch endpoint returns thumbnail URLs for all registered generators."""
        resp = client.get("/api/generators/thumbnails")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, dict)
        assert "torus" in data
        assert "/api/generators/torus/thumbnail" in data["torus"]
        assert "view_mode=points" in data["torus"]


class TestThumbnailDiskPath:
    """Tests for thumbnail disk storage paths."""

    def test_stored_at_expected_path(
        self, _mock_rendering: MagicMock, client: TestClient,
    ) -> None:
        """Thumbnails are stored at <thumbnails_dir>/<view_mode>/<name>.webp."""
        client.get("/api/generators/torus/thumbnail?view_mode=wireframe")

        expected = get_thumbnails_dir() / "wireframe" / "torus.webp"
        assert expected.is_file()
