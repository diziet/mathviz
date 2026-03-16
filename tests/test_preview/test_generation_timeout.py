"""Tests for generation timeout and cancel mechanism."""

import os
import pickle
import time
import threading
from typing import Any
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from mathviz.core.container import Container, PlacementPolicy
from mathviz.core.generator import register
from mathviz.core.math_object import MathObject, Mesh
from mathviz.generators.parametric.torus import TorusGenerator
from mathviz.preview.executor import (
    DEFAULT_TIMEOUT_SECONDS,
    GenerationExecutor,
    get_timeout_seconds,
)
from mathviz.preview.server import app, get_executor, reset_cache
import mathviz.preview.server as server_mod


def _ensure_torus_registered() -> None:
    """Re-register the torus generator if missing from the registry."""
    import mathviz.core.generator as gen_mod

    if "torus" not in gen_mod._alias_map:
        gen_mod._discovered = True
        register(TorusGenerator)


@pytest.fixture(autouse=True)
def _setup() -> None:
    """Ensure generators are registered and cache is clean."""
    _ensure_torus_registered()
    reset_cache()


@pytest.fixture
def client() -> TestClient:
    """Return a FastAPI test client."""
    return TestClient(app)


# --- Timeout returns HTTP 504 ---


class TestGenerationTimeout:
    """Tests for generation timeout behavior."""

    def test_timeout_returns_504(self, client: TestClient) -> None:
        """Generation that exceeds timeout returns HTTP 504."""
        def slow_submit(*args: Any, **kwargs: Any) -> None:
            from concurrent.futures import TimeoutError
            raise TimeoutError("timed out")

        with patch.object(server_mod._executor, "submit", side_effect=slow_submit):
            resp = client.post(
                "/api/generate",
                json={"generator": "torus", "seed": 42},
            )
        assert resp.status_code == 504

    def test_timeout_response_has_error_message(self, client: TestClient) -> None:
        """Response body contains a meaningful error message."""
        from concurrent.futures import TimeoutError

        with patch.object(server_mod._executor, "submit", side_effect=TimeoutError()):
            resp = client.post(
                "/api/generate",
                json={"generator": "torus", "seed": 42},
            )
        data = resp.json()
        assert "detail" in data
        assert "timed out" in data["detail"].lower()

    def test_timeout_configurable_via_env(self) -> None:
        """Timeout is configurable via MATHVIZ_GENERATION_TIMEOUT."""
        with patch.dict(os.environ, {"MATHVIZ_GENERATION_TIMEOUT": "60"}):
            assert get_timeout_seconds() == 60

    def test_timeout_default_is_300(self) -> None:
        """Default timeout is 300 seconds."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove the var if set
            os.environ.pop("MATHVIZ_GENERATION_TIMEOUT", None)
            assert get_timeout_seconds() == DEFAULT_TIMEOUT_SECONDS
            assert DEFAULT_TIMEOUT_SECONDS == 300

    def test_invalid_env_uses_default(self) -> None:
        """Invalid env var value falls back to default."""
        with patch.dict(os.environ, {"MATHVIZ_GENERATION_TIMEOUT": "abc"}):
            assert get_timeout_seconds() == DEFAULT_TIMEOUT_SECONDS

    def test_custom_timeout_passed_to_executor(self, client: TestClient) -> None:
        """Request with custom timeout passes that value to the executor."""
        captured_kwargs: dict[str, Any] = {}

        original_submit = server_mod._executor.submit

        def capturing_submit(*args: Any, **kwargs: Any) -> Any:
            captured_kwargs.update(kwargs)
            return original_submit(*args, **kwargs)

        with patch.object(server_mod._executor, "submit", side_effect=capturing_submit):
            resp = client.post(
                "/api/generate",
                json={"generator": "torus", "seed": 42, "timeout": 60},
            )
        assert resp.status_code == 200
        assert captured_kwargs.get("timeout_override") == 60

    def test_no_timeout_field_falls_back_to_default(self, client: TestClient) -> None:
        """Request without timeout field falls back to env var / 300s default."""
        captured_kwargs: dict[str, Any] = {}

        original_submit = server_mod._executor.submit

        def capturing_submit(*args: Any, **kwargs: Any) -> Any:
            captured_kwargs.update(kwargs)
            return original_submit(*args, **kwargs)

        with patch.object(server_mod._executor, "submit", side_effect=capturing_submit):
            resp = client.post(
                "/api/generate",
                json={"generator": "torus", "seed": 42},
            )
        assert resp.status_code == 200
        assert captured_kwargs.get("timeout_override") is None

    def test_custom_timeout_in_504_error_message(self, client: TestClient) -> None:
        """504 error message reflects the custom timeout, not the server default."""
        from concurrent.futures import TimeoutError

        with patch.object(server_mod._executor, "submit", side_effect=TimeoutError()):
            resp = client.post(
                "/api/generate",
                json={"generator": "torus", "seed": 42, "timeout": 45},
            )
        assert resp.status_code == 504
        assert "45 seconds" in resp.json()["detail"]


# --- Cancel endpoint ---


class TestCancelEndpoint:
    """Tests for POST /api/generate/cancel."""

    def test_cancel_returns_404_when_nothing_running(
        self, client: TestClient
    ) -> None:
        """POST /api/generate/cancel returns 404 when nothing is running."""
        resp = client.post("/api/generate/cancel")
        assert resp.status_code == 404
        data = resp.json()
        assert "detail" in data

    def test_cancel_returns_200_when_generation_running(
        self, client: TestClient
    ) -> None:
        """POST /api/generate/cancel returns 200 when generation is running."""
        with patch.object(server_mod._executor, "cancel", return_value=True):
            resp = client.post("/api/generate/cancel")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "cancelled"

    def test_cancel_does_not_leave_orphan_processes(self) -> None:
        """Cancelled generation does not leave orphan processes."""
        executor = GenerationExecutor()
        # After cancel + pool termination, pool should be None
        executor._pool = MagicMock()
        executor._current_task = MagicMock()
        executor._current_task.future = MagicMock()
        executor.cancel()
        executor._pool_was_terminated = True
        # Pool should have been shut down (set to None)
        assert executor._pool is None


# --- Normal generation still works ---


class TestNormalGeneration:
    """Tests that normal generation works with timeout in place."""

    def test_fast_generation_succeeds(self, client: TestClient) -> None:
        """Normal (fast) generation still works correctly with timeout."""
        resp = client.post(
            "/api/generate",
            json={"generator": "torus", "seed": 42},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "geometry_id" in data
        assert data["mesh_url"] is not None or data["cloud_url"] is not None


# --- UI cancel button ---


class TestPreviewUI:
    """Tests for cancel button presence in preview HTML."""

    def test_preview_html_has_cancel_button(self, client: TestClient) -> None:
        """Preview HTML shows a cancel button."""
        resp = client.get("/")
        assert resp.status_code == 200
        html = resp.text
        assert 'id="cancel-btn"' in html

    def test_preview_html_has_elapsed_timer(self, client: TestClient) -> None:
        """Preview HTML shows an elapsed time element."""
        resp = client.get("/")
        html = resp.text
        assert 'id="loading-elapsed"' in html

    def test_preview_html_has_cancel_endpoint_call(
        self, client: TestClient
    ) -> None:
        """Preview HTML contains JS that calls the cancel endpoint."""
        resp = client.get("/")
        html = resp.text
        assert "/api/generate/cancel" in html

    def test_preview_html_has_timeout_input(self, client: TestClient) -> None:
        """Preview HTML contains a timeout input control."""
        resp = client.get("/")
        html = resp.text
        assert 'id="timeout-input"' in html

    def test_preview_html_persists_timeout_in_localstorage(
        self, client: TestClient
    ) -> None:
        """Preview HTML uses localStorage to persist the timeout setting."""
        resp = client.get("/")
        html = resp.text
        assert "mathviz_generation_timeout" in html
        assert "localStorage" in html


# --- Picklability of pipeline args ---


class TestPicklability:
    """Ensure objects sent across process boundary survive pickling."""

    def test_container_survives_pickle_roundtrip(self) -> None:
        """Container can be pickled and unpickled for subprocess use."""
        container = Container.with_uniform_margin()
        restored = pickle.loads(pickle.dumps(container))
        assert restored.width_mm == container.width_mm
        assert restored.height_mm == container.height_mm
        assert restored.depth_mm == container.depth_mm

    def test_placement_policy_survives_pickle_roundtrip(self) -> None:
        """PlacementPolicy can be pickled and unpickled for subprocess use."""
        policy = PlacementPolicy()
        restored = pickle.loads(pickle.dumps(policy))
        assert type(restored) is PlacementPolicy
