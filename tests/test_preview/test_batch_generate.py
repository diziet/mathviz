"""Tests for POST /api/generate-batch endpoint."""

from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from mathviz.core.generator import register
from mathviz.generators.parametric.torus import TorusGenerator
from mathviz.preview.server import app, get_cache, reset_cache


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


def _make_panel(
    seed: int = 42,
    generator: str = "torus",
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a panel request dict."""
    return {
        "generator": generator,
        "params": params or {"major_radius": 1.0, "minor_radius": 0.4},
        "seed": seed,
        "resolution": {"grid_resolution": 16},
    }


class TestBatchGenerate:
    """Tests for POST /api/generate-batch."""

    def test_four_panels_returns_four_results(self, client: TestClient) -> None:
        """POST /api/generate-batch with 4 panels returns 4 geometry results."""
        panels = [_make_panel(seed=i) for i in range(4)]
        resp = client.post("/api/generate-batch", json={"panels": panels})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["panels"]) == 4
        for panel in data["panels"]:
            assert panel["geometry_id"] is not None
            assert panel["mesh_url"] is not None
            assert panel["error"] is None

    def test_batch_uses_parallel_execution(self, client: TestClient) -> None:
        """Batch generation submits jobs to a ProcessPoolExecutor."""
        import mathviz.preview.executor as executor_mod

        original_ensure = executor_mod.GenerationExecutor._ensure_batch_pool

        pool_workers: list[int] = []
        submit_calls: list[int] = []

        def _tracking_ensure(self_inner: Any, worker_count: int) -> Any:
            pool_workers.append(worker_count)
            pool = original_ensure(self_inner, worker_count)
            original_submit = pool.submit

            def _tracking_submit(*args: Any, **kwargs: Any) -> Any:
                submit_calls.append(1)
                return original_submit(*args, **kwargs)

            pool.submit = _tracking_submit  # type: ignore[method-assign]
            return pool

        with patch.object(
            executor_mod.GenerationExecutor,
            "_ensure_batch_pool",
            _tracking_ensure,
        ):
            panels = [_make_panel(seed=i) for i in range(4)]
            resp = client.post("/api/generate-batch", json={"panels": panels})

        assert resp.status_code == 200
        assert len(resp.json()["panels"]) == 4
        # Verify pool was created with multiple workers
        assert len(pool_workers) >= 1
        assert pool_workers[0] > 1, "Batch pool should use multiple workers"
        # Verify all 4 panels were submitted to the pool
        assert len(submit_calls) == 4

    def test_failed_panel_does_not_crash_batch(self, client: TestClient) -> None:
        """Failed panels return error without crashing the batch."""
        panels = [
            _make_panel(seed=1),
            {"generator": "nonexistent_gen_xyz", "params": {}, "seed": 2},
            _make_panel(seed=3),
        ]
        resp = client.post("/api/generate-batch", json={"panels": panels})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["panels"]) == 3
        # Panel 0 and 2 should succeed
        assert data["panels"][0]["geometry_id"] is not None
        assert data["panels"][0]["error"] is None
        # Panel 1 should have an error
        assert data["panels"][1]["error"] is not None
        assert data["panels"][1]["geometry_id"] is None
        # Panel 2 should succeed
        assert data["panels"][2]["geometry_id"] is not None
        assert data["panels"][2]["error"] is None

    def test_response_order_matches_request(self, client: TestClient) -> None:
        """Response order matches request order."""
        panels = [_make_panel(seed=i) for i in range(3)]
        resp = client.post("/api/generate-batch", json={"panels": panels})
        assert resp.status_code == 200
        data = resp.json()

        # Each panel should have a unique geometry_id
        ids = [p["geometry_id"] for p in data["panels"]]
        assert len(set(ids)) == 3

    def test_batch_respects_timeout(self, client: TestClient) -> None:
        """Batch respects the generation timeout."""
        from mathviz.preview.executor import (
            BATCH_TIMEOUT_ERROR,
            BatchPanelResult,
            BatchResult,
        )
        import mathviz.preview.server as server_mod

        timed_out_result = BatchResult(
            panels=[
                BatchPanelResult(index=0, error=BATCH_TIMEOUT_ERROR),
                BatchPanelResult(index=1, error=BATCH_TIMEOUT_ERROR),
            ],
            timed_out=True,
        )

        with patch.object(
            server_mod._executor, "submit_batch", return_value=timed_out_result
        ):
            panels = [_make_panel(seed=1), _make_panel(seed=2)]
            resp = client.post("/api/generate-batch", json={"panels": panels})

        assert resp.status_code == 200
        data = resp.json()
        assert data["timed_out"] is True
        for panel in data["panels"]:
            assert panel["error"] is not None

    def test_empty_panels_returns_400(self, client: TestClient) -> None:
        """Empty panels list returns 400."""
        resp = client.post("/api/generate-batch", json={"panels": []})
        assert resp.status_code == 400
        assert "empty" in resp.json()["detail"].lower()

    def test_single_panel_batch_works(self, client: TestClient) -> None:
        """Single-panel batch works the same as regular generate."""
        panel = _make_panel(seed=42)
        resp = client.post("/api/generate-batch", json={"panels": [panel]})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["panels"]) == 1
        assert data["panels"][0]["geometry_id"] is not None
        assert data["panels"][0]["mesh_url"] is not None
        assert data["panels"][0]["error"] is None

    def test_cached_panels_skip_generation(self, client: TestClient) -> None:
        """Already-cached panels are not regenerated."""
        panel = _make_panel(seed=42)

        # First call generates
        resp1 = client.post("/api/generate-batch", json={"panels": [panel]})
        assert resp1.status_code == 200
        gid1 = resp1.json()["panels"][0]["geometry_id"]

        # Second call should use cache
        resp2 = client.post("/api/generate-batch", json={"panels": [panel]})
        assert resp2.status_code == 200
        gid2 = resp2.json()["panels"][0]["geometry_id"]
        assert gid1 == gid2
