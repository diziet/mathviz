"""Batch generation endpoint for parallel multi-panel rendering."""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from mathviz.core.container import PlacementPolicy
from mathviz.preview.cache import CacheEntry, compute_cache_key

logger = logging.getLogger(__name__)

router = APIRouter()


class BatchPanelRequest(BaseModel):
    """A single panel in a batch generation request."""

    generator: str
    params: dict[str, Any] = Field(default_factory=dict)
    seed: int = 42
    resolution: dict[str, Any] = Field(default_factory=dict)
    container: Any | None = None


class BatchRequest(BaseModel):
    """Request body for POST /api/generate-batch."""

    panels: list[BatchPanelRequest]


class BatchPanelResponse(BaseModel):
    """Response for a single panel in a batch."""

    geometry_id: str | None = None
    mesh_url: str | None = None
    cloud_url: str | None = None
    error: str | None = None


class BatchResponse(BaseModel):
    """Response body for POST /api/generate-batch."""

    panels: list[BatchPanelResponse]
    timed_out: bool = False


def _get_server_helpers() -> tuple:
    """Import server helpers lazily to avoid circular imports."""
    from mathviz.preview.server import (
        ContainerParams,
        _build_container,
        _container_cache_dict,
        _normalize_params,
        get_cache,
        get_executor,
    )
    return (
        ContainerParams,
        _build_container,
        _container_cache_dict,
        _normalize_params,
        get_cache,
        get_executor,
    )


@router.post("/api/generate-batch", response_model=BatchResponse)
def generate_batch(req: BatchRequest) -> BatchResponse:
    """Run multiple panel generations in parallel."""
    if not req.panels:
        raise HTTPException(status_code=400, detail="Panels list must not be empty.")

    (
        ContainerParams,
        _build_container,
        _container_cache_dict,
        _normalize_params,
        get_cache,
        get_executor,
    ) = _get_server_helpers()

    cache = get_cache()
    executor = get_executor()

    # Check cache and collect jobs that need generation
    panel_cache_keys: list[str] = []
    jobs: list[dict[str, Any]] = []
    job_panel_indices: list[int] = []

    for i, panel in enumerate(req.panels):
        params = _normalize_params(panel.params)
        resolution = _normalize_params(panel.resolution)

        container_params = None
        if panel.container is not None:
            container_params = ContainerParams(**panel.container)

        cache_key = compute_cache_key(
            panel.generator,
            params or {},
            panel.seed,
            resolution or {},
            container_kwargs=_container_cache_dict(container_params),
        )
        panel_cache_keys.append(cache_key)

        if cache.get(cache_key) is not None:
            continue  # Already cached

        container = _build_container(container_params)
        jobs.append({
            "generator": panel.generator,
            "params": params,
            "seed": panel.seed,
            "resolution_kwargs": resolution,
            "container": container,
            "placement": PlacementPolicy(),
            "panel_index": i,
        })
        job_panel_indices.append(i)

    # Run uncached panels in parallel
    batch_result = None
    panel_errors: dict[int, str] = {}
    if jobs:
        batch_result = executor.submit_batch(jobs)

        for job_idx, panel_result in enumerate(batch_result.panels):
            original_index = job_panel_indices[job_idx]
            if panel_result.error:
                panel_errors[original_index] = panel_result.error
            elif panel_result.pipeline_result:
                panel_req = req.panels[original_index]
                entry = CacheEntry(
                    math_object=panel_result.pipeline_result.math_object,
                    generator_name=panel_req.generator,
                    params=panel_req.params,
                    seed=panel_req.seed,
                    resolution_kwargs=panel_req.resolution,
                )
                cache.put(panel_cache_keys[original_index], entry)
                logger.info(
                    "Batch: generated and cached panel %d as %s",
                    original_index,
                    panel_cache_keys[original_index],
                )

    # Build response in request order
    response_panels: list[BatchPanelResponse] = []
    for i in range(len(req.panels)):
        if i in panel_errors:
            response_panels.append(BatchPanelResponse(error=panel_errors[i]))
        else:
            entry = cache.get(panel_cache_keys[i])
            if entry is None:
                response_panels.append(
                    BatchPanelResponse(error="Generation produced no result")
                )
                continue
            obj = entry.math_object
            mesh_url = (
                f"/api/geometry/{panel_cache_keys[i]}/mesh"
                if obj.mesh is not None
                else None
            )
            cloud_url = (
                f"/api/geometry/{panel_cache_keys[i]}/cloud"
                if obj.point_cloud is not None
                else None
            )
            response_panels.append(
                BatchPanelResponse(
                    geometry_id=panel_cache_keys[i],
                    mesh_url=mesh_url,
                    cloud_url=cloud_url,
                )
            )

    timed_out = batch_result is not None and batch_result.timed_out
    return BatchResponse(panels=response_panels, timed_out=timed_out)
