"""Batch generation endpoint for parallel multi-panel rendering."""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from mathviz.core.container import PlacementPolicy
from mathviz.core.math_object import MathObject
from mathviz.preview.cache import CacheEntry, compute_cache_key

logger = logging.getLogger(__name__)

router = APIRouter()

MAX_BATCH_PANELS = 16


def build_geometry_urls(
    geometry_id: str,
    math_object: MathObject,
) -> tuple[str | None, str | None]:
    """Build mesh and cloud URLs for a geometry entry."""
    mesh_url = f"/api/geometry/{geometry_id}/mesh" if math_object.mesh is not None else None
    cloud_url = f"/api/geometry/{geometry_id}/cloud" if math_object.point_cloud is not None else None
    return mesh_url, cloud_url


class BatchPanelRequest(BaseModel):
    """A single panel in a batch generation request."""

    generator: str
    params: dict[str, Any] = Field(default_factory=dict)
    seed: int = 42
    resolution: dict[str, Any] = Field(default_factory=dict)
    container: dict[str, Any] | None = None


class BatchRequest(BaseModel):
    """Request body for POST /api/generate-batch."""

    panels: list[BatchPanelRequest] = Field(max_length=MAX_BATCH_PANELS)


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
    normalized_resolutions: list[dict[str, Any] | None] = []
    jobs: list[dict[str, Any]] = []
    job_panel_indices: list[int] = []

    for i, panel in enumerate(req.panels):
        params = _normalize_params(panel.params)
        resolution = _normalize_params(panel.resolution)
        normalized_resolutions.append(resolution)

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
                    resolution_kwargs=normalized_resolutions[original_index] or {},
                )
                cache.put(panel_cache_keys[original_index], entry)
                logger.info(
                    "Batch: generated and cached panel %d as %s",
                    original_index,
                    panel_cache_keys[original_index],
                )

    # Build response in request order
    response_panels = _build_response_panels(
        req.panels, panel_cache_keys, panel_errors, cache,
    )

    timed_out = batch_result is not None and batch_result.timed_out
    return BatchResponse(panels=response_panels, timed_out=timed_out)


def _build_response_panels(
    panels: list[BatchPanelRequest],
    cache_keys: list[str],
    errors: dict[int, str],
    cache: Any,
) -> list[BatchPanelResponse]:
    """Build the response panel list in request order."""
    response_panels: list[BatchPanelResponse] = []
    for i in range(len(panels)):
        if i in errors:
            response_panels.append(BatchPanelResponse(error=errors[i]))
            continue
        entry = cache.get(cache_keys[i])
        if entry is None:
            response_panels.append(
                BatchPanelResponse(error="Generation produced no result")
            )
            continue
        mesh_url, cloud_url = build_geometry_urls(cache_keys[i], entry.math_object)
        response_panels.append(
            BatchPanelResponse(
                geometry_id=cache_keys[i],
                mesh_url=mesh_url,
                cloud_url=cloud_url,
            )
        )
    return response_panels
