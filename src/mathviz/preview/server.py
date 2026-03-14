"""FastAPI preview server for the Tier 1 viewer."""

import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field

from mathviz.core.container import Container, PlacementPolicy
from mathviz.core.generator import get_generator, get_generator_meta, list_generators
from mathviz.pipeline.runner import run as run_pipeline
from mathviz.preview.cache import CacheEntry, GeometryCache, compute_cache_key
from mathviz.preview.lod import (
    cloud_to_binary_ply,
    decimate_mesh,
    mesh_to_glb,
    subsample_cloud,
)

logger = logging.getLogger(__name__)

_cache = GeometryCache()


def get_cache() -> GeometryCache:
    """Return the global geometry cache."""
    return _cache


def reset_cache(max_entries: int = 64) -> None:
    """Replace the global cache (for testing)."""
    global _cache
    _cache = GeometryCache(max_entries=max_entries)


app = FastAPI(title="MathViz Preview", version="0.1.0")


# --- Request / Response models ---


class GenerateRequest(BaseModel):
    """Request body for POST /api/generate."""

    generator: str
    params: dict[str, Any] = Field(default_factory=dict)
    seed: int = 42
    resolution: dict[str, Any] = Field(default_factory=dict)


class GenerateResponse(BaseModel):
    """Response body for POST /api/generate."""

    geometry_id: str
    mesh_url: str | None = None
    cloud_url: str | None = None


class GeneratorInfo(BaseModel):
    """Generator metadata returned by listing endpoints."""

    name: str
    category: str
    aliases: list[str]
    description: str
    resolution_params: dict[str, str]


# --- Endpoints ---


@app.get("/api/generators")
def list_all_generators() -> list[GeneratorInfo]:
    """Return metadata for all registered generators."""
    return [
        GeneratorInfo(
            name=m.name,
            category=m.category,
            aliases=m.aliases,
            description=m.description,
            resolution_params=m.resolution_params,
        )
        for m in list_generators()
    ]


@app.get("/api/generators/{name}")
def get_generator_details(name: str) -> GeneratorInfo:
    """Return metadata for a single generator by name or alias."""
    try:
        meta = get_generator_meta(name)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Generator {name!r} not found. Use GET /api/generators to list available.",
        )
    return GeneratorInfo(
        name=meta.name,
        category=meta.category,
        aliases=meta.aliases,
        description=meta.description,
        resolution_params=meta.resolution_params,
    )


@app.post("/api/generate", response_model=GenerateResponse)
def generate_geometry(req: GenerateRequest) -> GenerateResponse:
    """Run the pipeline and cache the result."""
    try:
        get_generator(req.generator)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Generator {req.generator!r} not found. "
            "Use GET /api/generators to list available.",
        )

    cache_key = compute_cache_key(req.generator, req.params, req.seed, req.resolution)
    cache = get_cache()
    entry = cache.get(cache_key)

    if entry is None:
        result = run_pipeline(
            req.generator,
            params=req.params or None,
            seed=req.seed,
            resolution_kwargs=req.resolution or None,
            container=Container.with_uniform_margin(),
            placement=PlacementPolicy(),
        )
        entry = CacheEntry(
            math_object=result.math_object,
            generator_name=req.generator,
            params=req.params,
            seed=req.seed,
            resolution_kwargs=req.resolution,
        )
        cache.put(cache_key, entry)
        logger.info("Generated and cached geometry %s", cache_key)
    else:
        logger.info("Serving geometry %s from cache", cache_key)

    obj = entry.math_object
    mesh_url = f"/api/geometry/{cache_key}/mesh" if obj.mesh is not None else None
    cloud_url = f"/api/geometry/{cache_key}/cloud" if obj.point_cloud is not None else None

    return GenerateResponse(geometry_id=cache_key, mesh_url=mesh_url, cloud_url=cloud_url)


@app.get("/api/geometry/{geometry_id}/mesh")
def get_mesh(
    geometry_id: str,
    lod: str = Query(default="preview", pattern="^(preview|full)$"),
) -> Response:
    """Return mesh as GLB binary, optionally decimated for preview."""
    entry = get_cache().get(geometry_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Geometry not found in cache.")

    mesh = entry.math_object.mesh
    if mesh is None:
        raise HTTPException(status_code=404, detail="This geometry has no mesh.")

    if lod == "preview":
        mesh = decimate_mesh(mesh)

    data = mesh_to_glb(mesh)
    return Response(content=data, media_type="model/gltf-binary")


@app.get("/api/geometry/{geometry_id}/cloud")
def get_cloud(
    geometry_id: str,
    lod: str = Query(default="preview", pattern="^(preview|full)$"),
) -> Response:
    """Return point cloud as binary PLY, optionally subsampled for preview."""
    entry = get_cache().get(geometry_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Geometry not found in cache.")

    cloud = entry.math_object.point_cloud
    if cloud is None:
        raise HTTPException(status_code=404, detail="This geometry has no point cloud.")

    if lod == "preview":
        cloud = subsample_cloud(cloud)

    data = cloud_to_binary_ply(cloud)
    return Response(content=data, media_type="application/x-ply")
