"""FastAPI preview server for the Tier 1 viewer."""

import importlib.resources
import io
import logging
from pathlib import Path
from typing import Any

import trimesh
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, Response
from pydantic import BaseModel, Field, ValidationError

from mathviz.core.container import Container, PlacementPolicy
from mathviz.core.generator import GeneratorMeta, get_generator_meta, list_generators
from mathviz.pipeline.runner import run as run_pipeline
from mathviz.preview.cache import CacheEntry, GeometryCache, compute_cache_key
from mathviz.preview.lod import (
    cloud_to_binary_ply,
    decimate_mesh,
    mesh_to_glb,
    subsample_cloud,
)
from mathviz.preview.snapshot_routes import router as snapshot_router
from mathviz.preview.snapshots import save_snapshot

logger = logging.getLogger(__name__)

_cache = GeometryCache()


def get_cache() -> GeometryCache:
    """Return the global geometry cache."""
    return _cache


def reset_cache() -> None:
    """Clear the global cache. Intended for testing only."""
    _cache.clear()


app = FastAPI(title="MathViz Preview", version="0.1.0")
app.include_router(snapshot_router)

_STATIC_DIR = importlib.resources.files("mathviz").joinpath("static")
_ALLOWED_FILE_EXTENSIONS = {".stl", ".ply", ".glb", ".gltf", ".obj"}

# File path configured by the preview CLI for serving local files
_served_file_path: str | None = None


def set_served_file(path: str | None) -> None:
    """Set the file path to serve via /api/file endpoint."""
    global _served_file_path  # noqa: PLW0603
    _served_file_path = path


def get_served_file() -> str | None:
    """Return the currently configured served file path."""
    return _served_file_path


# --- Request / Response models ---


class ContainerParams(BaseModel):
    """Optional container dimensions for POST /api/generate."""

    width_mm: float = 100.0
    height_mm: float = 100.0
    depth_mm: float = 100.0
    margin_x_mm: float = 5.0
    margin_y_mm: float = 5.0
    margin_z_mm: float = 5.0


class GenerateRequest(BaseModel):
    """Request body for POST /api/generate."""

    generator: str
    params: dict[str, Any] = Field(default_factory=dict)
    seed: int = 42
    resolution: dict[str, Any] = Field(default_factory=dict)
    container: ContainerParams | None = None


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


class SnapshotRequest(BaseModel):
    """Request body for POST /api/snapshots."""

    generator: str
    params: dict[str, Any] = Field(default_factory=dict)
    seed: int = 42
    container: ContainerParams | None = None
    geometry_id: str


class SnapshotResponse(BaseModel):
    """Response body for POST /api/snapshots."""

    snapshot_id: str


# --- Helpers ---


def _generator_info_from_meta(meta: GeneratorMeta) -> GeneratorInfo:
    """Convert a GeneratorMeta to a GeneratorInfo response model."""
    return GeneratorInfo(
        name=meta.name,
        category=meta.category,
        aliases=meta.aliases,
        description=meta.description,
        resolution_params=meta.resolution_params,
    )


def _generator_not_found(name: str) -> HTTPException:
    """Return a 404 HTTPException for an unknown generator."""
    return HTTPException(
        status_code=404,
        detail=f"Generator {name!r} not found. Use GET /api/generators to list available.",
    )


def _normalize_params(params: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize empty dict to None for consistent pipeline/cache behavior."""
    return params if params else None


# --- Endpoints ---


@app.get("/api/generators")
def list_all_generators() -> list[GeneratorInfo]:
    """Return metadata for all registered generators."""
    return [_generator_info_from_meta(m) for m in list_generators()]


@app.get("/api/generators/{name}")
def get_generator_details(name: str) -> GeneratorInfo:
    """Return metadata for a single generator by name or alias."""
    try:
        meta = get_generator_meta(name)
    except KeyError:
        raise _generator_not_found(name)
    return _generator_info_from_meta(meta)


@app.get("/api/generators/{name}/params")
def get_generator_params(name: str) -> dict[str, Any]:
    """Return default parameters, resolution defaults, and descriptions."""
    try:
        meta = get_generator_meta(name)
    except KeyError:
        raise _generator_not_found(name)

    instance = meta.generator_class.create(resolved_name=name)
    params = instance.get_default_params()
    resolution = instance.get_default_resolution()
    descriptions = dict(meta.resolution_params)

    return {
        "params": params,
        "resolution": resolution,
        "descriptions": descriptions,
    }


def _build_container(container_params: ContainerParams | None) -> Container:
    """Build a Container from request params, or return the default."""
    if container_params is None:
        return Container.with_uniform_margin()
    try:
        return Container(
            width_mm=container_params.width_mm,
            height_mm=container_params.height_mm,
            depth_mm=container_params.depth_mm,
            margin_x_mm=container_params.margin_x_mm,
            margin_y_mm=container_params.margin_y_mm,
            margin_z_mm=container_params.margin_z_mm,
        )
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _container_cache_dict(container_params: ContainerParams | None) -> dict[str, float]:
    """Return a dict of container params for cache key computation."""
    if container_params is None:
        return ContainerParams().model_dump()
    return container_params.model_dump()


@app.post("/api/generate", response_model=GenerateResponse)
def generate_geometry(req: GenerateRequest) -> GenerateResponse:
    """Run the pipeline and cache the result."""
    container = _build_container(req.container)

    params = _normalize_params(req.params)
    resolution = _normalize_params(req.resolution)

    cache_key = compute_cache_key(
        req.generator,
        params or {},
        req.seed,
        resolution or {},
        container_kwargs=_container_cache_dict(req.container),
    )
    cache = get_cache()
    entry = cache.get(cache_key)

    if entry is None:
        try:
            result = run_pipeline(
                req.generator,
                params=params,
                seed=req.seed,
                resolution_kwargs=resolution,
                container=container,
                placement=PlacementPolicy(),
            )
        except KeyError:
            raise _generator_not_found(req.generator)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

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


@app.get("/api/file")
def serve_local_file() -> Response:
    """Serve the configured local geometry file, converting STL to GLB if needed."""
    served = get_served_file()
    if served is None:
        raise HTTPException(status_code=404, detail="No file configured for serving.")

    resolved = Path(served).resolve()
    if not resolved.is_file():
        raise HTTPException(status_code=404, detail="Configured file not found on disk.")

    suffix = resolved.suffix.lower()
    if suffix not in _ALLOWED_FILE_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

    if suffix == ".stl":
        return _serve_stl_as_glb(resolved)
    if suffix == ".ply":
        return FileResponse(str(resolved), media_type="application/x-ply")
    if suffix == ".glb":
        return FileResponse(str(resolved), media_type="model/gltf-binary")
    if suffix == ".gltf":
        return FileResponse(str(resolved), media_type="model/gltf+json")
    return FileResponse(str(resolved), media_type="application/octet-stream")


def _serve_stl_as_glb(stl_path: Path) -> Response:
    """Convert an STL file to GLB and return it."""
    mesh = trimesh.load(str(stl_path), file_type="stl")
    buf = io.BytesIO()
    mesh.export(buf, file_type="glb")
    return Response(content=buf.getvalue(), media_type="model/gltf-binary")


@app.post("/api/snapshots", response_model=SnapshotResponse)
def create_snapshot(req: SnapshotRequest) -> SnapshotResponse:
    """Save current geometry and configuration as a snapshot."""
    entry = get_cache().get(req.geometry_id)
    if entry is None:
        raise HTTPException(
            status_code=400,
            detail="No generated geometry found for the given geometry_id.",
        )

    container_dict = (
        req.container.model_dump() if req.container else ContainerParams().model_dump()
    )

    snapshot_id, snapshot_path = save_snapshot(
        math_object=entry.math_object,
        generator=req.generator,
        params=req.params,
        seed=req.seed,
        container=container_dict,
        geometry_id=req.geometry_id,
    )

    return SnapshotResponse(snapshot_id=snapshot_id)


@app.get("/", response_class=HTMLResponse)
def serve_viewer() -> HTMLResponse:
    """Serve the Three.js viewer HTML."""
    index_resource = _STATIC_DIR.joinpath("index.html")
    try:
        content = index_resource.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Viewer HTML not found.")
    return HTMLResponse(content=content)
