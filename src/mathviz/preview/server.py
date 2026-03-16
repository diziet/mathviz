"""FastAPI preview server for the Tier 1 viewer."""

import importlib.resources
import io
import logging
from concurrent.futures import CancelledError, TimeoutError
from pathlib import Path
from typing import Any, Literal

import trimesh
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError, field_validator

from mathviz.core.container import Container, PlacementPolicy
from mathviz.core.generator import GeneratorMeta, get_generator_meta, list_generators
from mathviz.preview.cache import CacheEntry, GeometryCache, compute_cache_key
from mathviz.preview.cache_integration import load_from_disk, store_to_disk
from mathviz.preview.disk_cache import DiskCache
from mathviz.preview.executor import GenerationExecutor, MAX_TIMEOUT_SECONDS, get_timeout_seconds
from mathviz.preview.lod import (
    cloud_to_binary_ply,
    decimate_mesh,
    mesh_to_glb,
    subsample_cloud,
)
from mathviz.preview.batch_routes import router as batch_router
from mathviz.preview.snapshot_routes import router as snapshot_router
from mathviz.preview.thumbnail_routes import router as thumbnail_router
from mathviz.preview.snapshots import save_snapshot

logger = logging.getLogger(__name__)

_cache = GeometryCache()
_disk_cache: DiskCache | None = None
_executor = GenerationExecutor()


def get_cache() -> GeometryCache:
    """Return the global geometry cache."""
    return _cache


def get_disk_cache() -> DiskCache:
    """Return the global disk cache, creating it on first access."""
    global _disk_cache  # noqa: PLW0603
    if _disk_cache is None:
        _disk_cache = DiskCache()
    return _disk_cache


def get_executor() -> GenerationExecutor:
    """Return the global generation executor."""
    return _executor


def reset_cache() -> None:
    """Clear the global memory cache. Intended for testing only."""
    _cache.clear()


def set_disk_cache(dc: DiskCache) -> None:
    """Replace the global disk cache. Intended for testing only."""
    global _disk_cache  # noqa: PLW0603
    _disk_cache = dc


app = FastAPI(title="MathViz Preview", version="0.1.0")
app.include_router(batch_router)
app.include_router(snapshot_router)
app.include_router(thumbnail_router)

_STATIC_DIR = importlib.resources.files("mathviz").joinpath("static")
_STATIC_PATH = Path(str(_STATIC_DIR))
if _STATIC_PATH.is_dir():
    app.mount("/static", StaticFiles(directory=str(_STATIC_PATH)), name="static")
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


def _container_default(field_name: str) -> float:
    """Get a Container field's default value."""
    return Container.model_fields[field_name].default


class ContainerParams(BaseModel):
    """Optional container dimensions for POST /api/generate."""

    width_mm: float = _container_default("width_mm")
    height_mm: float = _container_default("height_mm")
    depth_mm: float = _container_default("depth_mm")
    margin_x_mm: float = _container_default("margin_x_mm")
    margin_y_mm: float = _container_default("margin_y_mm")
    margin_z_mm: float = _container_default("margin_z_mm")


class GenerateRequest(BaseModel):
    """Request body for POST /api/generate."""

    generator: str
    params: dict[str, Any] = Field(default_factory=dict)
    seed: int = 42
    resolution: dict[str, Any] = Field(default_factory=dict)
    container: ContainerParams | None = None
    force: bool = False
    timeout: int | None = Field(default=None, gt=0, le=MAX_TIMEOUT_SECONDS, description="Per-request timeout in seconds")
    # Maps to UI view_mode "dense" — JS sends sampling="post_transform"
    # when the user selects the Dense Cloud view mode.
    # "resolution_scaled" maps to "HD Cloud" — density scales with resolution.
    sampling: Literal["default", "post_transform", "resolution_scaled", "edge"] = "default"
    max_samples: int | None = Field(
        default=None, ge=1000, description="Override max sample count for dense/HD cloud",
    )



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


class CameraPosition(BaseModel):
    """3D position vector for camera state."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


class CameraState(BaseModel):
    """Camera position, target, and zoom level."""

    position: CameraPosition = Field(default_factory=CameraPosition)
    target: CameraPosition = Field(default_factory=CameraPosition)
    zoom: float = 1.0


class StretchState(BaseModel):
    """Per-axis scale factors."""

    x: float = 1.0
    y: float = 1.0
    z: float = 1.0


class UiState(BaseModel):
    """Display and control state saved with a snapshot."""

    camera: CameraState = Field(default_factory=CameraState)
    # "dense" maps to GenerateRequest.sampling="post_transform" at request time.
    # "hd_cloud" maps to GenerateRequest.sampling="resolution_scaled".
    view_mode: Literal["points", "shaded", "wireframe", "crystal", "dense", "hd_cloud", "edge_cloud", "colormap"] = "points"
    stretch: StretchState = Field(default_factory=StretchState)
    camera_lock: Literal["off", "render", "full"] = "render"
    show_bbox: bool = True
    show_axes: bool = False
    light_bg: bool = False
    point_size: float = 0.1

    @field_validator("point_size")
    @classmethod
    def clamp_point_size(cls, v: float) -> float:
        """Clamp point_size to slider range for backwards compat with old snapshots."""
        return max(0.02, min(0.25, v))


class SnapshotRequest(BaseModel):
    """Request body for POST /api/snapshots."""

    generator: str
    params: dict[str, Any] = Field(default_factory=dict)
    seed: int = 42
    container: ContainerParams | None = None
    geometry_id: str
    thumbnail: str | None = None
    ui_state: UiState | None = None


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


def _derive_param_range(value: Any) -> dict[str, float] | None:
    """Derive an exploration range from a parameter's default value."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        if value > 0:
            return {"min": 1, "max": max(value * 2, 10), "step": 1}
        if value == 0:
            return {"min": 0, "max": 10, "step": 1}
        return {"min": value * 2, "max": abs(value) * 2, "step": 1}
    if isinstance(value, float):
        if value > 0:
            step = max(round(value * 0.1, 6), 1e-6)
            return {"min": round(value * 0.25, 6), "max": round(value * 2.0, 6), "step": step}
        if value == 0.0:
            return {"min": -1.0, "max": 1.0, "step": 0.1}
        step = max(round(abs(value) * 0.1, 6), 1e-6)
        return {"min": round(value * 2.0, 6), "max": round(abs(value) * 2.0, 6), "step": step}
    return None


@app.get("/api/generators/{name}/param-ranges")
def get_generator_param_ranges(name: str) -> dict[str, dict[str, float]]:
    """Return exploration ranges for each parameter of a generator."""
    try:
        meta = get_generator_meta(name)
    except KeyError:
        raise _generator_not_found(name)

    instance = meta.generator_class.create(resolved_name=name)
    explicit_ranges = instance.get_param_ranges()
    defaults = instance.get_default_params()

    ranges: dict[str, dict[str, float]] = {}
    for param_name, default_value in defaults.items():
        if param_name in explicit_ranges:
            ranges[param_name] = explicit_ranges[param_name]
        else:
            derived = _derive_param_range(default_value)
            if derived is not None:
                ranges[param_name] = derived

    return ranges


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


@app.post("/api/generate")
def generate_geometry(req: GenerateRequest) -> Response:
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
        sampling=req.sampling,
        max_samples=req.max_samples,
    )
    cache = get_cache()
    disk_cache = get_disk_cache()
    container_dict = _container_cache_dict(req.container)

    entry, cache_status = _resolve_cached_entry(
        cache_key, req.force, cache, disk_cache,
    )
    if entry is None:
        is_post_transform = req.sampling == "post_transform"
        is_resolution_scaled = req.sampling == "resolution_scaled"
        is_edge = req.sampling == "edge"
        entry = _run_generation(
            req, params, resolution, container,
            post_transform_sampling=is_post_transform,
            resolution_scaled_sampling=is_resolution_scaled,
            edge_sampling=is_edge,
            max_samples=req.max_samples,
        )
        cache.put(cache_key, entry)
        store_to_disk(cache_key, entry, disk_cache, container_kwargs=container_dict)
        cache_status = "MISS"
        logger.info("Generated and cached geometry %s", cache_key)

    from mathviz.preview.batch_routes import build_geometry_urls

    mesh_url, cloud_url = build_geometry_urls(cache_key, entry.math_object)
    body = GenerateResponse(
        geometry_id=cache_key, mesh_url=mesh_url, cloud_url=cloud_url,
    )
    return Response(
        content=body.model_dump_json(),
        media_type="application/json",
        headers={"X-Cache": cache_status},
    )


def _resolve_cached_entry(
    cache_key: str,
    force: bool,
    cache: GeometryCache,
    disk_cache: DiskCache,
) -> tuple[CacheEntry | None, str]:
    """Check memory then disk cache. Returns (entry, cache_status)."""
    if force:
        return None, "MISS"

    entry = cache.get(cache_key)
    if entry is not None:
        logger.info("Serving geometry %s from memory cache", cache_key)
        return entry, "HIT"

    disk_entry = disk_cache.get(cache_key)
    if disk_entry is not None:
        entry = load_from_disk(cache_key, disk_entry, disk_cache, cache)
        if entry is not None:
            logger.info("Serving geometry %s from disk cache", cache_key)
            return entry, "HIT"

    return None, "MISS"


def _run_generation(
    req: GenerateRequest,
    params: dict[str, Any] | None,
    resolution: dict[str, Any] | None,
    container: Container,
    *,
    post_transform_sampling: bool = False,
    resolution_scaled_sampling: bool = False,
    edge_sampling: bool = False,
    max_samples: int | None = None,
) -> CacheEntry:
    """Execute the pipeline and return a CacheEntry."""
    timeout = req.timeout if req.timeout is not None else get_timeout_seconds()
    try:
        result = _executor.submit(
            req.generator,
            params=params,
            seed=req.seed,
            resolution_kwargs=resolution,
            container=container,
            placement=PlacementPolicy(),
            timeout_override=timeout,
            post_transform_sampling=post_transform_sampling,
            resolution_scaled_sampling=resolution_scaled_sampling,
            edge_sampling=edge_sampling,
            max_samples=max_samples,
        )
    except KeyError:
        raise _generator_not_found(req.generator)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except TimeoutError:
        logger.error("Generation timed out after %d seconds", timeout)
        raise HTTPException(
            status_code=504,
            detail=f"Generation timed out after {timeout} seconds",
        )
    except CancelledError:
        raise HTTPException(status_code=499, detail="Generation cancelled")
    return CacheEntry(
        math_object=result.math_object,
        generator_name=req.generator,
        params=req.params,
        seed=req.seed,
        resolution_kwargs=req.resolution,
    )


@app.post("/api/cache/clear")
def clear_cache() -> dict[str, Any]:
    """Clear all cached entries (memory and disk)."""
    get_cache().clear()
    count = get_disk_cache().clear()
    return {"status": "ok", "entries_removed": count}


@app.post("/api/generate/cancel")
def cancel_generation() -> dict[str, str]:
    """Cancel the currently running generation, if any."""
    if _executor.cancel():
        logger.info("Generation cancelled by user")
        return {"status": "cancelled"}
    raise HTTPException(status_code=404, detail="No generation in progress")


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

    thumbnail_png: bytes | None = None
    if req.thumbnail is not None:
        try:
            import base64

            thumbnail_png = base64.b64decode(req.thumbnail, validate=True)
        except (ValueError, base64.binascii.Error):
            raise HTTPException(
                status_code=400,
                detail="Invalid base64 thumbnail data.",
            )
        if not thumbnail_png[:8].startswith(b"\x89PNG"):
            raise HTTPException(
                status_code=400,
                detail="Invalid thumbnail: not a PNG image.",
            )

    container_dict = (
        req.container.model_dump() if req.container else ContainerParams().model_dump()
    )

    try:
        snapshot_id, _snapshot_path = save_snapshot(
            math_object=entry.math_object,
            generator=req.generator,
            params=req.params,
            seed=req.seed,
            container=container_dict,
            geometry_id=req.geometry_id,
            thumbnail_png=thumbnail_png,
            ui_state=req.ui_state.model_dump() if req.ui_state else None,
        )
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save snapshot: {exc}",
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
