"""Helpers for integrating disk cache with the in-memory geometry cache."""

import logging
from typing import Any

import numpy as np

from mathviz.core.math_object import MathObject, Mesh, PointCloud
from mathviz.preview.cache import CacheEntry, GeometryCache
from mathviz.preview.disk_cache import DiskCache, DiskCacheEntry
from mathviz.preview.lod import cloud_to_binary_ply, mesh_to_glb

logger = logging.getLogger(__name__)


def load_from_disk(
    cache_key: str,
    disk_entry: DiskCacheEntry,
    disk_cache: DiskCache,
    memory_cache: GeometryCache,
) -> CacheEntry | None:
    """Load geometry from disk cache into memory cache."""
    mesh_path = disk_cache.get_mesh_path(cache_key)
    cloud_path = disk_cache.get_cloud_path(cache_key)

    mesh_model = _load_mesh_from_path(mesh_path)
    cloud_model = _load_cloud_from_path(cloud_path)

    if mesh_model is None and cloud_model is None:
        return None

    math_object = MathObject(
        mesh=mesh_model,
        point_cloud=cloud_model,
        generator_name=disk_entry.generator_name,
    )
    entry = CacheEntry(
        math_object=math_object,
        generator_name=disk_entry.generator_name,
        params=disk_entry.params,
        seed=disk_entry.seed,
        resolution_kwargs=disk_entry.resolution_kwargs,
    )
    memory_cache.put(cache_key, entry)
    return entry


def store_to_disk(
    cache_key: str,
    entry: CacheEntry,
    disk_cache: DiskCache,
    container_kwargs: dict[str, Any] | None = None,
) -> None:
    """Store a generation result to disk cache."""
    mesh_data = None
    cloud_data = None
    if entry.math_object.mesh is not None:
        mesh_data = mesh_to_glb(entry.math_object.mesh)
    if entry.math_object.point_cloud is not None:
        cloud_data = cloud_to_binary_ply(entry.math_object.point_cloud)
    disk_cache.put(
        cache_key,
        generator_name=entry.generator_name,
        params=entry.params,
        seed=entry.seed,
        resolution_kwargs=entry.resolution_kwargs,
        container_kwargs=container_kwargs or {},
        mesh_data=mesh_data,
        cloud_data=cloud_data,
    )


def _load_mesh_from_path(mesh_path: Any) -> Mesh | None:
    """Load a Mesh model from a GLB file path."""
    if mesh_path is None:
        return None
    try:
        import trimesh
        loaded = trimesh.load(str(mesh_path), file_type="glb")
        if isinstance(loaded, trimesh.Scene):
            meshes = list(loaded.geometry.values())
            raw_mesh = meshes[0] if meshes else None
        else:
            raw_mesh = loaded
        if raw_mesh is None:
            return None
        return Mesh(
            vertices=np.asarray(raw_mesh.vertices, dtype=np.float64),
            faces=np.asarray(raw_mesh.faces, dtype=np.int64),
        )
    except (OSError, ValueError, RuntimeError) as exc:
        logger.warning("Failed to load mesh from disk cache: %s", exc)
        return None


def _load_cloud_from_path(cloud_path: Any) -> PointCloud | None:
    """Load a PointCloud model from a PLY file path."""
    if cloud_path is None:
        return None
    try:
        import trimesh
        loaded = trimesh.load(str(cloud_path), file_type="ply")
        points = loaded.vertices if hasattr(loaded, 'vertices') else None
        if points is not None:
            return PointCloud(points=points)
    except (OSError, ValueError, RuntimeError) as exc:
        logger.warning("Failed to load cloud from disk cache: %s", exc)
    return None
