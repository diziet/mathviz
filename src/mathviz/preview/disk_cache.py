"""Disk-based generation cache for the preview server."""

import hashlib
import json
import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path.home() / ".mathviz" / "cache"
DEFAULT_MAX_SIZE_BYTES = 5 * 1024 * 1024 * 1024  # 5 GB
METADATA_FILENAME = "metadata.json"


def _get_cache_dir() -> Path:
    """Return the cache directory from env or default."""
    env_val = os.environ.get("MATHVIZ_CACHE_DIR")
    if env_val:
        return Path(env_val)
    return DEFAULT_CACHE_DIR


def _get_max_size_bytes() -> int:
    """Return the max cache size from env or default."""
    env_val = os.environ.get("MATHVIZ_CACHE_MAX_SIZE")
    if env_val:
        try:
            return int(env_val)
        except ValueError:
            logger.warning("Invalid MATHVIZ_CACHE_MAX_SIZE=%r, using default", env_val)
    return DEFAULT_MAX_SIZE_BYTES


@dataclass
class DiskCacheEntry:
    """Metadata for a cached generation on disk."""

    cache_key: str
    generator_name: str
    params: dict[str, Any]
    seed: int
    resolution_kwargs: dict[str, Any]
    container_kwargs: dict[str, Any]
    timestamp: float
    has_mesh: bool
    has_cloud: bool


class DiskCache:
    """Disk-based cache for generated geometry files."""

    def __init__(
        self,
        cache_dir: Path | None = None,
        max_size_bytes: int | None = None,
    ) -> None:
        """Initialize the disk cache."""
        self._cache_dir = cache_dir if cache_dir is not None else _get_cache_dir()
        self._max_size_bytes = (
            max_size_bytes if max_size_bytes is not None else _get_max_size_bytes()
        )
        self._lock = threading.Lock()

    @property
    def cache_dir(self) -> Path:
        """Return the cache directory path."""
        return self._cache_dir

    def _entry_dir(self, cache_key: str) -> Path:
        """Return the directory for a specific cache entry."""
        return self._cache_dir / cache_key

    def _ensure_dir(self) -> None:
        """Create the cache directory if it doesn't exist."""
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, cache_key: str) -> DiskCacheEntry | None:
        """Look up a cache entry by key. Returns metadata if found."""
        with self._lock:
            entry_dir = self._entry_dir(cache_key)
            meta_path = entry_dir / METADATA_FILENAME
            if not meta_path.is_file():
                return None
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                # Touch to update access time for LRU eviction
                meta_path.touch()
                return DiskCacheEntry(
                    cache_key=cache_key,
                    generator_name=meta["generator_name"],
                    params=meta.get("params", {}),
                    seed=meta["seed"],
                    resolution_kwargs=meta.get("resolution_kwargs", {}),
                    container_kwargs=meta.get("container_kwargs", {}),
                    timestamp=meta["timestamp"],
                    has_mesh=meta.get("has_mesh", False),
                    has_cloud=meta.get("has_cloud", False),
                )
            except (json.JSONDecodeError, KeyError, OSError) as exc:
                logger.warning("Corrupt or evicted cache entry %s: %s", cache_key, exc)
                return None

    def _get_file_path(self, cache_key: str, filename: str) -> Path | None:
        """Return the path to a cached file if it exists, or None."""
        path = self._entry_dir(cache_key) / filename
        return path if path.is_file() else None

    def get_mesh_path(self, cache_key: str) -> Path | None:
        """Return the path to the cached mesh file, or None."""
        return self._get_file_path(cache_key, "mesh.glb")

    def get_cloud_path(self, cache_key: str) -> Path | None:
        """Return the path to the cached cloud file, or None."""
        return self._get_file_path(cache_key, "cloud.ply")

    def put(
        self,
        cache_key: str,
        generator_name: str,
        params: dict[str, Any],
        seed: int,
        resolution_kwargs: dict[str, Any],
        container_kwargs: dict[str, Any],
        mesh_data: bytes | None = None,
        cloud_data: bytes | None = None,
    ) -> None:
        """Store a generation result on disk."""
        with self._lock:
            self._ensure_dir()
            entry_dir = self._entry_dir(cache_key)
            entry_dir.mkdir(parents=True, exist_ok=True)

            if mesh_data is not None:
                (entry_dir / "mesh.glb").write_bytes(mesh_data)
            if cloud_data is not None:
                (entry_dir / "cloud.ply").write_bytes(cloud_data)

            meta = {
                "generator_name": generator_name,
                "params": params,
                "seed": seed,
                "resolution_kwargs": resolution_kwargs,
                "container_kwargs": container_kwargs,
                "timestamp": time.time(),
                "has_mesh": mesh_data is not None,
                "has_cloud": cloud_data is not None,
            }
            (entry_dir / METADATA_FILENAME).write_text(
                json.dumps(meta, indent=2, default=str),
                encoding="utf-8",
            )
            logger.info("Disk cache: stored %s", cache_key)

            self._evict_if_needed()

    def clear(self) -> int:
        """Remove all cached entries. Returns count of entries removed."""
        with self._lock:
            if not self._cache_dir.is_dir():
                return 0
            count = 0
            for child in self._cache_dir.iterdir():
                if child.is_dir() and (child / METADATA_FILENAME).is_file():
                    shutil.rmtree(child)
                    count += 1
            logger.info("Disk cache: cleared %d entries", count)
            return count

    def total_size_bytes(self) -> int:
        """Return total size of all cached files in bytes."""
        if not self._cache_dir.is_dir():
            return 0
        total = 0
        for path in self._cache_dir.rglob("*"):
            if path.is_file():
                total += path.stat().st_size
        return total

    def _evict_if_needed(self) -> None:
        """Evict oldest entries until total size is under the limit."""
        total = self.total_size_bytes()
        if total <= self._max_size_bytes:
            return

        entries = self._list_entries_by_age()
        for entry_dir, _mtime in entries:
            if total <= self._max_size_bytes:
                break
            entry_size = sum(
                f.stat().st_size for f in entry_dir.rglob("*") if f.is_file()
            )
            shutil.rmtree(entry_dir)
            total -= entry_size
            logger.info("Disk cache: evicted %s (freed %d bytes)", entry_dir.name, entry_size)

    def _list_entries_by_age(self) -> list[tuple[Path, float]]:
        """List cache entry dirs sorted by modification time (oldest first)."""
        entries: list[tuple[Path, float]] = []
        if not self._cache_dir.is_dir():
            return entries
        for child in self._cache_dir.iterdir():
            meta_path = child / METADATA_FILENAME
            if child.is_dir() and meta_path.is_file():
                mtime = meta_path.stat().st_mtime
                entries.append((child, mtime))
        entries.sort(key=lambda x: x[1])
        return entries
