"""LRU geometry cache for the preview server."""

import hashlib
import json
import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import numpy as np

from mathviz.core.math_object import MathObject

logger = logging.getLogger(__name__)

DEFAULT_MAX_ENTRIES = 64


@dataclass
class CacheEntry:
    """A cached geometry result with its generation metadata."""

    math_object: MathObject
    generator_name: str
    params: dict[str, Any]
    seed: int
    resolution_kwargs: dict[str, Any]


def _serialize_value(obj: Any) -> Any:
    """Convert non-JSON-serializable values to deterministic representations."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Cannot serialize {type(obj).__name__} for cache key")


def compute_cache_key(
    generator_name: str,
    params: dict[str, Any],
    seed: int,
    resolution_kwargs: dict[str, Any],
    container_kwargs: dict[str, float] | None = None,
    sampling: str = "default",
    max_samples: int | None = None,
) -> str:
    """Compute a deterministic cache key from generation parameters."""
    key_dict: dict[str, Any] = {
        "generator": generator_name,
        "params": params,
        "seed": seed,
        "resolution": resolution_kwargs,
        "container": container_kwargs or {},
    }
    if sampling != "default":
        key_dict["sampling"] = sampling
    if max_samples is not None and sampling != "default":
        key_dict["max_samples"] = max_samples
    key_data = json.dumps(key_dict, sort_keys=True, default=_serialize_value)
    return hashlib.sha256(key_data.encode()).hexdigest()[:32]


class GeometryCache:
    """LRU cache for generated geometry. All public methods are thread-safe."""

    def __init__(self, max_entries: int = DEFAULT_MAX_ENTRIES) -> None:
        """Initialize cache with a maximum number of entries."""
        self._max_entries = max_entries
        self._entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.Lock()

    @property
    def size(self) -> int:
        """Return the number of cached entries."""
        with self._lock:
            return len(self._entries)

    def get(self, key: str) -> CacheEntry | None:
        """Look up a cache entry by key, promoting it to most-recent."""
        with self._lock:
            if key not in self._entries:
                return None
            self._entries.move_to_end(key)
            logger.debug("Cache hit for key %s", key)
            return self._entries[key]

    def put(self, key: str, entry: CacheEntry) -> None:
        """Insert or update a cache entry, evicting oldest if at capacity."""
        with self._lock:
            if key in self._entries:
                self._entries.move_to_end(key)
                self._entries[key] = entry
                return
            if len(self._entries) >= self._max_entries:
                evicted_key, _ = self._entries.popitem(last=False)
                logger.debug("Evicted cache entry %s", evicted_key)
            self._entries[key] = entry
            logger.debug("Cached geometry with key %s", key)

    def clear(self) -> None:
        """Remove all entries from the cache."""
        with self._lock:
            self._entries.clear()
