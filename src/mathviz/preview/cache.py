"""LRU geometry cache for the preview server."""

import hashlib
import json
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

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


def compute_cache_key(
    generator_name: str,
    params: dict[str, Any],
    seed: int,
    resolution_kwargs: dict[str, Any],
) -> str:
    """Compute a deterministic cache key from generation parameters."""
    key_data = json.dumps(
        {
            "generator": generator_name,
            "params": params,
            "seed": seed,
            "resolution": resolution_kwargs,
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(key_data.encode()).hexdigest()[:16]


class GeometryCache:
    """Thread-safe LRU cache for generated geometry."""

    def __init__(self, max_entries: int = DEFAULT_MAX_ENTRIES) -> None:
        """Initialize cache with a maximum number of entries."""
        self._max_entries = max_entries
        self._entries: OrderedDict[str, CacheEntry] = OrderedDict()

    @property
    def size(self) -> int:
        """Return the number of cached entries."""
        return len(self._entries)

    def get(self, key: str) -> CacheEntry | None:
        """Look up a cache entry by key, promoting it to most-recent."""
        if key not in self._entries:
            return None
        self._entries.move_to_end(key)
        logger.debug("Cache hit for key %s", key)
        return self._entries[key]

    def put(self, key: str, entry: CacheEntry) -> None:
        """Insert or update a cache entry, evicting oldest if at capacity."""
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
        self._entries.clear()
