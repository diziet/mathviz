"""Root test configuration — shared fixtures for parallel execution."""

import pytest

import mathviz.core.generator as _gen_mod

# Snapshot of the fully-discovered registry, captured once per worker.
_snapshot_registry: dict | None = None
_snapshot_alias_map: dict | None = None


def _capture_snapshot() -> None:
    """Trigger discovery and snapshot the full registry state."""
    global _snapshot_registry, _snapshot_alias_map
    _gen_mod._ensure_discovered()
    _snapshot_registry = dict(_gen_mod._registry)
    _snapshot_alias_map = dict(_gen_mod._alias_map)


@pytest.fixture(autouse=True)
def _restore_registry_after_test() -> None:
    """Restore the generator registry to its fully-discovered state.

    Many test fixtures call ``clear_registry(suppress_discovery=True)``
    which empties the registry and prevents subsequent auto-discovery.
    When pytest-xdist assigns multiple modules to the same worker, this
    leaked state causes tests in later modules to see an empty registry.

    This fixture snapshots the fully-discovered registry on first use,
    then restores it after every test teardown — ensuring the next test
    always starts with a complete registry.
    """
    if _snapshot_registry is None:
        _capture_snapshot()
    yield
    _gen_mod._registry.clear()
    _gen_mod._registry.update(_snapshot_registry)
    _gen_mod._alias_map.clear()
    _gen_mod._alias_map.update(_snapshot_alias_map)
    _gen_mod._discovered = True
