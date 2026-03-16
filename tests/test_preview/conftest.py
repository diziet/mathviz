"""Shared test helpers for preview tests."""

from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from mathviz.core.container import Container
from mathviz.core.generator import register
from mathviz.generators.parametric.torus import TorusGenerator
from mathviz.preview.disk_cache import DiskCache
from mathviz.preview.server import get_disk_cache, set_disk_cache
from mathviz.preview.thumbnails import (
    THUMBNAIL_SIZE,
    THUMBNAILS_DIR_ENV_VAR,
    get_thumbnail_path,
)


@pytest.fixture(autouse=True)
def _use_temp_disk_cache(tmp_path: Path) -> None:
    """Use a temp directory for disk cache so tests don't hit ~/.mathviz/cache."""
    original = get_disk_cache()
    set_disk_cache(DiskCache(cache_dir=tmp_path / "disk_cache"))
    yield
    set_disk_cache(original)


def make_snapshot_request(geometry_id: str) -> dict[str, Any]:
    """Build a standard snapshot request body with default container."""
    c = Container()
    return {
        "generator": "torus",
        "params": {"major_radius": 1.0, "minor_radius": 0.4},
        "seed": 42,
        "container": {
            "width_mm": c.width_mm,
            "height_mm": c.height_mm,
            "depth_mm": c.depth_mm,
            "margin_x_mm": c.margin_x_mm,
            "margin_y_mm": c.margin_y_mm,
            "margin_z_mm": c.margin_z_mm,
        },
        "geometry_id": geometry_id,
    }


def ensure_torus_registered() -> None:
    """Re-register the torus generator if missing."""
    import mathviz.core.generator as gen_mod

    if "torus" not in gen_mod._alias_map:
        gen_mod._discovered = True
        register(TorusGenerator)


def create_fake_thumbnail(name: str, view_mode: str) -> Path:
    """Write a valid WebP file at the expected thumbnail cache path."""
    path = get_thumbnail_path(name, view_mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (THUMBNAIL_SIZE, THUMBNAIL_SIZE), color=(128, 128, 128))
    img.save(path, "webp")
    return path


@pytest.fixture
def thumbnail_setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Register generators and redirect thumbnail dir to temp."""
    ensure_torus_registered()
    monkeypatch.setenv(THUMBNAILS_DIR_ENV_VAR, str(tmp_path / "thumbnails"))
