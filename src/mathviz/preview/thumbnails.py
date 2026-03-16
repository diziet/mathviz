"""Thumbnail generation and persistent disk caching for generator previews.

Generates 256x256 PNG thumbnails using PyVista server-side rendering.
Thumbnails are cached at ~/.mathviz/thumbnails/<view_mode>/<generator_name>.png.
"""

import logging
import os
import shutil
from pathlib import Path
from typing import Literal

from mathviz.core.container import Container, PlacementPolicy
from mathviz.core.generator import get_generator_meta, list_generators
from mathviz.pipeline.runner import run as run_pipeline
from mathviz.preview.renderer import RenderConfig, render_to_png

logger = logging.getLogger(__name__)

THUMBNAIL_SIZE = 256
DEFAULT_THUMBNAILS_DIR = Path.home() / ".mathviz" / "thumbnails"
THUMBNAILS_DIR_ENV_VAR = "MATHVIZ_THUMBNAILS_DIR"

ViewMode = Literal["points", "shaded", "wireframe"]
VALID_VIEW_MODES: tuple[str, ...] = ("points", "shaded", "wireframe")
DEFAULT_VIEW_MODE: ViewMode = "points"


def get_thumbnails_dir() -> Path:
    """Return the configured thumbnails directory."""
    env_val = os.environ.get(THUMBNAILS_DIR_ENV_VAR)
    if env_val:
        return Path(env_val)
    return DEFAULT_THUMBNAILS_DIR


def get_thumbnail_path(generator_name: str, view_mode: str) -> Path:
    """Return the disk path for a cached thumbnail."""
    return get_thumbnails_dir() / view_mode / f"{generator_name}.png"


def _halve_resolution(resolution: dict) -> dict:
    """Reduce resolution values by half for faster thumbnail generation."""
    halved: dict = {}
    for key, value in resolution.items():
        if isinstance(value, int) and value > 1:
            halved[key] = max(value // 2, 1)
        elif isinstance(value, float) and value > 0:
            halved[key] = min(value, max(value / 2.0, 0.01))
        else:
            halved[key] = value
    return halved


def generate_thumbnail(generator_name: str, view_mode: str = DEFAULT_VIEW_MODE) -> Path:
    """Generate a thumbnail PNG for a generator and save it to disk cache."""
    meta = get_generator_meta(generator_name)
    instance = meta.generator_class.create(resolved_name=generator_name)

    default_resolution = instance.get_default_resolution()
    reduced_resolution = _halve_resolution(default_resolution)

    result = run_pipeline(
        generator_name,
        params=None,
        seed=42,
        resolution_kwargs=reduced_resolution,
        container=Container.with_uniform_margin(),
        placement=PlacementPolicy(),
    )

    config = RenderConfig(
        width=THUMBNAIL_SIZE,
        height=THUMBNAIL_SIZE,
        style=view_mode,
        point_size=2.0,
    )

    output_path = get_thumbnail_path(meta.name, view_mode)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    render_to_png(result.math_object, output_path, config=config, view="front-right-top")
    logger.info("Generated thumbnail for %s (%s) at %s", generator_name, view_mode, output_path)
    return output_path


def get_or_generate_thumbnail(generator_name: str, view_mode: str = DEFAULT_VIEW_MODE) -> Path:
    """Return cached thumbnail path, generating if missing."""
    meta = get_generator_meta(generator_name)
    path = get_thumbnail_path(meta.name, view_mode)
    if path.is_file():
        logger.debug("Thumbnail cache hit for %s (%s)", meta.name, view_mode)
        return path
    return generate_thumbnail(generator_name, view_mode)


def clear_all_thumbnails() -> int:
    """Remove all cached thumbnails. Returns count of files deleted."""
    thumbnails_dir = get_thumbnails_dir()
    if not thumbnails_dir.is_dir():
        return 0

    count = 0
    for png_file in thumbnails_dir.rglob("*.png"):
        png_file.unlink()
        count += 1

    # Clean up empty subdirectories
    for subdir in thumbnails_dir.iterdir():
        if subdir.is_dir() and not any(subdir.iterdir()):
            subdir.rmdir()

    logger.info("Cleared %d cached thumbnails", count)
    return count


def build_thumbnail_url(generator_name: str, view_mode: str = DEFAULT_VIEW_MODE) -> str:
    """Build the API URL for a generator's thumbnail."""
    return f"/api/generators/{generator_name}/thumbnail?view_mode={view_mode}"


def get_all_thumbnail_urls(view_mode: str = DEFAULT_VIEW_MODE) -> dict[str, str]:
    """Return a map of generator_name -> thumbnail URL for all generators."""
    return {
        meta.name: build_thumbnail_url(meta.name, view_mode)
        for meta in list_generators()
    }
