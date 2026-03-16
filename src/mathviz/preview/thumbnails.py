"""Thumbnail generation and persistent disk caching for generator previews.

Generates 472x472 WebP thumbnails (retina 2x for 236 CSS px) using PyVista
server-side rendering. Thumbnails are cached at
~/.mathviz/thumbnails/<view_mode>/<generator_name>.webp.
"""

import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Literal

from mathviz.core.container import Container, PlacementPolicy
from mathviz.core.generator import get_generator_meta, list_generators
from mathviz.pipeline.runner import run as run_pipeline
from mathviz.preview.renderer import RenderConfig, render_to_png

logger = logging.getLogger(__name__)

THUMBNAIL_SIZE = 472
WEBP_QUALITY = 85
DEFAULT_THUMBNAILS_DIR = Path.home() / ".mathviz" / "thumbnails"
THUMBNAILS_DIR_ENV_VAR = "MATHVIZ_THUMBNAILS_DIR"
SUBPROCESS_TIMEOUT_SECONDS = 60

ViewMode = Literal["vertex", "shaded", "wireframe"]
VALID_VIEW_MODES: tuple[str, ...] = ("vertex", "shaded", "wireframe")
DEFAULT_VIEW_MODE: ViewMode = "vertex"


def get_thumbnails_dir() -> Path:
    """Return the configured thumbnails directory."""
    env_val = os.environ.get(THUMBNAILS_DIR_ENV_VAR)
    if env_val:
        return Path(env_val)
    return DEFAULT_THUMBNAILS_DIR


def get_thumbnail_path(generator_name: str, view_mode: str) -> Path:
    """Return the disk path for a cached thumbnail."""
    return get_thumbnails_dir() / view_mode / f"{generator_name}.webp"


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
    """Generate a WebP thumbnail for a generator and save it to disk cache."""
    from PIL import Image

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

    webp_path = get_thumbnail_path(meta.name, view_mode)
    webp_path.parent.mkdir(parents=True, exist_ok=True)

    # Render to a temporary PNG, then convert to WebP
    fd, tmp_png = tempfile.mkstemp(suffix=".tmp.png", dir=webp_path.parent)
    os.close(fd)
    png_path = Path(tmp_png)
    render_to_png(result.math_object, png_path, config=config, view="front-right-top")

    try:
        Image.open(png_path).save(webp_path, "webp", quality=WEBP_QUALITY)
    finally:
        png_path.unlink(missing_ok=True)

    logger.info("Generated thumbnail for %s (%s) at %s", generator_name, view_mode, webp_path)
    return webp_path


class ThumbnailSubprocessError(RuntimeError):
    """Raised when the thumbnail subprocess exits non-zero."""


class ThumbnailTimeoutError(TimeoutError):
    """Raised when the thumbnail subprocess exceeds its timeout."""


def generate_thumbnail_subprocess(
    generator_name: str,
    view_mode: str = DEFAULT_VIEW_MODE,
    timeout: int = SUBPROCESS_TIMEOUT_SECONDS,
) -> Path:
    """Generate a thumbnail in a subprocess to avoid VTK main-thread crashes.

    Spawns `python -m mathviz.cli_thumbnail <name> <view_mode>` which runs
    on its own main thread, satisfying macOS Cocoa/AppKit requirements.
    """
    expected_path = get_thumbnail_path(generator_name, view_mode)
    env = os.environ.copy()
    env.setdefault(THUMBNAILS_DIR_ENV_VAR, str(get_thumbnails_dir()))

    cmd = [sys.executable, "-m", "mathviz.cli_thumbnail", generator_name, view_mode]
    logger.info("Spawning thumbnail subprocess: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, env=env,
        )
    except subprocess.TimeoutExpired as exc:
        logger.error("Thumbnail subprocess timed out after %ds for %s", timeout, generator_name)
        raise ThumbnailTimeoutError(
            f"Thumbnail generation timed out after {timeout}s for {generator_name!r}"
        ) from exc

    if result.returncode != 0:
        stderr = result.stderr.strip()
        logger.error(
            "Thumbnail subprocess failed (rc=%d) for %s: %s",
            result.returncode, generator_name, stderr,
        )
        raise ThumbnailSubprocessError(
            f"Thumbnail generation failed for {generator_name!r}: {stderr}"
        )

    if not expected_path.is_file():
        raise ThumbnailSubprocessError(
            f"Subprocess exited 0 but thumbnail not found at {expected_path}"
        )

    logger.info("Subprocess generated thumbnail for %s at %s", generator_name, expected_path)
    return expected_path


def get_or_generate_thumbnail(generator_name: str, view_mode: str = DEFAULT_VIEW_MODE) -> Path:
    """Return cached thumbnail path, generating via subprocess if missing."""
    meta = get_generator_meta(generator_name)
    path = get_thumbnail_path(meta.name, view_mode)
    if path.is_file():
        logger.debug("Thumbnail cache hit for %s (%s)", meta.name, view_mode)
        return path
    return generate_thumbnail_subprocess(meta.name, view_mode)


def clear_all_thumbnails() -> int:
    """Remove all cached thumbnails. Returns count of files deleted."""
    thumbnails_dir = get_thumbnails_dir()
    if not thumbnails_dir.is_dir():
        return 0

    count = 0
    for ext in ("*.webp", "*.png"):
        for thumb_file in thumbnails_dir.rglob(ext):
            thumb_file.unlink()
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
