"""Thumbnail API routes for generator preview images."""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from mathviz.core.generator import get_generator_meta
from mathviz.preview.thumbnails import (
    VALID_VIEW_MODES,
    ThumbnailSubprocessError,
    ThumbnailTimeoutError,
    clear_all_thumbnails,
    get_all_thumbnail_urls,
    get_or_generate_thumbnail,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["thumbnails"])


def _validate_view_mode(view_mode: str) -> str:
    """Validate and return view_mode, raising 400 on invalid values."""
    if view_mode not in VALID_VIEW_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid view_mode {view_mode!r}. Must be one of {VALID_VIEW_MODES}.",
        )
    return view_mode


@router.get("/api/generators/{name}/thumbnail")
def get_generator_thumbnail(
    name: str,
    view_mode: str = Query(default="points"),
) -> FileResponse:
    """Return a 472x472 WebP thumbnail for the given generator."""
    _validate_view_mode(view_mode)

    try:
        get_generator_meta(name)
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Generator {name!r} not found.",
        )

    try:
        path = get_or_generate_thumbnail(name, view_mode)
    except ThumbnailTimeoutError as exc:
        logger.error("Thumbnail generation timed out for %s: %s", name, exc)
        raise HTTPException(
            status_code=503,
            detail="Thumbnail generation timed out. Try again later.",
        ) from exc
    except ThumbnailSubprocessError as exc:
        logger.error("Thumbnail subprocess failed for %s: %s", name, exc)
        raise HTTPException(
            status_code=503,
            detail="Thumbnail generation failed. Please try again later.",
        ) from exc
    except ImportError as exc:
        logger.warning("Render dependencies unavailable for thumbnail %s: %s", name, exc)
        raise HTTPException(
            status_code=501,
            detail="Thumbnail rendering unavailable — install mathviz[render] dependencies.",
        ) from exc
    except Exception as exc:
        logger.error("Failed to generate thumbnail for %s: %s", name, exc)
        raise HTTPException(
            status_code=500,
            detail="Failed to generate thumbnail.",
        ) from exc

    return FileResponse(str(path), media_type="image/webp")


@router.get("/api/generators/thumbnails")
def get_all_thumbnails(
    view_mode: str = Query(default="points"),
) -> dict[str, str]:
    """Return a map of generator_name -> thumbnail_url for all generators."""
    _validate_view_mode(view_mode)
    return get_all_thumbnail_urls(view_mode)


@router.delete("/api/thumbnails")
def delete_all_thumbnails() -> dict[str, Any]:
    """Clear all cached thumbnails from disk."""
    count = clear_all_thumbnails()
    return {"status": "ok", "thumbnails_removed": count}
