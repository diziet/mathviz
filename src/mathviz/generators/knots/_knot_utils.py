"""Shared utilities for knot generators."""

import logging
from typing import Any

logger = logging.getLogger(__name__)

MIN_CURVE_POINTS = 16
DEFAULT_TUBE_RADIUS = 0.1


def validate_curve_points(curve_points: int) -> None:
    """Validate curve_points is above minimum."""
    if curve_points < MIN_CURVE_POINTS:
        raise ValueError(
            f"curve_points must be >= {MIN_CURVE_POINTS}, got {curve_points}"
        )


def extract_curve_points(
    merged: dict[str, Any],
    resolution_kwargs: dict[str, Any],
    default: int,
) -> tuple[dict[str, Any], int]:
    """Extract curve_points from params/kwargs, warn if in params."""
    if "curve_points" in merged:
        logger.warning(
            "curve_points should be passed as a resolution kwarg, "
            "not inside params; ignoring params value"
        )
        merged.pop("curve_points")
    curve_points = int(resolution_kwargs.get("curve_points", default))
    return merged, curve_points
