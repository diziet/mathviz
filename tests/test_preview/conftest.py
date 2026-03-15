"""Shared test helpers for preview tests."""

from typing import Any

from mathviz.core.container import Container


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
