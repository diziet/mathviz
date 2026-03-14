"""High-resolution offline renderer using PyVista/VTK.

Provides 3D rendering to PNG and 2D projection rendering.
Requires optional dependency: pip install mathviz[render]
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from mathviz.core.math_object import MathObject

logger = logging.getLogger(__name__)

PYVISTA_INSTALL_MSG = (
    "PyVista is required for rendering. Install it with: pip install mathviz[render]"
)

ProjectionView = Literal["top", "front", "side", "angle"]

# Camera positions for 2D projections: (position, viewup)
_PROJECTION_CAMERAS: dict[ProjectionView, tuple[tuple[float, ...], tuple[float, ...]]] = {
    "top": ((0, 0, 1), (0, 1, 0)),
    "front": ((0, -1, 0), (0, 0, 1)),
    "side": ((1, 0, 0), (0, 0, 1)),
    "angle": ((1, -1, 1), (0, 0, 1)),
}


@dataclass
class RenderConfig:
    """Configuration for high-resolution rendering."""

    width: int = 1920
    height: int = 1080
    background_color: str = "black"
    object_color: str = "white"
    opacity: float = 0.85
    lighting: str = "backlit"


def require_pyvista() -> None:
    """Check that pyvista is importable, raising with install instructions."""
    try:
        import pyvista  # noqa: F401
    except ImportError:
        raise ImportError(PYVISTA_INSTALL_MSG) from None


def _get_pyvista_mesh(obj: MathObject) -> "object":
    """Convert MathObject geometry to a PyVista mesh."""
    import pyvista as pv

    if obj.mesh is not None:
        faces_with_counts = np.column_stack(
            [
                np.full(len(obj.mesh.faces), 3, dtype=obj.mesh.faces.dtype),
                obj.mesh.faces,
            ]
        ).ravel()
        return pv.PolyData(obj.mesh.vertices, faces_with_counts)

    if obj.point_cloud is not None:
        return pv.PolyData(obj.point_cloud.points)

    if obj.curves is not None and len(obj.curves) > 0:
        all_points = np.vstack([c.points for c in obj.curves])
        return pv.PolyData(all_points)

    raise ValueError("MathObject has no renderable geometry")


def _setup_backlit_scene(plotter: "object", pv_mesh: "object", config: RenderConfig) -> None:
    """Set up a scene with lighting appropriate for simulating backlit glass."""
    import pyvista as pv

    plotter.set_background(config.background_color)

    plotter.add_mesh(
        pv_mesh,
        color=config.object_color,
        opacity=config.opacity,
        smooth_shading=True,
    )

    # Backlit glass simulation: strong light from behind, softer fill from front
    plotter.remove_all_lights()
    back_light = pv.Light(position=(0, 0, -5), focal_point=(0, 0, 0), intensity=1.0)
    back_light.positional = False
    plotter.add_light(back_light)

    fill_light = pv.Light(position=(0, 0, 5), focal_point=(0, 0, 0), intensity=0.3)
    fill_light.positional = False
    plotter.add_light(fill_light)

    rim_light = pv.Light(position=(5, 5, 0), focal_point=(0, 0, 0), intensity=0.2)
    rim_light.positional = False
    plotter.add_light(rim_light)


def render_to_png(
    obj: MathObject,
    output_path: Path,
    config: RenderConfig | None = None,
) -> Path:
    """Render a MathObject to a high-resolution PNG file."""
    require_pyvista()
    import pyvista as pv

    config = config or RenderConfig()
    pv_mesh = _get_pyvista_mesh(obj)

    pv.OFF_SCREEN = True
    plotter = pv.Plotter(
        off_screen=True,
        window_size=[config.width, config.height],
    )
    _setup_backlit_scene(plotter, pv_mesh, config)

    output_path = Path(output_path)
    plotter.screenshot(str(output_path))
    plotter.close()

    logger.info("Rendered %dx%d image to %s", config.width, config.height, output_path)
    return output_path


def render_2d_projection(
    obj: MathObject,
    output_path: Path,
    view: ProjectionView = "top",
    config: RenderConfig | None = None,
) -> Path:
    """Render a 2D projection of a MathObject to PNG."""
    require_pyvista()
    import pyvista as pv

    config = config or RenderConfig()
    pv_mesh = _get_pyvista_mesh(obj)

    pv.OFF_SCREEN = True
    plotter = pv.Plotter(
        off_screen=True,
        window_size=[config.width, config.height],
    )
    _setup_backlit_scene(plotter, pv_mesh, config)

    # Set up parallel projection for true 2D view
    plotter.enable_parallel_projection()

    camera_pos, view_up = _PROJECTION_CAMERAS[view]
    plotter.camera.position = camera_pos
    plotter.camera.focal_point = (0, 0, 0)
    plotter.camera.up = view_up
    plotter.reset_camera()

    output_path = Path(output_path)
    plotter.screenshot(str(output_path))
    plotter.close()

    logger.info("Rendered 2D %s projection to %s", view, output_path)
    return output_path
