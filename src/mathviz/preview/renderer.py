"""High-resolution offline renderer using PyVista/VTK.

Provides 3D rendering to PNG and 2D projection rendering.
Requires optional dependency: pip install mathviz[render]
"""

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

import numpy as np

from mathviz.core.math_object import MathObject

logger = logging.getLogger(__name__)

PYVISTA_INSTALL_MSG = (
    "PyVista is required for rendering. Install it with: pip install mathviz[render]"
)

ProjectionView = Literal["top", "front", "side", "angle"]
RenderStyle = Literal["shaded", "wireframe", "points"]
VALID_RENDER_STYLES = ("shaded", "wireframe", "points")

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
    style: RenderStyle = "points"
    point_size: float = 3.0

    def __post_init__(self) -> None:
        """Validate config fields."""
        if self.width <= 0:
            raise ValueError(f"width must be positive, got {self.width}")
        if self.height <= 0:
            raise ValueError(f"height must be positive, got {self.height}")
        if self.style not in VALID_RENDER_STYLES:
            raise ValueError(
                f"Invalid style {self.style!r}. Must be one of {VALID_RENDER_STYLES}"
            )
        if self.point_size <= 0:
            raise ValueError(f"point_size must be positive, got {self.point_size}")


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
        # TODO: Add line connectivity for proper curve rendering.
        # Currently renders curves as disconnected points.
        all_points = np.vstack([c.points for c in obj.curves])
        return pv.PolyData(all_points)

    raise ValueError("MathObject has no renderable geometry")


def _setup_backlit_scene(plotter: "object", pv_mesh: "object", config: RenderConfig) -> None:
    """Set up a scene with lighting appropriate for simulating backlit glass."""
    import pyvista as pv

    plotter.set_background(config.background_color)

    mesh_kwargs: dict[str, object] = {
        "color": config.object_color,
        "opacity": config.opacity,
    }
    if config.style == "shaded":
        mesh_kwargs["smooth_shading"] = True
    elif config.style == "wireframe":
        mesh_kwargs["style"] = "wireframe"
    elif config.style == "points":
        mesh_kwargs["style"] = "points"
        mesh_kwargs["point_size"] = config.point_size
        mesh_kwargs["render_points_as_spheres"] = True

    plotter.add_mesh(pv_mesh, **mesh_kwargs)

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


@contextmanager
def _create_plotter(
    obj: MathObject,
    output_path: Path,
    config: RenderConfig | None = None,
) -> Iterator[tuple["object", RenderConfig]]:
    """Create a PyVista plotter with scene setup, screenshot on exit."""
    require_pyvista()
    import pyvista as pv

    config = config or RenderConfig()
    pv_mesh = _get_pyvista_mesh(obj)

    pv.OFF_SCREEN = True
    plotter = pv.Plotter(
        off_screen=True,
        window_size=[config.width, config.height],
    )
    try:
        _setup_backlit_scene(plotter, pv_mesh, config)
        yield plotter, config
        plotter.screenshot(str(Path(output_path)))
    finally:
        plotter.close()


def render_to_png(
    obj: MathObject,
    output_path: Path,
    config: RenderConfig | None = None,
) -> Path:
    """Render a MathObject to a high-resolution PNG file."""
    output_path = Path(output_path)
    with _create_plotter(obj, output_path, config) as (plotter, resolved_config):
        pass  # Default scene is sufficient for 3D render

    logger.info(
        "Rendered %dx%d image to %s",
        resolved_config.width,
        resolved_config.height,
        output_path,
    )
    return output_path


def render_2d_projection(
    obj: MathObject,
    output_path: Path,
    view: ProjectionView = "top",
    config: RenderConfig | None = None,
) -> Path:
    """Render a 2D projection of a MathObject to PNG."""
    output_path = Path(output_path)
    with _create_plotter(obj, output_path, config) as (plotter, _):
        # Set up parallel projection for true 2D view
        plotter.enable_parallel_projection()

        camera_pos, view_up = _PROJECTION_CAMERAS[view]
        plotter.camera.position = camera_pos
        plotter.camera.focal_point = (0, 0, 0)
        plotter.camera.up = view_up
        plotter.reset_camera()

    logger.info("Rendered 2D %s projection to %s", view, output_path)
    return output_path
