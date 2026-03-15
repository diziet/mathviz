"""High-resolution offline renderer using PyVista/VTK.

Provides 3D rendering to PNG and 2D projection rendering.
Requires optional dependency: pip install mathviz[render]
"""

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal, get_args

import numpy as np

from mathviz.core.math_object import MathObject

logger = logging.getLogger(__name__)

PYVISTA_INSTALL_MSG = (
    "PyVista is required for rendering. Install it with: pip install mathviz[render]"
)

RenderStyle = Literal["shaded", "wireframe", "points"]
VALID_RENDER_STYLES: tuple[RenderStyle, ...] = get_args(RenderStyle)

# All named views: (direction_vector, view_up_vector)
# Direction is where the camera looks FROM (normalized to unit sphere).
_CameraSpec = tuple[tuple[float, ...], tuple[float, ...]]

_VIEW_CAMERAS: dict[str, _CameraSpec] = {
    # 6 face-on views
    "front": ((0, -1, 0), (0, 0, 1)),
    "back": ((0, 1, 0), (0, 0, 1)),
    "left": ((-1, 0, 0), (0, 0, 1)),
    "right": ((1, 0, 0), (0, 0, 1)),
    "top": ((0, 0, 1), (0, 1, 0)),
    "bottom": ((0, 0, -1), (0, -1, 0)),
    # 12 edge-on views
    "front-right": ((1, -1, 0), (0, 0, 1)),
    "front-left": ((-1, -1, 0), (0, 0, 1)),
    "back-right": ((1, 1, 0), (0, 0, 1)),
    "back-left": ((-1, 1, 0), (0, 0, 1)),
    "front-top": ((0, -1, 1), (0, 1, 1)),
    "front-bottom": ((0, -1, -1), (0, -1, 1)),
    "back-top": ((0, 1, 1), (0, 1, -1)),
    "back-bottom": ((0, 1, -1), (0, -1, -1)),
    "top-right": ((1, 0, 1), (0, 1, 0)),
    "top-left": ((-1, 0, 1), (0, 1, 0)),
    "bottom-right": ((1, 0, -1), (0, -1, 0)),
    "bottom-left": ((-1, 0, -1), (0, -1, 0)),
    # 8 vertex/corner views
    "front-right-top": ((1, -1, 1), (0, 0, 1)),
    "front-left-top": ((-1, -1, 1), (0, 0, 1)),
    "back-right-top": ((1, 1, 1), (0, 0, 1)),
    "back-left-top": ((-1, 1, 1), (0, 0, 1)),
    "front-right-bottom": ((1, -1, -1), (0, 0, 1)),
    "front-left-bottom": ((-1, -1, -1), (0, 0, 1)),
    "back-right-bottom": ((1, 1, -1), (0, 0, 1)),
    "back-left-bottom": ((-1, 1, -1), (0, 0, 1)),
}

# Aliases for backwards compatibility
_VIEW_ALIASES: dict[str, str] = {
    "side": "right",
    "angle": "front-right-top",
}

VALID_VIEW_NAMES: tuple[str, ...] = tuple(_VIEW_CAMERAS.keys()) + tuple(_VIEW_ALIASES.keys())


def resolve_view_name(name: str) -> str:
    """Resolve a view name, following aliases."""
    resolved = _VIEW_ALIASES.get(name, name)
    if resolved not in _VIEW_CAMERAS:
        raise ValueError(
            f"Invalid view name {name!r}. "
            f"Valid views: {', '.join(sorted(VALID_VIEW_NAMES))}"
        )
    return resolved


def get_view_camera(name: str) -> _CameraSpec:
    """Get camera spec (position, view_up) for a named view."""
    return _VIEW_CAMERAS[resolve_view_name(name)]


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
    else:
        raise ValueError(f"Unsupported render style: {config.style!r}")

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
    config = config or RenderConfig()
    with _create_plotter_bare(obj, config) as (plotter, resolved_config):
        yield plotter, resolved_config
        plotter.screenshot(str(Path(output_path)))


@contextmanager
def _create_plotter_bare(
    obj: MathObject,
    config: RenderConfig | None = None,
) -> Iterator[tuple["object", RenderConfig]]:
    """Create a PyVista plotter with scene setup, no automatic screenshot."""
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
    finally:
        plotter.close()


def _apply_camera_view(plotter: "object", view: str) -> None:
    """Position camera for a named view on a sphere around the geometry."""
    camera_pos, view_up = get_view_camera(view)
    plotter.camera.position = camera_pos
    plotter.camera.focal_point = (0, 0, 0)
    plotter.camera.up = view_up
    plotter.reset_camera()


def render_to_png(
    obj: MathObject,
    output_path: Path,
    config: RenderConfig | None = None,
    view: str = "front-right-top",
) -> Path:
    """Render a MathObject to a high-resolution PNG file."""
    output_path = Path(output_path)
    resolved_view = resolve_view_name(view)
    with _create_plotter(obj, output_path, config) as (plotter, resolved_config):
        _apply_camera_view(plotter, resolved_view)

    logger.info(
        "Rendered %dx%d %s view to %s",
        resolved_config.width,
        resolved_config.height,
        resolved_view,
        output_path,
    )
    return output_path


def render_2d_projection(
    obj: MathObject,
    output_path: Path,
    view: str = "top",
    config: RenderConfig | None = None,
) -> Path:
    """Render a 2D projection of a MathObject to PNG."""
    output_path = Path(output_path)
    resolved_view = resolve_view_name(view)
    with _create_plotter(obj, output_path, config) as (plotter, _):
        plotter.enable_parallel_projection()
        _apply_camera_view(plotter, resolved_view)

    logger.info("Rendered 2D %s projection to %s", resolved_view, output_path)
    return output_path


def render_all_views(
    obj: MathObject,
    output_path: Path,
    config: RenderConfig | None = None,
    use_2d: bool = False,
) -> list[Path]:
    """Render all named views with a single plotter for efficiency."""
    stem = output_path.stem
    suffix = output_path.suffix or ".png"
    parent = output_path.parent
    paths: list[Path] = []

    with _create_plotter_bare(obj, config) as (plotter, _):
        if use_2d:
            plotter.enable_parallel_projection()
        for view_name in _VIEW_CAMERAS:
            _apply_camera_view(plotter, view_name)
            view_path = parent / f"{stem}_{view_name}{suffix}"
            plotter.screenshot(str(view_path))
            paths.append(view_path)

    logger.info("Rendered %d views to %s", len(paths), parent)
    return paths
