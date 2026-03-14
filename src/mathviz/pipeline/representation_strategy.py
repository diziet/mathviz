"""Representation strategy: transform raw generator output into fabrication-ready form.

This is the key architectural seam between mathematical definition and
fabrication. Each RepresentationType maps to a concrete transformation
that produces geometry suitable for engraving in glass.
"""

import logging
from dataclasses import replace

import numpy as np

from mathviz.core.math_object import Curve, MathObject, Mesh, PointCloud
from mathviz.core.representation import RepresentationConfig, RepresentationType
from mathviz.shared.tube_thickening import thicken_curve

logger = logging.getLogger(__name__)

# Unimplemented types that raise NotImplementedError
_STUB_TYPES = frozenset({
    RepresentationType.SPARSE_SHELL,
    RepresentationType.VOLUME_FILL,
    RepresentationType.WIREFRAME,
    RepresentationType.WEIGHTED_CLOUD,
    RepresentationType.SLICE_STACK,
})

# Default representation configs by generator name
_GENERATOR_DEFAULTS: dict[str, RepresentationConfig] = {
    "torus": RepresentationConfig(type=RepresentationType.SURFACE_SHELL),
    "klein_bottle": RepresentationConfig(type=RepresentationType.SURFACE_SHELL),
    "sphere": RepresentationConfig(type=RepresentationType.SURFACE_SHELL),
    "lorenz": RepresentationConfig(
        type=RepresentationType.TUBE, tube_radius=0.05
    ),
    "rossler": RepresentationConfig(
        type=RepresentationType.TUBE, tube_radius=0.05
    ),
    "trefoil_knot": RepresentationConfig(
        type=RepresentationType.TUBE, tube_radius=0.1
    ),
    "lissajous": RepresentationConfig(
        type=RepresentationType.TUBE, tube_radius=0.05
    ),
    "mandelbrot": RepresentationConfig(
        type=RepresentationType.HEIGHTMAP_RELIEF,
    ),
}

_FALLBACK_DEFAULT = RepresentationConfig(type=RepresentationType.SURFACE_SHELL)


def get_default(generator_name: str) -> RepresentationConfig:
    """Return the recommended representation config for a generator."""
    return _GENERATOR_DEFAULTS.get(generator_name, _FALLBACK_DEFAULT)


def apply(
    obj: MathObject,
    config: RepresentationConfig,
    *,
    candidates: bool = False,
) -> MathObject:
    """Apply a representation strategy to a MathObject.

    When candidates=True, returns multiple representations for comparison.
    This mode is currently a stub and raises NotImplementedError.
    """
    if candidates:
        raise NotImplementedError(
            "candidates mode is not yet implemented — "
            "will return multiple representations for comparison"
        )

    obj.validate_or_raise()

    if config.type in _STUB_TYPES:
        raise NotImplementedError(
            f"Representation type '{config.type.value}' is not yet implemented"
        )

    handler = _HANDLERS[config.type]
    result = handler(obj, config)
    result = replace(result, representation=config.type.value)

    logger.info(
        "Applied representation %s to %s",
        config.type.value,
        obj.generator_name or "unnamed",
    )
    return result


def _apply_surface_shell(obj: MathObject, config: RepresentationConfig) -> MathObject:
    """Pass-through for mesh inputs."""
    if obj.mesh is None:
        raise ValueError(
            "SURFACE_SHELL requires a mesh input, but MathObject has no mesh"
        )
    return replace(obj)


def _apply_raw_point_cloud(
    obj: MathObject, config: RepresentationConfig
) -> MathObject:
    """Extract curve points into a PointCloud."""
    if not obj.curves:
        raise ValueError(
            "RAW_POINT_CLOUD requires curve input, "
            "but MathObject has no curves"
        )
    all_points = np.concatenate(
        [c.points for c in obj.curves], axis=0
    )
    cloud = PointCloud(points=all_points.astype(np.float64))
    return replace(obj, point_cloud=cloud)


def _apply_tube(obj: MathObject, config: RepresentationConfig) -> MathObject:
    """Apply tube thickening to curves."""
    if not obj.curves:
        raise ValueError(
            "TUBE requires curve input, but MathObject has no curves"
        )
    radius = config.tube_radius
    if radius is None:
        raise ValueError("TUBE representation requires tube_radius to be set")

    sides = config.tube_sides or 16
    meshes = _thicken_all_curves(obj.curves, radius, sides)
    merged = _merge_meshes(meshes)
    return replace(obj, mesh=merged)


def _thicken_all_curves(
    curves: list[Curve], radius: float, sides: int
) -> list[Mesh]:
    """Thicken each curve into a tube mesh."""
    return [thicken_curve(c, radius, sides) for c in curves]


def _merge_meshes(meshes: list[Mesh]) -> Mesh:
    """Merge multiple meshes into a single mesh."""
    if len(meshes) == 1:
        return meshes[0]

    all_vertices = []
    all_faces = []
    vertex_offset = 0

    for mesh in meshes:
        all_vertices.append(mesh.vertices)
        all_faces.append(mesh.faces + vertex_offset)
        vertex_offset += len(mesh.vertices)

    return Mesh(
        vertices=np.concatenate(all_vertices, axis=0),
        faces=np.concatenate(all_faces, axis=0),
    )


def _apply_heightmap_relief(
    obj: MathObject, config: RepresentationConfig
) -> MathObject:
    """Extrude a 2D scalar field into a surface mesh."""
    if obj.scalar_field is None:
        raise ValueError(
            "HEIGHTMAP_RELIEF requires a scalar_field, "
            "but MathObject has none"
        )
    field = obj.scalar_field
    if field.ndim != 2:
        raise ValueError(
            f"HEIGHTMAP_RELIEF scalar_field must be 2D, got {field.ndim}D"
        )

    mesh = _build_heightmap_mesh(field)
    return replace(obj, mesh=mesh)


def _build_heightmap_mesh(field: np.ndarray) -> Mesh:
    """Build a triangle mesh from a 2D scalar field treated as z-heights."""
    rows, cols = field.shape
    x = np.linspace(0.0, 1.0, cols)
    y = np.linspace(0.0, 1.0, rows)
    xx, yy = np.meshgrid(x, y)

    vertices = np.column_stack([
        xx.ravel(),
        yy.ravel(),
        field.ravel().astype(np.float64),
    ])

    # Build triangle faces from grid quads
    faces = _build_grid_faces(rows, cols)

    return Mesh(
        vertices=vertices.astype(np.float64),
        faces=faces.astype(np.int64),
    )


def _build_grid_faces(rows: int, cols: int) -> np.ndarray:
    """Build triangle faces for a rows x cols grid."""
    row_idx = np.arange(rows - 1)[:, np.newaxis]
    col_idx = np.arange(cols - 1)[np.newaxis, :]

    top_left = row_idx * cols + col_idx
    top_right = top_left + 1
    bot_left = top_left + cols
    bot_right = bot_left + 1

    tri1 = np.stack([top_left, bot_left, bot_right], axis=-1)
    tri2 = np.stack([top_left, bot_right, top_right], axis=-1)
    faces = np.concatenate([tri1, tri2], axis=-2)
    return faces.reshape(-1, 3).astype(np.int64)


_HANDLERS = {
    RepresentationType.SURFACE_SHELL: _apply_surface_shell,
    RepresentationType.RAW_POINT_CLOUD: _apply_raw_point_cloud,
    RepresentationType.TUBE: _apply_tube,
    RepresentationType.HEIGHTMAP_RELIEF: _apply_heightmap_relief,
}
