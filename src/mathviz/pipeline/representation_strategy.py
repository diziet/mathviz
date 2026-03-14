"""Representation strategy: transform raw generator output into fabrication-ready form.

This is the key architectural seam between mathematical definition and
fabrication. Each RepresentationType maps to a concrete transformation
that produces geometry suitable for engraving in glass.
"""

import logging
from dataclasses import replace

import numpy as np
import trimesh

from mathviz.core.generator import get_generator_meta
from mathviz.core.math_object import Curve, MathObject, Mesh, PointCloud
from mathviz.core.representation import RepresentationConfig, RepresentationType
from mathviz.pipeline.representation_handlers import (
    apply_slice_stack,
    apply_volume_fill,
    apply_wireframe,
)
from mathviz.shared.tube_thickening import thicken_curve

logger = logging.getLogger(__name__)

# Default representation configs by generator name
_TUBE_CONFIG = RepresentationConfig(type=RepresentationType.TUBE, tube_radius=0.05)
_SURFACE_CONFIG = RepresentationConfig(type=RepresentationType.SURFACE_SHELL)
_SPARSE_CONFIG = RepresentationConfig(type=RepresentationType.SPARSE_SHELL)
_HEIGHTMAP_CONFIG = RepresentationConfig(type=RepresentationType.HEIGHTMAP_RELIEF)

_GENERATOR_DEFAULTS: dict[str, RepresentationConfig] = {
    # Parametric surfaces (mesh)
    "torus": _SURFACE_CONFIG,
    "klein_bottle": _SURFACE_CONFIG,
    "sphere": _SURFACE_CONFIG,
    "boy_surface": _SURFACE_CONFIG,
    "costa_surface": _SURFACE_CONFIG,
    "enneper_surface": _SURFACE_CONFIG,
    "lissajous_surface": _SURFACE_CONFIG,
    "mobius_strip": _SURFACE_CONFIG,
    "spherical_harmonics": _SURFACE_CONFIG,
    "superellipsoid": _SURFACE_CONFIG,
    "generic_parametric": _SURFACE_CONFIG,
    # Implicit surfaces (mesh)
    "genus2_surface": _SURFACE_CONFIG,
    "gyroid": _SURFACE_CONFIG,
    "schwarz_d": _SURFACE_CONFIG,
    "schwarz_p": _SURFACE_CONFIG,
    # Data-driven mesh
    "building_extrude": _SURFACE_CONFIG,
    "parabolic_envelope": _SURFACE_CONFIG,
    # Attractors (curves)
    "lorenz": _TUBE_CONFIG,
    "rossler": _TUBE_CONFIG,
    "aizawa": _TUBE_CONFIG,
    "chen": _TUBE_CONFIG,
    "double_pendulum": _TUBE_CONFIG,
    "halvorsen": _TUBE_CONFIG,
    "thomas": _TUBE_CONFIG,
    # Knots (curves)
    "torus_knot": RepresentationConfig(
        type=RepresentationType.TUBE, tube_radius=0.1
    ),
    "figure_eight_knot": RepresentationConfig(
        type=RepresentationType.TUBE, tube_radius=0.1
    ),
    "lissajous_knot": RepresentationConfig(
        type=RepresentationType.TUBE, tube_radius=0.1
    ),
    "seven_crossing_knots": RepresentationConfig(
        type=RepresentationType.TUBE, tube_radius=0.1
    ),
    # Curves
    "lissajous": _TUBE_CONFIG,
    "lissajous_curve": _TUBE_CONFIG,
    "cardioid": _TUBE_CONFIG,
    "fibonacci_spiral": _TUBE_CONFIG,
    "logarithmic_spiral": _TUBE_CONFIG,
    "voronoi_3d": _TUBE_CONFIG,
    "soundwave": _TUBE_CONFIG,
    # Physics (curves)
    "kepler_orbit": _TUBE_CONFIG,
    "nbody": _TUBE_CONFIG,
    "planetary_positions": _TUBE_CONFIG,
    # Number theory (point clouds)
    "sacks_spiral": _SPARSE_CONFIG,
    "prime_gaps": _SPARSE_CONFIG,
    "ulam_spiral": _SPARSE_CONFIG,
    "digit_encoding": _SPARSE_CONFIG,
    # Fractals (sparse shell from mesh)
    "mandelbulb": _SPARSE_CONFIG,
    "julia3d": _SPARSE_CONFIG,
    # Heightmaps (scalar field)
    "mandelbrot": _HEIGHTMAP_CONFIG,
    "mandelbrot_heightmap": _HEIGHTMAP_CONFIG,
    "fractal_slice": _HEIGHTMAP_CONFIG,
    "heightmap": _HEIGHTMAP_CONFIG,
    "noise_surface": _HEIGHTMAP_CONFIG,
    "reaction_diffusion": _HEIGHTMAP_CONFIG,
    "terrain": _HEIGHTMAP_CONFIG,
}

_FALLBACK_TUBE_RADIUS_FRACTION = 0.01


def _get_fallback(obj: MathObject) -> RepresentationConfig:
    """Choose a compatible representation based on available geometry."""
    if obj.mesh is not None:
        return RepresentationConfig(type=RepresentationType.SURFACE_SHELL)
    if obj.curves:
        radius = _estimate_tube_radius(obj.curves)
        return RepresentationConfig(
            type=RepresentationType.TUBE, tube_radius=radius
        )
    if obj.point_cloud is not None:
        return RepresentationConfig(type=RepresentationType.SPARSE_SHELL)
    raise ValueError(
        "Cannot determine fallback representation: "
        "MathObject has no mesh, curves, or point_cloud"
    )


def _estimate_tube_radius(curves: list[Curve]) -> float:
    """Estimate a sensible tube radius as 1% of the bounding-box diagonal."""
    all_points = np.concatenate([c.points for c in curves], axis=0)
    bbox_min = all_points.min(axis=0)
    bbox_max = all_points.max(axis=0)
    diagonal = float(np.linalg.norm(bbox_max - bbox_min))
    radius = diagonal * _FALLBACK_TUBE_RADIUS_FRACTION
    return max(radius, 1e-6)


def _resolve_canonical(name: str) -> str:
    """Resolve an alias to its canonical generator name for defaults lookup."""
    try:
        return get_generator_meta(name).name
    except KeyError:
        return name


def get_default(
    generator_name: str, obj: MathObject | None = None
) -> RepresentationConfig:
    """Return the recommended representation config for a generator."""
    config = _GENERATOR_DEFAULTS.get(generator_name)
    if config is not None:
        return config
    canonical = _resolve_canonical(generator_name)
    config = _GENERATOR_DEFAULTS.get(canonical)
    if config is not None:
        return config
    if obj is not None:
        return _get_fallback(obj)
    return RepresentationConfig(type=RepresentationType.SURFACE_SHELL)


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

    handler = _HANDLERS.get(config.type)
    if handler is None:
        raise NotImplementedError(
            f"No handler registered for representation type '{config.type.value}'"
        )

    _validate_input(obj, config)
    result = handler(obj, config)
    result = replace(result, representation=config.type.value)

    logger.info(
        "Applied representation %s to %s",
        config.type.value,
        obj.generator_name or "unnamed",
    )
    return result


def _validate_input(obj: MathObject, config: RepresentationConfig) -> None:
    """Validate MathObject input, allowing scalar_field-only for HEIGHTMAP_RELIEF."""
    if config.type == RepresentationType.HEIGHTMAP_RELIEF:
        if obj.scalar_field is None:
            raise ValueError(
                "HEIGHTMAP_RELIEF requires a scalar_field, "
                "but MathObject has none"
            )
        return
    if config.type == RepresentationType.WEIGHTED_CLOUD:
        if obj.point_cloud is None:
            raise ValueError(
                "WEIGHTED_CLOUD requires a point_cloud input, "
                "but MathObject has no point_cloud"
            )
        return
    obj.validate_or_raise()


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

    if config.tube_sides is None:
        raise ValueError("TUBE representation requires tube_sides to be set")
    sides = config.tube_sides
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

    merged_normals = _merge_normals(meshes)

    return Mesh(
        vertices=np.concatenate(all_vertices, axis=0),
        faces=np.concatenate(all_faces, axis=0),
        normals=merged_normals,
    )


def _merge_normals(meshes: list[Mesh]) -> np.ndarray | None:
    """Merge normals from multiple meshes, or warn if some are missing."""
    has_normals = [m.normals is not None for m in meshes]
    if all(has_normals):
        return np.concatenate([m.normals for m in meshes], axis=0)
    if any(has_normals):
        logger.warning(
            "Dropping normals during mesh merge: %d of %d meshes have normals",
            sum(has_normals),
            len(meshes),
        )
    return None


def _apply_heightmap_relief(
    obj: MathObject, config: RepresentationConfig
) -> MathObject:
    """Extrude a 2D scalar field into a surface mesh."""
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


# Default surface density for SPARSE_SHELL: points per unit area.
# sample_count = total_mesh_area * surface_density.
_SPARSE_SHELL_DEFAULT_SURFACE_DENSITY = 100.0
_SPARSE_SHELL_MIN_SAMPLES = 10
_SPARSE_SHELL_SEED = 42


def _apply_sparse_shell(
    obj: MathObject, config: RepresentationConfig
) -> MathObject:
    """Sample mesh surface at reduced density to create a sparse point cloud.

    Uses surface_density (points per unit area) to determine sample count.
    Sampling is seeded for deterministic output.
    """
    if obj.mesh is None:
        raise ValueError(
            "SPARSE_SHELL requires a mesh input, but MathObject has no mesh"
        )

    density = config.surface_density or _SPARSE_SHELL_DEFAULT_SURFACE_DENSITY
    tm = trimesh.Trimesh(
        vertices=obj.mesh.vertices,
        faces=obj.mesh.faces,
        process=False,
    )

    total_area = float(tm.area)
    sample_count = max(
        _SPARSE_SHELL_MIN_SAMPLES,
        int(total_area * density),
    )

    # trimesh uses numpy's legacy RNG; seed it for deterministic output.
    np.random.seed(_SPARSE_SHELL_SEED)
    points, face_indices = tm.sample(sample_count, return_index=True)
    normals = tm.face_normals[face_indices]

    cloud = PointCloud(
        points=np.asarray(points, dtype=np.float64),
        normals=np.asarray(normals, dtype=np.float64),
    )

    logger.info(
        "SPARSE_SHELL: sampled %d points from %d faces "
        "(surface_density=%.1f pts/unit²)",
        len(cloud.points), len(obj.mesh.faces), density,
    )

    return replace(obj, point_cloud=cloud)


def _apply_weighted_cloud(
    obj: MathObject, config: RepresentationConfig
) -> MathObject:
    """Pass through a point cloud, preserving its per-point intensities."""
    return replace(obj)


_HANDLERS = {
    RepresentationType.SURFACE_SHELL: _apply_surface_shell,
    RepresentationType.RAW_POINT_CLOUD: _apply_raw_point_cloud,
    RepresentationType.TUBE: _apply_tube,
    RepresentationType.HEIGHTMAP_RELIEF: _apply_heightmap_relief,
    RepresentationType.SPARSE_SHELL: _apply_sparse_shell,
    RepresentationType.WEIGHTED_CLOUD: _apply_weighted_cloud,
    RepresentationType.VOLUME_FILL: apply_volume_fill,
    RepresentationType.SLICE_STACK: apply_slice_stack,
    RepresentationType.WIREFRAME: apply_wireframe,
}
