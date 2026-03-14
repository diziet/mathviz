"""Representation handlers for VOLUME_FILL, SLICE_STACK, and WIREFRAME.

These complete the full set of fabrication strategies by providing:
- VOLUME_FILL: interior point cloud from watertight mesh
- SLICE_STACK: parallel plane slicing with contour extraction
- WIREFRAME: edge extraction with tube thickening
"""

import logging
from dataclasses import replace

import numpy as np
import trimesh

from mathviz.core.math_object import Curve, MathObject, Mesh, PointCloud
from mathviz.core.representation import RepresentationConfig
from mathviz.shared.tube_thickening import thicken_curve

logger = logging.getLogger(__name__)

# Default volume density: points per cubic unit.
_VOLUME_FILL_DEFAULT_DENSITY = 50.0
_VOLUME_FILL_SEED = 42

# Default slice count for SLICE_STACK.
_SLICE_STACK_DEFAULT_COUNT = 10

# Default wireframe tube parameters.
_WIREFRAME_DEFAULT_THICKNESS = 0.02
_WIREFRAME_DEFAULT_SIDES = 6


def apply_volume_fill(
    obj: MathObject, config: RepresentationConfig
) -> MathObject:
    """Fill the interior of a watertight mesh with points."""
    if obj.mesh is None:
        raise ValueError(
            "VOLUME_FILL requires a mesh input, but MathObject has no mesh"
        )

    tm = trimesh.Trimesh(
        vertices=obj.mesh.vertices, faces=obj.mesh.faces, process=False
    )
    if not tm.is_watertight:
        raise ValueError("VOLUME_FILL requires a watertight mesh")

    density = config.volume_density or _VOLUME_FILL_DEFAULT_DENSITY
    points = _fill_volume(tm, density)
    cloud = PointCloud(points=points.astype(np.float64))

    logger.info(
        "VOLUME_FILL: generated %d interior points (density=%.1f pts/unit³)",
        len(cloud.points), density,
    )
    return replace(obj, point_cloud=cloud)


def _fill_volume(tm: trimesh.Trimesh, density: float) -> np.ndarray:
    """Generate interior points for a watertight mesh via jittered grid."""
    mesh_volume = abs(float(tm.volume))
    target_count = max(1, int(round(density * mesh_volume)))

    bounds = tm.bounds  # (2, 3): [min, max]
    bbox_size = bounds[1] - bounds[0]
    bbox_volume = float(np.prod(bbox_size))

    fill_ratio = mesh_volume / max(bbox_volume, 1e-15)
    adjusted_count = int(round(target_count / max(fill_ratio, 0.01)))
    spacing = (bbox_volume / max(adjusted_count, 1)) ** (1.0 / 3.0)

    rng = np.random.default_rng(_VOLUME_FILL_SEED)
    axes = [np.arange(bounds[0][i], bounds[1][i], spacing) for i in range(3)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    jitter = rng.uniform(-spacing * 0.4, spacing * 0.4, size=grid.shape)
    candidates = grid + jitter

    inside_mask = tm.contains(candidates)
    interior = candidates[inside_mask]

    if len(interior) == 0:
        raise ValueError("VOLUME_FILL produced no interior points")

    return interior


def apply_slice_stack(
    obj: MathObject, config: RepresentationConfig
) -> MathObject:
    """Slice a mesh with parallel planes and extract contour points."""
    if obj.mesh is None:
        raise ValueError(
            "SLICE_STACK requires a mesh input, but MathObject has no mesh"
        )

    slice_count = config.slice_count or _SLICE_STACK_DEFAULT_COUNT
    axis = config.slice_axis
    points = _slice_mesh(obj.mesh, slice_count, axis)
    cloud = PointCloud(points=points.astype(np.float64))

    logger.info(
        "SLICE_STACK: %d points from %d slices along %s axis",
        len(cloud.points), slice_count, axis,
    )
    return replace(obj, point_cloud=cloud)


def _slice_mesh(mesh: Mesh, slice_count: int, axis: str) -> np.ndarray:
    """Cut parallel slices through a mesh and collect contour points."""
    axis_idx = {"x": 0, "y": 1, "z": 2}[axis]
    normal = np.zeros(3)
    normal[axis_idx] = 1.0

    coords = mesh.vertices[:, axis_idx]
    lo, hi = float(coords.min()), float(coords.max())
    margin = (hi - lo) * 0.01
    origins = np.linspace(lo + margin, hi - margin, slice_count)

    tm = trimesh.Trimesh(
        vertices=mesh.vertices, faces=mesh.faces, process=False
    )

    all_points: list[np.ndarray] = []
    for origin_val in origins:
        origin = np.zeros(3)
        origin[axis_idx] = origin_val
        section = tm.section(plane_origin=origin, plane_normal=normal)
        if section is None:
            continue
        discrete = section.to_planar()[0]
        if discrete is None:
            continue
        verts_2d = np.asarray(discrete.vertices)
        if len(verts_2d) == 0:
            continue
        verts_3d = _embed_slice_points(verts_2d, axis_idx, origin_val)
        all_points.append(verts_3d)

    if not all_points:
        raise ValueError("SLICE_STACK produced no contour points")

    return np.concatenate(all_points, axis=0)


def _embed_slice_points(
    verts_2d: np.ndarray, axis_idx: int, axis_val: float
) -> np.ndarray:
    """Re-embed 2D slice vertices into 3D by inserting the slice axis coordinate."""
    n = len(verts_2d)
    verts_3d = np.zeros((n, 3), dtype=np.float64)
    other_axes = [i for i in range(3) if i != axis_idx]
    verts_3d[:, other_axes[0]] = verts_2d[:, 0]
    verts_3d[:, other_axes[1]] = verts_2d[:, 1]
    verts_3d[:, axis_idx] = axis_val
    return verts_3d


def apply_wireframe(
    obj: MathObject, config: RepresentationConfig
) -> MathObject:
    """Extract mesh edges and thicken into thin tubes."""
    if obj.mesh is None:
        raise ValueError(
            "WIREFRAME requires a mesh input, but MathObject has no mesh"
        )

    thickness = config.wireframe_thickness or _WIREFRAME_DEFAULT_THICKNESS
    sides = _WIREFRAME_DEFAULT_SIDES
    radius = thickness / 2.0

    edges = _extract_unique_edges(obj.mesh)
    edge_curves = _edges_to_curves(obj.mesh.vertices, edges)
    meshes = [thicken_curve(c, radius, sides) for c in edge_curves]
    merged = _merge_meshes(meshes)

    logger.info(
        "WIREFRAME: %d edges → %d vertices, %d faces (thickness=%.4f)",
        len(edges), len(merged.vertices), len(merged.faces), thickness,
    )
    return replace(obj, mesh=merged)


def _extract_unique_edges(mesh: Mesh) -> np.ndarray:
    """Extract unique edges from mesh faces as (E, 2) array."""
    faces = mesh.faces
    edges = np.concatenate([
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [2, 0]],
    ], axis=0)
    edges = np.sort(edges, axis=1)
    return np.unique(edges, axis=0)


def _edges_to_curves(vertices: np.ndarray, edges: np.ndarray) -> list[Curve]:
    """Convert edge pairs into short 2-point Curves."""
    return [
        Curve(points=vertices[edge].astype(np.float64), closed=False)
        for edge in edges
    ]


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
