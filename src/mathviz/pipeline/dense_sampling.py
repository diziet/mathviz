"""Post-transform dense sampling: sample mesh surface and edges after physical-space scaling."""

import logging
import threading
from dataclasses import replace
from typing import Any

import numpy as np
import trimesh

from mathviz.core.math_object import MathObject, PointCloud

logger = logging.getLogger(__name__)

MAX_DENSE_SAMPLES = 200_000
MAX_RESOLUTION_SCALED_SAMPLES = 500_000
_DENSE_SURFACE_DENSITY = 100.0
_DENSE_SEED = 42
_MIN_SAMPLES = 10
_DENSE_EDGE_FRACTION = 0.3

# trimesh.sample relies on numpy's legacy global RNG.  Protect the
# seed-then-sample sequence so concurrent threads don't interleave.
_rng_lock = threading.Lock()


def _sample_mesh_surface(
    obj: MathObject,
    surface_density: float,
    max_samples: int,
) -> tuple[PointCloud, float]:
    """Sample mesh surface, returning (cloud, total_area).

    Builds a trimesh, computes sample count from area × density (capped),
    and uses deterministic RNG-locked sampling.
    """
    assert obj.mesh is not None  # caller must check  # noqa: S101

    tm = trimesh.Trimesh(
        vertices=obj.mesh.vertices,
        faces=obj.mesh.faces,
        process=False,
    )

    total_area = float(tm.area)
    sample_count = max(_MIN_SAMPLES, int(total_area * surface_density))
    sample_count = min(sample_count, max_samples)

    with _rng_lock:
        np.random.seed(_DENSE_SEED)
        points, face_indices = tm.sample(sample_count, return_index=True)
    normals = tm.face_normals[face_indices]

    cloud = PointCloud(
        points=np.asarray(points, dtype=np.float64),
        normals=np.asarray(normals, dtype=np.float64),
    )
    return cloud, total_area


def _sample_mesh_edges(
    obj: MathObject,
    max_samples: int,
) -> PointCloud:
    """Sample points along mesh edges, distributing proportionally to edge length.

    Extracts unique edges from the mesh, computes each edge's length, then
    allocates points to each edge proportionally.  Points are interpolated
    at uniform spacing along each edge.
    """
    assert obj.mesh is not None  # noqa: S101
    mesh = obj.mesh

    if mesh.faces is None or len(mesh.faces) == 0:
        raise ValueError("Edge sampling requires a mesh with faces")

    # Extract unique edges
    faces = mesh.faces
    raw_edges = np.concatenate([
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [2, 0]],
    ], axis=0)
    raw_edges = np.sort(raw_edges, axis=1)
    edges = np.unique(raw_edges, axis=0)

    if len(edges) == 0:
        raise ValueError("Mesh has no edges to sample")

    verts = mesh.vertices
    starts = verts[edges[:, 0]]
    ends = verts[edges[:, 1]]
    lengths = np.linalg.norm(ends - starts, axis=1)
    total_length = lengths.sum()

    if total_length <= 0:
        raise ValueError("All mesh edges have zero length")

    sample_count = max(_MIN_SAMPLES, max_samples)

    # Distribute points proportionally to edge length
    raw_counts = (lengths / total_length) * sample_count
    floor_counts = np.floor(raw_counts).astype(int)
    remainder = sample_count - floor_counts.sum()

    # Distribute remaining points to edges with largest fractional parts
    if remainder > 0:
        fractions = raw_counts - floor_counts
        top_indices = np.argsort(fractions)[-remainder:]
        floor_counts[top_indices] += 1

    # Interpolate points along edges
    all_points = []
    for i, count in enumerate(floor_counts):
        if count <= 0:
            continue
        t_values = np.linspace(0.0, 1.0, count + 2)[1:-1]  # exclude endpoints
        if len(t_values) == 0:
            t_values = np.array([0.5])
        edge_points = starts[i] + t_values[:, np.newaxis] * (ends[i] - starts[i])
        all_points.append(edge_points)

    if not all_points:
        # Fallback: sample one point per edge at midpoint
        midpoints = (starts + ends) / 2.0
        all_points.append(midpoints[:max_samples])

    points = np.concatenate(all_points, axis=0)[:max_samples]

    return PointCloud(
        points=np.asarray(points, dtype=np.float64),
    )


def apply_edge_sampling(
    obj: MathObject,
    *,
    max_samples: int = MAX_DENSE_SAMPLES,
) -> MathObject:
    """Sample points along mesh edges only, creating a wireframe-like cloud."""
    if obj.mesh is None:
        raise ValueError("Edge sampling requires a mesh")

    cloud = _sample_mesh_edges(obj, max_samples)

    logger.info(
        "Edge sampling: %d points along edges (cap=%d)",
        len(cloud.points),
        max_samples,
    )

    return replace(obj, point_cloud=cloud)


def apply_post_transform_sampling(
    obj: MathObject,
    *,
    max_samples: int = MAX_DENSE_SAMPLES,
    surface_density: float = _DENSE_SURFACE_DENSITY,
    edge_fraction: float = _DENSE_EDGE_FRACTION,
) -> MathObject:
    """Sample the mesh surface and edges, combining into a single dense cloud.

    Splits the sample budget: (1 - edge_fraction) for surface samples,
    edge_fraction for edge samples.  The two are merged into one cloud.
    """
    if obj.mesh is None:
        raise ValueError("Post-transform sampling requires a mesh")

    edge_budget = int(max_samples * edge_fraction)
    surface_budget = max_samples - edge_budget

    surface_cloud, total_area = _sample_mesh_surface(obj, surface_density, surface_budget)
    edge_cloud = _sample_mesh_edges(obj, edge_budget)

    combined_points = np.concatenate([surface_cloud.points, edge_cloud.points], axis=0)
    # Surface normals exist; edge points have none — pad with zeros
    if surface_cloud.normals is not None:
        edge_normals = np.zeros_like(edge_cloud.points)
        combined_normals = np.concatenate([surface_cloud.normals, edge_normals], axis=0)
    else:
        combined_normals = None

    cloud = PointCloud(
        points=combined_points[:max_samples],
        normals=combined_normals[:max_samples] if combined_normals is not None else None,
    )

    logger.info(
        "Dense sampling: %d points (%d surface + %d edge) from %.1f mm² (cap=%d)",
        len(cloud.points),
        len(surface_cloud.points),
        len(edge_cloud.points),
        total_area,
        max_samples,
    )

    return replace(obj, point_cloud=cloud)


def _compute_resolution_scale(
    resolution_kwargs: dict[str, Any],
    default_resolution: dict[str, Any],
) -> float:
    """Compute the squared ratio of requested to default resolution.

    Finds the first matching numeric resolution parameter between the two
    dicts and returns (requested / default)².  Returns 1.0 when no match
    is found or when the default is zero.

    Generators typically expose a single resolution parameter (e.g.
    ``voxel_resolution`` or ``grid_resolution``).  If a generator has
    multiple numeric resolution keys and the caller supplies more than one,
    only the first iteration-order match is used.
    """
    for key, default_val in default_resolution.items():
        if key not in resolution_kwargs:
            continue
        requested_val = resolution_kwargs[key]
        if not isinstance(default_val, (int, float)) or default_val <= 0:
            continue
        if not isinstance(requested_val, (int, float)) or requested_val <= 0:
            continue
        ratio = float(requested_val) / float(default_val)
        return ratio * ratio
    return 1.0


def apply_resolution_scaled_sampling(
    obj: MathObject,
    *,
    resolution_kwargs: dict[str, Any],
    default_resolution: dict[str, Any],
    max_samples: int = MAX_RESOLUTION_SCALED_SAMPLES,
    base_density: float = _DENSE_SURFACE_DENSITY,
) -> MathObject:
    """Sample the mesh surface with density scaled by resolution ratio.

    Multiplies the base surface density by (resolution / default)² so that
    higher-resolution meshes produce proportionally denser point clouds.
    """
    if obj.mesh is None:
        raise ValueError("Resolution-scaled sampling requires a mesh")

    scale = _compute_resolution_scale(resolution_kwargs, default_resolution)
    surface_density = base_density * scale

    cloud, total_area = _sample_mesh_surface(obj, surface_density, max_samples)

    logger.info(
        "Resolution-scaled sampling: %d points from %.1f mm² surface "
        "(density=%.1f, scale=%.2fx, cap=%d)",
        len(cloud.points),
        total_area,
        surface_density,
        scale,
        max_samples,
    )

    return replace(obj, point_cloud=cloud)
