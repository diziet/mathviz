"""Mesh helpers for the reaction-diffusion surface generator.

Provides base surface mesh generation (torus, sphere, Klein bottle),
mesh Laplacian construction, and vertex normal computation.
"""

import numpy as np
from scipy import sparse

from mathviz.core.math_object import Mesh
from mathviz.generators.parametric._mesh_utils import (
    build_sphere_faces,
    build_wrapped_grid_faces,
)

VALID_SURFACES = ("torus", "sphere", "klein_bottle")


def generate_base_mesh(surface: str, resolution: int) -> Mesh:
    """Generate a base surface mesh by name."""
    builders = {
        "torus": _generate_torus,
        "sphere": _generate_sphere,
        "klein_bottle": _generate_klein_bottle,
    }
    if surface not in builders:
        raise ValueError(
            f"Unknown surface {surface!r}, expected one of {VALID_SURFACES}"
        )
    return builders[surface](resolution)


def _generate_torus(resolution: int) -> Mesh:
    """Generate a torus mesh with major_radius=1.0, minor_radius=0.4."""
    n = resolution
    major_r, minor_r = 1.0, 0.4
    u = np.linspace(0, 2 * np.pi, n, endpoint=False)
    v = np.linspace(0, 2 * np.pi, n, endpoint=False)
    uu, vv = np.meshgrid(u, v, indexing="ij")

    x = (major_r + minor_r * np.cos(vv)) * np.cos(uu)
    y = (major_r + minor_r * np.cos(vv)) * np.sin(uu)
    z = minor_r * np.sin(vv)

    vertices = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    faces = build_wrapped_grid_faces(n, n)
    return Mesh(vertices=vertices.astype(np.float64), faces=faces)


def _generate_sphere(resolution: int) -> Mesh:
    """Generate a UV-sphere mesh with radius 1.0."""
    n_lat = resolution
    n_lon = resolution
    lat = np.linspace(-np.pi / 2, np.pi / 2, n_lat + 2)[1:-1]
    lon = np.linspace(0, 2 * np.pi, n_lon, endpoint=False)
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")

    x = np.cos(lat_grid) * np.cos(lon_grid)
    y = np.cos(lat_grid) * np.sin(lon_grid)
    z = np.sin(lat_grid)

    body = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    south_pole = np.array([[0.0, 0.0, -1.0]])
    north_pole = np.array([[0.0, 0.0, 1.0]])
    vertices = np.concatenate([body, south_pole, north_pole], axis=0)

    faces = build_sphere_faces(n_lat, n_lon)
    return Mesh(vertices=vertices.astype(np.float64), faces=faces)


def _generate_klein_bottle(resolution: int) -> Mesh:
    """Generate a figure-8 Klein bottle mesh with scale 1.0."""
    n = resolution
    scale = 1.0
    u = np.linspace(0, 2 * np.pi, n, endpoint=False)
    v = np.linspace(0, 2 * np.pi, n, endpoint=False)
    uu, vv = np.meshgrid(u, v, indexing="ij")

    half_u = uu / 2.0
    cos_hu = np.cos(half_u)
    sin_hu = np.sin(half_u)
    sin_v = np.sin(vv)
    sin_2v = np.sin(2.0 * vv)

    r = scale * (2.0 + cos_hu * sin_v - sin_hu * sin_2v)
    x = r * np.cos(uu)
    y = r * np.sin(uu)
    z = scale * (sin_hu * sin_v + cos_hu * sin_2v)

    vertices = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    faces = build_wrapped_grid_faces(n, n)
    return Mesh(vertices=vertices.astype(np.float64), faces=faces)


def build_mesh_laplacian(
    faces: np.ndarray, num_vertices: int,
) -> sparse.csr_matrix:
    """Build a normalized graph Laplacian from mesh face connectivity.

    Returns a sparse matrix L such that L @ f computes the Laplacian of
    a per-vertex scalar field f. Uses the cotangent-weight-free
    (combinatorial) Laplacian: L = A / degree - I, where A is adjacency.
    """
    edges = np.concatenate([
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [0, 2]],
    ], axis=0)
    edges = np.concatenate([edges, edges[:, ::-1]], axis=0)

    rows = edges[:, 0]
    cols = edges[:, 1]
    data = np.ones(len(rows), dtype=np.float64)
    adjacency = sparse.coo_matrix(
        (data, (rows, cols)), shape=(num_vertices, num_vertices),
    ).tocsr()
    # Ensure binary adjacency (no double-counted edges)
    adjacency.data[:] = 1.0

    degree = np.array(adjacency.sum(axis=1)).flatten()
    degree[degree == 0] = 1.0  # avoid division by zero for isolated verts
    inv_degree = sparse.diags(1.0 / degree)
    identity = sparse.eye(num_vertices, format="csr")
    return inv_degree @ adjacency - identity


def compute_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Compute per-vertex normals by averaging adjacent face normals."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)

    vertex_normals = np.zeros_like(vertices)
    for i in range(3):
        np.add.at(vertex_normals, faces[:, i], face_normals)

    norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return vertex_normals / norms
