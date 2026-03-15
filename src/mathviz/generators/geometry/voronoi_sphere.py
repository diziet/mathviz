"""Spherical Voronoi tessellation with raised cell boundary ridges.

Distributes seed points on a sphere using a perturbed Fibonacci spiral,
computes the spherical Voronoi diagram via scipy.spatial.SphericalVoronoi,
and produces either ridge tubes, cell face meshes, or both.
"""

import logging
from typing import Any

import numpy as np
from numpy.random import default_rng
from scipy.spatial import SphericalVoronoi

from mathviz.core.generator import GeneratorBase, register
from mathviz.core.math_object import (
    BoundingBox,
    Curve,
    MathObject,
    Mesh,
)
from mathviz.core.representation import RepresentationConfig, RepresentationType
from mathviz.shared.tube_thickening import thicken_curve

logger = logging.getLogger(__name__)

_DEFAULT_NUM_CELLS = 64
_DEFAULT_RADIUS = 1.0
_DEFAULT_EDGE_WIDTH = 0.05
_DEFAULT_EDGE_HEIGHT = 0.1
_DEFAULT_CELL_STYLE = "ridges_only"
_VALID_CELL_STYLES = {"ridges_only", "cells_only", "both"}
_MIN_NUM_CELLS = 4
_ARC_RESOLUTION = 16


def _validate_params(
    num_cells: int, radius: float, edge_width: float, edge_height: float,
    cell_style: str,
) -> None:
    """Validate spherical Voronoi parameters."""
    if num_cells < _MIN_NUM_CELLS:
        raise ValueError(
            f"num_cells must be >= {_MIN_NUM_CELLS}, got {num_cells}"
        )
    if radius <= 0:
        raise ValueError(f"radius must be positive, got {radius}")
    if edge_width < 0:
        raise ValueError(f"edge_width must be >= 0, got {edge_width}")
    if edge_height < 0:
        raise ValueError(f"edge_height must be >= 0, got {edge_height}")
    if cell_style not in _VALID_CELL_STYLES:
        raise ValueError(
            f"cell_style must be one of {_VALID_CELL_STYLES}, got {cell_style!r}"
        )


def _fibonacci_sphere_points(
    num_points: int, radius: float, seed: int,
) -> np.ndarray:
    """Generate points on a sphere via perturbed Fibonacci spiral."""
    rng = default_rng(seed)
    golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0
    indices = np.arange(num_points, dtype=np.float64)

    # Fibonacci spiral latitude/longitude
    theta = 2.0 * np.pi * indices / golden_ratio
    phi = np.arccos(1.0 - 2.0 * (indices + 0.5) / num_points)

    # Perturb angles with seed-dependent noise
    perturbation = 0.3 / np.sqrt(num_points)
    theta += rng.uniform(-perturbation, perturbation, num_points)
    phi += rng.uniform(-perturbation, perturbation, num_points)
    phi = np.clip(phi, 0.01, np.pi - 0.01)

    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    return np.column_stack([x, y, z])


def _compute_spherical_voronoi(
    points: np.ndarray, radius: float,
) -> SphericalVoronoi:
    """Compute spherical Voronoi diagram and sort region vertices."""
    center = np.array([0.0, 0.0, 0.0])
    sv = SphericalVoronoi(points, radius=radius, center=center)
    sv.sort_vertices_of_regions()
    return sv


def _extract_ridge_curves(
    sv: SphericalVoronoi, radius: float, edge_height: float,
) -> list[Curve]:
    """Extract unique cell boundary edges as curves on the sphere."""
    seen_edges: set[tuple[int, int]] = set()
    curves: list[Curve] = []
    ridge_radius = radius + edge_height

    for region in sv.regions:
        num_verts = len(region)
        for i in range(num_verts):
            v0 = region[i]
            v1 = region[(i + 1) % num_verts]
            edge_key = (min(v0, v1), max(v0, v1))
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)

            arc_pts = _arc_on_sphere(
                sv.vertices[v0], sv.vertices[v1], ridge_radius,
            )
            if len(arc_pts) >= 2:
                curves.append(Curve(points=arc_pts, closed=False))

    return curves


def _arc_on_sphere(
    p0: np.ndarray, p1: np.ndarray, radius: float,
) -> np.ndarray:
    """Interpolate a great-circle arc between two points, projected to radius."""
    n0 = p0 / np.linalg.norm(p0)
    n1 = p1 / np.linalg.norm(p1)
    dot = np.clip(np.dot(n0, n1), -1.0, 1.0)
    angle = np.arccos(dot)

    if angle < 1e-10:
        return np.array([n0 * radius, n1 * radius], dtype=np.float64)

    t_values = np.linspace(0.0, 1.0, _ARC_RESOLUTION)
    sin_angle = np.sin(angle)
    weights_0 = np.sin((1.0 - t_values) * angle) / sin_angle
    weights_1 = np.sin(t_values * angle) / sin_angle

    pts = weights_0[:, np.newaxis] * n0 + weights_1[:, np.newaxis] * n1
    pts *= radius
    return pts.astype(np.float64)


def _build_ridge_mesh(
    curves: list[Curve], edge_width: float,
) -> Mesh:
    """Thicken ridge curves into tube meshes and merge."""
    all_verts: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    vertex_offset = 0

    tube_radius = max(edge_width / 2.0, 1e-6)
    for curve in curves:
        tube = thicken_curve(curve, radius=tube_radius, sides=8)
        all_verts.append(tube.vertices)
        all_faces.append(tube.faces + vertex_offset)
        vertex_offset += len(tube.vertices)

    vertices = np.concatenate(all_verts, axis=0)
    faces = np.concatenate(all_faces, axis=0)
    return Mesh(vertices=vertices.astype(np.float64), faces=faces.astype(np.int64))


def _build_cell_faces_mesh(
    sv: SphericalVoronoi, radius: float,
) -> Mesh:
    """Triangulate each Voronoi cell polygon into a surface mesh."""
    all_verts: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    vertex_offset = 0

    for region in sv.regions:
        if len(region) < 3:
            continue
        cell_verts = sv.vertices[region].astype(np.float64)
        # Normalize to exact radius
        norms = np.linalg.norm(cell_verts, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        cell_verts = cell_verts / norms * radius

        tri_faces = _fan_triangulate(len(cell_verts))
        all_verts.append(cell_verts)
        all_faces.append(tri_faces + vertex_offset)
        vertex_offset += len(cell_verts)

    vertices = np.concatenate(all_verts, axis=0)
    faces = np.concatenate(all_faces, axis=0)
    return Mesh(vertices=vertices.astype(np.float64), faces=faces.astype(np.int64))


def _fan_triangulate(num_verts: int) -> np.ndarray:
    """Fan-triangulate a convex polygon with num_verts vertices."""
    faces = []
    for i in range(1, num_verts - 1):
        faces.append([0, i, i + 1])
    return np.array(faces, dtype=np.int64)


def _merge_meshes(meshes: list[Mesh]) -> Mesh:
    """Merge multiple meshes into one."""
    all_verts: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    offset = 0
    for mesh in meshes:
        all_verts.append(mesh.vertices)
        all_faces.append(mesh.faces + offset)
        offset += len(mesh.vertices)
    return Mesh(
        vertices=np.concatenate(all_verts, axis=0).astype(np.float64),
        faces=np.concatenate(all_faces, axis=0).astype(np.int64),
    )


@register
class VoronoiSphereGenerator(GeneratorBase):
    """Spherical Voronoi tessellation with raised cell boundary ridges."""

    name = "voronoi_sphere"
    category = "geometry"
    aliases = ()
    description = "Voronoi tessellation on a sphere surface with geodesic cells"
    resolution_params = {}

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters for voronoi_sphere."""
        return {
            "num_cells": _DEFAULT_NUM_CELLS,
            "radius": _DEFAULT_RADIUS,
            "edge_width": _DEFAULT_EDGE_WIDTH,
            "edge_height": _DEFAULT_EDGE_HEIGHT,
            "cell_style": _DEFAULT_CELL_STYLE,
        }

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate spherical Voronoi tessellation geometry."""
        merged = self.get_default_params()
        if params:
            merged.update(params)

        num_cells = int(merged["num_cells"])
        radius = float(merged["radius"])
        edge_width = float(merged["edge_width"])
        edge_height = float(merged["edge_height"])
        cell_style = str(merged["cell_style"])

        _validate_params(num_cells, radius, edge_width, edge_height, cell_style)

        points = _fibonacci_sphere_points(num_cells, radius, seed)
        sv = _compute_spherical_voronoi(points, radius)

        mesh = _build_geometry(sv, radius, edge_width, edge_height, cell_style)

        bbox = BoundingBox.from_points(mesh.vertices)

        logger.info(
            "Generated voronoi_sphere: cells=%d, style=%s, verts=%d, seed=%d",
            num_cells, cell_style, len(mesh.vertices), seed,
        )

        return MathObject(
            mesh=mesh,
            generator_name=self.name,
            category=self.category,
            parameters=merged,
            seed=seed,
            bounding_box=bbox,
        )

    def get_default_representation(self) -> RepresentationConfig:
        """Return TUBE as the default representation for ridge geometry."""
        return RepresentationConfig(type=RepresentationType.TUBE)


def _build_geometry(
    sv: SphericalVoronoi, radius: float, edge_width: float,
    edge_height: float, cell_style: str,
) -> Mesh:
    """Build the mesh geometry based on cell_style."""
    if cell_style == "ridges_only":
        curves = _extract_ridge_curves(sv, radius, edge_height)
        return _build_ridge_mesh(curves, edge_width)
    if cell_style == "cells_only":
        return _build_cell_faces_mesh(sv, radius)

    # "both": combine ridges and cell faces
    curves = _extract_ridge_curves(sv, radius, edge_height)
    ridge_mesh = _build_ridge_mesh(curves, edge_width)
    cell_mesh = _build_cell_faces_mesh(sv, radius)
    return _merge_meshes([ridge_mesh, cell_mesh])
