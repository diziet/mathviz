"""Spherical Voronoi tessellation with raised cell boundary ridges.

Distributes seed points on a sphere using a perturbed Fibonacci spiral,
computes the spherical Voronoi diagram via scipy.spatial.SphericalVoronoi,
and produces either ridge curves, cell face meshes, or both.

For ``ridges_only``, the generator returns curves and the representation
layer handles thickening (matching the voronoi_3d pattern).
For ``cells_only`` and ``both``, it returns triangle meshes directly.
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

logger = logging.getLogger(__name__)

_DEFAULT_NUM_CELLS = 64
_DEFAULT_RADIUS = 1.0
_DEFAULT_EDGE_WIDTH = 0.05
_DEFAULT_EDGE_HEIGHT = 0.1
_DEFAULT_CELL_STYLE = "ridges_only"
_DEFAULT_ARC_RESOLUTION = 16
_VALID_CELL_STYLES = {"ridges_only", "cells_only", "both"}
_MIN_NUM_CELLS = 4


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
    if edge_width == 0.0 and cell_style in ("ridges_only", "both"):
        raise ValueError(
            "edge_width must be > 0 when cell_style involves ridges"
        )
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

    theta = 2.0 * np.pi * indices / golden_ratio
    phi = np.arccos(1.0 - 2.0 * (indices + 0.5) / num_points)

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
    arc_resolution: int,
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
                sv.vertices[v0], sv.vertices[v1],
                ridge_radius, arc_resolution,
            )
            if len(arc_pts) >= 2:
                curves.append(Curve(points=arc_pts, closed=False))

    return curves


def _arc_on_sphere(
    p0: np.ndarray, p1: np.ndarray, radius: float, arc_resolution: int,
) -> np.ndarray:
    """Interpolate a great-circle arc between two points, projected to radius."""
    n0 = p0 / np.linalg.norm(p0)
    n1 = p1 / np.linalg.norm(p1)
    dot = np.clip(np.dot(n0, n1), -1.0, 1.0)
    angle = np.arccos(dot)

    if angle < 1e-10:
        return np.array([n0 * radius, n1 * radius], dtype=np.float64)

    t_values = np.linspace(0.0, 1.0, arc_resolution)
    sin_angle = np.sin(angle)
    weights_0 = np.sin((1.0 - t_values) * angle) / sin_angle
    weights_1 = np.sin(t_values * angle) / sin_angle

    pts = weights_0[:, np.newaxis] * n0 + weights_1[:, np.newaxis] * n1
    pts *= radius
    return pts.astype(np.float64)


def _normalize_cell_verts(
    verts: np.ndarray, radius: float,
) -> np.ndarray:
    """Project Voronoi cell vertices onto the sphere at exact radius."""
    cell_verts = verts.astype(np.float64)
    norms = np.linalg.norm(cell_verts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return cell_verts / norms * radius


def _build_cell_faces_mesh(
    sv: SphericalVoronoi, radius: float,
) -> Mesh:
    """Triangulate each Voronoi cell polygon into a surface mesh."""
    cell_meshes: list[Mesh] = []
    for region in sv.regions:
        if len(region) < 3:
            continue
        cell_verts = _normalize_cell_verts(sv.vertices[region], radius)
        tri_faces = _fan_triangulate(len(cell_verts))
        cell_meshes.append(Mesh(vertices=cell_verts, faces=tri_faces))
    return _merge_meshes(cell_meshes)


def _fan_triangulate(num_verts: int) -> np.ndarray:
    """Fan-triangulate a convex polygon with num_verts vertices."""
    faces = []
    for i in range(1, num_verts - 1):
        faces.append([0, i, i + 1])
    return np.array(faces, dtype=np.int64)


def _merge_meshes(meshes: list[Mesh]) -> Mesh:
    """Merge multiple meshes into one with offset face indices."""
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


def _collect_all_curve_points(curves: list[Curve]) -> np.ndarray:
    """Gather all points from curves into a single array for bounding box."""
    if not curves:
        return np.zeros((1, 3), dtype=np.float64)
    return np.concatenate([c.points for c in curves], axis=0)


@register
class VoronoiSphereGenerator(GeneratorBase):
    """Spherical Voronoi tessellation with raised cell boundary ridges."""

    name = "voronoi_sphere"
    category = "geometry"
    aliases = ()
    description = "Voronoi tessellation on a sphere surface with geodesic cells"
    resolution_params = {"arc_resolution": "Points per great-circle arc segment"}
    _resolution_defaults = {"arc_resolution": _DEFAULT_ARC_RESOLUTION}

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

        res_defaults = self.get_default_resolution()
        res_defaults.update(resolution_kwargs)
        arc_resolution = int(res_defaults["arc_resolution"])

        _validate_params(num_cells, radius, edge_width, edge_height, cell_style)

        points = _fibonacci_sphere_points(num_cells, radius, seed)
        sv = _compute_spherical_voronoi(points, radius)

        return _build_math_object(
            self, sv, radius, edge_height, cell_style,
            arc_resolution, merged, seed,
        )

    def get_default_representation(self) -> RepresentationConfig:
        """Return TUBE as the default representation for ridge curves."""
        return RepresentationConfig(type=RepresentationType.TUBE)


def _build_math_object(
    gen: VoronoiSphereGenerator, sv: SphericalVoronoi,
    radius: float, edge_height: float,
    cell_style: str, arc_resolution: int,
    merged: dict[str, Any], seed: int,
) -> MathObject:
    """Build the MathObject with correct geometry type per cell_style."""
    if cell_style == "ridges_only":
        return _build_ridges_only(
            gen, sv, radius, edge_height, arc_resolution, merged, seed,
        )
    if cell_style == "cells_only":
        return _build_cells_only(gen, sv, radius, merged, seed)
    return _build_both(
        gen, sv, radius, edge_height, arc_resolution, merged, seed,
    )


def _build_ridges_only(
    gen: VoronoiSphereGenerator, sv: SphericalVoronoi,
    radius: float, edge_height: float, arc_resolution: int,
    merged: dict[str, Any], seed: int,
) -> MathObject:
    """Return curves for ridges — representation layer handles thickening."""
    curves = _extract_ridge_curves(sv, radius, edge_height, arc_resolution)
    all_pts = _collect_all_curve_points(curves)
    bbox = BoundingBox.from_points(all_pts)

    logger.info(
        "Generated voronoi_sphere: cells=%d, style=ridges_only, edges=%d, seed=%d",
        int(merged["num_cells"]), len(curves), seed,
    )
    return MathObject(
        curves=curves,
        generator_name=gen.name,
        category=gen.category,
        parameters=merged,
        seed=seed,
        bounding_box=bbox,
    )


def _build_cells_only(
    gen: VoronoiSphereGenerator, sv: SphericalVoronoi,
    radius: float, merged: dict[str, Any], seed: int,
) -> MathObject:
    """Return mesh of triangulated cell faces."""
    mesh = _build_cell_faces_mesh(sv, radius)
    bbox = BoundingBox.from_points(mesh.vertices)

    logger.info(
        "Generated voronoi_sphere: cells=%d, style=cells_only, verts=%d, seed=%d",
        int(merged["num_cells"]), len(mesh.vertices), seed,
    )
    return MathObject(
        mesh=mesh,
        generator_name=gen.name,
        category=gen.category,
        parameters=merged,
        seed=seed,
        bounding_box=bbox,
    )


def _build_both(
    gen: VoronoiSphereGenerator, sv: SphericalVoronoi,
    radius: float, edge_height: float,
    arc_resolution: int, merged: dict[str, Any], seed: int,
) -> MathObject:
    """Return mesh of cell faces plus curves for ridges."""
    mesh = _build_cell_faces_mesh(sv, radius)
    curves = _extract_ridge_curves(sv, radius, edge_height, arc_resolution)
    all_pts = np.concatenate(
        [mesh.vertices, _collect_all_curve_points(curves)], axis=0,
    )
    bbox = BoundingBox.from_points(all_pts)

    logger.info(
        "Generated voronoi_sphere: cells=%d, style=both, verts=%d, edges=%d, seed=%d",
        int(merged["num_cells"]), len(mesh.vertices), len(curves), seed,
    )
    return MathObject(
        mesh=mesh,
        curves=curves,
        generator_name=gen.name,
        category=gen.category,
        parameters=merged,
        seed=seed,
        bounding_box=bbox,
    )
