"""Weaire-Phelan foam structure generator.

Generates the Weaire-Phelan structure — the most efficient known foam
partition of space into equal-volume cells. The unit cell contains
2 irregular dodecahedra and 6 tetrakaidecahedra, derived from the
Voronoi tessellation of the A15 crystal structure.
"""

import logging
from typing import Any

import numpy as np
from scipy.spatial import Voronoi

from mathviz.core.generator import GeneratorBase, register
from mathviz.core.math_object import BoundingBox, Curve, Mesh, MathObject
from mathviz.core.representation import RepresentationConfig, RepresentationType

logger = logging.getLogger(__name__)

# A15 crystal structure seed points in fractional coords [0, 1).
# 2 BCC sites → irregular dodecahedra; 6 chain sites → tetrakaidecahedra.
_A15_SITES = np.array([
    [0.0, 0.0, 0.0],
    [0.5, 0.5, 0.5],
    [0.25, 0.0, 0.5],
    [0.75, 0.0, 0.5],
    [0.5, 0.25, 0.0],
    [0.5, 0.75, 0.0],
    [0.0, 0.5, 0.25],
    [0.0, 0.5, 0.75],
], dtype=np.float64)

CELLS_PER_UNIT = 8
_DEFAULT_CELLS_PER_AXIS = 2


def _validate_params(cells_per_axis: int) -> None:
    """Validate generator parameters."""
    if cells_per_axis < 1:
        raise ValueError(
            f"cells_per_axis must be >= 1, got {cells_per_axis}"
        )


def _build_sites(n: int) -> tuple[np.ndarray, list[int]]:
    """Build A15 sites with one layer of periodic padding for Voronoi."""
    sites: list[np.ndarray] = []
    interior: list[int] = []
    idx = 0
    for ix in range(-1, n + 1):
        for iy in range(-1, n + 1):
            for iz in range(-1, n + 1):
                is_inner = 0 <= ix < n and 0 <= iy < n and 0 <= iz < n
                offset = np.array([ix, iy, iz], dtype=np.float64)
                for site in _A15_SITES:
                    sites.append(site + offset)
                    if is_inner:
                        interior.append(idx)
                    idx += 1
    return np.array(sites, dtype=np.float64), interior


def _collect_edges(vor: Voronoi, indices: list[int]) -> list[Curve]:
    """Extract unique cell edges as line-segment Curves."""
    index_set = set(indices)
    unique_edges: set[tuple[int, int]] = set()

    for ridge_pts, ridge_verts in zip(
        vor.ridge_points, vor.ridge_vertices, strict=False,
    ):
        if -1 in ridge_verts:
            continue
        if ridge_pts[0] not in index_set and ridge_pts[1] not in index_set:
            continue
        n_v = len(ridge_verts)
        for i in range(n_v):
            edge = tuple(sorted((ridge_verts[i], ridge_verts[(i + 1) % n_v])))
            unique_edges.add(edge)

    curves: list[Curve] = []
    for v1, v2 in sorted(unique_edges):
        pts = vor.vertices[[v1, v2]]
        if not np.all(np.isfinite(pts)):
            continue
        curves.append(Curve(points=pts.astype(np.float64), closed=False))
    return curves


def _sort_polygon(vertices: np.ndarray, indices: list[int]) -> list[int]:
    """Sort polygon vertex indices by angle around centroid."""
    pts = vertices[indices]
    centroid = pts.mean(axis=0)
    relative = pts - centroid
    if len(indices) < 3:
        return indices
    normal = np.cross(relative[1] - relative[0], relative[2] - relative[0])
    norm_len = np.linalg.norm(normal)
    if norm_len < 1e-12:
        return indices
    normal /= norm_len
    u_axis = relative[0] - np.dot(relative[0], normal) * normal
    u_len = np.linalg.norm(u_axis)
    if u_len < 1e-12:
        return indices
    u_axis /= u_len
    v_axis = np.cross(normal, u_axis)
    angles = np.arctan2(np.dot(relative, v_axis), np.dot(relative, u_axis))
    order = np.argsort(angles)
    return [indices[o] for o in order]


def _collect_faces(vor: Voronoi, indices: list[int]) -> Mesh:
    """Extract cell faces as a triangulated mesh."""
    index_set = set(indices)
    seen: set[tuple[int, ...]] = set()
    face_polys: list[list[int]] = []

    for ridge_pts, ridge_verts in zip(
        vor.ridge_points, vor.ridge_vertices, strict=False,
    ):
        if -1 in ridge_verts:
            continue
        if ridge_pts[0] not in index_set and ridge_pts[1] not in index_set:
            continue
        key = tuple(sorted(ridge_verts))
        if key in seen:
            continue
        seen.add(key)
        face_polys.append(list(ridge_verts))

    # Remap vertex indices to compact range
    used = sorted({v for poly in face_polys for v in poly})
    remap = {old: new for new, old in enumerate(used)}
    vertices = vor.vertices[used].astype(np.float64)

    # Filter non-finite vertices
    finite = np.all(np.isfinite(vertices), axis=1)
    if not np.all(finite):
        good = {used[i] for i in range(len(used)) if finite[i]}
        face_polys = [f for f in face_polys if all(v in good for v in f)]
        used = sorted({v for poly in face_polys for v in poly})
        remap = {old: new for new, old in enumerate(used)}
        vertices = vor.vertices[used].astype(np.float64)

    # Fan-triangulate each sorted polygon
    triangles: list[list[int]] = []
    for poly in face_polys:
        mapped = [remap[v] for v in poly]
        if len(mapped) < 3:
            continue
        sorted_verts = _sort_polygon(vertices, mapped)
        for i in range(1, len(sorted_verts) - 1):
            triangles.append(
                [sorted_verts[0], sorted_verts[i], sorted_verts[i + 1]]
            )

    faces = np.array(triangles, dtype=np.int64)
    return Mesh(vertices=vertices, faces=faces)


@register
class WeairePhelanGenerator(GeneratorBase):
    """Weaire-Phelan foam structure generator."""

    name = "weaire_phelan"
    category = "geometry"
    description = "Weaire-Phelan foam: optimal equal-volume space partition"
    resolution_params: dict[str, str] = {}

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters."""
        return {
            "cells_per_axis": _DEFAULT_CELLS_PER_AXIS,
            "edge_only": True,
        }

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate Weaire-Phelan foam structure."""
        merged = self.get_default_params()
        if params:
            merged.update(params)

        cells_per_axis = int(merged["cells_per_axis"])
        edge_only = bool(merged["edge_only"])
        _validate_params(cells_per_axis)

        sites, interior = _build_sites(cells_per_axis)
        vor = Voronoi(sites)
        cell_count = len(interior)
        merged["cell_count"] = cell_count

        if edge_only:
            return self._build_edge_result(vor, interior, merged, seed)
        return self._build_face_result(vor, interior, merged, seed)

    def _build_edge_result(
        self,
        vor: Voronoi,
        interior: list[int],
        params: dict[str, Any],
        seed: int,
    ) -> MathObject:
        """Build MathObject with curve edges."""
        curves = _collect_edges(vor, interior)
        if not curves:
            raise ValueError("No finite edges produced")
        all_pts = np.concatenate([c.points for c in curves])
        logger.info(
            "Generated weaire_phelan: cells=%d, edges=%d",
            params["cell_count"], len(curves),
        )
        return MathObject(
            curves=curves,
            generator_name=self.name,
            category=self.category,
            parameters=params,
            seed=seed,
            bounding_box=BoundingBox.from_points(all_pts),
        )

    def _build_face_result(
        self,
        vor: Voronoi,
        interior: list[int],
        params: dict[str, Any],
        seed: int,
    ) -> MathObject:
        """Build MathObject with triangulated face mesh."""
        mesh = _collect_faces(vor, interior)
        logger.info(
            "Generated weaire_phelan: cells=%d, faces=%d",
            params["cell_count"], len(mesh.faces),
        )
        return MathObject(
            mesh=mesh,
            generator_name=self.name,
            category=self.category,
            parameters=params,
            seed=seed,
            bounding_box=BoundingBox.from_points(mesh.vertices),
        )

    def get_default_representation(self) -> RepresentationConfig:
        """Return TUBE as default for wireframe edges."""
        return RepresentationConfig(type=RepresentationType.TUBE)
