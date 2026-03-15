"""Linked tori generator — two or more interlocking torus shapes.

Each torus in the chain passes through its neighbor, like links in a chain.
Odd-indexed tori are rotated 90° so they interlock with their neighbors.
"""

import logging
from typing import Any

import numpy as np

from mathviz.core.generator import GeneratorBase, register
from mathviz.core.math_object import MathObject, Mesh
from mathviz.core.representation import RepresentationConfig, RepresentationType
from mathviz.generators.parametric._mesh_utils import (
    build_wrapped_grid_faces,
    compute_padded_bounding_box,
)

logger = logging.getLogger(__name__)

_DEFAULT_NUM_TORI = 2
_DEFAULT_MAJOR_RADIUS = 1.0
_DEFAULT_MINOR_RADIUS = 0.3
_DEFAULT_LINK_SPACING = 1.5
_DEFAULT_GRID_RESOLUTION = 64
_MIN_GRID_RESOLUTION = 3
_MIN_NUM_TORI = 2
_MAX_NUM_TORI = 20


def _validate_params(
    num_tori: int,
    major_radius: float,
    minor_radius: float,
    link_spacing: float,
    grid_resolution: int,
) -> None:
    """Validate linked tori parameters."""
    if num_tori < _MIN_NUM_TORI:
        raise ValueError(
            f"num_tori must be >= {_MIN_NUM_TORI}, got {num_tori}"
        )
    if num_tori > _MAX_NUM_TORI:
        raise ValueError(
            f"num_tori must be <= {_MAX_NUM_TORI}, got {num_tori}"
        )
    if major_radius <= 0:
        raise ValueError(f"major_radius must be positive, got {major_radius}")
    if minor_radius <= 0:
        raise ValueError(f"minor_radius must be positive, got {minor_radius}")
    if link_spacing <= 0:
        raise ValueError(f"link_spacing must be positive, got {link_spacing}")
    if grid_resolution < _MIN_GRID_RESOLUTION:
        raise ValueError(
            f"grid_resolution must be >= {_MIN_GRID_RESOLUTION}, "
            f"got {grid_resolution}"
        )


def _generate_single_torus(
    major_radius: float,
    minor_radius: float,
    grid_resolution: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate vertices and faces for a single torus at the origin in XY plane."""
    n = grid_resolution
    u_vals = np.linspace(0, 2 * np.pi, n, endpoint=False)
    v_vals = np.linspace(0, 2 * np.pi, n, endpoint=False)
    uu, vv = np.meshgrid(u_vals, v_vals, indexing="ij")

    x = (major_radius + minor_radius * np.cos(vv)) * np.cos(uu)
    y = (major_radius + minor_radius * np.cos(vv)) * np.sin(uu)
    z = minor_radius * np.sin(vv)

    vertices = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    vertices = vertices.astype(np.float64)
    faces = build_wrapped_grid_faces(n, n)
    return vertices, faces


def _rotate_vertices_x_90(vertices: np.ndarray) -> np.ndarray:
    """Rotate vertices 90° around the X axis (XY plane -> XZ plane)."""
    rotated = vertices.copy()
    rotated[:, 1] = -vertices[:, 2]
    rotated[:, 2] = vertices[:, 1]
    return rotated


def _build_linked_tori_mesh(
    num_tori: int,
    major_radius: float,
    minor_radius: float,
    link_spacing: float,
    grid_resolution: int,
) -> Mesh:
    """Build a combined mesh of interlocking tori."""
    base_verts, base_faces = _generate_single_torus(
        major_radius, minor_radius, grid_resolution,
    )

    all_vertices = []
    all_faces = []
    vertex_offset = 0

    for i in range(num_tori):
        verts = base_verts.copy()

        # Rotate odd-indexed tori 90° around X to interlock
        if i % 2 == 1:
            verts = _rotate_vertices_x_90(verts)

        # Translate along X axis
        verts[:, 0] += i * link_spacing

        all_vertices.append(verts)
        all_faces.append(base_faces + vertex_offset)
        vertex_offset += len(verts)

    combined_vertices = np.concatenate(all_vertices, axis=0)
    combined_faces = np.concatenate(all_faces, axis=0)
    return Mesh(vertices=combined_vertices, faces=combined_faces)


@register
class LinkedToriGenerator(GeneratorBase):
    """Generator for linked tori — interlocking torus chain."""

    name = "linked_tori"
    category = "parametric"
    aliases = ()
    description = "Chain of interlocking tori, like links in a chain"
    resolution_params = {
        "grid_resolution": "Number of grid divisions per axis per torus",
    }
    _resolution_defaults = {"grid_resolution": _DEFAULT_GRID_RESOLUTION}

    def get_param_ranges(self) -> dict[str, dict[str, float]]:
        """Return exploration ranges for linked tori parameters."""
        return {
            "num_tori": {"min": 2, "max": 10, "step": 1},
            "major_radius": {"min": 0.3, "max": 3.0, "step": 0.1},
            "minor_radius": {"min": 0.05, "max": 1.0, "step": 0.05},
            "link_spacing": {"min": 0.5, "max": 4.0, "step": 0.1},
        }

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters for the linked tori generator."""
        return {
            "num_tori": _DEFAULT_NUM_TORI,
            "major_radius": _DEFAULT_MAJOR_RADIUS,
            "minor_radius": _DEFAULT_MINOR_RADIUS,
            "link_spacing": _DEFAULT_LINK_SPACING,
        }

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate a linked tori mesh."""
        merged = self.get_default_params()
        if params:
            merged.update(params)

        num_tori = int(merged["num_tori"])
        major_radius = float(merged["major_radius"])
        minor_radius = float(merged["minor_radius"])
        link_spacing = float(merged["link_spacing"])
        grid_resolution = int(
            resolution_kwargs.get("grid_resolution", _DEFAULT_GRID_RESOLUTION)
        )

        _validate_params(
            num_tori, major_radius, minor_radius,
            link_spacing, grid_resolution,
        )

        mesh = _build_linked_tori_mesh(
            num_tori, major_radius, minor_radius,
            link_spacing, grid_resolution,
        )
        bbox = compute_padded_bounding_box(mesh.vertices)

        logger.info(
            "Generated linked tori: n=%d, R=%.3f, r=%.3f, spacing=%.3f, "
            "grid=%d, vertices=%d, faces=%d",
            num_tori, major_radius, minor_radius, link_spacing,
            grid_resolution, len(mesh.vertices), len(mesh.faces),
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
        """Return the recommended representation for linked tori."""
        return RepresentationConfig(type=RepresentationType.SURFACE_SHELL)
