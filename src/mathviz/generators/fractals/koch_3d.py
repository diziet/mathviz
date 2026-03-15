"""Koch snowflake 3D generator.

Generates the classic Koch snowflake curve at a given recursion level,
then produces 3D geometry by either extrusion (along the z-axis) or
revolution (around the y-axis). Default representation: SURFACE_SHELL.
"""

import logging
from typing import Any

import numpy as np
from scipy.spatial import Delaunay

from mathviz.core.generator import GeneratorBase, register
from mathviz.core.math_object import BoundingBox, MathObject, Mesh
from mathviz.core.representation import RepresentationConfig, RepresentationType

logger = logging.getLogger(__name__)

_DEFAULT_LEVEL = 4
_MAX_LEVEL = 6
_DEFAULT_MODE = "extrude"
_DEFAULT_HEIGHT = 0.3
_VALID_MODES = ("extrude", "revolve")
_MIN_HEIGHT = 1e-6
_REVOLVE_SEGMENTS = 64
_COS60 = np.cos(np.pi / 3)
_SIN60 = np.sin(np.pi / 3)


def _koch_snowflake_2d(level: int) -> np.ndarray:
    """Generate 2D Koch snowflake curve points at given recursion level.

    Returns (N, 2) array of points forming the closed snowflake curve.
    Level 0 produces an equilateral triangle.
    """
    angle_offsets = [np.pi / 2, np.pi / 2 - 2 * np.pi / 3,
                     np.pi / 2 - 4 * np.pi / 3]
    triangle = np.array([
        [np.cos(a), np.sin(a)] for a in angle_offsets
    ])

    if level == 0:
        return triangle

    segments = []
    for i in range(3):
        segments.append((triangle[i], triangle[(i + 1) % 3]))

    for _ in range(level):
        segments = _subdivide_segments(segments)

    points = np.array([seg[0] for seg in segments])
    return points


def _subdivide_segments(
    segments: list[tuple[np.ndarray, np.ndarray]],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Subdivide each segment into 4 Koch sub-segments."""
    new_segments: list[tuple[np.ndarray, np.ndarray]] = []
    for p_start, p_end in segments:
        delta = p_end - p_start
        p1 = p_start + delta / 3
        p2 = p_start + delta * 2 / 3
        dx, dy = delta[0] / 3, delta[1] / 3
        peak = p1 + np.array([
            dx * _COS60 - dy * _SIN60,
            dx * _SIN60 + dy * _COS60,
        ])
        new_segments.append((p_start, p1))
        new_segments.append((p1, peak))
        new_segments.append((peak, p2))
        new_segments.append((p2, p_end))
    return new_segments


def _triangulate_cap(curve_2d: np.ndarray) -> np.ndarray:
    """Triangulate a 2D polygon using Delaunay triangulation.

    Handles non-convex Koch snowflake curves correctly, unlike fan
    triangulation which produces self-intersecting faces.
    Returns (M, 3) array of triangle indices into curve_2d.
    """
    tri = Delaunay(curve_2d)
    return tri.simplices.astype(np.int64)


def _build_side_faces(
    num_pts: int, bottom_offset: int, top_offset: int,
) -> np.ndarray:
    """Build triangle faces for side walls between two rings."""
    idx = np.arange(num_pts)
    next_idx = (idx + 1) % num_pts
    b0 = bottom_offset + idx
    b1 = bottom_offset + next_idx
    t0 = top_offset + idx
    t1 = top_offset + next_idx
    faces = np.empty((num_pts * 2, 3), dtype=np.int64)
    faces[0::2] = np.column_stack([b0, b1, t1])
    faces[1::2] = np.column_stack([b0, t1, t0])
    return faces


def _build_extrusion_mesh(
    curve_2d: np.ndarray, height: float,
) -> Mesh:
    """Extrude a closed 2D curve along z to create a 3D mesh."""
    num_pts = len(curve_2d)

    bottom = np.column_stack([
        curve_2d[:, 0], curve_2d[:, 1],
        np.zeros(num_pts),
    ])
    top = np.column_stack([
        curve_2d[:, 0], curve_2d[:, 1],
        np.full(num_pts, height),
    ])
    vertices = np.vstack([bottom, top]).astype(np.float64)

    side_faces = _build_side_faces(num_pts, bottom_offset=0, top_offset=num_pts)

    # Delaunay triangulation for non-convex caps
    cap_tris = _triangulate_cap(curve_2d)
    bottom_cap = cap_tris[:, ::-1]  # Reverse winding for bottom face
    top_cap = cap_tris + num_pts    # Offset indices for top ring

    all_faces = np.vstack([side_faces, bottom_cap, top_cap]).astype(np.int64)
    return Mesh(vertices=vertices, faces=all_faces)


def _build_revolution_mesh(
    curve_2d: np.ndarray, num_segments: int,
) -> Mesh:
    """Revolve a closed 2D curve around the y-axis to create a 3D mesh."""
    num_pts = len(curve_2d)
    angles = np.linspace(0, 2 * np.pi, num_segments, endpoint=False)

    # Build vertices: rotate curve around y-axis at each angle
    vertices = np.empty((num_segments * num_pts, 3), dtype=np.float64)
    cos_angles = np.cos(angles)
    sin_angles = np.sin(angles)
    for seg_idx in range(num_segments):
        offset = seg_idx * num_pts
        vertices[offset:offset + num_pts, 0] = (
            curve_2d[:, 0] * cos_angles[seg_idx]
        )
        vertices[offset:offset + num_pts, 1] = curve_2d[:, 1]
        vertices[offset:offset + num_pts, 2] = (
            curve_2d[:, 0] * sin_angles[seg_idx]
        )

    # Vectorized face generation
    seg_indices = np.arange(num_segments)
    pt_indices = np.arange(num_pts)
    seg_grid, pt_grid = np.meshgrid(
        seg_indices, pt_indices, indexing="ij",
    )
    seg_flat = seg_grid.ravel()
    pt_flat = pt_grid.ravel()
    next_seg = (seg_flat + 1) % num_segments
    next_pt = (pt_flat + 1) % num_pts
    v00 = seg_flat * num_pts + pt_flat
    v01 = seg_flat * num_pts + next_pt
    v10 = next_seg * num_pts + pt_flat
    v11 = next_seg * num_pts + next_pt
    faces = np.empty((len(v00) * 2, 3), dtype=np.int64)
    faces[0::2] = np.column_stack([v00, v01, v11])
    faces[1::2] = np.column_stack([v00, v11, v10])

    return Mesh(vertices=vertices, faces=faces)


def _validate_params(level: int, mode: str, height: float) -> None:
    """Validate Koch 3D parameters."""
    if level < 0:
        raise ValueError(f"level must be >= 0, got {level}")
    if level > _MAX_LEVEL:
        raise ValueError(f"level must be <= {_MAX_LEVEL}, got {level}")
    if mode not in _VALID_MODES:
        raise ValueError(
            f"mode must be one of {_VALID_MODES}, got {mode!r}"
        )
    if mode == "extrude" and height < _MIN_HEIGHT:
        raise ValueError(f"height must be >= {_MIN_HEIGHT}, got {height}")


@register
class Koch3DGenerator(GeneratorBase):
    """Koch snowflake 3D — extruded or revolved fractal curve."""

    name = "koch_3d"
    category = "fractals"
    aliases = ("koch_snowflake_3d",)
    description = "Koch snowflake curve extruded or revolved into 3D"
    resolution_params: dict[str, str] = {}
    _resolution_defaults: dict[str, Any] = {}

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters for the Koch 3D generator."""
        return {
            "level": _DEFAULT_LEVEL,
            "mode": _DEFAULT_MODE,
            "height": _DEFAULT_HEIGHT,
        }

    def get_param_ranges(self) -> dict[str, dict[str, float]]:
        """Return exploration ranges for parameters."""
        return {
            "level": {"min": 0, "max": _MAX_LEVEL, "step": 1},
            "height": {"min": 0.05, "max": 2.0, "step": 0.05},
        }

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        # resolution_kwargs accepted per GeneratorBase contract but unused
        # — Koch geometry is fully determined by level, not resolution.
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate a Koch snowflake 3D mesh.

        Fully deterministic — seed is stored for metadata only.
        """
        merged = self.get_default_params()
        if params:
            merged.update(params)

        level = int(merged["level"])
        mode = str(merged["mode"])
        height = float(merged["height"])

        _validate_params(level, mode, height)

        curve_2d = _koch_snowflake_2d(level)

        if mode == "extrude":
            mesh = _build_extrusion_mesh(curve_2d, height)
        else:
            mesh = _build_revolution_mesh(curve_2d, _REVOLVE_SEGMENTS)

        bbox = BoundingBox.from_points(mesh.vertices)

        logger.info(
            "Generated koch_3d: level=%d, mode=%s, "
            "vertices=%d, faces=%d",
            level, mode, len(mesh.vertices), len(mesh.faces),
        )

        return MathObject(
            mesh=mesh,
            generator_name=self.resolved_name or self.name,
            category=self.category,
            parameters=merged,
            seed=seed,
            bounding_box=bbox,
        )

    def get_default_representation(self) -> RepresentationConfig:
        """Return SURFACE_SHELL as default representation."""
        return RepresentationConfig(type=RepresentationType.SURFACE_SHELL)
