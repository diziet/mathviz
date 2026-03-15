"""Koch snowflake 3D generator.

Generates the classic Koch snowflake curve at a given recursion level,
then produces 3D geometry by either extrusion (along the z-axis) or
revolution (around the y-axis). Default representation: SURFACE_SHELL.
"""

import logging
from typing import Any

import numpy as np

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


def _koch_snowflake_2d(level: int) -> np.ndarray:
    """Generate 2D Koch snowflake curve points at given recursion level.

    Returns (N, 2) array of points forming the closed snowflake curve.
    Level 0 produces an equilateral triangle.
    """
    # Start with equilateral triangle (vertices in CCW order)
    angle_offsets = [np.pi / 2, np.pi / 2 - 2 * np.pi / 3,
                     np.pi / 2 - 4 * np.pi / 3]
    triangle = np.array([
        [np.cos(a), np.sin(a)] for a in angle_offsets
    ])

    if level == 0:
        return triangle

    # Build segments from the triangle
    segments = []
    for i in range(3):
        segments.append((triangle[i], triangle[(i + 1) % 3]))

    for _ in range(level):
        segments = _subdivide_segments(segments)

    # Extract ordered points (first point of each segment)
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
        # Peak point: rotate delta/3 by -60 degrees from p1
        cos60, sin60 = np.cos(np.pi / 3), np.sin(np.pi / 3)
        dx, dy = delta[0] / 3, delta[1] / 3
        peak = p1 + np.array([
            dx * cos60 - dy * sin60,
            dx * sin60 + dy * cos60,
        ])
        new_segments.append((p_start, p1))
        new_segments.append((p1, peak))
        new_segments.append((peak, p2))
        new_segments.append((p2, p_end))
    return new_segments


def _build_extrusion_mesh(
    curve_2d: np.ndarray, height: float,
) -> Mesh:
    """Extrude a closed 2D curve along z to create a 3D mesh."""
    num_pts = len(curve_2d)

    # Bottom ring at z=0, top ring at z=height
    bottom = np.column_stack([
        curve_2d[:, 0], curve_2d[:, 1],
        np.zeros(num_pts),
    ])
    top = np.column_stack([
        curve_2d[:, 0], curve_2d[:, 1],
        np.full(num_pts, height),
    ])
    vertices = np.vstack([bottom, top]).astype(np.float64)

    # Side faces connecting bottom and top rings
    faces = _build_side_faces(num_pts, bottom_offset=0, top_offset=num_pts)

    # Cap faces via fan triangulation
    bottom_cap = _fan_triangulate(list(range(num_pts)), reverse=True)
    top_cap = _fan_triangulate(
        list(range(num_pts, 2 * num_pts)), reverse=False,
    )
    all_faces = np.vstack([faces, bottom_cap, top_cap]).astype(np.int64)

    return Mesh(vertices=vertices, faces=all_faces)


def _build_side_faces(
    num_pts: int, bottom_offset: int, top_offset: int,
) -> np.ndarray:
    """Build triangle faces for side walls between two rings."""
    faces = np.empty((num_pts * 2, 3), dtype=np.int64)
    for i in range(num_pts):
        j = (i + 1) % num_pts
        b0 = bottom_offset + i
        b1 = bottom_offset + j
        t0 = top_offset + i
        t1 = top_offset + j
        faces[2 * i] = [b0, b1, t1]
        faces[2 * i + 1] = [b0, t1, t0]
    return faces


def _fan_triangulate(
    indices: list[int], *, reverse: bool,
) -> np.ndarray:
    """Triangulate a polygon via fan from first vertex."""
    num_tris = len(indices) - 2
    faces = np.empty((num_tris, 3), dtype=np.int64)
    for i in range(num_tris):
        if reverse:
            faces[i] = [indices[0], indices[i + 2], indices[i + 1]]
        else:
            faces[i] = [indices[0], indices[i + 1], indices[i + 2]]
    return faces


def _build_revolution_mesh(
    curve_2d: np.ndarray, num_segments: int,
) -> Mesh:
    """Revolve a closed 2D curve around the y-axis to create a 3D mesh."""
    num_pts = len(curve_2d)
    angles = np.linspace(0, 2 * np.pi, num_segments, endpoint=False)

    # For each angle, rotate the 2D curve (x, y) around y-axis
    # x_3d = x_2d * cos(theta), z_3d = x_2d * sin(theta), y_3d = y_2d
    vertices = np.empty((num_segments * num_pts, 3), dtype=np.float64)
    for seg_idx, theta in enumerate(angles):
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        offset = seg_idx * num_pts
        vertices[offset:offset + num_pts, 0] = curve_2d[:, 0] * cos_t
        vertices[offset:offset + num_pts, 1] = curve_2d[:, 1]
        vertices[offset:offset + num_pts, 2] = curve_2d[:, 0] * sin_t

    # Connect adjacent rings
    faces_list = []
    for seg_idx in range(num_segments):
        next_seg = (seg_idx + 1) % num_segments
        for pt_idx in range(num_pts):
            next_pt = (pt_idx + 1) % num_pts
            v00 = seg_idx * num_pts + pt_idx
            v01 = seg_idx * num_pts + next_pt
            v10 = next_seg * num_pts + pt_idx
            v11 = next_seg * num_pts + next_pt
            faces_list.append([v00, v01, v11])
            faces_list.append([v00, v11, v10])

    faces = np.array(faces_list, dtype=np.int64)
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
    if height < _MIN_HEIGHT:
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
