"""Parabolic envelope generator.

A family of lines whose envelope forms a parabolic surface. The generator
creates a triangle mesh from the ruled surface formed by the line family,
producing a surface mesh rather than just a curve.
"""

import logging
from typing import Any

import numpy as np

from mathviz.core.generator import GeneratorBase, register
from mathviz.core.math_object import BoundingBox, Mesh, MathObject
from mathviz.core.representation import RepresentationConfig, RepresentationType

logger = logging.getLogger(__name__)

_DEFAULT_CURVE_POINTS = 64
_DEFAULT_LINE_COUNT = 32
_DEFAULT_SCALE = 1.0
_DEFAULT_HEIGHT = 0.5
_MIN_CURVE_POINTS = 4
_MIN_LINE_COUNT = 4


def _build_envelope_mesh(
    line_count: int,
    segments_per_line: int,
    scale: float,
    height: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a triangle mesh from the parabolic envelope ruled surface."""
    # Parameter along the family of lines
    s_values = np.linspace(0.0, 1.0, line_count)

    # Each line goes from (s, 0, h*s) to (0, 1-s, h*(1-s))
    # We sample each line at segments_per_line points
    t_values = np.linspace(0.0, 1.0, segments_per_line)

    vertices = []
    for s in s_values:
        # Line endpoints
        p0 = np.array([s, 0.0, height * s * (1.0 - s)])
        p1 = np.array([0.0, 1.0 - s, height * s * (1.0 - s)])
        for t in t_values:
            vertex = p0 * (1.0 - t) + p1 * t
            vertices.append(vertex)

    vertices_arr = np.array(vertices, dtype=np.float64) * scale

    # Build faces connecting adjacent lines
    faces = []
    for i in range(line_count - 1):
        for j in range(segments_per_line - 1):
            idx00 = i * segments_per_line + j
            idx01 = i * segments_per_line + j + 1
            idx10 = (i + 1) * segments_per_line + j
            idx11 = (i + 1) * segments_per_line + j + 1

            faces.append([idx00, idx10, idx01])
            faces.append([idx01, idx10, idx11])

    faces_arr = np.array(faces, dtype=np.intp)

    return vertices_arr, faces_arr


def _compute_bounding_box(vertices: np.ndarray) -> BoundingBox:
    """Compute axis-aligned bounding box from vertices."""
    min_corner = tuple(float(v) for v in vertices.min(axis=0))
    max_corner = tuple(float(v) for v in vertices.max(axis=0))
    return BoundingBox(min_corner=min_corner, max_corner=max_corner)


def _validate_params(
    line_count: int, scale: float, curve_points: int
) -> None:
    """Validate parabolic envelope parameters."""
    if line_count < _MIN_LINE_COUNT:
        raise ValueError(
            f"line_count must be >= {_MIN_LINE_COUNT}, got {line_count}"
        )
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")
    if curve_points < _MIN_CURVE_POINTS:
        raise ValueError(
            f"curve_points must be >= {_MIN_CURVE_POINTS}, got {curve_points}"
        )


@register
class ParabolicEnvelopeGenerator(GeneratorBase):
    """Parabolic envelope surface generator."""

    name = "parabolic_envelope"
    category = "curves"
    aliases = ()
    description = "Ruled surface from a family of lines forming a parabolic envelope"
    resolution_params = {
        "curve_points": "Number of sample points per line segment",
    }

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters."""
        return {
            "line_count": _DEFAULT_LINE_COUNT,
            "scale": _DEFAULT_SCALE,
            "height": _DEFAULT_HEIGHT,
        }

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate a parabolic envelope surface mesh."""
        merged = self.get_default_params()
        if params:
            merged.update(params)

        if "curve_points" in merged:
            logger.warning(
                "curve_points should be passed as a resolution kwarg, "
                "not inside params; ignoring params value"
            )
            merged.pop("curve_points")

        line_count = int(merged["line_count"])
        scale = float(merged["scale"])
        height = float(merged["height"])
        curve_points = int(
            resolution_kwargs.get("curve_points", _DEFAULT_CURVE_POINTS)
        )

        _validate_params(line_count, scale, curve_points)
        merged["curve_points"] = curve_points

        vertices, faces = _build_envelope_mesh(
            line_count, curve_points, scale, height,
        )

        mesh = Mesh(vertices=vertices, faces=faces)
        bbox = _compute_bounding_box(vertices)

        logger.info(
            "Generated parabolic envelope: lines=%d, segments=%d, "
            "verts=%d, faces=%d",
            line_count, curve_points, len(vertices), len(faces),
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
        """Return the recommended representation for parabolic envelopes."""
        return RepresentationConfig(type=RepresentationType.SURFACE_SHELL)
