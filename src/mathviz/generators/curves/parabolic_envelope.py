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
    s_values = np.linspace(0.0, 1.0, line_count)
    t_values = np.linspace(0.0, 1.0, segments_per_line)
    S, T = np.meshgrid(s_values, t_values, indexing="ij")

    # Line endpoints: p0=(s, 0, h*s*(1-s)), p1=(0, 1-s, h*s*(1-s))
    hz = height * S * (1.0 - S)
    x = S * (1.0 - T)
    y = (1.0 - S) * T
    z = hz

    vertices = (
        np.stack([x, y, z], axis=-1).reshape(-1, 3).astype(np.float64)
        * scale
    )

    # Build face indices with vectorized arange + broadcasting
    i_idx = np.arange(line_count - 1)
    j_idx = np.arange(segments_per_line - 1)
    I, J = np.meshgrid(i_idx, j_idx, indexing="ij")
    I_flat = I.ravel()
    J_flat = J.ravel()

    idx00 = I_flat * segments_per_line + J_flat
    idx01 = idx00 + 1
    idx10 = idx00 + segments_per_line
    idx11 = idx10 + 1

    tri_a = np.column_stack([idx00, idx10, idx01])
    tri_b = np.column_stack([idx01, idx10, idx11])
    faces = np.vstack([tri_a, tri_b]).astype(np.intp)

    return vertices, faces


def _validate_params(
    line_count: int, height: float, scale: float, curve_points: int
) -> None:
    """Validate parabolic envelope parameters."""
    if line_count < _MIN_LINE_COUNT:
        raise ValueError(
            f"line_count must be >= {_MIN_LINE_COUNT}, got {line_count}"
        )
    if not np.isfinite(height):
        raise ValueError(f"height must be finite, got {height}")
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
    category = "surfaces"
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

        _validate_params(line_count, height, scale, curve_points)
        merged["curve_points"] = curve_points

        vertices, faces = _build_envelope_mesh(
            line_count, curve_points, scale, height,
        )

        mesh = Mesh(vertices=vertices, faces=faces)
        bbox = BoundingBox.from_points(vertices)

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
