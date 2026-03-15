"""L-system / fractal tree generator.

Produces 3D branching structures from Lindenmayer system grammars. Supports
named presets (tree, bush, fern, hilbert3d, sierpinski) and custom angle,
length, and decay parameters for organic variation.
"""

import logging
from typing import Any

import numpy as np

from mathviz.core.generator import GeneratorBase, register
from mathviz.core.math_object import BoundingBox, Curve, MathObject
from mathviz.core.representation import RepresentationConfig, RepresentationType
from mathviz.generators.procedural._lsystem_engine import (
    PRESETS,
    interpret_turtle,
    rewrite,
)

logger = logging.getLogger(__name__)

_DEFAULT_PRESET = "tree"
_DEFAULT_ITERATIONS = 5
_DEFAULT_ANGLE = 25.0
_DEFAULT_LENGTH_SCALE = 1.0
_DEFAULT_LENGTH_DECAY = 0.7
_DEFAULT_THICKNESS_DECAY = 0.6
_DEFAULT_JITTER = 5.0
_DEFAULT_TUBE_RADIUS = 0.02
_MIN_ITERATIONS = 1
_MAX_ITERATIONS = 10
_MIN_ANGLE = 0.1
_MIN_LENGTH_SCALE = 0.001


def _validate_params(
    preset: str,
    iterations: int,
    angle: float,
    length_scale: float,
    length_decay: float,
    thickness_decay: float,
) -> None:
    """Validate L-system parameters."""
    if preset not in PRESETS:
        valid = ", ".join(sorted(PRESETS))
        raise ValueError(f"Unknown preset '{preset}'; valid: {valid}")
    if iterations < _MIN_ITERATIONS:
        raise ValueError(
            f"iterations must be >= {_MIN_ITERATIONS}, got {iterations}"
        )
    if iterations > _MAX_ITERATIONS:
        raise ValueError(
            f"iterations must be <= {_MAX_ITERATIONS}, got {iterations}"
        )
    if angle < _MIN_ANGLE:
        raise ValueError(f"angle must be >= {_MIN_ANGLE}, got {angle}")
    if length_scale < _MIN_LENGTH_SCALE:
        raise ValueError(
            f"length_scale must be >= {_MIN_LENGTH_SCALE}, got {length_scale}"
        )
    if not 0 < length_decay <= 1.0:
        raise ValueError(
            f"length_decay must be in (0, 1], got {length_decay}"
        )
    if not 0 < thickness_decay <= 1.0:
        raise ValueError(
            f"thickness_decay must be in (0, 1], got {thickness_decay}"
        )


def _segments_to_curves(
    segments: list[Any],
) -> list[Curve]:
    """Convert turtle segments into connected branch curves.

    Groups consecutive segments into branches, splitting at discontinuities
    (where one segment's end doesn't match the next's start).
    """
    if not segments:
        return []

    curves: list[Curve] = []
    current_points: list[np.ndarray] = [segments[0].start, segments[0].end]

    for seg in segments[1:]:
        if np.allclose(current_points[-1], seg.start, atol=1e-10):
            current_points.append(seg.end)
        else:
            if len(current_points) >= 2:
                points = np.array(current_points, dtype=np.float64)
                curves.append(Curve(points=points, closed=False))
            current_points = [seg.start, seg.end]

    if len(current_points) >= 2:
        points = np.array(current_points, dtype=np.float64)
        curves.append(Curve(points=points, closed=False))

    return curves


def _all_points_from_segments(segments: list[Any]) -> np.ndarray:
    """Collect all unique points from segments for bounding box."""
    if not segments:
        return np.zeros((1, 3), dtype=np.float64)
    points = []
    for seg in segments:
        points.append(seg.start)
        points.append(seg.end)
    return np.array(points, dtype=np.float64)


@register
class LSystemGenerator(GeneratorBase):
    """L-system fractal tree and branching structure generator."""

    name = "lsystem"
    category = "procedural"
    aliases = ()
    description = "L-system fractal trees, bushes, ferns, and space-filling curves"
    resolution_params: dict[str, str] = {}
    _resolution_defaults: dict[str, Any] = {}

    def get_default_params(self) -> dict[str, Any]:
        """Return default L-system parameters."""
        return {
            "preset": _DEFAULT_PRESET,
            "iterations": _DEFAULT_ITERATIONS,
            "angle": _DEFAULT_ANGLE,
            "length_scale": _DEFAULT_LENGTH_SCALE,
            "length_decay": _DEFAULT_LENGTH_DECAY,
            "thickness_decay": _DEFAULT_THICKNESS_DECAY,
            "jitter": _DEFAULT_JITTER,
        }

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate an L-system branching structure."""
        merged = self.get_default_params()
        if params:
            merged.update(params)

        preset_name = str(merged["preset"])
        iterations = int(merged["iterations"])
        angle = float(merged["angle"])
        length_scale = float(merged["length_scale"])
        length_decay = float(merged["length_decay"])
        thickness_decay = float(merged["thickness_decay"])
        jitter = float(merged["jitter"])

        _validate_params(
            preset_name, iterations, angle,
            length_scale, length_decay, thickness_decay,
        )

        preset = PRESETS[preset_name]

        instruction_string = rewrite(preset.axiom, preset.rules, iterations)
        rng = np.random.default_rng(seed)

        segments = interpret_turtle(
            instruction_string=instruction_string,
            angle_deg=angle,
            length=length_scale,
            length_decay=length_decay,
            thickness_decay=thickness_decay,
            jitter_deg=jitter,
            rng=rng,
        )

        if not segments:
            logger.warning(
                "L-system produced no segments for preset=%s, iterations=%d",
                preset_name, iterations,
            )
            point = np.zeros((2, 3), dtype=np.float64)
            point[1, 2] = length_scale
            curves = [Curve(points=point, closed=False)]
            bbox = BoundingBox.from_points(point)
        else:
            curves = _segments_to_curves(segments)
            all_pts = _all_points_from_segments(segments)
            bbox = BoundingBox.from_points(all_pts)

        logger.info(
            "Generated L-system: preset=%s, iterations=%d, angle=%.1f, "
            "segments=%d, curves=%d",
            preset_name, iterations, angle, len(segments), len(curves),
        )

        return MathObject(
            curves=curves,
            generator_name=self.resolved_name or self.name,
            category=self.category,
            parameters=merged,
            seed=seed,
            bounding_box=bbox,
        )

    def get_default_representation(self) -> RepresentationConfig:
        """Return TUBE as default representation for L-system curves."""
        return RepresentationConfig(
            type=RepresentationType.TUBE, tube_radius=_DEFAULT_TUBE_RADIUS,
        )
