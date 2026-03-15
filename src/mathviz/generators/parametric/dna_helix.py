"""DNA double helix generator.

Generates a DNA double helix structure with twin parametric helices offset
by 180 degrees, connected by base pair rungs at regular intervals.
"""

import logging
from typing import Any

import numpy as np

from mathviz.core.generator import GeneratorBase, register
from mathviz.core.math_object import BoundingBox, Curve, MathObject
from mathviz.core.representation import RepresentationConfig, RepresentationType

logger = logging.getLogger(__name__)

_DEFAULT_TURNS = 3
_DEFAULT_RADIUS = 1.0
_DEFAULT_RISE_PER_TURN = 3.4
_DEFAULT_BASE_PAIRS_PER_TURN = 10
_DEFAULT_CURVE_POINTS = 512

_MIN_TURNS = 0.5
_MIN_RADIUS = 0.01
_MIN_RISE_PER_TURN = 0.01
_MIN_BASE_PAIRS_PER_TURN = 1
_MIN_CURVE_POINTS = 4


def _validate_params(
    turns: float,
    radius: float,
    rise_per_turn: float,
    base_pairs_per_turn: int,
    curve_points: int,
) -> None:
    """Validate DNA helix parameters."""
    if turns < _MIN_TURNS:
        raise ValueError(f"turns must be >= {_MIN_TURNS}, got {turns}")
    if radius < _MIN_RADIUS:
        raise ValueError(f"radius must be >= {_MIN_RADIUS}, got {radius}")
    if rise_per_turn < _MIN_RISE_PER_TURN:
        raise ValueError(
            f"rise_per_turn must be >= {_MIN_RISE_PER_TURN}, got {rise_per_turn}"
        )
    if base_pairs_per_turn < _MIN_BASE_PAIRS_PER_TURN:
        raise ValueError(
            f"base_pairs_per_turn must be >= {_MIN_BASE_PAIRS_PER_TURN}, "
            f"got {base_pairs_per_turn}"
        )
    if curve_points < _MIN_CURVE_POINTS:
        raise ValueError(
            f"curve_points must be >= {_MIN_CURVE_POINTS}, got {curve_points}"
        )


def _compute_helix(
    turns: float,
    radius: float,
    rise_per_turn: float,
    curve_points: int,
    phase_offset: float,
) -> np.ndarray:
    """Compute a single helix backbone curve."""
    t = np.linspace(0.0, turns * 2.0 * np.pi, curve_points)
    x = radius * np.cos(t + phase_offset)
    y = radius * np.sin(t + phase_offset)
    z = (t / (2.0 * np.pi)) * rise_per_turn
    return np.column_stack([x, y, z])


def _compute_rungs(
    turns: float,
    radius: float,
    rise_per_turn: float,
    base_pairs_per_turn: int,
    rung_points: int,
) -> list[np.ndarray]:
    """Compute base pair rung curves connecting the two helices."""
    total_rungs = int(turns * base_pairs_per_turn)
    rungs = []
    for i in range(total_rungs):
        angle = (i / base_pairs_per_turn) * 2.0 * np.pi
        z_val = (angle / (2.0 * np.pi)) * rise_per_turn

        # Points on helix A (phase=0) and helix B (phase=pi)
        t_param = np.linspace(0.0, 1.0, rung_points)
        cos_a = radius * np.cos(angle)
        sin_a = radius * np.sin(angle)
        cos_b = radius * np.cos(angle + np.pi)
        sin_b = radius * np.sin(angle + np.pi)

        x = cos_a + t_param * (cos_b - cos_a)
        y = sin_a + t_param * (sin_b - sin_a)
        z = np.full(rung_points, z_val)
        rungs.append(np.column_stack([x, y, z]))
    return rungs


@register
class DNAHelixGenerator(GeneratorBase):
    """DNA double helix generator with backbone helices and base pair rungs."""

    name = "dna_helix"
    category = "parametric"
    aliases = ("dna", "double_helix")
    description = "DNA double helix with twin helices and base pair rungs"
    resolution_params = {
        "curve_points": "Number of sample points per helix backbone",
    }
    _resolution_defaults = {"curve_points": _DEFAULT_CURVE_POINTS}

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters."""
        return {
            "turns": _DEFAULT_TURNS,
            "radius": _DEFAULT_RADIUS,
            "rise_per_turn": _DEFAULT_RISE_PER_TURN,
            "base_pairs_per_turn": _DEFAULT_BASE_PAIRS_PER_TURN,
        }

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate DNA double helix as curves."""
        merged = self.get_default_params()
        if params:
            merged.update(params)

        if "curve_points" in merged:
            logger.warning(
                "curve_points should be passed as a resolution kwarg, "
                "not inside params; ignoring params value"
            )
            merged.pop("curve_points")

        turns = float(merged["turns"])
        radius = float(merged["radius"])
        rise_per_turn = float(merged["rise_per_turn"])
        base_pairs_per_turn = int(merged["base_pairs_per_turn"])
        curve_points = int(
            resolution_kwargs.get(
                "curve_points", self._resolution_defaults["curve_points"],
            )
        )

        _validate_params(
            turns, radius, rise_per_turn, base_pairs_per_turn, curve_points,
        )
        merged["curve_points"] = curve_points

        # Backbone helices: offset by pi (180 degrees)
        helix_a = _compute_helix(turns, radius, rise_per_turn, curve_points, 0.0)
        helix_b = _compute_helix(
            turns, radius, rise_per_turn, curve_points, np.pi,
        )

        curves: list[Curve] = [
            Curve(points=helix_a, closed=False),
            Curve(points=helix_b, closed=False),
        ]

        # Base pair rungs
        rung_point_count = 8
        rung_arrays = _compute_rungs(
            turns, radius, rise_per_turn, base_pairs_per_turn, rung_point_count,
        )
        for rung in rung_arrays:
            curves.append(Curve(points=rung, closed=False))

        all_points = np.concatenate([c.points for c in curves], axis=0)
        bbox = BoundingBox.from_points(all_points)

        total_rungs = int(turns * base_pairs_per_turn)
        logger.info(
            "Generated dna_helix: turns=%.1f, radius=%.2f, rungs=%d, "
            "curve_points=%d",
            turns, radius, total_rungs, curve_points,
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
        """Return the recommended representation for DNA helix."""
        return RepresentationConfig(type=RepresentationType.TUBE)
