"""Digit encoding generator.

Maps digits of mathematical constants (π, e, φ) to 3D point positions.
Each digit determines the z-height, while x/y follow a linear or spiral
layout. Default representation: WEIGHTED_CLOUD.
"""

import logging
import math
from typing import Any

import numpy as np

from mathviz.core.generator import GeneratorBase, register
from mathviz.core.math_object import BoundingBox, MathObject, PointCloud
from mathviz.core.representation import RepresentationConfig, RepresentationType

logger = logging.getLogger(__name__)

# Pre-computed digit strings for mathematical constants.
# mpmath would give arbitrary precision, but these suffice for typical use.
_PI_DIGITS = (
    "3141592653589793238462643383279502884197169399375105820974944592307816"
    "4062862089986280348253421170679821480865132823066470938446095505822317"
    "2535940812848111745028410270193852110555964462294895493038196442881097"
    "5665933446128475648233786783165271201909145648566923460348610454326648"
    "2133936072602491412737245870066063155881748815209209628292540917153643"
)

_E_DIGITS = (
    "2718281828459045235360287471352662497757247093699959574966967627724076"
    "6303535475945713821785251664274274663919320030599218174135966290435729"
    "0033429526059563073813232862794349076323382988075319525101901157383418"
    "7930702154089149934884167509244761460668082264800168477411853742345442"
    "4371075390777449920695517027618386062613313845830007520449338265602976"
)

_PHI_DIGITS = (
    "1618033988749894848204586834365638117720309179805762862135448622705260"
    "4628189024497072072041893911374847540880753868917521266338622235369317"
    "9318006076672635443338908659593958290563832266131992829026788067520876"
    "6892501711696207032221043216269548626296313614438149758701220340805887"
    "9544547492461856953648644492410443207713449470495658467885098743394422"
)

_CONSTANT_DIGITS: dict[str, str] = {
    "pi": _PI_DIGITS,
    "e": _E_DIGITS,
    "phi": _PHI_DIGITS,
}

_VALID_CONSTANTS = frozenset(_CONSTANT_DIGITS.keys())


def _validate_params(constant: str, num_digits: int) -> None:
    """Validate digit encoding parameters."""
    if constant not in _VALID_CONSTANTS:
        raise ValueError(
            f"constant must be one of {sorted(_VALID_CONSTANTS)}, "
            f"got {constant!r}"
        )
    if num_digits < 1:
        raise ValueError(f"num_digits must be >= 1, got {num_digits}")


def _get_digits(constant: str, num_digits: int) -> np.ndarray:
    """Extract integer digits from a constant string."""
    digit_str = _CONSTANT_DIGITS[constant]
    available = len(digit_str)
    if num_digits > available:
        logger.warning(
            "Requested %d digits of %s but only %d available, clamping",
            num_digits, constant, available,
        )
        num_digits = available
    return np.array([int(c) for c in digit_str[:num_digits]], dtype=np.int64)


def _build_digit_cloud(
    digits: np.ndarray, height_scale: float, spacing: float
) -> tuple[np.ndarray, np.ndarray]:
    """Build 3D points and intensities from digit values."""
    num_digits = len(digits)
    # Layout digits in a grid: wrap into rows
    cols = int(np.ceil(np.sqrt(num_digits)))
    rows = int(np.ceil(num_digits / cols))

    points = np.zeros((num_digits, 3), dtype=np.float64)
    for i in range(num_digits):
        row = i // cols
        col = i % cols
        points[i, 0] = col * spacing
        points[i, 1] = row * spacing
        points[i, 2] = digits[i] * height_scale

    # Intensity proportional to digit value (0-9 normalized)
    intensities = (digits / 9.0).astype(np.float64)
    return points, intensities


@register
class DigitEncodingGenerator(GeneratorBase):
    """Digit encoding: digits of π, e, or φ mapped to 3D point positions."""

    name = "digit_encoding"
    category = "number_theory"
    aliases = ("digits",)
    description = "Digits of mathematical constants as 3D point heights"
    resolution_params = {
        "num_digits": "Number of digits to encode",
    }

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters for digit encoding."""
        return {
            "constant": "pi",
            "height_scale": 0.1,
            "spacing": 0.1,
        }

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate a digit encoding point cloud."""
        merged = self.get_default_params()
        if params:
            merged.update(params)

        num_digits = int(resolution_kwargs.get("num_digits", 100))
        constant = str(merged["constant"])
        height_scale = float(merged["height_scale"])
        spacing = float(merged["spacing"])
        _validate_params(constant, num_digits)

        digits = _get_digits(constant, num_digits)
        points, intensities = _build_digit_cloud(digits, height_scale, spacing)
        cloud = PointCloud(points=points, intensities=intensities)
        bbox = BoundingBox.from_points(points)

        logger.info(
            "Generated digit encoding for %s with %d digits",
            constant, len(digits),
        )

        return MathObject(
            point_cloud=cloud,
            generator_name=self.resolved_name or self.name,
            category=self.category,
            parameters={**merged, "num_digits": num_digits},
            seed=seed,
            bounding_box=bbox,
        )

    def get_default_representation(self) -> RepresentationConfig:
        """Return WEIGHTED_CLOUD as the default representation."""
        return RepresentationConfig(type=RepresentationType.WEIGHTED_CLOUD)
