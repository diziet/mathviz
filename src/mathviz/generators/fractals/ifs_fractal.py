"""Iterated Function System (IFS) fractal generator.

An IFS is defined by a set of affine transformations with associated
probabilities.  The chaos-game algorithm iterates by randomly choosing
a transform and applying it to the current point, producing a fractal
point cloud.

Presets: barnsley_fern, maple_leaf, spiral, custom.
Modes: 3d (native 3×3 affine), 2d_extruded (classic 2D IFS with z-thickness).
"""

import logging
from typing import Any

import numpy as np
from numpy.random import default_rng

from mathviz.core.generator import GeneratorBase, register
from mathviz.core.math_object import BoundingBox, MathObject, PointCloud
from mathviz.core.representation import RepresentationConfig, RepresentationType

logger = logging.getLogger(__name__)

DEFAULT_NUM_POINTS = 500_000
MIN_NUM_POINTS = 100
_SKIP_INITIAL = 50  # transient points to discard

VALID_PRESETS = ("barnsley_fern", "maple_leaf", "spiral", "custom")
VALID_DIMENSIONS = ("2d_extruded", "3d")


# ---------------------------------------------------------------------------
# Preset definitions (2D)
# ---------------------------------------------------------------------------


def _barnsley_fern_2d() -> tuple[list[np.ndarray], list[np.ndarray], list[float]]:
    """Return classic Barnsley fern 2D IFS (matrices, offsets, probs)."""
    matrices = [
        np.array([[0.0, 0.0], [0.0, 0.16]]),
        np.array([[0.85, 0.04], [-0.04, 0.85]]),
        np.array([[0.20, -0.26], [0.23, 0.22]]),
        np.array([[-0.15, 0.28], [0.26, 0.24]]),
    ]
    offsets = [
        np.array([0.0, 0.0]),
        np.array([0.0, 1.6]),
        np.array([0.0, 1.6]),
        np.array([0.0, 0.44]),
    ]
    probs = [0.01, 0.85, 0.07, 0.07]
    return matrices, offsets, probs


def _maple_leaf_2d() -> tuple[list[np.ndarray], list[np.ndarray], list[float]]:
    """Return maple-leaf IFS (matrices, offsets, probs)."""
    matrices = [
        np.array([[0.14, 0.01], [0.0, 0.51]]),
        np.array([[0.43, 0.52], [-0.45, 0.50]]),
        np.array([[0.45, -0.49], [0.47, 0.47]]),
        np.array([[0.49, 0.0], [0.0, 0.51]]),
    ]
    offsets = [
        np.array([-0.08, -1.31]),
        np.array([1.49, -0.75]),
        np.array([-1.62, -0.74]),
        np.array([0.02, 1.62]),
    ]
    probs = [0.25, 0.25, 0.25, 0.25]
    return matrices, offsets, probs


def _spiral_2d() -> tuple[list[np.ndarray], list[np.ndarray], list[float]]:
    """Return spiral IFS (matrices, offsets, probs)."""
    cos30 = np.cos(np.radians(30))
    sin30 = np.sin(np.radians(30))
    matrices = [
        np.array([[0.7 * cos30, -0.7 * sin30],
                  [0.7 * sin30, 0.7 * cos30]]),
        np.array([[0.3 * cos30, -0.3 * sin30],
                  [0.3 * sin30, 0.3 * cos30]]),
    ]
    offsets = [
        np.array([0.0, 0.0]),
        np.array([1.0, 0.0]),
    ]
    probs = [0.8, 0.2]
    return matrices, offsets, probs


def _get_2d_preset(
    preset: str,
) -> tuple[list[np.ndarray], list[np.ndarray], list[float]]:
    """Look up a 2D IFS preset by name."""
    presets = {
        "barnsley_fern": _barnsley_fern_2d,
        "maple_leaf": _maple_leaf_2d,
        "spiral": _spiral_2d,
    }
    return presets[preset]()


# ---------------------------------------------------------------------------
# 2D → 3D embedding helpers
# ---------------------------------------------------------------------------


def _embed_2d_to_3d(mat_2d: np.ndarray, z_scale: float) -> np.ndarray:
    """Embed a 2×2 matrix into 3×3 with z_scale on diagonal."""
    mat_3d = np.zeros((3, 3), dtype=np.float64)
    mat_3d[:2, :2] = mat_2d
    mat_3d[2, 2] = z_scale
    return mat_3d


def _embed_offsets_2d_to_3d(offsets_2d: list[np.ndarray]) -> list[np.ndarray]:
    """Embed 2D offset vectors into 3D with z=0."""
    return [np.array([o[0], o[1], 0.0]) for o in offsets_2d]


# ---------------------------------------------------------------------------
# Preset definitions (3D)
# ---------------------------------------------------------------------------


def _barnsley_fern_3d() -> tuple[list[np.ndarray], list[np.ndarray], list[float]]:
    """Return Barnsley fern extended to 3D affine transforms."""
    matrices = [
        np.array([[0.0, 0.0, 0.0],
                  [0.0, 0.16, 0.0],
                  [0.0, 0.0, 0.0]]),
        np.array([[0.85, 0.04, 0.0],
                  [-0.04, 0.85, 0.0],
                  [0.0, 0.0, 0.85]]),
        np.array([[0.20, -0.26, 0.0],
                  [0.23, 0.22, 0.0],
                  [0.0, 0.0, 0.20]]),
        np.array([[-0.15, 0.28, 0.0],
                  [0.26, 0.24, 0.0],
                  [0.0, 0.0, 0.24]]),
    ]
    offsets = [
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 1.6, 0.0]),
        np.array([0.0, 1.6, 0.0]),
        np.array([0.0, 0.44, 0.0]),
    ]
    probs = [0.01, 0.85, 0.07, 0.07]
    return matrices, offsets, probs


def _maple_leaf_3d() -> tuple[list[np.ndarray], list[np.ndarray], list[float]]:
    """Return maple leaf extended to 3D."""
    m2, o2, probs = _maple_leaf_2d()
    matrices = [_embed_2d_to_3d(m, 0.3) for m in m2]
    offsets = _embed_offsets_2d_to_3d(o2)
    return matrices, offsets, probs


def _spiral_3d() -> tuple[list[np.ndarray], list[np.ndarray], list[float]]:
    """Return spiral extended to 3D with z-axis rotation on second transform."""
    m2, o2, probs = _spiral_2d()
    matrices = [_embed_2d_to_3d(m, 0.7) for m in m2]
    offsets = _embed_offsets_2d_to_3d(o2)
    # Add z-axis rotation to second transform for 3D interest
    cos15 = np.cos(np.radians(15))
    sin15 = np.sin(np.radians(15))
    matrices[1] = np.array([
        [0.3 * cos15, 0.0, -0.3 * sin15],
        [0.0, 0.3, 0.0],
        [0.3 * sin15, 0.0, 0.3 * cos15],
    ])
    offsets[1] = np.array([1.0, 0.0, 0.5])
    return matrices, offsets, probs


def _get_3d_preset(
    preset: str,
) -> tuple[list[np.ndarray], list[np.ndarray], list[float]]:
    """Look up a 3D IFS preset by name."""
    presets = {
        "barnsley_fern": _barnsley_fern_3d,
        "maple_leaf": _maple_leaf_3d,
        "spiral": _spiral_3d,
    }
    return presets[preset]()


# ---------------------------------------------------------------------------
# Chaos-game iteration (dimension-generic)
# ---------------------------------------------------------------------------


def _iterate(
    matrices: list[np.ndarray],
    offsets: list[np.ndarray],
    probs: list[float],
    num_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Run the chaos game and return (N, ndim) points.

    Pre-stacks matrices/offsets into contiguous arrays and uses scalar
    arithmetic to avoid numpy dispatch overhead on tiny vectors.
    """
    ndim = matrices[0].shape[0]
    total = num_points + _SKIP_INITIAL

    # Pre-stack for contiguous memory access
    mat_stack = np.array(matrices, dtype=np.float64)  # (K, ndim, ndim)
    off_stack = np.array(offsets, dtype=np.float64)    # (K, ndim)

    choices = rng.choice(len(matrices), size=total, p=probs)
    points = np.empty((total, ndim), dtype=np.float64)
    point = rng.normal(scale=0.01, size=ndim)

    if ndim == 2:
        _iterate_2d_scalar(mat_stack, off_stack, choices, points, point)
    elif ndim == 3:
        _iterate_3d_scalar(mat_stack, off_stack, choices, points, point)
    else:
        # Generic fallback for arbitrary dimensions
        for i in range(total):
            idx = choices[i]
            point = mat_stack[idx] @ point + off_stack[idx]
            points[i] = point

    return points[_SKIP_INITIAL:]


def _iterate_2d_scalar(
    mat_stack: np.ndarray,
    off_stack: np.ndarray,
    choices: np.ndarray,
    points: np.ndarray,
    point: np.ndarray,
) -> None:
    """Scalar-arithmetic 2D chaos game loop (avoids numpy dispatch)."""
    x, y = float(point[0]), float(point[1])
    for i in range(len(choices)):
        idx = choices[i]
        m = mat_stack[idx]
        o = off_stack[idx]
        x_new = m[0, 0] * x + m[0, 1] * y + o[0]
        y_new = m[1, 0] * x + m[1, 1] * y + o[1]
        x, y = x_new, y_new
        points[i, 0] = x
        points[i, 1] = y


def _iterate_3d_scalar(
    mat_stack: np.ndarray,
    off_stack: np.ndarray,
    choices: np.ndarray,
    points: np.ndarray,
    point: np.ndarray,
) -> None:
    """Scalar-arithmetic 3D chaos game loop (avoids numpy dispatch)."""
    x, y, z = float(point[0]), float(point[1]), float(point[2])
    for i in range(len(choices)):
        idx = choices[i]
        m = mat_stack[idx]
        o = off_stack[idx]
        x_new = m[0, 0] * x + m[0, 1] * y + m[0, 2] * z + o[0]
        y_new = m[1, 0] * x + m[1, 1] * y + m[1, 2] * z + o[1]
        z_new = m[2, 0] * x + m[2, 1] * y + m[2, 2] * z + o[2]
        x, y, z = x_new, y_new, z_new
        points[i, 0] = x
        points[i, 1] = y
        points[i, 2] = z


def _extrude_to_3d(
    points_2d: np.ndarray,
    rng: np.random.Generator,
    thickness: float = 0.05,
) -> np.ndarray:
    """Add z-axis noise to extrude 2D points into a thin 3D slab."""
    z = rng.normal(scale=thickness, size=len(points_2d))
    return np.column_stack([points_2d, z])


# ---------------------------------------------------------------------------
# Custom transform validation
# ---------------------------------------------------------------------------

_VALID_MATRIX_SHAPES = {(2, 2), (3, 3)}


def _validate_custom_transforms(
    params: dict[str, Any],
    dimensions: str,
) -> None:
    """Validate custom IFS transform parameters."""
    matrices = params.get("matrices")
    offsets = params.get("offsets")
    probs = params.get("probabilities")

    if matrices is None or offsets is None or probs is None:
        raise ValueError(
            "Custom preset requires 'matrices', 'offsets', and "
            "'probabilities' parameters"
        )
    if len(matrices) != len(offsets) or len(matrices) != len(probs):
        raise ValueError(
            "matrices, offsets, and probabilities must have equal length"
        )

    # Validate probabilities
    if any(p < 0 for p in probs):
        raise ValueError("probabilities must be non-negative")
    prob_sum = sum(probs)
    if abs(prob_sum - 1.0) > 1e-6:
        raise ValueError(
            f"probabilities must sum to 1.0, got {prob_sum}"
        )

    # Validate matrix shapes
    for i, m in enumerate(matrices):
        arr = np.asarray(m)
        if arr.shape not in _VALID_MATRIX_SHAPES:
            raise ValueError(
                f"matrices[{i}] shape {arr.shape}, expected (2, 2) or (3, 3)"
            )
        if i > 0 and arr.shape != np.asarray(matrices[0]).shape:
            raise ValueError(
                f"matrices[{i}] shape {arr.shape} differs from "
                f"matrices[0] shape {np.asarray(matrices[0]).shape}"
            )

    # Validate offset dimensions match matrix dimensions
    ndim = np.asarray(matrices[0]).shape[0]
    for i, o in enumerate(offsets):
        arr = np.asarray(o)
        if arr.shape != (ndim,):
            raise ValueError(
                f"offsets[{i}] shape {arr.shape}, expected ({ndim},)"
            )

    # Validate dimension compatibility
    if ndim == 3 and dimensions == "2d_extruded":
        raise ValueError(
            "3×3 custom matrices are incompatible with dimensions='2d_extruded'; "
            "use dimensions='3d' or provide 2×2 matrices"
        )


# ---------------------------------------------------------------------------
# Generator class
# ---------------------------------------------------------------------------


@register
class IFSFractalGenerator(GeneratorBase):
    """Iterated Function System fractal generator."""

    name = "ifs_fractal"
    aliases = ("ifs", "barnsley_fern")
    description = "IFS fractal generator (Barnsley fern, maple leaf, spiral, custom)"
    category = "fractals"

    resolution_params = {"num_points": "Number of chaos-game points to generate"}
    _resolution_defaults = {"num_points": DEFAULT_NUM_POINTS}

    def get_default_params(self) -> dict[str, Any]:
        """Return default IFS parameters."""
        return {
            "preset": "barnsley_fern",
            "dimensions": "3d",
        }

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate an IFS fractal point cloud."""
        merged = self.get_default_params()
        if params:
            merged.update(params)

        num_points = int(
            resolution_kwargs.get("num_points", DEFAULT_NUM_POINTS)
        )
        if num_points < MIN_NUM_POINTS:
            raise ValueError(
                f"num_points must be >= {MIN_NUM_POINTS}, got {num_points}"
            )

        preset = str(merged["preset"])
        dimensions = str(merged["dimensions"])

        if preset not in VALID_PRESETS:
            raise ValueError(
                f"preset must be one of {VALID_PRESETS}, got {preset!r}"
            )
        if dimensions not in VALID_DIMENSIONS:
            raise ValueError(
                f"dimensions must be one of {VALID_DIMENSIONS}, "
                f"got {dimensions!r}"
            )

        rng = default_rng(seed)

        if preset == "custom":
            points = self._generate_custom(merged, dimensions, num_points, rng)
        elif dimensions == "3d":
            points = self._generate_3d(preset, num_points, rng)
        else:
            points = self._generate_2d_extruded(preset, num_points, rng)

        merged["num_points"] = num_points
        cloud = PointCloud(points=points)
        bbox = BoundingBox.from_points(points)

        logger.info(
            "Generated IFS fractal (%s, %s): %d points",
            preset, dimensions, num_points,
        )

        return MathObject(
            point_cloud=cloud,
            generator_name=self.name,
            category=self.category,
            parameters=merged,
            seed=seed,
            bounding_box=bbox,
        )

    def _generate_3d(
        self, preset: str, num_points: int, rng: np.random.Generator,
    ) -> np.ndarray:
        """Generate points using native 3D affine transforms."""
        matrices, offsets, probs = _get_3d_preset(preset)
        return _iterate(matrices, offsets, probs, num_points, rng)

    def _generate_2d_extruded(
        self, preset: str, num_points: int, rng: np.random.Generator,
    ) -> np.ndarray:
        """Generate 2D IFS points and extrude to 3D."""
        matrices, offsets, probs = _get_2d_preset(preset)
        points_2d = _iterate(matrices, offsets, probs, num_points, rng)
        return _extrude_to_3d(points_2d, rng)

    def _generate_custom(
        self,
        params: dict[str, Any],
        dimensions: str,
        num_points: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Generate points from user-supplied affine transforms."""
        _validate_custom_transforms(params, dimensions)
        raw_matrices = params["matrices"]
        raw_offsets = params["offsets"]
        probs = list(params["probabilities"])

        matrices = [np.asarray(m, dtype=np.float64) for m in raw_matrices]
        offsets = [np.asarray(o, dtype=np.float64) for o in raw_offsets]

        ndim = matrices[0].shape[0]
        if ndim == 2 and dimensions == "3d":
            matrices = [_embed_2d_to_3d(m, 0.3) for m in matrices]
            offsets = _embed_offsets_2d_to_3d(offsets)
            return _iterate(matrices, offsets, probs, num_points, rng)
        if ndim == 2:
            pts_2d = _iterate(matrices, offsets, probs, num_points, rng)
            return _extrude_to_3d(pts_2d, rng)
        return _iterate(matrices, offsets, probs, num_points, rng)

    def get_default_representation(self) -> RepresentationConfig:
        """Return SPARSE_SHELL as the default representation."""
        return RepresentationConfig(type=RepresentationType.SPARSE_SHELL)
