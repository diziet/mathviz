"""Lissajous knot tube surface generator.

Generates a closed Lissajous knot curve and sweeps a circular cross-section
along it using parallel-transport (Bishop) frames. The result is a tubular
surface that wraps in both the longitudinal and cross-sectional directions.
"""

import logging
from typing import Any

import numpy as np

from mathviz.core.generator import GeneratorBase, register
from mathviz.core.math_object import BoundingBox, MathObject, Mesh
from mathviz.core.representation import RepresentationConfig, RepresentationType

logger = logging.getLogger(__name__)

_DEFAULT_NX = 2
_DEFAULT_NY = 3
_DEFAULT_NZ = 5
_DEFAULT_PHASE_X = 0.0
_DEFAULT_PHASE_Y = 0.0
_DEFAULT_PHASE_Z = 0.0
_DEFAULT_TUBE_RADIUS = 0.1
_DEFAULT_GRID_RESOLUTION = 128
_MIN_GRID_RESOLUTION = 4


def _evaluate_lissajous_curve(
    t: np.ndarray, nx: int, ny: int, nz: int,
    phase_x: float, phase_y: float, phase_z: float,
) -> np.ndarray:
    """Evaluate the Lissajous knot curve at parameter values t in [0, 2pi)."""
    x = np.cos(nx * t + phase_x)
    y = np.cos(ny * t + phase_y)
    z = np.cos(nz * t + phase_z)
    return np.column_stack([x, y, z])


def _compute_bishop_frames(
    curve: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute parallel-transport frames along a closed curve.

    Returns tangent, normal, binormal arrays each of shape (N, 3).
    """
    n = len(curve)
    tangents = np.roll(curve, -1, axis=0) - np.roll(curve, 1, axis=0)
    tangent_norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    tangent_norms = np.clip(tangent_norms, 1e-10, None)
    tangents = tangents / tangent_norms

    # Initial normal: perpendicular to first tangent
    t0 = tangents[0]
    arbitrary = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(t0, arbitrary)) > 0.9:
        arbitrary = np.array([1.0, 0.0, 0.0])
    normal = np.cross(t0, arbitrary)
    normal = normal / np.linalg.norm(normal)

    normals = np.empty((n, 3), dtype=np.float64)
    normals[0] = normal

    # Propagate via parallel transport
    for i in range(1, n):
        b = np.cross(tangents[i - 1], tangents[i])
        b_norm = np.linalg.norm(b)
        if b_norm < 1e-10:
            normals[i] = normals[i - 1]
        else:
            b = b / b_norm
            angle = np.arccos(np.clip(
                np.dot(tangents[i - 1], tangents[i]), -1.0, 1.0,
            ))
            normals[i] = _rotate_around_axis(normals[i - 1], b, angle)

    binormals = np.cross(tangents, normals)
    return tangents, normals, binormals


def _rotate_around_axis(
    vec: np.ndarray, axis: np.ndarray, angle: float,
) -> np.ndarray:
    """Rodrigues' rotation of vec around axis by angle."""
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    return (
        vec * cos_a
        + np.cross(axis, vec) * sin_a
        + axis * np.dot(axis, vec) * (1 - cos_a)
    )


def _generate_tube_mesh(
    curve: np.ndarray, normals: np.ndarray, binormals: np.ndarray,
    tube_radius: float, n_cross: int,
) -> Mesh:
    """Sweep a circle along curve to create a tube mesh."""
    n_along = len(curve)
    theta = np.linspace(0, 2 * np.pi, n_cross, endpoint=False)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    vertices = np.empty((n_along * n_cross, 3), dtype=np.float64)
    for i in range(n_along):
        for j in range(n_cross):
            offset = tube_radius * (
                cos_t[j] * normals[i] + sin_t[j] * binormals[i]
            )
            vertices[i * n_cross + j] = curve[i] + offset

    faces = _build_wrapped_grid_faces(n_along, n_cross)
    return Mesh(vertices=vertices, faces=faces)


def _build_wrapped_grid_faces(n_u: int, n_v: int) -> np.ndarray:
    """Build triangle faces for a grid periodic in both u and v."""
    rows = np.arange(n_u)
    cols = np.arange(n_v)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    rr, cc = rr.ravel(), cc.ravel()

    i00 = rr * n_v + cc
    i10 = ((rr + 1) % n_u) * n_v + cc
    i01 = rr * n_v + ((cc + 1) % n_v)
    i11 = ((rr + 1) % n_u) * n_v + ((cc + 1) % n_v)

    tri1 = np.stack([i00, i10, i11], axis=-1)
    tri2 = np.stack([i00, i11, i01], axis=-1)
    return np.concatenate([tri1, tri2], axis=0).astype(np.int64)


def _validate_params(
    nx: int, ny: int, nz: int, tube_radius: float, grid_resolution: int,
) -> None:
    """Validate Lissajous surface parameters."""
    for name, val in [("nx", nx), ("ny", ny), ("nz", nz)]:
        if val < 1:
            raise ValueError(f"{name} must be >= 1, got {val}")
    if tube_radius <= 0:
        raise ValueError(f"tube_radius must be positive, got {tube_radius}")
    if grid_resolution < _MIN_GRID_RESOLUTION:
        raise ValueError(
            f"grid_resolution must be >= {_MIN_GRID_RESOLUTION}, "
            f"got {grid_resolution}"
        )


@register
class LissajousSurfaceGenerator(GeneratorBase):
    """Lissajous knot tube surface generator."""

    name = "lissajous_surface"
    category = "parametric"
    aliases = ()
    description = "Tubular surface around a Lissajous knot curve"
    resolution_params = {"grid_resolution": "Number of grid divisions per axis"}

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters for the Lissajous surface."""
        return {
            "nx": _DEFAULT_NX, "ny": _DEFAULT_NY, "nz": _DEFAULT_NZ,
            "phase_x": _DEFAULT_PHASE_X,
            "phase_y": _DEFAULT_PHASE_Y,
            "phase_z": _DEFAULT_PHASE_Z,
            "tube_radius": _DEFAULT_TUBE_RADIUS,
        }

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate a Lissajous knot tube mesh."""
        merged = self.get_default_params()
        if params:
            merged.update(params)

        nx = int(merged["nx"])
        ny = int(merged["ny"])
        nz = int(merged["nz"])
        phase_x = float(merged["phase_x"])
        phase_y = float(merged["phase_y"])
        phase_z = float(merged["phase_z"])
        tube_radius = float(merged["tube_radius"])
        grid_resolution = int(
            resolution_kwargs.get("grid_resolution", _DEFAULT_GRID_RESOLUTION)
        )

        _validate_params(nx, ny, nz, tube_radius, grid_resolution)

        n_along = grid_resolution
        n_cross = max(grid_resolution // 4, _MIN_GRID_RESOLUTION)
        t = np.linspace(0, 2 * np.pi, n_along, endpoint=False)
        curve = _evaluate_lissajous_curve(t, nx, ny, nz, phase_x, phase_y, phase_z)
        _, normals, binormals = _compute_bishop_frames(curve)
        mesh = _generate_tube_mesh(curve, normals, binormals, tube_radius, n_cross)

        extent = 1.0 + tube_radius
        bbox = BoundingBox(
            min_corner=(-extent, -extent, -extent),
            max_corner=(extent, extent, extent),
        )

        logger.info(
            "Generated lissajous_surface: nx=%d, ny=%d, nz=%d, r=%.3f, "
            "grid=%d, vertices=%d, faces=%d",
            nx, ny, nz, tube_radius, grid_resolution,
            len(mesh.vertices), len(mesh.faces),
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
        """Return the recommended representation."""
        return RepresentationConfig(type=RepresentationType.SURFACE_SHELL)
