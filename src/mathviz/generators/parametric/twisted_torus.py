"""Twisted torus parametric surface generator.

A torus where the circular cross-section rotates N half-twists as it
traverses the loop. twist=0 is a standard torus; twist=1 yields a
Möbius-torus (non-orientable). Even twists produce orientable surfaces.
"""

import logging
from typing import Any

import numpy as np

from mathviz.core.generator import GeneratorBase, register
from mathviz.core.math_object import MathObject, Mesh
from mathviz.core.representation import RepresentationConfig, RepresentationType
from mathviz.generators.parametric._mesh_utils import compute_padded_bounding_box

logger = logging.getLogger(__name__)

_DEFAULT_TWIST = 3
_DEFAULT_MAJOR_RADIUS = 1.0
_DEFAULT_MINOR_RADIUS = 0.3
_DEFAULT_GRID_RESOLUTION = 128
_MIN_GRID_RESOLUTION = 3
_MIN_TWIST = 0


def _evaluate_twisted_torus(
    u: np.ndarray,
    v: np.ndarray,
    major_radius: float,
    minor_radius: float,
    twist: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the twisted torus surface f(u, v) -> (x, y, z).

    The cross-section angle is offset by twist * u / 2 half-twists,
    so after a full loop in u the section has rotated by twist * pi.
    """
    v_eff = v + twist * u / 2.0
    x = (major_radius + minor_radius * np.cos(v_eff)) * np.cos(u)
    y = (major_radius + minor_radius * np.cos(v_eff)) * np.sin(u)
    z = minor_radius * np.sin(v_eff)
    return x, y, z


def _build_even_twist_faces(n: int) -> np.ndarray:
    """Build faces for even twist (wraps in both u and v)."""
    row = np.arange(n)
    col = np.arange(n)
    rr, cc = np.meshgrid(row, col, indexing="ij")
    rr, cc = rr.ravel(), cc.ravel()

    i00 = rr * n + cc
    i10 = ((rr + 1) % n) * n + cc
    i01 = rr * n + ((cc + 1) % n)
    i11 = ((rr + 1) % n) * n + ((cc + 1) % n)

    tri1 = np.stack([i00, i10, i11], axis=-1)
    tri2 = np.stack([i00, i11, i01], axis=-1)
    return np.concatenate([tri1, tri2], axis=0).astype(np.int64)


def _build_odd_twist_faces(n: int) -> np.ndarray:
    """Build faces for odd twist (non-orientable seam).

    Interior rows (0..n-2) wrap in v normally. At the u-seam (row n-1
    to row 0), vertex (n-1, j) connects to (0, (j + n//2) % n) with
    reversed winding, creating a non-orientable surface analogous to
    a Klein bottle or Möbius strip.
    """
    half = n // 2

    # Interior faces: rows 0..n-2
    rows = np.arange(n - 1)
    cols = np.arange(n)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    rr, cc = rr.ravel(), cc.ravel()

    i00 = rr * n + cc
    i10 = (rr + 1) * n + cc
    i01 = rr * n + ((cc + 1) % n)
    i11 = (rr + 1) * n + ((cc + 1) % n)

    interior_t1 = np.stack([i00, i10, i11], axis=-1)
    interior_t2 = np.stack([i00, i11, i01], axis=-1)

    # Seam faces: row n-1 connects to row 0 with half-period v shift.
    # Winding is reversed to make the surface non-orientable — the face
    # normals at the seam are inconsistent with interior normals, which
    # is the defining property of a non-orientable surface.
    sc = np.arange(n)
    s00 = (n - 1) * n + sc
    s01 = (n - 1) * n + ((sc + 1) % n)
    s10 = (sc + half) % n  # row 0, shifted by half period
    s11 = ((sc + 1 + half) % n)  # row 0, next column shifted

    seam_t1 = np.stack([s00, s11, s10], axis=-1)
    seam_t2 = np.stack([s00, s01, s11], axis=-1)

    return np.concatenate(
        [interior_t1, interior_t2, seam_t1, seam_t2], axis=0,
    ).astype(np.int64)


def _build_twisted_torus_mesh(
    major_radius: float,
    minor_radius: float,
    twist: int,
    grid_resolution: int,
) -> Mesh:
    """Build triangle mesh for a twisted torus."""
    n = grid_resolution
    u_vals = np.linspace(0, 2 * np.pi, n, endpoint=False)
    v_vals = np.linspace(0, 2 * np.pi, n, endpoint=False)
    uu, vv = np.meshgrid(u_vals, v_vals, indexing="ij")

    x, y, z = _evaluate_twisted_torus(uu, vv, major_radius, minor_radius, twist)

    vertices = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    vertices = vertices.astype(np.float64)

    is_even_twist = (twist % 2 == 0)
    if is_even_twist:
        faces = _build_even_twist_faces(n)
    else:
        faces = _build_odd_twist_faces(n)

    return Mesh(vertices=vertices, faces=faces)


def _validate_params(
    major_radius: float,
    minor_radius: float,
    twist: int,
    grid_resolution: int,
) -> None:
    """Validate twisted torus parameters."""
    if major_radius <= 0:
        raise ValueError(f"major_radius must be positive, got {major_radius}")
    if minor_radius <= 0:
        raise ValueError(f"minor_radius must be positive, got {minor_radius}")
    if twist < _MIN_TWIST:
        raise ValueError(f"twist must be >= {_MIN_TWIST}, got {twist}")
    if grid_resolution < _MIN_GRID_RESOLUTION:
        raise ValueError(
            f"grid_resolution must be >= {_MIN_GRID_RESOLUTION}, "
            f"got {grid_resolution}"
        )
    if minor_radius >= major_radius:
        logger.warning(
            "minor_radius (%.3f) >= major_radius (%.3f): "
            "torus will self-intersect",
            minor_radius, major_radius,
        )


@register
class TwistedTorusGenerator(GeneratorBase):
    """Parametric twisted torus surface generator."""

    name = "twisted_torus"
    category = "parametric"
    aliases = ()
    description = (
        "Torus with cross-section rotating N half-twists around the loop; "
        "twist=1 produces a Möbius-torus"
    )
    resolution_params = {"grid_resolution": "Number of grid divisions per axis"}
    _resolution_defaults = {"grid_resolution": _DEFAULT_GRID_RESOLUTION}

    def get_param_ranges(self) -> dict[str, dict[str, float]]:
        """Return exploration ranges for twisted torus parameters."""
        return {
            "twist": {"min": 0, "max": 12, "step": 1},
            "major_radius": {"min": 0.3, "max": 3.0, "step": 0.1},
            "minor_radius": {"min": 0.05, "max": 1.0, "step": 0.05},
        }

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters for the twisted torus."""
        return {
            "twist": _DEFAULT_TWIST,
            "major_radius": _DEFAULT_MAJOR_RADIUS,
            "minor_radius": _DEFAULT_MINOR_RADIUS,
        }

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate a twisted torus mesh."""
        merged = self.get_default_params()
        if params:
            merged.update(params)

        twist = int(merged["twist"])
        major_radius = float(merged["major_radius"])
        minor_radius = float(merged["minor_radius"])
        grid_resolution = int(
            resolution_kwargs.get("grid_resolution", _DEFAULT_GRID_RESOLUTION)
        )

        _validate_params(major_radius, minor_radius, twist, grid_resolution)

        mesh = _build_twisted_torus_mesh(
            major_radius, minor_radius, twist, grid_resolution,
        )
        bbox = compute_padded_bounding_box(mesh.vertices)

        merged["grid_resolution"] = grid_resolution

        logger.info(
            "Generated twisted_torus: twist=%d, R=%.3f, r=%.3f, "
            "grid=%d, vertices=%d, faces=%d",
            twist, major_radius, minor_radius, grid_resolution,
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
        """Return the recommended representation for the twisted torus."""
        return RepresentationConfig(type=RepresentationType.SURFACE_SHELL)
