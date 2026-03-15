"""Hydrogen atom electron orbital generator.

Computes probability density isosurfaces |ψ(r,θ,φ)|² for hydrogen atom
wavefunctions using the radial function R_nl(r) and spherical harmonics
Y_lm(θ,φ). Different (n,l,m) quantum numbers produce iconic orbital shapes:
(1,0,0) = sphere, (2,1,0) = dumbbell, (3,2,0) = cloverleaf.
"""

import logging
from typing import Any

import numpy as np
from scipy.special import assoc_laguerre, factorial, sph_harm_y

from mathviz.core.generator import GeneratorBase, register
from mathviz.core.math_object import BoundingBox, MathObject
from mathviz.core.representation import RepresentationConfig, RepresentationType
from mathviz.shared.marching_cubes import SpatialBounds, extract_mesh

logger = logging.getLogger(__name__)

# Bohr radius (atomic units)
_BOHR_RADIUS = 1.0

# Default parameters
_DEFAULT_N = 3
_DEFAULT_L = 2
_DEFAULT_M = 0
_DEFAULT_VOXEL_RESOLUTION = 128
_DEFAULT_ISO_LEVEL = 0.01

# Grid extent scaling factor per principal quantum number
_EXTENT_SCALE = 6.0

# Minimum voxel resolution
_MIN_VOXEL_RESOLUTION = 16


def _validate_quantum_numbers(n: int, l: int, m: int) -> None:
    """Validate hydrogen atom quantum numbers."""
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if l < 0:
        raise ValueError(f"l must be >= 0, got {l}")
    if l >= n:
        raise ValueError(f"l must be < n, got l={l} with n={n}")
    if abs(m) > l:
        raise ValueError(f"|m| must be <= l, got m={m} with l={l}")


def _validate_grid_params(voxel_resolution: int, iso_level: float) -> None:
    """Validate grid and isosurface parameters."""
    if voxel_resolution < _MIN_VOXEL_RESOLUTION:
        raise ValueError(
            f"voxel_resolution must be >= {_MIN_VOXEL_RESOLUTION}, "
            f"got {voxel_resolution}"
        )
    if iso_level <= 0:
        raise ValueError(f"iso_level must be > 0, got {iso_level}")


def _compute_radial(n: int, l: int, r: np.ndarray) -> np.ndarray:
    """Compute the radial wavefunction R_nl(r) for hydrogen."""
    rho = 2.0 * r / (n * _BOHR_RADIUS)
    norm_num = (2.0 / (n * _BOHR_RADIUS)) ** 3 * factorial(n - l - 1, exact=True)
    norm_den = 2.0 * n * factorial(n + l, exact=True)
    normalization = np.sqrt(norm_num / norm_den)
    laguerre = assoc_laguerre(rho, n - l - 1, 2 * l + 1)
    return normalization * np.exp(-rho / 2.0) * rho**l * laguerre


def _compute_probability_density(
    n: int, l: int, m: int, grid_extent: float, voxel_resolution: int
) -> tuple[np.ndarray, SpatialBounds]:
    """Evaluate |ψ(r,θ,φ)|² on a 3D Cartesian grid."""
    coords = np.linspace(-grid_extent, grid_extent, voxel_resolution)
    x, y, z = np.meshgrid(coords, coords, coords, indexing="ij")

    r = np.sqrt(x**2 + y**2 + z**2)
    r = np.maximum(r, 1e-12)  # avoid division by zero
    theta = np.arccos(np.clip(z / r, -1.0, 1.0))
    phi = np.arctan2(y, x) % (2.0 * np.pi)

    radial = _compute_radial(n, l, r)
    angular = sph_harm_y(l, m, theta, phi)
    psi = radial * np.real(angular)
    density = psi**2

    bounds = SpatialBounds(
        min_corner=(-grid_extent, -grid_extent, -grid_extent),
        max_corner=(grid_extent, grid_extent, grid_extent),
    )
    return density, bounds


@register
class ElectronOrbitalGenerator(GeneratorBase):
    """Hydrogen atom electron orbital probability density isosurface."""

    name = "electron_orbital"
    category = "physics"
    aliases = ("hydrogen_orbital",)
    description = (
        "Hydrogen atom electron orbital probability density isosurface"
    )
    resolution_params = {
        "voxel_resolution": "Voxels per axis (N³ cost)",
    }
    _resolution_defaults = {"voxel_resolution": _DEFAULT_VOXEL_RESOLUTION}

    def get_default_params(self) -> dict[str, Any]:
        """Return default quantum number parameters."""
        return {
            "n": _DEFAULT_N,
            "l": _DEFAULT_L,
            "m": _DEFAULT_M,
            "iso_level": _DEFAULT_ISO_LEVEL,
        }

    def get_param_ranges(self) -> dict[str, dict[str, float]]:
        """Return exploration ranges for orbital parameters."""
        return {
            "n": {"min": 1, "max": 4, "step": 1},
            "l": {"min": 0, "max": 3, "step": 1},
            "m": {"min": -3, "max": 3, "step": 1},
            "iso_level": {"min": 0.001, "max": 0.1, "step": 0.001},
        }

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate an electron orbital isosurface mesh."""
        merged = self.get_default_params()
        if params:
            merged.update(params)

        n = int(merged["n"])
        l = int(merged["l"])
        m = int(merged["m"])
        iso_level = float(merged["iso_level"])
        voxel_resolution = int(
            resolution_kwargs.get("voxel_resolution", _DEFAULT_VOXEL_RESOLUTION)
        )

        _validate_quantum_numbers(n, l, m)
        _validate_grid_params(voxel_resolution, iso_level)
        merged["voxel_resolution"] = voxel_resolution

        grid_extent = _EXTENT_SCALE * n * _BOHR_RADIUS
        density, bounds = _compute_probability_density(
            n, l, m, grid_extent, voxel_resolution
        )

        # Normalize density so max = 1 for consistent iso_level interpretation
        max_density = density.max()
        if max_density > 0:
            density = density / max_density

        mesh = extract_mesh(density, bounds, isolevel=iso_level)
        bbox = BoundingBox(
            min_corner=bounds.min_corner,
            max_corner=bounds.max_corner,
        )

        logger.info(
            "Generated electron_orbital: n=%d, l=%d, m=%d, "
            "voxel_res=%d, vertices=%d, faces=%d",
            n, l, m, voxel_resolution, len(mesh.vertices), len(mesh.faces),
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
