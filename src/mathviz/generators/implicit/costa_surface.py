"""Costa surface generator via Weierstrass-Enneper representation.

The Costa surface is a complete embedded minimal surface of genus 1 with three
ends (one flat, two catenoidal). It was the first known example of a complete
embedded minimal surface of finite topology beyond the plane, catenoid, and
helicoid.

Mathematical construction:
- Domain: a patch of the fundamental domain of the square torus C/(Z+iZ),
  avoiding punctures at z=0, z=1/2, z=i/2
- Weierstrass data: f(z) = P(z), g(z) = C/P'(z) where P is the Weierstrass
  elliptic function for the square lattice, computed via truncated lattice sums
- Coordinates obtained via numerical integration of the Weierstrass-Enneper
  representation along L-shaped paths on the grid
"""

import logging
from typing import Any

import numpy as np

from mathviz.core.generator import GeneratorBase, register
from mathviz.core.math_object import BoundingBox, MathObject, Mesh
from mathviz.core.representation import RepresentationConfig, RepresentationType
from mathviz.generators.parametric._mesh_utils import build_open_grid_faces

logger = logging.getLogger(__name__)

_DEFAULT_GRID_RESOLUTION = 64
_MIN_GRID_RESOLUTION = 8
_DEFAULT_SCALE = 0.1
_LATTICE_TERMS = 3
_DOMAIN_MARGIN = 0.06


def _weierstrass_p(z: np.ndarray) -> np.ndarray:
    """Compute Weierstrass P(z) for the square lattice Z+iZ via lattice sum."""
    result = 1.0 / z**2
    for m in range(-_LATTICE_TERMS, _LATTICE_TERMS + 1):
        for n in range(-_LATTICE_TERMS, _LATTICE_TERMS + 1):
            if m == 0 and n == 0:
                continue
            omega = complex(m, n)
            result += 1.0 / (z - omega) ** 2 - 1.0 / omega**2
    return result


def _weierstrass_p_prime(z: np.ndarray) -> np.ndarray:
    """Compute P'(z) for the square lattice Z+iZ via lattice sum."""
    result = -2.0 / z**3
    for m in range(-_LATTICE_TERMS, _LATTICE_TERMS + 1):
        for n in range(-_LATTICE_TERMS, _LATTICE_TERMS + 1):
            if m == 0 and n == 0:
                continue
            omega = complex(m, n)
            result += -2.0 / (z - omega) ** 3
    return result


def _cumtrapz_complex(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Cumulative trapezoidal integration for complex-valued y over real x."""
    dx = np.diff(x)
    mid = (y[:-1] + y[1:]) / 2.0
    return np.concatenate([[0.0 + 0.0j], np.cumsum(mid * dx)])


def _compute_costa_mesh(
    grid_resolution: int, scale: float,
) -> tuple[Mesh, BoundingBox]:
    """Build the Costa surface mesh via Weierstrass-Enneper integration."""
    delta = _DOMAIN_MARGIN
    u_vals = np.linspace(delta, 0.5 - delta, grid_resolution)
    v_vals = np.linspace(delta, 0.5 - delta, grid_resolution)

    uu, vv = np.meshgrid(u_vals, v_vals, indexing="ij")
    z_grid = uu + 1j * vv

    p_val = _weierstrass_p(z_grid)
    pp_val = _weierstrass_p_prime(z_grid)

    e1_val = np.real(_weierstrass_p(np.array([0.5 + 0.0j]))[0])
    gauss_map_scale = np.sqrt(2.0 * abs(e1_val))

    g = gauss_map_scale / pp_val
    g2 = g * g

    phi1 = p_val * (1.0 - g2) / 2.0
    phi2 = 1j * p_val * (1.0 + g2) / 2.0
    phi3 = p_val * g

    coords = np.zeros((3, grid_resolution, grid_resolution))

    for k, phi in enumerate([phi1, phi2, phi3]):
        bottom = _cumtrapz_complex(phi[:, 0], u_vals)
        for i in range(grid_resolution):
            col_integrand = phi[i, :] * 1j
            col_integral = _cumtrapz_complex(col_integrand, v_vals)
            coords[k, i, :] = np.real(bottom[i] + col_integral)

    x = coords[0].ravel() * scale
    y = coords[1].ravel() * scale
    z = coords[2].ravel() * scale
    vertices = np.column_stack([x, y, z]).astype(np.float64)

    _replace_nan_with_neighbors(vertices)

    faces = build_open_grid_faces(grid_resolution, grid_resolution)
    bbox = _compute_bounding_box(vertices)

    return Mesh(vertices=vertices, faces=faces), bbox


def _replace_nan_with_neighbors(vertices: np.ndarray) -> None:
    """Replace any NaN vertices with the mean of their non-NaN neighbors."""
    nan_mask = np.any(np.isnan(vertices), axis=1)
    if not np.any(nan_mask):
        return
    good_mean = np.nanmean(vertices, axis=0)
    vertices[nan_mask] = good_mean


def _compute_bounding_box(vertices: np.ndarray) -> BoundingBox:
    """Compute tight bounding box from vertex positions."""
    min_c = tuple(float(v) for v in vertices.min(axis=0))
    max_c = tuple(float(v) for v in vertices.max(axis=0))
    return BoundingBox(min_corner=min_c, max_corner=max_c)


def _validate_params(grid_resolution: int, scale: float) -> None:
    """Validate Costa surface parameters."""
    if grid_resolution < _MIN_GRID_RESOLUTION:
        raise ValueError(
            f"grid_resolution must be >= {_MIN_GRID_RESOLUTION}, "
            f"got {grid_resolution}"
        )
    if scale <= 0:
        raise ValueError(f"scale must be positive, got {scale}")


@register
class CostaSurfaceGenerator(GeneratorBase):
    """Costa minimal surface generator via Weierstrass-Enneper representation.

    Computes one patch of the Costa surface on the fundamental domain of the
    square torus, using numerically evaluated Weierstrass elliptic functions.
    The surface has genus 1 with three ends (one flat, two catenoidal).

    Seed has no effect; the surface is fully deterministic for given parameters.
    """

    name = "costa_surface"
    category = "implicit"
    aliases = ()
    description = "Costa minimal surface via Weierstrass-Enneper representation"
    resolution_params = {
        "grid_resolution": "Number of grid divisions per axis",
    }

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters for the Costa surface generator."""
        return {"scale": _DEFAULT_SCALE}

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate a Costa surface mesh."""
        merged = self.get_default_params()
        if params:
            merged.update(params)

        scale = float(merged["scale"])
        grid_resolution = int(
            resolution_kwargs.get("grid_resolution", _DEFAULT_GRID_RESOLUTION)
        )

        _validate_params(grid_resolution, scale)

        merged["grid_resolution"] = grid_resolution

        mesh, bbox = _compute_costa_mesh(grid_resolution, scale)

        logger.info(
            "Generated costa_surface: grid=%d, scale=%.3f, "
            "vertices=%d, faces=%d",
            grid_resolution, scale,
            len(mesh.vertices), len(mesh.faces),
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
        """Return the recommended representation for the Costa surface."""
        return RepresentationConfig(type=RepresentationType.SURFACE_SHELL)
