"""Reaction-diffusion (Gray-Scott) on curved surfaces.

Runs the Gray-Scott model on a surface mesh (torus, sphere, or Klein bottle)
using the mesh Laplacian for diffusion. Displaces vertices along normals
proportional to the V concentration to produce Turing-pattern geometry.

Pattern presets (feed_rate, kill_rate):
  - Spots:   (0.035, 0.065)
  - Stripes: (0.055, 0.062)
  - Maze:    (0.029, 0.057)
"""

import logging
from typing import Any

import numpy as np
from numpy.random import default_rng

from mathviz.core.generator import GeneratorBase, register
from mathviz.core.math_object import BoundingBox, MathObject, Mesh
from mathviz.core.representation import RepresentationConfig, RepresentationType
from mathviz.generators.procedural._rd_surface_mesh import (
    VALID_SURFACES,
    build_mesh_laplacian,
    compute_vertex_normals,
    generate_base_mesh,
)

logger = logging.getLogger(__name__)

_DEFAULT_BASE_SURFACE = "torus"
_DEFAULT_FEED_RATE = 0.055
_DEFAULT_KILL_RATE = 0.062
_DEFAULT_DIFFUSION_U = 0.16
_DEFAULT_DIFFUSION_V = 0.08
_DEFAULT_ITERATIONS = 5000
_DEFAULT_DISPLACEMENT_SCALE = 0.1
_DEFAULT_GRID_RESOLUTION = 128
_MIN_GRID_RESOLUTION = 8
_MIN_ITERATIONS = 100
_MAX_ITERATIONS = 50000


def _require_positive(name: str, value: float) -> None:
    """Raise ValueError if value is not positive."""
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def _validate_params(
    base_surface: str,
    feed_rate: float,
    kill_rate: float,
    diffusion_u: float,
    diffusion_v: float,
    iterations: int,
    displacement_scale: float,
    grid_resolution: int,
) -> None:
    """Validate reaction-diffusion surface parameters."""
    if base_surface not in VALID_SURFACES:
        raise ValueError(
            f"base_surface must be one of {VALID_SURFACES}, got {base_surface!r}"
        )
    _require_positive("feed_rate", feed_rate)
    _require_positive("kill_rate", kill_rate)
    _require_positive("diffusion_u", diffusion_u)
    _require_positive("diffusion_v", diffusion_v)
    if displacement_scale < 0:
        raise ValueError(
            f"displacement_scale must be >= 0, got {displacement_scale}"
        )
    if iterations < _MIN_ITERATIONS:
        raise ValueError(
            f"iterations must be >= {_MIN_ITERATIONS}, got {iterations}"
        )
    if iterations > _MAX_ITERATIONS:
        raise ValueError(
            f"iterations must be <= {_MAX_ITERATIONS}, got {iterations}"
        )
    if grid_resolution < _MIN_GRID_RESOLUTION:
        raise ValueError(
            f"grid_resolution must be >= {_MIN_GRID_RESOLUTION}, got {grid_resolution}"
        )


def _init_vertex_concentrations(
    num_vertices: int, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Initialize U=1, V=0 everywhere, then seed V with random patches."""
    rng = default_rng(seed)
    u_field = np.ones(num_vertices, dtype=np.float64)
    v_field = np.zeros(num_vertices, dtype=np.float64)

    num_patches = rng.integers(3, 8)
    patch_fraction = 0.05

    for _ in range(num_patches):
        patch_size = max(1, int(num_vertices * patch_fraction))
        indices = rng.choice(num_vertices, size=patch_size, replace=False)
        u_field[indices] = 0.50
        v_field[indices] = 0.25
        noise = rng.uniform(-0.01, 0.01, patch_size)
        u_field[indices] += noise
        v_field[indices] += noise * 0.5

    return u_field, v_field


def _run_gray_scott_mesh(
    u_field: np.ndarray,
    v_field: np.ndarray,
    laplacian: Any,
    feed_rate: float,
    kill_rate: float,
    diffusion_u: float,
    diffusion_v: float,
    iterations: int,
) -> np.ndarray:
    """Run Gray-Scott iteration on mesh vertices using the mesh Laplacian."""
    dt = 1.0
    check_interval = max(1, iterations // 10)
    for step in range(iterations):
        lap_u = laplacian @ u_field
        lap_v = laplacian @ v_field
        uvv = u_field * v_field * v_field

        u_field += dt * (diffusion_u * lap_u - uvv + feed_rate * (1.0 - u_field))
        v_field += dt * (diffusion_v * lap_v + uvv - (feed_rate + kill_rate) * v_field)

        if step % check_interval == 0:
            if not (np.all(np.isfinite(u_field)) and np.all(np.isfinite(v_field))):
                raise RuntimeError(
                    f"Gray-Scott diverged at step {step}: NaN/Inf detected. "
                    f"Reduce diffusion_u/diffusion_v or increase grid_resolution."
                )

        np.clip(u_field, 0.0, 1.0, out=u_field)
        np.clip(v_field, 0.0, 1.0, out=v_field)

    return v_field


def _displace_mesh(
    mesh: Mesh, v_field: np.ndarray, displacement_scale: float,
) -> Mesh:
    """Displace mesh vertices along normals by V * displacement_scale."""
    normals = compute_vertex_normals(mesh.vertices, mesh.faces)
    displaced = mesh.vertices + normals * (v_field[:, np.newaxis] * displacement_scale)
    new_normals = compute_vertex_normals(displaced, mesh.faces)
    return Mesh(
        vertices=displaced.astype(np.float64),
        faces=mesh.faces,
        normals=new_normals,
    )


@register
class ReactionDiffusionSurface(GeneratorBase):
    """Gray-Scott reaction-diffusion on curved surfaces.

    Simulates reaction-diffusion directly on a surface mesh, displacing
    vertices along normals to produce organic Turing-pattern geometry.
    """

    name = "rd_surface"
    category = "procedural"
    aliases = ("reaction_diffusion_surface",)
    description = "Gray-Scott reaction-diffusion on curved surface meshes"
    resolution_params = {
        "grid_resolution": "Base surface mesh resolution per axis",
    }
    _resolution_defaults = {"grid_resolution": _DEFAULT_GRID_RESOLUTION}

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters for reaction-diffusion surface."""
        return {
            "base_surface": _DEFAULT_BASE_SURFACE,
            "feed_rate": _DEFAULT_FEED_RATE,
            "kill_rate": _DEFAULT_KILL_RATE,
            "diffusion_u": _DEFAULT_DIFFUSION_U,
            "diffusion_v": _DEFAULT_DIFFUSION_V,
            "iterations": _DEFAULT_ITERATIONS,
            "displacement_scale": _DEFAULT_DISPLACEMENT_SCALE,
        }

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate a reaction-diffusion patterned surface mesh.

        Note: grid_resolution is a resolution kwarg, not a params entry.
        Pass it as a keyword argument: generate(grid_resolution=64).
        """
        merged = self.get_default_params()
        if params:
            merged.update(params)

        base_surface = str(merged["base_surface"])
        feed_rate = float(merged["feed_rate"])
        kill_rate = float(merged["kill_rate"])
        diffusion_u = float(merged["diffusion_u"])
        diffusion_v = float(merged["diffusion_v"])
        iterations = int(merged["iterations"])
        displacement_scale = float(merged["displacement_scale"])
        grid_resolution = int(
            resolution_kwargs.get("grid_resolution", _DEFAULT_GRID_RESOLUTION)
        )

        _validate_params(
            base_surface, feed_rate, kill_rate, diffusion_u, diffusion_v,
            iterations, displacement_scale, grid_resolution,
        )

        merged["grid_resolution"] = grid_resolution

        base_mesh = generate_base_mesh(base_surface, grid_resolution)
        num_verts = len(base_mesh.vertices)

        laplacian = build_mesh_laplacian(base_mesh.faces, num_verts)
        u_field, v_field = _init_vertex_concentrations(num_verts, seed)

        v_result = _run_gray_scott_mesh(
            u_field, v_field, laplacian,
            feed_rate, kill_rate, diffusion_u, diffusion_v, iterations,
        )

        result_mesh = _displace_mesh(base_mesh, v_result, displacement_scale)
        bbox = BoundingBox.from_points(result_mesh.vertices)

        logger.info(
            "Generated rd_surface: surface=%s, f=%.4f, k=%.4f, "
            "iters=%d, grid=%d, verts=%d",
            base_surface, feed_rate, kill_rate,
            iterations, grid_resolution, num_verts,
        )

        return MathObject(
            mesh=result_mesh,
            generator_name=self.resolved_name or self.name,
            category=self.category,
            parameters=merged,
            seed=seed,
            bounding_box=bbox,
        )

    def get_default_representation(self) -> RepresentationConfig:
        """Return SURFACE_SHELL as default representation."""
        return RepresentationConfig(type=RepresentationType.SURFACE_SHELL)
