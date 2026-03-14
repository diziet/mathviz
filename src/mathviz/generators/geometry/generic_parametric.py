"""Generic parametric surface from user-supplied string expressions.

Evaluates f(u, v) -> (x, y, z) from three string expressions using a
restricted namespace containing only numpy math functions. No exec,
no imports, no file I/O — safe against code injection.
"""

import logging
import re
from typing import Any

import numpy as np

from mathviz.core.generator import GeneratorBase, register
from mathviz.core.math_object import BoundingBox, MathObject, Mesh
from mathviz.core.representation import RepresentationConfig, RepresentationType
from mathviz.generators.parametric._mesh_utils import (
    build_open_grid_faces,
    build_wrapped_grid_faces,
)

logger = logging.getLogger(__name__)

_DEFAULT_GRID_RESOLUTION = 64
_MIN_GRID_RESOLUTION = 3
_DEFAULT_U_RANGE = (0.0, 2 * np.pi)
_DEFAULT_V_RANGE = (0.0, 2 * np.pi)

_FORBIDDEN_PATTERNS = re.compile(
    r"(__\w+__|import\s|exec\s*\(|eval\s*\(|open\s*\(|compile\s*\("
    r"|getattr\s*\(|setattr\s*\(|delattr\s*\(|globals\s*\(|locals\s*\("
    r"|vars\s*\(|dir\s*\(|type\s*\(|__builtins__|breakpoint\s*\("
    r"|os\.|sys\.|subprocess\.|shutil\.)",
)

_SAFE_NAMESPACE: dict[str, Any] = {
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "arcsin": np.arcsin,
    "arccos": np.arccos,
    "arctan": np.arctan,
    "arctan2": np.arctan2,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
    "exp": np.exp,
    "log": np.log,
    "log2": np.log2,
    "log10": np.log10,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "floor": np.floor,
    "ceil": np.ceil,
    "pi": np.pi,
    "e": np.e,
    "power": np.power,
    "sign": np.sign,
    "clip": np.clip,
    "minimum": np.minimum,
    "maximum": np.maximum,
}


def validate_expression(expr: str) -> None:
    """Validate a user expression for safety.

    Raises ValueError if the expression contains forbidden patterns.
    """
    if _FORBIDDEN_PATTERNS.search(expr):
        raise ValueError(f"Expression contains forbidden pattern: {expr!r}")


def evaluate_expression(
    expr: str, u: np.ndarray, v: np.ndarray,
) -> np.ndarray:
    """Safely evaluate a parametric expression with u and v arrays.

    Only numpy math functions are available in the evaluation namespace.
    """
    validate_expression(expr)

    namespace = dict(_SAFE_NAMESPACE)
    namespace["u"] = u
    namespace["v"] = v
    namespace["__builtins__"] = {}

    try:
        result = eval(expr, namespace)  # noqa: S307
    except NameError as exc:
        raise ValueError(f"Unknown name in expression {expr!r}: {exc}") from exc
    except SyntaxError as exc:
        raise ValueError(f"Syntax error in expression {expr!r}: {exc}") from exc

    return np.asarray(result, dtype=np.float64)


def _validate_params(
    grid_resolution: int, u_range: tuple[float, float], v_range: tuple[float, float],
) -> None:
    """Validate generic parametric parameters."""
    if grid_resolution < _MIN_GRID_RESOLUTION:
        raise ValueError(
            f"grid_resolution must be >= {_MIN_GRID_RESOLUTION}, "
            f"got {grid_resolution}"
        )
    if u_range[0] >= u_range[1]:
        raise ValueError(f"u_range start must be < end, got {u_range}")
    if v_range[0] >= v_range[1]:
        raise ValueError(f"v_range start must be < end, got {v_range}")


@register
class GenericParametricGenerator(GeneratorBase):
    """Generic parametric surface from user-supplied expressions."""

    name = "generic_parametric"
    category = "geometry"
    aliases = ()
    description = "Parametric surface from user-supplied f(u,v) expressions"
    resolution_params = {"grid_resolution": "Number of grid divisions per axis"}

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters for generic_parametric."""
        return {
            "x_expr": "(1 + 0.4 * cos(v)) * cos(u)",
            "y_expr": "(1 + 0.4 * cos(v)) * sin(u)",
            "z_expr": "0.4 * sin(v)",
            "u_range": list(_DEFAULT_U_RANGE),
            "v_range": list(_DEFAULT_V_RANGE),
            "wrap_u": True,
            "wrap_v": True,
        }

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate a parametric surface mesh from string expressions."""
        merged = self.get_default_params()
        if params:
            merged.update(params)

        x_expr = str(merged["x_expr"])
        y_expr = str(merged["y_expr"])
        z_expr = str(merged["z_expr"])
        u_range = tuple(float(v) for v in merged["u_range"])
        v_range = tuple(float(v) for v in merged["v_range"])
        wrap_u = bool(merged["wrap_u"])
        wrap_v = bool(merged["wrap_v"])
        grid_resolution = int(
            resolution_kwargs.get("grid_resolution", _DEFAULT_GRID_RESOLUTION)
        )

        _validate_params(grid_resolution, u_range, v_range)

        for expr in (x_expr, y_expr, z_expr):
            validate_expression(expr)

        mesh = _build_parametric_mesh(
            x_expr, y_expr, z_expr,
            u_range, v_range,
            wrap_u, wrap_v,
            grid_resolution,
        )

        bbox = BoundingBox.from_points(mesh.vertices)

        logger.info(
            "Generated generic_parametric: grid=%d, vertices=%d, faces=%d",
            grid_resolution, len(mesh.vertices), len(mesh.faces),
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
        """Return surface shell as default representation."""
        return RepresentationConfig(type=RepresentationType.SURFACE_SHELL)


def _build_parametric_mesh(
    x_expr: str,
    y_expr: str,
    z_expr: str,
    u_range: tuple[float, float],
    v_range: tuple[float, float],
    wrap_u: bool,
    wrap_v: bool,
    grid_resolution: int,
) -> Mesh:
    """Build a triangle mesh from parametric expressions."""
    n = grid_resolution
    endpoint_u = not wrap_u
    endpoint_v = not wrap_v

    u_vals = np.linspace(u_range[0], u_range[1], n, endpoint=endpoint_u)
    v_vals = np.linspace(v_range[0], v_range[1], n, endpoint=endpoint_v)
    uu, vv = np.meshgrid(u_vals, v_vals, indexing="ij")

    x = evaluate_expression(x_expr, uu, vv)
    y = evaluate_expression(y_expr, uu, vv)
    z = evaluate_expression(z_expr, uu, vv)

    vertices = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    vertices = vertices.astype(np.float64)

    if wrap_u and wrap_v:
        faces = build_wrapped_grid_faces(n, n)
    else:
        faces = build_open_grid_faces(n, n)

    return Mesh(vertices=vertices, faces=faces)
