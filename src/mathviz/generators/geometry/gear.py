"""Involute gear tooth profile generator.

Generates mechanical gear tooth profiles using standard involute geometry
(base circle, involute curve, tip circle, root circle, fillet) extruded
into a 3D spur or helical gear solid.
"""

import logging
from typing import Any

import numpy as np

from mathviz.core.generator import GeneratorBase, register
from mathviz.core.math_object import BoundingBox, Mesh, MathObject
from mathviz.core.representation import RepresentationConfig, RepresentationType

logger = logging.getLogger(__name__)

_DEFAULT_NUM_TEETH = 20
_DEFAULT_MODULE = 1.0
_DEFAULT_PRESSURE_ANGLE = 20.0
_DEFAULT_FACE_WIDTH = 0.5
_DEFAULT_HELIX_ANGLE = 0.0
_DEFAULT_CURVE_POINTS = 32
_MIN_TEETH = 6
_MAX_TEETH = 200
_ADDENDUM_FACTOR = 1.0
_DEDENDUM_FACTOR = 1.25


def _validate_params(
    num_teeth: int,
    module: float,
    pressure_angle: float,
    face_width: float,
    curve_points: int,
) -> None:
    """Validate gear generator parameters."""
    if num_teeth < _MIN_TEETH:
        raise ValueError(f"num_teeth must be >= {_MIN_TEETH}, got {num_teeth}")
    if num_teeth > _MAX_TEETH:
        raise ValueError(f"num_teeth must be <= {_MAX_TEETH}, got {num_teeth}")
    if module <= 0:
        raise ValueError(f"module must be > 0, got {module}")
    if pressure_angle <= 0 or pressure_angle >= 45:
        raise ValueError(
            f"pressure_angle must be in (0, 45) degrees, got {pressure_angle}"
        )
    if face_width <= 0:
        raise ValueError(f"face_width must be > 0, got {face_width}")
    if curve_points < 8:
        raise ValueError(f"curve_points must be >= 8, got {curve_points}")


def _involute_offset(base_radius: float, radius: float) -> float:
    """Compute involute angular offset at a given radius."""
    alpha = np.arccos(np.clip(base_radius / radius, -1.0, 1.0))
    return float(np.tan(alpha) - alpha)


def _build_gear_profile(
    num_teeth: int,
    module: float,
    pressure_angle_rad: float,
    curve_points: int,
) -> np.ndarray:
    """Build closed 2D gear profile as (P, 2) array of xy points."""
    pitch_radius = module * num_teeth / 2.0
    base_radius = pitch_radius * np.cos(pressure_angle_rad)
    tip_radius = pitch_radius + _ADDENDUM_FACTOR * module
    root_radius = max(pitch_radius - _DEDENDUM_FACTOR * module, 0.1 * module)

    inv_pa = float(np.tan(pressure_angle_rad) - pressure_angle_rad)
    half_tooth_angle = np.pi / (2 * num_teeth) + inv_pa
    tooth_pitch = 2 * np.pi / num_teeth

    n_inv = max(curve_points // 4, 3)
    n_tip = max(curve_points // 8, 2)
    n_root = max(curve_points // 8, 2)
    has_fillet = root_radius < base_radius
    n_fillet = max(curve_points // 8, 2) if has_fillet else 0

    all_points: list[list[float]] = []

    for i in range(num_teeth):
        center = i * tooth_pitch

        # Negative fillet: root → base (radial line)
        if has_fillet:
            for r in np.linspace(root_radius, base_radius, n_fillet, endpoint=False):
                theta = center - half_tooth_angle
                all_points.append([r * np.cos(theta), r * np.sin(theta)])

        # Negative involute: base → tip
        for r in np.linspace(base_radius, tip_radius, n_inv):
            inv_a = _involute_offset(base_radius, float(r))
            theta = center - half_tooth_angle + inv_a
            all_points.append([float(r) * np.cos(theta), float(r) * np.sin(theta)])

        # Tip arc
        tip_inv = _involute_offset(base_radius, tip_radius)
        theta_neg_tip = center - half_tooth_angle + tip_inv
        theta_pos_tip = center + half_tooth_angle - tip_inv
        for theta in np.linspace(theta_neg_tip, theta_pos_tip, n_tip + 2)[1:-1]:
            all_points.append(
                [tip_radius * np.cos(theta), tip_radius * np.sin(theta)]
            )

        # Positive involute: tip → base
        for r in np.linspace(tip_radius, base_radius, n_inv):
            inv_a = _involute_offset(base_radius, float(r))
            theta = center + half_tooth_angle - inv_a
            all_points.append([float(r) * np.cos(theta), float(r) * np.sin(theta)])

        # Positive fillet: base → root
        if has_fillet:
            for r in np.linspace(base_radius, root_radius, n_fillet, endpoint=False):
                theta = center + half_tooth_angle
                all_points.append([r * np.cos(theta), r * np.sin(theta)])

        # Root arc to next tooth
        gap_start = center + half_tooth_angle
        gap_end = center + tooth_pitch - half_tooth_angle
        for theta in np.linspace(gap_start, gap_end, n_root + 2)[1:-1]:
            all_points.append(
                [root_radius * np.cos(theta), root_radius * np.sin(theta)]
            )

    return np.array(all_points, dtype=np.float64)


def _extrude_to_mesh(
    profile: np.ndarray,
    face_width: float,
    helix_angle_rad: float,
    pitch_radius: float,
) -> Mesh:
    """Extrude a 2D profile into a 3D mesh along z-axis."""
    num_profile = len(profile)

    # Determine twist and number of layers
    if abs(helix_angle_rad) < 1e-10:
        total_twist = 0.0
        num_layers = 2
    else:
        total_twist = float(np.tan(helix_angle_rad) * face_width / pitch_radius)
        num_layers = max(2, int(np.ceil(abs(np.degrees(total_twist)) / 5.0)) + 1)

    # Build vertices: profile points at each layer + 2 center vertices
    num_verts = num_profile * num_layers + 2
    vertices = np.zeros((num_verts, 3), dtype=np.float64)

    for layer_idx in range(num_layers):
        t = layer_idx / max(num_layers - 1, 1)
        z = t * face_width
        twist = t * total_twist

        cos_tw = np.cos(twist)
        sin_tw = np.sin(twist)
        rotated_x = profile[:, 0] * cos_tw - profile[:, 1] * sin_tw
        rotated_y = profile[:, 0] * sin_tw + profile[:, 1] * cos_tw

        offset = layer_idx * num_profile
        vertices[offset:offset + num_profile, 0] = rotated_x
        vertices[offset:offset + num_profile, 1] = rotated_y
        vertices[offset:offset + num_profile, 2] = z

    # Center vertices for caps
    bottom_center_idx = num_profile * num_layers
    top_center_idx = bottom_center_idx + 1
    vertices[bottom_center_idx] = [0.0, 0.0, 0.0]
    vertices[top_center_idx] = [0.0, 0.0, face_width]

    faces: list[list[int]] = []

    # Side faces: connect adjacent layers
    for layer_idx in range(num_layers - 1):
        off_curr = layer_idx * num_profile
        off_next = (layer_idx + 1) * num_profile
        for j in range(num_profile):
            j_next = (j + 1) % num_profile
            faces.append([off_curr + j, off_next + j, off_next + j_next])
            faces.append([off_curr + j, off_next + j_next, off_curr + j_next])

    # Bottom cap (z=0): fan from center, winding order for outward normal
    for j in range(num_profile):
        j_next = (j + 1) % num_profile
        faces.append([bottom_center_idx, j_next, j])

    # Top cap (z=face_width): fan from center
    top_off = (num_layers - 1) * num_profile
    for j in range(num_profile):
        j_next = (j + 1) % num_profile
        faces.append([top_center_idx, top_off + j, top_off + j_next])

    faces_arr = np.array(faces, dtype=np.int64)
    return Mesh(vertices=vertices, faces=faces_arr)


@register
class GearGenerator(GeneratorBase):
    """Involute gear tooth profile generator."""

    name = "gear"
    category = "geometry"
    description = (
        "Involute gear: mechanical gear tooth profiles extruded "
        "as spur or helical gear solids"
    )
    resolution_params: dict[str, str] = {}

    def get_default_params(self) -> dict[str, Any]:
        """Return default gear parameters."""
        return {
            "num_teeth": _DEFAULT_NUM_TEETH,
            "module": _DEFAULT_MODULE,
            "pressure_angle": _DEFAULT_PRESSURE_ANGLE,
            "face_width": _DEFAULT_FACE_WIDTH,
            "helix_angle": _DEFAULT_HELIX_ANGLE,
            "curve_points": _DEFAULT_CURVE_POINTS,
        }

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate involute gear geometry."""
        merged = self.get_default_params()
        if params:
            merged.update(params)

        num_teeth = int(merged["num_teeth"])
        module = float(merged["module"])
        pressure_angle = float(merged["pressure_angle"])
        face_width = float(merged["face_width"])
        helix_angle = float(merged["helix_angle"])
        curve_points = int(merged["curve_points"])

        _validate_params(num_teeth, module, pressure_angle, face_width, curve_points)

        pressure_angle_rad = np.radians(pressure_angle)
        helix_angle_rad = np.radians(helix_angle)
        pitch_radius = module * num_teeth / 2.0

        profile = _build_gear_profile(
            num_teeth, module, pressure_angle_rad, curve_points
        )
        mesh = _extrude_to_mesh(profile, face_width, helix_angle_rad, pitch_radius)

        logger.info(
            "Generated gear: teeth=%d, module=%.2f, verts=%d, faces=%d",
            num_teeth, module, len(mesh.vertices), len(mesh.faces),
        )
        return MathObject(
            mesh=mesh,
            generator_name=self.name,
            category=self.category,
            parameters=merged,
            seed=seed,
            bounding_box=BoundingBox.from_points(mesh.vertices),
        )

    def get_default_representation(self) -> RepresentationConfig:
        """Return SURFACE_SHELL as default representation."""
        return RepresentationConfig(type=RepresentationType.SURFACE_SHELL)
