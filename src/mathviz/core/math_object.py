"""Geometry container dataclasses for the MathViz pipeline."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class CoordSpace(str, Enum):
    """Coordinate space of a MathObject."""

    ABSTRACT = "abstract"
    PHYSICAL = "physical"


def _validate_float64_nx3(arr: np.ndarray, name: str) -> list[str]:
    """Validate that an array is (N, 3) float64 with no NaNs."""
    errors: list[str] = []
    if arr.ndim != 2 or arr.shape[1] != 3:
        errors.append(f"{name} shape {arr.shape}, expected (N, 3)")
    if arr.dtype != np.float64:
        errors.append(f"{name} dtype {arr.dtype}, expected float64")
    if np.any(np.isnan(arr)):
        errors.append(f"{name} contain NaN")
    return errors


@dataclass
class Mesh:
    """Triangle mesh geometry."""

    vertices: np.ndarray  # (N, 3) float64
    faces: np.ndarray  # (M, 3) int
    normals: Optional[np.ndarray] = None  # (M, 3) or (N, 3)

    def validate(self) -> list[str]:
        """Return list of validation errors."""
        errors = _validate_float64_nx3(self.vertices, "vertices")
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            errors.append(f"faces shape {self.faces.shape}, expected (M, 3)")
        if self.faces.dtype.kind not in ("i", "u"):
            errors.append(f"faces dtype {self.faces.dtype}, expected integer")
        if self.faces.size > 0 and self.faces.max() >= len(self.vertices):
            errors.append("face index out of bounds")
        if self.faces.size > 0 and self.faces.min() < 0:
            errors.append("face index is negative")
        if self.normals is not None:
            if self.normals.ndim != 2 or self.normals.shape[1] != 3:
                errors.append(f"normals shape {self.normals.shape}, expected (K, 3)")
        return errors


@dataclass
class PointCloud:
    """Point cloud geometry."""

    points: np.ndarray  # (N, 3) float64
    normals: Optional[np.ndarray] = None
    intensities: Optional[np.ndarray] = None  # per-point scalar

    def validate(self) -> list[str]:
        """Return list of validation errors."""
        errors = _validate_float64_nx3(self.points, "points")
        if self.normals is not None and len(self.normals) != len(self.points):
            errors.append(
                f"normals length {len(self.normals)} != points {len(self.points)}"
            )
        if self.intensities is not None and len(self.intensities) != len(self.points):
            errors.append(
                f"intensities length {len(self.intensities)} != points {len(self.points)}"
            )
        return errors


@dataclass
class Curve:
    """3D curve geometry."""

    points: np.ndarray  # (K, 3) float64
    closed: bool = False

    def validate(self) -> list[str]:
        """Return list of validation errors."""
        return _validate_float64_nx3(self.points, "points")


@dataclass
class BoundingBox:
    """Axis-aligned bounding box."""

    min_corner: tuple[float, float, float]
    max_corner: tuple[float, float, float]


@dataclass
class MathObject:
    """Universal geometry container.

    At least one of mesh, point_cloud, or curves must be populated.
    coord_space tracks whether geometry is in abstract or physical units.
    """

    # Geometry — at least one must be populated
    mesh: Optional[Mesh] = None
    point_cloud: Optional[PointCloud] = None
    curves: Optional[list[Curve]] = None

    # Metadata
    generator_name: str = ""
    category: str = ""
    parameters: dict = field(default_factory=dict)
    seed: int = 42
    coord_space: CoordSpace = CoordSpace.ABSTRACT
    bounding_box: Optional[BoundingBox] = None
    representation: Optional[str] = None

    # Optional
    scalar_field: Optional[np.ndarray] = None
    description: str = ""

    # Performance
    generation_time_seconds: Optional[float] = None

    def validate(self) -> list[str]:
        """Validate all geometry components. Returns list of errors."""
        errors: list[str] = []
        has_geometry = False
        if self.mesh is not None:
            has_geometry = True
            errors.extend(f"mesh: {e}" for e in self.mesh.validate())
        if self.point_cloud is not None:
            has_geometry = True
            errors.extend(f"point_cloud: {e}" for e in self.point_cloud.validate())
        if self.curves is not None and len(self.curves) > 0:
            has_geometry = True
            for i, c in enumerate(self.curves):
                errors.extend(f"curve[{i}]: {e}" for e in c.validate())
        if not has_geometry:
            errors.append(
                "MathObject has no geometry (mesh, point_cloud, and curves are all None)"
            )
        return errors

    def validate_or_raise(self) -> None:
        """Validate and raise ValueError if invalid."""
        errors = self.validate()
        if errors:
            raise ValueError(f"Invalid MathObject: {'; '.join(errors)}")
