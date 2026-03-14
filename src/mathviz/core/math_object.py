"""Geometry container dataclasses for the MathViz pipeline."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class CoordSpace(str, Enum):
    """Coordinate space of a MathObject."""

    ABSTRACT = "abstract"
    PHYSICAL = "physical"


@dataclass
class Mesh:
    """Triangle mesh geometry."""

    vertices: np.ndarray  # (N, 3) float64
    faces: np.ndarray  # (M, 3) int
    normals: Optional[np.ndarray] = None  # (M, 3) or (N, 3)

    def validate(self) -> list[str]:
        """Return list of validation errors."""
        errors: list[str] = []
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 3:
            errors.append(f"vertices shape {self.vertices.shape}, expected (N, 3)")
        if self.vertices.dtype != np.float64:
            errors.append(f"vertices dtype {self.vertices.dtype}, expected float64")
        if np.any(np.isnan(self.vertices)):
            errors.append("vertices contain NaN")
        if self.faces.ndim != 2 or self.faces.shape[1] != 3:
            errors.append(f"faces shape {self.faces.shape}, expected (M, 3)")
        if self.faces.dtype.kind not in ("i", "u"):
            errors.append(f"faces dtype {self.faces.dtype}, expected integer")
        if self.faces.size > 0 and self.faces.max() >= len(self.vertices):
            errors.append("face index out of bounds")
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
        errors: list[str] = []
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            errors.append(f"points shape {self.points.shape}, expected (N, 3)")
        if self.points.dtype != np.float64:
            errors.append(f"points dtype {self.points.dtype}, expected float64")
        if np.any(np.isnan(self.points)):
            errors.append("points contain NaN")
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
        errors: list[str] = []
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            errors.append(f"points shape {self.points.shape}, expected (K, 3)")
        if self.points.dtype != np.float64:
            errors.append(f"points dtype {self.points.dtype}, expected float64")
        if np.any(np.isnan(self.points)):
            errors.append("points contain NaN")
        return errors


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
    bounding_box: BoundingBox = field(
        default_factory=lambda: BoundingBox((0, 0, 0), (0, 0, 0))
    )
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
        if self.curves is not None:
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
