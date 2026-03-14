"""Two-tier validation: mesh checks and engraving checks."""

import logging
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import trimesh
from scipy.spatial import cKDTree

from mathviz.core.container import Container
from mathviz.core.engraving import EngravingProfile
from mathviz.core.math_object import Mesh, PointCloud

logger = logging.getLogger(__name__)


class Severity(str, Enum):
    """Severity level for a validation check."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class CheckResult:
    """Result of a single validation check."""

    name: str
    passed: bool
    severity: Severity
    message: str


@dataclass
class ValidationResult:
    """Aggregated result of all validation checks."""

    checks: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True only when there are zero errors (warnings allowed)."""
        return all(
            check.passed or check.severity != Severity.ERROR for check in self.checks
        )

    @property
    def errors(self) -> list[CheckResult]:
        """Return all failed error-severity checks."""
        return [c for c in self.checks if not c.passed and c.severity == Severity.ERROR]

    @property
    def warnings(self) -> list[CheckResult]:
        """Return all failed warning-severity checks."""
        return [c for c in self.checks if not c.passed and c.severity == Severity.WARNING]


def validate_mesh(mesh: Mesh, container: Container | None = None) -> ValidationResult:
    """Run mesh validation checks and return structured results."""
    result = ValidationResult()
    tm = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)

    _check_watertight(tm, result)
    _check_manifold(tm, result)
    _check_degenerate_faces(tm, result)
    _check_normals(mesh, tm, result)
    if container is not None:
        _check_bounding_box(mesh, container, result)

    logger.info("Mesh validation complete: %d checks, passed=%s", len(result.checks), result.passed)
    return result


def validate_engraving(
    cloud: PointCloud,
    profile: EngravingProfile,
    container: Container | None = None,
) -> ValidationResult:
    """Run engraving validation checks on a point cloud."""
    result = ValidationResult()

    _check_point_budget(cloud, profile, result)
    nearest = _compute_nearest_distances(cloud.points) if len(cloud.points) >= 2 else None
    _check_min_spacing(profile, result, nearest)
    _check_max_gap(profile, result, nearest)
    _check_opacity(cloud, result)
    if container is not None:
        _check_points_in_container(cloud, container, result)

    logger.info(
        "Engraving validation complete: %d checks, passed=%s",
        len(result.checks),
        result.passed,
    )
    return result


def _compute_nearest_distances(points: np.ndarray) -> np.ndarray:
    """Return nearest-neighbor distance for each point."""
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=2)
    return distances[:, 1]


# --- Mesh check helpers ---


def _check_watertight(tm: trimesh.Trimesh, result: ValidationResult) -> None:
    """Check if the mesh is watertight."""
    if tm.is_watertight:
        result.checks.append(CheckResult("watertight", True, Severity.INFO, "Mesh is watertight"))
    else:
        result.checks.append(
            CheckResult("watertight", False, Severity.WARNING, "Mesh is not watertight")
        )


def _check_manifold(tm: trimesh.Trimesh, result: ValidationResult) -> None:
    """Check for boundary edges (non-manifold / open surface)."""
    boundary_edges = _count_boundary_edges(tm)
    if boundary_edges == 0:
        result.checks.append(
            CheckResult("manifold", True, Severity.INFO, "No boundary edges detected")
        )
    else:
        result.checks.append(
            CheckResult(
                "manifold",
                False,
                Severity.WARNING,
                f"Mesh has {boundary_edges} boundary edges (open surface)",
            )
        )


def _count_boundary_edges(tm: trimesh.Trimesh) -> int:
    """Count edges that belong to only one face."""
    if len(tm.faces) == 0:
        return 0
    edges = tm.edges_sorted
    unique_edges, counts = np.unique(edges, axis=0, return_counts=True)
    return int(np.sum(counts == 1))


def _check_degenerate_faces(tm: trimesh.Trimesh, result: ValidationResult) -> None:
    """Check for zero-area (degenerate) faces."""
    if len(tm.faces) == 0:
        result.checks.append(
            CheckResult("degenerate_faces", True, Severity.INFO, "No faces to check")
        )
        return
    areas = tm.area_faces
    degenerate_count = int(np.sum(areas < 1e-12))
    if degenerate_count == 0:
        result.checks.append(
            CheckResult("degenerate_faces", True, Severity.INFO, "No degenerate faces")
        )
    else:
        result.checks.append(
            CheckResult(
                "degenerate_faces",
                False,
                Severity.WARNING,
                f"{degenerate_count} degenerate (zero-area) faces detected",
            )
        )


def _check_normals(mesh: Mesh, tm: trimesh.Trimesh, result: ValidationResult) -> None:
    """Check for NaN values in supplied normals."""
    if mesh.normals is None:
        msg = "No normals to check" if len(tm.faces) == 0 else "No normals provided (optional)"
        result.checks.append(CheckResult("normals", True, Severity.INFO, msg))
        return
    if bool(np.any(np.isnan(mesh.normals))):
        result.checks.append(
            CheckResult("normals", False, Severity.ERROR, "Normals contain NaN values")
        )
        return
    result.checks.append(CheckResult("normals", True, Severity.INFO, "Normals OK"))


def _check_bounding_box(mesh: Mesh, container: Container, result: ValidationResult) -> None:
    """Check that mesh fits within the container's usable volume."""
    _check_geometry_in_container(mesh.vertices, container, "Mesh", result)


# --- Engraving check helpers ---


def _check_point_budget(
    cloud: PointCloud, profile: EngravingProfile, result: ValidationResult
) -> None:
    """Check point count against budget."""
    count = len(cloud.points)
    if count <= profile.point_budget:
        result.checks.append(
            CheckResult(
                "point_budget",
                True,
                Severity.INFO,
                f"Point count {count} within budget {profile.point_budget}",
            )
        )
    else:
        result.checks.append(
            CheckResult(
                "point_budget",
                False,
                Severity.ERROR,
                f"Point count {count} exceeds budget {profile.point_budget}",
            )
        )


def _check_min_spacing(
    profile: EngravingProfile,
    result: ValidationResult,
    nearest: np.ndarray | None,
) -> None:
    """Check minimum spacing between points using precomputed nearest distances."""
    if nearest is None:
        result.checks.append(
            CheckResult("min_spacing", True, Severity.INFO, "Too few points to check spacing")
        )
        return
    min_dist = float(np.min(nearest))
    if min_dist >= profile.min_point_spacing_mm:
        result.checks.append(
            CheckResult(
                "min_spacing",
                True,
                Severity.INFO,
                f"Min spacing {min_dist:.4f} mm >= {profile.min_point_spacing_mm} mm",
            )
        )
    else:
        result.checks.append(
            CheckResult(
                "min_spacing",
                False,
                Severity.WARNING,
                f"Min spacing {min_dist:.4f} mm < {profile.min_point_spacing_mm} mm",
            )
        )


def _check_max_gap(
    profile: EngravingProfile,
    result: ValidationResult,
    nearest: np.ndarray | None,
) -> None:
    """Check for gaps exceeding max spacing."""
    if nearest is None:
        result.checks.append(
            CheckResult("max_gap", True, Severity.INFO, "Too few points to check gap")
        )
        return
    max_nearest = float(np.max(nearest))
    if max_nearest <= profile.max_point_spacing_mm:
        result.checks.append(
            CheckResult(
                "max_gap",
                True,
                Severity.INFO,
                f"Max gap {max_nearest:.4f} mm <= {profile.max_point_spacing_mm} mm",
            )
        )
    else:
        result.checks.append(
            CheckResult(
                "max_gap",
                False,
                Severity.WARNING,
                f"Max gap {max_nearest:.4f} mm > {profile.max_point_spacing_mm} mm",
            )
        )


def _check_opacity(cloud: PointCloud, result: ValidationResult) -> None:
    """Warn if >70% of voxels are occupied in any axis projection."""
    if len(cloud.points) < 2:
        result.checks.append(
            CheckResult("opacity", True, Severity.INFO, "Too few points to check opacity")
        )
        return

    threshold = 0.70
    axis_names = ["X (YZ projection)", "Y (XZ projection)", "Z (XY projection)"]
    # Project along each axis: for axis i, project onto the other two axes
    for axis_idx, axis_name in enumerate(axis_names):
        other_axes = [j for j in range(3) if j != axis_idx]
        projected = cloud.points[:, other_axes]
        density = _compute_projection_density(projected)
        if density > threshold:
            result.checks.append(
                CheckResult(
                    "opacity",
                    False,
                    Severity.WARNING,
                    f"Opacity warning: {density:.1%} voxel occupancy in {axis_name} "
                    f"exceeds {threshold:.0%} threshold",
                )
            )
            return

    result.checks.append(
        CheckResult("opacity", True, Severity.INFO, "Opacity within acceptable limits")
    )


def _compute_projection_density(points_2d: np.ndarray) -> float:
    """Compute fraction of occupied cells in a 2D grid of projected points."""
    mins = points_2d.min(axis=0)
    maxs = points_2d.max(axis=0)
    ranges = maxs - mins
    if np.any(ranges < 1e-12):
        return 0.0

    # Estimate grid resolution from unique projected point count
    unique_count = len(np.unique(points_2d, axis=0))
    grid_size = min(64, max(4, int(np.sqrt(unique_count))))
    cell_size = ranges / grid_size
    indices = ((points_2d - mins) / cell_size).astype(int)
    indices = np.clip(indices, 0, grid_size - 1)
    occupied = len(set(map(tuple, indices)))
    total_cells = grid_size * grid_size
    return occupied / total_cells


def _check_points_in_container(
    cloud: PointCloud, container: Container, result: ValidationResult
) -> None:
    """Check that all points fit within the container's usable volume."""
    _check_geometry_in_container(cloud.points, container, "Point cloud", result)


def _check_geometry_in_container(
    points: np.ndarray,
    container: Container,
    label: str,
    result: ValidationResult,
) -> None:
    """Check that all points fit within the container's usable volume."""
    if len(points) == 0:
        result.checks.append(
            CheckResult("container_bounds", True, Severity.INFO, f"{label} has no points to check")
        )
        return

    usable = container.usable_volume
    half_extents = np.array([usable[0] / 2, usable[1] / 2, usable[2] / 2])

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    outside = bool(np.any(mins < -half_extents) or np.any(maxs > half_extents))

    if outside:
        result.checks.append(
            CheckResult(
                "container_bounds",
                False,
                Severity.ERROR,
                f"{label} extends outside container usable volume "
                f"(extents: [{mins[0]:.2f}..{maxs[0]:.2f}], "
                f"[{mins[1]:.2f}..{maxs[1]:.2f}], "
                f"[{mins[2]:.2f}..{maxs[2]:.2f}]; "
                f"half-extents: {half_extents[0]:.2f}, {half_extents[1]:.2f}, "
                f"{half_extents[2]:.2f})",
            )
        )
    else:
        result.checks.append(
            CheckResult(
                "container_bounds",
                True,
                Severity.INFO,
                f"{label} fits within container usable volume",
            )
        )
