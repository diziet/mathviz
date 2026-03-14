"""PointCloudExporter: exports MathObject point cloud geometry to PLY, XYZ, and PCD."""

import logging
from pathlib import Path

import numpy as np

from mathviz.core.math_object import MathObject
from mathviz.pipeline.metadata import resolve_format, validate_format, write_metadata

logger = logging.getLogger(__name__)

SUPPORTED_CLOUD_FORMATS = {"ply", "xyz", "pcd"}


class PointCloudExportError(ValueError):
    """Raised when a point cloud export fails due to missing geometry."""


def export_point_cloud(obj: MathObject, path: Path, fmt: str | None = None) -> Path:
    """Export MathObject point cloud to a file and write sidecar metadata.

    Args:
        obj: MathObject with point_cloud geometry.
        path: Output file path.
        fmt: Format override ("ply", "xyz", "pcd"). Inferred from suffix if None.

    Returns:
        The path written to.

    Raises:
        PointCloudExportError: If the MathObject has no point_cloud.
        ImportError: If PCD format is requested without open3d.
        ValueError: If the format is unsupported.
    """
    if obj.point_cloud is None:
        raise PointCloudExportError(
            f"Cannot export point cloud: MathObject '{obj.generator_name}' has no point_cloud "
            "geometry. Generate or convert to point cloud before exporting."
        )

    resolved_fmt = resolve_format(path, fmt)
    validate_format(resolved_fmt, SUPPORTED_CLOUD_FORMATS, "point cloud")

    path.parent.mkdir(parents=True, exist_ok=True)

    if resolved_fmt == "ply":
        _write_ply_cloud(path, obj.point_cloud.points)
    elif resolved_fmt == "xyz":
        _write_xyz(path, obj.point_cloud.points)
    elif resolved_fmt == "pcd":
        _write_pcd(path, obj.point_cloud.points)
    else:
        raise NotImplementedError(f"Writer not implemented for '{resolved_fmt}'")

    write_metadata(path, obj, export_format=resolved_fmt)
    logger.info("Exported point cloud to %s (format=%s)", path, resolved_fmt)
    return path


def _write_ply_cloud(path: Path, points: np.ndarray) -> None:
    """Write points to a PLY file in ASCII format (cloud mode, no faces)."""
    num_points = len(points)
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {num_points}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(header)
        np.savetxt(f, points, fmt="%.6f", delimiter=" ")


def _write_xyz(path: Path, points: np.ndarray) -> None:
    """Write points to an XYZ text file (one point per line)."""
    np.savetxt(str(path), points, fmt="%.6f", delimiter=" ")


def _write_pcd(path: Path, points: np.ndarray) -> None:
    """Write points to PCD format using open3d.

    Raises ImportError with install instructions if open3d is not available.
    """
    try:
        import open3d  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "PCD export requires open3d, which is not installed. "
            "Install it with: pip install mathviz[open3d]"
        ) from e

    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points)
    open3d.io.write_point_cloud(str(path), pcd)
