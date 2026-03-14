"""MeshExporter: exports MathObject mesh geometry to STL, OBJ, and PLY."""

import logging
from pathlib import Path

import trimesh

from mathviz.core.math_object import MathObject
from mathviz.pipeline.metadata import resolve_format, validate_format, write_metadata

logger = logging.getLogger(__name__)

SUPPORTED_MESH_FORMATS = {"stl", "obj", "ply"}


class MeshExportError(ValueError):
    """Raised when a mesh export fails due to missing geometry."""


def export_mesh(obj: MathObject, path: Path, fmt: str | None = None) -> Path:
    """Export MathObject mesh to a file and write sidecar metadata.

    Args:
        obj: MathObject with mesh geometry.
        path: Output file path.
        fmt: Format override ("stl", "obj", "ply"). Inferred from suffix if None.

    Returns:
        The path written to.

    Raises:
        MeshExportError: If the MathObject has no mesh.
        ValueError: If the format is unsupported.
    """
    if obj.mesh is None:
        raise MeshExportError(
            f"Cannot export mesh: MathObject '{obj.generator_name}' has no mesh geometry. "
            "Generate or convert to mesh before exporting."
        )

    resolved_fmt = resolve_format(path, fmt)
    validate_format(resolved_fmt, SUPPORTED_MESH_FORMATS, "mesh")

    tri_mesh = trimesh.Trimesh(
        vertices=obj.mesh.vertices,
        faces=obj.mesh.faces,
        process=False,
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    tri_mesh.export(str(path), file_type=resolved_fmt)

    write_metadata(path, obj, export_format=resolved_fmt)
    logger.info("Exported mesh to %s (format=%s)", path, resolved_fmt)
    return path
