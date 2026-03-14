"""LOD decimation for preview serving."""

import io
import logging

import numpy as np
import trimesh

from mathviz.core.math_object import Mesh, PointCloud

logger = logging.getLogger(__name__)

PREVIEW_MAX_FACES = 100_000
PREVIEW_MAX_POINTS = 200_000


def decimate_mesh(mesh: Mesh, max_faces: int = PREVIEW_MAX_FACES) -> Mesh:
    """Decimate a mesh to at most max_faces triangles."""
    face_count = len(mesh.faces)
    if face_count <= max_faces:
        return mesh

    import fast_simplification

    vertices_out, faces_out = fast_simplification.simplify(
        mesh.vertices,
        mesh.faces,
        target_count=max_faces,
    )
    logger.info("Decimated mesh from %d to %d faces", face_count, len(faces_out))
    return Mesh(
        vertices=np.asarray(vertices_out, dtype=np.float64),
        faces=np.asarray(faces_out, dtype=np.int64),
    )


def subsample_cloud(cloud: PointCloud, max_points: int = PREVIEW_MAX_POINTS) -> PointCloud:
    """Subsample a point cloud to at most max_points points."""
    point_count = len(cloud.points)
    if point_count <= max_points:
        return cloud

    rng = np.random.default_rng(0)
    indices = rng.choice(point_count, size=max_points, replace=False)
    indices.sort()

    normals = cloud.normals[indices] if cloud.normals is not None else None
    intensities = cloud.intensities[indices] if cloud.intensities is not None else None

    logger.info("Subsampled cloud from %d to %d points", point_count, max_points)
    return PointCloud(
        points=cloud.points[indices],
        normals=normals,
        intensities=intensities,
    )


def mesh_to_glb(mesh: Mesh) -> bytes:
    """Serialize a Mesh to GLB binary format."""
    tri = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
    with io.BytesIO() as buf:
        tri.export(buf, file_type="glb")
        return buf.getvalue()


def cloud_to_binary_ply(cloud: PointCloud) -> bytes:
    """Serialize a PointCloud to binary PLY format."""
    points = cloud.points
    num_points = len(points)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {num_points}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    )
    header_bytes = header.encode("ascii")
    point_data = points.astype(np.float32).tobytes()
    return header_bytes + point_data
