"""Building extrusion generator from GeoJSON files.

Parses GeoJSON polygons and extrudes them vertically to create 3D building
meshes. Each polygon feature is extruded to a height specified by a feature
property or a default height parameter.
"""

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from mathviz.core.generator import GeneratorBase, register
from mathviz.core.math_object import BoundingBox, MathObject, Mesh
from mathviz.core.representation import RepresentationConfig, RepresentationType
from mathviz.generators.data_driven._file_utils import validate_input_file

logger = logging.getLogger(__name__)

_DEFAULT_HEIGHT = 1.0
_DEFAULT_HEIGHT_PROPERTY = "height"
_SUPPORTED_EXTENSIONS = {".geojson", ".json"}


def _load_geojson(path: Path) -> dict:
    """Load and validate a GeoJSON file."""
    with open(path) as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("GeoJSON root must be an object")
    geojson_type = data.get("type", "")
    if geojson_type not in ("FeatureCollection", "Feature", "Polygon"):
        raise ValueError(
            f"Unsupported GeoJSON type '{geojson_type}'. "
            "Expected FeatureCollection, Feature, or Polygon"
        )
    return data


def _extract_polygons(data: dict, height_property: str, default_height: float) -> list:
    """Extract (coordinates, height) tuples from GeoJSON data."""
    polygons: list[tuple[list, float]] = []
    geojson_type = data.get("type", "")

    if geojson_type == "FeatureCollection":
        for feature in data.get("features", []):
            polygons.extend(
                _extract_from_feature(feature, height_property, default_height)
            )
    elif geojson_type == "Feature":
        polygons.extend(
            _extract_from_feature(data, height_property, default_height)
        )
    elif geojson_type == "Polygon":
        coordinates = data.get("coordinates")
        if not coordinates:
            raise ValueError("Polygon is missing 'coordinates' key")
        polygons.append((coordinates[0], default_height))

    if not polygons:
        raise ValueError("No polygons found in GeoJSON file")
    return polygons


def _extract_from_feature(
    feature: dict, height_property: str, default_height: float,
) -> list[tuple[list, float]]:
    """Extract polygons from a single GeoJSON Feature."""
    geometry = feature.get("geometry", {})
    properties = feature.get("properties", {}) or {}
    raw_height = properties.get(height_property, default_height)
    height = float(raw_height)
    if height <= 0:
        raise ValueError(
            f"Feature height must be positive, got {height} "
            f"(from property '{height_property}')"
        )
    geo_type = geometry.get("type", "")

    results: list[tuple[list, float]] = []
    if geo_type == "Polygon":
        coords = geometry.get("coordinates", [[]])
        results.append((coords[0], height))
    elif geo_type == "MultiPolygon":
        for polygon_coords in geometry.get("coordinates", []):
            results.append((polygon_coords[0], height))
    return results


def _extrude_polygon(
    ring: list, height: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Extrude a 2D polygon ring vertically to create a 3D mesh."""
    coords = np.array(ring, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError(f"Invalid polygon coordinates shape: {coords.shape}")

    # Use only x, y; drop z if present
    xy = coords[:, :2]

    # Remove closing vertex if it duplicates the first
    if np.allclose(xy[0], xy[-1]) and len(xy) > 1:
        xy = xy[:-1]

    num_points = len(xy)
    if num_points < 3:
        raise ValueError(f"Polygon must have >= 3 vertices, got {num_points}")

    # Build vertices: bottom ring (z=0) + top ring (z=height)
    bottom = np.column_stack([xy, np.zeros(num_points)])
    top = np.column_stack([xy, np.full(num_points, height)])
    vertices = np.vstack([bottom, top])

    faces_list: list[list[int]] = []

    # Side faces
    for i in range(num_points):
        next_i = (i + 1) % num_points
        b0, b1 = i, next_i
        t0, t1 = i + num_points, next_i + num_points
        faces_list.append([b0, b1, t1])
        faces_list.append([b0, t1, t0])

    # Top and bottom caps via fan triangulation
    for i in range(1, num_points - 1):
        faces_list.append([0, i + 1, i])  # bottom (reversed winding)
        faces_list.append([num_points, num_points + i, num_points + i + 1])  # top

    faces = np.array(faces_list, dtype=np.intp)
    return vertices, faces


def _merge_meshes(
    mesh_parts: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Merge multiple (vertices, faces) pairs into a single mesh."""
    all_vertices: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    vertex_offset = 0

    for vertices, faces in mesh_parts:
        all_vertices.append(vertices)
        all_faces.append(faces + vertex_offset)
        vertex_offset += len(vertices)

    return np.vstack(all_vertices), np.vstack(all_faces)


@register
class BuildingExtrudeGenerator(GeneratorBase):
    """Extrude GeoJSON polygons into 3D building meshes.

    Reads polygon geometries from a GeoJSON file and extrudes each one
    vertically. Height can be specified per-feature via a property field
    or as a global default. The seed parameter is accepted for interface
    conformance but unused — output is fully determined by the input file.
    """

    name = "building_extrude"
    category = "data_driven"
    aliases = ()
    description = "Extrude GeoJSON polygons into 3D building meshes"
    resolution_params = {}

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters for building extrusion."""
        return {
            "input_file": "",
            "default_height": _DEFAULT_HEIGHT,
            "height_property": _DEFAULT_HEIGHT_PROPERTY,
        }

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate extruded building meshes from a GeoJSON file.

        The seed parameter is accepted for interface conformance but does
        not affect output — the result is fully determined by the input file.
        """
        merged = self.get_default_params()
        if params:
            merged.update(params)

        input_file = str(merged["input_file"])
        default_height = float(merged["default_height"])
        height_property = str(merged["height_property"])

        if not input_file:
            raise ValueError("input_file parameter is required")
        if default_height <= 0:
            raise ValueError(
                f"default_height must be positive, got {default_height}"
            )

        path = validate_input_file(input_file, _SUPPORTED_EXTENSIONS)
        data = _load_geojson(path)
        polygons = _extract_polygons(data, height_property, default_height)

        mesh_parts: list[tuple[np.ndarray, np.ndarray]] = []
        for ring, height in polygons:
            vertices, faces = _extrude_polygon(ring, height)
            mesh_parts.append((vertices, faces))

        all_vertices, all_faces = _merge_meshes(mesh_parts)
        bbox = BoundingBox.from_points(all_vertices)

        logger.info(
            "Generated building_extrude: file=%s, polygons=%d, "
            "vertices=%d, faces=%d",
            path.name, len(polygons), len(all_vertices), len(all_faces),
        )

        return MathObject(
            mesh=Mesh(vertices=all_vertices, faces=all_faces),
            generator_name=self.resolved_name or self.name,
            category=self.category,
            parameters=merged,
            seed=seed,
            bounding_box=bbox,
        )

    def get_default_representation(self) -> RepresentationConfig:
        """Return SURFACE_SHELL as default representation."""
        return RepresentationConfig(type=RepresentationType.SURFACE_SHELL)
