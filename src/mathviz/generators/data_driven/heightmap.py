"""Heightmap generator from image files.

Reads a PNG/JPEG image (via Pillow) and converts pixel luminance to a 2D
scalar field for HEIGHTMAP_RELIEF representation. Optionally supports GeoTIFF
via rasterio if installed. When no input file is provided, generates a
built-in demo heightmap procedurally.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np

from mathviz.core.generator import GeneratorBase, register
from mathviz.core.math_object import BoundingBox, MathObject
from mathviz.core.representation import RepresentationConfig, RepresentationType
from mathviz.generators.data_driven._file_utils import validate_input_file

logger = logging.getLogger(__name__)

_DEFAULT_HEIGHT_SCALE = 1.0
_DEFAULT_DOWNSAMPLE = 1
_SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
_SUPPORTED_GEOTIFF_EXTENSIONS = {".tif", ".tiff"}
_MAX_PIXEL_DIMENSION = 4096
_MIN_PIXEL_DIMENSION = 2


_DEMO_RESOLUTION = 64


def _synthesize_demo_heightmap(seed: int) -> np.ndarray:
    """Generate a procedural demo heightmap with concentric rings and peaks."""
    rng = np.random.default_rng(seed)
    size = _DEMO_RESOLUTION
    y, x = np.mgrid[0:size, 0:size].astype(np.float64)

    # Normalize to [-1, 1]
    cx, cy = size / 2.0, size / 2.0
    nx = (x - cx) / cx
    ny = (y - cy) / cy
    r = np.sqrt(nx**2 + ny**2)

    # Concentric rings pattern
    rings = 0.5 * (1.0 + np.cos(r * 6.0 * np.pi))

    # Radial falloff
    falloff = np.exp(-2.0 * r**2)

    # Combine with slight noise for texture
    noise = rng.normal(0, 0.05, (size, size))
    field = (rings * falloff + noise).clip(0, None)

    return field.astype(np.float64)


def _try_load_geotiff(path: Path) -> np.ndarray | None:
    """Attempt to load a GeoTIFF via rasterio. Returns None if unavailable."""
    try:
        import rasterio
    except ImportError:
        return None

    suffix = path.suffix.lower()
    if suffix not in _SUPPORTED_GEOTIFF_EXTENSIONS:
        return None

    try:
        with rasterio.open(path) as dataset:
            band = dataset.read(1).astype(np.float64)
            return band
    except rasterio.errors.RasterioIOError:
        logger.warning(
            "rasterio failed to open %s, falling back to PIL", path,
            exc_info=True,
        )
        return None


def _load_image_as_array(path: Path) -> np.ndarray:
    """Load an image file and convert to a 2D float64 luminance array."""
    from PIL import Image

    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=np.float64)
    return arr


def _validate_dimensions(arr: np.ndarray) -> None:
    """Validate that field dimensions are reasonable after processing."""
    if arr.ndim != 2:
        raise ValueError(
            f"Expected 2D array from image, got shape {arr.shape}"
        )
    height, width = arr.shape
    if height < _MIN_PIXEL_DIMENSION or width < _MIN_PIXEL_DIMENSION:
        raise ValueError(
            f"Image too small: {width}x{height}, "
            f"minimum is {_MIN_PIXEL_DIMENSION}x{_MIN_PIXEL_DIMENSION}"
        )
    if height > _MAX_PIXEL_DIMENSION or width > _MAX_PIXEL_DIMENSION:
        raise ValueError(
            f"Image too large after downsampling: {width}x{height}, "
            f"maximum is {_MAX_PIXEL_DIMENSION}x{_MAX_PIXEL_DIMENSION}"
        )


def _downsample(arr: np.ndarray, factor: int) -> np.ndarray:
    """Downsample a 2D array by the given integer factor."""
    if factor <= 1:
        return arr
    return arr[::factor, ::factor]


def _normalize_and_scale(arr: np.ndarray, height_scale: float) -> np.ndarray:
    """Normalize array to [0, 1] and apply height scale."""
    arr_min = float(arr.min())
    arr_max = float(arr.max())
    value_range = arr_max - arr_min
    if value_range < 1e-12:
        logger.warning("Image is uniform (range=%.2e); returning flat field", value_range)
        return np.zeros_like(arr)
    normalized = (arr - arr_min) / value_range
    return normalized * height_scale


def _compute_aspect_bbox(field: np.ndarray, z_min: float, z_max: float) -> BoundingBox:
    """Compute bounding box preserving image aspect ratio in XY."""
    height, width = field.shape
    max_dim = max(width, height)
    x_extent = width / max_dim
    y_extent = height / max_dim
    return BoundingBox(
        min_corner=(0.0, 0.0, z_min),
        max_corner=(x_extent, y_extent, z_max),
    )


@register
class HeightmapGenerator(GeneratorBase):
    """Heightmap from image or GeoTIFF file.

    Reads pixel luminance from an image file and produces a scalar field
    for HEIGHTMAP_RELIEF representation. Supports PNG, JPEG, BMP, and
    optionally GeoTIFF (requires rasterio). The seed parameter is accepted
    for interface conformance but unused — output is fully determined by
    the input file.
    """

    name = "heightmap"
    category = "data_driven"
    aliases = ("heightmap_image",)
    description = "Heightmap relief from image or GeoTIFF file"
    resolution_params = {}

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters for heightmap generation."""
        return {
            "input_file": "",
            "height_scale": _DEFAULT_HEIGHT_SCALE,
            "downsample": _DEFAULT_DOWNSAMPLE,
        }

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate a heightmap scalar field from an image file.

        The seed parameter is accepted for interface conformance but does
        not affect output — the result is fully determined by the input file.
        """
        merged = self.get_default_params()
        if params:
            merged.update(params)

        input_file = str(merged["input_file"])
        height_scale = float(merged["height_scale"])
        downsample = int(merged["downsample"])

        if height_scale <= 0:
            raise ValueError(f"height_scale must be positive, got {height_scale}")
        if downsample < 1:
            raise ValueError(f"downsample must be >= 1, got {downsample}")

        if not input_file:
            logger.info("No input_file provided, using built-in demo heightmap")
            arr = _synthesize_demo_heightmap(seed)
            merged["demo_mode"] = True
        else:
            all_supported = _SUPPORTED_IMAGE_EXTENSIONS | _SUPPORTED_GEOTIFF_EXTENSIONS
            path = validate_input_file(input_file, all_supported)
            arr = _try_load_geotiff(path)
            if arr is None:
                arr = _load_image_as_array(path)

        # Downsample before dimension validation so large images with high
        # downsample factors are not unnecessarily rejected.
        arr = _downsample(arr, downsample)
        _validate_dimensions(arr)
        field = _normalize_and_scale(arr, height_scale)

        z_min = float(field.min())
        z_max = float(field.max())
        bbox = _compute_aspect_bbox(field, z_min, z_max)

        if input_file:
            logger.info(
                "Generated heightmap: file=%s, shape=%s, z_range=[%.4f, %.4f]",
                path.name, field.shape, z_min, z_max,
            )
        else:
            logger.info(
                "Generated heightmap demo: shape=%s, z_range=[%.4f, %.4f]",
                field.shape, z_min, z_max,
            )

        return MathObject(
            scalar_field=field,
            generator_name=self.resolved_name or self.name,
            category=self.category,
            parameters=merged,
            seed=seed,
            bounding_box=bbox,
        )

    def get_default_representation(self) -> RepresentationConfig:
        """Return HEIGHTMAP_RELIEF as default representation."""
        return RepresentationConfig(type=RepresentationType.HEIGHTMAP_RELIEF)
