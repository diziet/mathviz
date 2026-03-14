"""Heightmap generator from image files.

Reads a PNG/JPEG image (via Pillow) and converts pixel luminance to a 2D
scalar field for HEIGHTMAP_RELIEF representation. Optionally supports GeoTIFF
via rasterio if installed.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np

from mathviz.core.generator import GeneratorBase, register
from mathviz.core.math_object import BoundingBox, MathObject
from mathviz.core.representation import RepresentationConfig, RepresentationType

logger = logging.getLogger(__name__)

_DEFAULT_HEIGHT_SCALE = 1.0
_DEFAULT_DOWNSAMPLE = 1
_SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
_SUPPORTED_GEOTIFF_EXTENSIONS = {".tif", ".tiff"}
_MAX_PIXEL_DIMENSION = 4096
_MIN_PIXEL_DIMENSION = 2


def _validate_input_file(input_file: str) -> Path:
    """Validate that the input file exists and has a supported extension."""
    path = Path(input_file)
    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_file}"
        )
    suffix = path.suffix.lower()
    all_supported = _SUPPORTED_IMAGE_EXTENSIONS | _SUPPORTED_GEOTIFF_EXTENSIONS
    if suffix not in all_supported:
        raise ValueError(
            f"Unsupported file format '{suffix}'. "
            f"Supported formats: {sorted(all_supported)}"
        )
    return path


def _try_load_geotiff(path: Path) -> np.ndarray | None:
    """Attempt to load a GeoTIFF via rasterio. Returns None if unavailable."""
    try:
        import rasterio  # noqa: F401
    except ImportError:
        return None

    suffix = path.suffix.lower()
    if suffix not in _SUPPORTED_GEOTIFF_EXTENSIONS:
        return None

    try:
        with rasterio.open(path) as dataset:
            band = dataset.read(1).astype(np.float64)
            return band
    except Exception:
        return None


def _load_image_as_array(path: Path) -> np.ndarray:
    """Load an image file and convert to a 2D float64 luminance array."""
    from PIL import Image

    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=np.float64)
    return arr


def _validate_dimensions(arr: np.ndarray) -> None:
    """Validate that image dimensions are reasonable."""
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
            f"Image too large: {width}x{height}, "
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


@register
class HeightmapGenerator(GeneratorBase):
    """Heightmap from image or GeoTIFF file.

    Reads pixel luminance from an image file and produces a scalar field
    for HEIGHTMAP_RELIEF representation. Supports PNG, JPEG, BMP, and
    optionally GeoTIFF (requires rasterio).
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
        """Generate a heightmap scalar field from an image file."""
        merged = self.get_default_params()
        if params:
            merged.update(params)

        input_file = str(merged["input_file"])
        height_scale = float(merged["height_scale"])
        downsample = int(merged["downsample"])

        if not input_file:
            raise ValueError("input_file parameter is required")
        if height_scale <= 0:
            raise ValueError(f"height_scale must be positive, got {height_scale}")
        if downsample < 1:
            raise ValueError(f"downsample must be >= 1, got {downsample}")

        path = _validate_input_file(input_file)

        # Try GeoTIFF first, fall back to PIL
        arr = _try_load_geotiff(path)
        if arr is None:
            arr = _load_image_as_array(path)

        _validate_dimensions(arr)
        arr = _downsample(arr, downsample)
        field = _normalize_and_scale(arr, height_scale)

        z_min = float(field.min())
        z_max = float(field.max())
        bbox = BoundingBox(
            min_corner=(0.0, 0.0, z_min),
            max_corner=(1.0, 1.0, z_max),
        )

        logger.info(
            "Generated heightmap: file=%s, shape=%s, z_range=[%.4f, %.4f]",
            path.name, field.shape, z_min, z_max,
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
