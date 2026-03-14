"""Soundwave generator from audio files.

Reads a WAV file via scipy.io.wavfile, extracts the amplitude envelope,
and maps it to a 3D curve or point cloud. The waveform is laid out along
the x-axis with amplitude on y and optional z modulation.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import wavfile

from mathviz.core.generator import GeneratorBase, register
from mathviz.core.math_object import BoundingBox, Curve, MathObject
from mathviz.core.representation import RepresentationConfig, RepresentationType

logger = logging.getLogger(__name__)

_DEFAULT_NUM_SAMPLES = 2048
_DEFAULT_AMPLITUDE_SCALE = 1.0
_DEFAULT_LENGTH = 2.0
_SUPPORTED_EXTENSIONS = {".wav"}
_MIN_NUM_SAMPLES = 16
_MAX_NUM_SAMPLES = 65536


def _validate_input_file(input_file: str) -> Path:
    """Validate that the input file exists and has a supported extension."""
    path = Path(input_file)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    suffix = path.suffix.lower()
    if suffix not in _SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format '{suffix}'. "
            f"Supported formats: {sorted(_SUPPORTED_EXTENSIONS)}"
        )
    return path


def _load_wav(path: Path) -> tuple[int, np.ndarray]:
    """Load a WAV file and return (sample_rate, mono_samples)."""
    sample_rate, data = wavfile.read(str(path))

    # Convert to float64
    if data.dtype.kind == "i":
        max_val = float(np.iinfo(data.dtype).max)
        data = data.astype(np.float64) / max_val
    elif data.dtype.kind == "u":
        max_val = float(np.iinfo(data.dtype).max)
        data = data.astype(np.float64) / max_val * 2.0 - 1.0
    else:
        data = data.astype(np.float64)

    # Mix to mono if stereo
    if data.ndim == 2:
        data = data.mean(axis=1)

    return sample_rate, data


def _compute_envelope(samples: np.ndarray, num_output: int) -> np.ndarray:
    """Compute amplitude envelope by chunking and taking max absolute value."""
    total = len(samples)
    if total <= num_output:
        # Not enough samples to chunk — use absolute values directly
        envelope = np.abs(samples)
        # Pad or interpolate to desired length
        x_orig = np.linspace(0, 1, len(envelope))
        x_new = np.linspace(0, 1, num_output)
        envelope = np.interp(x_new, x_orig, envelope)
        return envelope

    chunk_size = total // num_output
    trimmed = samples[: chunk_size * num_output]
    chunks = trimmed.reshape(num_output, chunk_size)
    return np.max(np.abs(chunks), axis=1)


def _build_waveform_curve(
    envelope: np.ndarray,
    length: float,
    amplitude_scale: float,
) -> np.ndarray:
    """Map amplitude envelope to 3D points along a curve."""
    num_points = len(envelope)
    x = np.linspace(0, length, num_points)
    y = envelope * amplitude_scale
    z = np.zeros(num_points)
    return np.column_stack([x, y, z]).astype(np.float64)


@register
class SoundwaveGenerator(GeneratorBase):
    """3D waveform from a WAV audio file.

    Extracts the amplitude envelope from an audio file and maps it to
    a 3D curve. The waveform extends along the x-axis with amplitude
    on the y-axis.
    """

    name = "soundwave"
    category = "data_driven"
    aliases = ("audio_waveform",)
    description = "3D waveform visualization from WAV audio file"
    resolution_params = {
        "num_samples": "Number of envelope sample points",
    }

    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters for soundwave generation."""
        return {
            "input_file": "",
            "amplitude_scale": _DEFAULT_AMPLITUDE_SCALE,
            "length": _DEFAULT_LENGTH,
        }

    def generate(
        self,
        params: dict[str, Any] | None = None,
        seed: int = 42,
        **resolution_kwargs: Any,
    ) -> MathObject:
        """Generate a 3D waveform curve from a WAV file."""
        merged = self.get_default_params()
        if params:
            merged.update(params)

        input_file = str(merged["input_file"])
        amplitude_scale = float(merged["amplitude_scale"])
        length = float(merged["length"])
        num_samples = int(
            resolution_kwargs.get("num_samples", _DEFAULT_NUM_SAMPLES)
        )

        if not input_file:
            raise ValueError("input_file parameter is required")
        if amplitude_scale <= 0:
            raise ValueError(
                f"amplitude_scale must be positive, got {amplitude_scale}"
            )
        if length <= 0:
            raise ValueError(f"length must be positive, got {length}")
        if num_samples < _MIN_NUM_SAMPLES:
            raise ValueError(
                f"num_samples must be >= {_MIN_NUM_SAMPLES}, got {num_samples}"
            )
        if num_samples > _MAX_NUM_SAMPLES:
            raise ValueError(
                f"num_samples must be <= {_MAX_NUM_SAMPLES}, got {num_samples}"
            )

        path = _validate_input_file(input_file)
        sample_rate, samples = _load_wav(path)
        merged["num_samples"] = num_samples
        merged["sample_rate"] = sample_rate
        merged["total_audio_samples"] = len(samples)

        envelope = _compute_envelope(samples, num_samples)
        points = _build_waveform_curve(envelope, length, amplitude_scale)
        bbox = BoundingBox.from_points(points)

        logger.info(
            "Generated soundwave: file=%s, sample_rate=%d, "
            "audio_samples=%d, envelope_points=%d",
            path.name, sample_rate, len(samples), num_samples,
        )

        curve = Curve(points=points, closed=False)
        return MathObject(
            curves=[curve],
            generator_name=self.resolved_name or self.name,
            category=self.category,
            parameters=merged,
            seed=seed,
            bounding_box=bbox,
        )

    def get_default_representation(self) -> RepresentationConfig:
        """Return TUBE as default representation."""
        return RepresentationConfig(
            type=RepresentationType.TUBE,
            tube_radius=0.02,
        )
