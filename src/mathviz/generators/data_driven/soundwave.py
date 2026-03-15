"""Soundwave generator from audio files.

Reads a WAV file via scipy.io.wavfile, extracts the amplitude envelope,
and maps it to a 3D curve. The waveform is laid out along the x-axis with
amplitude on the y-axis. When no input file is provided, synthesizes a
built-in demo waveform.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import wavfile

from mathviz.core.generator import GeneratorBase, register
from mathviz.core.math_object import BoundingBox, Curve, MathObject
from mathviz.core.representation import RepresentationConfig, RepresentationType
from mathviz.generators.data_driven._file_utils import validate_input_file

logger = logging.getLogger(__name__)

_DEFAULT_NUM_SAMPLES = 2048
_DEFAULT_AMPLITUDE_SCALE = 1.0
_DEFAULT_LENGTH = 2.0
_SUPPORTED_EXTENSIONS = {".wav"}
_MIN_NUM_SAMPLES = 16
_MAX_NUM_SAMPLES = 65536


def _load_wav(path: Path) -> tuple[int, np.ndarray]:
    """Load a WAV file and return (sample_rate, mono_samples)."""
    sample_rate, data = wavfile.read(str(path))

    # Convert to float64 in [-1, 1] range
    if data.dtype.kind == "i":
        max_val = float(np.iinfo(data.dtype).max)
        data = data.astype(np.float64) / max_val
    elif data.dtype.kind == "u":
        # Unsigned: midpoint is at half of max+1 (e.g. 128 for uint8)
        bits = data.dtype.itemsize * 8
        midpoint = 2 ** (bits - 1)
        data = (data.astype(np.float64) - midpoint) / midpoint
    else:
        data = data.astype(np.float64)

    # Mix to mono if stereo
    if data.ndim == 2:
        data = data.mean(axis=1)

    return sample_rate, data


def _compute_envelope(samples: np.ndarray, num_output: int) -> np.ndarray:
    """Compute amplitude envelope by chunking and taking max absolute value.

    Uses np.array_split to distribute all samples evenly across chunks,
    ensuring no trailing samples are discarded.
    """
    total = len(samples)
    if total <= num_output:
        # Not enough samples to chunk — interpolate to desired length
        envelope = np.abs(samples)
        x_orig = np.linspace(0, 1, len(envelope))
        x_new = np.linspace(0, 1, num_output)
        envelope = np.interp(x_new, x_orig, envelope)
        return envelope

    chunks = np.array_split(samples, num_output)
    return np.array([np.max(np.abs(chunk)) for chunk in chunks])


def _synthesize_demo_envelope(num_samples: int, seed: int) -> np.ndarray:
    """Synthesize a demo amplitude envelope from a 440 Hz sine wave."""
    rng = np.random.default_rng(seed)
    sample_rate = 44100
    duration = 2.0
    total_samples = int(sample_rate * duration)
    t = np.linspace(0, duration, total_samples, endpoint=False)

    # 440 Hz sine with amplitude envelope (attack-sustain-release)
    freq = 440.0
    envelope_shape = np.clip(t / 0.1, 0, 1) * np.clip((duration - t) / 0.3, 0, 1)
    signal = np.sin(2 * np.pi * freq * t) * envelope_shape

    # Add slight randomness for visual interest
    noise = rng.normal(0, 0.02, total_samples)
    signal = signal + noise * envelope_shape

    return _compute_envelope(signal, num_samples)


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
    on the y-axis. The seed parameter is accepted for interface conformance
    but unused — output is fully determined by the input file.
    """

    name = "soundwave"
    category = "data_driven"
    aliases = ("audio_waveform",)
    description = "3D waveform visualization from WAV audio file"
    resolution_params = {
        "num_samples": "Number of envelope sample points",
    }
    _resolution_defaults = {"num_samples": _DEFAULT_NUM_SAMPLES}

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
        """Generate a 3D waveform curve from a WAV file.

        The seed parameter is accepted for interface conformance but does
        not affect output — the result is fully determined by the input file.
        """
        merged = self.get_default_params()
        if params:
            merged.update(params)

        input_file = str(merged["input_file"])
        amplitude_scale = float(merged["amplitude_scale"])
        length = float(merged["length"])
        num_samples = int(
            resolution_kwargs.get("num_samples", _DEFAULT_NUM_SAMPLES)
        )

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

        merged["num_samples"] = num_samples
        path: Path | None = None

        if not input_file:
            logger.info("No input_file provided, using built-in demo waveform")
            envelope = _synthesize_demo_envelope(num_samples, seed)
            merged["demo_mode"] = True
        else:
            path = validate_input_file(input_file, _SUPPORTED_EXTENSIONS)
            sample_rate, samples = _load_wav(path)
            merged["sample_rate"] = sample_rate
            merged["total_audio_samples"] = len(samples)
            envelope = _compute_envelope(samples, num_samples)

        points = _build_waveform_curve(envelope, length, amplitude_scale)
        bbox = BoundingBox.from_points(points)

        source = f"file={path.name}, " if path else "demo, "
        logger.info(
            "Generated soundwave: %senvelope_points=%d",
            source, num_samples,
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
