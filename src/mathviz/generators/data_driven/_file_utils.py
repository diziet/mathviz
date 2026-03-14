"""Shared file validation utilities for data-driven generators."""

from pathlib import Path


def validate_input_file(input_file: str, supported_extensions: set[str]) -> Path:
    """Validate that the input file exists and has a supported extension."""
    path = Path(input_file)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    suffix = path.suffix.lower()
    if suffix not in supported_extensions:
        raise ValueError(
            f"Unsupported file format '{suffix}'. "
            f"Supported formats: {sorted(supported_extensions)}"
        )
    return path
