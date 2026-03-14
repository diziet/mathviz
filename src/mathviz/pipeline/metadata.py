"""MetadataExporter: writes sidecar .meta.json files for reproducibility."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mathviz.core.math_object import MathObject

logger = logging.getLogger(__name__)

VERSION = "0.1.0"


def build_metadata(obj: MathObject, **extra: Any) -> dict[str, Any]:
    """Build a metadata dict from a MathObject for JSON serialization."""
    meta: dict[str, Any] = {
        "generator_name": obj.generator_name,
        "params": obj.parameters,
        "seed": obj.seed,
        "category": obj.category,
        "coord_space": obj.coord_space.value,
        "representation": obj.representation,
        "description": obj.description,
        "generation_time_seconds": obj.generation_time_seconds,
        "export_timestamp": datetime.now(timezone.utc).isoformat(),
        "version": VERSION,
    }
    if obj.bounding_box is not None:
        meta["bounding_box"] = {
            "min_corner": list(obj.bounding_box.min_corner),
            "max_corner": list(obj.bounding_box.max_corner),
        }
    meta.update(extra)
    return meta


def write_metadata(path: Path, obj: MathObject, **extra: Any) -> Path:
    """Write a .meta.json sidecar file next to the given export path."""
    meta = build_metadata(obj, **extra)
    meta_path = path.with_suffix(path.suffix + ".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote metadata sidecar: %s", meta_path)
    return meta_path


def resolve_format(path: Path, fmt: str | None) -> str:
    """Resolve export format from explicit argument or file suffix."""
    if fmt is not None:
        return fmt.lower().lstrip(".")
    suffix = path.suffix.lower().lstrip(".")
    if not suffix:
        raise ValueError(f"Cannot infer format from path '{path}' — specify fmt explicitly")
    return suffix


def validate_format(fmt: str, supported: set[str], label: str) -> None:
    """Validate that the format is in the supported set."""
    if fmt not in supported:
        raise ValueError(
            f"Unsupported {label} format '{fmt}'. Supported: {sorted(supported)}"
        )
