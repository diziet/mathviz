#!/usr/bin/env python3
"""Export snapshot presets to demo_presets.json.

Reads all snapshots from ~/.mathviz/snapshots/ (or MATHVIZ_SNAPSHOTS_DIR),
picks the newest snapshot per generator, and writes a demo_presets.json file
with the curated params and seed for each generator.

Usage:
    python scripts/export_presets.py
    python scripts/export_presets.py --output src/mathviz/data/demo_presets.json
    python scripts/export_presets.py --snapshots-dir /path/to/snapshots
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from mathviz.preview.snapshots import get_snapshots_dir, list_snapshots

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "src" / "mathviz" / "data" / "demo_presets.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for export_presets."""
    parser = argparse.ArgumentParser(
        description="Export snapshot presets to demo_presets.json"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output file path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--snapshots-dir",
        type=Path,
        default=None,
        help="Override snapshots directory (default: from env or ~/.mathviz/snapshots)",
    )
    return parser.parse_args(argv)


def build_presets_from_snapshots(snapshots_dir: Path | None = None) -> dict[str, dict]:
    """Read snapshots and return a dict of generator name -> {params, seed}.

    Picks the newest snapshot per generator (list_snapshots returns newest-first).
    """
    import os

    if snapshots_dir is not None:
        old_val = os.environ.get("MATHVIZ_SNAPSHOTS_DIR")
        os.environ["MATHVIZ_SNAPSHOTS_DIR"] = str(snapshots_dir)

    try:
        snapshots = list_snapshots()
    finally:
        if snapshots_dir is not None:
            if old_val is None:
                os.environ.pop("MATHVIZ_SNAPSHOTS_DIR", None)
            else:
                os.environ["MATHVIZ_SNAPSHOTS_DIR"] = old_val

    presets: dict[str, dict] = {}
    for snap in snapshots:
        if snap.generator not in presets:
            presets[snap.generator] = {
                "params": snap.params,
                "seed": snap.seed,
            }
    return presets


def write_presets(presets: dict[str, dict], output_path: Path) -> None:
    """Write presets dict to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(presets, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.info("Wrote %d preset(s) to %s", len(presets), output_path)


def main(argv: list[str] | None = None) -> None:
    """Entry point for export_presets script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    args = parse_args(argv)

    snapshots_dir = args.snapshots_dir
    presets = build_presets_from_snapshots(snapshots_dir)

    if not presets:
        logger.warning("No snapshots found; writing empty presets file")

    write_presets(presets, args.output)
    print(f"Exported {len(presets)} preset(s) to {args.output}")


if __name__ == "__main__":
    main()
