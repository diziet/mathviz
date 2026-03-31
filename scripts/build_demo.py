#!/usr/bin/env python3
"""Build a self-contained static demo site from generator pipeline output.

Produces a deployable directory with index.html, geometry files, thumbnails,
and manifest.json suitable for Cloudflare Pages or any static host.

Delegates to :mod:`mathviz.demo_builder` for the actual build logic.

Usage:
    python scripts/build_demo.py                         # all generators
    python scripts/build_demo.py --generators lorenz,gyroid
    python scripts/build_demo.py --output build/ --profile preview
    python scripts/build_demo.py --no-presets             # ignore presets
"""

import argparse
import logging
import sys
from pathlib import Path

from mathviz.demo_builder import build_demo as _lib_build_demo

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = Path("dist")
DEFAULT_PROFILE = "preview"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for build_demo."""
    parser = argparse.ArgumentParser(
        description="Build a self-contained static demo site"
    )
    parser.add_argument(
        "--generators",
        default="all",
        help="Comma-separated generator names, or 'all' (default: all)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--profile",
        default=DEFAULT_PROFILE,
        help=f"Sampling profile name (default: {DEFAULT_PROFILE})",
    )
    parser.add_argument(
        "--no-presets",
        action="store_true",
        default=False,
        help="Ignore snapshots and presets file; use generator defaults",
    )
    return parser.parse_args(argv)


def build_demo(
    generator_spec: str,
    output_dir: Path,
    profile: str,
    use_presets: bool = True,
) -> int:
    """Build the full demo site. Returns count of successful generators."""
    result = _lib_build_demo(generator_spec, output_dir, profile, use_presets)
    return result.succeeded


def main(argv: list[str] | None = None) -> None:
    """Entry point for the build_demo script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    args = parse_args(argv)
    succeeded = build_demo(
        args.generators, args.output, args.profile,
        use_presets=not args.no_presets,
    )
    if succeeded == 0:
        logger.error("No generators exported successfully")
        sys.exit(1)


if __name__ == "__main__":
    main()
