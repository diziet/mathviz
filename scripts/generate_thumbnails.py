#!/usr/bin/env python3
"""Pre-render thumbnail PNGs for all generators into docs/images/generators/.

Uses the thumbnail system from Task 123 (mathviz.preview.thumbnails) to
generate each thumbnail, then copies it to the docs image directory.

Usage:
    python scripts/generate_thumbnails.py               # generate all
    python scripts/generate_thumbnails.py --skip-existing  # skip cached
"""

import argparse
import logging
import shutil
from pathlib import Path

from mathviz.core.generator import list_generators
from mathviz.preview.thumbnails import generate_thumbnail

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
IMAGES_DIR = ROOT / "docs" / "images" / "generators"
VIEW_MODE = "points"


def generate_all_thumbnails(skip_existing: bool = False) -> int:
    """Generate thumbnails for every registered generator.

    Returns the number of thumbnails generated.
    """
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    generators = list_generators()
    generated = 0

    for meta in sorted(generators, key=lambda m: m.name):
        output_path = IMAGES_DIR / f"{meta.name}.png"
        if skip_existing and output_path.is_file():
            logger.debug("Skipping %s (already exists)", meta.name)
            continue

        try:
            cached_path = generate_thumbnail(meta.name, VIEW_MODE)
            shutil.copy2(cached_path, output_path)
            generated += 1
            logger.info("Generated thumbnail for %s", meta.name)
        except Exception:
            logger.error(
                "Failed to generate thumbnail for %s",
                meta.name,
                exc_info=True,
            )

    return generated


def main() -> None:
    """Entry point for thumbnail generation."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="Generate thumbnail images for all generators"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip generators that already have thumbnails in docs/",
    )
    args = parser.parse_args()

    count = generate_all_thumbnails(skip_existing=args.skip_existing)
    total = len(list_generators())
    logger.info("Generated %d/%d thumbnails in %s", count, total, IMAGES_DIR)


if __name__ == "__main__":
    main()
