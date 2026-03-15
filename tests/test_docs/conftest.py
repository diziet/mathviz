"""Shared fixtures and helpers for documentation tests."""

from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent.parent
README_PATH = ROOT / "README.md"
DOCS_DIR = ROOT / "docs"
GENERATORS_DOC = DOCS_DIR / "generators.md"


def read_text(path: Path) -> str:
    """Read file contents as UTF-8 text."""
    return path.read_text(encoding="utf-8")


@pytest.fixture()
def gen_doc() -> str:
    """Load generators.md content."""
    return read_text(GENERATORS_DOC)
