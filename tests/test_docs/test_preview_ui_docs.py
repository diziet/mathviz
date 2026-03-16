"""Tests for preview UI documentation completeness."""

import re

import pytest

from tests.test_docs.conftest import DOCS_DIR, README_PATH, read_text

PREVIEW_UI_DOC = DOCS_DIR / "preview-ui.md"

# Every documented UI feature that must appear in preview-ui.md
REQUIRED_FEATURES = [
    "generator browser",
    "cmd+k",
    "category",
    "search",
    "thumbnail",
    "parameters panel",
    "auto-apply",
    "randomize",
    "dice",
    "enter to apply",
    "resolution",
    "container panel",
    "uniform margin",
    "collapsible",
    "point cloud",
    "shaded mesh",
    "wireframe",
    "crystal preview",
    "color map",
    "camera controls",
    "orbit",
    "pan",
    "zoom",
    "render lock",
    "full lock",
    "reset view",
    "bounding box",
    "axes",
    "light background",
    "stretch",
    "density slider",
    "turntable",
    "auto-rotate",
    "gif",
    "webm",
    "compare mode",
    "2x2",
    "3x3",
    "per-panel",
    "save",
    "load",
    "snapshot",
    "gallery",
    "disk cache",
    "cache indicator",
    "force regenerate",
    "keyboard shortcuts",
    "screenshot",
]

# Keyboard shortcuts that exist in the source code (index.html)
# Each tuple is (shortcut description, expected doc text)
CODE_KEYBOARD_SHORTCUTS = [
    "cmd+k",
    "ctrl+k",
    "home",
    "escape",
    "backspace",
    "arrow",
    "enter",
    "r",
]


class TestPreviewUiDocExists:
    """Tests that preview-ui.md exists and is valid."""

    def test_doc_exists(self) -> None:
        """docs/preview-ui.md exists."""
        assert PREVIEW_UI_DOC.is_file(), "docs/preview-ui.md not found"

    def test_doc_not_empty(self) -> None:
        """docs/preview-ui.md has substantial content."""
        content = read_text(PREVIEW_UI_DOC)
        assert len(content.strip()) > 1000, "docs/preview-ui.md is too short"

    def test_doc_starts_with_heading(self) -> None:
        """docs/preview-ui.md starts with a level-1 heading."""
        content = read_text(PREVIEW_UI_DOC)
        first_line = content.strip().split("\n")[0]
        assert first_line.startswith("# "), (
            "docs/preview-ui.md does not start with a level-1 heading"
        )

    def test_no_empty_headings(self) -> None:
        """No empty headings in preview-ui.md."""
        content = read_text(PREVIEW_UI_DOC)
        for i, line in enumerate(content.split("\n"), 1):
            if re.match(r"^#{1,6}\s*$", line):
                pytest.fail(
                    f"docs/preview-ui.md:{i} has empty heading: {line!r}"
                )


class TestPreviewUiFeatureCoverage:
    """Tests that preview-ui.md covers all documented features."""

    @pytest.mark.parametrize("feature", REQUIRED_FEATURES)
    def test_feature_documented(self, feature: str) -> None:
        """Each UI feature is mentioned in preview-ui.md."""
        content = read_text(PREVIEW_UI_DOC).lower()
        assert feature.lower() in content, (
            f"UI feature '{feature}' not found in docs/preview-ui.md"
        )


class TestKeyboardShortcutCoverage:
    """Tests that every keyboard shortcut in code has a docs entry."""

    @pytest.mark.parametrize("shortcut", CODE_KEYBOARD_SHORTCUTS)
    def test_shortcut_documented(self, shortcut: str) -> None:
        """Each keyboard shortcut from code appears in preview-ui.md."""
        content = read_text(PREVIEW_UI_DOC).lower()
        assert shortcut.lower() in content, (
            f"Keyboard shortcut '{shortcut}' not found in docs/preview-ui.md"
        )

    def test_shortcut_section_exists(self) -> None:
        """A dedicated keyboard shortcuts section exists."""
        content = read_text(PREVIEW_UI_DOC)
        assert "## Keyboard Shortcuts" in content, (
            "docs/preview-ui.md missing '## Keyboard Shortcuts' section"
        )

    def test_shortcut_table_has_entries(self) -> None:
        """The keyboard shortcuts section contains a table with entries."""
        content = read_text(PREVIEW_UI_DOC)
        shortcuts_section = content.split("## Keyboard Shortcuts")[-1]
        # Count table rows (lines starting with |)
        table_rows = [
            line for line in shortcuts_section.split("\n")
            if line.strip().startswith("|") and "---" not in line
        ]
        # Subtract header row
        data_rows = len(table_rows) - 1
        assert data_rows >= 8, (
            f"Keyboard shortcuts table has only {data_rows} entries, "
            f"expected at least 8"
        )


class TestReadmeLinksToPreviewUi:
    """Tests that README links to preview-ui.md."""

    def test_readme_links_to_preview_ui(self) -> None:
        """README.md contains a link to docs/preview-ui.md."""
        content = read_text(README_PATH)
        assert "docs/preview-ui.md" in content, (
            "README.md does not link to docs/preview-ui.md"
        )
