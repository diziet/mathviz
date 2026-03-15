"""Tests for comprehensive documentation completeness across all features."""

import re

import pytest

from tests.test_docs.conftest import DOCS_DIR, README_PATH, read_text

# UI features that must appear in preview.md
PREVIEW_UI_FEATURES = [
    "generator switcher",
    "container",
    "parameter editor",
    "shaded",
    "wireframe",
    "point cloud",
    "lock camera",
    "auto-apply",
    "reset view",
    "save",
    "load",
    "snapshot",
    "comparison mode",
    "2×2",
    "3×3",
    "keyboard shortcuts",
]

# Pipeline stages that must appear in pipeline.md
PIPELINE_STAGES = [
    "generate",
    "represent",
    "transform",
    "sample",
    "validate",
    "export",
]


class TestReadmeCompleteness:
    """README contains all required sections."""

    def test_readme_has_install_section(self) -> None:
        """README contains install instructions."""
        content = read_text(README_PATH).lower()
        assert "install" in content, "README.md missing install section"

    def test_readme_has_run_section(self) -> None:
        """README contains instructions to run the tool."""
        content = read_text(README_PATH).lower()
        assert "quickstart" in content or "run" in content, (
            "README.md missing run/quickstart section"
        )

    def test_readme_has_test_section(self) -> None:
        """README contains testing instructions."""
        content = read_text(README_PATH).lower()
        assert "testing" in content or "pytest" in content, (
            "README.md missing test section"
        )

    def test_readme_has_render_extras(self) -> None:
        """README mentions the [render] extras for installation."""
        content = read_text(README_PATH)
        assert "[render]" in content, (
            "README.md does not mention [render] extras"
        )


class TestPreviewDocCompleteness:
    """docs/preview.md covers all UI features."""

    def test_preview_doc_exists(self) -> None:
        """docs/preview.md exists."""
        path = DOCS_DIR / "preview.md"
        assert path.is_file(), "docs/preview.md not found"

    @pytest.mark.parametrize("feature", PREVIEW_UI_FEATURES)
    def test_preview_covers_feature(self, feature: str) -> None:
        """Each UI feature is mentioned in preview.md."""
        content = read_text(DOCS_DIR / "preview.md").lower()
        assert feature.lower() in content, (
            f"UI feature '{feature}' not found in docs/preview.md"
        )


class TestPipelineDocCompleteness:
    """docs/pipeline.md covers all pipeline stages."""

    def test_pipeline_doc_exists(self) -> None:
        """docs/pipeline.md exists."""
        path = DOCS_DIR / "pipeline.md"
        assert path.is_file(), "docs/pipeline.md not found"

    @pytest.mark.parametrize("stage", PIPELINE_STAGES)
    def test_pipeline_covers_stage(self, stage: str) -> None:
        """Each pipeline stage is documented."""
        content = read_text(DOCS_DIR / "pipeline.md").lower()
        assert stage in content, (
            f"Pipeline stage '{stage}' not found in docs/pipeline.md"
        )


class TestMarkdownValidity:
    """All doc files are valid markdown with no broken headers."""

    def _get_all_doc_files(self) -> list[str]:
        """List all .md files in docs/."""
        return sorted(p.name for p in DOCS_DIR.glob("*.md"))

    @pytest.mark.parametrize(
        "filename",
        sorted(p.name for p in (DOCS_DIR).glob("*.md"))
        if DOCS_DIR.is_dir()
        else [],
    )
    def test_no_broken_headers(self, filename: str) -> None:
        """No empty headings (# with no text) in any doc file."""
        path = DOCS_DIR / filename
        content = read_text(path)
        for i, line in enumerate(content.split("\n"), 1):
            if re.match(r"^#{1,6}\s*$", line):
                pytest.fail(
                    f"docs/{filename}:{i} has empty heading: {line!r}"
                )


class TestCrossDocLinks:
    """No dead links between documentation files."""

    def _extract_md_links(self, content: str) -> list[str]:
        """Extract markdown link targets from content."""
        return re.findall(r"\[.*?\]\((.*?\.md(?:#[^)]*)?)\)", content)

    def test_no_dead_links_in_docs(self) -> None:
        """Every markdown link in docs/ points to an existing file."""
        if not DOCS_DIR.is_dir():
            pytest.skip("docs/ directory not found")

        for doc_path in sorted(DOCS_DIR.glob("*.md")):
            content = read_text(doc_path)
            links = self._extract_md_links(content)
            for link in links:
                # Strip anchor fragment
                file_part = link.split("#")[0]
                if not file_part:
                    continue
                target = DOCS_DIR / file_part
                assert target.is_file(), (
                    f"docs/{doc_path.name} links to '{link}' "
                    f"but {target} does not exist"
                )

    def test_no_dead_links_in_readme(self) -> None:
        """Every markdown link in README.md points to an existing file."""
        content = read_text(README_PATH)
        links = self._extract_md_links(content)
        root = README_PATH.parent
        for link in links:
            file_part = link.split("#")[0]
            if not file_part:
                continue
            target = root / file_part
            assert target.is_file(), (
                f"README.md links to '{link}' but {target} does not exist"
            )
