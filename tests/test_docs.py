"""Tests for documentation completeness and validity."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
README_PATH = ROOT / "README.md"
DOCS_DIR = ROOT / "docs"

# Expected doc files
EXPECTED_DOC_FILES = [
    "generators.md",
    "pipeline.md",
    "cli.md",
    "configuration.md",
    "representation.md",
    "rendering.md",
    "grid.md",
    "api.md",
]


def _read_text(path: Path) -> str:
    """Read file contents as UTF-8 text."""
    return path.read_text(encoding="utf-8")


class TestReadme:
    """Tests for README.md content."""

    def test_readme_exists(self) -> None:
        """README.md exists at project root."""
        assert README_PATH.is_file(), "README.md not found at project root"

    def test_readme_not_empty(self) -> None:
        """README.md is not empty."""
        content = _read_text(README_PATH)
        assert len(content.strip()) > 0, "README.md is empty"

    def test_readme_has_install_section(self) -> None:
        """README.md contains an install section."""
        content = _read_text(README_PATH).lower()
        assert "install" in content, "README.md missing install section"

    def test_readme_has_quickstart_section(self) -> None:
        """README.md contains a quickstart section."""
        content = _read_text(README_PATH).lower()
        assert "quickstart" in content, "README.md missing quickstart section"

    def test_readme_has_generators_section(self) -> None:
        """README.md contains a generators section."""
        content = _read_text(README_PATH).lower()
        assert "generator" in content, "README.md missing generators section"

    def test_readme_has_cli_section(self) -> None:
        """README.md contains a CLI section."""
        content = _read_text(README_PATH).lower()
        assert "cli" in content, "README.md missing CLI section"


class TestDocsStructure:
    """Tests for docs/ directory structure."""

    def test_docs_dir_exists(self) -> None:
        """docs/ directory exists."""
        assert DOCS_DIR.is_dir(), "docs/ directory not found"

    @pytest.mark.parametrize("filename", EXPECTED_DOC_FILES)
    def test_doc_file_exists(self, filename: str) -> None:
        """Each expected doc file exists."""
        path = DOCS_DIR / filename
        assert path.is_file(), f"docs/{filename} not found"

    @pytest.mark.parametrize("filename", EXPECTED_DOC_FILES)
    def test_doc_file_not_empty(self, filename: str) -> None:
        """Each doc file has non-empty content."""
        path = DOCS_DIR / filename
        content = _read_text(path)
        assert len(content.strip()) > 100, f"docs/{filename} appears empty or too short"

    @pytest.mark.parametrize("filename", EXPECTED_DOC_FILES)
    def test_doc_file_valid_markdown(self, filename: str) -> None:
        """Each doc file has valid markdown (starts with heading, no broken headers)."""
        path = DOCS_DIR / filename
        content = _read_text(path)
        lines = content.strip().split("\n")
        assert lines[0].startswith("# "), (
            f"docs/{filename} does not start with a level-1 heading"
        )
        # Check for broken headers (# with no text after)
        for i, line in enumerate(lines, 1):
            if re.match(r"^#{1,6}\s*$", line):
                pytest.fail(f"docs/{filename}:{i} has empty heading: {line!r}")


class TestDocsIndex:
    """Tests that README links to all doc files."""

    def test_readme_links_to_all_docs(self) -> None:
        """README.md contains links to every file in docs/."""
        readme_content = _read_text(README_PATH)
        existing_docs = sorted(p.name for p in DOCS_DIR.glob("*.md"))
        for doc_name in existing_docs:
            link_target = f"docs/{doc_name}"
            assert link_target in readme_content, (
                f"README.md does not link to {link_target}"
            )


class TestCliCompleteness:
    """Tests that docs/cli.md documents all CLI commands."""

    def _get_registered_commands(self) -> list[str]:
        """Get CLI command names from the codebase."""
        # Commands registered in cli.py
        commands = ["generate", "list", "info", "validate"]
        # Commands from cli_preview.py
        commands.append("preview")
        # Commands from cli_render.py
        commands.extend(["render", "render-2d"])
        # Commands from cli_utils.py
        commands.extend(["convert", "sample", "transform", "schema"])
        # Grid subcommands from cli_grid.py
        commands.extend([
            "grid init", "grid show", "grid assign",
            "grid status", "grid neighbors", "grid summary",
            "grid export-all",
        ])
        return commands

    def test_all_cli_commands_documented(self) -> None:
        """Every CLI command in the codebase appears in docs/cli.md."""
        cli_doc = _read_text(DOCS_DIR / "cli.md")
        for command in self._get_registered_commands():
            # Check the command name appears (without mathviz prefix)
            assert command in cli_doc, (
                f"CLI command '{command}' not found in docs/cli.md"
            )


class TestGeneratorCompleteness:
    """Tests that docs/generators.md documents all registered generators."""

    def test_all_generators_documented(self) -> None:
        """Every generator in the registry appears in docs/generators.md."""
        from mathviz.core.generator import list_generators

        gen_doc = _read_text(DOCS_DIR / "generators.md")
        for meta in list_generators():
            assert meta.name in gen_doc, (
                f"Generator '{meta.name}' not found in docs/generators.md"
            )


class TestRepresentationCompleteness:
    """Tests that docs/representation.md documents all strategies."""

    def test_all_strategies_documented(self) -> None:
        """Every RepresentationType enum value appears in docs/representation.md."""
        from mathviz.core.representation import RepresentationType

        rep_doc = _read_text(DOCS_DIR / "representation.md")
        for strategy in RepresentationType:
            assert strategy.value in rep_doc, (
                f"Representation strategy '{strategy.value}' not found "
                f"in docs/representation.md"
            )
