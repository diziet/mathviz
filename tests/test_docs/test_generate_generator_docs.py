"""Tests for the generate_generator_docs.py script and its output.

Verifies that:
- The script runs without errors and produces docs/generators.md
- Every registered generator has a section in the output
- Parameter tables match actual defaults from the registry
- Thumbnail images exist for all generators
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

import mathviz.core.generator as _gen_module

_gen_module._ensure_discovered()

from mathviz.core.generator import GeneratorMeta, list_generators

ROOT = Path(__file__).parent.parent.parent
DOCS_DIR = ROOT / "docs"
GENERATORS_MD = DOCS_DIR / "generators.md"
IMAGES_DIR = DOCS_DIR / "images" / "generators"
SCRIPT_PATH = ROOT / "scripts" / "generate_generator_docs.py"


def _read_generators_md() -> str:
    """Read the current generators.md content."""
    return GENERATORS_MD.read_text(encoding="utf-8")


def _get_all_generators() -> list[GeneratorMeta]:
    """Return all registered generators sorted by name."""
    return sorted(list_generators(), key=lambda m: m.name)


def _extract_h3_headings(content: str) -> set[str]:
    """Extract all h3 headings from markdown."""
    return {m.group(1) for m in re.finditer(r"^### (\S+)", content, re.MULTILINE)}


def _get_section(content: str, name: str) -> str:
    """Extract the section for a generator between its h3 and the next h2/h3."""
    pattern = rf"^### {re.escape(name)}\s*\n(.*?)(?=^###? |\Z)"
    match = re.search(pattern, content, re.MULTILINE | re.DOTALL)
    if match is None:
        return ""
    return match.group(1)


class TestScriptRuns:
    """The doc generation script runs without errors."""

    def test_script_produces_output(self) -> None:
        """Running the script with --stdout produces markdown output."""
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--stdout"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, (
            f"Script failed with stderr: {result.stderr}"
        )
        assert "# Generators" in result.stdout
        assert len(result.stdout) > 1000

    def test_script_check_mode(self) -> None:
        """--check exits 0 when generators.md is up to date."""
        result = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--check"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, (
            f"generators.md is out of date: {result.stderr}"
        )

    def test_generators_md_exists(self) -> None:
        """docs/generators.md exists after script has run."""
        assert GENERATORS_MD.is_file(), "docs/generators.md not found"


class TestEveryGeneratorHasSection:
    """Every registered generator has a section in generators.md."""

    def test_all_generators_present(self) -> None:
        """Every generator name appears as an h3 heading."""
        content = _read_generators_md()
        headings = _extract_h3_headings(content)
        for meta in _get_all_generators():
            assert meta.name in headings, (
                f"Generator '{meta.name}' missing from docs/generators.md"
            )

    def test_no_extra_generators(self) -> None:
        """No h3 heading exists that isn't a registered generator."""
        content = _read_generators_md()
        headings = _extract_h3_headings(content)
        registered = {m.name for m in _get_all_generators()}
        extra = headings - registered
        assert not extra, (
            f"docs/generators.md has entries for unregistered generators: {extra}"
        )


class TestParameterTablesMatchDefaults:
    """Parameter tables in the docs match actual defaults from the registry."""

    @pytest.mark.parametrize(
        "meta",
        _get_all_generators(),
        ids=[m.name for m in _get_all_generators()],
    )
    def test_defaults_match(self, meta: GeneratorMeta) -> None:
        """Each parameter's default value in the doc matches the registry."""
        content = _read_generators_md()
        section = _get_section(content, meta.name)
        assert section, f"No section found for {meta.name}"

        inst = meta.generator_class.create(meta.name)
        params = inst.get_default_params()

        for pname, pvalue in params.items():
            # Check that the parameter name appears in the section
            assert f"`{pname}`" in section, (
                f"Parameter '{pname}' not found in section for {meta.name}"
            )
            # Check that the default value is present
            if isinstance(pvalue, float):
                formatted = f"{pvalue:.6g}"
                assert formatted in section, (
                    f"Default value {formatted} for '{pname}' not found "
                    f"in {meta.name} section"
                )
            elif isinstance(pvalue, bool):
                assert str(pvalue) in section
            elif isinstance(pvalue, int):
                assert str(pvalue) in section
            elif isinstance(pvalue, str) and pvalue:
                assert pvalue in section


class TestThumbnailsExist:
    """Thumbnail images exist for all generators."""

    def test_images_directory_exists(self) -> None:
        """docs/images/generators/ directory exists."""
        assert IMAGES_DIR.is_dir(), (
            "docs/images/generators/ directory not found"
        )

    @pytest.mark.parametrize(
        "meta",
        _get_all_generators(),
        ids=[m.name for m in _get_all_generators()],
    )
    def test_thumbnail_exists(self, meta: GeneratorMeta) -> None:
        """Each generator has a PNG thumbnail in docs/images/generators/."""
        thumb_path = IMAGES_DIR / f"{meta.name}.png"
        assert thumb_path.is_file(), (
            f"Thumbnail not found for {meta.name}: {thumb_path}"
        )

    @pytest.mark.parametrize(
        "meta",
        _get_all_generators(),
        ids=[m.name for m in _get_all_generators()],
    )
    def test_thumbnail_not_empty(self, meta: GeneratorMeta) -> None:
        """Each thumbnail file has non-zero size (valid PNG)."""
        thumb_path = IMAGES_DIR / f"{meta.name}.png"
        if thumb_path.is_file():
            assert thumb_path.stat().st_size > 100, (
                f"Thumbnail for {meta.name} appears to be empty/corrupt"
            )


class TestDocStructure:
    """The generated docs have proper structure."""

    def test_has_table_of_contents(self) -> None:
        """docs/generators.md has a table of contents section."""
        content = _read_generators_md()
        assert "## Table of Contents" in content

    def test_has_category_headings(self) -> None:
        """Each category has an h2 heading."""
        content = _read_generators_md()
        categories = {m.category for m in _get_all_generators()}
        h2_headings = {
            m.group(1).lower()
            for m in re.finditer(r"^## (.+)$", content, re.MULTILINE)
        }
        # Map categories to expected heading patterns
        heading_map = {
            "attractors": "attractors",
            "curves": "curves",
            "data_driven": "data-driven",
            "fractals": "fractals",
            "geometry": "geometry",
            "implicit": "implicit surfaces",
            "knots": "knots",
            "number_theory": "number theory",
            "parametric": "parametric surfaces",
            "physics": "physics",
            "procedural": "procedural",
            "surfaces": "surfaces",
        }
        for cat in categories:
            expected = heading_map.get(cat, cat)
            found = any(expected in h for h in h2_headings)
            assert found, (
                f"Category '{cat}' missing h2 heading (expected '{expected}')"
            )

    def test_generator_count_in_header(self) -> None:
        """The header mentions the correct generator count."""
        content = _read_generators_md()
        total = len(_get_all_generators())
        assert f"{total} generators" in content

    @pytest.mark.parametrize(
        "meta",
        _get_all_generators(),
        ids=[m.name for m in _get_all_generators()],
    )
    def test_section_has_thumbnail_reference(self, meta: GeneratorMeta) -> None:
        """Each generator section references a thumbnail image."""
        content = _read_generators_md()
        section = _get_section(content, meta.name)
        assert f"![{meta.name}]" in section, (
            f"No thumbnail image reference in section for {meta.name}"
        )

    @pytest.mark.parametrize(
        "meta",
        _get_all_generators(),
        ids=[m.name for m in _get_all_generators()],
    )
    def test_section_has_representation(self, meta: GeneratorMeta) -> None:
        """Each section mentions the recommended representation."""
        content = _read_generators_md()
        section = _get_section(content, meta.name)
        assert "Recommended representation:" in section, (
            f"No representation info in section for {meta.name}"
        )
