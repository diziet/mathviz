"""Tests for generator documentation completeness.

Verifies that docs/generators.md and README.md accurately reflect
every generator in the registry.
"""

import re
from pathlib import Path

import pytest

from mathviz.core.generator import GeneratorBase

# Trigger initial import of all generator modules so subclasses exist
import mathviz.core.generator as _gen_module
_gen_module._ensure_discovered()

ROOT = Path(__file__).parent.parent.parent
GENERATORS_DOC = ROOT / "docs" / "generators.md"
README_PATH = ROOT / "README.md"


def _read_text(path: Path) -> str:
    """Read file contents as UTF-8 text."""
    return path.read_text(encoding="utf-8")


def _get_concrete_generators() -> list[type[GeneratorBase]]:
    """Return all concrete GeneratorBase subclasses from the generators package."""
    result: list[type[GeneratorBase]] = []
    stack = list(GeneratorBase.__subclasses__())
    while stack:
        cls = stack.pop()
        sub = cls.__subclasses__()
        if sub:
            stack.extend(sub)
        # Only include generators from the mathviz.generators package
        module = getattr(cls, "__module__", "") or ""
        if cls.name and module.startswith("mathviz.generators"):
            result.append(cls)
    return result


def _get_all_generator_names() -> list[str]:
    """Return sorted list of all registered generator names."""
    return sorted({cls.name for cls in _get_concrete_generators()})


def _get_all_categories() -> set[str]:
    """Return set of all registered generator categories."""
    return {cls.category for cls in _get_concrete_generators()}


def _get_generator_count() -> int:
    """Return the total number of generators."""
    return len(_get_all_generator_names())


def _extract_doc_headings(content: str) -> set[str]:
    """Extract all h3 headings (### name) from markdown content."""
    return {m.group(1) for m in re.finditer(r"^### (\S+)", content, re.MULTILINE)}


def _extract_doc_h2_headings(content: str) -> set[str]:
    """Extract all h2 headings (## Name) from markdown content."""
    return {m.group(1) for m in re.finditer(r"^## (.+)$", content, re.MULTILINE)}


class TestEveryGeneratorDocumented:
    """Every generator in the registry has an entry in docs/generators.md."""

    @pytest.fixture()
    def gen_doc(self) -> str:
        """Load generators.md content."""
        return _read_text(GENERATORS_DOC)

    def test_all_generators_have_entry(self, gen_doc: str) -> None:
        """Every registered generator name appears as an h3 heading."""
        doc_headings = _extract_doc_headings(gen_doc)
        for name in _get_all_generator_names():
            assert name in doc_headings, (
                f"Generator '{name}' is registered but has no ### heading "
                f"in docs/generators.md"
            )


class TestEntryContents:
    """Every entry includes name, category, parameters, and description."""

    @pytest.fixture()
    def gen_doc(self) -> str:
        """Load generators.md content."""
        return _read_text(GENERATORS_DOC)

    def _get_section(self, gen_doc: str, name: str) -> str:
        """Extract the section for a generator between its h3 and the next h2/h3."""
        pattern = rf"^### {re.escape(name)}\s*\n(.*?)(?=^###? |\Z)"
        match = re.search(pattern, gen_doc, re.MULTILINE | re.DOTALL)
        assert match is not None, f"Could not find section for '{name}'"
        return match.group(1)

    @pytest.mark.parametrize("name", _get_all_generator_names())
    def test_entry_has_description(self, gen_doc: str, name: str) -> None:
        """Each generator entry has descriptive text (not just a table)."""
        section = self._get_section(gen_doc, name)
        # Strip tables and code blocks, check remaining text
        text_lines = [
            line for line in section.strip().split("\n")
            if line.strip()
            and not line.strip().startswith("|")
            and not line.strip().startswith("```")
            and not line.strip().startswith("Aliases:")
            and not line.strip().startswith("Resolution:")
            and not line.strip().startswith("Output varies")
            and not line.strip().startswith("Recommended")
            and not line.strip().startswith("No additional")
            and not line.strip().startswith("**Parameter")
        ]
        assert len(text_lines) > 0, (
            f"Generator '{name}' entry has no description text"
        )

    @pytest.mark.parametrize("name", _get_all_generator_names())
    def test_entry_has_parameters_or_note(self, gen_doc: str, name: str) -> None:
        """Each generator entry has a parameter table or 'No additional parameters'."""
        section = self._get_section(gen_doc, name)
        has_table = "| Parameter" in section or "| `" in section
        has_no_params_note = "No additional parameters" in section
        assert has_table or has_no_params_note, (
            f"Generator '{name}' entry has no parameter table and no "
            f"'No additional parameters' note"
        )

    @pytest.mark.parametrize("name", _get_all_generator_names())
    def test_entry_has_example_command(self, gen_doc: str, name: str) -> None:
        """Each generator entry has a sample render command."""
        section = self._get_section(gen_doc, name)
        assert "mathviz generate" in section, (
            f"Generator '{name}' entry has no sample render command"
        )


class TestNoPhantomGenerators:
    """No generator is listed in docs that doesn't exist in the registry."""

    def test_no_phantom_generators(self) -> None:
        """Every h3 heading in docs/generators.md is a registered generator."""
        gen_doc = _read_text(GENERATORS_DOC)
        doc_headings = _extract_doc_headings(gen_doc)
        registered = set(_get_all_generator_names())
        phantoms = doc_headings - registered
        assert not phantoms, (
            f"docs/generators.md lists generators not in registry: {phantoms}"
        )


class TestReadmeGeneratorCount:
    """README generator count matches the actual registry count."""

    def test_readme_count_matches_registry(self) -> None:
        """The '**N generators**' in README matches the registry count."""
        readme = _read_text(README_PATH)
        match = re.search(r"\*\*(\d+) generators\*\*", readme)
        assert match is not None, (
            "README.md does not contain a '**N generators**' count"
        )
        readme_count = int(match.group(1))
        actual_count = _get_generator_count()
        assert readme_count == actual_count, (
            f"README says {readme_count} generators but registry has {actual_count}"
        )


class TestCategoryHeadings:
    """docs/generators.md has a heading for each category."""

    def test_all_categories_have_heading(self) -> None:
        """Every category in the registry has a corresponding h2 heading."""
        gen_doc = _read_text(GENERATORS_DOC)
        doc_h2 = _extract_doc_h2_headings(gen_doc)
        # Normalize to lowercase for matching
        doc_h2_lower = {h.lower() for h in doc_h2}

        # Map category names to expected heading patterns
        category_heading_map = {
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

        for category in _get_all_categories():
            expected = category_heading_map.get(category, category)
            found = any(expected in h for h in doc_h2_lower)
            assert found, (
                f"Category '{category}' has no matching h2 heading in "
                f"docs/generators.md (expected '{expected}' in headings: "
                f"{sorted(doc_h2_lower)})"
            )
