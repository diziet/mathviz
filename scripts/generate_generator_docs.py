#!/usr/bin/env python3
"""Auto-generate docs/generators.md from the generator registry.

Introspects every registered generator for defaults, parameter ranges,
resolution params, and representation type.  Merges manual descriptions
and example configurations from docs/generator_notes.yaml.

Usage:
    python scripts/generate_generator_docs.py          # write docs/generators.md
    python scripts/generate_generator_docs.py --check   # exit 1 if file would change
    python scripts/generate_generator_docs.py --stdout   # print to stdout
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

from mathviz.core.generator import GeneratorMeta, list_generators

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = ROOT / "docs"
NOTES_PATH = DOCS_DIR / "generator_notes.yaml"
OUTPUT_PATH = DOCS_DIR / "generators.md"
IMAGES_DIR = Path("images/generators")

# Display names for categories (used as h2 headings).
CATEGORY_HEADINGS: dict[str, str] = {
    "attractors": "Attractors",
    "curves": "Curves",
    "data_driven": "Data-Driven",
    "fractals": "Fractals",
    "geometry": "Geometry",
    "implicit": "Implicit Surfaces",
    "knots": "Knots",
    "number_theory": "Number Theory",
    "parametric": "Parametric Surfaces",
    "physics": "Physics",
    "procedural": "Procedural",
    "surfaces": "Surfaces",
}

CATEGORY_BLURBS: dict[str, str] = {
    "attractors": (
        "Strange attractor trajectories computed by integrating "
        "dynamical systems."
    ),
    "curves": "3D curves with configurable shape and resolution.",
    "data_driven": (
        "Generators that read external data files to produce geometry."
    ),
    "fractals": (
        "Self-similar structures from recursive or escape-time algorithms."
    ),
    "geometry": "Geometric primitives and constructions.",
    "implicit": (
        "Surfaces defined by implicit equations, extracted via "
        "marching cubes."
    ),
    "knots": "Mathematical knots and linked structures rendered as tube meshes.",
    "number_theory": "Visualizations of number-theoretic patterns.",
    "parametric": "Parametric surfaces defined by explicit coordinate formulas.",
    "physics": "Physics simulations and field visualizations.",
    "procedural": (
        "Procedurally generated geometry from noise, L-systems, "
        "and reaction-diffusion."
    ),
    "surfaces": "Miscellaneous surface generators.",
}


def _load_notes() -> dict[str, Any]:
    """Load manual notes YAML, returning empty dict on missing file."""
    if not NOTES_PATH.is_file():
        logger.warning("Notes file not found at %s", NOTES_PATH)
        return {}
    return yaml.safe_load(NOTES_PATH.read_text(encoding="utf-8")) or {}


def _python_type_name(value: Any) -> str:
    """Return a short human-readable type name for a parameter value."""
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    return type(value).__name__


def _format_default(value: Any) -> str:
    """Format a default value for display in a markdown table."""
    if isinstance(value, float):
        formatted = f"{value:.6g}"
        return formatted
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, str):
        return f'`"{value}"`' if value else '`""`'
    if isinstance(value, list):
        return f"`{value}`"
    return str(value)


def _format_range(ranges: dict[str, dict[str, float]], param: str) -> str:
    """Format the range column for a parameter."""
    if param not in ranges:
        return "—"
    r = ranges[param]
    lo = r.get("min", "")
    hi = r.get("max", "")
    step = r.get("step", "")
    parts = f"{lo}–{hi}"
    if step:
        parts += f" (step {step})"
    return parts


def _build_toc(categories: list[str], gen_by_cat: dict[str, list[str]]) -> str:
    """Build a table of contents grouped by category with links."""
    lines: list[str] = ["## Table of Contents", ""]
    for cat in categories:
        heading = CATEGORY_HEADINGS.get(cat, cat.replace("_", " ").title())
        anchor = heading.lower().replace(" ", "-").replace("–", "")
        lines.append(f"- **[{heading}](#{anchor})**")
        for name in gen_by_cat[cat]:
            lines.append(f"  - [{name}](#{name})")
    lines.append("")
    return "\n".join(lines)


def _build_generator_section(
    meta: GeneratorMeta,
    notes: dict[str, Any],
) -> str:
    """Build the markdown section for a single generator."""
    inst = meta.generator_class.create(meta.name)
    params = inst.get_default_params()
    ranges = inst.get_param_ranges()
    resolution = inst.get_default_resolution()
    representation = inst.get_default_representation()
    gen_notes = notes.get(meta.name, {})

    lines: list[str] = [f"### {meta.name}", ""]

    # Thumbnail image
    thumb_path = IMAGES_DIR / f"{meta.name}.png"
    lines.append(f"![{meta.name}]({thumb_path})")
    lines.append("")

    # Description from registry
    lines.append(meta.description)
    lines.append("")

    # Math description from notes
    math_desc = gen_notes.get("math_description", "")
    if math_desc:
        lines.append(math_desc.strip())
        lines.append("")

    # Aliases
    if meta.aliases:
        lines.append(f"Aliases: {', '.join(f'`{a}`' for a in meta.aliases)}")
        lines.append("")

    # Representation
    lines.append(
        f"Recommended representation: **{representation.type.value}**"
    )
    lines.append("")

    # Seed behavior
    seed_behavior = gen_notes.get("seed_behavior", "")
    if seed_behavior:
        lines.append(f"Seed behavior: **{seed_behavior}**")
        lines.append("")

    # Performance notes
    perf_notes = gen_notes.get("performance_notes", "")
    if perf_notes:
        lines.append(f"Performance: {perf_notes}")
        lines.append("")

    # Parameter table
    if params:
        lines.append("**Parameters:**")
        lines.append("")
        lines.append("| Parameter | Type | Default | Range | Description |")
        lines.append("|---|---|---|---|---|")
        for pname, pvalue in params.items():
            ptype = _python_type_name(pvalue)
            pdefault = _format_default(pvalue)
            prange = _format_range(ranges, pname)
            pdesc = _get_param_description(meta, pname)
            lines.append(
                f"| `{pname}` | {ptype} | {pdefault} | {prange} | {pdesc} |"
            )
        lines.append("")
    else:
        lines.append("No additional parameters.")
        lines.append("")

    # Resolution parameters
    if meta.resolution_params:
        lines.append("**Resolution parameters:**")
        lines.append("")
        lines.append("| Parameter | Default | Description |")
        lines.append("|---|---|---|")
        for rname, rdesc in meta.resolution_params.items():
            rdefault = resolution.get(rname, "—")
            lines.append(f"| `{rname}` | {rdefault} | {rdesc} |")
        lines.append("")

    # Example configurations
    examples = gen_notes.get("examples", [])
    if examples:
        lines.append("**Example configurations:**")
        lines.append("")
        for ex in examples:
            desc = ex.get("description", "")
            ex_params = ex.get("params", {})
            param_strs = " ".join(
                f"--param {k}={v}" for k, v in ex_params.items()
            )
            lines.append(f"- {desc}")
            lines.append(f"  ```bash")
            lines.append(
                f"  mathviz generate {meta.name} {param_strs} "
                f"--output {meta.name}.ply"
            )
            lines.append(f"  ```")
        lines.append("")
    else:
        # Default example command
        lines.append("```bash")
        lines.append(
            f"mathviz generate {meta.name} --output {meta.name}.ply"
        )
        lines.append("```")
        lines.append("")

    return "\n".join(lines)


def _get_param_description(meta: GeneratorMeta, param_name: str) -> str:
    """Get a description for a parameter from the schema or a fallback."""
    inst = meta.generator_class.create(meta.name)
    schema = inst.get_param_schema()
    if schema and param_name in schema:
        entry = schema[param_name]
        if isinstance(entry, dict) and "description" in entry:
            return entry["description"]
    # Fallback: capitalize and humanize the parameter name
    return param_name.replace("_", " ").capitalize()


def generate_docs() -> str:
    """Generate the full generators.md content."""
    notes = _load_notes()
    all_generators = list_generators()

    # Group by category
    gen_by_cat: dict[str, list[GeneratorMeta]] = {}
    for meta in all_generators:
        gen_by_cat.setdefault(meta.category, []).append(meta)

    # Sort generators within each category by name
    for cat in gen_by_cat:
        gen_by_cat[cat].sort(key=lambda m: m.name)

    # Sort categories by display name
    sorted_cats = sorted(
        gen_by_cat.keys(),
        key=lambda c: CATEGORY_HEADINGS.get(c, c),
    )

    total = len(all_generators)
    cat_count = len(sorted_cats)

    # Build name-only map for TOC
    name_by_cat = {
        cat: [m.name for m in gen_by_cat[cat]] for cat in sorted_cats
    }

    parts: list[str] = []

    # Header
    parts.append("# Generators")
    parts.append("")
    parts.append(
        f"MathViz includes {total} generators across {cat_count} categories. "
        f"Each generator produces a deterministic 3D mathematical form from "
        f"a seed and a set of parameters."
    )
    parts.append("")

    # TOC
    parts.append(_build_toc(sorted_cats, name_by_cat))

    # Generator sections by category
    for cat in sorted_cats:
        heading = CATEGORY_HEADINGS.get(cat, cat.replace("_", " ").title())
        blurb = CATEGORY_BLURBS.get(cat, "")

        parts.append(f"## {heading}")
        parts.append("")
        if blurb:
            parts.append(blurb)
            parts.append("")

        for meta in gen_by_cat[cat]:
            parts.append(_build_generator_section(meta, notes))

    result = "\n".join(parts)
    # Ensure single trailing newline
    return result.rstrip("\n") + "\n"


def main() -> None:
    """Entry point for the script."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(
        description="Generate docs/generators.md from the registry"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if the file is up to date (exit 1 if not)",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print to stdout instead of writing file",
    )
    args = parser.parse_args()

    content = generate_docs()

    if args.stdout:
        sys.stdout.write(content)
        return

    if args.check:
        if not OUTPUT_PATH.is_file():
            logger.error("docs/generators.md does not exist")
            sys.exit(1)
        existing = OUTPUT_PATH.read_text(encoding="utf-8")
        if existing != content:
            logger.error("docs/generators.md is out of date")
            sys.exit(1)
        logger.info("docs/generators.md is up to date")
        return

    OUTPUT_PATH.write_text(content, encoding="utf-8")
    logger.info("Wrote %s (%d bytes)", OUTPUT_PATH, len(content))


if __name__ == "__main__":
    main()
