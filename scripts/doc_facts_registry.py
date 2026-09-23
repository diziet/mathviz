"""The facts this repo's docs state, and the source each value is computed from.

scripts/doc_facts.py rewrites `<!-- fact:NAME -->VALUE<!-- /fact -->` markers in
tracked .md files from these functions. To add a fact, register it here, wrap the
value in a doc with a marker, and run `make doc-facts`. The porting guide is
docs/doc-checks.md in the llm-reliability-benchmark repository.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from doc_facts_sources import Fact, FactError, read_text, regex_group

if TYPE_CHECKING:
    from pathlib import Path

RENDER_BATCH = "src/mathviz/cli_render_batch.py"
RENDERER = "src/mathviz/preview/renderer.py"
THUMBNAILS = "src/mathviz/preview/thumbnails.py"
CLI_THUMBNAIL = "src/mathviz/cli_thumbnail.py"
PREVIEW_PAGE = "src/mathviz/static/index.html"
QUOTED = re.compile(r'"([^"]+)"')
VIEW_MODE_SELECT = re.compile(r'<select id="view-mode">(.*?)</select>', re.DOTALL)
OPTION_LABEL = re.compile(r"<option value=\"[^\"]*\"[^>]*>([^<]+)</option>")


def _quoted_list(root: Path, relative: str, pattern: str) -> str:
    """Return the quoted strings inside group 1 of `pattern`, joined with commas."""
    values = QUOTED.findall(regex_group(root, relative, pattern))
    if not values:
        raise FactError(f"{relative}: {pattern!r} matched no quoted values")
    return ", ".join(values)


def render_all_style_default(root: Path) -> str:
    """Return the default of `render-all --style`."""
    return regex_group(
        root, RENDER_BATCH, r'style: str = typer\.Option\(\s*"([^"]+)",\s*"--style"'
    )


def render_styles(root: Path) -> str:
    """Return the styles the `RenderStyle` Literal allows, in declaration order."""
    return _quoted_list(root, RENDERER, r"^RenderStyle = Literal\[(.+)\]$")


def thumbnail_view_mode_default(root: Path) -> str:
    """Return DEFAULT_VIEW_MODE, after checking `render-thumbnail --view-mode` uses it."""
    regex_group(root, CLI_THUMBNAIL, r'(DEFAULT_VIEW_MODE),\s*"--view-mode"')
    return regex_group(root, THUMBNAILS, r'^DEFAULT_VIEW_MODE: \w+ = "([^"]+)"$')


def thumbnail_view_modes(root: Path) -> str:
    """Return VALID_VIEW_MODES, the modes `render-thumbnail` accepts."""
    return _quoted_list(root, THUMBNAILS, r"^VALID_VIEW_MODES: .* = \((.+)\)$")


def _preview_view_mode_labels(root: Path) -> list[str]:
    """Return the option labels of the preview page's View Mode dropdown."""
    select = VIEW_MODE_SELECT.search(read_text(root, PREVIEW_PAGE))
    labels = OPTION_LABEL.findall(select.group(1)) if select else []
    if not labels:
        raise FactError(f'{PREVIEW_PAGE} has no <select id="view-mode"> with options')
    return [label.strip() for label in labels]


def preview_view_mode_count(root: Path) -> int:
    """Return how many modes the preview View Mode dropdown offers."""
    return len(_preview_view_mode_labels(root))


def preview_view_modes(root: Path) -> str:
    """Return the preview View Mode labels in dropdown order."""
    return ", ".join(_preview_view_mode_labels(root))


FACTS: dict[str, Fact] = {
    "render-all-style-default": Fact(
        f'the `"--style"` Option default in {RENDER_BATCH}', render_all_style_default
    ),
    "render-styles": Fact(f"the `RenderStyle` Literal in {RENDERER}", render_styles),
    "thumbnail-view-mode-default": Fact(
        f"`DEFAULT_VIEW_MODE` in {THUMBNAILS}, passed to --view-mode in {CLI_THUMBNAIL}",
        thumbnail_view_mode_default,
    ),
    "thumbnail-view-modes": Fact(
        f"`VALID_VIEW_MODES` in {THUMBNAILS}", thumbnail_view_modes
    ),
    "preview-view-mode-count": Fact(
        f'options of <select id="view-mode"> in {PREVIEW_PAGE}', preview_view_mode_count
    ),
    "preview-view-modes": Fact(
        f'option labels of <select id="view-mode"> in {PREVIEW_PAGE}', preview_view_modes
    ),
}
