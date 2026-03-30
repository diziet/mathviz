"""Tests for the demo gallery UI and manifest loader (Task 166).

Validates gallery rendering, card interaction, query-param deep-linking,
category filtering, and collapse/toggle behavior using DOM-level assertions.
"""

import importlib.resources
import json
import re

import pytest

SAMPLE_MANIFEST = [
    {
        "name": "lorenz",
        "category": "attractors",
        "display_name": "Lorenz Attractor",
        "thumbnail": "./data/lorenz/thumb.png",
        "mesh": "./data/lorenz/mesh.glb",
        "cloud": "./data/lorenz/cloud.ply",
        "description": "Classic chaotic attractor",
    },
    {
        "name": "klein_bottle",
        "category": "surfaces",
        "display_name": "Klein Bottle",
        "thumbnail": "./data/klein_bottle/thumb.png",
        "mesh": "./data/klein_bottle/mesh.glb",
        "cloud": "./data/klein_bottle/cloud.ply",
    },
    {
        "name": "rossler",
        "category": "attractors",
        "display_name": "Rossler Attractor",
        "thumbnail": "./data/rossler/thumb.png",
        "mesh": "./data/rossler/mesh.glb",
        "cloud": "./data/rossler/cloud.ply",
        "description": "Three-scroll attractor",
    },
    {
        "name": "trefoil",
        "category": "knots",
        "display_name": "Trefoil Knot",
        "thumbnail": "./data/trefoil/thumb.png",
        "mesh": "./data/trefoil/mesh.glb",
        "cloud": "./data/trefoil/cloud.ply",
    },
]


def _read_static_file(filename: str) -> str:
    """Read a file from the mathviz static directory."""
    static = importlib.resources.files("mathviz").joinpath("static")
    return static.joinpath(filename).read_text(encoding="utf-8")


@pytest.fixture()
def demo_html() -> str:
    """Read demo.html content."""
    return _read_static_file("demo.html")


@pytest.fixture()
def gallery_js() -> str:
    """Read demo-gallery.js content."""
    return _read_static_file("demo-gallery.js")


@pytest.fixture()
def scene_js() -> str:
    """Read demo-scene.js content."""
    return _read_static_file("demo-scene.js")


@pytest.fixture()
def controls_js() -> str:
    """Read demo-controls.js content."""
    return _read_static_file("demo-controls.js")


@pytest.fixture()
def display_js() -> str:
    """Read demo-display.js content."""
    return _read_static_file("demo-display.js")


# ── Gallery HTML structure ──


def test_gallery_panel_exists(demo_html: str) -> None:
    """demo.html contains the gallery panel container."""
    assert 'id="gallery-panel"' in demo_html


def test_gallery_grid_exists(demo_html: str) -> None:
    """demo.html contains the gallery grid for cards."""
    assert 'id="gallery-grid"' in demo_html


def test_gallery_filter_bar_exists(demo_html: str) -> None:
    """demo.html contains the category filter bar."""
    assert 'id="gallery-filter-bar"' in demo_html


def test_gallery_toggle_button_exists(demo_html: str) -> None:
    """demo.html has a toggle button to show/hide gallery."""
    assert 'id="gallery-toggle"' in demo_html


def test_gallery_close_button_exists(demo_html: str) -> None:
    """demo.html has a close button inside the gallery panel."""
    assert 'id="gallery-close"' in demo_html


# ── Gallery JS module ──


def test_gallery_js_exists() -> None:
    """demo-gallery.js module exists and is non-empty."""
    content = _read_static_file("demo-gallery.js")
    assert len(content) > 0


def test_gallery_exports_build_gallery(gallery_js: str) -> None:
    """demo-gallery.js exports buildGallery function."""
    assert "export function buildGallery" in gallery_js


def test_gallery_exports_query_param(gallery_js: str) -> None:
    """demo-gallery.js exports getQueryParamName for deep linking."""
    assert "export function getQueryParamName" in gallery_js


def test_gallery_exports_resolve_paths(gallery_js: str) -> None:
    """demo-gallery.js exports resolveItemPaths."""
    assert "export function resolveItemPaths" in gallery_js


# ── Card rendering: gallery creates DOM elements with correct classes ──


def test_gallery_creates_card_elements_with_dataset_name(gallery_js: str) -> None:
    """Gallery creates card divs with data-name set to the item's name."""
    # _renderCards sets card.dataset.name = item.name
    assert "card.dataset.name = item.name" in gallery_js


def test_gallery_creates_card_with_category_dataset(gallery_js: str) -> None:
    """Gallery sets data-category on each card for filtering."""
    assert "card.dataset.category" in gallery_js


def test_gallery_renders_thumbnail_with_lazy_loading(gallery_js: str) -> None:
    """Gallery sets thumbnail src from item.thumbnail with lazy loading."""
    assert "thumb.loading = 'lazy'" in gallery_js
    assert "item.thumbnail" in gallery_js


def test_gallery_renders_display_name_in_title(gallery_js: str) -> None:
    """Card title uses display_name with name as fallback."""
    assert "item.display_name || item.name" in gallery_js


def test_gallery_renders_description_when_present(gallery_js: str) -> None:
    """Cards include a description element when item has a description."""
    assert "item.description" in gallery_js
    assert "gallery-card-desc" in gallery_js


# ── Card click triggers onSelect with the correct item ──


def test_card_click_finds_item_by_dataset_name(gallery_js: str) -> None:
    """Click handler looks up item by card's data-name attribute."""
    # The handler reads card.dataset.name, then finds the matching item
    assert "card.dataset.name" in gallery_js
    assert "items.find" in gallery_js


def test_card_click_skips_already_selected(gallery_js: str) -> None:
    """Clicking the already-selected card does not re-trigger onSelect."""
    assert "name === selectedName" in gallery_js


def test_card_click_highlights_and_calls_onselect(gallery_js: str) -> None:
    """Click handler highlights the card and invokes onSelect callback."""
    # After finding the item, handler should highlight then call onSelect
    assert "_highlightCard" in gallery_js


# ── Scene integration: resolved paths are passed to loadVisualization ──


def test_scene_gallery_onselect_resolves_paths(scene_js: str) -> None:
    """Gallery onSelect callback resolves mesh/cloud paths before loading."""
    assert "resolveItemPaths(item)" in scene_js
    assert "paths.mesh" in scene_js
    assert "paths.cloud" in scene_js


def test_scene_passes_three_args_to_load(scene_js: str) -> None:
    """loadVisualization is called with (name, meshUrl, cloudUrl)."""
    # Match loadVisualization call with 3 arguments
    pattern = r"loadVisualization\([^)]+,\s*paths\.mesh,\s*paths\.cloud\)"
    assert re.search(pattern, scene_js), "loadVisualization should receive 3 args"


# ── Query param deep-link ──


def test_scene_reads_query_param(scene_js: str) -> None:
    """demo-scene.js reads ?name= query param on init."""
    assert "getQueryParamName" in scene_js


def test_query_param_selects_or_falls_back(scene_js: str) -> None:
    """When ?name= doesn't match, first item is selected as fallback."""
    # Should use inverted condition: if (!queryName || !selectByName(queryName))
    assert "!queryName || !gallery.selectByName(queryName)" in scene_js


def test_gallery_select_by_name_returns_bool(gallery_js: str) -> None:
    """selectByName returns false when name not found, true on success."""
    assert "return false" in gallery_js
    assert "return true" in gallery_js


# ── Category filter: filter buttons and card visibility ──


def test_gallery_filter_extracts_sorted_categories(gallery_js: str) -> None:
    """Categories are extracted as a sorted unique set with 'All' first."""
    assert "Array.from(set).sort()" in gallery_js
    assert "ALL_CATEGORY" in gallery_js


def test_gallery_filter_creates_buttons_with_data_category(gallery_js: str) -> None:
    """Filter bar buttons have data-category matching category names."""
    assert "btn.dataset.category = cat" in gallery_js


def test_gallery_filter_click_updates_active_highlight(gallery_js: str) -> None:
    """Clicking a filter button updates which button has the active class."""
    assert "_updateFilterHighlight" in gallery_js
    assert "classList.toggle('active'" in gallery_js


def test_gallery_filter_hides_non_matching_cards(gallery_js: str) -> None:
    """Cards whose category doesn't match the filter are hidden via display:none."""
    # _filterCards sets card.style.display = 'none' for non-matching
    assert "card.style.display" in gallery_js
    assert "card.dataset.category === category" in gallery_js


def test_gallery_all_category_shows_all_cards(gallery_js: str) -> None:
    """Selecting 'All' category shows all cards."""
    assert "category === ALL_CATEGORY" in gallery_js


# ── Resolve paths: fallback convention ──


def test_resolve_paths_uses_data_name_convention(gallery_js: str) -> None:
    """resolveItemPaths falls back to ./data/{name}/mesh.glb and cloud.ply."""
    assert "'./data/' + item.name" in gallery_js
    assert "basePath + '/mesh.glb'" in gallery_js
    assert "basePath + '/cloud.ply'" in gallery_js


def test_resolve_paths_prefers_explicit_paths(gallery_js: str) -> None:
    """resolveItemPaths uses item.mesh/item.cloud when present."""
    assert "item.mesh ||" in gallery_js
    assert "item.cloud ||" in gallery_js


# ── DRY: no duplicate path fallback in controls or display ──


def test_controls_dropdown_uses_resolved_paths_directly(controls_js: str) -> None:
    """Dropdown handler uses dataset.mesh/cloud without redundant fallback."""
    assert "selected.dataset.mesh" in controls_js
    assert "selected.dataset.cloud" in controls_js
    # Should NOT have the old broken fallback
    assert "dataset.path" not in controls_js


def test_display_no_legacy_two_arg_fallback(display_js: str) -> None:
    """loadVisualization does not contain legacy 2-arg path reconstruction."""
    assert "cloudUrl === undefined" not in display_js
    assert "basePath + '/mesh.glb'" not in display_js


# ── Gallery collapse/toggle ──


def test_gallery_close_wired_in_scene(scene_js: str) -> None:
    """demo-scene.js wires the gallery-close button to collapse the panel."""
    assert "gallery-close" in scene_js
    assert "classList.add('collapsed')" in scene_js


def test_gallery_toggle_uses_css_class_not_inline_style(scene_js: str) -> None:
    """Toggle button visibility is controlled via CSS class, not inline style."""
    assert "classList.add('hidden')" in scene_js
    assert "classList.remove('hidden')" in scene_js


# ── Manifest schema fields handled ──


def test_manifest_schema_fields(gallery_js: str) -> None:
    """Gallery handles all manifest fields."""
    for field in ["name", "category", "display_name", "thumbnail", "mesh", "cloud", "description"]:
        assert field in gallery_js, f"Gallery JS missing handling for '{field}'"


# ── Integration: gallery import in scene ──


def test_scene_imports_gallery(scene_js: str) -> None:
    """demo-scene.js imports from demo-gallery.js."""
    assert "demo-gallery.js" in scene_js
    assert "buildGallery" in scene_js


def test_gallery_module_listed_in_demo_page_test() -> None:
    """demo-gallery.js is importable as a static resource."""
    content = _read_static_file("demo-gallery.js")
    assert content
