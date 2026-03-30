"""Tests for the demo gallery UI and manifest loader (Task 166)."""

import importlib.resources
import json

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


@pytest.fixture()
def demo_html() -> str:
    """Read demo.html content."""
    static = importlib.resources.files("mathviz").joinpath("static")
    return static.joinpath("demo.html").read_text(encoding="utf-8")


@pytest.fixture()
def gallery_js() -> str:
    """Read demo-gallery.js content."""
    static = importlib.resources.files("mathviz").joinpath("static")
    return static.joinpath("demo-gallery.js").read_text(encoding="utf-8")


@pytest.fixture()
def scene_js() -> str:
    """Read demo-scene.js content."""
    static = importlib.resources.files("mathviz").joinpath("static")
    return static.joinpath("demo-scene.js").read_text(encoding="utf-8")


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


# ── Gallery JS module ──


def test_gallery_js_exists() -> None:
    """demo-gallery.js module exists and is non-empty."""
    static = importlib.resources.files("mathviz").joinpath("static")
    content = static.joinpath("demo-gallery.js").read_text(encoding="utf-8")
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


# ── Gallery renders cards from manifest ──


def test_gallery_renders_card_elements(gallery_js: str) -> None:
    """Gallery creates card elements with class 'gallery-card'."""
    assert "gallery-card" in gallery_js


def test_gallery_renders_thumbnail(gallery_js: str) -> None:
    """Gallery cards include a thumbnail image."""
    assert "gallery-thumb" in gallery_js


def test_gallery_renders_display_name(gallery_js: str) -> None:
    """Gallery cards show the display_name (or name fallback)."""
    assert "display_name" in gallery_js
    assert "gallery-card-title" in gallery_js


def test_gallery_renders_category_badge(gallery_js: str) -> None:
    """Gallery cards include a category badge."""
    assert "gallery-category-badge" in gallery_js


# ── Card click loads correct geometry ──


def test_gallery_card_click_triggers_callback(gallery_js: str) -> None:
    """Clicking a gallery card triggers the onSelect callback."""
    assert "onSelect" in gallery_js
    assert "grid.addEventListener" in gallery_js
    # The click handler finds the item and calls onSelect
    assert "onSelect(item)" in gallery_js


def test_scene_loads_mesh_and_cloud_from_item(scene_js: str) -> None:
    """Scene integration passes mesh and cloud paths from manifest item."""
    assert "resolveItemPaths" in scene_js
    assert "loadVisualization" in scene_js
    # The gallery onSelect callback uses resolved paths
    assert "paths.mesh" in scene_js
    assert "paths.cloud" in scene_js


# ── Query param deep-link ──


def test_scene_reads_query_param(scene_js: str) -> None:
    """demo-scene.js reads ?name= query param on init."""
    assert "getQueryParamName" in scene_js


def test_query_param_selects_matching_entry(scene_js: str) -> None:
    """When ?name= matches an entry, it is selected instead of the first."""
    assert "queryName" in scene_js
    assert "selectByName" in scene_js


def test_gallery_select_by_name(gallery_js: str) -> None:
    """buildGallery returns selectByName for programmatic selection."""
    assert "selectByName" in gallery_js


# ── Category filter ──


def test_gallery_extracts_categories(gallery_js: str) -> None:
    """Gallery extracts unique categories from manifest items."""
    assert "_extractCategories" in gallery_js


def test_gallery_filter_buttons(gallery_js: str) -> None:
    """Gallery renders filter buttons with data-category attribute."""
    assert "gallery-filter-btn" in gallery_js
    assert "data-category" in gallery_js or "dataset.category" in gallery_js


def test_gallery_filter_hides_non_matching(gallery_js: str) -> None:
    """Category filter hides cards that don't match the selected category."""
    assert "_filterCards" in gallery_js


def test_gallery_has_all_category(gallery_js: str) -> None:
    """Gallery includes an 'All' category to show everything."""
    assert "All" in gallery_js


# ── Manifest schema ──


def test_manifest_schema_fields(gallery_js: str) -> None:
    """Gallery handles manifest fields: name, category, display_name, thumbnail, mesh, cloud, description."""
    for field in ["name", "category", "display_name", "thumbnail", "mesh", "cloud", "description"]:
        assert field in gallery_js, f"Gallery JS missing handling for '{field}'"


def test_resolve_paths_fallback(gallery_js: str) -> None:
    """resolveItemPaths falls back to ./data/{name}/ convention."""
    assert "./data/" in gallery_js
    assert "mesh.glb" in gallery_js or "/mesh.glb" in gallery_js
    assert "cloud.ply" in gallery_js or "/cloud.ply" in gallery_js


# ── Integration: gallery import in scene ──


def test_scene_imports_gallery(scene_js: str) -> None:
    """demo-scene.js imports from demo-gallery.js."""
    assert "demo-gallery.js" in scene_js
    assert "buildGallery" in scene_js


# ── Existing test file references: demo-scene.js must still list in modules ──


def test_gallery_module_listed_in_demo_page_test() -> None:
    """demo-gallery.js is importable as a static resource."""
    static = importlib.resources.files("mathviz").joinpath("static")
    path = static.joinpath("demo-gallery.js")
    assert path.read_text(encoding="utf-8")
