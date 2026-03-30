"""Tests for the static demo page (demo.html)."""

import importlib.resources
from pathlib import Path

import pytest

DEMO_JS_MODULES = [
    "demo-scene.js",
    "demo-display.js",
    "demo-controls.js",
    "demo-render.js",
    "demo-crystal.js",
    "demo-export.js",
    "demo-gallery.js",
]


@pytest.fixture()
def demo_html() -> str:
    """Read demo.html content."""
    static = importlib.resources.files("mathviz").joinpath("static")
    return static.joinpath("demo.html").read_text(encoding="utf-8")


@pytest.fixture()
def demo_js_files() -> dict[str, str]:
    """Read all demo JS module contents."""
    static = importlib.resources.files("mathviz").joinpath("static")
    result = {}
    for name in DEMO_JS_MODULES:
        result[name] = static.joinpath(name).read_text(encoding="utf-8")
    return result


def test_demo_html_exists() -> None:
    """demo.html is accessible via importlib.resources."""
    static = importlib.resources.files("mathviz").joinpath("static")
    demo = static.joinpath("demo.html")
    content = demo.read_text(encoding="utf-8")
    assert len(content) > 0
    assert "<!DOCTYPE html>" in content


@pytest.mark.parametrize("api_path", ["/api/generate", "/api/generators", "/api/snapshots"])
def test_demo_no_api_references(
    demo_html: str, demo_js_files: dict[str, str], api_path: str
) -> None:
    """demo.html and JS modules contain no references to backend API paths."""
    assert api_path not in demo_html
    for name, content in demo_js_files.items():
        assert api_path not in content, f"{name} contains {api_path}"


def test_demo_contains_threejs_import(demo_html: str) -> None:
    """demo.html imports Three.js."""
    assert "three" in demo_html.lower()
    assert "importmap" in demo_html


def test_demo_contains_all_view_modes(demo_html: str) -> None:
    """demo.html includes all 8 view mode options."""
    for mode in [
        "shaded", "wireframe", "vertex", "dense",
        "edge_cloud", "surface", "crystal", "colormap",
    ]:
        assert f'value="{mode}"' in demo_html, f"Missing view mode: {mode}"


def test_demo_contains_visualization_selector(demo_html: str) -> None:
    """demo.html has a visualization selector dropdown."""
    assert 'id="viz-selector"' in demo_html


def test_demo_has_manifest_loading(demo_js_files: dict[str, str]) -> None:
    """demo-scene.js loads manifest.json."""
    assert "manifest.json" in demo_js_files["demo-scene.js"]


def test_demo_loads_glb_from_relative_paths(
    demo_js_files: dict[str, str],
) -> None:
    """Gallery module resolves geometry from relative paths like ./data/{name}/mesh.glb."""
    gallery_js = demo_js_files["demo-gallery.js"]
    assert "/mesh.glb" in gallery_js
    assert "/cloud.ply" in gallery_js


def test_demo_has_no_backend_ui_elements(demo_html: str) -> None:
    """demo.html does not include backend-dependent UI elements."""
    # No generator search input
    assert 'id="generator-search"' not in demo_html
    # No seed input
    assert 'id="seed-input"' not in demo_html
    # No parameter panel
    assert 'id="param-panel"' not in demo_html
    # No container editor
    assert 'id="container-panel"' not in demo_html
    # No save/load buttons
    assert 'id="save-btn"' not in demo_html
    assert 'id="load-btn"' not in demo_html
    # No snapshot gallery
    assert 'id="snapshot-gallery"' not in demo_html
    # No regenerate button
    assert 'id="regenerate-btn"' not in demo_html
    # No compare mode
    assert 'id="compare-mode"' not in demo_html


def test_demo_retains_client_side_features(demo_html: str) -> None:
    """demo.html retains all client-side rendering features."""
    assert 'id="point-size"' in demo_html
    assert 'id="density-slider"' in demo_html
    assert 'id="show-bbox"' in demo_html
    assert 'id="show-axes"' in demo_html
    assert 'id="turntable-toggle"' in demo_html
    assert 'id="screenshot-btn"' in demo_html
    assert 'id="lock-camera"' in demo_html
    assert 'id="stretch-panel"' in demo_html
    assert 'id="crystal-controls"' in demo_html
    assert 'id="colormap-controls"' in demo_html


def test_demo_js_modules_exist() -> None:
    """All required JS modules for demo.html exist."""
    static = importlib.resources.files("mathviz").joinpath("static")
    for name in DEMO_JS_MODULES:
        content = static.joinpath(name).read_text(encoding="utf-8")
        assert len(content) > 0, f"{name} is empty"
