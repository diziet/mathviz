"""Tests for expanded render view system (26 views + aliases)."""

import sys
from pathlib import Path
from typing import Iterator
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mathviz.core.math_object import MathObject, Mesh, PointCloud
from mathviz.preview.renderer import (
    VALID_VIEW_NAMES,
    RenderConfig,
    _VIEW_CAMERAS,
    get_view_camera,
    resolve_view_name,
)


def _sphere_mesh() -> MathObject:
    """Create a simple sphere-like mesh for testing."""
    phi = (1 + np.sqrt(5)) / 2
    raw_verts = np.array(
        [
            [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
            [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
            [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
        ],
        dtype=np.float64,
    )
    norms = np.linalg.norm(raw_verts, axis=1, keepdims=True)
    vertices = raw_verts / norms
    faces = np.array(
        [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
        ],
        dtype=np.int64,
    )
    return MathObject(mesh=Mesh(vertices=vertices, faces=faces), generator_name="test_sphere")


def _make_mock_pyvista() -> MagicMock:
    """Create a mock pyvista module with required classes and attributes."""
    mock_pv = MagicMock(spec=ModuleType)
    mock_pv.__name__ = "pyvista"
    mock_pv.PolyData = MagicMock(return_value=MagicMock())
    mock_plotter = MagicMock()
    mock_plotter.camera = MagicMock()
    mock_pv.Plotter = MagicMock(return_value=mock_plotter)
    mock_pv.Light = MagicMock(return_value=MagicMock())
    mock_pv.OFF_SCREEN = False
    return mock_pv


@pytest.fixture()
def pyvista_env(tmp_path: Path) -> Iterator[tuple[MagicMock, "ModuleType", Path]]:
    """Provide a mocked pyvista environment with (mock_plotter, renderer, tmp_path)."""
    mock_pv = _make_mock_pyvista()
    mock_plotter = mock_pv.Plotter.return_value
    mock_plotter.screenshot.side_effect = lambda p: Path(p).write_bytes(
        b"\x89PNG" + b"\x01" * 50
    )
    with patch.dict(sys.modules, {"pyvista": mock_pv}):
        from mathviz.preview import renderer

        yield mock_plotter, renderer, tmp_path


def _render_with_view(
    tmp_path: Path, view: str, use_2d: bool = False
) -> tuple[MagicMock, Path]:
    """Render a sphere with the given view, return (mock_plotter, output_path)."""
    mock_pv = _make_mock_pyvista()
    output_file = tmp_path / f"{view}.png"
    mock_plotter = mock_pv.Plotter.return_value
    mock_plotter.screenshot.side_effect = lambda p: Path(p).write_bytes(
        b"\x89PNG\r\n\x1a\n" + view.encode() + b"\x00" * 50
    )

    with patch.dict(sys.modules, {"pyvista": mock_pv}):
        from mathviz.preview import renderer

        if use_2d:
            renderer.render_2d_projection(_sphere_mesh(), output_file, view=view)
        else:
            renderer.render_to_png(_sphere_mesh(), output_file, view=view)

    return mock_plotter, output_file


class TestViewNameResolution:
    """Tests for view name resolution and validation."""

    def test_resolve_direct_view(self) -> None:
        """Direct view names resolve to themselves."""
        assert resolve_view_name("front") == "front"
        assert resolve_view_name("top") == "top"

    def test_resolve_alias_side(self) -> None:
        """'side' resolves to 'right'."""
        assert resolve_view_name("side") == "right"

    def test_resolve_alias_angle(self) -> None:
        """'angle' resolves to 'front-right-top'."""
        assert resolve_view_name("angle") == "front-right-top"

    def test_invalid_view_raises(self) -> None:
        """Invalid view name raises ValueError with available views."""
        with pytest.raises(ValueError, match="Invalid view name"):
            resolve_view_name("invalid-view")

    def test_invalid_view_lists_valid_names(self) -> None:
        """Error message lists valid view names."""
        with pytest.raises(ValueError, match="front"):
            resolve_view_name("nonexistent")

    def test_all_26_views_defined(self) -> None:
        """All 26 camera positions are defined (6 face + 12 edge + 8 vertex)."""
        assert len(_VIEW_CAMERAS) == 26

    def test_valid_view_names_includes_aliases(self) -> None:
        """VALID_VIEW_NAMES includes both direct views and aliases."""
        assert "side" in VALID_VIEW_NAMES
        assert "angle" in VALID_VIEW_NAMES
        assert "front" in VALID_VIEW_NAMES
        assert len(VALID_VIEW_NAMES) == 28  # 26 + 2 aliases


class TestRenderWithView:
    """Tests for render_to_png with --view flag."""

    def test_render_front_produces_png(self, tmp_path: Path) -> None:
        """render with --view front produces a non-trivial PNG."""
        _, output_file = _render_with_view(tmp_path, "front")
        assert output_file.exists()
        assert output_file.stat().st_size > 0

    def test_render_top_produces_png(self, tmp_path: Path) -> None:
        """render with --view top produces a non-trivial PNG."""
        _, output_file = _render_with_view(tmp_path, "top")
        assert output_file.exists()
        assert output_file.stat().st_size > 0

    def test_front_and_top_have_different_camera(self, tmp_path: Path) -> None:
        """--view front and --view top set different camera positions."""
        front_plotter, _ = _render_with_view(tmp_path, "front")
        top_plotter, _ = _render_with_view(tmp_path, "top")
        assert front_plotter.camera.position != top_plotter.camera.position

    def test_front_right_top_matches_angle(self, tmp_path: Path) -> None:
        """--view front-right-top and --view angle set same camera position."""
        frt_plotter, _ = _render_with_view(tmp_path, "front-right-top")
        angle_plotter, _ = _render_with_view(tmp_path, "angle")
        assert frt_plotter.camera.position == angle_plotter.camera.position
        assert frt_plotter.camera.up == angle_plotter.camera.up

    def test_default_view_is_front_right_top(self, pyvista_env: tuple) -> None:
        """Default render (no explicit view) uses front-right-top camera."""
        mock_plotter, renderer, tmp_path = pyvista_env
        output_file = tmp_path / "default.png"
        renderer.render_to_png(_sphere_mesh(), output_file)

        expected_pos, expected_up = get_view_camera("front-right-top")
        assert mock_plotter.camera.position == expected_pos
        assert mock_plotter.camera.up == expected_up

    def test_all_six_face_views_have_distinct_cameras(self, tmp_path: Path) -> None:
        """All 6 face-on views produce distinct camera positions."""
        face_views = ["front", "back", "left", "right", "top", "bottom"]
        positions = set()
        for view in face_views:
            plotter, _ = _render_with_view(tmp_path, view)
            positions.add(plotter.camera.position)
        assert len(positions) == 6, "Expected 6 distinct camera positions for face views"

    def test_edge_view_differs_from_face_views(self, tmp_path: Path) -> None:
        """Edge-on view front-right differs from front and right."""
        fr_plotter, _ = _render_with_view(tmp_path, "front-right")
        f_plotter, _ = _render_with_view(tmp_path, "front")
        r_plotter, _ = _render_with_view(tmp_path, "right")
        assert fr_plotter.camera.position != f_plotter.camera.position
        assert fr_plotter.camera.position != r_plotter.camera.position


class TestRender2dWithExpandedViews:
    """Tests for render-2d with expanded view names."""

    def test_render_2d_back_produces_png(self, tmp_path: Path) -> None:
        """render-2d with --view back produces a non-trivial PNG."""
        _, output_file = _render_with_view(tmp_path, "back", use_2d=True)
        assert output_file.exists()
        assert output_file.stat().st_size > 0

    def test_render_2d_right_produces_png(self, tmp_path: Path) -> None:
        """render-2d with --view right produces a non-trivial PNG."""
        _, output_file = _render_with_view(tmp_path, "right", use_2d=True)
        assert output_file.exists()
        assert output_file.stat().st_size > 0

    def test_render_2d_enables_parallel_projection(self, tmp_path: Path) -> None:
        """render-2d with new view names still uses parallel projection."""
        plotter, _ = _render_with_view(tmp_path, "back", use_2d=True)
        plotter.enable_parallel_projection.assert_called_once()

    def test_render_2d_side_alias_works(self, tmp_path: Path) -> None:
        """render-2d --view side still works via alias to right."""
        plotter, output_file = _render_with_view(tmp_path, "side", use_2d=True)
        assert output_file.exists()
        expected_pos, expected_up = get_view_camera("right")
        assert plotter.camera.position == expected_pos


class TestRenderAllViews:
    """Tests for --view all rendering multiple files."""

    def test_render_all_produces_multiple_files(self, pyvista_env: tuple) -> None:
        """--view all produces one file per named view."""
        mock_plotter, renderer, tmp_path = pyvista_env
        output_file = tmp_path / "torus.png"
        paths = renderer.render_all_views(_sphere_mesh(), output_file)

        assert len(paths) == 26
        for p in paths:
            assert p.exists()

    def test_render_all_names_files_with_view_suffix(self, pyvista_env: tuple) -> None:
        """--view all names files with view name appended to stem."""
        mock_plotter, renderer, tmp_path = pyvista_env
        output_file = tmp_path / "torus.png"
        paths = renderer.render_all_views(_sphere_mesh(), output_file)

        filenames = {p.name for p in paths}
        assert "torus_front.png" in filenames
        assert "torus_top.png" in filenames
        assert "torus_front-right-top.png" in filenames

    def test_render_all_2d_uses_parallel_projection(self, pyvista_env: tuple) -> None:
        """--view all with use_2d=True uses parallel projection."""
        mock_plotter, renderer, tmp_path = pyvista_env
        output_file = tmp_path / "torus.png"
        renderer.render_all_views(_sphere_mesh(), output_file, use_2d=True)

        mock_plotter.enable_parallel_projection.assert_called()


class TestCameraPositions:
    """Tests verifying camera direction vectors match task spec."""

    @pytest.mark.parametrize(
        "view,expected_pos",
        [
            ("front", (0, -1, 0)),
            ("back", (0, 1, 0)),
            ("left", (-1, 0, 0)),
            ("right", (1, 0, 0)),
            ("top", (0, 0, 1)),
            ("bottom", (0, 0, -1)),
            ("front-right", (1, -1, 0)),
            ("front-left", (-1, -1, 0)),
            ("front-right-top", (1, -1, 1)),
            ("back-left-bottom", (-1, 1, -1)),
        ],
    )
    def test_camera_position_matches_spec(
        self, view: str, expected_pos: tuple[float, ...]
    ) -> None:
        """Camera position direction vector matches task specification."""
        pos, _ = get_view_camera(view)
        assert pos == expected_pos
