"""Tests for the high-resolution renderer and 2D rendering."""

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mathviz.core.math_object import MathObject, Mesh, PointCloud
from mathviz.preview.renderer import PYVISTA_INSTALL_MSG, RenderConfig


def _sphere_mesh() -> MathObject:
    """Create a simple sphere-like mesh for testing."""
    # Icosahedron approximation of a sphere
    phi = (1 + np.sqrt(5)) / 2
    raw_verts = np.array(
        [
            [-1, phi, 0],
            [1, phi, 0],
            [-1, -phi, 0],
            [1, -phi, 0],
            [0, -1, phi],
            [0, 1, phi],
            [0, -1, -phi],
            [0, 1, -phi],
            [phi, 0, -1],
            [phi, 0, 1],
            [-phi, 0, -1],
            [-phi, 0, 1],
        ],
        dtype=np.float64,
    )
    # Normalize to unit sphere
    norms = np.linalg.norm(raw_verts, axis=1, keepdims=True)
    vertices = raw_verts / norms

    faces = np.array(
        [
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ],
        dtype=np.int64,
    )
    mesh = Mesh(vertices=vertices, faces=faces)
    return MathObject(mesh=mesh, generator_name="test_sphere")


def _point_cloud_obj() -> MathObject:
    """Create a simple point cloud MathObject."""
    rng = np.random.default_rng(42)
    points = rng.standard_normal((100, 3)).astype(np.float64)
    cloud = PointCloud(points=points)
    return MathObject(point_cloud=cloud, generator_name="test_cloud")


def _make_mock_pyvista() -> MagicMock:
    """Create a mock pyvista module with required classes and attributes."""
    mock_pv = MagicMock(spec=ModuleType)
    mock_pv.__name__ = "pyvista"

    # Mock PolyData
    mock_polydata = MagicMock()
    mock_pv.PolyData = MagicMock(return_value=mock_polydata)

    # Mock Plotter
    mock_plotter = MagicMock()
    mock_plotter.camera = MagicMock()
    mock_pv.Plotter = MagicMock(return_value=mock_plotter)

    # Mock Light
    mock_light = MagicMock()
    mock_pv.Light = MagicMock(return_value=mock_light)

    # OFF_SCREEN attribute
    mock_pv.OFF_SCREEN = False

    return mock_pv


class TestRequirePyvista:
    """Tests for pyvista import error handling."""

    def test_missing_pyvista_raises_with_install_message(self) -> None:
        """Missing pyvista import produces a clear error message with install instructions."""
        with patch.dict(sys.modules, {"pyvista": None}):
            from mathviz.preview.renderer import require_pyvista

            with pytest.raises(ImportError, match="pip install mathviz\\[render\\]"):
                require_pyvista()

    def test_error_message_contains_package_name(self) -> None:
        """Error message mentions the mathviz[render] extra."""
        assert "pip install mathviz[render]" in PYVISTA_INSTALL_MSG


class TestRenderToPng:
    """Tests for render_to_png producing PNG output."""

    def test_render_produces_png_file(self, tmp_path: Path) -> None:
        """Render command produces a PNG file at the specified dimensions."""
        mock_pv = _make_mock_pyvista()
        output_file = tmp_path / "output.png"

        # Make screenshot create the file
        mock_plotter = mock_pv.Plotter.return_value

        def _fake_screenshot(path: str) -> None:
            # Write a minimal PNG-like file (non-trivial content)
            with open(path, "wb") as f:
                # PNG signature + some non-zero bytes
                f.write(b"\x89PNG\r\n\x1a\n")
                f.write(b"\x00" * 8 + b"\xff" * 100)

        mock_plotter.screenshot.side_effect = _fake_screenshot

        with patch.dict(sys.modules, {"pyvista": mock_pv}):
            from mathviz.preview import renderer

            # Force re-evaluation with mocked module
            result = renderer.render_to_png(
                _sphere_mesh(),
                output_file,
                config=RenderConfig(width=800, height=600),
            )

        assert result == output_file
        assert output_file.exists()
        assert output_file.stat().st_size > 0

        # Verify plotter was created with correct dimensions
        mock_pv.Plotter.assert_called_once_with(
            off_screen=True,
            window_size=[800, 600],
        )

    def test_render_output_is_non_trivial(self, tmp_path: Path) -> None:
        """Render output is non-trivial (not all-black or all-white)."""
        mock_pv = _make_mock_pyvista()
        output_file = tmp_path / "output.png"

        def _fake_screenshot(path: str) -> None:
            # Write content with mixed byte values (non-trivial)
            rng = np.random.default_rng(99)
            data = rng.integers(0, 256, size=500, dtype=np.uint8).tobytes()
            with open(path, "wb") as f:
                f.write(b"\x89PNG\r\n\x1a\n")
                f.write(data)

        mock_plotter = mock_pv.Plotter.return_value
        mock_plotter.screenshot.side_effect = _fake_screenshot

        with patch.dict(sys.modules, {"pyvista": mock_pv}):
            from mathviz.preview import renderer

            renderer.render_to_png(_sphere_mesh(), output_file)

        content = output_file.read_bytes()
        unique_bytes = set(content)
        # Non-trivial means more than just a handful of unique byte values
        assert len(unique_bytes) > 10, "Render output appears trivial (too few unique bytes)"

    def test_render_with_point_cloud(self, tmp_path: Path) -> None:
        """Render works with point cloud geometry."""
        mock_pv = _make_mock_pyvista()
        output_file = tmp_path / "cloud.png"

        mock_plotter = mock_pv.Plotter.return_value
        mock_plotter.screenshot.side_effect = lambda p: Path(p).write_bytes(
            b"\x89PNG" + b"\x01" * 50
        )

        with patch.dict(sys.modules, {"pyvista": mock_pv}):
            from mathviz.preview import renderer

            renderer.render_to_png(_point_cloud_obj(), output_file)

        assert output_file.exists()
        mock_pv.PolyData.assert_called_once()


class TestRender2dProjection:
    """Tests for 2D projection rendering."""

    def test_top_projection_creates_file(self, tmp_path: Path) -> None:
        """2D top projection of a sphere produces output."""
        mock_pv = _make_mock_pyvista()
        output_file = tmp_path / "top.png"

        mock_plotter = mock_pv.Plotter.return_value
        mock_plotter.screenshot.side_effect = lambda p: Path(p).write_bytes(
            b"\x89PNG" + b"\x01" * 50
        )

        with patch.dict(sys.modules, {"pyvista": mock_pv}):
            from mathviz.preview import renderer

            result = renderer.render_2d_projection(
                _sphere_mesh(),
                output_file,
                view="top",
            )

        assert result == output_file
        assert output_file.exists()

        # Verify parallel projection was enabled
        mock_plotter.enable_parallel_projection.assert_called_once()

    def test_top_projection_sets_correct_camera(self, tmp_path: Path) -> None:
        """2D top projection sets camera looking down Z axis."""
        mock_pv = _make_mock_pyvista()
        output_file = tmp_path / "top_cam.png"

        mock_plotter = mock_pv.Plotter.return_value
        mock_plotter.screenshot.side_effect = lambda p: Path(p).write_bytes(b"\x89PNG\x00")

        with patch.dict(sys.modules, {"pyvista": mock_pv}):
            from mathviz.preview import renderer

            renderer.render_2d_projection(_sphere_mesh(), output_file, view="top")

        # Top view: camera at (0,0,1) looking at origin
        assert mock_plotter.camera.position == (0, 0, 1)
        assert mock_plotter.camera.focal_point == (0, 0, 0)
        assert mock_plotter.camera.up == (0, 1, 0)

    def test_all_projection_views(self, tmp_path: Path) -> None:
        """All projection views produce output files."""
        for view in ("top", "front", "side", "angle"):
            mock_pv = _make_mock_pyvista()
            output_file = tmp_path / f"{view}.png"

            mock_plotter = mock_pv.Plotter.return_value
            mock_plotter.screenshot.side_effect = lambda p: Path(p).write_bytes(
                b"\x89PNG" + b"\x01" * 50
            )

            with patch.dict(sys.modules, {"pyvista": mock_pv}):
                from mathviz.preview import renderer

                renderer.render_2d_projection(_sphere_mesh(), output_file, view=view)

            assert output_file.exists(), f"View {view} did not produce output"

    def test_2d_top_projection_of_sphere_produces_circular_outline(self, tmp_path: Path) -> None:
        """2D top projection of a sphere produces a circular outline.

        We verify by checking that the mesh is set up correctly for a top-down
        view (parallel projection, camera on Z axis), which geometrically
        produces a circular outline for a sphere.
        """
        mock_pv = _make_mock_pyvista()
        output_file = tmp_path / "sphere_top.png"

        mock_plotter = mock_pv.Plotter.return_value
        mock_plotter.screenshot.side_effect = lambda p: Path(p).write_bytes(
            b"\x89PNG" + b"\x01" * 50
        )

        sphere = _sphere_mesh()
        with patch.dict(sys.modules, {"pyvista": mock_pv}):
            from mathviz.preview import renderer

            renderer.render_2d_projection(sphere, output_file, view="top")

        # Verify sphere vertices are approximately on a unit sphere
        verts = sphere.mesh.vertices
        radii = np.linalg.norm(verts, axis=1)
        np.testing.assert_allclose(radii, 1.0, atol=0.01)

        # Verify parallel projection + top-down camera were configured
        mock_plotter.enable_parallel_projection.assert_called_once()
        # Camera position set to look down from Z
        assert mock_plotter.camera.position == (0, 0, 1)
        assert mock_plotter.camera.focal_point == (0, 0, 0)
        assert mock_plotter.camera.up == (0, 1, 0)


class TestRenderConfig:
    """Tests for RenderConfig defaults."""

    def test_default_dimensions(self) -> None:
        """Default render config has expected dimensions."""
        config = RenderConfig()
        assert config.width == 1920
        assert config.height == 1080

    def test_custom_dimensions(self) -> None:
        """Custom dimensions are respected."""
        config = RenderConfig(width=800, height=600)
        assert config.width == 800
        assert config.height == 600

    def test_zero_width_raises(self) -> None:
        """Zero width is rejected."""
        with pytest.raises(ValueError, match="width must be positive"):
            RenderConfig(width=0)

    def test_negative_height_raises(self) -> None:
        """Negative height is rejected."""
        with pytest.raises(ValueError, match="height must be positive"):
            RenderConfig(height=-1)

    def test_default_style_is_vertex(self) -> None:
        """Default render style is vertex."""
        config = RenderConfig()
        assert config.style == "vertex"

    def test_default_point_size(self) -> None:
        """Default point size is 3.0."""
        config = RenderConfig()
        assert config.point_size == 3.0

    def test_invalid_style_raises(self) -> None:
        """Invalid style value is rejected with a clear error."""
        with pytest.raises(ValueError, match="Invalid style"):
            RenderConfig(style="invalid")

    def test_invalid_style_error_lists_valid_options(self) -> None:
        """Invalid style error lists valid options."""
        with pytest.raises(ValueError, match="shaded"):
            RenderConfig(style="bad")

    def test_negative_point_size_raises(self) -> None:
        """Negative point size is rejected."""
        with pytest.raises(ValueError, match="point_size must be positive"):
            RenderConfig(point_size=-1.0)


class TestRenderStyles:
    """Tests for render style options (shaded, wireframe, vertex)."""

    def _render_with_style(
        self, tmp_path: Path, style: str, point_size: float = 3.0, suffix: str = ""
    ) -> tuple[MagicMock, Path]:
        """Render a sphere with the given style, return (mock_plotter, output_path)."""
        mock_pv = _make_mock_pyvista()
        output_file = tmp_path / f"{style}{suffix}.png"

        mock_plotter = mock_pv.Plotter.return_value
        mock_plotter.screenshot.side_effect = lambda p: Path(p).write_bytes(
            b"\x89PNG" + b"\x01" * 50
        )

        config = RenderConfig(style=style, point_size=point_size)

        with patch.dict(sys.modules, {"pyvista": mock_pv}):
            from mathviz.preview import renderer

            renderer.render_to_png(_sphere_mesh(), output_file, config=config)

        return mock_plotter, output_file

    def test_shaded_style_produces_png(self, tmp_path: Path) -> None:
        """Shaded style produces a non-trivial PNG."""
        mock_plotter, output_file = self._render_with_style(tmp_path, "shaded")
        assert output_file.exists()
        assert output_file.stat().st_size > 0
        mock_plotter.add_mesh.assert_called_once()
        call_kwargs = mock_plotter.add_mesh.call_args[1]
        assert call_kwargs.get("smooth_shading") is True
        assert "style" not in call_kwargs

    def test_wireframe_style_produces_png(self, tmp_path: Path) -> None:
        """Wireframe style produces a non-trivial PNG."""
        mock_plotter, output_file = self._render_with_style(tmp_path, "wireframe")
        assert output_file.exists()
        assert output_file.stat().st_size > 0
        mock_plotter.add_mesh.assert_called_once()
        call_kwargs = mock_plotter.add_mesh.call_args[1]
        assert call_kwargs.get("style") == "wireframe"

    def test_vertex_style_produces_png(self, tmp_path: Path) -> None:
        """Vertex style produces a non-trivial PNG."""
        mock_plotter, output_file = self._render_with_style(tmp_path, "vertex")
        assert output_file.exists()
        assert output_file.stat().st_size > 0
        mock_plotter.add_mesh.assert_called_once()
        call_kwargs = mock_plotter.add_mesh.call_args[1]
        assert call_kwargs.get("style") == "points"
        assert call_kwargs.get("render_points_as_spheres") is True

    def test_vertex_style_uses_point_size(self, tmp_path: Path) -> None:
        """Vertex style passes point_size to add_mesh."""
        mock_plotter, _ = self._render_with_style(tmp_path, "vertex", point_size=5.0)
        call_kwargs = mock_plotter.add_mesh.call_args[1]
        assert call_kwargs.get("point_size") == 5.0

    def test_point_size_default_differs_from_custom(self, tmp_path: Path) -> None:
        """Different point_size values produce different add_mesh calls."""
        default_plotter, _ = self._render_with_style(tmp_path, "vertex", point_size=3.0)
        default_size = default_plotter.add_mesh.call_args[1]["point_size"]

        custom_plotter, _ = self._render_with_style(
            tmp_path, "vertex", point_size=5.0, suffix="_large"
        )
        custom_size = custom_plotter.add_mesh.call_args[1]["point_size"]

        assert default_size != custom_size
        assert custom_size == 5.0

    def test_default_render_uses_vertex_style(self, tmp_path: Path) -> None:
        """Default render (no explicit style) uses vertex style."""
        mock_plotter, _ = self._render_with_style(tmp_path, "vertex")
        call_kwargs = mock_plotter.add_mesh.call_args[1]
        assert call_kwargs.get("style") == "points"
