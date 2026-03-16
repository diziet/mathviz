"""Tests for the render-thumbnail CLI command."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image
from typer.testing import CliRunner

from mathviz.cli import app
from mathviz.core.generator import register
from mathviz.generators.parametric.torus import TorusGenerator
from mathviz.preview.thumbnails import (
    THUMBNAIL_SIZE,
    THUMBNAILS_DIR_ENV_VAR,
    get_thumbnail_path,
)


def _ensure_torus_registered() -> None:
    """Re-register the torus generator if missing."""
    import mathviz.core.generator as gen_mod

    if "torus" not in gen_mod._alias_map:
        gen_mod._discovered = True
        register(TorusGenerator)


def _fake_generate_thumbnail(name: str, view_mode: str = "points") -> Path:
    """Write a fake WebP at the expected cache path and return it."""
    path = get_thumbnail_path(name, view_mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (THUMBNAIL_SIZE, THUMBNAIL_SIZE), color=(64, 64, 64))
    img.save(path, "webp")
    return path


@pytest.fixture(autouse=True)
def _setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Register generators and redirect thumbnail dir to temp."""
    _ensure_torus_registered()
    monkeypatch.setenv(THUMBNAILS_DIR_ENV_VAR, str(tmp_path / "thumbnails"))


@pytest.fixture
def runner() -> CliRunner:
    """Return a Typer CLI test runner."""
    return CliRunner()


class TestRenderThumbnailCli:
    """Tests for `mathviz render-thumbnail`."""

    def test_render_thumbnail_single(self, runner: CliRunner) -> None:
        """CLI command generates thumbnail and exits 0."""
        with patch(
            "mathviz.cli_thumbnail.generate_thumbnail",
            side_effect=_fake_generate_thumbnail,
        ):
            result = runner.invoke(app, ["render-thumbnail", "torus"])
        assert result.exit_code == 0
        assert "saved" in result.output.lower() or "torus" in result.output.lower()

    def test_render_thumbnail_all(self, runner: CliRunner) -> None:
        """`--all` flag generates thumbnails for all generators."""
        with patch(
            "mathviz.cli_thumbnail.generate_thumbnail",
            side_effect=_fake_generate_thumbnail,
        ):
            result = runner.invoke(app, ["render-thumbnail", "--all"])
        assert result.exit_code == 0
        assert "generated" in result.output.lower()

    def test_render_thumbnail_invalid_view_mode(self, runner: CliRunner) -> None:
        """Invalid view_mode exits with error."""
        result = runner.invoke(
            app, ["render-thumbnail", "torus", "--view-mode", "invalid"],
        )
        assert result.exit_code != 0
        assert "invalid" in result.output.lower()

    def test_render_thumbnail_no_args(self, runner: CliRunner) -> None:
        """No name and no --all exits with error."""
        result = runner.invoke(app, ["render-thumbnail"])
        assert result.exit_code != 0

    def test_render_thumbnail_unknown_generator(self, runner: CliRunner) -> None:
        """Unknown generator name exits with error."""
        with patch(
            "mathviz.cli_thumbnail.generate_thumbnail",
            side_effect=KeyError("not_real"),
        ):
            result = runner.invoke(app, ["render-thumbnail", "not_real"])
        assert result.exit_code != 0
        assert "unknown" in result.output.lower()
