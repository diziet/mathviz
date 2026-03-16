"""Tests for the render-thumbnail CLI command."""

from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from mathviz.cli import app
from tests.test_preview.conftest import create_fake_thumbnail


@pytest.fixture(autouse=True)
def _setup(thumbnail_setup: None) -> None:
    """Use shared thumbnail setup fixture."""


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
            side_effect=create_fake_thumbnail,
        ):
            result = runner.invoke(app, ["render-thumbnail", "torus"])
        assert result.exit_code == 0
        assert "saved" in result.output.lower() or "torus" in result.output.lower()

    def test_render_thumbnail_all(self, runner: CliRunner) -> None:
        """`--all` flag generates thumbnails for all generators."""
        with patch(
            "mathviz.cli_thumbnail.generate_thumbnail",
            side_effect=create_fake_thumbnail,
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
