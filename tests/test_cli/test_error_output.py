"""Tests that CLI error messages print literally, without Rich markup or emoji codes."""

import json
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from mathviz import cli_output
from mathviz.cli import app
from mathviz.core.representation import RepresentationConfig

runner = CliRunner()

BRACKET_KEY_TOML = '[representation]\ntype = "tube"\n"tube[radius]" = 0.15\n'
BRACKET_KEY_ERROR = (
    "Invalid representation config: tube[radius]: unknown key. "
    f"Valid keys: {', '.join(RepresentationConfig.model_fields)}"
)
ANSI_RED = "\x1b[31m"
ANSI_RESET = "\x1b[0m"


@pytest.fixture(autouse=True)
def isolated_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run each test in an empty directory so no stray mathviz.toml is discovered."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _unwrapped(output: str) -> str:
    """Join the lines that Rich wraps at the console width into one line."""
    return " ".join(output.split())


class TestErrorExitRichOutput:
    """Test the error text that error_exit prints without --json."""

    def test_quoted_toml_key_with_brackets_prints_in_full(self, isolated_cwd: Path) -> None:
        """The quoted [representation] key "tube[radius]" prints with its brackets."""
        config = isolated_cwd / "bracket.toml"
        config.write_text(BRACKET_KEY_TOML)
        result = runner.invoke(app, ["generate", "trefoil", "--config", str(config)])
        assert result.exit_code == 2
        assert f"Error: {BRACKET_KEY_ERROR}" in _unwrapped(result.output)

    def test_missing_config_path_with_brackets_prints_in_full(self) -> None:
        """A missing --config path such as cfg[v2].toml prints with its brackets."""
        result = runner.invoke(app, ["generate", "trefoil", "--config", "cfg[v2].toml"])
        assert result.exit_code == 2
        assert "Config file not found: cfg[v2].toml" in _unwrapped(result.output)

    def test_closing_markup_tag_in_message_exits_2(self) -> None:
        """A generator name of "[/red]" exits 2 with the name, not a MarkupError."""
        result = runner.invoke(app, ["generate", "[/red]"])
        assert result.exit_code == 2, repr(result.exception)
        assert "Error: Unknown generator: '[/red]'" in result.output

    def test_emoji_code_in_message_prints_literally(self) -> None:
        """A generator name of ":fire:" prints as typed, not as an emoji."""
        result = runner.invoke(app, ["generate", ":fire:"])
        assert result.exit_code == 2
        assert "Error: Unknown generator: ':fire:'" in result.output

    def test_error_prints_in_red_without_highlighting(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """On a color terminal the whole error line is red, with no other styles."""
        color_console = Console(
            force_terminal=True, color_system="standard", no_color=False, width=200
        )
        monkeypatch.setattr(cli_output, "console", color_console)
        result = runner.invoke(app, ["generate", "tube[radius]"])
        assert result.exit_code == 2
        expected = f"{ANSI_RED}Error: Unknown generator: 'tube[radius]'{ANSI_RESET}\n"
        assert result.output == expected


class TestErrorExitJsonOutput:
    """Test the error payload that error_exit prints with --json."""

    def test_json_error_keeps_quoted_toml_key_with_brackets(self, isolated_cwd: Path) -> None:
        """--json prints the unknown-key error unchanged, brackets included."""
        config = isolated_cwd / "bracket.toml"
        config.write_text(BRACKET_KEY_TOML)
        result = runner.invoke(
            app, ["generate", "trefoil", "--json", "--config", str(config)]
        )
        assert result.exit_code == 2
        assert json.loads(result.output) == {"error": BRACKET_KEY_ERROR}
