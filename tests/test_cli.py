from __future__ import annotations

from typer.testing import CliRunner

from scperturb_cmap.cli import app


def test_cli_help_lists_commands():
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    text = result.stdout
    for cmd in [
        "prepare-lincs",
        "make-target",
        "score",
        "train",
        "evaluate",
        "ui",
    ]:
        assert cmd in text


def test_cli_score_help():
    runner = CliRunner()
    result = runner.invoke(app, ["score", "--help"])
    assert result.exit_code == 0
    assert "--target-json" in result.stdout
    assert "--library" in result.stdout

