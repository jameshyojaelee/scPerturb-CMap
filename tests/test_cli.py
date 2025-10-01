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
        "power",
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


def test_cli_power_help_lists_commands():
    runner = CliRunner()
    result = runner.invoke(app, ["power", "--help"])
    assert result.exit_code == 0
    text = result.stdout
    for cmd in [
        "sample-size",
        "min-cells",
        "rank-ci",
        "stability",
        "fdr",
        "permutation-test",
    ]:
        assert cmd in text


def test_cli_power_permutation_test_runs():
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "power",
            "permutation-test",
            "--group-a",
            "0.1",
            "--group-a",
            "0.2",
            "--group-b",
            "0.3",
            "--group-b",
            "0.4",
            "--n-permutations",
            "10",
            "--random-seed",
            "1",
        ],
    )
    assert result.exit_code == 0
    assert "p_value" in result.stdout
