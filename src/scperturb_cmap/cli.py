from __future__ import annotations

import typer

from . import __version__
from .utils.device import get_device


app = typer.Typer(help="scPerturb-CMap command line interface")


@app.command()
def version() -> None:
    """Print package version."""
    typer.echo(__version__)


@app.command()
def device() -> None:
    """Print the selected compute device (cuda|mps|cpu)."""
    typer.echo(get_device())


def main() -> None:
    app()

