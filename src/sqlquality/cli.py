"""Command-line interface for sqlquality."""

from __future__ import annotations

import typer

from sqlquality import __version__

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Measure dbt model performance and complexity.",
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def _root(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show the version and exit.",
    ),
) -> None:
    """sqlquality — measure dbt model performance and complexity."""


def main() -> None:
    """Console-script entry point."""
    app()
