"""Command-line interface for sqlquality."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from sqlquality import __version__
from sqlquality.complexity import ComplexityEngine
from sqlquality.sqlast import SqlParseError, analyze_sql

console = Console()

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


@app.command()
def complexity(
    path: Path = typer.Argument(
        ...,
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to a .sql file (compiled/plain SQL, not Jinja).",
    ),
    dialect: str = typer.Option(
        "postgres", "--dialect", "-d", help="SQL dialect (e.g. postgres, redshift)."
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Score the structural complexity of a single SQL file."""
    sql = path.read_text()
    try:
        metrics = analyze_sql(sql, dialect)
    except SqlParseError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2)

    result = ComplexityEngine().score(metrics)

    if json_out:
        payload = {
            "path": str(path),
            "dialect": dialect,
            "composite": result.composite,
            "components": result.components,
            "metrics": asdict(metrics),
        }
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    table = Table(
        title=f"Complexity — {path.name}  (composite {result.composite}/100)"
    )
    table.add_column("metric")
    table.add_column("value", justify="right")
    table.add_column("contribution", justify="right")
    for name, contribution in result.components.items():
        raw_value = getattr(metrics, name, None)
        table.add_row(
            name,
            "" if raw_value is None else str(raw_value),
            str(contribution),
        )
    console.print(table)


def main() -> None:
    """Console-script entry point."""
    app()
