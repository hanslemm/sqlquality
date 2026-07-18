"""Command-line interface for sqlquality."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from sqlquality import __version__
from sqlquality.changeset import ChangeSetError, compute_changeset, run_state_modified
from sqlquality.checkcmd import run_check
from sqlquality.complexity import ComplexityEngine
from sqlquality.dbtproject import DbtProject, DbtProjectError
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


@app.command()
def check(
    project_dir: Path = typer.Option(
        ..., "--project-dir", help="dbt project dir containing target/manifest.json."
    ),
    state: Path = typer.Option(
        ..., "--state", help="Baseline artifacts dir for `dbt ls --state`."
    ),
    dialect: str = typer.Option("postgres", "--dialect", "-d", help="SQL dialect."),
    json_out: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
    dbt: str = typer.Option("dbt", "--dbt", help="dbt executable to invoke."),
) -> None:
    """Score the complexity of models changed vs a baseline dbt state."""
    manifest_path = project_dir / "target" / "manifest.json"
    try:
        project = DbtProject.from_path(manifest_path)
    except DbtProjectError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2)

    try:
        ls_stdout = run_state_modified(project_dir, state, dbt)
    except ChangeSetError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2)
    changeset = compute_changeset(project, ls_stdout)
    results, skipped = run_check(project, changeset.changed, dialect)

    if json_out:
        payload = {
            "changed": changeset.changed,
            "neighbors": changeset.neighbors,
            "results": [
                {
                    "unique_id": r.unique_id,
                    "composite": r.composite,
                    "fan_in": r.dag.fan_in,
                    "fan_out": r.dag.fan_out,
                    "lineage_depth": r.dag.lineage_depth,
                }
                for r in results
            ],
            "skipped": [{"unique_id": uid, "reason": reason} for uid, reason in skipped],
        }
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    table = Table(title=f"Changed models: {len(changeset.changed)}  (neighbors: {len(changeset.neighbors)})")
    table.add_column("model")
    table.add_column("complexity", justify="right")
    table.add_column("fan_in", justify="right")
    table.add_column("fan_out", justify="right")
    table.add_column("depth", justify="right")
    for r in results:
        table.add_row(
            r.unique_id, str(r.composite), str(r.dag.fan_in), str(r.dag.fan_out), str(r.dag.lineage_depth)
        )
    console.print(table)
    for uid, reason in skipped:
        console.print(f"[yellow]skipped[/] {uid}: {reason}")


def main() -> None:
    """Console-script entry point."""
    app()
