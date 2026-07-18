"""Command-line interface for sqlquality."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from sqlquality import __version__
from sqlquality.adapters import get_adapter
from sqlquality.changeset import ChangeSetError, compute_changeset, run_state_modified
from sqlquality.complexity import ComplexityEngine
from sqlquality.config import load_config
from sqlquality.dbtproject import DbtProject, DbtProjectError
from sqlquality.delta import compute_deltas
from sqlquality.gate import evaluate_gate
from sqlquality.linter import fix_sql, lint_sql
from sqlquality.models import Severity
from sqlquality.report import gate_payload, render_html
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

    table = Table(title=f"Complexity — {path.name}  (composite {result.composite}/100)")
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
        ..., "--state", help="Baseline artifacts dir (contains manifest.json)."
    ),
    config: Path | None = typer.Option(
        None, "--config", help="Path to sqlquality.yml (default: <project-dir>/sqlquality.yml)."
    ),
    dialect: str = typer.Option("postgres", "--dialect", "-d", help="SQL dialect."),
    json_out: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
    html: Path | None = typer.Option(None, "--html", help="Write a self-contained HTML report."),
    dbt: str = typer.Option("dbt", "--dbt", help="dbt executable to invoke."),
) -> None:
    """Gate a dbt change on the complexity delta of its changed models."""
    cfg_path = config if config is not None else project_dir / "sqlquality.yml"
    cfg = load_config(cfg_path)

    manifest_path = project_dir / "target" / "manifest.json"
    try:
        candidate = DbtProject.from_path(manifest_path)
    except DbtProjectError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2)

    try:
        ls_stdout = run_state_modified(project_dir, state, dbt)
    except ChangeSetError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2)

    changeset = compute_changeset(candidate, ls_stdout)

    baseline_path = state / "manifest.json"
    try:
        baseline = DbtProject.from_path(baseline_path) if baseline_path.exists() else None
    except DbtProjectError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2)

    deltas, skipped = compute_deltas(baseline, candidate, changeset.changed, dialect)
    report = evaluate_gate(deltas, cfg)

    if html is not None:
        Path(html).write_text(render_html(report, skipped))

    if json_out:
        typer.echo(
            json.dumps(gate_payload(report, changeset.neighbors, skipped), indent=2, sort_keys=True)
        )
    else:
        verdict = "PASS" if report.passed else "FAIL"
        table = Table(
            title=f"sqlquality: {verdict}  (changed {len(deltas)}, neighbors {len(changeset.neighbors)})"
        )
        table.add_column("model")
        table.add_column("baseline", justify="right")
        table.add_column("candidate", justify="right")
        table.add_column("delta", justify="right")
        table.add_column("", justify="center")
        for d in report.deltas:
            flag = "⚠️" if d.unique_id in report.regressions else ("new" if d.is_new else "")
            table.add_row(d.unique_id, str(d.baseline), str(d.candidate), f"{d.delta:+}", flag)
        console.print(table)
        for uid, reason in skipped:
            console.print(f"[yellow]skipped[/] {uid}: {reason}")

    raise typer.Exit(code=0 if report.passed else 1)


@app.command()
def lint(
    path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, readable=True, help="Path to a .sql file."
    ),
    dialect: str = typer.Option("postgres", "--dialect", "-d", help="SQL dialect."),
    fix: bool = typer.Option(False, "--fix", help="Rewrite the file with auto-fixes."),
    exclude_rules: str | None = typer.Option(
        None, "--exclude-rules", help="Comma-separated rule codes to skip."
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Lint a SQL file for best-practice violations (SQLFluff); --fix rewrites it."""
    sql = path.read_text()
    excl = [r.strip() for r in exclude_rules.split(",")] if exclude_rules else None
    findings = lint_sql(sql, dialect, excl)

    changed = False
    if fix:
        fixed_sql = fix_sql(sql, dialect, excl)
        if fixed_sql != sql:
            path.write_text(fixed_sql)
            changed = True

    if json_out:
        payload = {
            "path": str(path),
            "fixed": changed,
            "findings": [
                {
                    "code": f.code,
                    "message": f.message,
                    "line": f.line,
                    "severity": f.severity.value,
                    "fixable": f.fixable,
                }
                for f in findings
            ],
        }
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        table = Table(title=f"Lint — {path.name}  ({len(findings)} findings)")
        table.add_column("line", justify="right")
        table.add_column("code")
        table.add_column("severity")
        table.add_column("fix?", justify="center")
        table.add_column("message")
        for f in findings:
            table.add_row(
                str(f.line), f.code, f.severity.value, "✓" if f.fixable else "", f.message
            )
        console.print(table)
        if fix:
            console.print(
                "[green]Applied auto-fixes and rewrote the file.[/]"
                if changed
                else "[yellow]No auto-fixable changes.[/]"
            )

    has_error = any(f.severity is Severity.ERROR for f in findings)
    raise typer.Exit(code=1 if has_error else 0)


@app.command()
def perf(
    path: Path = typer.Argument(
        ..., exists=True, dir_okay=False, readable=True, help="Path to a .sql file."
    ),
    dialect: str = typer.Option("postgres", "--dialect", "-d", help="SQL dialect/engine."),
    explain: Path | None = typer.Option(
        None,
        "--explain",
        exists=True,
        dir_okay=False,
        readable=True,
        help="A captured EXPLAIN file (FORMAT JSON for Postgres; plan text for Redshift).",
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Analyze a SQL file for performance anti-patterns (+ optional EXPLAIN plan)."""
    sql = path.read_text()
    try:
        adapter = get_adapter(dialect)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2)

    findings = adapter.static_findings(sql)
    if explain is not None:
        try:
            findings = findings + adapter.plan_findings(explain.read_text())
        except ValueError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=2)

    if json_out:
        payload = {
            "path": str(path),
            "dialect": dialect,
            "findings": [
                {
                    "code": f.code,
                    "message": f.message,
                    "line": f.line,
                    "severity": f.severity.value,
                    "fixable": f.fixable,
                }
                for f in findings
            ],
        }
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        table = Table(title=f"Perf — {path.name} ({dialect}, {len(findings)} findings)")
        table.add_column("code")
        table.add_column("severity")
        table.add_column("message")
        for f in findings:
            table.add_row(f.code, f.severity.value, f.message)
        console.print(table)

    has_error = any(f.severity is Severity.ERROR for f in findings)
    raise typer.Exit(code=1 if has_error else 0)


def main() -> None:
    """Console-script entry point."""
    app()
