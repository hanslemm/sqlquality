"""Command-line interface for sqlquality."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import NoReturn

import typer
from rich.console import Console
from rich.table import Table

from sqlquality import __version__
from sqlquality.adapters import get_adapter
from sqlquality.changeset import ChangeSetError, compute_changeset, run_state_modified
from sqlquality.complexity import ComplexityEngine
from sqlquality.config import ConfigError, load_config
from sqlquality.dbtproject import DbtProject, DbtProjectError
from sqlquality.delta import compute_deltas
from sqlquality.dialects import validate_dialect
from sqlquality.gate import evaluate_gate
from sqlquality.linter import fix_sql, lint_sql
from sqlquality.llm import Suggestion, enrich_findings, resolve_provider
from sqlquality.models import Severity
from sqlquality.report import gate_payload, render_html, render_markdown, verdict_label
from sqlquality.sqlast import SqlParseError, analyze_sql, parse, strip_jinja

console = Console()

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Measure dbt model performance and complexity.",
    epilog=(
        "Exit codes: 0 = pass / no findings; 1 = findings or gate failure; "
        "2 = usage, config, or input error."
    ),
)

#: Substrings whose presence marks a source as containing dbt/Jinja templating.
_JINJA_MARKERS = ("{{", "{%")
#: Notice emitted (stderr only) when analysis falls back to Jinja placeholders.
_JINJA_NOTICE = (
    "analyzed with Jinja placeholders — results are approximate; "
    "prefer compiled SQL from target/compiled/"
)
#: Appended to a parse-error message when the source contained Jinja markers.
_COMPILED_HINT = (
    " — the source contains Jinja; supply compiled SQL from target/compiled/ for accurate results"
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


def _has_jinja(sql: str) -> bool:
    """True if the source contains any dbt/Jinja templating markers."""
    return any(marker in sql for marker in _JINJA_MARKERS)


def _labels(path: Path) -> tuple[str, str]:
    """Return (display_name, machine_path) for a source; '<stdin>' when path is '-'."""
    if str(path) == "-":
        return "<stdin>", "<stdin>"
    return path.name, str(path)


def read_sql_file(path: Path) -> str:
    """Read SQL text from a file, or from stdin when ``path`` is ``-``.

    Prints a friendly message and exits 2 on a missing file, a non-UTF-8 file, or
    any other read error, so callers get a consistent input-error experience.
    """
    if str(path) == "-":
        return sys.stdin.read()
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        typer.echo(f"No such file: {path}", err=True)
        raise typer.Exit(code=2)
    except UnicodeDecodeError:
        typer.echo(f"{path} is not valid UTF-8 — supply a UTF-8 encoded SQL file.", err=True)
        raise typer.Exit(code=2)
    except OSError as exc:
        typer.echo(f"Could not read {path}: {exc}", err=True)
        raise typer.Exit(code=2)


def _validate_dialect_or_exit(name: str) -> str:
    """Normalize a dialect name or print the friendly error and exit 2."""
    try:
        return validate_dialect(name)
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2)


def _fail_parse(exc: SqlParseError, *, had_jinja: bool) -> NoReturn:
    """Print a parse-error message (with a compiled-SQL hint for Jinja) and exit 2."""
    message = str(exc)
    if had_jinja:
        message += _COMPILED_HINT
    typer.echo(message, err=True)
    raise typer.Exit(code=2)


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
        dir_okay=False,
        help="Path to a .sql file (or '-' for stdin).",
    ),
    dialect: str = typer.Option(
        "postgres", "--dialect", "-d", help="SQL dialect (e.g. postgres, redshift)."
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Score the structural complexity of a single SQL file."""
    dialect = _validate_dialect_or_exit(dialect)
    sql = read_sql_file(path)
    display_name, machine_path = _labels(path)

    jinja_notice = False
    try:
        metrics = analyze_sql(sql, dialect)
    except SqlParseError as exc:
        if not _has_jinja(sql):
            _fail_parse(exc, had_jinja=False)
        try:
            metrics = analyze_sql(strip_jinja(sql), dialect)
        except SqlParseError as retry_exc:
            _fail_parse(retry_exc, had_jinja=True)
        jinja_notice = True

    if jinja_notice:
        typer.echo(_JINJA_NOTICE, err=True)

    result = ComplexityEngine().score(metrics)

    if json_out:
        payload = {
            "path": machine_path,
            "dialect": dialect,
            "composite": result.composite,
            "components": result.components,
            "metrics": asdict(metrics),
        }
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        return

    table = Table(title=f"Complexity — {display_name}  (composite {result.composite})")
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


def _resolve_check_dialect(candidate: DbtProject) -> str:
    """Resolve check's dialect from the manifest's adapter_type, else postgres.

    Emits a stderr notice describing the source. Only called when no explicit
    ``--dialect`` was given.
    """
    adapter_type = candidate.adapter_type()
    if adapter_type:
        try:
            resolved = validate_dialect(adapter_type)
        except ValueError:
            resolved = None
        if resolved is not None:
            typer.echo(f"dialect: {resolved} (from manifest adapter_type)", err=True)
            return resolved
    typer.echo(
        "dialect: postgres (default — manifest adapter_type absent or unrecognized)",
        err=True,
    )
    return "postgres"


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
    dialect: str | None = typer.Option(
        None,
        "--dialect",
        "-d",
        help="SQL dialect (default: manifest adapter_type, else postgres).",
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
    html: Path | None = typer.Option(None, "--html", help="Write a self-contained HTML report."),
    markdown: Path | None = typer.Option(
        None, "--markdown", help="Write a markdown report (e.g. for a PR comment)."
    ),
    dbt: str = typer.Option("dbt", "--dbt", help="dbt executable to invoke."),
) -> None:
    """Gate a dbt change on the complexity delta of its changed models."""
    # An explicit --config that isn't a readable file (missing, or a directory)
    # is a user error; the implicit <project-dir>/sqlquality.yml default stays
    # lenient (absent -> defaults).
    if config is not None and not config.is_file():
        typer.echo(f"--config path is not a file: {config}", err=True)
        raise typer.Exit(code=2)
    # An explicit --dialect is validated up front; the manifest-derived default is
    # resolved after the candidate manifest loads.
    if dialect is not None:
        dialect = _validate_dialect_or_exit(dialect)
    cfg_path = config if config is not None else project_dir / "sqlquality.yml"
    try:
        cfg = load_config(cfg_path)
    except ConfigError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2)

    manifest_path = project_dir / "target" / "manifest.json"
    try:
        candidate = DbtProject.from_path(manifest_path)
    except DbtProjectError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2)

    resolved_dialect = dialect if dialect is not None else _resolve_check_dialect(candidate)

    schema_version = candidate.schema_version()
    if "/v12" not in schema_version:
        found = schema_version or "(absent)"
        typer.echo(
            f"warning: candidate manifest dbt_schema_version is {found}, "
            "expected a v12 schema — results may be unreliable",
            err=True,
        )

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

    deltas, skipped = compute_deltas(baseline, candidate, changeset.changed, resolved_dialect)
    report = evaluate_gate(deltas, cfg)

    if html is not None:
        Path(html).write_text(render_html(report, skipped))

    if markdown is not None:
        Path(markdown).write_text(render_markdown(report, skipped))

    if json_out:
        typer.echo(
            json.dumps(gate_payload(report, changeset.neighbors, skipped), indent=2, sort_keys=True)
        )
    else:
        verdict = verdict_label(report, emoji=True)
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
    paths: list[Path] = typer.Argument(
        ..., dir_okay=False, help="One or more .sql files (or '-' for stdin)."
    ),
    dialect: str = typer.Option("postgres", "--dialect", "-d", help="SQL dialect."),
    fix: bool = typer.Option(
        False,
        "--fix",
        help="Rewrite the file with auto-fixes. Exit code reflects pre-fix findings "
        "(a fully-fixed file still exits 1).",
    ),
    exclude_rules: str | None = typer.Option(
        None, "--exclude-rules", help="Comma-separated rule codes to skip."
    ),
    sqlfluff_config: Path | None = typer.Option(
        None,
        "--sqlfluff-config",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Path to a SQLFluff config file (e.g. .sqlfluff) to apply.",
    ),
    warn_only: bool = typer.Option(
        False, "--warn-only", help="Print/emit findings but always exit 0."
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Lint SQL files for best-practice violations (SQLFluff); --fix rewrites them."""
    dialect = _validate_dialect_or_exit(dialect)
    if fix and any(str(p) == "-" for p in paths):
        typer.echo("--fix cannot rewrite stdin ('-'); pass a file path instead.", err=True)
        raise typer.Exit(code=2)
    excl = [r.strip() for r in exclude_rules.split(",")] if exclude_rules else None
    config_path = str(sqlfluff_config) if sqlfluff_config is not None else None
    file_reports: list[dict] = []
    gating = False
    for path in paths:
        sql = read_sql_file(path)
        _, machine_path = _labels(path)
        findings = lint_sql(sql, dialect, excl, config_path)
        changed = False
        if fix:
            fixed_sql = fix_sql(sql, dialect, excl, config_path)
            if fixed_sql != sql:
                path.write_text(fixed_sql)
                changed = True
        # INFO (unresolved-Jinja) findings are advisory and never gate the commit.
        gating = gating or any(f.severity in (Severity.WARNING, Severity.ERROR) for f in findings)
        file_reports.append(
            {
                "path": machine_path,
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
        )

    if json_out:
        typer.echo(json.dumps({"files": file_reports}, indent=2, sort_keys=True))
    else:
        for report in file_reports:
            table = Table(title=f"Lint — {report['path']} ({len(report['findings'])} findings)")
            table.add_column("line", justify="right")
            table.add_column("code")
            table.add_column("severity")
            table.add_column("fix?", justify="center")
            table.add_column("message")
            for item in report["findings"]:
                table.add_row(
                    str(item["line"]),
                    item["code"],
                    item["severity"],
                    "✓" if item["fixable"] else "",
                    item["message"],
                )
            console.print(table)
            if report["fixed"]:
                console.print(f"[green]Rewrote {report['path']} with auto-fixes.[/]")

    raise typer.Exit(code=1 if gating and not warn_only else 0)


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
    suggest: bool = typer.Option(
        False, "--suggest", help="Enrich findings with LLM suggestions (needs SQLQUALITY_LLM set)."
    ),
) -> None:
    """Analyze a SQL file for performance anti-patterns (+ optional EXPLAIN plan)."""
    dialect = _validate_dialect_or_exit(dialect)
    try:
        adapter = get_adapter(dialect)
    except ValueError as exc:
        # A valid-but-unsupported dialect (only postgres/redshift have perf adapters).
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2)

    sql = read_sql_file(path)
    display_name, machine_path = _labels(path)

    # static_findings swallows parse errors into an SQ000 finding, so a raw dbt
    # model would otherwise yield only SQ000. Parse-check first and, when the source
    # is Jinja, retry against stripped placeholders so we get real anti-pattern
    # findings; if stripping still fails, fall back to SQ000 (annotated below).
    analyze_target = sql
    jinja_notice = False
    had_jinja = _has_jinja(sql)
    try:
        parse(sql, dialect)
    except SqlParseError:
        if had_jinja:
            stripped = strip_jinja(sql)
            try:
                parse(stripped, dialect)
            except SqlParseError:
                pass
            else:
                analyze_target = stripped
                jinja_notice = True

    findings = adapter.static_findings(analyze_target)
    if had_jinja and not jinja_notice:
        # Stripping did not yield parseable SQL: annotate the SQ000 parse error with
        # the compiled-SQL hint so the user knows why and what to do.
        findings = [
            replace(f, message=f.message + _COMPILED_HINT) if f.code == "SQ000" else f
            for f in findings
        ]
    if jinja_notice:
        typer.echo(_JINJA_NOTICE, err=True)

    if explain is not None:
        try:
            findings = findings + adapter.plan_findings(explain.read_text())
        except ValueError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=2)

    suggestions: list[Suggestion] = []
    if suggest:
        try:
            provider = resolve_provider()
            if provider is None:
                typer.echo(
                    "LLM suggestions require SQLQUALITY_LLM=anthropic (or 1/true) "
                    "(and `pip install 'sqlquality[llm]'` + credentials).",
                    err=True,
                )
            else:
                suggestions = enrich_findings(findings, sql, provider)
                if len(suggestions) < len(findings):
                    # enrich_findings skips per-finding call failures silently, so
                    # surface a single note when some (or all) calls dropped out.
                    missing = len(findings) - len(suggestions)
                    typer.echo(f"LLM suggestions unavailable for {missing} finding(s).", err=True)
        except Exception as exc:  # advisory-only: never affect the exit code or report
            # Covers provider construction (missing package/credentials); findings
            # still print and the exit code is unchanged.
            typer.echo(f"LLM suggestions unavailable: {exc}", err=True)

    if json_out:
        payload = {
            "path": machine_path,
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
            "suggestions": [{"code": s.code, "text": s.text} for s in suggestions],
        }
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
    else:
        table = Table(title=f"Perf — {display_name} ({dialect}, {len(findings)} findings)")
        table.add_column("code")
        table.add_column("severity")
        table.add_column("message")
        for f in findings:
            table.add_row(f.code, f.severity.value, f.message)
        console.print(table)
        for s in suggestions:
            console.print(f"[cyan]{s.code}[/]: {s.text}")

    has_error = any(f.severity is Severity.ERROR for f in findings)
    raise typer.Exit(code=1 if has_error else 0)


def main() -> None:
    """Console-script entry point."""
    app()
