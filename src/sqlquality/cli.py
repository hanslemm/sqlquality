"""Command-line interface for sqlquality."""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import asdict, replace
from datetime import timedelta
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
from sqlquality.models import Aggregation, Severity, Workload
from sqlquality.report import (
    advise_payload,
    gate_payload,
    render_advise_markdown,
    render_html,
    render_markdown,
    verdict_label,
)
from sqlquality.sqlast import SqlParseError, analyze_sql, parse, strip_jinja
from sqlquality.workload import get_workload_adapter
from sqlquality.workload.aggregate import aggregate
from sqlquality.workload.connection import ConnectionResolutionError, resolve_connection
from sqlquality.workload.fingerprint import ingest

console = Console()

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Measure dbt model performance and complexity, and advise on database optimizations.",
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

    Prints a friendly message and exits 2 on a missing file, a non-UTF-8 source, or
    any other read error, so callers get a consistent input-error experience. Stdin
    is decoded from raw bytes so a non-UTF-8 pipe fails the same way a file does
    (never a traceback / exit 1 that CI would misread as findings).
    """
    is_stdin = str(path) == "-"
    source = "<stdin>" if is_stdin else str(path)
    try:
        if is_stdin:
            return sys.stdin.buffer.read().decode("utf-8")
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        typer.echo(f"No such file: {path}", err=True)
        raise typer.Exit(code=2)
    except UnicodeDecodeError:
        typer.echo(f"{source} is not valid UTF-8 — supply UTF-8 encoded SQL.", err=True)
        raise typer.Exit(code=2)
    except OSError as exc:
        typer.echo(f"Could not read {source}: {exc}", err=True)
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
    if isinstance(adapter_type, str) and adapter_type:
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
    # Pre-flight every path before touching any file: a bad path (missing, non-UTF-8)
    # must exit 2 with no side effects, not rewrite earlier files then abort.
    sources = [(path, read_sql_file(path)) for path in paths]
    file_reports: list[dict] = []
    gating = False
    for path, sql in sources:
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
    if str(path) == "-":
        typer.echo("stdin ('-') is not supported for perf; pass a file path.", err=True)
        raise typer.Exit(code=2)
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
    # model would otherwise yield only SQ000. When the source is Jinja, parse-check
    # first and retry against stripped placeholders so we get real anti-pattern
    # findings; if stripping still fails, fall back to SQ000 (annotated below).
    # Plain SQL skips the pre-parse — static_findings parses it once itself.
    analyze_target = sql
    jinja_notice = False
    had_jinja = _has_jinja(sql)
    if had_jinja:
        try:
            parse(sql, dialect)
        except SqlParseError:
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


_SINCE_UNITS = {"h": "hours", "d": "days", "w": "weeks"}
#: Accepted --timeout range, in seconds. The adapter clamps to the same bounds as a safety
#: net, but silently altering a number the user typed is worse than telling them it is out
#: of range — 0 in particular means "no limit" to Postgres, the opposite of a timeout.
_TIMEOUT_MIN_S = 1
_TIMEOUT_MAX_S = 3600
#: Warn when this share of the *analyzable* statements could not be analyzed. Coverage is
#: not cosmetic: cost_share divides by the whole window's cost including skipped statements,
#: so poor coverage dilutes every share and makes --min-cost-share progressively stricter.
#: Without this warning, "no proposals" is indistinguishable from "I understood almost none
#: of your workload".
_LOW_COVERAGE_FRACTION = 0.2


def _plural(count: int, noun: str) -> str:
    """`3 proposals` / `1 proposal`. Small, but "1 proposals" reads as a bug in the tool."""
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _validate_timeout(value: int) -> int:
    """Return a timeout in seconds, or exit 2 with a message naming the accepted range."""
    if not _TIMEOUT_MIN_S <= value <= _TIMEOUT_MAX_S:
        typer.echo(
            f"--timeout must be between {_TIMEOUT_MIN_S} and {_TIMEOUT_MAX_S} seconds "
            f"(got {value}). Postgres treats 0 as no limit, which would defeat the flag.",
            err=True,
        )
        raise typer.Exit(code=2)
    return value


def _analyzed_count(workload: Workload, aggregation: Aggregation) -> int:
    """Query groups whose usage was actually extracted.

    Unresolvable groups are a *subset* of ``workload.stats``, not a separate pool, so
    ``len(stats)`` overstates what was understood.
    """
    return max(0, len(workload.stats) - aggregation.skipped_unqualifiable)


def _coverage_line(workload: Workload, aggregation: Aggregation) -> str:
    """One-line coverage disclosure, printed on every run.

    Says "N of M" rather than just "N". Printing ``len(stats)`` as *analyzed* next to
    "2 unresolvable" contradicts itself on a single line — and does so least accurately
    exactly when coverage is worst, which is the situation the line exists to reveal.
    """
    return (
        f"analyzed {_analyzed_count(workload, aggregation)} of {len(workload.stats)} "
        f"query group(s); skipped {workload.skipped_unparseable} unparseable, "
        f"{workload.skipped_noise} introspection/DDL, "
        f"{aggregation.skipped_unqualifiable} unresolvable"
    )


def _coverage_warning(workload: Workload, aggregation: Aggregation) -> str | None:
    """A warning when too little of the workload could be analyzed, else None.

    Noise (introspection, DDL, session control) is excluded from the denominator: those are
    deliberately filtered, not failures to understand. Only statements we tried and failed
    to use count against coverage.
    """
    analyzed = _analyzed_count(workload, aggregation)
    unexplained = workload.skipped_unparseable + aggregation.skipped_unqualifiable
    considered = analyzed + unexplained
    if not considered:
        return None
    share = unexplained / considered
    if share <= _LOW_COVERAGE_FRACTION:
        return None
    return (
        f"low coverage: {share:.0%} of candidate statements could not be analyzed "
        f"({workload.skipped_unparseable} unparseable, "
        f"{aggregation.skipped_unqualifiable} unresolvable against the schema). "
        "Cost shares are computed against the whole window, so they are diluted and "
        "--min-cost-share is effectively stricter — few or no proposals may reflect "
        "coverage rather than a healthy workload."
    )


def _validate_schemas(values: list[str]) -> tuple[str, ...]:
    """Accept exactly one distinct schema, or exit 2 explaining why.

    Every catalog fact `advise` collects is keyed on the bare relation name: table sizes,
    NDV maps, index lists and the qualify() schema all merge across schemas, so two
    schemas each holding an `orders` alias into one another silently — the last catalog
    row wins the row estimate, and `qualify()` resolves columns against a union of the
    two column sets. Rejecting is the honest minimum until the keys are schema-qualified.
    """
    distinct = tuple(dict.fromkeys(values))
    if len(distinct) > 1:
        typer.echo(
            f"--schema accepts one schema at a time (got {', '.join(distinct)}). "
            "Multi-schema introspection is not supported yet: table facts, index lists "
            "and NDV statistics are keyed on the bare table name, not schema-qualified, "
            "so same-named tables in two schemas would silently alias. Run advise once "
            "per schema.",
            err=True,
        )
        raise typer.Exit(code=2)
    return distinct


def _parse_since(value: str | None) -> timedelta | None:
    """Parse a '7d' / '24h' / '2w' duration, or exit 2."""
    if value is None:
        return None
    match = re.fullmatch(r"(\d+)([hdw])", value.strip().lower())
    if match is None:
        typer.echo(
            f"Could not parse --since {value!r}. Use a count and a unit, e.g. 24h, 7d, 2w.",
            err=True,
        )
        raise typer.Exit(code=2)
    return timedelta(**{_SINCE_UNITS[match.group(2)]: int(match.group(1))})


@app.command()
def advise(
    engine: str | None = typer.Option(
        None, "--engine", help="postgres | redshift | snowflake. Inferred from the DSN if unset."
    ),
    dsn: str | None = typer.Option(None, "--dsn", help="Database URL. Overrides SQLQUALITY_DSN."),
    profile: str | None = typer.Option(None, "--profile", help="dbt profile name (optional)."),
    target: str | None = typer.Option(None, "--target", help="dbt target within the profile."),
    profiles_dir: Path | None = typer.Option(
        None, "--profiles-dir", help="Directory holding profiles.yml (default: ~/.dbt)."
    ),
    schema: list[str] = typer.Option(
        ["public"], "--schema", help="Schema to introspect. One at a time (see --help notes)."
    ),
    since: str | None = typer.Option(
        None, "--since", help="Window, e.g. 7d. Not supported by pg_stat_statements."
    ),
    limit: int = typer.Option(500, "--limit", help="Max query-history rows to read."),
    min_cost_share: float = typer.Option(
        0.01, "--min-cost-share", help="Suppress proposals below this share of workload cost."
    ),
    keep_literals: bool = typer.Option(
        False, "--keep-literals", help="Do NOT redact literal values from query text."
    ),
    timeout: int = typer.Option(30, "--timeout", help="Statement timeout in seconds."),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print the statements that would run, then exit without connecting.",
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
    markdown: Path | None = typer.Option(None, "--markdown", help="Write a markdown report."),
    ddl: Path | None = typer.Option(None, "--ddl", help="Write proposed DDL for review."),
) -> None:
    """Propose database optimizations from query history and catalog metadata."""
    since_delta = _parse_since(since)
    timeout = _validate_timeout(timeout)
    schemas = _validate_schemas(schema)

    # --dry-run must work with no credentials at all: it is how you audit what we would run.
    if dry_run:
        try:
            adapter = get_workload_adapter(engine or "postgres")
        except ValueError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=2)
        statements = adapter.introspection_sql()
        if json_out:
            # Honour --json here too: auditing what a tool would run against your database
            # is exactly the kind of thing someone wants to diff or feed to review tooling.
            typer.echo(
                json.dumps(
                    {
                        "engine": engine or "postgres",
                        "statements": [
                            {
                                "capability": s.capability,
                                "privilege_hint": s.privilege_hint,
                                "sql": s.sql,
                            }
                            for s in statements
                        ],
                    },
                    indent=2,
                    sort_keys=True,
                )
            )
            raise typer.Exit(code=0)
        for statement in statements:
            typer.echo(f"-- {statement.capability}: {statement.privilege_hint}")
            typer.echo(statement.sql)
            typer.echo("")
        raise typer.Exit(code=0)

    try:
        params = resolve_connection(
            dsn=dsn,
            engine=engine,
            profile=profile,
            target=target,
            profiles_dir=profiles_dir,
            env=os.environ,
        )
        adapter = get_workload_adapter(params.engine)
    except (ConnectionResolutionError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2)

    typer.echo(f"engine: {params.engine} (credentials from {params.source})", err=True)

    adapter.schemas = schemas
    try:
        adapter.connect(params, timeout)
    except ImportError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2)
    except Exception as exc:  # driver-specific connection failures
        typer.echo(f"Could not connect: {exc}", err=True)
        raise typer.Exit(code=2)

    fetch = adapter.fetch_workload(since_delta, limit)
    workload = ingest(fetch, params.engine, keep_literals=keep_literals)
    db_schema = adapter.fetch_schema(schemas)
    aggregation = aggregate(workload, db_schema, params.engine)
    facts = adapter.fetch_table_facts(schemas, aggregation.tables)
    proposals = adapter.propose(aggregation, facts, workload, min_cost_share=min_cost_share)

    payload = advise_payload(
        proposals,
        workload,
        aggregation,
        engine=params.engine,
        redacted=not keep_literals,
        degraded=adapter.degraded,
    )
    if markdown is not None:
        markdown.write_text(
            render_advise_markdown(
                proposals,
                workload,
                aggregation,
                engine=params.engine,
                redacted=not keep_literals,
                degraded=adapter.degraded,
            )
        )
    if ddl is not None:
        ddl.write_text(adapter.render_ddl(proposals))

    if json_out:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        raise typer.Exit(code=0)

    for capability, reason in adapter.degraded:
        typer.echo(f"reduced coverage — {capability}: {reason}", err=True)
    typer.echo(f"window: {workload.window_description}", err=True)
    # Disclose coverage on every run, not only when it is bad. The markdown and JSON
    # reports always carry these counts; the terminal path should not be the one place a
    # user cannot see how much of their workload was actually understood.
    typer.echo(_coverage_line(workload, aggregation), err=True)
    coverage = _coverage_warning(workload, aggregation)
    if coverage is not None:
        typer.echo(coverage, err=True)

    table = Table(
        title=(
            f"Advise — {params.engine} "
            f"({_plural(len(proposals), 'proposal')}, "
            f"{_plural(len(workload.stats), 'query group')})"
        )
    )
    table.add_column("code")
    table.add_column("conf")
    table.add_column("cost share", justify="right")
    table.add_column("proposal")
    for proposal in proposals:
        share = proposal.evidence.get("cost_share")
        share_text = f"{float(share):.1%}" if isinstance(share, (int, float)) else "—"
        table.add_row(proposal.code, proposal.confidence.value, share_text, proposal.title)
    console.print(table)
    # Proposals are advisory: advise never gates.
    raise typer.Exit(code=0)


def main() -> None:
    """Console-script entry point."""
    app()
