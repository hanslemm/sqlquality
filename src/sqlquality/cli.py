"""Command-line interface for sqlquality."""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import asdict, replace
from datetime import datetime, timedelta
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
from sqlquality.models import (
    Aggregation,
    Severity,
    Workload,
    analyzed_query_groups,
    cost_share_of,
    proposal_relations,
)
from sqlquality.report import (
    advise_payload,
    gate_payload,
    render_advise_markdown,
    render_html,
    render_markdown,
    render_verify_markdown,
    verdict_label,
    verify_applied_label,
    verify_caveats,
    verify_context,
    verify_mean_cell,
    verify_payload,
    verify_proposal_label,
    verify_summary,
    verify_workload_line,
)
from sqlquality.sqlast import SqlParseError, analyze_sql, parse, strip_jinja
from sqlquality.verify import artifact_incomparabilities, verdicts
from sqlquality.workload import get_workload_adapter
from sqlquality.workload.aggregate import aggregate, star_tables
from sqlquality.workload.base import MAX_TIMEOUT_S, MIN_TIMEOUT_S
from sqlquality.workload.connection import ConnectionResolutionError, resolve_connection
from sqlquality.workload.dbt import (
    describe_rewrites,
    enrich_proposals,
    load_dbt_context,
    propose_materialization,
    propose_unused_models,
    resolve_manifest_path,
)
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


def _write_report_or_exit(path: Path, text: str, flag: str) -> None:
    """Write a rendered report, or exit 2 naming the flag that pointed at the bad path.

    Two failures are folded together here because both used to escape as exit **1**, and
    exit 1 is how `check` and `lint` report real findings — so an unwritable path was
    indistinguishable from a failed gate, and CI would block on a healthy run.

    * ``encoding="utf-8"`` because the default is platform-dependent and these reports
      carry non-ASCII (the gate verdict is an emoji), so an ASCII locale made every
      ``--markdown`` write a guaranteed crash.
    * ``UnicodeError`` alongside ``OSError`` because ``UnicodeEncodeError`` is a
      ``ValueError``: pinning the encoding makes it unlikely, not unreachable, since a lone
      surrogate in an identifier is unencodable in any codec.
    """
    try:
        path.write_text(text, encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        typer.echo(f"Could not write {flag} to {path}: {exc}", err=True)
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

    # Rendered first, then written, so a renderer bug cannot be reported as a write failure.
    if html is not None:
        _write_report_or_exit(Path(html), render_html(report, skipped), "--html")

    if markdown is not None:
        _write_report_or_exit(Path(markdown), render_markdown(report, skipped), "--markdown")

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
                # `--fix` rewrites the user's own source. read_sql_file already reads as
                # UTF-8, so writing without an encoding could round-trip a non-ASCII
                # comment into mojibake, or raise and exit 1 — which `lint` also uses for
                # findings.
                _write_report_or_exit(path, fixed_sql, "--fix")
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
    """Return a timeout in seconds, or exit 2 with a message naming the accepted range.

    The bounds are imported, not restated: the adapter clamps to the same pair as a safety
    net, and two independent literals drift into an error message that promises a range the
    adapter does not honor.
    """
    if not MIN_TIMEOUT_S <= value <= MAX_TIMEOUT_S:
        typer.echo(
            f"--timeout must be between {MIN_TIMEOUT_S} and {MAX_TIMEOUT_S} seconds "
            f"(got {value}). Postgres treats 0 as no limit, which would defeat the flag.",
            err=True,
        )
        raise typer.Exit(code=2)
    return value


def _coverage_line(workload: Workload, aggregation: Aggregation) -> str:
    """One-line coverage disclosure, printed on every run.

    Says "N of M" rather than just "N". Printing ``len(stats)`` as *analyzed* next to
    "2 unresolvable" contradicts itself on a single line — and does so least accurately
    exactly when coverage is worst, which is the situation the line exists to reveal.

    ``skipped_noise`` is reported as "filtered", not "introspection/DDL": it covers session
    control, DDL, maintenance, introspection, a whole-table `COPY ... TO`, and a cursor
    statement that carries no query at all (`FETCH`, `CLOSE`). `DECLARE cur CURSOR FOR
    SELECT ...` and `COPY (SELECT ...) TO STDOUT` — what every psycopg2 server-side cursor
    emits, and ordinary reads with real predicates — are unwrapped to their inner query
    before this count is taken, so they are analysed rather than filtered. Calling
    `skipped_noise` "introspection or DDL" would tell the user their hot reads were
    maintenance traffic; "filtered" claims only what is true.
    """
    return (
        f"analyzed {analyzed_query_groups(workload, aggregation)} of {len(workload.stats)} "
        f"query group(s); skipped {workload.skipped_unparseable} unparseable, "
        f"{workload.skipped_noise} filtered, "
        f"{aggregation.skipped_unqualifiable} unresolvable, "
        f"{aggregation.skipped_ambiguous} ambiguous"
    )


def _coverage_warning(workload: Workload, aggregation: Aggregation) -> str | None:
    """A warning when too little of the workload could be analyzed, else None.

    Noise (introspection, DDL, session control) is excluded from the denominator: those are
    deliberately filtered, not failures to understand. Only statements we tried and failed
    to use count against coverage — an ambiguous statement belongs in that sum exactly like
    an unresolvable one: both are statements `aggregate()` tried and failed to attribute.
    """
    analyzed = analyzed_query_groups(workload, aggregation)
    unexplained = (
        workload.skipped_unparseable
        + aggregation.skipped_unqualifiable
        + aggregation.skipped_ambiguous
    )
    considered = analyzed + unexplained
    if not considered:
        return None
    share = unexplained / considered
    if share <= _LOW_COVERAGE_FRACTION:
        return None
    return (
        f"low coverage: {share:.0%} of candidate statements could not be analyzed "
        f"({workload.skipped_unparseable} unparseable, "
        f"{aggregation.skipped_unqualifiable} unresolvable against the schema, "
        f"{aggregation.skipped_ambiguous} ambiguous across the introspected schemas). "
        "Cost shares are computed against the whole window, so they are diluted and "
        "--min-cost-share is effectively stricter — few or no proposals may reflect "
        "coverage rather than a healthy workload."
    )


def _ambiguity_warning(aggregation: Aggregation) -> str | None:
    """A warning naming the remedy for schema-ambiguous statements, or None.

    Separate from `_coverage_warning`, which fires on a *fraction* and says "coverage is
    low". This fires on any occurrence at all, because the remedy is specific and
    actionable — and because a handful of ambiguous statements can be the hottest ones in
    the workload without moving the coverage fraction enough to trip a threshold.
    """
    if not aggregation.skipped_ambiguous:
        return None
    return (
        f"{aggregation.skipped_ambiguous} statement(s) named a table held by more than one "
        "of the introspected schemas without qualifying it, so they could not be attributed "
        "and were dropped. Qualify the table in the query, or run advise once per --schema."
    )


def _validate_schemas(values: list[str]) -> tuple[str, ...]:
    """Deduplicate `--schema` values, preserving the order they were given in.

    Multiple schemas used to be rejected because every catalog fact was keyed on the bare
    relation name, so two schemas each holding an `orders` aliased into one another. Facts,
    NDV maps, index lists and the `qualify()` schema are all keyed by `Relation` now, so the
    rejection is gone. What survives is a narrower caveat, surfaced by
    `_ambiguity_warning`: a query that says `from orders` when two introspected schemas both
    hold `orders` is genuinely ambiguous, and is counted and reported rather than guessed at.
    """
    return tuple(dict.fromkeys(values))


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
    project_dir: Path | None = typer.Option(
        None,
        "--project-dir",
        help="dbt project dir; reads target/manifest.json to enrich proposals (optional).",
    ),
    manifest: Path | None = typer.Option(
        None, "--manifest", help="Path to a dbt manifest.json. Overrides --project-dir."
    ),
    schema: list[str] = typer.Option(
        ["public"],
        "--schema",
        help="Schema to introspect. Repeat for several: --schema public --schema sales.",
    ),
    since: str | None = typer.Option(
        None, "--since", help="Window, e.g. 7d. Not supported by pg_stat_statements."
    ),
    limit: int = typer.Option(500, "--limit", help="Max query-history rows to read."),
    min_cost_share: float = typer.Option(
        0.01,
        "--min-cost-share",
        # The unqualified "suppress proposals below this share" was a promise the flag
        # cannot keep: propose_unused_indexes and propose_redundant_indexes do not take the
        # parameter, because index hygiene is read out of the catalog and has no cost
        # evidence to weigh. --min-cost-share 5 -- an impossible threshold -- still returned
        # both. Naming the rules it does not reach is the honest fix; filtering them on a
        # share they do not have would be inventing evidence.
        help=(
            "Suppress proposals below this share of workload cost. Applies to the "
            "cost-weighted rules (ADV001, ADV004, ADV005, ADV006, ADV007, ADV008, ADV301 "
            "-- the last only with --project-dir/--manifest -- and, on Redshift, ADV101 "
            "SORTKEY, ADV102 DISTKEY, ADV103 DISTSTYLE ALL); the index-hygiene rules "
            "ADV002 and ADV003, ADV303 (its evidence is absence, not cost, so there is "
            "no share to threshold), and, on Redshift, ADV104 VACUUM/ANALYZE (its evidence "
            "is a catalog measurement about the table itself, not the workload) and ADV105 "
            "(Redshift Advisor's own recommendations, not ours to threshold), carry no "
            "cost evidence and are reported whatever the threshold. ADV303 has its own "
            "non-threshold suppression: it emits nothing when no query usage could be "
            "extracted at all."
        ),
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
    # A `select *` with no predicates produces no column usage, so its table never reaches
    # `aggregation.tables` — and the wide-table rule that exists to catch exactly that
    # query had no column count to test against. Union those tables in before fetching.
    facts = adapter.fetch_table_facts(
        schemas, aggregation.tables | star_tables(workload, db_schema)
    )
    proposals = adapter.propose(aggregation, facts, workload, min_cost_share=min_cost_share)

    # Optional dbt enrichment: neither option given means (None, None) and nothing below
    # fires, so every existing `advise` invocation behaves identically without a manifest —
    # that identity is proved byte-for-byte in tests/test_advise_cli.py and is the
    # constraint this whole block exists to honour.
    dbt_context, dbt_disclosure = load_dbt_context(project_dir, manifest)
    if dbt_disclosure is not None:
        typer.echo(dbt_disclosure, err=True)

    dbt_payload: dict | None = None
    if dbt_context is not None:
        # Rewrite index-creating proposals for dbt-managed relations (ADV302), then add
        # ADV301 (materialize a hot view) and ADV303 (a model the workload never touched).
        # `enrich_proposals` and both `propose_*` calls return an *unsorted-relative-to-
        # each-other* concatenation, so the combined list is re-sorted with the adapter's
        # own ranking key: without this, a proposal `enrich_proposals` downgrades (e.g. a
        # view it strips DDL from, dropping it to LOW) would keep its old, now-wrong
        # position, and the terminal table, the markdown and the DDL file would each see a
        # different order depending on which pass touched them last.
        proposals = enrich_proposals(proposals, dbt_context)
        proposals = proposals + propose_materialization(
            aggregation, dbt_context, min_cost_share=min_cost_share
        )
        proposals = proposals + propose_unused_models(aggregation, dbt_context, workload)
        # `adapter.ranking_key`, not one specific adapter's: ordering is each adapter's own
        # responsibility, and reaching into `PostgresWorkloadAdapter` here meant a future
        # engine would silently get Postgres's ordering on the dbt path while keeping its
        # own everywhere else.
        proposals = sorted(proposals, key=adapter.ranking_key)

        # ADV302 is a rewrite, not a proposal code: an enriched row in the table above is
        # byte-identical to the same proposal from a dbt-free run, and the terminal never
        # prints `rationale`. Without this line a terminal-only user cannot tell that
        # enrichment fired at all.
        rewrite_note = describe_rewrites(proposals)
        if rewrite_note is not None:
            typer.echo(rewrite_note, err=True)

        # The same resolution `load_dbt_context` itself used, so the path disclosed here is
        # exactly the one it loaded — one shared function rather than a second copy of the
        # precedence, which could be (and was) changed in one place only.
        resolved_manifest = resolve_manifest_path(project_dir, manifest)
        assert resolved_manifest is not None  # dbt_context is only ever set when one was given
        dbt_payload = {
            "manifest": str(resolved_manifest),
            "models": len(dbt_context.models),
            "dropped_collisions": dbt_context.dropped_collisions,
        }

    # Union with `aggregation.tables`, not `proposal_relations(proposals)` alone: a
    # relation whose proposal got resolved between two `advise` runs (the index was
    # created, so the rule no longer fires) would otherwise carry no `physical_state`
    # entry at all in the run where it stopped being a finding — which is precisely the
    # run `sqlquality verify` needs it in, to confirm the fix actually landed. Every
    # relation in `aggregation.tables` already has its `CAP_TABLE_FACTS`/`CAP_INDEXES`
    # facts fetched regardless (see `fetch_table_facts`/`fetch_indexes` above and in
    # `propose()`), so this reports more from data already gathered — no new SQL, no
    # behaviour change to `proposals`, `workload` or any non-JSON surface below.
    physical_state_relations = proposal_relations(proposals) | aggregation.tables
    payload = advise_payload(
        proposals,
        workload,
        aggregation,
        engine=params.engine,
        redacted=not keep_literals,
        degraded=adapter.degraded,
        dbt=dbt_payload,
        window_facts=adapter.window_facts(),
        physical_state=adapter.physical_state(physical_state_relations),
    )
    # Both writes happen after the whole analysis, so an unwritable path would otherwise
    # discard the work *and* exit 1 — the code the epilog reserves for "findings or gate
    # failure", which would make CI read a healthy advisory run as a failed gate. Same
    # house pattern as read_sql_file: name the path, exit 2.
    #
    # `encoding="utf-8"` for the same reason profiles.py reads with it: without it Python
    # uses the *platform's* preferred encoding, and both renderers always emit an em dash
    # (render_ddl's header contains one), so under an ASCII locale every --ddl run failed.
    # `UnicodeError` is caught alongside `OSError` because UnicodeEncodeError is a
    # ValueError — it is not an OSError, so the handler let it through and the process
    # exited 1. utf-8 makes that near-unreachable but not unreachable: a lone surrogate in
    # a catalog identifier still cannot be encoded.
    #
    # Only the write is inside the `try`. The renderers run first, so the handler's
    # message — which names a path and claims a write failed — can only fire for something
    # that actually is a write failure.
    if markdown is not None:
        markdown_text = render_advise_markdown(
            proposals,
            workload,
            aggregation,
            engine=params.engine,
            redacted=not keep_literals,
            degraded=adapter.degraded,
            dbt=dbt_payload,
        )
        try:
            markdown.write_text(markdown_text, encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            typer.echo(f"Could not write --markdown {markdown}: {exc}", err=True)
            raise typer.Exit(code=2)
    if ddl is not None:
        ddl_text = adapter.render_ddl(proposals)
        try:
            ddl.write_text(ddl_text, encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            typer.echo(f"Could not write --ddl {ddl}: {exc}", err=True)
            raise typer.Exit(code=2)

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
    ambiguity = _ambiguity_warning(aggregation)
    if ambiguity is not None:
        typer.echo(ambiguity, err=True)

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
        share = cost_share_of(proposal.evidence)
        share_text = f"{share:.1%}" if share is not None else "—"
        table.add_row(proposal.code, proposal.confidence.value, share_text, proposal.title)
    console.print(table)
    # Proposals are advisory: advise never gates.
    raise typer.Exit(code=0)


# --- verify -------------------------------------------------------------------------------
#
# `verify` is fully offline: it reads two `advise --json` artifacts and connects to nothing.
# Every check below refuses rather than guesses, because the one thing this command must never
# do is present an absence or an incommensurability produced by one run's own conditions as a
# measurement about the user's database.


#: Every top-level key `advise_payload` has emitted since this feature landed. An artifact
#: missing any of them predates it (0.3.0, or an intermediate build), and `verify` refuses it
#: rather than deriving a verdict from absent data.
#:
#: **Not just `window`-being-a-string.** That is the 0.3.0 marker specifically, and checking
#: it alone was this refusal's first design. `physical_state` and `query_groups` are emitted
#: as `{}`/`[]` rather than omitted-when-empty *precisely* so that an absent key can mean
#: "this artifact predates the feature" and nothing else (see `advise_payload`'s docstring),
#: which only works if something actually checks for them. An artifact from an intermediate
#: version can carry the `window` object and still lack the rest, and half-understanding such
#: an artifact is the failure this refusal exists to prevent.
_VERIFY_REQUIRED_KEYS: tuple[str, ...] = (
    "analyzed",
    "degraded",
    "engine",
    "physical_state",
    "proposals",
    "query_groups",
    "redacted",
    "skipped",
    "window",
)

#: Every field of the `window` object, `since_duration_seconds` included — it is what
#: `classify_windows` grades `COMPARABLE` on, and an artifact predating it would silently
#: classify as though neither run had ever passed `--since`.
_VERIFY_REQUIRED_WINDOW_KEYS: tuple[str, ...] = (
    "description",
    "engine",
    "limit",
    "since",
    "since_duration_seconds",
    "stats_reset_at",
)


def _refuse(*lines: str) -> NoReturn:
    """Print a refusal to stderr and exit 2 — the house code for "usage or input error".

    Never exit 1: that is what `check` and `lint` use for real findings, so a CI job cannot
    distinguish it from a failed gate. `verify` reports and never gates, so 1 is not a code
    it can produce at all.
    """
    for line in lines:
        typer.echo(line, err=True)
    raise typer.Exit(code=2)


def _read_verify_artifact(path: Path, label: str) -> tuple[bytes, dict[str, object]]:
    """One `advise --json` artifact, as raw bytes and as a parsed object.

    The bytes are returned alongside the parse because byte-identity is one of the two ways
    "the same artifact twice" is detected (see the command body). Every failure exits 2 with
    the argument named: a user who mixed up two paths needs to know *which* one is wrong.
    """
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        _refuse(f"{label}: no such file: {path}")
    except OSError as exc:
        _refuse(f"{label}: could not read {path}: {exc}")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        _refuse(
            f"{label} ({path}) is not valid UTF-8, so it cannot be the JSON `advise --json` "
            "writes. Regenerate it with `sqlquality advise --json > artifact.json`."
        )
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        _refuse(
            f"{label} ({path}) is not valid JSON: {exc}. verify reads two `advise --json` "
            "artifacts; regenerate this one with `sqlquality advise --json > artifact.json`."
        )
    if not isinstance(parsed, dict):
        _refuse(
            f"{label} ({path}) is valid JSON but not a JSON object (found "
            f"{type(parsed).__name__}), so it is not an `advise --json` artifact."
        )
    return raw, parsed


def _verify_artifact_complaints(payload: dict[str, object]) -> list[str]:
    """Every way `payload` falls short of the artifact contract `verify` needs, or `[]`.

    Names what is actually wrong rather than reporting the first problem found: an artifact
    can be missing several keys at once, and a user regenerating it wants the whole list.
    """
    complaints: list[str] = []
    missing = [key for key in _VERIFY_REQUIRED_KEYS if key not in payload]
    if missing:
        complaints.append("missing top-level key(s) " + ", ".join(missing))
    window = payload.get("window")
    if "window" in payload and not isinstance(window, dict):
        complaints.append(
            f"`window` is a {type(window).__name__}, not an object — 0.3.0 wrote a prose "
            "sentence here, and a sentence cannot be compared across two runs"
        )
    elif isinstance(window, dict):
        missing_window = [key for key in _VERIFY_REQUIRED_WINDOW_KEYS if key not in window]
        if missing_window:
            complaints.append("missing `window` field(s) " + ", ".join(missing_window))
    return complaints


def _verify_reset_instant(payload: dict[str, object]) -> datetime | None:
    """`window["stats_reset_at"]` as a `datetime`, or `None` when it cannot be established.

    `None` for an absent, null, non-string or unparseable value — never a substituted
    instant. The one thing this feeds is the swapped-artifact refusal below, and refusing a
    pair on a value that could not be read would be its own false claim.
    """
    window = payload.get("window")
    if not isinstance(window, dict):
        return None
    value = window.get("stats_reset_at")
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


@app.command()
def verify(
    before: Path = typer.Argument(
        ..., dir_okay=False, help="The earlier `advise --json` artifact (the baseline)."
    ),
    after: Path = typer.Argument(
        ..., dir_okay=False, help="The later `advise --json` artifact, taken after the change."
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
    markdown: Path | None = typer.Option(None, "--markdown", help="Write a markdown report."),
) -> None:
    """Diff two `advise --json` artifacts: was each proposal applied, and did it help?

    Fully offline — it reads two files and never opens a connection, so anyone reviewing a
    change can run it, including people who will never have production access.

    Run order is taken from the argument order: BEFORE, then AFTER. An advise artifact
    carries no run timestamp (deliberately: its bytes are reproducible, which is what makes
    two runs diffable at all), so verify cannot detect a swapped pair in general. The one
    exception is refused: a server's statistics-reset instant cannot move backwards, so when
    both artifacts report a stats_reset_at and AFTER's is earlier, the two were passed the
    wrong way round.

    Reports and never gates: exit 0 whenever a comparison was reported, exit 2 when it was
    refused (unreadable or pre-0.4.0 artifacts, the same artifact twice, two engines, or two
    runs that disagree about literal redaction). There is no --gate flag.
    """
    before_raw, before_payload = _read_verify_artifact(before, "BEFORE")
    after_raw, after_payload = _read_verify_artifact(after, "AFTER")

    # Checked before anything compares the two, so a user who saved a baseline with an older
    # sqlquality is told exactly that rather than being handed a verdict derived from keys
    # the artifact never carried.
    for label, path, payload in (
        ("BEFORE", before, before_payload),
        ("AFTER", after, after_payload),
    ):
        complaints = _verify_artifact_complaints(payload)
        if complaints:
            _refuse(
                f"{label} ({path}) was not produced by sqlquality 0.4.0 or later: "
                + "; ".join(complaints)
                + ". Every verdict verify reports is derived from those keys, so regenerate "
                "the artifact with `sqlquality advise --json` on 0.4.0 or later rather than "
                "letting a verdict rest on absent data."
            )

    # The same artifact twice would report every proposal unchanged — which reads as a
    # finding rather than as a mistake. Two arms: the same resolved path, and byte-identical
    # content (two genuinely distinct runs cannot be byte-identical, because
    # pg_stat_statements' counters accumulate, so identical bytes are sound evidence of the
    # same run rather than of a real no-change result).
    if before.resolve() == after.resolve():
        _refuse(
            f"BEFORE and AFTER are the same file ({before.resolve()}). Comparing a run with "
            "itself reports every proposal unchanged, which reads as a finding rather than "
            "as a mistake. Pass artifacts from two different advise runs."
        )
    if before_raw == after_raw:
        _refuse(
            f"BEFORE ({before}) and AFTER ({after}) are byte-identical, so they are the same "
            "run saved twice. Two genuinely distinct runs cannot be: pg_stat_statements' "
            "counters accumulate, so even an unchanged workload moves the numbers. Comparing "
            "a run with itself would report every proposal unchanged, which reads as a "
            "finding rather than as a mistake."
        )

    # Two engines, and two disagreeing (or unreadable) `redacted` flags, are both
    # incommensurability rather than weak evidence: the two artifacts do not share a
    # coordinate system, so there is no confidence at which a query-group comparison could be
    # stated. Rendered once here rather than repeated on every proposal's note.
    incomparabilities = artifact_incomparabilities(before_payload, after_payload)
    if incomparabilities:
        _refuse(
            *[item.detail for item in incomparabilities],
            "verify refuses this pair rather than grading it at low confidence: a weak "
            "comparison and a meaningless one are different things. Regenerate both "
            "artifacts from the same engine, with the same --keep-literals setting.",
        )

    before_reset = _verify_reset_instant(before_payload)
    after_reset = _verify_reset_instant(after_payload)
    if before_reset is not None and after_reset is not None:
        try:
            reversed_pair = after_reset < before_reset
        except TypeError:
            # One instant is timezone-aware and the other naive, so they cannot be ordered at
            # all. That is "cannot establish", not "in order" — disclosed by the run-order
            # caveat every report carries, never resolved by guessing here.
            reversed_pair = False
        if reversed_pair:
            _refuse(
                f"AFTER's window reports an earlier stats_reset_at ({after_reset}) than "
                f"BEFORE's ({before_reset}). A server's statistics-reset instant cannot move "
                "backwards, so these two artifacts were passed in the wrong order — pass the "
                "earlier run first. (This is the only swapped pair verify can detect: an "
                "advise artifact carries no run timestamp, so run order is otherwise taken "
                "from the argument order and never guessed at.)"
            )

    results = verdicts(before_payload, after_payload)
    context = verify_context(before_payload, after_payload)

    # Rendered first, then written, so a renderer bug cannot be reported as a write failure —
    # and exit 2 rather than 1 for an unwritable path, exactly as `advise` and `check` do.
    if markdown is not None:
        _write_report_or_exit(
            markdown, render_verify_markdown(results, before_payload, after_payload), "--markdown"
        )

    if json_out:
        typer.echo(
            json.dumps(
                verify_payload(results, before_payload, after_payload), indent=2, sort_keys=True
            )
        )
        raise typer.Exit(code=0)

    # Every caveat, on stderr, before the table: each one names a condition under which the
    # table below means less than it appears to. The nested-window case is the important one —
    # it is the common Postgres path, and on it a real improvement is understated, so a user
    # who is not told will read `unchanged` as "it did not work".
    for caveat in verify_caveats(context):
        typer.echo(caveat, err=True)
    typer.echo(verify_workload_line(context), err=True)

    summary = verify_summary(results)
    outcomes = summary["outcomes"]
    improved = outcomes["improved"] if isinstance(outcomes, dict) else 0
    table = Table(
        title=(
            f"Verify — {context.before.engine} "
            f"({_plural(len(results), 'proposal')}, "
            f"{summary['applied']} applied, {improved} improved)"
        )
    )
    table.add_column("proposal")
    table.add_column("applied")
    table.add_column("outcome")
    table.add_column("mean per call", justify="right")
    table.add_column("conf")
    for verdict in results:
        table.add_row(
            verify_proposal_label(verdict.key),
            verify_applied_label(verdict.applied),
            verdict.outcome.value,
            verify_mean_cell(verdict),
            verdict.confidence.value,
        )
    console.print(table)
    # The notes carry the substance the five columns cannot: the applied-but-unchanged
    # headline, a collision, a limit mismatch, a degraded read. Printing the table without
    # them would be this command withholding exactly what it exists to say.
    for verdict in results:
        if verdict.note:
            console.print(f"[cyan]{verify_proposal_label(verdict.key)}[/]: {verdict.note}")
    # verify reports; it never gates. There is no --gate flag (deliberately out of scope).
    raise typer.Exit(code=0)


def main() -> None:
    """Console-script entry point."""
    app()
