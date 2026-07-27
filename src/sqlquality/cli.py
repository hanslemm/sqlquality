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
from sqlquality.models import Aggregation, Severity, Workload, cost_share_of
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
from sqlquality.workload.aggregate import aggregate, star_tables
from sqlquality.workload.base import MAX_TIMEOUT_S, MIN_TIMEOUT_S
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


def _analyzed_count(workload: Workload, aggregation: Aggregation) -> int:
    """Query groups whose usage was actually extracted.

    Unresolvable *and* ambiguous groups are both a *subset* of ``workload.stats``, not a
    separate pool, so ``len(stats)`` overstates what was understood unless both are
    subtracted. Omitting `skipped_ambiguous` here used to let an ambiguous statement count
    as "analyzed" in this line while the *same* statement counted as "unexplained" in
    `_coverage_warning`'s share — a statement cannot honestly be both, and the share was the
    one that mattered: it silently deflated toward "coverage is fine", suppressing the
    low-coverage warning exactly when ambiguity was the reason coverage was bad.
    """
    return max(
        0,
        len(workload.stats) - aggregation.skipped_unqualifiable - aggregation.skipped_ambiguous,
    )


def _coverage_line(workload: Workload, aggregation: Aggregation) -> str:
    """One-line coverage disclosure, printed on every run.

    Says "N of M" rather than just "N". Printing ``len(stats)`` as *analyzed* next to
    "2 unresolvable" contradicts itself on a single line — and does so least accurately
    exactly when coverage is worst, which is the situation the line exists to reveal.

    ``skipped_noise`` is reported as "filtered", not "introspection/DDL". The filter is a
    statement-prefix match, so it also swallows `DECLARE cur CURSOR FOR SELECT ...` and
    `COPY (SELECT ...) TO STDOUT` — ordinary reads with real predicates, and what every
    psycopg2 server-side cursor emits. Calling those introspection or DDL told the user
    their hot reads were maintenance traffic. "filtered" claims only what is true.
    """
    return (
        f"analyzed {_analyzed_count(workload, aggregation)} of {len(workload.stats)} "
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
    analyzed = _analyzed_count(workload, aggregation)
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
            "cost-weighted rules (ADV001, ADV004, ADV005, ADV006); the index-hygiene rules "
            "ADV002 and ADV003 carry no cost evidence and are always reported."
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

    payload = advise_payload(
        proposals,
        workload,
        aggregation,
        engine=params.engine,
        redacted=not keep_literals,
        degraded=adapter.degraded,
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


def main() -> None:
    """Console-script entry point."""
    app()
