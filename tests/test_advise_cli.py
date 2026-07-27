import json
from pathlib import Path

from typer.testing import CliRunner

from sqlquality.cli import (
    _ambiguity_warning,
    _coverage_line,
    _validate_schemas,
    app,
)
from sqlquality.models import Aggregation, QueryStat, Relation, Workload
from sqlquality.report import advise_payload

runner = CliRunner()


def test_dry_run_prints_statements_and_never_connects(monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("--dry-run must not connect")

    monkeypatch.setattr("sqlquality.workload.postgres.PostgresWorkloadAdapter.connect", explode)
    result = runner.invoke(app, ["advise", "--engine", "postgres", "--dry-run"])
    assert result.exit_code == 0
    assert "pg_stat_statements" in result.stdout


def test_dry_run_needs_no_credentials():
    result = runner.invoke(app, ["advise", "--engine", "postgres", "--dry-run"])
    assert result.exit_code == 0


def test_missing_credentials_exit_2():
    result = runner.invoke(app, ["advise"])
    assert result.exit_code == 2
    assert "--dsn" in result.output


def test_unsupported_engine_exit_2():
    result = runner.invoke(app, ["advise", "--dsn", "postgresql://u@h/db", "--engine", "duckdb"])
    assert result.exit_code == 2


def test_bad_dsn_scheme_exit_2():
    result = runner.invoke(app, ["advise", "--dsn", "mysql://u@h/db"])
    assert result.exit_code == 2


def test_malformed_since_exit_2():
    result = runner.invoke(app, ["advise", "--dsn", "postgresql://u@h/db", "--since", "banana"])
    assert result.exit_code == 2


def test_out_of_range_timeout_exit_2_before_connecting(monkeypatch):
    """Reject rather than silently clamp. 0 means "no limit" to Postgres — the opposite."""

    def explode(*args, **kwargs):
        raise AssertionError("must not connect with an invalid --timeout")

    monkeypatch.setattr("sqlquality.workload.postgres.PostgresWorkloadAdapter.connect", explode)
    for bad in ("0", "-5", "99999"):
        result = runner.invoke(app, ["advise", "--dsn", "postgresql://u@h/db", "--timeout", bad])
        assert result.exit_code == 2, f"--timeout {bad} should be rejected"
        assert "between 1 and 3600" in result.output


def test_two_schemas_are_accepted(monkeypatch):
    """Facts, NDV maps, index lists and qualify() are all keyed by `Relation` now, so a
    second `--schema` no longer aliases same-named tables across schemas together."""
    _stub_adapter(monkeypatch, {"pg_stat_statements": [], "pg_stat_database": [("2026-07-01",)]})
    result = runner.invoke(
        app,
        [
            "advise",
            "--dsn",
            "postgresql://u@h/db",
            "--schema",
            "sales",
            "--schema",
            "staging",
            "--json",
        ],
    )
    assert result.exit_code == 0


def test_duplicate_schemas_are_deduplicated():
    assert _validate_schemas(["public", "public"]) == ("public",)


def test_schema_order_is_preserved():
    assert _validate_schemas(["b", "a"]) == ("b", "a")


def test_coverage_line_reports_ambiguous_separately():
    workload = _workload_with(stats=3, unparseable=1, noise=0)
    aggregation = _aggregation_with(skipped_unqualifiable=1, skipped_ambiguous=2)
    line = _coverage_line(workload, aggregation)
    assert "2 ambiguous" in line


def test_ambiguity_warning_names_the_remedy():
    aggregation = _aggregation_with(skipped_unqualifiable=0, skipped_ambiguous=4)
    warning = _ambiguity_warning(aggregation)
    assert warning is not None
    assert "--schema" in warning


def test_no_ambiguity_means_no_warning():
    """The warning must not fire on the single-schema path, which is every existing run."""
    aggregation = _aggregation_with(skipped_unqualifiable=3, skipped_ambiguous=0)
    assert _ambiguity_warning(aggregation) is None


def test_payload_tables_are_qualified_strings():
    payload = advise_payload(
        [],
        _workload_with(stats=0, unparseable=0, noise=0),
        _aggregation_with(tables=frozenset({Relation("sales", "orders")})),
        engine="postgres",
        redacted=True,
        degraded=[],
    )
    assert payload["analyzed"]["tables"] == ["sales.orders"]
    json.dumps(payload)  # must not raise


def _workload_with(*, stats: int, unparseable: int, noise: int) -> Workload:
    return Workload(
        stats=tuple(
            QueryStat(fingerprint=f"fp{i}", sql="select 1", calls=1, total_time_ms=1.0)
            for i in range(stats)
        ),
        window_description="w",
        skipped_unparseable=unparseable,
        skipped_noise=noise,
    )


def _aggregation_with(
    *,
    skipped_unqualifiable: int = 0,
    skipped_ambiguous: int = 0,
    tables: frozenset[Relation] = frozenset(),
) -> Aggregation:
    return Aggregation(
        usage=(),
        total_cost_ms=0.0,
        skipped_unqualifiable=skipped_unqualifiable,
        tables=tables,
        skipped_ambiguous=skipped_ambiguous,
    )


def test_a_single_schema_is_still_accepted(monkeypatch):
    _stub_adapter(monkeypatch, {"pg_stat_statements": [], "pg_stat_database": [("2026-07-01",)]})
    result = runner.invoke(
        app, ["advise", "--dsn", "postgresql://u@h/db", "--schema", "analytics", "--json"]
    )
    assert result.exit_code == 0


def test_low_coverage_warns_on_stderr(monkeypatch):
    """ "No proposals" must be distinguishable from "I understood almost none of this"."""
    _stub_adapter(
        monkeypatch,
        {
            "pg_stat_statements": [
                ("select id from mystery_table where a = $1", 100, 5000.0, 10),
                ("select id from other_mystery where b = $1", 100, 5000.0, 10),
                ("select id from orders where status = $1", 1, 10.0, 1),
            ],
            "pg_stat_database": [("2026-07-01",)],
            "information_schema.columns": WIDE_COLUMNS,
            "pg_total_relation_size": [("public", "orders", 5_000_000, 10**8)],
            "pg_stats": [("public", "orders", "status", 5000.0)],
            "pg_index": [],
        },
    )
    result = runner.invoke(app, ["advise", "--dsn", "postgresql://u@h/db"])
    assert result.exit_code == 0
    assert "analyzed 1 of 3 query group(s)" in result.output
    assert "low coverage" in result.output
    assert "min-cost-share" in result.output


def test_good_coverage_does_not_warn(monkeypatch):
    _stub_adapter(
        monkeypatch,
        {
            "pg_stat_statements": [
                ("select id from orders where status = $1 and created_at > $2", 100, 5000.0, 10),
            ],
            "pg_stat_database": [("2026-07-01",)],
            "information_schema.columns": WIDE_COLUMNS,
            "pg_total_relation_size": [("public", "orders", 5_000_000, 10**8)],
            "pg_stats": [("public", "orders", "status", 5000.0)],
            "pg_index": [],
        },
    )
    result = runner.invoke(app, ["advise", "--dsn", "postgresql://u@h/db"])
    assert result.exit_code == 0
    assert "low coverage" not in result.output


def _stub_adapter(monkeypatch, rows):
    """Replace connect() with an injected fake querier."""
    from sqlquality.workload.postgres import PostgresWorkloadAdapter

    def fake_connect(self, params, timeout_s):
        self.schemas = ("public",)

        def query(sql, bind):
            for marker, result in rows.items():
                if marker in sql:
                    return result
            return []

        self._query = query

    monkeypatch.setattr(PostgresWorkloadAdapter, "connect", fake_connect)


WIDE_COLUMNS = [
    ("public", "orders", "id", "integer"),
    ("public", "orders", "status", "text"),
    ("public", "orders", "created_at", "timestamp"),
]


#: A table wide enough for ADV006, reachable only by a bare star query.
STAR_ONLY_ROWS = {
    "pg_stat_statements": [("select * from wide_t", 100, 5000.0, 10)],
    "pg_stat_database": [("2026-07-01",)],
    "information_schema.columns": [("public", "wide_t", f"c{i}", "text") for i in range(20)],
    "pg_total_relation_size": [("public", "wide_t", 5_000_000, 10**8)],
    "pg_stats": [],
    "pg_index": [],
}


def test_the_filtered_counter_does_not_claim_introspection_or_ddl(monkeypatch):
    """`DECLARE cur CURSOR FOR SELECT ...` and `COPY (SELECT ...) TO STDOUT` are reads.

    Django's `QuerySet.iterator()` and every psycopg2 server-side cursor emit exactly the
    first form, so a Django shop's hot reads land in this counter — and were then reported
    as "introspection/DDL", i.e. as maintenance traffic nobody needed to care about, on the
    one line that exists to disclose what was lost.
    """
    _stub_adapter(
        monkeypatch,
        {
            "pg_stat_statements": [
                ("declare cur cursor for select id from orders where status = $1", 9, 900.0, 9),
                ("copy (select id from orders where status = $1) to stdout", 5, 500.0, 5),
            ],
            "pg_stat_database": [("2026-07-01",)],
        },
    )
    result = runner.invoke(app, ["advise", "--dsn", "postgresql://u@h/db"])
    assert result.exit_code == 0
    assert "2 filtered" in result.output
    assert "introspection/DDL" not in result.output


def test_a_bare_select_star_table_is_still_introspected(monkeypatch):
    """ADV006 was inert for exactly the case it exists to catch.

    `select * from wide_t` has no predicates, so it contributes no column usage, so the
    table never reached `aggregation.tables`, so `fetch_table_facts` never fetched its
    column count, so the wide-table test could not pass. Adding `where c1 = $1` made the
    proposal appear — which is the wrong way round.
    """
    _stub_adapter(monkeypatch, STAR_ONLY_ROWS)
    result = runner.invoke(app, ["advise", "--dsn", "postgresql://u@h/db", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert any(p["code"] == "ADV006" for p in payload["proposals"]), payload["proposals"]


def test_an_unwritable_markdown_path_exits_2_not_1(monkeypatch, tmp_path):
    """1 is reserved for "findings or gate failure" — a typo must not read as a failed gate.

    It also happened after the whole analysis, so the work was discarded with a traceback.
    """
    _stub_adapter(monkeypatch, STAR_ONLY_ROWS)
    missing = tmp_path / "no" / "such" / "dir" / "report.md"
    result = runner.invoke(
        app, ["advise", "--dsn", "postgresql://u@h/db", "--markdown", str(missing)]
    )
    assert result.exit_code == 2
    assert str(missing) in result.output
    assert "Traceback" not in result.output


def test_a_ddl_path_that_is_a_directory_exits_2_not_1(monkeypatch, tmp_path):
    _stub_adapter(monkeypatch, STAR_ONLY_ROWS)
    result = runner.invoke(app, ["advise", "--dsn", "postgresql://u@h/db", "--ddl", str(tmp_path)])
    assert result.exit_code == 2
    assert str(tmp_path) in result.output
    assert "Traceback" not in result.output


def test_reports_are_written_as_utf8_whatever_the_platform_encoding_is(monkeypatch, tmp_path):
    """Both renderers *always* emit an em dash — render_ddl's header contains one.

    `write_text()` with no encoding uses the platform's preferred encoding, so under an
    ASCII locale (LC_ALL=C) every single `advise --ddl` run raised UnicodeEncodeError. And
    UnicodeEncodeError is a ValueError, not an OSError, so the write handler did not catch
    it and the process exited 1 — the exact CI-misreads-a-healthy-run failure the handler
    was added to prevent. Same bug class as the read side, which got `encoding="utf-8"`.
    """
    real_write_text = Path.write_text

    def ascii_locale(self, data, encoding=None, *args, **kwargs):
        # Emulate a machine whose preferred encoding is ASCII: an omitted encoding is
        # resolved to it, exactly as CPython would.
        resolved = encoding or "ascii"
        data.encode(resolved)
        return real_write_text(self, data, encoding=resolved)

    monkeypatch.setattr(Path, "write_text", ascii_locale)
    _stub_adapter(monkeypatch, STAR_ONLY_ROWS)
    ddl = tmp_path / "out.sql"
    md = tmp_path / "out.md"
    result = runner.invoke(
        app, ["advise", "--dsn", "postgresql://u@h/db", "--ddl", str(ddl), "--markdown", str(md)]
    )
    assert result.exit_code == 0, result.output
    assert "—" in ddl.read_text(encoding="utf-8")
    assert md.read_text(encoding="utf-8")


def test_an_unencodable_report_exits_2_not_1(monkeypatch, tmp_path):
    """utf-8 encodes almost everything — but not a lone surrogate.

    Identifiers reach the DDL from a live catalog, so an unencodable string is not purely
    hypothetical, and the handler must own the whole failure class rather than the one
    exception type that happens to be an OSError.
    """
    from sqlquality.workload.postgres import PostgresWorkloadAdapter

    _stub_adapter(monkeypatch, STAR_ONLY_ROWS)
    monkeypatch.setattr(
        PostgresWorkloadAdapter, "render_ddl", lambda self, proposals: "-- \ud800\n"
    )
    ddl = tmp_path / "out.sql"
    result = runner.invoke(app, ["advise", "--dsn", "postgresql://u@h/db", "--ddl", str(ddl)])
    assert result.exit_code == 2
    assert str(ddl) in result.output
    assert "Traceback" not in result.output


def test_successful_run_exits_0_and_emits_json(monkeypatch):
    _stub_adapter(
        monkeypatch,
        {
            "pg_stat_statements": [
                ("select id from orders where status = $1 and created_at > $2", 100, 5000.0, 10),
            ],
            "pg_stat_database": [("2026-07-01",)],
            "information_schema.columns": WIDE_COLUMNS,
            "pg_total_relation_size": [("public", "orders", 5_000_000, 10**8)],
            "pg_stats": [("public", "orders", "status", 5000.0)],
            "pg_index": [],
        },
    )
    result = runner.invoke(app, ["advise", "--dsn", "postgresql://u@h/db", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["engine"] == "postgres"
    assert payload["redacted"] is True
    assert any(p["code"] == "ADV001" for p in payload["proposals"])


def test_empty_workload_exits_0(monkeypatch):
    _stub_adapter(monkeypatch, {"pg_stat_statements": [], "pg_stat_database": [("2026-07-01",)]})
    result = runner.invoke(app, ["advise", "--dsn", "postgresql://u@h/db", "--json"])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["proposals"] == []


def test_ddl_and_markdown_files_are_written(monkeypatch, tmp_path):
    _stub_adapter(
        monkeypatch,
        {
            "pg_stat_statements": [
                ("select id from orders where status = $1 and created_at > $2", 100, 5000.0, 10),
            ],
            "pg_stat_database": [("2026-07-01",)],
            "information_schema.columns": WIDE_COLUMNS,
            "pg_total_relation_size": [("public", "orders", 5_000_000, 10**8)],
            "pg_stats": [("public", "orders", "status", 5000.0)],
            "pg_index": [],
        },
    )
    ddl = tmp_path / "out.sql"
    md = tmp_path / "out.md"
    result = runner.invoke(
        app, ["advise", "--dsn", "postgresql://u@h/db", "--ddl", str(ddl), "--markdown", str(md)]
    )
    assert result.exit_code == 0
    assert "CREATE INDEX" in ddl.read_text()
    assert "sqlquality advise" in md.read_text()
    assert "REVIEW BEFORE RUNNING" in ddl.read_text()


def test_real_adapter_connection_failure_is_reported_once(monkeypatch):
    """Exercises the real adapter's error string through the CLI's handler.

    Every other connection test monkeypatches `PostgresWorkloadAdapter.connect` itself, so
    none of them cover the seam between the adapter's message and the CLI's prefix — which
    is where a doubled "Could not connect: Could not connect: ..." went unnoticed. This
    patches `psycopg` instead, leaving the real `connect()` to run.
    """
    import sys
    import types

    fake_psycopg = types.ModuleType("psycopg")

    def explode(conninfo, **kwargs):
        # The conninfo is echoed back deliberately, matching
        # test_workload_postgres.py::test_connect_scrubs_a_password_from_a_driver_failure.
        # A fixed message would make the "hunter2 not in output" assertion below unable to
        # fail: it would be asserting the absence of a string nothing ever produced.
        # Measured: with `scrub` (sqlquality.workload.secrets) replaced by the identity
        # function, that assertion now fails, and with a fixed message it did not.
        raise RuntimeError(f"connection failed for conninfo {conninfo}")

    fake_psycopg.connect = explode  # type: ignore[attr-defined]
    fake_psycopg.conninfo = types.SimpleNamespace(  # type: ignore[attr-defined]
        make_conninfo=lambda **kw: " ".join(f"{k}={v}" for k, v in kw.items())
    )
    monkeypatch.setitem(sys.modules, "psycopg", fake_psycopg)

    result = runner.invoke(app, ["advise", "--dsn", "postgresql://u:hunter2@db/x"])
    assert result.exit_code == 2
    assert result.output.count("Could not connect") == 1
    assert "connection failed for conninfo" in result.output
    # And the inline DSN password still must not surface on this path — load-bearing now.
    assert "hunter2" not in result.output


def test_dry_run_honours_json(monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("--dry-run must not connect")

    monkeypatch.setattr("sqlquality.workload.postgres.PostgresWorkloadAdapter.connect", explode)
    result = runner.invoke(app, ["advise", "--engine", "postgres", "--dry-run", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["engine"] == "postgres"
    capabilities = {s["capability"] for s in payload["statements"]}
    assert "workload" in capabilities
    assert all(s["privilege_hint"] for s in payload["statements"])


def test_coverage_is_disclosed_even_on_a_clean_run(monkeypatch):
    """The terminal path should not be the one place coverage is invisible."""
    _stub_adapter(
        monkeypatch,
        {
            "pg_stat_statements": [
                ("select id from orders where status = $1 and created_at > $2", 100, 5000.0, 10),
            ],
            "pg_stat_database": [("2026-07-01",)],
            "information_schema.columns": WIDE_COLUMNS,
            "pg_total_relation_size": [("public", "orders", 5_000_000, 10**8)],
            "pg_stats": [("public", "orders", "status", 5000.0)],
            "pg_index": [],
        },
    )
    result = runner.invoke(app, ["advise", "--dsn", "postgresql://u@h/db"])
    assert result.exit_code == 0
    assert "analyzed 1 of 1 query group(s)" in result.output
    assert "unparseable" in result.output
    assert "low coverage" not in result.output


def test_connection_failure_exits_2(monkeypatch):
    from sqlquality.workload.postgres import PostgresWorkloadAdapter

    def boom(self, params, timeout_s):
        raise RuntimeError("could not connect to server")

    monkeypatch.setattr(PostgresWorkloadAdapter, "connect", boom)
    result = runner.invoke(app, ["advise", "--dsn", "postgresql://u@h/db"])
    assert result.exit_code == 2
    assert "could not connect" in result.output


def test_missing_driver_exits_2_with_install_hint(monkeypatch):
    from sqlquality.workload.postgres import PostgresWorkloadAdapter

    def no_driver(self, params, timeout_s):
        raise ImportError("Install it with: pip install 'sqlquality[postgres]'")

    monkeypatch.setattr(PostgresWorkloadAdapter, "connect", no_driver)
    result = runner.invoke(app, ["advise", "--dsn", "postgresql://u@h/db"])
    assert result.exit_code == 2
    assert "sqlquality[postgres]" in result.output
