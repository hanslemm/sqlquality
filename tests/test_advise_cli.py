import dataclasses
import json
from pathlib import Path

from typer.testing import CliRunner

from sqlquality.cli import (
    _ambiguity_warning,
    _coverage_line,
    _coverage_warning,
    _validate_schemas,
    app,
)
from sqlquality.models import Aggregation, Confidence, Proposal, QueryStat, Relation, Workload
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


def test_two_schemas_are_accepted_and_both_reach_every_catalog_query(monkeypatch):
    """A second `--schema` is accepted *and* forwarded to every schema-scoped statement.

    The docstring used to claim this test proved `Relation` keying prevented cross-schema
    aliasing; all it actually asserted was `exit_code == 0`, and `_stub_adapter` overwrote
    `adapter.schemas` with `("public",)` so it could not have proved anything about
    `--schema` at all. It now asserts the bind parameter of each schema-scoped query, and
    names both members rather than checking that the list is non-empty.
    """
    recorded = _stub_adapter(
        monkeypatch, {"pg_stat_statements": [], "pg_stat_database": [("2026-07-01",)]}
    )
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
    # Every statement that takes a schema list: the qualify() schema map, table facts, NDV
    # and the existing-index catalog. Each is checked separately — one of the four carrying
    # both schemas while another silently narrowed to one is the failure being excluded.
    # One marker per statement, each unique to it: `pg_class` would have matched both the
    # table-facts and the index statement and so could not tell which one narrowed.
    for marker in ("information_schema.columns", "pg_total_relation_size", "pg_stats", "pg_index"):
        binds = [bind for sql, bind in recorded if marker in sql]
        assert binds, f"no query ran against {marker}"
        for bind in binds:
            assert bind[0] == ["sales", "staging"], f"{marker} received {bind[0]!r}"


def test_the_resolved_schemas_reach_the_existing_index_query(monkeypatch):
    """`adapter.schemas` is the *only* route `--schema` takes into `fetch_indexes`.

    `fetch_schema`, `fetch_table_facts` and `fetch_ndv` are handed the resolved tuple
    directly by the CLI; `propose()` reads `self.schemas` instead. Drop the CLI's
    `adapter.schemas = schemas` assignment and `fetch_indexes` silently queries `("public",)`
    — which returns zero rows *without raising*, so nothing lands in `degraded`,
    `have_index_data` stays True, and ADV001/ADV007 then claim "no existing index leads with
    them" at HIGH for tables that are fully indexed while ADV002/ADV003 go silent. That is a
    check that could not run, reported as a check that ran and passed.
    """
    recorded = _stub_adapter(
        monkeypatch, {"pg_stat_statements": [], "pg_stat_database": [("2026-07-01",)]}
    )
    result = runner.invoke(
        app,
        ["advise", "--dsn", "postgresql://u@h/db", "--schema", "sales", "--schema", "staging"],
    )
    assert result.exit_code == 0
    index_binds = [bind for sql, bind in recorded if "pg_index" in sql]
    assert index_binds, "the existing-index catalog query never ran"
    assert all(bind[0] == ["sales", "staging"] for bind in index_binds), (
        f"fetch_indexes was called with {[bind[0] for bind in index_binds]!r} — the CLI's "
        "resolved --schema tuple never reached the adapter"
    )


def test_duplicate_schemas_are_deduplicated():
    assert _validate_schemas(["public", "public"]) == ("public",)


def test_schema_order_is_preserved():
    assert _validate_schemas(["b", "a"]) == ("b", "a")


def test_coverage_line_reports_ambiguous_separately():
    workload = _workload_with(stats=3, unparseable=1, noise=0)
    aggregation = _aggregation_with(skipped_unqualifiable=1, skipped_ambiguous=2)
    line = _coverage_line(workload, aggregation)
    assert "2 ambiguous" in line


def test_analyzed_count_excludes_ambiguous_statements():
    """`analyzed N of M` must not double-book an ambiguous statement as both analysed here
    and unexplained in `_coverage_warning`'s share — it cannot honestly be both. Of 4 query
    groups, 2 were dropped as ambiguous and 0 as otherwise unresolvable, so only 2 were
    actually analyzed."""
    workload = _workload_with(stats=4, unparseable=0, noise=0)
    aggregation = _aggregation_with(skipped_unqualifiable=0, skipped_ambiguous=2)
    line = _coverage_line(workload, aggregation)
    assert "analyzed 2 of 4" in line


def test_coverage_warning_fires_when_ambiguity_alone_crosses_the_threshold():
    """100 stats, 25 ambiguous, nothing else unexplained: the true unexplained share is
    25/100 = 25%, above the 20% low-coverage threshold. Before `analyzed_query_groups` subtracted
    `skipped_ambiguous`, the 25 ambiguous statements were counted as both analyzed (inflating
    `considered`) and unexplained, diluting the share to exactly 20% — at the threshold, not
    above it — so the warning never fired precisely when ambiguity was the whole reason
    coverage was bad."""
    workload = _workload_with(stats=100, unparseable=0, noise=0)
    aggregation = _aggregation_with(skipped_unqualifiable=0, skipped_ambiguous=25)
    assert _coverage_warning(workload, aggregation) is not None


def test_ambiguity_warning_names_the_remedy():
    aggregation = _aggregation_with(skipped_unqualifiable=0, skipped_ambiguous=4)
    warning = _ambiguity_warning(aggregation)
    assert warning is not None
    assert "--schema" in warning


def test_no_ambiguity_means_no_warning():
    """The warning must not fire on the single-schema path, which is every existing run."""
    aggregation = _aggregation_with(skipped_unqualifiable=3, skipped_ambiguous=0)
    assert _ambiguity_warning(aggregation) is None


def test_ambiguity_warning_reaches_the_user_on_a_real_run(monkeypatch):
    """`_ambiguity_warning` is unit-tested above in isolation, but nothing else in the
    suite exercises the wiring that actually echoes it from the `advise` command body —
    deleting that echo leaves every other test green. Two introspected schemas both hold
    `orders`; the query names it bare, so it cannot be attributed and must surface here."""
    from sqlquality.workload.postgres import PostgresWorkloadAdapter

    rows = {
        "pg_stat_statements": [
            ("select id from orders where status = $1", 5, 100.0, 5),
        ],
        "pg_stat_database": [("2026-07-01",)],
        "information_schema.columns": [
            ("sales", "orders", "id", "integer"),
            ("sales", "orders", "status", "text"),
            ("staging", "orders", "id", "integer"),
            ("staging", "orders", "status", "text"),
        ],
        "pg_total_relation_size": [],
        "pg_stats": [],
        "pg_index": [],
    }

    def fake_connect(self, params, timeout_s):
        # Unlike `_stub_adapter`, this does not overwrite `self.schemas`: the CLI already
        # set it from `--schema` before calling `connect()`, and this scenario needs both.
        def query(sql, bind):
            for marker, result in rows.items():
                if marker in sql:
                    return result
            return []

        self._query = query

    monkeypatch.setattr(PostgresWorkloadAdapter, "connect", fake_connect)
    result = runner.invoke(
        app,
        ["advise", "--dsn", "postgresql://u@h/db", "--schema", "sales", "--schema", "staging"],
    )
    assert result.exit_code == 0
    assert "could not be attributed" in result.output
    assert "--schema" in result.output


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
    """Replace connect() with an injected fake querier. Returns the recorded `(sql, bind)`.

    Deliberately does **not** assign `self.schemas`. It used to hard-code `("public",)`,
    *after* the CLI had already resolved `--schema` onto the adapter — so every test in this
    module ran `fetch_indexes(("public",), ...)` no matter what schemas it passed, and
    deleting the CLI's `adapter.schemas = schemas` line left the whole default suite green.
    The adapter's own `__init__` default is `("public",)` already, so single-schema tests are
    unaffected; multi-schema ones now exercise the real wiring.
    """
    from sqlquality.workload.postgres import PostgresWorkloadAdapter

    recorded: list[tuple[str, object]] = []

    def fake_connect(self, params, timeout_s):
        def query(sql, bind):
            recorded.append((sql, bind))
            for marker, result in rows.items():
                if marker in sql:
                    return result
            return []

        self._query = query

    monkeypatch.setattr(PostgresWorkloadAdapter, "connect", fake_connect)
    return recorded


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


def test_declared_cursors_and_copy_subqueries_are_analyzed_not_filtered(monkeypatch):
    """`DECLARE cur CURSOR FOR SELECT ...` and `COPY (SELECT ...) TO STDOUT` are reads.

    Django's `QuerySet.iterator()` and every psycopg2 server-side cursor emit exactly the
    first form, so a Django shop's hot reads used to land in the "filtered" counter and be
    thrown away entirely — and reported as "introspection/DDL" on the one line that exists
    to disclose what was lost. `unwrap()` (`sqlquality.workload.fingerprint`) now recovers
    the inner query from both statements before the noise test runs, so each is analyzed
    as its own query group instead.
    """
    _stub_adapter(
        monkeypatch,
        {
            "pg_stat_statements": [
                ("declare cur cursor for select id from orders where status = $1", 9, 900.0, 9),
                (
                    "copy (select id, status from orders where status = $1) to stdout",
                    5,
                    500.0,
                    5,
                ),
            ],
            "pg_stat_database": [("2026-07-01",)],
        },
    )
    result = runner.invoke(app, ["advise", "--dsn", "postgresql://u@h/db"])
    assert result.exit_code == 0
    assert "analyzed 2 of 2" in result.output
    assert "0 filtered" in result.output
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


DBT_FIXTURE = Path(__file__).parent / "fixtures" / "manifest_v12.json"


def test_project_dir_loads_a_manifest_and_discloses_only_on_stderr(monkeypatch, tmp_path):
    """--project-dir must actually reach `load_dbt_context` — replacing `target/manifest.json`
    with garbage used to leave every test in this module green, because nothing exercised the
    option at all. The disclosure it produces must land on stderr: stdout has to stay valid
    JSON under --json, since that is what a later task diffs byte-for-byte against `main`.
    """
    _stub_adapter(monkeypatch, {"pg_stat_statements": [], "pg_stat_database": [("2026-07-01",)]})
    target = tmp_path / "target"
    target.mkdir()
    (target / "manifest.json").write_text(DBT_FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")

    result = runner.invoke(
        app,
        ["advise", "--dsn", "postgresql://u@h/db", "--project-dir", str(tmp_path), "--json"],
    )
    assert result.exit_code == 0
    assert "dbt enrichment" in result.stderr
    assert "dbt enrichment" not in result.stdout
    payload = json.loads(result.stdout)  # stdout must still be pure, parseable JSON
    assert payload["proposals"] == []
    # The payload's manifest path must resolve via --project-dir/target/manifest.json —
    # the same precedence load_dbt_context itself used to load this file.
    assert payload["dbt"]["manifest"] == str(target / "manifest.json")
    assert payload["dbt"]["models"] == 3


def test_project_dir_with_a_broken_manifest_does_not_abort_the_run(monkeypatch, tmp_path):
    """A garbage `target/manifest.json` must degrade to 'no enrichment', not crash a run
    that already did the whole catalog analysis."""
    _stub_adapter(monkeypatch, {"pg_stat_statements": [], "pg_stat_database": [("2026-07-01",)]})
    target = tmp_path / "target"
    target.mkdir()
    (target / "manifest.json").write_text("{not json", encoding="utf-8")

    result = runner.invoke(
        app, ["advise", "--dsn", "postgresql://u@h/db", "--project-dir", str(tmp_path)]
    )
    assert result.exit_code == 0
    assert "dbt enrichment unavailable" in result.stderr
    assert "Traceback" not in result.output


def test_manifest_option_loads_and_discloses_the_source(monkeypatch):
    _stub_adapter(monkeypatch, {"pg_stat_statements": [], "pg_stat_database": [("2026-07-01",)]})
    result = runner.invoke(
        app, ["advise", "--dsn", "postgresql://u@h/db", "--manifest", str(DBT_FIXTURE)]
    )
    assert result.exit_code == 0
    assert str(DBT_FIXTURE) in result.stderr


def test_no_dbt_option_means_no_disclosure_anywhere(monkeypatch):
    """The no-manifest path is every existing `advise` invocation. A later task proves
    byte-identical output against `main` by diffing artifacts, so nothing dbt-shaped may
    appear anywhere in the output without either --project-dir or --manifest.
    """
    _stub_adapter(monkeypatch, {"pg_stat_statements": [], "pg_stat_database": [("2026-07-01",)]})
    result = runner.invoke(app, ["advise", "--dsn", "postgresql://u@h/db"])
    assert result.exit_code == 0
    assert "dbt enrichment" not in result.output


def test_no_manifest_means_no_behaviour_change(monkeypatch):
    """The dbt-free path is first-class, so enrichment must be additive by construction.

    This is a unit-level pin of the same constraint the task proves by diffing a whole run's
    artifacts against `main`: with neither `--project-dir` nor `--manifest`, no proposal may
    carry dbt evidence and the payload must carry no `"dbt"` key at all — not even one set
    to `None` — so the payload stays byte-identical to what `main` produced before this key
    existed, rather than merely equal apart from one known extra key.
    """
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
    assert "dbt" not in result.stderr.lower()
    payload = json.loads(result.stdout)
    assert payload["proposals"], "the scenario must produce at least one proposal to test"
    for proposal in payload["proposals"]:
        assert "dbt_model" not in proposal["evidence"]
    assert "dbt" not in payload


def test_an_unreadable_manifest_via_the_flag_does_not_fail_the_run(monkeypatch, tmp_path):
    """Exit 0 with a disclosure — the catalog work already happened, and dbt is optional.

    Distinct from `test_project_dir_with_a_broken_manifest_does_not_abort_the_run`: that one
    exercises a malformed *file* reached via `--project-dir`; this one exercises `--manifest`
    naming a path that does not exist at all.
    """
    _stub_adapter(monkeypatch, {"pg_stat_statements": [], "pg_stat_database": [("2026-07-01",)]})
    missing = tmp_path / "no.json"
    result = runner.invoke(
        app, ["advise", "--dsn", "postgresql://u@h/db", "--manifest", str(missing), "--json"]
    )
    assert result.exit_code == 0
    assert "dbt enrichment unavailable" in result.stderr
    payload = json.loads(result.stdout)
    assert "dbt" not in payload


def test_the_payload_records_which_manifest_was_used(monkeypatch):
    """The mirror image of `test_no_manifest_means_no_behaviour_change`: with a manifest,
    the `"dbt"` key must be present (not merely non-`None` — `"dbt" in payload` is the
    actual claim, since the no-manifest test now pins its *absence*) and carry the
    manifest path, model count and collision count."""
    _stub_adapter(monkeypatch, {"pg_stat_statements": [], "pg_stat_database": [("2026-07-01",)]})
    result = runner.invoke(
        app,
        ["advise", "--dsn", "postgresql://u@h/db", "--manifest", str(DBT_FIXTURE), "--json"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert "dbt" in payload
    assert payload["dbt"]["manifest"] == str(DBT_FIXTURE)
    # The fixture carries exactly 3 models: stg_orders, orders, customer_orders.
    assert payload["dbt"]["models"] == 3
    assert payload["dbt"]["dropped_collisions"] == 0


def test_the_json_payload_is_serializable_with_a_manifest_loaded(monkeypatch):
    """`--manifest` is parsed by typer as a `Path`, and `json.dumps` cannot encode one.

    If `cli.advise` ever handed a raw `Path` into `dbt_payload["manifest"]` instead of
    `str(resolved_manifest)`, `json.dumps(payload, ...)` would raise `TypeError` — *after*
    the whole catalog analysis had already run, the same late-failure shape the write-
    failure handlers elsewhere in this module exist to avoid. Asserting `isinstance(...,
    str)` pins the actual hazard directly, rather than only failing coincidentally were a
    future encoder ever more lenient than the stdlib's about non-`str` dict values.
    """
    _stub_adapter(monkeypatch, {"pg_stat_statements": [], "pg_stat_database": [("2026-07-01",)]})
    result = runner.invoke(
        app,
        ["advise", "--dsn", "postgresql://u@h/db", "--manifest", str(DBT_FIXTURE), "--json"],
    )
    assert result.exit_code == 0
    assert "Traceback" not in result.output
    payload = json.loads(result.stdout)
    assert isinstance(payload["dbt"]["manifest"], str)
    json.dumps(payload)  # must not raise


def test_the_payload_reports_a_nonzero_dropped_collision_count(monkeypatch, tmp_path):
    """`dropped_collisions` must reflect `DbtContext.dropped_collisions`, not a hardcoded 0.

    Two models here build the same `(schema, table)` in two different databases — the
    cross-database collision `DbtContext.from_project` refuses to guess at (see
    workload/dbt.py). Task 1 counts it precisely so a user can learn a relation was
    silently dropped from the index; this pins that the count actually reaches the CLI
    payload rather than a value that happens to already be right for the shared fixture,
    which has zero collisions and so cannot catch a hardcoded 0.
    """
    manifest = {
        "metadata": {
            "dbt_schema_version": "https://schemas.getdbt.com/dbt/manifest/v12.json",
            "adapter_type": "postgres",
        },
        "nodes": {
            "model.demo.a": {
                "unique_id": "model.demo.a",
                "name": "a",
                "resource_type": "model",
                "config": {"materialized": "table"},
                "compiled_code": "select 1",
                "relation_name": '"prod"."main"."orders"',
                "depends_on": {"macros": [], "nodes": []},
            },
            "model.demo.b": {
                "unique_id": "model.demo.b",
                "name": "b",
                "resource_type": "model",
                "config": {"materialized": "table"},
                "compiled_code": "select 1",
                "relation_name": '"stage"."main"."orders"',
                "depends_on": {"macros": [], "nodes": []},
            },
        },
        "sources": {},
        "parent_map": {"model.demo.a": [], "model.demo.b": []},
        "child_map": {"model.demo.a": [], "model.demo.b": []},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    _stub_adapter(monkeypatch, {"pg_stat_statements": [], "pg_stat_database": [("2026-07-01",)]})
    result = runner.invoke(
        app,
        ["advise", "--dsn", "postgresql://u@h/db", "--manifest", str(manifest_path), "--json"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["dbt"]["dropped_collisions"] == 1
    assert payload["dbt"]["models"] == 0


def test_adv301_and_adv303_only_appear_with_a_manifest(monkeypatch):
    """ADV301/ADV303 need the model graph a manifest carries, so neither may appear without
    one — and at least one must appear with the fixture manifest, or this test proves
    nothing about the wiring at all.

    `customer_orders` in the fixture manifest has no declared consumer, so
    `propose_unused_models` (ADV303) flags it as soon as the workload has *any* usage at
    all — regardless of which schema that usage is in, since the negative check is simply
    "not in aggregation.tables". The workload below queries `public.orders`, wholly
    unrelated to the fixture's `main` schema, so ADV303 firing here is attributable only to
    the manifest being loaded, not to any accidental overlap with the query below.
    """
    rows = {
        "pg_stat_statements": [
            ("select id from orders where status = $1", 5, 100.0, 5),
        ],
        "pg_stat_database": [("2026-07-01",)],
        "information_schema.columns": [
            ("public", "orders", "id", "integer"),
            ("public", "orders", "status", "text"),
        ],
        "pg_total_relation_size": [("public", "orders", 5_000_000, 10**8)],
        "pg_stats": [("public", "orders", "status", 5000.0)],
        "pg_index": [],
    }
    _stub_adapter(monkeypatch, rows)
    without = runner.invoke(app, ["advise", "--dsn", "postgresql://u@h/db", "--json"])
    _stub_adapter(monkeypatch, rows)
    with_dbt = runner.invoke(
        app,
        ["advise", "--dsn", "postgresql://u@h/db", "--manifest", str(DBT_FIXTURE), "--json"],
    )
    assert without.exit_code == 0
    assert with_dbt.exit_code == 0
    without_codes = {p["code"] for p in json.loads(without.stdout)["proposals"]}
    with_codes = {p["code"] for p in json.loads(with_dbt.stdout)["proposals"]}
    assert not ({"ADV301", "ADV303"} & without_codes), without_codes
    assert {"ADV301", "ADV303"} & with_codes, with_codes


def test_enrichment_output_is_resorted_by_the_adapters_ranking_key(monkeypatch):
    """After enriching and extending with ADV301/ADV303, the combined list must be re-sorted
    by `PostgresWorkloadAdapter._ranking_key`, not left in call order — otherwise the
    terminal table, the markdown and the DDL file could each disagree on the order.

    The base adapter is stubbed to return one LOW-confidence proposal; `propose_materialization`
    is stubbed to contribute one HIGH-confidence proposal. Concatenation in call order would
    put the LOW proposal first; the ranking key puts HIGH first. Only a real re-sort produces
    the HIGH-first order asserted below.
    """
    _stub_adapter(monkeypatch, {"pg_stat_statements": [], "pg_stat_database": [("2026-07-01",)]})

    low = Proposal(
        code="ADV999",
        title="low one",
        rationale="r",
        evidence={},
        confidence=Confidence.LOW,
        ddl=None,
    )
    high = Proposal(
        code="ADV001",
        title="high one",
        rationale="r",
        evidence={},
        confidence=Confidence.HIGH,
        ddl=None,
    )

    monkeypatch.setattr(
        "sqlquality.workload.postgres.PostgresWorkloadAdapter.propose",
        lambda self, *a, **k: [low],
    )
    monkeypatch.setattr("sqlquality.cli.enrich_proposals", lambda proposals, context: proposals)
    monkeypatch.setattr("sqlquality.cli.propose_materialization", lambda *a, **k: [high])
    monkeypatch.setattr("sqlquality.cli.propose_unused_models", lambda *a, **k: [])

    result = runner.invoke(
        app,
        ["advise", "--dsn", "postgresql://u@h/db", "--manifest", str(DBT_FIXTURE), "--json"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    codes = [p["code"] for p in payload["proposals"]]
    assert codes == ["ADV001", "ADV999"], codes


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


def test_coverage_warning_is_silent_exactly_at_the_threshold():
    """`share <= _LOW_COVERAGE_FRACTION` returns None, and the boundary is deliberate.

    Nothing pinned which comparison was used, so flipping `<=` to `<` — making the warning
    fire at exactly 20% — passed the whole suite. Either choice is defensible; leaving it
    unpinned is not, because the threshold is what decides whether a user is told their
    proposals may reflect coverage rather than a healthy workload.

    20 unexplained of 100 candidates is exactly 0.2. The pair below it and above it are
    asserted too, so the test fails whichever direction the comparison is flipped rather than
    only one of them.
    """
    # exactly at the threshold: silent
    at = _coverage_warning(_workload_with(stats=80, unparseable=20, noise=0), _aggregation_with())
    assert at is None, "the warning fired at exactly the threshold"

    # one statement worse: 21 of 101 is above 0.2, so it must fire
    above = _coverage_warning(
        _workload_with(stats=80, unparseable=21, noise=0), _aggregation_with()
    )
    assert above is not None, "the warning stayed silent above the threshold"

    # one statement better: 19 of 99 is below 0.2, so it must stay silent
    below = _coverage_warning(
        _workload_with(stats=80, unparseable=19, noise=0), _aggregation_with()
    )
    assert below is None, "the warning fired below the threshold"


#: A workload whose hot predicate and hot join key both land on `public.orders`, plus a join
#: key on an unrelated `public.payments`. Two survivors on one relation is the *normal* shape
#: — the adapter's collapse layer never folds non-prefix column lists — and `payments` is the
#: control: nothing dbt-managed, so nothing about it may change.
TWO_INDEXES_ON_ORDERS_ROWS = {
    "pg_stat_statements": [
        ("select id from orders where status = $1 and created_at > $2", 100, 5000.0, 10),
        (
            "select o.id from orders o join payments p on p.customer_id = o.customer_id",
            80,
            4000.0,
            8,
        ),
    ],
    "pg_stat_database": [("2026-07-01",)],
    "information_schema.columns": [
        ("public", "orders", "id", "integer"),
        ("public", "orders", "status", "text"),
        ("public", "orders", "created_at", "timestamp"),
        ("public", "orders", "customer_id", "integer"),
        ("public", "payments", "id", "integer"),
        ("public", "payments", "customer_id", "integer"),
    ],
    "pg_total_relation_size": [
        ("public", "orders", 5_000_000, 10**8),
        ("public", "payments", 5_000_000, 10**8),
    ],
    "pg_stats": [
        ("public", "orders", "status", 5000.0),
        ("public", "orders", "customer_id", 5000.0),
    ],
    #: An index on the dbt-managed relation that the workload never scans, so ADV002 fires
    #: through the real rules and the run contains a genuine `DROP INDEX` for a dbt model.
    "pg_index": [
        (
            "public",
            "orders",
            "idx_orders_cold",
            "id",
            1,
            False,  # unique
            False,  # primary
            0,  # scans
            8192,
            False,  # partial
            None,  # predicate
            False,  # expressions
            "CREATE INDEX idx_orders_cold ON public.orders USING btree (id)",
        )
    ],
}


def _orders_manifest(tmp_path, materialized="table"):
    """A manifest declaring `public.orders` — the schema the stubbed workload really uses.

    `tests/fixtures/manifest_v12.json` cannot serve here: its `relation_name`s are all schema
    `main`, matching is on the qualified `(schema, table)` pair with no bare-name fallback, so
    a fixture built from a schema the workload never touches would match nothing and every
    assertion below would pass while proving nothing. The database part is deliberately not
    the connected one — `parse_relation_name` drops it.
    """
    manifest = {
        "metadata": {
            "dbt_schema_version": "https://schemas.getdbt.com/dbt/manifest/v12.json",
            "adapter_type": "postgres",
        },
        "nodes": {
            "model.demo.orders": {
                "unique_id": "model.demo.orders",
                "name": "orders",
                "resource_type": "model",
                "config": {"materialized": materialized},
                "compiled_code": "select 1",
                "relation_name": '"analytics"."public"."orders"',
                "depends_on": {"macros": [], "nodes": []},
            }
        },
        "sources": {},
        "parent_map": {"model.demo.orders": []},
        "child_map": {"model.demo.orders": []},
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_adv302_rewrites_an_index_proposal_into_dbt_config_through_the_cli(monkeypatch, tmp_path):
    """ADV302's entire CLI wiring, pinned in the suite CI actually runs.

    Nothing in the default suite pinned that `advise` calls `enrich_proposals` at all:
    replacing that one line with `pass` — which disables the branch's headline feature
    completely — left all 665 tests passing. The only guard was
    `tests/integration/test_advise_live.py`, and `pyproject.toml` sets
    `addopts = "-m 'not integration'"` while `ci.yml` provisioned no Postgres at the time, so CI
    never ran it: ADV302 could have been deleted from the CLI with every check green. This is the
    third instance of that defect class on this branch, so it is pinned here — no Docker, no
    extras, no live database, because the `no-extras` CI job depends on that.

    CI now has an `integration` job that does provision Postgres, which closes the other half of
    that gap. It does not make this test redundant: the live suite needs Docker and the postgres
    extra, so it is still the wrong place to pin CLI wiring that must hold for every contributor
    running a bare `uv run pytest`.
    """
    _stub_adapter(monkeypatch, TWO_INDEXES_ON_ORDERS_ROWS)
    result = runner.invoke(
        app,
        [
            "advise",
            "--dsn",
            "postgresql://u@h/db",
            "--manifest",
            str(_orders_manifest(tmp_path)),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    proposals = json.loads(result.stdout)["proposals"]
    orders = [p for p in proposals if p["evidence"].get("table") == "orders"]
    assert orders, f"the scenario must produce a proposal for public.orders: {proposals}"
    rewritten = [p for p in orders if p["evidence"].get("dbt_index_config") is True]
    assert rewritten, f"no index proposal was expressed as dbt config: {orders}"
    for proposal in orders:
        assert not (proposal["ddl"] or "").upper().lstrip().startswith("CREATE INDEX"), proposal
        assert proposal["evidence"]["dbt_model"] == "model.demo.orders"
    for proposal in rewritten:
        assert "ADV302" in proposal["ddl"], proposal
    # The control: `payments` is not dbt-managed, so its proposal must be untouched.
    [payments] = [p for p in proposals if p["evidence"].get("table") == "payments"]
    assert payments["ddl"] == 'CREATE INDEX ON "public"."payments" ("customer_id");'
    assert "dbt_model" not in payments["evidence"]


def test_two_index_proposals_for_one_dbt_model_yield_one_config_block_through_the_cli(
    monkeypatch, tmp_path
):
    """The end-to-end form of the duplicate-YAML-key data loss.

    Two ordinary proposals on one dbt-managed relation each emitted a complete, standalone
    `indexes:` block. Pasted under one model's `config:` that is a duplicate mapping key, and
    PyYAML — dbt's own parser — silently keeps the last: the other recommended index is
    discarded with no error. Asserted on the `--ddl` artifact because that is the file a human
    copies from.
    """
    import yaml

    _stub_adapter(monkeypatch, TWO_INDEXES_ON_ORDERS_ROWS)
    ddl_path = tmp_path / "out.sql"
    result = runner.invoke(
        app,
        [
            "advise",
            "--dsn",
            "postgresql://u@h/db",
            "--manifest",
            str(_orders_manifest(tmp_path)),
            "--ddl",
            str(ddl_path),
        ],
    )
    assert result.exit_code == 0, result.output
    script = ddl_path.read_text(encoding="utf-8")
    lines = [ln.removeprefix("--").strip() for ln in script.splitlines()]
    assert lines.count("indexes:") == 1, f"one model, one `indexes:` block:\n{script}"

    # Un-comment from the `indexes:` line to the end of that comment run — exactly the region
    # a human copies into a model config — keeping the original indentation, which is what
    # makes it YAML at all.
    body_lines: list[str] = []
    started = False
    for raw in script.splitlines():
        if not raw.startswith("--"):
            if started:
                break
            continue
        content = raw[3:] if raw.startswith("-- ") else raw[2:]
        if content.strip() == "indexes:":
            started = True
        if started:
            body_lines.append(content)
    body = "\n".join(body_lines)
    parsed = yaml.safe_load(body)
    assert [entry["columns"] for entry in parsed["indexes"]] == [
        ["status", "created_at"],
        ["customer_id"],
    ], f"both recommended indexes must survive in the one block:\n{body}"


def _warned_statements(script: str) -> dict[str, bool]:
    """`{statement: whether its block carries a dbt warning}` for every statement in a script.

    Keyed by the statement itself rather than by the relation named in it, because the two are
    not the same thing: `DROP INDEX "public"."idx_orders_cold";` names an index, so filtering
    blocks on the *table* name skipped every drop — which is how a bare `DROP INDEX` for a
    dbt-managed relation sat in the same file that declared that relation dbt-managed.
    """
    result: dict[str, bool] = {}
    for block in script.split("\n\n"):
        lines = block.splitlines()
        statements = [ln for ln in lines if ln.strip() and not ln.startswith("--")]
        if not statements:
            continue
        warned = any("dbt WARNING" in ln for ln in lines)
        for statement in statements:
            result[statement] = warned
    return result


def test_the_ddl_file_warns_beside_every_statement_it_keeps_for_a_dbt_relation(
    monkeypatch, tmp_path
):
    """The constraint: the `--ddl` file must never carry an executable statement for a
    dbt-managed relation without an adjacent comment saying dbt will destroy it.

    `render_ddl` emits only the code/confidence header, the title and the DDL — `rationale`,
    where every ADV302 disclosure used to live, never reaches this file. So one file held a
    config block explaining that raw DDL is destroyed by `dbt run` and, below it, a bare
    `CREATE INDEX` on that same dbt-managed table. Here the manifest declares an
    *unrecognised* materialization, which is a real end-to-end decline path: the DDL is
    deliberately kept, so the warning has to be in the file.
    """
    _stub_adapter(monkeypatch, TWO_INDEXES_ON_ORDERS_ROWS)
    ddl_path = tmp_path / "out.sql"
    result = runner.invoke(
        app,
        [
            "advise",
            "--dsn",
            "postgresql://u@h/db",
            "--manifest",
            str(_orders_manifest(tmp_path, materialized="exotic")),
            "--json",
            "--ddl",
            str(ddl_path),
        ],
    )
    assert result.exit_code == 0, result.output
    script = ddl_path.read_text(encoding="utf-8")
    payload = json.loads(result.stdout)

    # Matched by statement text against the payload, deliberately **not** by looking for the
    # relation's name in the DDL: `DROP INDEX "public"."idx_orders_cold";` names the *index*,
    # so a `'"public"."orders"' in block` filter silently skipped every drop — the exact shape
    # of statement that turned out to be missing its warning.
    warned = _warned_statements(script)
    dbt_statements = {
        p["ddl"] for p in payload["proposals"] if "dbt_model" in p["evidence"] and p["ddl"]
    }
    plain_statements = {
        p["ddl"] for p in payload["proposals"] if "dbt_model" not in p["evidence"] and p["ddl"]
    }
    assert dbt_statements, f"the scenario must keep executable DDL for a dbt model:\n{script}"
    assert plain_statements, "and at least one statement for a relation dbt does not manage"
    assert any(s.upper().startswith("CREATE INDEX") for s in dbt_statements), dbt_statements
    assert any(s.upper().startswith("DROP INDEX") for s in dbt_statements), (
        f"ADV002 must fire for the dbt-managed relation: {dbt_statements}"
    )
    for statement in dbt_statements:
        assert warned.get(statement) is True, f"no dbt warning beside: {statement}\n{script}"
    # Discriminating in the other direction: a renderer that warned on everything would
    # satisfy the loop above and mean nothing.
    for statement in plain_statements:
        assert warned.get(statement) is False, f"spurious dbt warning beside: {statement}"


def test_the_ddl_file_carries_a_warning_on_every_adv302_decline_shape(monkeypatch, tmp_path):
    """The same constraint over all four decline paths at once, including the two no live
    workload reaches (a proposal with no plain column list, and a non-btree access method).

    `propose` is stubbed here precisely because the point is coverage of the *shapes* ADV302
    declines on rather than of the rules that produce them: `_is_index_creating` matches by
    DDL prefix specifically so future rules are covered without being enumerated, so these
    paths must hold for a proposal shape, not for today's four rule codes.
    """
    from sqlquality.workload.postgres import PostgresWorkloadAdapter

    def _p(code, ddl, extra=None):
        evidence = {
            "schema": "public",
            "table": "orders",
            "columns": ("status",),
            "cost_share": 0.5,
        }
        evidence.update(extra or {})
        return Proposal(
            code=code,
            title=f"{code} on public.orders",
            rationale="r.",
            evidence=evidence,
            confidence=Confidence.HIGH,
            ddl=ddl,
        )

    no_columns = _p("ADV008", 'CREATE INDEX ON "public"."orders" (lower("email"));')
    no_columns = dataclasses.replace(
        no_columns, evidence={k: v for k, v in no_columns.evidence.items() if k != "columns"}
    )
    stubbed = [
        _p(
            "ADV004",
            'CREATE INDEX ON "public"."orders" ("region") WHERE "deleted" IS NULL;',
            {"guard_column": "deleted", "guard_predicate": "IS NULL"},
        ),
        no_columns,
        _p("ADV009", 'CREATE INDEX ON "public"."orders" USING gin ("payload");'),
        _p("ADV001", 'CREATE INDEX ON "public"."orders" ("status");'),
    ]
    _stub_adapter(monkeypatch, TWO_INDEXES_ON_ORDERS_ROWS)
    monkeypatch.setattr(PostgresWorkloadAdapter, "propose", lambda self, *a, **k: list(stubbed))

    ddl_path = tmp_path / "out.sql"
    result = runner.invoke(
        app,
        [
            "advise",
            "--dsn",
            "postgresql://u@h/db",
            "--manifest",
            str(_orders_manifest(tmp_path)),
            "--ddl",
            str(ddl_path),
        ],
    )
    assert result.exit_code == 0, result.output
    script = ddl_path.read_text(encoding="utf-8")

    warned = _warned_statements(script)
    assert len(warned) == 3, f"three declines keep their DDL; ADV001 becomes config:\n{script}"
    for statement, has_warning in warned.items():
        assert has_warning, f"no dbt warning beside: {statement}\n{script}"
    assert "ADV004" in script and "ADV009" in script and "ADV008" in script
    # And the one that *was* rewritten carries no bare statement at all.
    assert "-- ADV001 [high" in script
    assert "  indexes:" in script.replace("--", "")


def test_the_terminal_says_adv302_fired(monkeypatch, tmp_path):
    """ADV302 is never a proposal `code`, so an enriched row in the terminal table is
    byte-identical to the same proposal from a dbt-free run — same code, confidence, cost
    share and title — and the terminal never prints `rationale`. Without a line of its own a
    terminal user cannot tell enrichment happened at all.

    On stderr, like every other disclosure this command makes, so stdout stays pure JSON.
    """
    _stub_adapter(monkeypatch, TWO_INDEXES_ON_ORDERS_ROWS)
    result = runner.invoke(
        app,
        [
            "advise",
            "--dsn",
            "postgresql://u@h/db",
            "--manifest",
            str(_orders_manifest(tmp_path)),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "ADV302" in result.stderr
    assert "ADV302" not in result.stdout.split('"proposals"')[0]
    json.loads(result.stdout)  # stdout is still pure JSON

    _stub_adapter(monkeypatch, TWO_INDEXES_ON_ORDERS_ROWS)
    without = runner.invoke(app, ["advise", "--dsn", "postgresql://u@h/db", "--json"])
    assert "ADV302" not in without.stderr, "no manifest, no rewrite, no line"


def test_an_explicit_manifest_wins_over_project_dir_in_both_the_load_and_the_payload(
    monkeypatch, tmp_path
):
    """The precedence existed twice — in `load_dbt_context` and in the CLI's payload builder —
    and swapping it in *either* copy alone left the whole suite green, because no test passed
    both flags. The payload could therefore name a manifest that was never read.

    Both manifests are valid and differ in model count, so the payload's `models` proves which
    file was actually loaded rather than merely which path was formatted into a string.
    """
    project_dir = tmp_path / "proj"
    (project_dir / "target").mkdir(parents=True)
    (project_dir / "target" / "manifest.json").write_text(
        DBT_FIXTURE.read_text(encoding="utf-8"), encoding="utf-8"
    )  # 3 models
    explicit = _orders_manifest(tmp_path)  # 1 model

    _stub_adapter(monkeypatch, {"pg_stat_statements": [], "pg_stat_database": [("2026-07-01",)]})
    result = runner.invoke(
        app,
        [
            "advise",
            "--dsn",
            "postgresql://u@h/db",
            "--project-dir",
            str(project_dir),
            "--manifest",
            str(explicit),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["dbt"]["manifest"] == str(explicit)
    assert payload["dbt"]["models"] == 1, "the *explicit* manifest is the one that was read"
    assert str(explicit) in result.stderr


def test_the_resort_uses_the_resolved_adapters_ranking_key(monkeypatch, tmp_path):
    """The re-sort after enrichment must go through the adapter it resolved.

    It reached `PostgresWorkloadAdapter._ranking_key` — a private classmethod of one specific
    adapter, from the engine-agnostic CLI — while the resolved adapter instance was in scope,
    so a future engine would silently have got Postgres's ordering on the dbt path only.
    Overriding the public hook must change the order the CLI emits.
    """
    from sqlquality.workload.postgres import PostgresWorkloadAdapter

    class _TitleRankingAdapter(PostgresWorkloadAdapter):
        """Stands in for a second engine: same rules, its own reading order."""

        @classmethod
        def ranking_key(cls, proposal):
            return (proposal.title, proposal.code)

    # A *subclass*, resolved the way the CLI resolves any adapter. Patching
    # `PostgresWorkloadAdapter.ranking_key` itself could not discriminate: the un-fixed code
    # named that same class, so the override would have been picked up either way.
    monkeypatch.setattr(
        "sqlquality.cli.get_workload_adapter", lambda engine: _TitleRankingAdapter()
    )
    _stub_adapter(monkeypatch, TWO_INDEXES_ON_ORDERS_ROWS)
    result = runner.invoke(
        app,
        [
            "advise",
            "--dsn",
            "postgresql://u@h/db",
            "--manifest",
            str(_orders_manifest(tmp_path)),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    titles = [p["title"] for p in json.loads(result.stdout)["proposals"]]
    assert titles == sorted(titles), f"the overridden ranking key must be the one used: {titles}"


def test_the_no_manifest_run_contains_no_dbt_conditional_element_anywhere(monkeypatch, tmp_path):
    """A regression guard for the branch's headline compatibility claim.

    "The no-manifest path is byte-identical to `main`" was established by a one-off manual
    diff of all four artifacts; nothing in the suite performed it, so the claim was
    unprotected. A diff against another commit is not something a unit test can do, but the
    equivalent property is: on a run that produces real proposals and writes every artifact,
    no dbt-conditional element may appear in any of them.
    """
    _stub_adapter(monkeypatch, TWO_INDEXES_ON_ORDERS_ROWS)
    ddl_path = tmp_path / "out.sql"
    md_path = tmp_path / "out.md"
    result = runner.invoke(
        app,
        [
            "advise",
            "--dsn",
            "postgresql://u@h/db",
            "--json",
            "--ddl",
            str(ddl_path),
            "--markdown",
            str(md_path),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["proposals"], "a vacuous run would satisfy every assertion below"
    assert ddl_path.read_text(encoding="utf-8").count("CREATE INDEX ON") == 3, (
        "two indexes on orders and one on payments — the count pins that the artifacts are "
        "non-vacuous, since a run with no DDL would satisfy every absence check below"
    )
    assert "dbt" not in payload

    surfaces = {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "ddl": ddl_path.read_text(encoding="utf-8"),
        "markdown": md_path.read_text(encoding="utf-8"),
    }
    # Every dbt-conditional element this branch can emit, each checked against every
    # surface — a single "dbt" substring check would pass while ADV301 leaked.
    forbidden = [
        "dbt",
        "ADV301",
        "ADV302",
        "ADV303",
        "indexes:",
        "dbt_model",
        "dbt_materialized",
        "dbt_index_config",
        "materialized as",
        "manifest",
    ]
    for name, text in surfaces.items():
        for token in forbidden:
            assert token not in text, f"{token!r} leaked into {name} on a dbt-free run"
    for proposal in payload["proposals"]:
        assert not any(k.startswith("dbt") for k in proposal["evidence"]), proposal


def test_redshift_read_only_degradation_reaches_stderr_end_to_end(monkeypatch):
    """The carried-forward item from Tasks 2-4: `connect()`'s read-only degradation was
    recorded correctly from the start, but could never reach a user because `propose()`
    raised `NotImplementedError` before `cli.py` ever got to the loop that prints
    `adapter.degraded` to stderr — see `RedshiftWorkloadAdapter.propose`'s docstring and
    `tests/test_workload_redshift.py`'s `test_the_read_only_degradation_survives_past_
    fetch_workload`, which proved only that `fetch_workload` no longer crashed, and said so
    explicitly rather than claiming the full run worked.

    Now that ADV101-105 make `propose()` a real method, a full `advise` run against a
    Redshift adapter whose `connect()` recorded a read-only degradation must complete and
    print it — this is the first test that exercises `cli.py` end to end for that engine
    rather than stopping at `fetch_workload`.
    """
    from sqlquality.workload.redshift import DEGRADATION_READ_ONLY, RedshiftWorkloadAdapter

    def fake_connect(self, params, timeout_s):
        def query(sql, bind):
            return []

        self._query = query
        self.degraded.append(
            (
                DEGRADATION_READ_ONLY,
                "the session could not be proven read-only (belt-and-braces guard refused) — ***",
            )
        )

    monkeypatch.setattr(RedshiftWorkloadAdapter, "connect", fake_connect)
    result = runner.invoke(app, ["advise", "--engine", "redshift", "--dsn", "postgresql://u@h/db"])
    assert result.exit_code == 0, result.output
    assert f"reduced coverage — {DEGRADATION_READ_ONLY}:" in result.stderr
    assert "could not be proven read-only" in result.stderr


def test_redshift_advise_run_with_no_degradation_prints_none(monkeypatch):
    """Guards the test above: a clean `connect()` must not print a `reduced coverage` line
    at all, so the assertion above is attributable to the recorded degradation, not to
    `cli.py` always printing something regardless of `adapter.degraded`'s contents."""
    from sqlquality.workload.redshift import RedshiftWorkloadAdapter

    def fake_connect(self, params, timeout_s):
        self._query = lambda sql, bind: []

    monkeypatch.setattr(RedshiftWorkloadAdapter, "connect", fake_connect)
    result = runner.invoke(app, ["advise", "--engine", "redshift", "--dsn", "postgresql://u@h/db"])
    assert result.exit_code == 0, result.output
    assert "reduced coverage" not in result.stderr


def _redshift_dbt_run(monkeypatch, *, extra_args=()):
    """A full `advise --engine redshift` run over a fake querier, on the dbt fixture's
    `main.orders` (a `table`-materialized model), returning the CliRunner result.

    The rows are shaped for the real statements: one hot range predicate on
    `main.orders.created_at` so ADV101 fires, and `svv_table_info` facts saying the table is
    sorted on a *different* column so the proposal is not suppressed.
    """
    from sqlquality.workload.redshift import (
        CAP_ADVISOR,
        CAP_SCHEMA,
        CAP_TABLE_FACTS,
        CAP_WORKLOAD,
        RedshiftWorkloadAdapter,
    )

    rows = {
        CAP_WORKLOAD: [("select id from main.orders where created_at > '2026-01-01'", 5_000_000)],
        CAP_SCHEMA: [
            ("main", "orders", "id", "integer"),
            ("main", "orders", "created_at", "timestamp"),
        ],
        CAP_TABLE_FACTS: [("main", "orders", 10_000, 50, 0.0, 0.0, "EVEN", "id", 0.0)],
        CAP_ADVISOR: [],
    }

    def fake_connect(self, params, timeout_s):
        def query(sql, bind):
            for capability, canned in rows.items():
                if RedshiftWorkloadAdapter.SQL[capability] in sql:
                    return canned
            return []

        self._query = query

    monkeypatch.setattr(RedshiftWorkloadAdapter, "connect", fake_connect)
    return runner.invoke(
        app,
        [
            "advise",
            "--engine",
            "redshift",
            "--dsn",
            "postgresql://u@h/db",
            "--schema",
            "main",
            *extra_args,
        ],
    )


def test_redshift_dbt_enrichment_is_disclosed_in_the_terminal_end_to_end(monkeypatch):
    """dbt enrichment must not be invisible to a terminal-only user on Redshift.

    `describe_rewrites` counted only ADV302's config-block rewrite, and no Redshift proposal
    can reach that path — nothing this adapter emits is a `CREATE INDEX` — so the stderr line
    never appeared: the terminal row for an enriched ADV101 was byte-identical to the same
    proposal from a dbt-free run (same code, same confidence, same cost share, same title),
    while the warning that `dbt run` will undo an hours-long full-table rewrite sat in
    `rationale` and in the `--ddl` note, neither of which the terminal prints. This pins the
    whole chain end to end — the evidence flag, the count, and `cli.py`'s echo — rather than
    only the counting function, because each link in it has been broken separately before.
    """
    result = _redshift_dbt_run(monkeypatch, extra_args=("--manifest", str(DBT_FIXTURE)))
    assert result.exit_code == 0, result.output
    assert "ADV101" in result.stdout, "a vacuous run would satisfy the assertion below"
    assert "cannot be expressed as dbt config" in result.stderr
    assert "may undo it" in result.stderr


def test_redshift_run_without_a_manifest_says_nothing_about_dbt(monkeypatch):
    """Control for the test above: the same run with no manifest must print no enrichment
    line at all, so the disclosure is attributable to dbt enrichment rather than to
    `cli.py` always printing something."""
    result = _redshift_dbt_run(monkeypatch)
    assert result.exit_code == 0, result.output
    assert "ADV101" in result.stdout
    assert "dbt" not in result.stderr
    assert "cannot be expressed" not in result.stderr


def test_redshift_dry_run_prints_all_four_statements_and_never_connects(monkeypatch):
    """Task 8's proof that a user can inspect Redshift's introspection SQL — every column
    name unverified against a live cluster (see `redshift.py`'s module docstring) — before
    trusting it with a real connection. Mirrors `test_dry_run_prints_statements_and_never_
    connects` above, for the engine whose SQL genuinely needs this escape hatch most.
    """

    def explode(*args, **kwargs):
        raise AssertionError("--dry-run must not connect")

    monkeypatch.setattr("sqlquality.workload.redshift.RedshiftWorkloadAdapter.connect", explode)
    result = runner.invoke(app, ["advise", "--engine", "redshift", "--dry-run"])
    assert result.exit_code == 0, result.output
    for marker in ("sys_query_history", "svv_columns", "svv_table_info", "svv_alter_table"):
        assert marker in result.stdout


def test_redshift_dry_run_needs_no_credentials():
    result = runner.invoke(app, ["advise", "--engine", "redshift", "--dry-run"])
    assert result.exit_code == 0, result.output
