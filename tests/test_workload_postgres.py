import inspect
import re
from datetime import timedelta
from pathlib import Path

import pytest

from sqlquality.models import ConnectionParams, Relation
from sqlquality.workload import get_workload_adapter
from sqlquality.workload import postgres as postgres_module
from sqlquality.workload.base import MAX_TIMEOUT_S
from sqlquality.workload.postgres import (
    CAP_INDEXES,
    CAP_NDV,
    CAP_SCHEMA,
    CAP_STATS_RESET,
    CAP_TABLE_FACTS,
    CAP_WORKLOAD,
    PostgresWorkloadAdapter,
)

EXPECTED_CAPABILITIES = {
    CAP_WORKLOAD,
    CAP_STATS_RESET,
    CAP_SCHEMA,
    CAP_TABLE_FACTS,
    CAP_NDV,
    CAP_INDEXES,
}


def test_registry_returns_postgres_adapter():
    adapter = get_workload_adapter("postgres")
    assert adapter.engine == "postgres"


def test_registry_rejects_unsupported_engine_with_a_useful_message():
    with pytest.raises(ValueError) as exc:
        get_workload_adapter("duckdb")
    assert "duckdb" in str(exc.value)


def test_introspection_statements_are_named_and_carry_privilege_hints():
    statements = get_workload_adapter("postgres").introspection_sql()
    assert statements, "adapter must declare its introspection statements for --dry-run"
    capabilities = {s.capability for s in statements}
    assert "workload" in capabilities
    assert all(s.privilege_hint for s in statements)


def test_every_capability_has_a_statement_and_a_hint():
    statements = PostgresWorkloadAdapter().introspection_sql()
    assert {s.capability for s in statements} == EXPECTED_CAPABILITIES
    for statement in statements:
        assert statement.sql.strip()
        assert statement.privilege_hint.strip()


#: Write verbs that must never appear in an introspection statement.
FORBIDDEN_VERBS = (
    "insert",
    "update",
    "delete",
    "create",
    "drop",
    "alter",
    "truncate",
    "grant",
    "revoke",
)


def _write_verbs_in(sql: str) -> set[str]:
    """Write verbs appearing as whole words in ``sql``.

    Word-boundary matching, not whitespace tokenizing. Two earlier attempts at this guard
    each left a hole: ``f" {verb} " in f" {sql.lower()} "`` misses a verb after a newline
    or at the statement start, and collapsing whitespace first still misses one glued to
    punctuation (``";drop table foo"``, ``"(delete from t)"``). ``\\b`` covers every
    position while still leaving ``created_at``, ``deleted`` and ``pg_stat_user_indexes``
    alone, since a following word character means no boundary.
    """
    lowered = sql.lower()
    return {verb for verb in FORBIDDEN_VERBS if re.search(rf"\b{verb}\b", lowered)}


def test_the_write_verb_detector_catches_every_adjacency():
    """Guard the guard. Each case below defeated an earlier version of this detector."""
    # Whitespace-separated — caught by every version.
    assert _write_verbs_in("select 1 from t; drop table foo") == {"drop"}
    # Newline, tab, and leading position — defeated the space-padded version.
    assert _write_verbs_in("select 1 from t limit 1;\ndrop table foo") == {"drop"}
    assert _write_verbs_in("select 1\n\tgrant select on x to y") == {"grant"}
    assert _write_verbs_in("delete from t") == {"delete"}
    # Punctuation-adjacent — defeated the whitespace-collapsing version too.
    assert _write_verbs_in("select 1;drop table foo") == {"drop"}
    assert _write_verbs_in("select(delete from t)") == {"delete"}


def test_the_write_verb_detector_does_not_fire_on_ordinary_identifiers():
    """A guard that flags `created_at` gets weakened to shut it up, so it must not."""
    assert _write_verbs_in("select created_at, updated_at from t") == set()
    assert _write_verbs_in("select deleted, insertion_id from pg_stat_user_indexes") == set()
    assert _write_verbs_in("select 1 from t where a = 2") == set()


def test_no_introspection_statement_writes():
    for statement in PostgresWorkloadAdapter().introspection_sql():
        found = _write_verbs_in(statement.sql)
        assert not found, f"{statement.capability} contains write verb(s): {sorted(found)}"


def test_workload_statement_is_scoped_to_the_current_database():
    sql = PostgresWorkloadAdapter().SQL[CAP_WORKLOAD].lower()
    assert "pg_stat_statements" in sql
    assert "current_database()" in sql
    assert "order by" in sql and "limit" in sql


def test_workload_statement_does_not_filter_on_toplevel():
    """Deliberately not filtered — a documented trade-off, not an oversight.

    A blanket `AND s.toplevel` would deduplicate a `COPY (...) TO` execution under
    `pg_stat_statements.track = all` (see the README's "Prerequisites and limits"), but
    `toplevel = false` is also the *only* way Postgres exposes the SQL executed inside a
    PL/pgSQL function body. Tried and reverted: verified live that the filter made a
    genuinely hot, function-wrapped query disappear from evidence entirely while a colder
    query took its place as a `high`-confidence proposal — confidently wrong, which is worse
    than the double-count it would have fixed. A *narrow* predicate does work — the two
    nested forms are textually distinguishable, see
    `test_the_toplevel_tradeoff_is_documented_as_a_price_not_an_impossibility` — and is
    declined only because naming `s.toplevel` at all raises the floor to PostgreSQL 14. This
    test exists so a future attempt to reintroduce the blanket filter fails here first,
    rather than silently reopening that regression.
    """
    sql = PostgresWorkloadAdapter().SQL[CAP_WORKLOAD].lower()
    assert "toplevel" not in sql


def _source_of(module) -> str:
    return Path(inspect.getsourcefile(module)).read_text(encoding="utf-8")


def test_the_toplevel_tradeoff_is_documented_as_a_price_not_an_impossibility():
    """The reason the filter is absent must be the reason it is actually absent.

    Both the source comment and the README claimed no `s.query` text pattern could separate
    a COPY's nested duplicate from a PL/pgSQL function's nested statement. Measured on
    PostgreSQL 16 under `track = all` that is false: the COPY's nested row keeps its wrapper
    while a function body is recorded bare, and
    `NOT (s.toplevel = false AND s.query ~* '^\\s*COPY\\s*\\(')` removed exactly the
    duplicate (4 rows -> 3). The filter is declined because *naming* `s.toplevel` requires
    PostgreSQL 14 while the supported floor is 13 — a price, not an impossibility. A comment
    that justifies an absence with a false premise is how the wrong decision gets made next
    time, so the claim is pinned here.
    """
    source = _source_of(postgres_module)
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    for text, where in ((source, "the CAP_WORKLOAD comment"), (readme, "the README")):
        lowered = text.lower()
        assert "no `s.query`" not in lowered, f"{where} still claims no predicate can work"
        assert "text pattern separates" not in lowered, f"{where} still claims impossibility"
        assert "text pattern tells the two apart" not in lowered, (
            f"{where} still claims impossibility"
        )
        assert "postgresql 14" in lowered or "postgres 14" in lowered, (
            f"{where} does not state the version cost that is the actual reason"
        )


def test_the_plpgsql_double_count_is_documented_as_a_limitation():
    """The larger, unfixable half of the `track = all` inaccuracy.

    Every PL/pgSQL call is counted twice under `track = all` — measured, `SELECT lc.hot()`
    at 68.21 ms plus its body at 67.67 ms for one execution — which roughly halves every
    `cost_share`. Unlike the `COPY` duplicate no predicate can fix it (the call carries the
    cost, the body carries the predicates), so disclosure is the only honest treatment, and
    the README documented only the smaller `COPY` half.
    """
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    assert "PL/pgSQL function call is counted twice" in readme
    assert "68.21" in readme and "67.67" in readme, (
        "the measurement behind the claim is not recorded"
    )
    assert "no predicate can fix it" in readme


class FakeQuerier:
    """Returns canned rows per capability, keyed by a distinctive SQL substring."""

    def __init__(self, rows_by_marker, fail_markers=()):
        self.rows_by_marker = rows_by_marker
        self.fail_markers = fail_markers
        self.calls = []

    def __call__(self, sql, params):
        self.calls.append((sql, params))
        for marker in self.fail_markers:
            if marker in sql:
                raise RuntimeError(f"permission denied for {marker}")
        for marker, rows in self.rows_by_marker.items():
            if marker in sql:
                return rows
        return []


def _canned(rows_by_capability):
    """A FakeQuerier addressed by capability constant rather than a raw SQL substring.

    Same dispatch as FakeQuerier — a capability's own statement text is already a unique
    substring of itself — just keyed by the name a test actually cares about instead of a
    fragile fragment of SQL.
    """
    return FakeQuerier(
        {
            PostgresWorkloadAdapter.SQL[capability]: rows
            for capability, rows in rows_by_capability.items()
        }
    )


def test_fetch_workload_maps_rows_and_reports_the_window():
    querier = FakeQuerier(
        {
            "pg_stat_statements": [("select id from orders where status = $1", 10, 250.0, 100)],
            "pg_stat_database": [("2026-07-01 00:00:00",)],
        }
    )
    fetch = PostgresWorkloadAdapter(querier=querier).fetch_workload(None, 500)
    assert fetch.rows[0].sql == "select id from orders where status = $1"
    assert fetch.rows[0].calls == 10
    assert fetch.rows[0].total_time_ms == 250.0
    assert "2026-07-01" in fetch.window_description


def test_a_copy_executions_two_rows_both_pass_through_under_track_all():
    """Pins the accepted, documented limitation — not a filter that no longer exists.

    Under `pg_stat_statements.track = all`, one `COPY (SELECT ...) TO ...` execution
    produces two rows: the verbatim top-level statement and its normalised nested query.
    `fetch_workload` deliberately does not filter either out (see
    `test_workload_statement_does_not_filter_on_toplevel`), so both reach `ingest`, which
    fingerprints them differently (a real literal survives redaction in one, `$1` sits in
    the other) and counts the one execution as two query groups at roughly twice its true
    cost. If this assertion ever fails, either the double-count was fixed some other way
    (update the README's "Prerequisites and limits") or the row pass-through broke by
    accident.
    """
    querier = FakeQuerier(
        {
            "pg_stat_statements": [
                ("copy (select id from orders where status = 'x') to stdout", 1, 20.1, 40000),
                ("select id from orders where status = $1", 1, 19.8, 40000),
            ],
            "pg_stat_database": [("2026-07-01",)],
        }
    )
    fetch = PostgresWorkloadAdapter(querier=querier).fetch_workload(None, 500)
    assert len(fetch.rows) == 2


def test_fetch_workload_window_is_honest_that_since_is_not_supported():
    querier = FakeQuerier(
        {
            "pg_stat_statements": [],
            "pg_stat_database": [("2026-07-01 00:00:00",)],
        }
    )
    fetch = PostgresWorkloadAdapter(querier=querier).fetch_workload(timedelta(days=7), 500)
    assert "since stats reset" in fetch.window_description.lower()


def test_a_null_stats_reset_reads_as_an_unknown_time_not_as_None():
    """`stats_reset` is SQL NULL until someone resets statistics — the *default* state.

    The row is then `(None,)`: non-empty, so truthy, so a guard testing the row's emptiness
    lets the None straight through and the window line reads "since stats reset at None".
    That line is the sole statement of what period the advice covers, and ADV002's rationale
    tells the operator to check it before dropping an index.
    """
    querier = FakeQuerier(
        {
            "pg_stat_statements": [("select id from orders where status = $1", 10, 250.0, 100)],
            "pg_stat_database": [(None,)],
        }
    )
    fetch = PostgresWorkloadAdapter(querier=querier).fetch_workload(None, 500)
    assert "an unknown time" in fetch.window_description
    assert "None" not in fetch.window_description


def test_a_denied_stats_reset_statement_also_reads_as_an_unknown_time():
    """The control: the empty-row path must keep working once the guard tests the value.

    Written as a pair with the test above because the obvious fix — `reset[0][0] is not
    None` — reads element 0 of a row that may not exist, and an IndexError on a denied
    grant would cost the whole run for a missing privilege (invariant 4).
    """
    querier = FakeQuerier(
        {"pg_stat_statements": [("select id from orders where status = $1", 10, 250.0, 100)]},
        fail_markers=("pg_stat_database",),
    )
    adapter = PostgresWorkloadAdapter(querier=querier)
    fetch = adapter.fetch_workload(None, 500)
    assert "an unknown time" in fetch.window_description
    assert any(cap == CAP_STATS_RESET for cap, _ in adapter.degraded)


def test_fetch_schema_builds_a_sqlglot_schema_mapping():
    querier = FakeQuerier(
        {
            "information_schema.columns": [
                ("public", "orders", "id", "integer"),
                ("public", "orders", "status", "text"),
                ("public", "customers", "id", "integer"),
            ]
        }
    )
    schema = PostgresWorkloadAdapter(querier=querier).fetch_schema(("public",))
    assert schema == {
        "public": {
            "orders": {"id": "integer", "status": "text"},
            "customers": {"id": "integer"},
        },
    }


def test_fetch_schema_is_nested_by_schema():
    rows = {
        CAP_SCHEMA: [
            ("sales", "orders", "id", "integer"),
            ("sales", "orders", "status", "text"),
            ("staging", "orders", "id", "integer"),
        ]
    }
    adapter = PostgresWorkloadAdapter(querier=_canned(rows))
    assert adapter.fetch_schema(("sales", "staging")) == {
        "sales": {"orders": {"id": "integer", "status": "text"}},
        "staging": {"orders": {"id": "integer"}},
    }


def test_table_facts_do_not_alias_across_schemas():
    """Two same-named tables must keep their own row estimates."""
    rows = {
        CAP_SCHEMA: [("sales", "orders", "id", "integer"), ("staging", "orders", "id", "integer")],
        CAP_TABLE_FACTS: [("sales", "orders", 50_000, 1024), ("staging", "orders", 7, 64)],
        CAP_NDV: [],
    }
    adapter = PostgresWorkloadAdapter(querier=_canned(rows))
    facts = adapter.fetch_table_facts(
        ("sales", "staging"),
        frozenset({Relation("sales", "orders"), Relation("staging", "orders")}),
    )
    assert facts[Relation("sales", "orders")].row_estimate == 50_000
    assert facts[Relation("staging", "orders")].row_estimate == 7


def test_ndv_does_not_leak_between_same_named_tables():
    rows = {
        CAP_SCHEMA: [("sales", "orders", "id", "integer"), ("staging", "orders", "id", "integer")],
        CAP_TABLE_FACTS: [("sales", "orders", 50_000, 1024), ("staging", "orders", 50_000, 1024)],
        CAP_NDV: [("sales", "orders", "id", 5000.0), ("staging", "orders", "id", 3.0)],
    }
    adapter = PostgresWorkloadAdapter(querier=_canned(rows))
    facts = adapter.fetch_table_facts(
        ("sales", "staging"),
        frozenset({Relation("sales", "orders"), Relation("staging", "orders")}),
    )
    assert facts[Relation("sales", "orders")].ndv["id"] == 5000.0
    assert facts[Relation("staging", "orders")].ndv["id"] == 3.0


def test_indexes_do_not_alias_across_schemas():
    rows = {
        CAP_INDEXES: [
            ("sales", "orders", "idx_a", "id", 1, False, False, 0, 100, False, None, False, "..."),
            (
                "staging",
                "orders",
                "idx_b",
                "id",
                1,
                False,
                False,
                9,
                200,
                False,
                None,
                False,
                "...",
            ),
        ]
    }
    adapter = PostgresWorkloadAdapter(querier=_canned(rows))
    indexes = adapter.fetch_indexes(
        ("sales", "staging"),
        frozenset({Relation("sales", "orders"), Relation("staging", "orders")}),
    )
    assert [i.name for i in indexes[Relation("sales", "orders")]] == ["idx_a"]
    assert [i.name for i in indexes[Relation("staging", "orders")]] == ["idx_b"]
    assert indexes[Relation("staging", "orders")][0].scans == 9


def _select_list(sql: str) -> str:
    """The text between `SELECT` and the first `FROM` — the columns actually returned.

    Grepping the whole statement cannot tell a `SELECT` list from a `WHERE` clause, and
    every one of these statements already filtered on the schema before this task — the
    substring the naive version of this check looked for was there from the start, in the
    WHERE clause, regardless of what the SELECT list returned.
    """
    match = re.search(r"select\s+(.*?)\s+from\b", sql, re.IGNORECASE | re.DOTALL)
    assert match, f"no SELECT ... FROM found in statement: {sql!r}"
    return match.group(1)


def test_every_relation_returning_statement_selects_its_schema():
    """A statement that filters on schema but does not return it cannot be keyed by it.

    This is the whole defect class of this task: the rows come back indistinguishable and
    the last one silently wins. An earlier version of this test grepped the *entire*
    statement for `nspname`/`schemaname`/`table_schema` and passed even with the schema
    column stripped from the SELECT list — those substrings were already present in every
    WHERE clause at d0421d0, since each statement already filtered on schema without
    returning it. Restricting the search to the select list (see `_select_list`) is what
    actually pins the defect this task exists to close.
    """
    for capability in (CAP_SCHEMA, CAP_TABLE_FACTS, CAP_NDV, CAP_INDEXES):
        select_list = _select_list(PostgresWorkloadAdapter.SQL[capability])
        assert (
            "nspname" in select_list or "schemaname" in select_list or "table_schema" in select_list
        ), capability


def test_fetch_table_facts_resolves_negative_n_distinct_as_a_row_fraction():
    querier = FakeQuerier(
        {
            "pg_total_relation_size": [("public", "orders", 1000, 8192)],
            "information_schema.columns": [
                ("public", "orders", "id", "integer"),
                ("public", "orders", "s", "text"),
            ],
            "pg_stats": [("public", "orders", "id", 500.0), ("public", "orders", "s", -0.25)],
        }
    )
    facts = PostgresWorkloadAdapter(querier=querier).fetch_table_facts(
        ("public",), frozenset({Relation("public", "orders")})
    )
    assert facts[Relation("public", "orders")].row_estimate == 1000
    assert facts[Relation("public", "orders")].ndv["id"] == 500.0
    # -0.25 means "a quarter of the rows are distinct"
    assert facts[Relation("public", "orders")].ndv["s"] == 250.0


def test_negative_n_distinct_without_a_row_count_is_omitted_not_zeroed():
    """A negative n_distinct is a *fraction*, so it needs the row count to mean anything.

    The two facts come from different statements, so different privileges can hide one and
    not the other. Defaulting the missing row estimate to 0 would fabricate "zero distinct
    values" — a confident, wrong LOW-confidence signal for every proposal on the table.
    """
    querier = FakeQuerier(
        {
            "information_schema.columns": [("public", "orders", "id", "integer")],
            "pg_stats": [("public", "orders", "id", -0.25)],
            # No pg_total_relation_size rows: the row count is unknown.
        }
    )
    facts = PostgresWorkloadAdapter(querier=querier).fetch_table_facts(
        ("public",), frozenset({Relation("public", "orders")})
    )
    assert facts[Relation("public", "orders")].row_estimate is None
    assert "id" not in facts[Relation("public", "orders")].ndv


def test_absolute_n_distinct_survives_a_missing_row_count():
    """A positive n_distinct is an absolute count and needs no row estimate."""
    querier = FakeQuerier(
        {
            "information_schema.columns": [("public", "orders", "id", "integer")],
            "pg_stats": [("public", "orders", "id", 500.0)],
        }
    )
    facts = PostgresWorkloadAdapter(querier=querier).fetch_table_facts(
        ("public",), frozenset({Relation("public", "orders")})
    )
    assert facts[Relation("public", "orders")].ndv["id"] == 500.0


def test_a_never_analyzed_table_reports_an_unknown_row_count():
    """Postgres 14+ stores -1 in reltuples for a table that has never been analyzed.

    Passed through, the small-table gate reads it as a tiny table and suppresses every
    proposal — silently, and precisely in the window after a load or migration when someone
    would run advise. -1 means unknown, and unknown already has a correct path.
    """
    querier = FakeQuerier(
        {
            "information_schema.columns": [("public", "orders", "id", "integer")],
            "pg_total_relation_size": [("public", "orders", -1, 10**9)],
        }
    )
    facts = PostgresWorkloadAdapter(querier=querier).fetch_table_facts(
        ("public",), frozenset({Relation("public", "orders")})
    )
    assert facts[Relation("public", "orders")].row_estimate is None


def test_an_analyzed_empty_table_still_reports_zero():
    """0 is a real answer — analyzed and empty — and must not be conflated with unknown."""
    querier = FakeQuerier(
        {
            "information_schema.columns": [("public", "orders", "id", "integer")],
            "pg_total_relation_size": [("public", "orders", 0, 8192)],
        }
    )
    facts = PostgresWorkloadAdapter(querier=querier).fetch_table_facts(
        ("public",), frozenset({Relation("public", "orders")})
    )
    assert facts[Relation("public", "orders")].row_estimate == 0


def test_fetch_indexes_restores_column_order_from_ordinality():
    """Rows arriving out of order must still yield the right composite order.

    The statement does ORDER BY k.ordinality, but composite column order decides whether a
    proposal is correct, and a fixture that pre-sorts its rows cannot catch a regression
    here. Feeding them backwards is the only way to test the property.
    """
    querier = FakeQuerier(
        {
            "pg_index": [
                (
                    "public",
                    "orders",
                    "idx_status_created",
                    "created_at",
                    2,
                    False,
                    False,
                    0,
                    8192,
                    False,
                    None,
                    False,
                    "CREATE INDEX idx_status_created ON orders (status, created_at)",
                ),
                (
                    "public",
                    "orders",
                    "idx_status_created",
                    "status",
                    1,
                    False,
                    False,
                    0,
                    8192,
                    False,
                    None,
                    False,
                    "CREATE INDEX idx_status_created ON orders (status, created_at)",
                ),
            ]
        }
    )
    indexes = PostgresWorkloadAdapter(querier=querier).fetch_indexes(
        ("public",), frozenset({Relation("public", "orders")})
    )
    assert indexes[Relation("public", "orders")][0].columns == ("status", "created_at")


def test_connect_scrubs_a_password_from_a_driver_failure(monkeypatch):
    """A driver exception is where this class of leak hides — see Task 6.

    psycopg is not believed to echo a password, but the auth-failure path cannot be
    exercised without a live server, so the secret is scrubbed rather than trusted.
    """
    import sys
    import types

    fake_psycopg = types.ModuleType("psycopg")

    def explode(conninfo, **kwargs):
        raise RuntimeError(f"connection failed for conninfo {conninfo}")

    fake_psycopg.connect = explode  # type: ignore[attr-defined]
    fake_psycopg.conninfo = types.SimpleNamespace(  # type: ignore[attr-defined]
        make_conninfo=lambda **kw: " ".join(f"{k}={v}" for k, v in kw.items())
    )
    monkeypatch.setitem(sys.modules, "psycopg", fake_psycopg)

    params = ConnectionParams(
        engine="postgres",
        dsn=None,
        fields={"host": "db", "user": "hans", "password": "hunter2"},
        source="profiles.yml",
    )
    with pytest.raises(ConnectionError) as exc:
        PostgresWorkloadAdapter().connect(params, 30)
    assert "hunter2" not in str(exc.value)
    assert "***" in str(exc.value)
    # And the unscrubbed original must not be reachable through the chain.
    assert exc.value.__cause__ is None
    assert exc.value.__context__ is None


def test_fetch_indexes_groups_columns_in_ordinal_order():
    querier = FakeQuerier(
        {
            "pg_index": [
                (
                    "public",
                    "orders",
                    "orders_pkey",
                    "id",
                    1,
                    True,
                    True,
                    900,
                    4096,
                    False,
                    None,
                    False,
                    "CREATE UNIQUE INDEX orders_pkey ON orders (id)",
                ),
                (
                    "public",
                    "orders",
                    "idx_status_created",
                    "status",
                    1,
                    False,
                    False,
                    0,
                    8192,
                    False,
                    None,
                    False,
                    "CREATE INDEX idx_status_created ON orders (status, created_at)",
                ),
                (
                    "public",
                    "orders",
                    "idx_status_created",
                    "created_at",
                    2,
                    False,
                    False,
                    0,
                    8192,
                    False,
                    None,
                    False,
                    "CREATE INDEX idx_status_created ON orders (status, created_at)",
                ),
            ]
        }
    )
    indexes = PostgresWorkloadAdapter(querier=querier).fetch_indexes(
        ("public",), frozenset({Relation("public", "orders")})
    )
    by_name = {i.name: i for i in indexes[Relation("public", "orders")]}
    assert by_name["idx_status_created"].columns == ("status", "created_at")
    assert by_name["orders_pkey"].is_primary is True
    assert by_name["idx_status_created"].scans == 0


def test_a_denied_statement_degrades_and_names_the_privilege():
    """The fixture supplies real pg_stats rows *and* denies them.

    Both halves are necessary. `assert facts == {} or facts["orders"].ndv == {}` proved
    nothing before: `wanted` always populates facts["orders"], and with no pg_stats rows in
    the fixture `ndv == {}` held whether or not the statement was denied — removing
    `fail_markers` left the test passing. With rows present, the empty ndv is caused by the
    denial and by nothing else.
    """
    querier = FakeQuerier(
        {
            "information_schema.columns": [("public", "orders", "id", "integer")],
            "pg_stats": [("public", "orders", "id", 500.0)],
        },
        fail_markers=("pg_stats",),
    )
    adapter = PostgresWorkloadAdapter(querier=querier)
    facts = adapter.fetch_table_facts(("public",), frozenset({Relation("public", "orders")}))
    assert facts[Relation("public", "orders")].ndv == {}
    assert any(cap == CAP_NDV for cap, _ in adapter.degraded)
    assert any("pg_stats" in reason for _, reason in adapter.degraded)


def test_the_denial_fixture_would_otherwise_have_returned_statistics():
    """Guards the guard above: without the denial the same fixture yields a non-empty ndv,
    so the emptiness there is attributable to the denial rather than to an empty fixture."""
    querier = FakeQuerier(
        {
            "information_schema.columns": [("public", "orders", "id", "integer")],
            "pg_stats": [("public", "orders", "id", 500.0)],
        }
    )
    facts = PostgresWorkloadAdapter(querier=querier).fetch_table_facts(
        ("public",), frozenset({Relation("public", "orders")})
    )
    assert facts[Relation("public", "orders")].ndv == {"id": 500.0}


class _FakeCursor:
    """Enough of a psycopg cursor for connect()'s two session-setup statements.

    `executed` records this cursor's own statements; `log` is the connection-wide
    transcript, so the *relative* order of session setup and later queries is observable.
    Ordering across cursors is the whole point of invariant 2 — a read-only setting applied
    after the first query would be no protection at all.
    """

    def __init__(self, log: list[tuple] | None = None) -> None:
        self.executed: list[tuple] = []
        self._log = log if log is not None else []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.executed.append((sql, params))
        self._log.append((sql, params))

    def fetchall(self):
        return []


class _FakeConnection:
    def __init__(self) -> None:
        self.cursors: list[_FakeCursor] = []
        self.log: list[tuple] = []

    def cursor(self):
        cursor = _FakeCursor(self.log)
        self.cursors.append(cursor)
        return cursor


def _install_fake_psycopg(monkeypatch, seen: dict):
    """A psycopg that records the conninfo it was handed and connects successfully."""
    import sys
    import types

    module = types.ModuleType("psycopg")

    def connect(conninfo, **kwargs):
        seen["conninfo"] = conninfo
        seen["connection"] = _FakeConnection()
        return seen["connection"]

    module.connect = connect  # type: ignore[attr-defined]
    module.conninfo = types.SimpleNamespace(  # type: ignore[attr-defined]
        make_conninfo=lambda **kw: " ".join(f"{k}={v}" for k, v in kw.items())
    )
    monkeypatch.setitem(sys.modules, "psycopg", module)
    return module


def test_profile_tls_settings_are_forwarded_to_the_driver(monkeypatch):
    """A profile asking for verify-full must not connect under libpq's default `prefer`.

    Silently downgrading certificate verification for a tool pitched as safe to point at
    production is the wrong default, and the user was never told.
    """
    seen: dict = {}
    _install_fake_psycopg(monkeypatch, seen)
    params = ConnectionParams(
        engine="postgres",
        dsn=None,
        fields={
            "host": "db",
            "user": "hans",
            "password": "hunter2",
            "sslmode": "verify-full",
            "sslrootcert": "/etc/ssl/ca.crt",
            "sslcert": "/etc/ssl/client.crt",
            "sslkey": "/etc/ssl/client.key",
            "connect_timeout": "10",
        },
        source="profiles.yml",
    )
    PostgresWorkloadAdapter().connect(params, 30)
    conninfo = seen["conninfo"]
    assert "sslmode=verify-full" in conninfo
    assert "sslrootcert=/etc/ssl/ca.crt" in conninfo
    assert "sslcert=/etc/ssl/client.crt" in conninfo
    assert "sslkey=/etc/ssl/client.key" in conninfo
    assert "connect_timeout=10" in conninfo


def test_dropped_profile_keys_are_named_on_stderr(monkeypatch, capsys):
    """A key we cannot forward must be reported, not discarded in silence."""
    seen: dict = {}
    _install_fake_psycopg(monkeypatch, seen)
    params = ConnectionParams(
        engine="postgres",
        dsn=None,
        fields={
            "host": "db",
            "user": "hans",
            "password": "hunter2",
            "search_path": "app",
            "threads": "4",
        },
        source="profiles.yml",
    )
    PostgresWorkloadAdapter().connect(params, 30)
    warning = capsys.readouterr().err
    assert "search_path" in warning
    assert "threads" in warning
    # Key names only — never a value, since one of them could be a secret.
    assert "hunter2" not in warning
    assert "app" not in warning


def test_forwarded_and_mapped_keys_are_not_reported_as_dropped(monkeypatch, capsys):
    seen: dict = {}
    _install_fake_psycopg(monkeypatch, seen)
    params = ConnectionParams(
        engine="postgres",
        dsn=None,
        fields={"host": "db", "dbname": "x", "user": "u", "password": "p", "sslmode": "require"},
        source="profiles.yml",
    )
    PostgresWorkloadAdapter().connect(params, 30)
    assert capsys.readouterr().err == ""


def test_connect_arms_read_only_and_a_statement_timeout_before_the_querier_is_usable(
    monkeypatch,
):
    """Invariant 2, at unit level: neither session-setup statement had a unit guard.

    Removing `SET default_transaction_read_only = on` left the whole default suite green —
    the read-only claim rested solely on an integration test that is deselected by default
    and needs Docker. The statement timeout was asserted nowhere at all, unit or live: a
    session with no timeout can pin a production server on a catalog query, which is the
    opposite of the "safe to point at production" promise.

    `before the querier is usable` is asserted against the connection-wide transcript, not
    just per-cursor: setup applied after the first query would protect nothing, and only the
    relative order can tell the difference.
    """
    seen: dict = {}
    _install_fake_psycopg(monkeypatch, seen)
    adapter = PostgresWorkloadAdapter()
    adapter.connect(
        ConnectionParams(engine="postgres", dsn="postgresql:///x", fields={}, source="--dsn"), 30
    )

    setup = seen["connection"].log[:]
    assert setup == [
        ("SET default_transaction_read_only = on", None),
        ("SELECT set_config('statement_timeout', %s, false)", ("30000ms",)),
    ], setup

    # Usable only now, and every later statement lands after both setup statements.
    adapter._query("SELECT 1", ())
    assert seen["connection"].log[:2] == setup
    assert seen["connection"].log[2] == ("SELECT 1", ())


def test_an_out_of_range_timeout_is_clamped_before_it_reaches_the_session(monkeypatch):
    """The value is `clamp_timeout_ms`'s output in milliseconds, not the raw seconds.

    Passing `7200` through unclamped would arm a two-hour statement timeout, and passing it
    as `7200` rather than `7200ms` would be read by Postgres as milliseconds — a 7-second
    ceiling. Both are silent.
    """
    seen: dict = {}
    _install_fake_psycopg(monkeypatch, seen)
    PostgresWorkloadAdapter().connect(
        ConnectionParams(engine="postgres", dsn="postgresql:///x", fields={}, source="--dsn"), 7200
    )
    assert seen["connection"].log[1][1] == (f"{MAX_TIMEOUT_S * 1000}ms",)


def test_a_conninfo_build_failure_is_scrubbed_like_a_connect_failure(monkeypatch):
    """make_conninfo ran outside the scrubbing envelope, so its message was unfiltered.

    psycopg raises from make_conninfo on an unusable keyword, and that message can quote
    the offending value — which for a `password` keyword is the password.
    """
    import sys
    import types

    module = types.ModuleType("psycopg")

    def never_called(conninfo, **kwargs):
        raise AssertionError("must not reach connect() when the conninfo cannot be built")

    def explode(**kwargs):
        raise RuntimeError(f"invalid connection option: {sorted(kwargs.items())}")

    module.connect = never_called  # type: ignore[attr-defined]
    module.conninfo = types.SimpleNamespace(make_conninfo=explode)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "psycopg", module)

    params = ConnectionParams(
        engine="postgres",
        dsn=None,
        fields={"host": "db", "user": "hans", "password": "hunter2"},
        source="profiles.yml",
    )
    with pytest.raises(ConnectionError) as exc:
        PostgresWorkloadAdapter().connect(params, 30)
    assert "hunter2" not in str(exc.value)
    assert "***" in str(exc.value)
    assert exc.value.__context__ is None


def test_connect_without_psycopg_installed_raises_a_helpful_import_error(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_psycopg(name, *args, **kwargs):
        if name == "psycopg":
            raise ImportError("No module named 'psycopg'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_psycopg)
    adapter = PostgresWorkloadAdapter()
    params = ConnectionParams(
        engine="postgres", dsn="postgresql://u@h/db", fields={}, source="--dsn"
    )
    with pytest.raises(ImportError) as exc:
        adapter.connect(params, 30)
    assert "sqlquality[postgres]" in str(exc.value)


def test_the_ranking_key_ignores_a_boolean_cost_share():
    """The proposal sort had the same missing bool guard as the two renderers.

    `-float(True)` is -1.0, so a proposal carrying a stray True would sort ahead of a
    genuinely hot one at the same confidence — the ordering the CLI presents as "read this
    first".
    """
    from sqlquality.models import Confidence, Proposal

    stray = Proposal(
        code="ADV001",
        title="t",
        rationale="r",
        evidence={"cost_share": True},
        confidence=Confidence.HIGH,
    )
    hot = Proposal(
        code="ADV001",
        title="t",
        rationale="r",
        evidence={"cost_share": 0.5},
        confidence=Confidence.HIGH,
    )
    key = PostgresWorkloadAdapter._ranking_key
    assert key(hot) < key(stray)


def test_the_schema_statement_runs_once_per_run():
    """CAP_SCHEMA was executed by both fetch_schema and fetch_table_facts.

    Twice the catalog work, and — worse — two identical `degraded` entries when it is
    denied, so the user is told the same thing twice.
    """
    querier = FakeQuerier({"information_schema.columns": [("public", "orders", "id", "integer")]})
    adapter = PostgresWorkloadAdapter(querier=querier)
    adapter.fetch_schema(("public",))
    adapter.fetch_table_facts(("public",), frozenset({Relation("public", "orders")}))
    schema_calls = [sql for sql, _ in querier.calls if "information_schema.columns" in sql]
    assert len(schema_calls) == 1


def test_a_denied_schema_statement_is_reported_once_not_twice():
    querier = FakeQuerier({}, fail_markers=("information_schema.columns",))
    adapter = PostgresWorkloadAdapter(querier=querier)
    adapter.fetch_schema(("public",))
    adapter.fetch_table_facts(("public",), frozenset({Relation("public", "orders")}))
    assert [cap for cap, _ in adapter.degraded].count(CAP_SCHEMA) == 1


def test_the_timeout_bounds_have_a_single_definition():
    """Two independent constant pairs drift: the CLI would reject what the adapter accepts,
    or the adapter would silently clamp past the range the CLI's error message promises."""
    import inspect
    from pathlib import Path

    from sqlquality import cli
    from sqlquality.workload import base

    assert cli.MIN_TIMEOUT_S is base.MIN_TIMEOUT_S
    assert cli.MAX_TIMEOUT_S is base.MAX_TIMEOUT_S
    from sqlquality.workload import postgres

    assert postgres.MAX_TIMEOUT_S is base.MAX_TIMEOUT_S
    # And imported, not re-typed — equal literals in two files are still two definitions.
    for module in (cli, postgres):
        source = Path(inspect.getfile(module)).read_text(encoding="utf-8")
        assert str(base.MAX_TIMEOUT_S) not in source, (
            f"{module.__name__} restates the --timeout ceiling as a literal"
        )


def test_fetch_indexes_records_an_expression_index_rather_than_dropping_it():
    """`indkey` holds 0 for an expression column and no pg_attribute row has attnum 0.

    The old inner join therefore discarded those rows, so an index on `lower(status)`
    arrived with an empty column tuple and could not be reasoned about at all.
    """
    querier = FakeQuerier(
        {
            "pg_index": [
                # attname is NULL for the expression column, as a LEFT JOIN yields.
                (
                    "public",
                    "orders",
                    "idx_lower_status",
                    None,
                    1,
                    False,
                    False,
                    3,
                    8192,
                    False,
                    None,
                    True,
                    "CREATE INDEX idx_lower_status ON orders (lower(status))",
                ),
            ]
        }
    )
    indexes = PostgresWorkloadAdapter(querier=querier).fetch_indexes(
        ("public",), frozenset({Relation("public", "orders")})
    )
    index = indexes[Relation("public", "orders")][0]
    assert index.has_expressions is True
    assert index.columns == ()
    assert "lower(status)" in (index.definition or "")


def test_fetch_indexes_records_a_partial_index_predicate():
    querier = FakeQuerier(
        {
            "pg_index": [
                (
                    "public",
                    "orders",
                    "idx_open",
                    "status",
                    1,
                    False,
                    False,
                    7,
                    4096,
                    True,
                    "(shipped_at IS NULL)",
                    False,
                    "CREATE INDEX idx_open ON orders (status) WHERE shipped_at IS NULL",
                ),
            ]
        }
    )
    index = PostgresWorkloadAdapter(querier=querier).fetch_indexes(
        ("public",), frozenset({Relation("public", "orders")})
    )[Relation("public", "orders")][0]
    assert index.is_partial is True
    assert index.predicate == "(shipped_at IS NULL)"
    assert index.columns == ("status",)


def test_fetch_indexes_leaves_a_plain_index_unmarked():
    querier = FakeQuerier(
        {
            "pg_index": [
                (
                    "public",
                    "orders",
                    "idx_status",
                    "status",
                    1,
                    False,
                    False,
                    12,
                    4096,
                    False,
                    None,
                    False,
                    "CREATE INDEX idx_status ON orders (status)",
                ),
            ]
        }
    )
    index = PostgresWorkloadAdapter(querier=querier).fetch_indexes(
        ("public",), frozenset({Relation("public", "orders")})
    )[Relation("public", "orders")][0]
    assert (index.is_partial, index.predicate, index.has_expressions) == (False, None, False)


def test_the_indexes_statement_reads_predicate_and_expression_metadata():
    sql = PostgresWorkloadAdapter().SQL[CAP_INDEXES].lower()
    assert "indpred" in sql, "the partial-index predicate must be selected"
    assert "indexprs" in sql, "expression presence must be selected"
    assert "left join pg_attribute" in sql, (
        "an inner join drops expression columns, whose indkey entry is 0"
    )
