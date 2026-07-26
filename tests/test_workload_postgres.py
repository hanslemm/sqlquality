import re
from datetime import timedelta

import pytest

from sqlquality.models import ConnectionParams
from sqlquality.workload import get_workload_adapter
from sqlquality.workload.postgres import (
    CAP_INDEXES,
    CAP_NDV,
    CAP_SCHEMA,
    CAP_STATS_RESET,
    CAP_TABLE_FACTS,
    CAP_WORKLOAD,
    PostgresWorkloadAdapter,
    _scrub,
    _secrets_for,
    _WITHHELD,
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


def test_fetch_workload_window_is_honest_that_since_is_not_supported():
    querier = FakeQuerier(
        {
            "pg_stat_statements": [],
            "pg_stat_database": [("2026-07-01 00:00:00",)],
        }
    )
    fetch = PostgresWorkloadAdapter(querier=querier).fetch_workload(timedelta(days=7), 500)
    assert "since stats reset" in fetch.window_description.lower()


def test_fetch_schema_builds_a_sqlglot_schema_mapping():
    querier = FakeQuerier(
        {
            "information_schema.columns": [
                ("orders", "id", "integer"),
                ("orders", "status", "text"),
                ("customers", "id", "integer"),
            ]
        }
    )
    schema = PostgresWorkloadAdapter(querier=querier).fetch_schema(("public",))
    assert schema == {
        "orders": {"id": "integer", "status": "text"},
        "customers": {"id": "integer"},
    }


def test_fetch_table_facts_resolves_negative_n_distinct_as_a_row_fraction():
    querier = FakeQuerier(
        {
            "pg_total_relation_size": [("orders", 1000, 8192)],
            "information_schema.columns": [("orders", "id", "integer"), ("orders", "s", "text")],
            "pg_stats": [("orders", "id", 500.0), ("orders", "s", -0.25)],
        }
    )
    facts = PostgresWorkloadAdapter(querier=querier).fetch_table_facts(
        ("public",), frozenset({"orders"})
    )
    assert facts["orders"].row_estimate == 1000
    assert facts["orders"].ndv["id"] == 500.0
    # -0.25 means "a quarter of the rows are distinct"
    assert facts["orders"].ndv["s"] == 250.0


def test_negative_n_distinct_without_a_row_count_is_omitted_not_zeroed():
    """A negative n_distinct is a *fraction*, so it needs the row count to mean anything.

    The two facts come from different statements, so different privileges can hide one and
    not the other. Defaulting the missing row estimate to 0 would fabricate "zero distinct
    values" — a confident, wrong LOW-confidence signal for every proposal on the table.
    """
    querier = FakeQuerier(
        {
            "information_schema.columns": [("orders", "id", "integer")],
            "pg_stats": [("orders", "id", -0.25)],
            # No pg_total_relation_size rows: the row count is unknown.
        }
    )
    facts = PostgresWorkloadAdapter(querier=querier).fetch_table_facts(
        ("public",), frozenset({"orders"})
    )
    assert facts["orders"].row_estimate is None
    assert "id" not in facts["orders"].ndv


def test_absolute_n_distinct_survives_a_missing_row_count():
    """A positive n_distinct is an absolute count and needs no row estimate."""
    querier = FakeQuerier(
        {
            "information_schema.columns": [("orders", "id", "integer")],
            "pg_stats": [("orders", "id", 500.0)],
        }
    )
    facts = PostgresWorkloadAdapter(querier=querier).fetch_table_facts(
        ("public",), frozenset({"orders"})
    )
    assert facts["orders"].ndv["id"] == 500.0


def test_fetch_indexes_restores_column_order_from_ordinality():
    """Rows arriving out of order must still yield the right composite order.

    The statement does ORDER BY k.ordinality, but composite column order decides whether a
    proposal is correct, and a fixture that pre-sorts its rows cannot catch a regression
    here. Feeding them backwards is the only way to test the property.
    """
    querier = FakeQuerier(
        {
            "pg_index": [
                ("orders", "idx_status_created", "created_at", 2, False, False, 0, 8192),
                ("orders", "idx_status_created", "status", 1, False, False, 0, 8192),
            ]
        }
    )
    indexes = PostgresWorkloadAdapter(querier=querier).fetch_indexes(
        ("public",), frozenset({"orders"})
    )
    assert indexes["orders"][0].columns == ("status", "created_at")


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


def test_secrets_for_extracts_the_password_from_an_inline_dsn():
    """The realistic leak shape: a driver reports the bad password on its own.

    It never echoes the whole connection string back, so a whole-DSN token alone would
    never match and DSN connections would have no protection.
    """
    params = ConnectionParams(
        engine="postgres",
        dsn="postgresql://u:hunter2@db:5432/analytics",
        fields={},
        source="--dsn",
    )
    secrets = _secrets_for(params)
    assert "hunter2" in secrets
    realistic = 'connection failed: password authentication failed for user "u" (hunter2)'
    assert "hunter2" not in _scrub(realistic, secrets)


def test_scrub_withholds_rather_than_mangles_an_unredactable_secret():
    """A one-character password would blank every occurrence of that letter.

    Nothing leaks either way, but a message redacted into unreadability is worse than an
    honest refusal to show it.
    """
    mangled = _scrub("a database has an admin at a table", ("a",))
    assert mangled == _WITHHELD
    # A short secret that does not actually appear must not suppress a usable message.
    assert _scrub("connection refused", ("a",)) == "connection refused"


def test_fetch_indexes_groups_columns_in_ordinal_order():
    querier = FakeQuerier(
        {
            "pg_index": [
                ("orders", "orders_pkey", "id", 1, True, True, 900, 4096),
                ("orders", "idx_status_created", "status", 1, False, False, 0, 8192),
                ("orders", "idx_status_created", "created_at", 2, False, False, 0, 8192),
            ]
        }
    )
    indexes = PostgresWorkloadAdapter(querier=querier).fetch_indexes(
        ("public",), frozenset({"orders"})
    )
    by_name = {i.name: i for i in indexes["orders"]}
    assert by_name["idx_status_created"].columns == ("status", "created_at")
    assert by_name["orders_pkey"].is_primary is True
    assert by_name["idx_status_created"].scans == 0


def test_a_denied_statement_degrades_and_names_the_privilege():
    querier = FakeQuerier({"information_schema.columns": []}, fail_markers=("pg_stats",))
    adapter = PostgresWorkloadAdapter(querier=querier)
    facts = adapter.fetch_table_facts(("public",), frozenset({"orders"}))
    assert facts == {} or facts["orders"].ndv == {}
    assert any(cap == CAP_NDV for cap, _ in adapter.degraded)
    assert any("pg_stats" in reason for _, reason in adapter.degraded)


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
