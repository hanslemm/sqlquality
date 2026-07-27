"""Execute every introspection statement against a real server.

The unit suite only checks these statements for drift, which cannot catch a wrong column
name, a wrong join, or a view that does not exist. This is the only place they run.
"""

from __future__ import annotations

import pytest

from sqlquality.models import ConnectionParams
from sqlquality.workload.postgres import PostgresWorkloadAdapter


@pytest.fixture
def adapter(seeded: tuple[str, str]) -> PostgresWorkloadAdapter:
    dsn, schema = seeded
    a = PostgresWorkloadAdapter()
    a.schemas = (schema,)
    a.connect(ConnectionParams(engine="postgres", dsn=dsn, fields={}, source="--dsn"), 30)
    return a


def test_every_introspection_statement_executes(adapter, seeded):
    """No statement may raise, and none may report a degraded capability."""
    _dsn, schema = seeded
    adapter.fetch_workload(None, 500)
    adapter.fetch_schema((schema,))
    adapter.fetch_table_facts((schema,), frozenset({"orders"}))
    adapter.fetch_indexes((schema,), frozenset({"orders"}))
    assert adapter.degraded == [], f"a statement failed against a real server: {adapter.degraded}"


def test_workload_statement_returns_our_own_queries(adapter):
    fetch = adapter.fetch_workload(None, 500)
    assert fetch.rows, "pg_stat_statements returned nothing"
    assert "since stats reset at" in fetch.window_description


def test_table_facts_reports_a_real_row_estimate_and_ndv(adapter, seeded):
    _dsn, schema = seeded
    facts = adapter.fetch_table_facts((schema,), frozenset({"orders"}))["orders"]
    assert facts.row_estimate is not None and facts.row_estimate > 0
    assert "status" in facts.columns
    assert facts.ndv, "pg_stats returned no distinct-value estimates"


def test_indexes_statement_reads_partial_and_expression_metadata(adapter, seeded):
    """The reason Task 2 exists, verified against a real catalog rather than a fixture."""
    _dsn, schema = seeded
    by_name = {i.name: i for i in adapter.fetch_indexes((schema,), frozenset({"orders"}))["orders"]}

    assert by_name["idx_plain"].columns == ("status", "created_at")
    assert by_name["idx_plain"].is_partial is False
    assert by_name["idx_plain"].has_expressions is False

    assert by_name["idx_open"].is_partial is True
    assert "shipped_at IS NULL" in (by_name["idx_open"].predicate or "")

    # The row the shipped statement silently dropped.
    assert by_name["idx_lower_note"].has_expressions is True
    assert "lower(note)" in (by_name["idx_lower_note"].definition or "")

    assert by_name["orders_pkey"].is_primary is True


def test_the_session_really_is_read_only(adapter, seeded):
    """Invariant 2, against a real server: the session must refuse a write."""
    import psycopg

    _dsn, schema = seeded
    with pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
        adapter._query(f"CREATE TABLE {schema}.should_not_exist (x int)", ())
