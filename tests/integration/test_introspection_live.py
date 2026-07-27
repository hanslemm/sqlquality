"""Execute every introspection statement against a real server.

The unit suite only checks these statements for drift, which cannot catch a wrong column
name, a wrong join, or a view that does not exist. This is the only place they run.
"""

from __future__ import annotations

import pytest

from sqlquality.models import ConnectionParams, Relation
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
    orders = Relation(schema, "orders")
    adapter.fetch_workload(None, 500)
    adapter.fetch_schema((schema,))
    adapter.fetch_table_facts((schema,), frozenset({orders}))
    adapter.fetch_indexes((schema,), frozenset({orders}))
    assert adapter.degraded == [], f"a statement failed against a real server: {adapter.degraded}"


def test_workload_statement_returns_our_own_queries(adapter):
    """The window line must name a real time, not merely start with the right prefix.

    `"since stats reset at" in ...` was satisfied by the broken `"since stats reset at
    None"` — the prefix is the boilerplate, the payload is the suffix. The `seeded` fixture
    calls `pg_stat_statements_reset()`, but `pg_stat_database.stats_reset` is a *separate*
    counter that a fresh container has never reset, so this assertion is precisely where a
    NULL surfaces live.
    """
    fetch = adapter.fetch_workload(None, 500)
    assert fetch.rows, "pg_stat_statements returned nothing"
    assert "None" not in fetch.window_description, fetch.window_description
    assert "since stats reset at" in fetch.window_description


def test_table_facts_reports_a_real_row_estimate_and_ndv(adapter, seeded):
    _dsn, schema = seeded
    orders = Relation(schema, "orders")
    facts = adapter.fetch_table_facts((schema,), frozenset({orders}))[orders]
    assert facts.row_estimate is not None and facts.row_estimate > 0
    assert "status" in facts.columns
    assert facts.ndv, "pg_stats returned no distinct-value estimates"


def test_indexes_statement_reads_partial_and_expression_metadata(adapter, seeded):
    """The reason Task 2 exists, verified against a real catalog rather than a fixture."""
    _dsn, schema = seeded
    orders = Relation(schema, "orders")
    by_name = {i.name: i for i in adapter.fetch_indexes((schema,), frozenset({orders}))[orders]}

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


def test_fetch_schema_nests_columns_under_schema_then_table(seeded):
    """`fetch_schema`'s nested shape, asserted directly rather than called for side effect.

    A flat `{table: {column: type}}` map cannot tell `public.orders` and `staging.orders`
    apart — a column lookup for one resolves against the union of both column sets. This
    is the specific shape guarantee `qualify()` depends on to keep them separate, and
    nothing in this suite checked it directly before now.
    """
    dsn, _schema = seeded
    adapter = PostgresWorkloadAdapter()
    adapter.schemas = ("public", "staging")
    adapter.connect(ConnectionParams(engine="postgres", dsn=dsn, fields={}, source="--dsn"), 30)
    db_schema = adapter.fetch_schema(("public", "staging"))

    assert "public" in db_schema and "staging" in db_schema, db_schema.keys()
    assert "orders" in db_schema["public"], db_schema["public"].keys()
    assert "orders" in db_schema["staging"], db_schema["staging"].keys()
    assert db_schema["public"]["orders"] is not db_schema["staging"]["orders"], (
        "both schemas' orders table resolved to the same column dict — flat, aliased shape"
    )
    assert "status" in db_schema["public"]["orders"]
    assert "status" in db_schema["staging"]["orders"]


def test_two_same_named_tables_keep_their_own_row_estimates(seeded):
    """The aliasing bug, against real catalog rows rather than canned ones.

    Bare-name keying used to merge `public.orders` and `staging.orders` into one entry, so
    the last catalog row read won the row estimate for both. `seeded` loads them with
    deliberately different row counts (20,000 vs 50,000) specifically so an aliasing
    regression cannot pass this assertion by coincidence.
    """
    dsn, _schema = seeded
    adapter = PostgresWorkloadAdapter()
    adapter.schemas = ("public", "staging")
    adapter.connect(ConnectionParams(engine="postgres", dsn=dsn, fields={}, source="--dsn"), 30)
    facts = adapter.fetch_table_facts(
        ("public", "staging"),
        frozenset({Relation("public", "orders"), Relation("staging", "orders")}),
    )
    public_rows = facts[Relation("public", "orders")].row_estimate
    staging_rows = facts[Relation("staging", "orders")].row_estimate
    assert public_rows is not None and public_rows > 0
    assert staging_rows is not None and staging_rows > 0
    assert public_rows != staging_rows, "both relations reported the same estimate"
