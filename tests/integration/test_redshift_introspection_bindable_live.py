"""Prove every Redshift introspection statement is *bindable*, not merely syntax-checked.

The unit suite only parses these statements with sqlglot, which can catch a malformed
statement but not a parameter the driver cannot bind — and whether a statement's parameters
can be bound at all is the driver's job, not Redshift's. A bind failure happens identically
whether the FROM clause names `svv_table_info` or `pg_class`.

**Why this creates throwaway stand-in tables instead of just running the real statements.**
The first version of this file ran each statement as-is against `postgres:16`, on the
theory that the expected, accepted failure would be `UndefinedTable` (the view genuinely
does not exist here) with a parameter-binding failure (`IndeterminateDatatype`) being the
one thing that must not happen instead. Verified empirically that this does not
discriminate anything: Postgres's analyzer resolves table references before parameter
types, so a statement whose FROM-clause table does not exist *always* fails with
`UndefinedTable`, regardless of whether its parameters would otherwise bind — reintroducing
the exact bug this file exists to catch (a bare `(%s IS NULL OR start_time >= %s)`) still
produced only `UndefinedTable` against a real, missing `sys_query_history`. So this instead
creates a same-named, same-shaped (but empty) real table for each capability, which lets
Postgres's analyzer get past the FROM clause and actually resolve every parameter's type —
at which point the *bindable* form succeeds outright (no exception, zero rows) and the
*unbindable* form still fails with `IndeterminateDatatype`, exactly as it does for a
freshly-loaded relation on a real Redshift cluster. See `CAP_WORKLOAD`'s comment in
`redshift.py` for the production bug this reproduces and fixes.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timezone

import pytest

from sqlquality.models import ConnectionParams
from sqlquality.workload.redshift import (
    CAP_ADVISOR,
    CAP_SCHEMA,
    CAP_TABLE_FACTS,
    CAP_WORKLOAD,
    RedshiftWorkloadAdapter,
)

#: One throwaway real table per capability's FROM clause, shaped to match every column
#: each statement selects or filters on — just enough for Postgres's analyzer to resolve
#: the statement fully, including its parameters. Types are the obvious Postgres
#: equivalent of AWS's documented Redshift column types; the *rows* are irrelevant (every
#: table stays empty), only whether the statement can be prepared and run against them.
_SHIM_DDL = {
    "sys_query_history": """
        CREATE TABLE sys_query_history (
            query_text text, elapsed_time bigint, database_name text,
            status text, start_time timestamptz
        )
    """,
    "svv_columns": """
        CREATE TABLE svv_columns (
            schema_name text, table_name text, column_name text, data_type text
        )
    """,
    "svv_table_info": """
        CREATE TABLE svv_table_info (
            "schema" text, "table" text, tbl_rows bigint, size bigint,
            unsorted float8, stats_off float8, diststyle text, sortkey1 text,
            skew_rows float8
        )
    """,
    "svv_alter_table_recommendations": """
        CREATE TABLE svv_alter_table_recommendations (
            database_name text, schema_name text, table_name text, type text,
            current_ddl text, recommended_ddl text
        )
    """,
}

#: Representative binds for each capability, matching what its own fetch_* method passes.
#: CAP_WORKLOAD gets two entries — with and without a `--since` cutoff — since those are
#: two different parameter shapes over the wire and the bug this file guards against only
#: reproduced in the no-`--since` (both-NULL) shape.
_BINDS: dict[str, list[tuple[object, ...]]] = {
    CAP_WORKLOAD: [
        (None, None, 10),
        (datetime.now(timezone.utc), datetime.now(timezone.utc), 10),
    ],
    CAP_SCHEMA: [(["public"],)],
    CAP_TABLE_FACTS: [(["public"], ["orders"])],
    CAP_ADVISOR: [(["public"], ["orders"])],
}


@pytest.fixture(scope="module")
def shim_tables(live_dsn: str) -> Iterator[None]:
    """Create, then drop, one throwaway real table per statement's FROM clause. See the
    module docstring for why a real table (rather than the statement as-is against a
    genuinely missing relation) is required to make this check discriminating at all.
    """
    import psycopg

    with psycopg.connect(live_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            for name, ddl in _SHIM_DDL.items():
                cur.execute(f"DROP TABLE IF EXISTS {name}")
                cur.execute(ddl)
    try:
        yield
    finally:
        with psycopg.connect(live_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                for name in _SHIM_DDL:
                    cur.execute(f"DROP TABLE IF EXISTS {name}")


@pytest.fixture
def adapter(live_dsn: str, shim_tables: None) -> RedshiftWorkloadAdapter:
    a = RedshiftWorkloadAdapter()
    a.connect(ConnectionParams(engine="redshift", dsn=live_dsn, fields={}, source="--dsn"), 30)
    return a


@pytest.mark.parametrize("capability", sorted(_BINDS))
def test_every_statement_binds_its_parameters(adapter: RedshiftWorkloadAdapter, capability: str):
    """Every one of this adapter's four statements must actually run, with representative
    binds, against a table shaped like the view it targets — proving its parameters bind
    over the wire, which is the driver's job and has nothing to do with which system view
    sits behind the FROM clause.

    Calls `adapter._query` directly, bypassing `_run`'s try/except: this test needs to see
    a real bind failure if one occurs, not have it swallowed into `degraded` the way a
    production run correctly does. `connect(autocommit=True)` (see `session.py`) means a
    failed statement does not abort a shared transaction, so every bind in the list runs
    independently on the same connection.
    """
    for params in _BINDS[capability]:
        adapter._query(RedshiftWorkloadAdapter.SQL[capability], params)
