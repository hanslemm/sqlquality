import re

import sqlglot
import pytest

from sqlquality.models import Aggregation, ConnectionParams, Workload
from sqlquality.workload import get_workload_adapter
from sqlquality.workload.redshift import (
    CAP_ADVISOR,
    CAP_SCHEMA,
    CAP_TABLE_FACTS,
    CAP_WORKLOAD,
    RedshiftWorkloadAdapter,
)

EXPECTED_CAPABILITIES = {CAP_WORKLOAD, CAP_SCHEMA, CAP_TABLE_FACTS, CAP_ADVISOR}


def test_registry_returns_the_redshift_adapter():
    adapter = get_workload_adapter("redshift")
    assert adapter.engine == "redshift"


def test_every_capability_has_a_statement_and_a_hint():
    statements = RedshiftWorkloadAdapter().introspection_sql()
    assert {s.capability for s in statements} == EXPECTED_CAPABILITIES
    for statement in statements:
        assert statement.sql.strip()
        assert statement.privilege_hint.strip()


@pytest.mark.parametrize("capability", sorted(EXPECTED_CAPABILITIES))
def test_every_statement_parses_as_redshift_sql(capability):
    """Syntax validation is the one correctness check available without a cluster.

    The catalog SQL in this adapter cannot be executed during development — there is no
    Redshift container and `svv_*`/`sys_*` do not exist in Postgres. Parsing each statement
    with sqlglot's redshift dialect cannot catch a wrong column name, but it catches a
    malformed statement, which would otherwise be invisible until a user ran it.
    """
    sql = RedshiftWorkloadAdapter.SQL[capability]
    # `%s` placeholders are libpq's, not SQL — sqlglot cannot parse them, so they become
    # bind markers for the purposes of this check.
    parsed = sqlglot.parse_one(sql.replace("%s", "?"), dialect="redshift")
    assert parsed is not None


@pytest.mark.parametrize("capability", sorted(EXPECTED_CAPABILITIES))
def test_no_statement_writes(capability):
    """Same guard the Postgres adapter carries, for the same reason."""
    forbidden = (
        "insert",
        "update",
        "delete",
        "create",
        "drop",
        "alter",
        "truncate",
        "grant",
        "revoke",
        "vacuum",
        "analyze",
    )
    lowered = RedshiftWorkloadAdapter.SQL[capability].lower()
    found = {verb for verb in forbidden if re.search(rf"\b{verb}\b", lowered)}
    assert not found, f"{capability} contains write verb(s): {sorted(found)}"


def test_there_is_no_ndv_or_index_capability():
    """Redshift exposes no `pg_stats.n_distinct` equivalent and has no indexes.

    Declaring either capability would invite a rule to assume evidence that cannot exist.
    """
    capabilities = {s.capability for s in RedshiftWorkloadAdapter().introspection_sql()}
    assert not any("ndv" in c or "index" in c for c in capabilities)


#: Every `WorkloadAdapter` method this task deliberately leaves unbuilt, with a call that
#: reaches it. Named individually rather than discovered by reflection: a later task that
#: implements one of these must delete its entry here, which is a visible, reviewable edit —
#: whereas a reflective sweep would silently stop covering whatever got implemented.
UNIMPLEMENTED = {
    "connect": lambda a: a.connect(
        ConnectionParams(engine="redshift", dsn="postgresql://h/d", fields={}, source="test"), 30
    ),
    "fetch_workload": lambda a: a.fetch_workload(None, 10),
    "fetch_schema": lambda a: a.fetch_schema(("public",)),
    "fetch_table_facts": lambda a: a.fetch_table_facts(("public",), frozenset()),
    "propose": lambda a: a.propose(
        Aggregation(usage=(), total_cost_ms=0.0, skipped_unqualifiable=0, tables=frozenset()),
        {},
        Workload(stats=(), window_description="w"),
        min_cost_share=0.01,
    ),
    "render_ddl": lambda a: a.render_ddl([]),
}


@pytest.mark.parametrize("method", sorted(UNIMPLEMENTED))
def test_unimplemented_methods_say_so_rather_than_returning_empty(method):
    """A half-built adapter that returns nothing looks exactly like a healthy cluster with
    no workload, which is the worst possible failure mode for this command.

    Every unbuilt method is covered, not just one. Task 1 originally pinned `fetch_schema`
    alone, which would have let a later task implement `fetch_workload` and silently leave
    `fetch_table_facts` returning `[]` — the run would then report a healthy cluster with no
    catalog facts rather than an unfinished adapter.
    """
    adapter = RedshiftWorkloadAdapter()
    with pytest.raises(NotImplementedError):
        UNIMPLEMENTED[method](adapter)
