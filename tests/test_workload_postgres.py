import pytest

from sqlquality.workload import get_workload_adapter
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

    Whitespace is collapsed before matching, and the result padded, so a verb preceded by a
    newline or sitting at the very start of the statement is still caught. A naive
    ``f" {verb} " in f" {sql.lower()} "`` check misses `"... LIMIT %s;\\ndrop table foo"`
    entirely — which is precisely the stacked-statement mistake this guard exists to catch.
    """
    normalized = f" {' '.join(sql.lower().split())} "
    return {verb for verb in FORBIDDEN_VERBS if f" {verb} " in normalized}


def test_the_write_verb_detector_catches_newline_and_leading_positions():
    """Guard the guard: this test is the reason the detector normalizes whitespace."""
    assert _write_verbs_in("select 1 from t limit 1;\ndrop table foo") == {"drop"}
    assert _write_verbs_in("delete from t") == {"delete"}
    assert _write_verbs_in("select 1\n\tgrant select on x to y") == {"grant"}
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
