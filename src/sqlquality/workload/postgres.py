"""Postgres workload adapter: pg_stat_statements + catalog introspection, index rules."""

from __future__ import annotations

from datetime import timedelta

from sqlquality.models import (
    Aggregation,
    ConnectionParams,
    Proposal,
    TableFacts,
    Workload,
    WorkloadFetch,
)
from sqlquality.workload.base import IntrospectionStatement, Querier, WorkloadAdapter

CAP_WORKLOAD = "workload"
CAP_STATS_RESET = "stats_reset"
CAP_SCHEMA = "schema"
CAP_TABLE_FACTS = "table_facts"
CAP_NDV = "ndv"
CAP_INDEXES = "indexes"

#: What to tell the user when a capability's statement is refused. These strings are what
#: someone hands their DBA, so they distinguish the one capability needing a real grant
#: from the four that read views already world-readable in stock Postgres — overstating
#: the ask would make a routine request look alarming.
_HINTS = {
    CAP_WORKLOAD: (
        "requires the pg_stat_statements extension (PostgreSQL 13+) and pg_read_all_stats "
        "or superuser; enable via shared_preload_libraries then CREATE EXTENSION. On "
        "PostgreSQL 12 and older the view lacks total_exec_time and this will fail."
    ),
    CAP_STATS_RESET: "reads pg_stat_database; world-readable unless explicitly revoked",
    CAP_SCHEMA: (
        "reads information_schema.columns; shows only tables the current role can access, "
        "so a partial result means missing table privileges rather than a missing grant"
    ),
    CAP_TABLE_FACTS: "reads pg_class and pg_namespace; world-readable unless revoked",
    CAP_NDV: (
        "reads pg_stats, which exposes only rows for tables the current role owns or can "
        "select from — a role without table access silently sees no statistics"
    ),
    CAP_INDEXES: "reads pg_index and pg_stat_user_indexes; world-readable unless revoked",
}


class PostgresWorkloadAdapter(WorkloadAdapter):
    engine = "postgres"

    SQL: dict[str, str] = {
        # `s.rows` is selected but currently discarded. It is kept deliberately: rows per
        # call is the natural selectivity signal for a future confidence refinement
        # ("returns 3 rows from 8M — an excellent index candidate"), and fetching it costs
        # nothing. Task 8 unpacks it as `_rows`.
        # `total_exec_time` requires PostgreSQL 13+; it was `total_time` on 12 and older,
        # both long past end-of-life. The privilege hint states the floor.
        CAP_WORKLOAD: """
            SELECT s.query, s.calls, s.total_exec_time, s.rows
            FROM pg_stat_statements s
            JOIN pg_database d ON d.oid = s.dbid
            WHERE d.datname = current_database()
            ORDER BY s.total_exec_time DESC
            LIMIT %s
        """,
        CAP_STATS_RESET: """
            SELECT stats_reset
            FROM pg_stat_database
            WHERE datname = current_database()
        """,
        CAP_SCHEMA: """
            SELECT c.table_name, c.column_name, c.data_type
            FROM information_schema.columns c
            WHERE c.table_schema = ANY(%s)
        """,
        CAP_TABLE_FACTS: """
            SELECT c.relname, c.reltuples::bigint, pg_total_relation_size(c.oid)
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE c.relkind = 'r' AND n.nspname = ANY(%s) AND c.relname = ANY(%s)
        """,
        CAP_NDV: """
            SELECT s.tablename, s.attname, s.n_distinct
            FROM pg_stats s
            WHERE s.schemaname = ANY(%s) AND s.tablename = ANY(%s)
        """,
        # Known limitation: the pg_attribute join silently omits expression indexes.
        # `indkey` holds 0 for an expression column, which matches no pg_attribute row, so
        # an index on `lower(status)` is invisible here. Consequence: ADV001 may propose an
        # index whose expression equivalent already exists. Reading pg_get_indexdef() would
        # fix it; deferred rather than silently ignored.
        CAP_INDEXES: """
            SELECT t.relname, i.relname, a.attname, k.ordinality,
                   ix.indisunique, ix.indisprimary,
                   COALESCE(psui.idx_scan, 0), pg_relation_size(i.oid)
            FROM pg_index ix
            JOIN pg_class i ON i.oid = ix.indexrelid
            JOIN pg_class t ON t.oid = ix.indrelid
            JOIN pg_namespace n ON n.oid = t.relnamespace
            JOIN LATERAL unnest(ix.indkey) WITH ORDINALITY AS k(attnum, ordinality) ON TRUE
            JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = k.attnum
            LEFT JOIN pg_stat_user_indexes psui ON psui.indexrelid = i.oid
            WHERE n.nspname = ANY(%s) AND t.relname = ANY(%s)
            ORDER BY t.relname, i.relname, k.ordinality
        """,
    }

    def __init__(self, querier: Querier | None = None) -> None:
        super().__init__()
        self._query = querier

    def introspection_sql(self) -> list[IntrospectionStatement]:
        return [
            IntrospectionStatement(capability=cap, sql=sql.strip(), privilege_hint=_HINTS[cap])
            for cap, sql in self.SQL.items()
        ]

    def connect(self, params: ConnectionParams, timeout_s: int) -> None:
        raise NotImplementedError  # Task 8

    def fetch_workload(self, since: timedelta | None, limit: int) -> WorkloadFetch:
        raise NotImplementedError  # Task 8

    def fetch_schema(self, schemas: tuple[str, ...]) -> dict:
        raise NotImplementedError  # Task 8

    def fetch_table_facts(
        self, schemas: tuple[str, ...], tables: frozenset[str]
    ) -> dict[str, TableFacts]:
        raise NotImplementedError  # Task 8

    def propose(
        self,
        aggregation: Aggregation,
        facts: dict[str, TableFacts],
        workload: Workload,
        *,
        min_cost_share: float,
    ) -> list[Proposal]:
        raise NotImplementedError  # Tasks 9-10

    def render_ddl(self, proposals: list[Proposal]) -> str:
        raise NotImplementedError  # Task 11
