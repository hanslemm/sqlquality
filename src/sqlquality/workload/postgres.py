"""Postgres workload adapter: pg_stat_statements + catalog introspection, index rules."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from sqlquality.models import (
    Aggregation,
    ConnectionParams,
    Proposal,
    RawQueryRow,
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

#: dbt profiles.yml field names -> libpq connection keywords.
_PG_FIELD_MAP = {
    "dbname": "dbname",
    "database": "dbname",
    "host": "host",
    "port": "port",
    "user": "user",
    "username": "user",
    "password": "password",
}


def _pg_fields(fields: dict[str, str]) -> dict[str, str]:
    """Translate profiles.yml keys to libpq keywords, dropping anything unrecognized."""
    return {_PG_FIELD_MAP[k]: v for k, v in fields.items() if k in _PG_FIELD_MAP}


@dataclass(frozen=True)
class PgIndex:
    """One existing Postgres index, with its ordered column list and usage counter."""

    name: str
    columns: tuple[str, ...]
    is_unique: bool
    is_primary: bool
    scans: int
    size_bytes: int


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

    def _run(self, capability: str, params: tuple[object, ...]) -> list[tuple[object, ...]]:
        """Run one introspection statement, recording degradation rather than raising.

        A single missing grant must cost only that capability — never the whole run.
        """
        if self._query is None:
            raise RuntimeError("connect() must be called before fetching")
        try:
            return self._query(self.SQL[capability], params)
        except Exception as exc:  # driver-specific; we only need the message
            self.degraded.append((capability, f"{exc} — {_HINTS[capability]}"))
            return []

    def connect(self, params: ConnectionParams, timeout_s: int) -> None:
        try:
            import psycopg
        except ImportError as exc:
            raise ImportError(
                "Postgres support requires psycopg. "
                "Install it with: pip install 'sqlquality[postgres]'"
            ) from exc

        conninfo = params.dsn or psycopg.conninfo.make_conninfo(**_pg_fields(params.fields))
        connection = psycopg.connect(conninfo, autocommit=True)
        with connection.cursor() as cursor:
            # Belt and braces: the session cannot write even if a statement tried to.
            cursor.execute("SET default_transaction_read_only = on")
            cursor.execute(f"SET statement_timeout = '{int(timeout_s)}s'")

        def query(sql: str, bind: tuple[object, ...]) -> list[tuple[object, ...]]:
            with connection.cursor() as cur:
                cur.execute(sql, bind)
                return list(cur.fetchall())

        self._query = query

    def fetch_workload(self, since: timedelta | None, limit: int) -> WorkloadFetch:
        rows = self._run(CAP_WORKLOAD, (limit,))
        reset = self._run(CAP_STATS_RESET, ())
        reset_at = reset[0][0] if reset and reset[0] else "an unknown time"
        # pg_stat_statements is cumulative since reset and carries no per-statement
        # timestamps before PG 17, so --since cannot be honored. Say so rather than
        # implying the requested window was applied.
        window = f"since stats reset at {reset_at}"
        if since is not None:
            window += " (--since is not supported by pg_stat_statements)"
        return WorkloadFetch(
            rows=tuple(
                RawQueryRow(
                    sql=str(sql),
                    calls=int(calls),  # type: ignore[call-overload]
                    total_time_ms=float(total_ms),  # type: ignore[arg-type]
                )
                for sql, calls, total_ms, _rows in rows
            ),
            window_description=window,
        )

    def fetch_schema(self, schemas: tuple[str, ...]) -> dict:
        schema: dict[str, dict[str, str]] = {}
        for table, column, data_type in self._run(CAP_SCHEMA, (list(schemas),)):
            schema.setdefault(str(table), {})[str(column)] = str(data_type)
        return schema

    def fetch_table_facts(
        self, schemas: tuple[str, ...], tables: frozenset[str]
    ) -> dict[str, TableFacts]:
        wanted = sorted(tables)
        sizes = {
            str(name): (
                int(rows),  # type: ignore[call-overload]
                int(size) if size is not None else None,  # type: ignore[call-overload]
            )
            for name, rows, size in self._run(CAP_TABLE_FACTS, (list(schemas), wanted))
        }
        columns: dict[str, list[str]] = {}
        for table, column, _type in self._run(CAP_SCHEMA, (list(schemas),)):
            if str(table) in tables:
                columns.setdefault(str(table), []).append(str(column))

        ndv: dict[str, dict[str, float]] = {}
        for table, column, n_distinct in self._run(CAP_NDV, (list(schemas), wanted)):
            if n_distinct is None:
                continue
            value = float(n_distinct)  # type: ignore[arg-type]
            row_estimate = sizes.get(str(table), (0, None))[0]
            # Postgres encodes "distinct as a fraction of row count" as a negative value.
            resolved = -value * row_estimate if value < 0 else value
            ndv.setdefault(str(table), {})[str(column)] = resolved

        facts: dict[str, TableFacts] = {}
        for table in wanted:
            rows, size = sizes.get(table, (None, None))
            facts[table] = TableFacts(
                name=table,
                row_estimate=rows,
                size_bytes=size,
                columns=tuple(columns.get(table, ())),
                ndv=ndv.get(table, {}),
            )
        return facts

    def fetch_indexes(
        self, schemas: tuple[str, ...], tables: frozenset[str]
    ) -> dict[str, tuple[PgIndex, ...]]:
        """Existing indexes per table, columns in ordinal order."""
        grouped: dict[tuple[str, str], dict[str, object]] = {}
        for row in self._run(CAP_INDEXES, (list(schemas), sorted(tables))):
            table, index, column, _ordinality, unique, primary, scans, size = row
            size_bytes = int(size) if size is not None else 0  # type: ignore[call-overload]
            entry = grouped.setdefault(
                (str(table), str(index)),
                {
                    "columns": [],
                    "is_unique": bool(unique),
                    "is_primary": bool(primary),
                    "scans": int(scans),  # type: ignore[call-overload]
                    "size_bytes": size_bytes,
                },
            )
            columns = entry["columns"]
            assert isinstance(columns, list)
            columns.append(str(column))

        result: dict[str, list[PgIndex]] = {}
        for (table, index), entry in grouped.items():
            result.setdefault(table, []).append(
                PgIndex(
                    name=index,
                    columns=tuple(entry["columns"]),  # type: ignore[arg-type]
                    is_unique=bool(entry["is_unique"]),
                    is_primary=bool(entry["is_primary"]),
                    scans=int(entry["scans"]),  # type: ignore[call-overload]
                    size_bytes=int(entry["size_bytes"]),  # type: ignore[call-overload]
                )
            )
        return {table: tuple(indexes) for table, indexes in result.items()}

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
