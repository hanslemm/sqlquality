"""Postgres workload adapter: pg_stat_statements + catalog introspection, index rules."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import timedelta
from urllib.parse import urlparse

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
#: profiles.yml keys whose values must never appear in any message we emit.
_SECRET_FIELDS = frozenset({"password", "pass"})
#: Bounds for --timeout, in seconds. A statement timeout of 0 means "no limit" in Postgres,
#: which would defeat the point, and an absurd value is more likely a typo than an intent.
_MIN_TIMEOUT_S = 1
_MAX_TIMEOUT_S = 3600


def _pg_fields(fields: dict[str, str]) -> dict[str, str]:
    """Translate profiles.yml keys to libpq keywords, dropping anything unrecognized."""
    return {_PG_FIELD_MAP[k]: v for k, v in fields.items() if k in _PG_FIELD_MAP}


def _clamp_timeout_ms(timeout_s: int) -> int:
    """Statement timeout in milliseconds, clamped into a sane range."""
    return max(_MIN_TIMEOUT_S, min(int(timeout_s), _MAX_TIMEOUT_S)) * 1000


#: A secret shorter than this cannot be redacted by substring replacement without
#: destroying the message — a one-character password would blank every occurrence of that
#: letter. When one actually appears, the driver's text is withheld rather than mangled.
_MIN_SCRUBBABLE_SECRET = 4
_WITHHELD = "(driver message withheld: it contained a value too short to redact safely)"


def _secrets_for(params: ConnectionParams) -> tuple[str, ...]:
    """Every value we know to be secret for this connection.

    A DSN is added *and* its password extracted separately. The whole-DSN token only helps
    if the driver echoes the connection string back verbatim, which real libpq errors do
    not do — they report the offending value on its own. Without the extracted password,
    DSN-based connections would have no effective protection at all.
    """
    secrets = tuple(
        value for key, value in params.fields.items() if key in _SECRET_FIELDS and value
    )
    if params.dsn:
        secrets += (params.dsn,)
        dsn_password = urlparse(params.dsn).password
        if dsn_password:
            secrets += (dsn_password,)
    return secrets


def _scrub(text: str, secrets: Iterable[str]) -> str:
    """Replace any known secret occurring in ``text`` with a redaction marker.

    Defence in depth for driver exceptions. libpq is not believed to echo a password, but
    the auth-failure path — the most common real connect failure — cannot be exercised
    without a live server, and we hold the secret anyway, so its absence can be guaranteed
    instead of trusted.
    """
    present = [secret for secret in secrets if secret and secret in text]
    if any(len(secret) < _MIN_SCRUBBABLE_SECRET for secret in present):
        return _WITHHELD
    scrubbed = text
    for secret in present:
        scrubbed = scrubbed.replace(secret, "***")
    return scrubbed


def _as_int(value: object) -> int:
    """Coerce a driver row value to int.

    Querier rows are `tuple[object, ...]`, so this coercion is unavoidably unchecked. It
    lives in one auditable helper rather than at a dozen call sites.
    """
    return int(value)  # type: ignore[call-overload]


def _as_float(value: object) -> float:
    """Coerce a driver row value to float. See _as_int."""
    return float(value)  # type: ignore[arg-type]


@dataclass
class _IndexRows:
    """Mutable per-index collector while unnested index rows are grouped.

    A typed accumulator rather than a ``dict[str, object]``: the dict re-boxed already
    correctly-typed ints and bools as ``object``, forcing them to be re-coerced (and
    type-ignored) a few lines later for no gain.
    """

    is_unique: bool
    is_primary: bool
    scans: int
    size_bytes: int
    #: (ordinality, column) so the column order can be restored by sorting.
    columns: list[tuple[int, str]] = field(default_factory=list)


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
        # Everything we know to be secret, so a driver exception can be proven clean rather
        # than trusted.
        secrets = _secrets_for(params)

        failure: str | None = None
        try:
            connection = psycopg.connect(conninfo, autocommit=True)
            with connection.cursor() as cursor:
                # Belt and braces: the session cannot write even if a statement tried to.
                cursor.execute("SET default_transaction_read_only = on")
                # set_config() rather than `SET`, because Postgres does not accept bind
                # parameters in a SET statement and string-building one with a caller value
                # is the wrong habit to establish in the one place we talk to a database.
                cursor.execute(
                    "SELECT set_config('statement_timeout', %s, false)",
                    (f"{_clamp_timeout_ms(timeout_s)}ms",),
                )
        except Exception as exc:
            failure = _scrub(str(exc), secrets)
        if failure is not None:
            # Raised after the handler, and scrubbed: Task 6 established that a dependency's
            # exception text is exactly where this class of leak hides, and that leaving the
            # handler is the only way to keep the original out of __context__.
            raise ConnectionError(f"Could not connect: {failure}")

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
                RawQueryRow(sql=str(sql), calls=_as_int(calls), total_time_ms=_as_float(total_ms))
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
                _as_int(rows),
                _as_int(size) if size is not None else None,
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
            value = _as_float(n_distinct)
            if value < 0:
                # Postgres encodes "distinct as a fraction of row count" as a negative
                # value, which is meaningless without the row count. If the row-count query
                # returned nothing for this table — different statement, so different
                # privileges can hide it — omit the column so it reads as *unknown*.
                # Defaulting the row estimate to 0 would fabricate "zero distinct values"
                # and hand every proposal on this table a false LOW-confidence rating.
                row_estimate = sizes.get(str(table), (None, None))[0]
                if row_estimate is None:
                    continue
                resolved = -value * row_estimate
            else:
                resolved = value
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
        grouped: dict[tuple[str, str], _IndexRows] = {}
        for row in self._run(CAP_INDEXES, (list(schemas), sorted(tables))):
            table, index, column, ordinality, unique, primary, scans, size = row
            entry = grouped.setdefault(
                (str(table), str(index)),
                _IndexRows(
                    is_unique=bool(unique),
                    is_primary=bool(primary),
                    scans=_as_int(scans),
                    size_bytes=_as_int(size) if size is not None else 0,
                ),
            )
            # Keyed by ordinality and sorted below rather than trusting arrival order. The
            # statement does ORDER BY k.ordinality, but composite-index column order decides
            # whether a proposal is right, and a fixture test that pre-sorts its canned rows
            # cannot notice the difference. Cheap defence in depth.
            entry.columns.append((_as_int(ordinality), str(column)))

        result: dict[str, list[PgIndex]] = {}
        for (table, index), entry in grouped.items():
            result.setdefault(table, []).append(
                PgIndex(
                    name=index,
                    columns=tuple(column for _ordinality, column in sorted(entry.columns)),
                    is_unique=entry.is_unique,
                    is_primary=entry.is_primary,
                    scans=entry.scans,
                    size_bytes=entry.size_bytes,
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
