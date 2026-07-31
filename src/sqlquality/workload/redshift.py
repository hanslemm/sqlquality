"""Redshift workload adapter: sys_/svv_ system-view introspection, connect() is real.

**Provenance warning.** None of the SQL in this module has been executed against a live
Redshift cluster: there is no Redshift container available for development, and Postgres —
where every other adapter's SQL gets exercised during tests — does not implement `svv_*` or
`sys_*` views at all, so nothing here can be run locally either. Every statement's column
names come from AWS's published Redshift system-view documentation, not from an observed
row. That is exactly why every consumer of these rows (added in later tasks) is required to
unpack them defensively, and why `_run` records a denied or malformed statement as one entry
in `self.degraded` rather than letting the exception propagate: a wrong column name should
cost this run exactly one capability, never the whole run. The one correctness check
available without a cluster is syntax — see `tests/test_workload_redshift.py`, which parses
every statement with sqlglot's `redshift` dialect.

**Deliberately no `CAP_NDV`, no `CAP_INDEXES`.** Redshift exposes no equivalent of
`pg_stats.n_distinct`, and it has no indexes at all — its physical-design levers are
SORTKEY, DISTKEY/DISTSTYLE, and VACUUM/ANALYZE staleness. Declaring either capability here
would invite a later rule to assume evidence that cannot exist on this engine. Those levers
are read through `CAP_ADVISOR` below and turned into proposals (ADV101-ADV105) by a later
task; this module is the skeleton — the capability set, the statements, and registration.

Every method beyond `introspection_sql()` and `connect()` raises `NotImplementedError` here
on purpose. A `fetch_*` method that silently returned empty results would be
indistinguishable from a healthy cluster running no workload at all, which is a worse
failure mode than an explicit "not implemented yet" — see `WorkloadAdapter`'s docstring on
`--dry-run`, the thing Task 1 delivered.

`connect()` is the exception, and deliberately so: Redshift speaks the PostgreSQL wire
protocol through psycopg, so it is the one part of this adapter genuinely exercisable
against the `postgres:16` container the rest of the suite already runs against — see
`tests/integration/test_redshift_connect_live.py`. Its session setup is shared with
`PostgresWorkloadAdapter.connect` via `workload/session.py`, and the one behavioral
difference — Redshift refusing `SET default_transaction_read_only` is a recorded
degradation rather than a hard failure — is documented on `connect()` itself.
"""

from __future__ import annotations

import sys
from datetime import timedelta

from sqlquality.models import (
    Aggregation,
    ConnectionParams,
    Proposal,
    Relation,
    TableFacts,
    Workload,
    WorkloadFetch,
)
from sqlquality.workload.base import (
    MAX_TIMEOUT_S,
    MIN_TIMEOUT_S,
    IntrospectionStatement,
    Querier,
    WorkloadAdapter,
)
from sqlquality.workload.secrets import secrets_for
from sqlquality.workload.session import (
    READ_ONLY_SQL,
    dropped_libpq_fields,
    import_psycopg,
    open_session,
    translate_libpq_fields,
)

CAP_WORKLOAD = "workload"
CAP_SCHEMA = "schema"
CAP_TABLE_FACTS = "table_facts"
CAP_ADVISOR = "advisor"

#: Pseudo-capability name `connect()` uses to record a read-only degradation in
#: `self.degraded` — not one of `introspection_sql()`'s four statements, since arming
#: read-only intent happens once at connect time rather than per fetch. Named distinctly
#: from every real `CAP_*` so a report reader cannot mistake it for a denied SELECT.
DEGRADATION_READ_ONLY = "read_only"

#: dbt profiles.yml field names -> libpq connection keywords, for a Redshift target.
#: Redshift's dbt adapter accepts the same core keywords Postgres does — see
#: `translate_libpq_fields` in `session.py`, which this and `postgres.py`'s own
#: `_PG_FIELD_MAP` both feed. IAM-based fields (`cluster_identifier`, `iam`, `region`)
#: are not psycopg keywords and are deliberately not mapped here; a profile using them
#: falls through to `dropped_libpq_fields` and is named on stderr rather than silently
#: dropped.
_REDSHIFT_FIELD_MAP = {
    "dbname": "dbname",
    "database": "dbname",
    "host": "host",
    "port": "port",
    "user": "user",
    "username": "user",
    "password": "password",
}
#: profiles.yml keys forwarded to libpq unchanged — see `postgres._PG_PASSTHROUGH_FIELDS`
#: for why the TLS group in particular is here rather than silently dropped.
_REDSHIFT_PASSTHROUGH_FIELDS = frozenset(
    {"sslmode", "sslcert", "sslkey", "sslrootcert", "connect_timeout"}
)

#: What to tell the user when a capability's statement is refused. These strings are what
#: someone hands their DBA, so each one names the actual failure mode rather than a generic
#: "requires access" — in particular `CAP_WORKLOAD`'s partial-result trap, which is exactly
#: the kind the Postgres adapter already warns about for `pg_stats`: a role without the
#: right grant does not get denied, it gets a workload that looks thin or empty.
_HINTS = {
    CAP_WORKLOAD: (
        "reads sys_query_history, which without the SYSLOG ACCESS UNRESTRICTED privilege "
        "shows only the connecting user's own queries — a role lacking it sees a workload "
        "that looks merely small, not one that was denied, so the gap has no error to "
        "notice. Grant with ALTER USER <user> SYSLOG ACCESS UNRESTRICTED (superuser-only)."
    ),
    CAP_SCHEMA: (
        "reads svv_columns; like information_schema, it returns only columns of tables the "
        "current user can already see, so a partial result means missing table privileges "
        "rather than a missing grant on this view itself"
    ),
    CAP_TABLE_FACTS: (
        "reads svv_table_info; rows are limited to tables the current user has been granted "
        "access to, so an unexpectedly short result reads as a small schema rather than a "
        "denial — there is no error to distinguish the two"
    ),
    CAP_ADVISOR: (
        "reads svv_alter_table_recommendations, Amazon Redshift Advisor's own SORTKEY/"
        "DISTSTYLE recommendations; visible only for tables the current user can access, "
        "and only after Advisor has run its analysis — a fresh or lightly queried cluster "
        "can return nothing here even with every grant in place"
    ),
}


class RedshiftWorkloadAdapter(WorkloadAdapter):
    engine = "redshift"

    SQL: dict[str, str] = {
        # Column names below come from AWS's Redshift system-view documentation and have
        # NOT been executed against a live cluster (see the module docstring). Every
        # consumer unpacks defensively and a denied or malformed statement is recorded in
        # `degraded` rather than raised, so a wrong name costs one capability instead of
        # the run.
        #
        # `status = 'success'` excludes failed and cancelled statements, which carry no
        # useful cost signal and would otherwise dilute cost_share with executions that
        # never finished. `database_name = current_database()` scopes to the connected
        # database exactly as the Postgres adapter's CAP_WORKLOAD scopes to
        # `current_database()` via `pg_database`.
        CAP_WORKLOAD: """
            SELECT query_text, elapsed_time
            FROM sys_query_history
            WHERE database_name = current_database()
              AND status = 'success'
            ORDER BY elapsed_time DESC
            LIMIT %s
        """,
        # svv_columns is Redshift's own columns view (distinct from information_schema,
        # which Redshift also exposes but which AWS documents less completely for this
        # engine). No reserved words here, unlike CAP_TABLE_FACTS below.
        CAP_SCHEMA: """
            SELECT schema_name, table_name, column_name, data_type
            FROM svv_columns
            WHERE schema_name = ANY(%s)
        """,
        # svv_table_info's own column names are the reserved words "schema" and "table" —
        # both stay double-quoted so the statement parses at all; dropping either quote
        # breaks the statement (verified with sqlglot's redshift dialect — see
        # test_every_statement_parses_as_redshift_sql). `tbl_rows` and `size` are the row
        # estimate and size-in-MB columns per AWS's documentation.
        CAP_TABLE_FACTS: """
            SELECT "schema", "table", tbl_rows, size
            FROM svv_table_info
            WHERE "schema" = ANY(%s) AND "table" = ANY(%s)
        """,
        # svv_alter_table_recommendations is Redshift Advisor's own view of ALTER TABLE
        # ... ALTER DISTSTYLE / ALTER SORTKEY recommendations, which is what "advisor" names
        # here — the source for ADV101-ADV105 in a later task. Column names are AWS's
        # documented ones for this view; unlike svv_table_info none of them are reserved
        # words.
        CAP_ADVISOR: """
            SELECT database_name, schema_name, table_name, type, current_ddl, recommended_ddl
            FROM svv_alter_table_recommendations
            WHERE schema_name = ANY(%s) AND table_name = ANY(%s)
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

        A single missing grant must cost only that capability — never the whole run. See
        `PostgresWorkloadAdapter._run`, which this mirrors exactly.
        """
        if self._query is None:
            raise RuntimeError("connect() must be called before fetching")
        try:
            return self._query(self.SQL[capability], params)
        except Exception as exc:  # driver-specific; we only need the message
            self.degraded.append((capability, f"{exc} — {_HINTS[capability]}"))
            return []

    def connect(self, params: ConnectionParams, timeout_s: int) -> None:
        """Open a read-only session over the PostgreSQL wire protocol.

        Redshift speaks libpq through psycopg exactly as Postgres does, so the whole
        session-setup mechanism — driver import, conninfo construction inside the
        scrubbing envelope, the statement timeout, secret scrubbing on a driver failure —
        is shared with `PostgresWorkloadAdapter.connect` via `open_session`; see
        `session.py`'s module docstring for why it lives there rather than being copied.

        The one thing this adapter does differently is what a refused
        `SET default_transaction_read_only = on` means: Redshift does not accept that
        statement in every configuration, and unlike Postgres its refusal here does not
        abort the connection. It is recorded in `self.degraded` instead, so the report
        says plainly that the session could not be proven read-only — continuing
        silently as though the statement had succeeded would misstate the one guarantee
        this tool exists to keep. The connection is still safe to use regardless: this
        adapter only ever issues the four `SELECT` statements in `SQL` above, pinned by
        `test_no_statement_writes`.
        """
        psycopg = import_psycopg("Redshift", "warehouse")

        # Silence is the failure mode being fixed here: a dropped `sslmode` downgrades
        # the connection with no signal at all. Key names only — see
        # `dropped_libpq_fields`.
        dropped = dropped_libpq_fields(
            params.fields, _REDSHIFT_FIELD_MAP, _REDSHIFT_PASSTHROUGH_FIELDS
        )
        if dropped:
            print(
                f"warning: ignoring connection setting(s) not supported by the Redshift "
                f"adapter: {', '.join(dropped)}. Pass --dsn if you need them.",
                file=sys.stderr,
            )

        # Everything we know to be secret, so a driver exception can be proven clean
        # rather than trusted.
        secrets = secrets_for(params)

        # `read_only_required=False`: a refusal here is recorded as a degradation, not
        # raised — see this method's own docstring and `open_session`'s parameter of the
        # same name.
        query, degradation = open_session(
            psycopg=psycopg,
            # Inside the scrubbing envelope: psycopg raises from make_conninfo on an
            # unusable keyword, and that message can quote the offending value — which
            # for the `password` keyword is the password.
            conninfo_factory=lambda: (
                params.dsn
                or psycopg.conninfo.make_conninfo(
                    **translate_libpq_fields(
                        params.fields, _REDSHIFT_FIELD_MAP, _REDSHIFT_PASSTHROUGH_FIELDS
                    )
                )
            ),
            secrets=secrets,
            timeout_s=timeout_s,
            min_timeout_s=MIN_TIMEOUT_S,
            max_timeout_s=MAX_TIMEOUT_S,
            read_only_sql=READ_ONLY_SQL,
            read_only_required=False,
        )
        self._query = query
        if degradation is not None:
            self.degraded.append((DEGRADATION_READ_ONLY, degradation))

    def fetch_workload(self, since: timedelta | None, limit: int) -> WorkloadFetch:
        raise NotImplementedError("Redshift fetch_workload() is not implemented yet.")

    def fetch_schema(self, schemas: tuple[str, ...]) -> dict:
        raise NotImplementedError("Redshift fetch_schema() is not implemented yet.")

    def fetch_table_facts(
        self, schemas: tuple[str, ...], relations: frozenset[Relation]
    ) -> dict[Relation, TableFacts]:
        raise NotImplementedError("Redshift fetch_table_facts() is not implemented yet.")

    def propose(
        self,
        aggregation: Aggregation,
        facts: dict[Relation, TableFacts],
        workload: Workload,
        *,
        min_cost_share: float,
    ) -> list[Proposal]:
        raise NotImplementedError("Redshift propose() is not implemented yet.")

    def render_ddl(self, proposals: list[Proposal]) -> str:
        raise NotImplementedError("Redshift render_ddl() is not implemented yet.")
