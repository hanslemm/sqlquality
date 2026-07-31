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
available without a cluster for *column names* is syntax — see
`tests/test_workload_redshift.py`, which parses every statement with sqlglot's `redshift`
dialect.

That is a narrower boundary than it first looks, and framing it any wider cost a day: a
statement's *parameters* are bound by the driver over the wire, which has nothing to do
with whether the table behind `FROM` is Redshift-only — a parameter psycopg cannot type
fails identically whether it is headed at `svv_table_info` or `pg_class`. So every one of
this module's four statements is also executed, with representative binds, against a
same-named, same-shaped throwaway table created in the same `postgres:16` the rest of the
suite runs against — see `tests/integration/test_redshift_introspection_bindable_live.py`,
and its docstring for why a *real* stand-in table is required: Postgres's analyzer
resolves table references before parameter types, so running a statement against the
genuinely missing view always fails with `UndefinedTable` regardless of whether its
parameters would otherwise bind, which cannot discriminate anything. A parameter-binding
failure (`IndeterminateDatatype`, previously reproducible on every default `advise` run —
see `CAP_WORKLOAD`'s comment) is exactly the class of bug this closes locally.

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

import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlquality.models import (
    Aggregation,
    ColumnRole,
    ColumnUsage,
    ConnectionParams,
    Confidence,
    Proposal,
    RawQueryRow,
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
from sqlquality.workload.postgres import _by_relation
from sqlquality.workload.secrets import secrets_for
from sqlquality.workload.session import (
    LIBPQ_FIELD_MAP,
    LIBPQ_PASSTHROUGH_FIELDS,
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

#: Redshift's dbt adapter accepts the same core libpq keywords Postgres does, so field
#: translation uses the one shared table in `session.py` (`LIBPQ_FIELD_MAP` /
#: `LIBPQ_PASSTHROUGH_FIELDS`) rather than a second, Redshift-named copy of the same
#: data — exactly the drift the brief for this adapter warned against. IAM-based fields
#: (`cluster_identifier`, `iam`, `region`) are not psycopg keywords and are deliberately
#: not in that table; a profile using them falls through to `dropped_libpq_fields` and is
#: named on stderr rather than silently dropped.

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
        "reads svv_table_info, which is superuser-only unless the connecting role has an "
        "explicit SELECT grant on it; rows are also limited to tables the current user has "
        "been granted access to, so an unexpectedly short (or empty) result reads as a "
        "small schema rather than a denial — there is no error to distinguish the two"
    ),
    CAP_ADVISOR: (
        "reads svv_alter_table_recommendations, Amazon Redshift Advisor's own SORTKEY/"
        "DISTSTYLE recommendations; visible only for tables the current user can access, "
        "and only after Advisor has run its analysis — a fresh or lightly queried cluster "
        "can return nothing here even with every grant in place"
    ),
}


def _as_int(value: object) -> int:
    """Coerce a driver row value to int. See `postgres.py`'s identical helper — Querier
    rows are `tuple[object, ...]`, so this coercion is unavoidably unchecked and lives in
    one auditable place rather than at a dozen call sites."""
    return int(value)  # type: ignore[call-overload]


def _as_float(value: object) -> float:
    """Coerce a driver row value to float. See `_as_int`."""
    return float(value)  # type: ignore[arg-type]


#: `svv_table_info.size` is documented in 1 MB blocks; `TableFacts.size_bytes` is bytes.
_MB_BYTES = 1024 * 1024


def _row_estimate(tbl_rows: object) -> int | None:
    """`svv_table_info.tbl_rows`, with a NULL row translated to unknown.

    A review of this module's first version gated this on `stats_off` too — reading
    `stats_off = 100` as Redshift's equivalent of `pg_class.reltuples = -1`, Postgres's
    "never analyzed" sentinel that silently suppressed every proposal for a table. That
    premise was inverted and is corrected here: AWS documents `tbl_rows` as the table's
    actual row count and `stats_off` as a 0-100 *staleness percentage* for the planner
    statistics, not an "ever analyzed" flag — neither `tbl_rows` nor `size` is itself
    derived from ANALYZE, so nulling them out on a high `stats_off` discarded accurate
    facts about a merely-stale table and then claimed the row count "could not be
    checked" when it plainly could. `stats_off` is still real evidence — see
    `RedshiftTableFacts.stats_off` — it is disclosed as a staleness caveat by a later
    task's rules, not used here to erase a fact this column was never responsible for.
    A NULL `tbl_rows` (the row was never populated at all) is the one genuine unknown.
    """
    return None if tbl_rows is None else _as_int(tbl_rows)


def _size_bytes(size: object) -> int | None:
    """`svv_table_info.size` (1 MB blocks) converted to bytes, or unknown if NULL.

    See `_row_estimate` for why this is no longer gated on `stats_off`.
    """
    return None if size is None else _as_int(size) * _MB_BYTES


@dataclass(frozen=True)
class RedshiftTableFacts:
    """Redshift's own physical-design facts, which `TableFacts` deliberately does not
    model — see that dataclass's docstring: it is engine-neutral, and SORTKEY/DISTKEY/
    staleness are Redshift-specific levers. Held in the adapter, keyed by `Relation`, the
    way `postgres.py` holds `PgIndex`.

    `stats_off` is a 0-100 staleness *percentage* for this table's planner statistics — 0
    is current, 100 is maximally stale — **not** a flag for "never analyzed" and not a
    reason to distrust `tbl_rows`/`size` on the engine-neutral `TableFacts` this adapter
    also builds: AWS documents both of those as physical facts about the table itself,
    not values ANALYZE produces. A later task's rules should disclose `stats_off` as a
    caveat ("statistics are N% stale") rather than treat it as a reason to null out a row
    estimate or size that was never derived from statistics in the first place.

    Every field is `None` only when its own source value was SQL NULL.

    A relation absent entirely from the dict this is stored in (`RedshiftWorkloadAdapter
    .physical_facts`) is a *distinct* condition from every field here being `None`: absence
    means the relation never appeared in `svv_table_info` at all. **This is not, by
    itself, proof of a Spectrum (external) table** — AWS also omits genuinely *empty*
    tables from `svv_table_info`, so a later task proposing SORTKEY/DISTKEY from this
    absence needs an additional signal (e.g. cross-referencing `svv_external_tables`) to
    tell the two cases apart; recorded here as a carry-forward, not solved by this task.
    """

    unsorted: float | None
    stats_off: float | None
    diststyle: str | None
    sortkey1: str | None
    skew_rows: float | None


#: Matches `KEY(column)`, whether bare or nested inside `AUTO(...)` — the shapes
#: `svv_table_info.diststyle` takes when the table has an explicit distribution key. Not
#: verified against a live cluster (see the module docstring); parsing is defensive and
#: case-insensitive, matching this whole module's discipline for column *values* it cannot
#: exercise locally.
_DISTSTYLE_KEY_RE = re.compile(r"KEY\(\s*([^)]+?)\s*\)", re.IGNORECASE)


def _diststyle_key_column(diststyle: str) -> str | None:
    """The column name inside `KEY(column)` (or `AUTO(KEY(column))`), or `None`.

    `svv_table_info` carries no separate "DISTKEY column" field — the column name is
    embedded in `diststyle`'s text, in one of AWS's documented shapes ('KEY(col)', 'EVEN',
    'ALL', or any of those wrapped in `AUTO(...)`). Parsed rather than string-matched
    verbatim so `propose_distkey` can tell "already distributed on this column" apart from
    "distributed on a different one" without a second introspection round trip.
    """
    match = _DISTSTYLE_KEY_RE.search(diststyle)
    return match.group(1).strip() if match else None


def _diststyle_is_all(diststyle: str) -> bool:
    """True for Redshift's own `ALL` or `AUTO(ALL)` diststyle text.

    Checked as "names ALL and no KEY column" rather than an exact match against the finite
    AWS-documented value set, because `AUTO(...)` wraps any of the other shapes too and
    telling those apart needs `_diststyle_key_column`'s parsing either way.
    """
    return "ALL" in diststyle.upper() and _diststyle_key_column(diststyle) is None


def _quote_ident(name: str) -> str:
    """Quote an identifier, doubling any embedded double quote.

    See `postgres.py`'s identical helper. Duplicated rather than imported: DDL quoting is
    each adapter's own small, self-contained concern, and it is not on the reuse list this
    module was given (`_by_relation`, `_is_prefix`, `cost_share_of`, `ranking_key`, the
    proposal-collapse machinery) — those are the pieces of shared *logic*, not this
    engine-agnostic one-liner.
    """
    return '"' + name.replace('"', '""') + '"'


def _qualified(schema: str, name: str) -> str:
    """`"schema"."name"`, both parts quoted. See `postgres.py`'s identical helper."""
    return f"{_quote_ident(schema)}.{_quote_ident(name)}"


def propose_sortkey(
    usage: Sequence[ColumnUsage],
    facts: Mapping[Relation, TableFacts],
    physical: Mapping[Relation, RedshiftTableFacts],
    *,
    min_cost_share: float,
) -> list[Proposal]:
    """ADV101 — a SORTKEY candidate from the table's hottest RANGE/EQUALITY predicate.

    Redshift's zone maps store a min/max per 1MB block for the sort key column, so a scan
    can skip whole blocks when the predicate is on that column. A time-series column under
    a range predicate is the canonical win, but a hot equality predicate benefits the same
    way, so both roles are pooled into one candidate list.

    **Confidence is capped at MEDIUM and there is deliberately no HIGH branch — do not add
    one for symmetry with a Postgres index rule.** A SORTKEY change is only worth its
    rewrite if the predicate is *selective*, and selectivity is exactly what cannot be
    measured without per-column NDV, which Redshift does not expose at all (see the module
    docstring's `CAP_NDV` note). Claiming HIGH here would assert something about data
    distribution this tool cannot see, while recommending `ALTER TABLE ... ALTER SORTKEY`,
    which rewrites the entire table. See `propose_distkey` for the DISTKEY-specific version
    of the same argument, and ADV008 in `postgres.py` for the precedent this follows.

    Suppressed when the table's existing `sortkey1` already *is* the candidate column — the
    SORTKEY equivalent of `postgres.py`'s `_covered`. If `sortkey1` itself could not be
    read (its source value was SQL NULL), the claim "the table is not already sorted on
    this column" is unknowable, so confidence drops to LOW and the rationale names the gap
    — the same trap `_covered`'s docstring describes for an unreadable index catalog.

    A relation entirely absent from `physical` — as opposed to present with `sortkey1 is
    None` — is a different, and materially worse, gap: `svv_table_info` omits both external
    (Spectrum) tables, which cannot carry a SORTKEY at all, and genuinely empty local
    tables, and nothing available anywhere in this adapter distinguishes the two (see
    `RedshiftTableFacts`'s docstring). Proposing a table rewrite for something that might
    not even support one is worse than proposing nothing, so this rule does not propose for
    it at all — a documented gap, not a silent one, and not a guess either way.

    `facts` is accepted but not read: this rule's absence-of-evidence gate is entirely
    `physical`'s (`RedshiftTableFacts` carries `sortkey1`; the engine-neutral `TableFacts`
    row estimate has nothing this rule needs), kept in the signature so every ADV10x
    proposal function takes the same four-argument shape from `propose()`'s call sites.
    """
    proposals: list[Proposal] = []
    for relation, items in sorted(_by_relation(usage).items()):
        candidates = sorted(
            (i for i in items if i.role in (ColumnRole.RANGE, ColumnRole.EQUALITY)),
            key=lambda i: (-i.cost_ms, i.column),
        )
        if not candidates:
            continue
        best = candidates[0]
        if best.cost_share < min_cost_share:
            continue

        phys = physical.get(relation)
        if phys is None:
            # Cannot tell a Spectrum table from a genuinely empty one — see this
            # function's own docstring. Not proposed, and not a silent skip: the reasoning
            # lives above rather than in a per-run message, the same discipline this
            # module already uses for every other documented gap.
            continue

        if phys.sortkey1 is not None and phys.sortkey1 == best.column:
            continue

        if phys.sortkey1 is None:
            confidence = Confidence.LOW
            rationale = (
                f"{best.column} carries the table's hottest range/equality predicate. "
                "The table's existing sort key could not be read, so whether it is "
                "already sorted on this column is unknown — confirm before applying."
            )
        else:
            confidence = Confidence.MEDIUM
            rationale = (
                f"{best.column} carries the table's hottest range/equality predicate, and "
                f"the table is currently sorted on {phys.sortkey1!r}, not this column. "
                "Zone maps let a scan skip whole 1MB blocks when the predicate matches the "
                "sort key, which the current sort key cannot provide for this predicate."
            )
        rationale += (
            " Confidence is capped at MEDIUM: a SORTKEY change only repays the rewrite if "
            "this predicate is selective, and Redshift exposes no per-column "
            "distinct-value statistics to check that."
        )
        if phys.stats_off is not None and phys.stats_off > 0:
            rationale += (
                f" This table's planner statistics are {phys.stats_off:.0f}% stale "
                "(stats_off) — treat any row-count-based reasoning elsewhere in this "
                "report with that in mind."
            )

        proposals.append(
            Proposal(
                code="ADV101",
                title=f"Consider SORTKEY on {relation}({best.column})",
                rationale=rationale,
                evidence={
                    "schema": relation.schema,
                    "table": relation.table,
                    "column": best.column,
                    "role": best.role.value,
                    "cost_share": best.cost_share,
                    "calls": best.calls,
                    "current_sortkey1": phys.sortkey1,
                    "stats_off": phys.stats_off,
                },
                confidence=confidence,
                ddl=(
                    f"ALTER TABLE {_qualified(relation.schema, relation.table)} "
                    f"ALTER SORTKEY ({_quote_ident(best.column)});"
                ),
                note=(
                    "ALTER SORTKEY rewrites the entire table: Redshift copies every row, "
                    "holding a lock for the duration, and needs roughly the table's own "
                    "size again in free disk space while the rewrite runs. There is no "
                    "CONCURRENTLY equivalent — unlike a Postgres index, this cannot be "
                    "built alongside normal traffic. Run it in a maintenance window and "
                    "confirm free disk space first."
                ),
            )
        )
    return proposals


def propose_distkey(
    usage: Sequence[ColumnUsage],
    facts: Mapping[Relation, TableFacts],
    physical: Mapping[Relation, RedshiftTableFacts],
    *,
    min_cost_share: float,
) -> list[Proposal]:
    """ADV102 — a DISTKEY candidate from the table's hottest JOIN predicate.

    A join whose two sides are not distributed on the join key forces Redshift to
    redistribute rows across the cluster before the join can run — `DS_BCAST_INNER` or the
    heavier `DS_DIST_BOTH`, the same redistribution markers the offline `RedshiftAdapter`
    names from an EXPLAIN plan (`sqlquality/adapters/redshift.py`). Distributing both sides
    on the join key removes that step entirely.

    **Confidence is capped at MEDIUM, deliberately, with no HIGH branch** — see
    `propose_sortkey`'s docstring for the shared reasoning, and do not add a HIGH branch
    here either. The DISTKEY-specific version of it: `svv_table_info.skew_rows` describes
    the table's *current* distribution, not the skew the proposed key would produce, and
    Redshift exposes no per-column NDV to predict it — a bad DISTKEY choice does not merely
    cost a slower scan, it can concentrate the whole table onto one node, which is exactly
    the failure mode this rule cannot see coming.

    Suppressed when the table is already distributed on the candidate column — parsed out
    of `svv_table_info.diststyle`'s `KEY(column)` (or `AUTO(KEY(column))`) text, since that
    view carries no separate DISTKEY column (see `_diststyle_key_column`) — or when it is
    already `DISTSTYLE ALL`, which already avoids redistribution entirely and is a strictly
    better outcome than any single-column DISTKEY could offer (see `propose_diststyle_all`,
    which proposes moving *to* ALL under its own, narrower gate).

    See `propose_sortkey` for the absence-from-`physical` handling: identical reasoning,
    identical outcome — no proposal, not a guess. `facts` is accepted but not read, for the
    same interface-symmetry reason `propose_sortkey` gives.
    """
    proposals: list[Proposal] = []
    for relation, items in sorted(_by_relation(usage).items()):
        candidates = sorted(
            (i for i in items if i.role is ColumnRole.JOIN),
            key=lambda i: (-i.cost_ms, i.column),
        )
        if not candidates:
            continue
        best = candidates[0]
        if best.cost_share < min_cost_share:
            continue

        phys = physical.get(relation)
        if phys is None:
            continue

        diststyle = phys.diststyle
        if diststyle is not None:
            if _diststyle_is_all(diststyle):
                continue
            existing_key = _diststyle_key_column(diststyle)
            if existing_key is not None and existing_key == best.column:
                continue

        if diststyle is None:
            confidence = Confidence.LOW
            rationale = (
                f"{best.column} carries the table's hottest join predicate. The table's "
                "current distribution style could not be read, so whether it is already "
                "distributed on this column is unknown — confirm before applying."
            )
        else:
            confidence = Confidence.MEDIUM
            rationale = (
                f"{best.column} carries the table's hottest join predicate, and the "
                f"table's current distribution style is {diststyle!r}, not keyed on this "
                "column. A join whose sides are not co-located on the join key forces "
                "Redshift to redistribute rows across the cluster before it can complete "
                "the join."
            )
        rationale += (
            " Confidence is capped at MEDIUM: distribution skew is what makes a DISTKEY "
            "choice good or catastrophic, and Redshift exposes no per-column "
            "distinct-value statistics to predict it."
        )
        if phys.skew_rows is not None:
            rationale += (
                f" This table's current skew_rows is {phys.skew_rows:.2f}, but that "
                "describes its *existing* distribution, not the skew this DISTKEY would "
                "produce, which cannot be predicted from it."
            )
        if phys.stats_off is not None and phys.stats_off > 0:
            rationale += (
                f" This table's planner statistics are {phys.stats_off:.0f}% stale "
                "(stats_off) — treat any row-count-based reasoning elsewhere in this "
                "report with that in mind."
            )

        proposals.append(
            Proposal(
                code="ADV102",
                title=f"Consider DISTKEY on {relation}({best.column})",
                rationale=rationale,
                evidence={
                    "schema": relation.schema,
                    "table": relation.table,
                    "column": best.column,
                    "role": best.role.value,
                    "cost_share": best.cost_share,
                    "calls": best.calls,
                    "current_diststyle": diststyle,
                    "skew_rows": phys.skew_rows,
                    "stats_off": phys.stats_off,
                },
                confidence=confidence,
                ddl=(
                    f"ALTER TABLE {_qualified(relation.schema, relation.table)} "
                    f"ALTER DISTKEY {_quote_ident(best.column)};"
                ),
                note=(
                    "ALTER DISTKEY rewrites the entire table: Redshift redistributes and "
                    "copies every row across every node, holding a lock for the duration, "
                    "and needs roughly the table's own size again in free disk space while "
                    "the rewrite runs. There is no CONCURRENTLY equivalent. Run it in a "
                    "maintenance window and confirm free disk space first."
                ),
            )
        )
    return proposals


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
        #
        # `elapsed_time` is documented as microseconds, unlike `pg_stat_statements
        # .total_exec_time`'s milliseconds — fetch_workload() divides by 1000.
        #
        # `(CAST(%s AS timestamptz) IS NULL OR start_time >= %s)`, not a bare
        # `start_time >= %s`: unlike `pg_stat_statements`, which carries no per-statement
        # timestamp at all, `sys_query_history.start_time` genuinely lets `--since` be
        # honoured here — see fetch_workload()'s docstring and its honest
        # `window_description` either way. The same bind value is passed twice (`None`
        # when `--since` was not given) so one static, syntax-checkable statement serves
        # both cases rather than two near-duplicate strings that could drift apart.
        #
        # The explicit `CAST(... AS timestamptz)` is load-bearing, not decoration. A bare
        # `%s IS NULL` gives the driver no other typed operand in that branch of the OR to
        # infer a type from, and every run *without* `--since` binds `None` there —
        # reproduced live against `postgres:16` through the identical psycopg wire path:
        # `IndeterminateDatatype: could not determine data type of parameter $1`. `_run`
        # then swallows it into `degraded`, so the run reported a zero-query workload —
        # exactly the "healthy cluster, no traffic" failure mode this module's own
        # docstring says it exists to prevent, for the *default* invocation with no
        # `--since` at all. See `tests/integration/test_redshift_introspection_bindable_live
        # .py`, which executes every one of this adapter's four statements against
        # `postgres:16` with representative binds specifically to catch this class of bug
        # — a statement that cannot even be prepared — locally, rather than assuming
        # bindability is untestable just because the tables underneath are Redshift-only.
        CAP_WORKLOAD: """
            SELECT query_text, elapsed_time
            FROM sys_query_history
            WHERE database_name = current_database()
              AND status = 'success'
              AND (CAST(%s AS timestamptz) IS NULL OR start_time >= %s)
            ORDER BY elapsed_time DESC
            LIMIT %s
        """,
        # svv_columns is Redshift's own columns view (distinct from information_schema,
        # which Redshift also exposes but which AWS documents less completely for this
        # engine). No reserved words here, unlike CAP_TABLE_FACTS below. It also includes
        # external (Spectrum) tables, unlike CAP_TABLE_FACTS's svv_table_info — see
        # fetch_schema()'s docstring.
        CAP_SCHEMA: """
            SELECT schema_name, table_name, column_name, data_type
            FROM svv_columns
            WHERE schema_name = ANY(%s)
        """,
        # svv_table_info's own column names are the reserved words "schema" and "table" —
        # both stay double-quoted so the statement parses at all; dropping either quote
        # breaks the statement (verified with sqlglot's redshift dialect — see
        # test_every_statement_parses_as_redshift_sql). `tbl_rows` and `size` are the row
        # estimate and size-in-MB columns per AWS's documentation; `unsorted`, `stats_off`,
        # `diststyle`, `sortkey1` and `skew_rows` are the physical-design evidence ADV103
        # (DISTSTYLE ALL) and ADV104 (VACUUM/ANALYZE) need — see `RedshiftTableFacts`.
        # `stats_off` is a staleness *percentage*, not a never-analyzed flag — see
        # `RedshiftTableFacts`'s docstring — and does not gate `tbl_rows`/`size`.
        CAP_TABLE_FACTS: """
            SELECT "schema", "table", tbl_rows, size, unsorted, stats_off, diststyle,
                   sortkey1, skew_rows
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
        #: CAP_SCHEMA rows per schema tuple. Both fetch_schema and fetch_table_facts need
        #: them, and running the statement twice did twice the catalog work and — worse —
        #: would append two identical entries to `degraded` when it was denied. Mirrors
        #: `PostgresWorkloadAdapter`'s identical cache.
        self._schema_cache: dict[tuple[str, ...], list[tuple[object, ...]]] = {}
        #: Redshift-specific physical facts from the most recent `fetch_table_facts` call,
        #: keyed the same way as its `TableFacts` return value — see `RedshiftTableFacts`.
        #: A later task's SORTKEY/DISTKEY rules read this directly rather than through a
        #: second introspection round trip, since one CAP_TABLE_FACTS query already carries
        #: both the engine-neutral and the Redshift-specific columns.
        self.physical_facts: dict[Relation, RedshiftTableFacts] = {}

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
        dropped = dropped_libpq_fields(params.fields, LIBPQ_FIELD_MAP, LIBPQ_PASSTHROUGH_FIELDS)
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
                        params.fields, LIBPQ_FIELD_MAP, LIBPQ_PASSTHROUGH_FIELDS
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
        """Raw query-history rows plus an honest description of the window they cover.

        Unlike `pg_stat_statements`, `sys_query_history` carries a `start_time` per
        execution — so unlike `PostgresWorkloadAdapter.fetch_workload`, `--since` genuinely
        can be honoured here, and `window_description` says so plainly either way, the same
        discipline the Postgres adapter uses to say the opposite. The statement is also
        `ORDER BY elapsed_time DESC LIMIT n`, though, so what is actually returned is *the
        n most expensive successful queries* since that cutoff (or overall, with no
        `--since`) — not literally everything since then. `window_description` says so
        explicitly rather than implying full coverage, because `cost_share` denominators
        throughout the rest of this run are computed over exactly that truncated set.

        `sys_query_history` returns one row per *execution*, not per normalised statement —
        `pg_stat_statements` pre-aggregates by fingerprint, this view does not. So `calls`
        is always 1 on the `RawQueryRow`s built here; the collapse into one `QueryStat` per
        fingerprint, with `calls` and `total_time_ms` summed, happens in `ingest()` — see
        `tests/test_workload_redshift.py`'s test pinning that two executions of the same
        statement actually do collapse, rather than assuming it.

        That collapse is keyed on `ingest()`'s redacted, re-serialised SQL text, which is
        sensitive to exactly how the raw text was written — and `sys_query_history` stores
        the *verbatim* text the client sent, unlike `pg_stat_statements`, which Postgres has
        already parsed and re-serialised (identifiers folded to lowercase) before storing.
        So two executions that are, semantically, one statement can still fingerprint as two
        separate `QueryStat`s here if they differ only in identifier case or in an attached
        comment (an ORM query tag, for instance) — deliberately left undisclosed-but-real
        rather than "fixed" by normalising identifiers before `ingest()` runs, since a
        general case-fold cannot tell an unquoted identifier (case-insensitive) apart from a
        deliberately-quoted, case-sensitive one without risking folding a real distinction
        away. Pinned by
        `test_identifier_case_and_comments_can_still_split_one_statement_into_two_stats`.
        The consequence is real, not merely cosmetic: splitting one statement's cost across
        two `QueryStat`s inflates the number of groups the workload's total cost is spread
        over, which shrinks every `cost_share` and makes `--min-cost-share` correspondingly
        stricter for the affected statement.
        """
        cutoff = None if since is None else datetime.now(timezone.utc) - since
        rows = self._run(CAP_WORKLOAD, (cutoff, cutoff, limit))
        if cutoff is not None:
            window = (
                f"the {limit} most expensive successful queries since {cutoff.isoformat()} "
                "in sys_query_history (--since is honoured: sys_query_history carries a "
                "per-execution start_time, unlike pg_stat_statements)"
            )
        else:
            window = (
                f"the {limit} most expensive successful queries recorded in "
                "sys_query_history (no --since filter applied)"
            )
        return WorkloadFetch(
            rows=tuple(
                RawQueryRow(
                    sql=str(sql),
                    calls=1,
                    # elapsed_time is documented in microseconds; total_time_ms wants
                    # milliseconds. A NULL elapsed_time (not documented as possible for a
                    # 'success' row, but nothing here can prove it can't happen) must not
                    # raise past this point: an uncaught TypeError here would crash the
                    # whole run for one malformed row, exactly the failure `_run`'s
                    # try/except exists to prevent for a denied statement — that guarantee
                    # is worthless if a single bad row can still take down the run one
                    # level up. Treated as zero cost rather than dropping the row, so the
                    # call is still counted; zero cost is honestly conservative, since
                    # `total_time_ms` has no `None`/unknown state to fall back to.
                    total_time_ms=(_as_float(elapsed) / 1000.0) if elapsed is not None else 0.0,
                )
                for sql, elapsed in rows
            ),
            window_description=window,
        )

    def _schema_rows(self, schemas: tuple[str, ...]) -> list[tuple[object, ...]]:
        """CAP_SCHEMA rows, fetched at most once per schema tuple. See `_schema_cache`."""
        if schemas not in self._schema_cache:
            self._schema_cache[schemas] = self._run(CAP_SCHEMA, (list(schemas),))
        return self._schema_cache[schemas]

    def fetch_schema(self, schemas: tuple[str, ...]) -> dict:
        """Nested schema mapping for sqlglot qualify(): {schema: {table: {column: type}}}.

        `svv_columns` includes external (Spectrum) tables; `svv_table_info` — what
        `fetch_table_facts` reads — does not (an external table cannot carry SORTKEY or
        DISTSTYLE, so Redshift never lists one there). So this map can, correctly, carry a
        relation `fetch_table_facts` never returns a fact for: a query joining an external
        table still needs its columns to qualify, or the whole statement is dropped as
        unqualifiable — see `fetch_table_facts`'s docstring for the consequence on the
        other side of that gap.

        Nested rather than flat for the same reason `PostgresWorkloadAdapter.fetch_schema`
        is: a flat map cannot tell two same-named tables in different schemas apart.
        """
        schema: dict[str, dict[str, dict[str, str]]] = {}
        for schema_name, table, column, data_type in self._schema_rows(schemas):
            schema.setdefault(str(schema_name), {}).setdefault(str(table), {})[str(column)] = str(
                data_type
            )
        return schema

    def fetch_table_facts(
        self, schemas: tuple[str, ...], relations: frozenset[Relation]
    ) -> dict[Relation, TableFacts]:
        """Row estimates, sizes and columns — plus Redshift's own physical-design facts,
        stashed on `self.physical_facts` for a later task (see `RedshiftTableFacts`).

        Deliberately does **not** create an entry for every relation in `relations`, unlike
        `PostgresWorkloadAdapter.fetch_table_facts`. `svv_columns` includes external
        (Spectrum) tables and `svv_table_info` does not (see `fetch_schema`'s docstring),
        so a relation named in `relations` can legitimately never appear in this method's
        `svv_table_info` rows at all — forcing an entry anyway, every field `None`, would
        make that indistinguishable from a real table this method genuinely has no facts
        for. A relation's simple absence from the returned dict (and from
        `self.physical_facts`) is the signal a later task's rules must check for instead.

        **That absence is not, by itself, proof of a Spectrum table** — AWS also omits
        genuinely *empty* tables from `svv_table_info` — so a later task proposing
        SORTKEY/DISTKEY from this absence needs an additional signal to tell the two cases
        apart; see `RedshiftTableFacts`'s docstring. Recorded as a carry-forward, not
        solved here.
        """
        wanted = sorted({relation.table for relation in relations})
        columns: dict[Relation, list[str]] = {}
        for schema_name, table, column, _type in self._schema_rows(schemas):
            relation = Relation(schema=str(schema_name), table=str(table))
            if relation in relations:
                columns.setdefault(relation, []).append(str(column))

        facts: dict[Relation, TableFacts] = {}
        physical: dict[Relation, RedshiftTableFacts] = {}
        for (
            schema_name,
            table,
            tbl_rows,
            size,
            unsorted,
            stats_off,
            diststyle,
            sortkey1,
            skew_rows,
        ) in self._run(CAP_TABLE_FACTS, (list(schemas), wanted)):
            relation = Relation(schema=str(schema_name), table=str(table))
            # Same over-fetch guard `PostgresWorkloadAdapter.fetch_table_facts` documents:
            # the table parameter is bare names, so a same-named table in a *different*
            # requested schema that is not itself in `relations` can come back too.
            if relation not in relations:
                continue
            facts[relation] = TableFacts(
                relation=relation,
                row_estimate=_row_estimate(tbl_rows),
                size_bytes=_size_bytes(size),
                columns=tuple(columns.get(relation, ())),
            )
            physical[relation] = RedshiftTableFacts(
                unsorted=_as_float(unsorted) if unsorted is not None else None,
                stats_off=_as_float(stats_off) if stats_off is not None else None,
                diststyle=str(diststyle) if diststyle is not None else None,
                sortkey1=str(sortkey1) if sortkey1 is not None else None,
                skew_rows=_as_float(skew_rows) if skew_rows is not None else None,
            )
        self.physical_facts = physical
        return facts

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
