"""Postgres workload adapter: pg_stat_statements + catalog introspection, index rules."""

from __future__ import annotations

import hashlib
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from urllib.parse import unquote, urlparse

from sqlquality.models import (
    Aggregation,
    ColumnRole,
    ColumnUsage,
    ConnectionParams,
    Confidence,
    Proposal,
    RawQueryRow,
    TableFacts,
    Workload,
    WorkloadFetch,
    cost_share_of,
)
from sqlquality.workload.aggregate import mentions_table
from sqlquality.workload.base import (
    MAX_TIMEOUT_S,
    MIN_TIMEOUT_S,
    IntrospectionStatement,
    Querier,
    WorkloadAdapter,
)
from sqlquality.workload.fingerprint import FLAG_LEADING_WILDCARD_LIKE, FLAG_SELECT_STAR

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
#: profiles.yml keys forwarded to libpq unchanged, because the name already *is* the libpq
#: keyword. The TLS group is here for a security reason, not a completeness one: a profile
#: saying `sslmode: verify-full` that silently connects under libpq's default `prefer`
#: performs no certificate verification at all, and the user is never told. For a tool
#: pitched as safe to point at production that is the wrong way to fail.
_PG_PASSTHROUGH_FIELDS = frozenset(
    {"sslmode", "sslcert", "sslkey", "sslrootcert", "connect_timeout"}
)
#: profiles.yml keys whose values must never appear in any message we emit.
_SECRET_FIELDS = frozenset({"password", "pass"})


def _pg_fields(fields: dict[str, str]) -> dict[str, str]:
    """Translate profiles.yml keys to libpq keywords, dropping anything unrecognized."""
    translated = {_PG_FIELD_MAP[k]: v for k, v in fields.items() if k in _PG_FIELD_MAP}
    translated.update({k: v for k, v in fields.items() if k in _PG_PASSTHROUGH_FIELDS})
    return translated


def _dropped_pg_fields(fields: dict[str, str]) -> tuple[str, ...]:
    """profiles.yml keys this adapter cannot forward, by name.

    Names only, never values: one of them could be a secret (`sslpassword`), and this text
    goes to stderr and from there into CI logs.
    """
    return tuple(
        sorted(k for k in fields if k not in _PG_FIELD_MAP and k not in _PG_PASSTHROUGH_FIELDS)
    )


def _clamp_timeout_ms(timeout_s: int) -> int:
    """Statement timeout in milliseconds, clamped into a sane range.

    The CLI rejects an out-of-range value before reaching here; this is the safety net
    for any other caller. Bounds come from workload.base so the two cannot drift.
    """
    return max(MIN_TIMEOUT_S, min(int(timeout_s), MAX_TIMEOUT_S)) * 1000


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

    The password is added in **both** its percent-encoded and decoded forms.
    ``urlparse().password`` returns it still encoded, but libpq decodes a URI DSN before
    authenticating, so the value a real auth-failure message carries is the decoded one:
    for ``postgresql://u:p%40ss@h/db`` the driver reports ``p@ss`` while urlparse yields
    ``p%40ss``, and a token of only the encoded form never matches. Any password containing
    ``@``, ``:``, ``/``, ``%`` or a space hits this. The encoded form is kept too, since a
    URI-parse error can echo the raw string back instead.
    """
    secrets = tuple(
        value for key, value in params.fields.items() if key in _SECRET_FIELDS and value
    )
    if params.dsn:
        secrets += (params.dsn,)
        encoded = urlparse(params.dsn).password
        if encoded:
            secrets += (encoded,)
            decoded = unquote(encoded)
            if decoded != encoded:
                secrets += (decoded,)
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


#: Characters of hex kept from the fingerprint digest. 12 is 48 bits — ample for telling
#: apart the few hundred query groups one run reads, and short enough to sit in a table cell.
_FINGERPRINT_ID_LEN = 12


def _fingerprint_id(fingerprint: str) -> str:
    """A short, stable identity for a query group.

    `QueryStat.fingerprint` is the *entire* canonical SQL, so emitting it as evidence
    printed the whole statement a second time next to `sql` — for a long query, most of the
    proposal's evidence block, duplicated. What the field is for is identity: telling two
    query groups apart and correlating a proposal with a later run. A digest does that in
    twelve characters. The readable text stays in `sql`.

    Not a security boundary — the fingerprint is already redacted — so a fast digest is
    fine; sha256 is used because it is the unsurprising choice.
    """
    return hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:_FINGERPRINT_ID_LEN]


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


#: Below this row estimate a sequential scan is the right plan; an index is pure overhead.
MIN_ROWS_FOR_INDEX = 10_000
#: Wider composite indexes cost more to maintain than they repay in practice.
MAX_INDEX_ARITY = 3
#: An equality column with fewer distinct values than this is not selective enough to
#: justify HIGH confidence on its own.
SELECTIVE_NDV = 100.0

#: Appended to any index-creation rationale when the small-table gate could not run.
#: Shared by ADV001 and ADV004 so both index-creation rules disclose the same gap in the
#: same words — the operator reads the report, not the rule that produced it.
_UNKNOWN_ROWS_NOTE = (
    " The row count for this table is unknown, so it could not be checked against the "
    "small-table floor — on a small table an index is pure write overhead. Confirm the "
    "table's size before applying."
)


def _by_table(usage: Sequence[ColumnUsage]) -> dict[str, list[ColumnUsage]]:
    grouped: dict[str, list[ColumnUsage]] = {}
    for item in usage:
        grouped.setdefault(item.table, []).append(item)
    return grouped


def _is_prefix(shorter: tuple[str, ...], longer: tuple[str, ...]) -> bool:
    """True if ``shorter`` is a leading prefix of ``longer`` (equal lists included).

    The slice direction is the whole point and is easy to invert: an index on (a, b) covers
    a candidate (a), but an index on (a) does *not* cover a candidate (a, b). Both
    directions are tested.
    """
    return longer[: len(shorter)] == shorter


def _covered(candidate: tuple[str, ...], existing: Sequence[PgIndex]) -> str | None:
    """Name of an existing index whose leading columns already cover ``candidate``."""
    for index in existing:
        if _is_prefix(candidate, index.columns):
            return index.name
    return None


#: Schema every rule qualifies its DDL with unless told otherwise. It matches the CLI's
#: `--schema` default, so the default is truthful rather than merely convenient; the CLI
#: always passes the resolved schema explicitly (see PostgresWorkloadAdapter.propose).
DEFAULT_SCHEMA = "public"


def _qualified(schema: str, name: str) -> str:
    """`"schema"."name"` — both parts quoted.

    Generated DDL is read and run by an operator whose `search_path` we do not control.
    An unqualified `DROP INDEX "idx_cold"` drops whichever `idx_cold` their search_path
    resolves first, which is not necessarily the one we introspected.
    """
    return f"{_quote_ident(schema)}.{_quote_ident(name)}"


def propose_indexes(
    usage: Sequence[ColumnUsage],
    facts: Mapping[str, TableFacts],
    existing: Mapping[str, Sequence[PgIndex]],
    *,
    min_cost_share: float,
    min_rows: int = MIN_ROWS_FOR_INDEX,
    max_arity: int = MAX_INDEX_ARITY,
    schema: str = DEFAULT_SCHEMA,
    have_index_data: bool = True,
) -> list[Proposal]:
    """ADV001 — composite index candidates: equality columns first, one range column last.

    Equality-then-range is the standard B-tree ordering: once a range predicate is used,
    later columns can no longer be probed by equality.

    ``have_index_data`` is False when the existing-index catalog query was denied. The
    rule's central claim — "no existing index leads with these columns" — is then
    unknowable, so it is not made and confidence is capped at LOW. ``existing`` being
    empty cannot distinguish "no such index" from "could not look", which is exactly the
    conflation the row-estimate branch below exists to prevent.
    """
    proposals: list[Proposal] = []
    for table, items in sorted(_by_table(usage).items()):
        table_facts = facts.get(table)
        rows = table_facts.row_estimate if table_facts else None
        if rows is not None and rows < min_rows:
            continue

        equality = sorted(
            (i for i in items if i.role is ColumnRole.EQUALITY),
            key=lambda i: i.cost_ms,
            reverse=True,
        )
        ranges = sorted(
            (i for i in items if i.role in (ColumnRole.RANGE, ColumnRole.SORT)),
            key=lambda i: i.cost_ms,
            reverse=True,
        )
        if ranges:
            candidate = equality[: max_arity - 1] + ranges[:1]
        else:
            candidate = equality[:max_arity]
        # `equality` and `ranges` are filtered from the same list by *role*, so a column
        # used in two roles — `where id = $1` alongside `order by id desc`, entirely
        # ordinary — lands in both and the concatenation above repeats it. That produced
        # `CREATE INDEX ON orders (id, id)` at HIGH confidence, which `_covered` could not
        # even suppress: `_is_prefix(("id","id"), ("id",))` is False, so the table's own
        # primary key did not match. Equality wins because equality-first is the B-tree
        # ordering the whole rule is built on.
        chosen: list[ColumnUsage] = []
        picked: set[str] = set()
        for item in candidate:
            if item.column in picked:
                continue
            picked.add(item.column)
            chosen.append(item)
        if not chosen:
            continue

        cost_share = max(i.cost_share for i in chosen)
        if cost_share < min_cost_share:
            continue

        columns = tuple(i.column for i in chosen)
        covered_by = _covered(columns, existing.get(table, ()))
        if covered_by is not None:
            continue

        ndv = table_facts.ndv if table_facts else {}
        leading_ndv = ndv.get(columns[0])
        if rows is None or not have_index_data:
            # Either the small-table gate could not run, or we could not check whether an
            # index already leads with these columns. The cost evidence is real, so the
            # proposal is kept — a user whose grants are incomplete should still get
            # advice — but at LOW, and the rationale says which check was skipped.
            # Reporting MEDIUM or HIGH here would be the exact confident-but-wrong failure
            # this rule set exists to avoid: nothing is assumed away, it is unknown.
            confidence = Confidence.LOW
        elif leading_ndv is None:
            confidence = Confidence.MEDIUM
        elif leading_ndv >= SELECTIVE_NDV:
            confidence = Confidence.HIGH
        else:
            confidence = Confidence.LOW

        if have_index_data:
            rationale = (
                "These columns carry the table's hottest predicates and no existing index "
                "leads with them. Equality columns come first so the range column can be "
                "scanned last."
            )
        else:
            rationale = (
                "These columns carry the table's hottest predicates. Equality columns come "
                "first so the range column can be scanned last. The existing-index list "
                "could not be read, so whether an index already leads with these columns "
                "is unknown — check before applying."
            )
        if rows is None:
            rationale += _UNKNOWN_ROWS_NOTE

        proposals.append(
            Proposal(
                code="ADV001",
                title=f"Add index on {table}({', '.join(columns)})",
                rationale=rationale,
                evidence={
                    "table": table,
                    "columns": columns,
                    "roles": tuple(i.role.value for i in chosen),
                    "cost_share": cost_share,
                    "calls": max(i.calls for i in chosen),
                    "fingerprints": max(i.fingerprints for i in chosen),
                    "row_estimate": rows,
                    "leading_ndv": leading_ndv,
                },
                confidence=confidence,
                ddl=(
                    f"CREATE INDEX ON {_qualified(schema, table)} "
                    f"({', '.join(_quote_ident(c) for c in columns)});"
                ),
            )
        )
    return proposals


def propose_unused_indexes(
    existing: Mapping[str, Sequence[PgIndex]],
    *,
    hot_tables: frozenset[str],
    schema: str = DEFAULT_SCHEMA,
) -> list[Proposal]:
    """ADV002 — indexes with zero recorded scans, excluding constraint-backing indexes.

    Confidence is capped at MEDIUM: idx_scan accumulates only since the last statistics
    reset, so zero scans cannot prove an index is unused across a full business cycle.
    """
    proposals: list[Proposal] = []
    for table in sorted(hot_tables):
        for index in existing.get(table, ()):
            if index.scans != 0 or index.is_unique or index.is_primary:
                continue
            proposals.append(
                Proposal(
                    code="ADV002",
                    title=f"Drop unused index {index.name} on {table}",
                    rationale=(
                        "No recorded scans since the last statistics reset. Verify the "
                        "reset time covers a full business cycle before dropping."
                    ),
                    evidence={
                        "table": table,
                        "index": index.name,
                        "columns": index.columns,
                        "scans": index.scans,
                        "size_bytes": index.size_bytes,
                    },
                    confidence=Confidence.MEDIUM,
                    ddl=f"DROP INDEX {_qualified(schema, index.name)};",
                )
            )
    return proposals


def propose_redundant_indexes(
    existing: Mapping[str, Sequence[PgIndex]],
    *,
    schema: str = DEFAULT_SCHEMA,
) -> list[Proposal]:
    """ADV003 — an index whose column list is a strict prefix of another's is redundant.

    Capped at MEDIUM, not HIGH. ``PgIndex`` carries no ``indpred``/``indexprs``, so a
    partial index (``WHERE shipped_at IS NULL``) and an expression index are both
    indistinguishable from plain ones here — and for a partial index "serves the same
    lookups" is simply false. The README says so, but a README does not travel inside the
    ``.sql`` file the operator runs, so the rationale carries the caveat too.
    """
    proposals: list[Proposal] = []
    for table, indexes in sorted(existing.items()):
        for narrow in indexes:
            if narrow.is_unique or narrow.is_primary:
                continue
            wider = next(
                (
                    other
                    for other in indexes
                    if other.name != narrow.name
                    and len(other.columns) > len(narrow.columns)
                    and _is_prefix(narrow.columns, other.columns)
                ),
                None,
            )
            if wider is None:
                continue
            proposals.append(
                Proposal(
                    code="ADV003",
                    title=f"Drop redundant index {narrow.name} on {table}",
                    rationale=(
                        f"Its columns are a leading prefix of {wider.name}, which can "
                        "serve the same lookups. This comparison is on column lists only: "
                        "sqlquality cannot see a partial index's WHERE predicate or an "
                        "expression index's expressions, and a partial index does not "
                        "cover the same rows as a wider full one. Confirm that neither "
                        "index is partial or expression-based before dropping."
                    ),
                    evidence={
                        "table": table,
                        "index": narrow.name,
                        "columns": narrow.columns,
                        "superseded_by": wider.name,
                        "superseding_columns": wider.columns,
                        "size_bytes": narrow.size_bytes,
                    },
                    confidence=Confidence.MEDIUM,
                    ddl=f"DROP INDEX {_qualified(schema, narrow.name)};",
                )
            )
    return proposals


#: A table with at least this many columns makes `SELECT *` materially wasteful.
WIDE_TABLE_COLUMNS = 15

_NULL_ROLE_PREDICATE = {
    ColumnRole.NULL_CHECK: "IS NULL",
    ColumnRole.NOT_NULL_CHECK: "IS NOT NULL",
}


def _first_co_occurring(
    equality: Sequence[ColumnUsage], null_checks: Sequence[ColumnUsage]
) -> tuple[ColumnUsage, ColumnUsage, frozenset[str]] | None:
    """Highest-cost (equality, null-check) pair that some single query uses together.

    Both inputs arrive sorted by cost descending, so the first overlapping pair found is
    the most expensive supported one. Returns the shared fingerprints alongside, since that
    overlap *is* the evidence for the proposal.
    """
    for left in equality:
        for right in null_checks:
            shared = left.fingerprint_ids & right.fingerprint_ids
            if shared:
                return left, right, shared
    return None


def propose_partial_indexes(
    usage: Sequence[ColumnUsage],
    facts: Mapping[str, TableFacts],
    *,
    min_cost_share: float,
    min_rows: int = MIN_ROWS_FOR_INDEX,
    schema: str = DEFAULT_SCHEMA,
) -> list[Proposal]:
    """ADV004 — index the hot equality column, restricted by a hot null-check predicate.

    Only structural predicates qualify. Literal-valued partial indexes would need the
    literals retained, which default redaction deliberately discards.

    Gated on table size exactly as ADV001 is: both rules create an index, and below the
    floor a sequential scan is the right plan whichever rule proposed it. An unknown row
    count caps confidence at LOW rather than being assumed large.
    """
    proposals: list[Proposal] = []
    for table, items in sorted(_by_table(usage).items()):
        table_facts = facts.get(table)
        rows = table_facts.row_estimate if table_facts else None
        if rows is not None and rows < min_rows:
            continue
        equality = sorted(
            (i for i in items if i.role is ColumnRole.EQUALITY),
            key=lambda i: i.cost_ms,
            reverse=True,
        )
        null_checks = sorted(
            (i for i in items if i.role in _NULL_ROLE_PREDICATE),
            key=lambda i: i.cost_ms,
            reverse=True,
        )
        if not equality or not null_checks:
            continue
        # Pair only columns that some single query actually filters on *together*. Taking
        # the hottest of each list independently can pair a filter from query A with a null
        # check from query B that never co-occur — and a partial index whose guard no
        # filtering query uses helps nothing while still costing writes. The overlap is the
        # evidence for this proposal, so without one there is no proposal.
        pair = _first_co_occurring(equality, null_checks)
        if pair is None:
            continue
        leading, guard, shared = pair
        cost_share = max(leading.cost_share, guard.cost_share)
        if cost_share < min_cost_share:
            continue
        predicate = _NULL_ROLE_PREDICATE[guard.role]
        rationale = (
            "The hot predicates always pair this lookup with the same null check, "
            "so a partial index covers them at a fraction of the size."
        )
        if rows is None:
            rationale += _UNKNOWN_ROWS_NOTE
        proposals.append(
            Proposal(
                code="ADV004",
                title=(
                    f"Partial index on {table}({leading.column}) WHERE {guard.column} {predicate}"
                ),
                rationale=rationale,
                evidence={
                    "table": table,
                    "columns": (leading.column,),
                    "guard_column": guard.column,
                    "guard_predicate": predicate,
                    "cost_share": cost_share,
                    "calls": max(leading.calls, guard.calls),
                    "row_estimate": rows,
                    #: How many query groups filter on both columns together. This is what
                    #: makes the proposal supported rather than a guess.
                    "co_occurring_fingerprints": len(shared),
                },
                confidence=Confidence.LOW if rows is None else Confidence.MEDIUM,
                ddl=(
                    f"CREATE INDEX ON {_qualified(schema, table)} "
                    f"({_quote_ident(leading.column)}) "
                    f"WHERE {_quote_ident(guard.column)} {predicate};"
                ),
            )
        )
    return proposals


def propose_sargability(
    usage: Sequence[ColumnUsage],
    workload: Workload,
    *,
    min_cost_share: float,
) -> list[Proposal]:
    """ADV005 — predicates an index cannot serve, ranked by the cost they carry."""
    proposals: list[Proposal] = []
    for item in sorted(usage, key=lambda i: i.cost_ms, reverse=True):
        if item.role is not ColumnRole.NON_SARGABLE or item.cost_share < min_cost_share:
            continue
        proposals.append(
            Proposal(
                code="ADV005",
                title=f"Non-sargable predicate on {item.table}.{item.column}",
                rationale=(
                    "The column is wrapped in a cast or function inside a predicate, so a "
                    "plain B-tree index cannot be used. Rewrite the predicate to leave the "
                    "column bare, or add a matching expression index."
                ),
                evidence={
                    "table": item.table,
                    "column": item.column,
                    "cost_share": item.cost_share,
                    "calls": item.calls,
                    "fingerprints": item.fingerprints,
                },
                confidence=Confidence.HIGH,
                ddl=None,
            )
        )

    total = workload.total_cost_ms
    for stat in workload.stats:
        if FLAG_LEADING_WILDCARD_LIKE not in stat.flags:
            continue
        share = (stat.total_time_ms / total) if total else 0.0
        if share < min_cost_share:
            continue
        proposals.append(
            Proposal(
                code="ADV005",
                title="Leading-wildcard LIKE in a hot query group",
                rationale=(
                    "A LIKE pattern beginning with '%' cannot use a B-tree index. Consider "
                    "a trigram index or full-text search. The pattern itself was redacted, "
                    "so the specific column is not attributed here."
                ),
                evidence={
                    "fingerprint": _fingerprint_id(stat.fingerprint),
                    "sql": stat.sql,
                    "cost_share": share,
                    "calls": stat.calls,
                },
                confidence=Confidence.MEDIUM,
                ddl=None,
            )
        )
    return proposals


def propose_select_star(
    workload: Workload,
    facts: Mapping[str, TableFacts],
    *,
    min_cost_share: float,
    min_columns: int = WIDE_TABLE_COLUMNS,
) -> list[Proposal]:
    """ADV006 — hot query groups projecting a star from a wide table."""
    wide = {name for name, fact in facts.items() if len(fact.columns) >= min_columns}
    if not wide:
        return []
    total = workload.total_cost_ms
    proposals: list[Proposal] = []
    for stat in workload.stats:
        if FLAG_SELECT_STAR not in stat.flags:
            continue
        touched = sorted(name for name in wide if mentions_table(name, stat.sql))
        if not touched:
            continue
        share = (stat.total_time_ms / total) if total else 0.0
        if share < min_cost_share:
            continue
        proposals.append(
            Proposal(
                code="ADV006",
                title=f"Hot SELECT * over wide table(s): {', '.join(touched)}",
                rationale=(
                    "Projecting every column of a wide table moves data no consumer asked "
                    "for. List the columns the query actually needs."
                ),
                evidence={
                    "tables": tuple(touched),
                    "column_counts": {name: len(facts[name].columns) for name in touched},
                    "cost_share": share,
                    "calls": stat.calls,
                    "fingerprint": _fingerprint_id(stat.fingerprint),
                    # The identity above is a digest, so the readable text has to be
                    # carried explicitly — ADV005 already does this. Redacted by
                    # default, like every other query text in the report.
                    "sql": stat.sql,
                },
                confidence=Confidence.MEDIUM,
                ddl=None,
            )
        )
    return proposals


def _quote_ident(name: str) -> str:
    """Quote an identifier the way Postgres does, doubling any embedded double quote.

    Unquoted identifiers produce invalid DDL for anything needing quoting — a mixed-case
    name, or one colliding with a reserved word. Quoting makes the identifier a single
    token, safe to parse even if it contains unusual characters. Identifiers containing
    line breaks are handled specially in render_ddl to keep the output visually safe.
    """
    return '"' + name.replace('"', '""') + '"'


def _comment_lines(text: str) -> list[str]:
    """Comment each line of text for safe inclusion in a SQL script.

    Each line is prefixed with '-- ', ensuring that even a multi-line title from
    a schema identifier cannot break out of comment mode and become executable.
    """
    return [f"-- {line}" for line in text.splitlines()]


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
        #: CAP_SCHEMA rows per schema tuple. Both fetch_schema and fetch_table_facts need
        #: them, and running the statement twice did twice the catalog work and — worse —
        #: appended two identical entries to `degraded` when it was denied, telling the user
        #: about one missing grant twice.
        self._schema_cache: dict[tuple[str, ...], list[tuple[object, ...]]] = {}

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

        # Silence is the failure mode being fixed here: a dropped `sslmode` downgrades the
        # connection with no signal at all. Key names only — see _dropped_pg_fields.
        dropped = _dropped_pg_fields(params.fields)
        if dropped:
            print(
                f"warning: ignoring connection setting(s) not supported by the Postgres "
                f"adapter: {', '.join(dropped)}. Pass --dsn if you need them.",
                file=sys.stderr,
            )

        # Everything we know to be secret, so a driver exception can be proven clean rather
        # than trusted.
        secrets = _secrets_for(params)

        failure: str | None = None
        try:
            # Inside the scrubbing envelope: psycopg raises from make_conninfo on an
            # unusable keyword, and that message can quote the offending value — which for
            # the `password` keyword is the password.
            conninfo = params.dsn or psycopg.conninfo.make_conninfo(**_pg_fields(params.fields))
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
            # No "Could not connect" prefix here — the CLI adds it. Prefixing at both
            # layers printed "Could not connect: Could not connect: ..." on the most
            # common failure a user hits.
            raise ConnectionError(failure)

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

    def _schema_rows(self, schemas: tuple[str, ...]) -> list[tuple[object, ...]]:
        """CAP_SCHEMA rows, fetched at most once per schema tuple. See `_schema_cache`."""
        if schemas not in self._schema_cache:
            self._schema_cache[schemas] = self._run(CAP_SCHEMA, (list(schemas),))
        return self._schema_cache[schemas]

    def fetch_schema(self, schemas: tuple[str, ...]) -> dict:
        schema: dict[str, dict[str, str]] = {}
        for table, column, data_type in self._schema_rows(schemas):
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
        for table, column, _type in self._schema_rows(schemas):
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

    #: Highest confidence first, then largest cost share — the reading order a human wants.
    _CONFIDENCE_ORDER = {Confidence.HIGH: 0, Confidence.MEDIUM: 1, Confidence.LOW: 2}

    @classmethod
    def _dedupe_by_ddl(cls, proposals: list[Proposal]) -> list[Proposal]:
        """Collapse proposals that would run identical DDL, keeping the strongest evidence.

        An index with no recorded scans that is *also* a prefix of a wider index gets
        flagged by both ADV002 and ADV003, producing two entries with the same
        `DROP INDEX`. They do not contradict each other, but a reader should not have to
        notice they are the same object twice. ADV003 wins ties because prefix redundancy
        is structural — provable from the column lists alone — whereas ADV002 rests on a
        scan counter that only covers the window since the last statistics reset.

        That preference used to fall out of the confidence order on its own, when ADV003
        was HIGH. Capping ADV003 at MEDIUM (it cannot see partial-index predicates) made
        the two rules tie, and a tie was resolved by list order — silently handing the
        collapse to ADV002. `_RULE_PRECEDENCE` states the preference instead of relying on
        it emerging.
        """
        best: dict[str, Proposal] = {}
        for proposal in proposals:
            if not proposal.ddl:
                continue
            incumbent = best.get(proposal.ddl)
            if incumbent is None or cls._dedupe_rank(proposal) < cls._dedupe_rank(incumbent):
                best[proposal.ddl] = proposal
        return [p for p in proposals if not p.ddl or best[p.ddl] is p]

    #: Lower wins when two rules propose identical DDL at the same confidence. Only ADV003
    #: is named: its evidence is structural, every other rule's rests on a counter or an
    #: estimate. Anything unlisted sorts after it.
    _RULE_PRECEDENCE = {"ADV003": 0}

    @classmethod
    def _dedupe_rank(cls, proposal: Proposal) -> tuple[int, int]:
        return (
            cls._CONFIDENCE_ORDER[proposal.confidence],
            cls._RULE_PRECEDENCE.get(proposal.code, 1),
        )

    def propose(
        self,
        aggregation: Aggregation,
        facts: dict[str, TableFacts],
        workload: Workload,
        *,
        min_cost_share: float,
    ) -> list[Proposal]:
        existing = self.fetch_indexes(self.schemas, aggregation.tables)
        # The rules are module-level and emit DDL, so they need the schema they are talking
        # about. Only one schema is ever introspected (the CLI rejects more than one), so
        # this is unambiguous rather than a guess about which one a proposal belongs to.
        schema = self.schemas[0] if self.schemas else DEFAULT_SCHEMA
        # An empty `existing` means one of two very different things: this table genuinely
        # has no indexes, or the catalog query was denied. Only the adapter can tell, so it
        # is the adapter's job to say — ADV001 must not claim "no existing index leads with
        # them" on the strength of a permission error.
        have_index_data = not any(cap == CAP_INDEXES for cap, _ in self.degraded)
        proposals = [
            *propose_indexes(
                aggregation.usage,
                facts,
                existing,
                min_cost_share=min_cost_share,
                schema=schema,
                have_index_data=have_index_data,
            ),
            *propose_partial_indexes(
                aggregation.usage, facts, min_cost_share=min_cost_share, schema=schema
            ),
            *propose_sargability(aggregation.usage, workload, min_cost_share=min_cost_share),
            *propose_select_star(workload, facts, min_cost_share=min_cost_share),
            *propose_unused_indexes(existing, hot_tables=aggregation.tables, schema=schema),
            *propose_redundant_indexes(existing, schema=schema),
        ]
        return sorted(self._dedupe_by_ddl(proposals), key=self._ranking_key)

    @classmethod
    def _ranking_key(cls, proposal: Proposal) -> tuple[int, float, str, str]:
        """Highest confidence first, then largest cost share — the reading order a human
        wants — with a canonical tiebreak so equal-confidence equal-cost proposals do not
        reorder between runs and make the CLI's tests flaky.

        `cost_share_of` rather than `float(evidence.get(...))`: bool is an int subclass, so
        a stray True became -1.0 and sorted a fabricated share ahead of a genuinely hot
        proposal, at the top of the list the CLI presents as "read this first".
        """
        return (
            cls._CONFIDENCE_ORDER[proposal.confidence],
            -(cost_share_of(proposal.evidence) or 0.0),
            proposal.code,
            proposal.title,
        )

    def render_ddl(self, proposals: list[Proposal]) -> str:
        """A commented, reviewable script. sqlquality never executes this."""
        header = [
            "-- Generated by `sqlquality advise` — REVIEW BEFORE RUNNING.",
            "-- sqlquality does not execute this script and has not validated it against",
            "-- your workload's write patterns. Each statement is advisory.",
            "--",
            "-- On a live table prefer CREATE INDEX CONCURRENTLY / DROP INDEX CONCURRENTLY:",
            "-- the plain forms below take a lock that blocks writes for the duration.",
            "-- Note that CONCURRENTLY cannot run inside a transaction block, so apply those",
            "-- statements individually rather than piping this whole file into one.",
            "",
        ]
        body: list[str] = []
        for proposal in proposals:
            if not proposal.ddl:
                continue
            if "\n" in proposal.ddl or "\r" in proposal.ddl:
                # An identifier containing a line break cannot be emitted as a single-line
                # statement. Quoting already makes it *semantically* safe — psql parses the
                # whole thing as one quoted identifier, so nothing extra executes — but the
                # file would still show a line reading like a statement to anyone skimming,
                # which is the property this script exists to provide. Truncating the name
                # instead would emit DDL targeting an object that does not exist. So it is
                # commented out in full with the reason, rather than rendered wrong or
                # silently dropped.
                body.append("-- NOT RENDERED: an identifier in this proposal contains a line")
                body.append("-- break, so it cannot be emitted as a single-line statement.")
                body.append("-- Verify the name and apply this by hand:")
                body.extend(_comment_lines(proposal.ddl))
                body.append("")
                continue
            share = cost_share_of(proposal.evidence)
            share_text = f", {share:.1%} of workload cost" if share is not None else ""
            body.append(f"-- {proposal.code} [{proposal.confidence.value}{share_text}]")
            body.extend(_comment_lines(proposal.title))
            body.append(proposal.ddl)
            body.append("")
        if not body:
            body = ["-- No DDL proposals — every finding is advisory-only.", ""]
        return "\n".join(header + body)
