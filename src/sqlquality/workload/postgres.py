"""Postgres workload adapter: pg_stat_statements + catalog introspection, index rules."""

from __future__ import annotations

import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import timedelta

from sqlglot import exp

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
    cost_share_of,
)
from sqlquality.sqlast import parse
from sqlquality.workload.aggregate import mentions_identifier
from sqlquality.workload.base import (
    MAX_TIMEOUT_S,
    MIN_TIMEOUT_S,
    IntrospectionStatement,
    Querier,
    WorkloadAdapter,
)
from sqlquality.workload.fingerprint import (
    FLAG_LEADING_WILDCARD_LIKE,
    FLAG_SELECT_STAR,
    fingerprint_digests,
    fingerprint_id,
)
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


def _as_int(value: object) -> int:
    """Coerce a driver row value to int.

    Querier rows are `tuple[object, ...]`, so this coercion is unavoidably unchecked. It
    lives in one auditable helper rather than at a dozen call sites.
    """
    return int(value)  # type: ignore[call-overload]


def _as_float(value: object) -> float:
    """Coerce a driver row value to float. See _as_int."""
    return float(value)  # type: ignore[arg-type]


def _row_estimate(value: object) -> int | None:
    """`pg_class.reltuples`, with Postgres's never-analyzed sentinel translated to unknown.

    Postgres 14+ stores -1 in `reltuples` for a table that has never been analyzed —
    distinct from 0, which means analyzed and genuinely empty. Passed through as-is, -1
    reads as a tiny table to the small-table gate in `propose_indexes`, which then
    suppresses every proposal for that table with no message. The window where this bites
    is exactly when someone reaches for `advise`: a freshly loaded or migrated table,
    before autovacuum's first ANALYZE, with slow queries. `None` already means "unknown"
    throughout — it proposes at LOW and says the row count could not be checked — so
    translating the sentinel here is the whole fix.
    """
    rows = _as_int(value)
    return None if rows < 0 else rows


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
    is_partial: bool = False
    predicate: str | None = None
    has_expressions: bool = False
    definition: str | None = None
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
    #: True when the index has a WHERE predicate. A partial index does not serve an
    #: unfiltered lookup, so it can never be assumed to cover a proposed index.
    is_partial: bool = False
    #: The rendered predicate, for showing an operator why a drop was not recommended.
    predicate: str | None = None
    #: True when any indexed position is an expression rather than a plain column. Such a
    #: position contributes no name to `columns`, so the tuple understates the index.
    has_expressions: bool = False
    #: The full CREATE INDEX text, the only place an expression is legible.
    definition: str | None = None


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


def _by_relation(usage: Sequence[ColumnUsage]) -> dict[Relation, list[ColumnUsage]]:
    grouped: dict[Relation, list[ColumnUsage]] = {}
    for item in usage:
        grouped.setdefault(item.relation, []).append(item)
    return grouped


def _is_prefix(shorter: tuple[str, ...], longer: tuple[str, ...]) -> bool:
    """True if ``shorter`` is a leading prefix of ``longer`` (equal lists included).

    The slice direction is the whole point and is easy to invert: an index on (a, b) covers
    a candidate (a), but an index on (a) does *not* cover a candidate (a, b). Both
    directions are tested.
    """
    return longer[: len(shorter)] == shorter


def _covered(candidate: tuple[str, ...], existing: Sequence[PgIndex]) -> str | None:
    """Name of a *plain* existing index whose leading columns already cover ``candidate``.

    Partial and expression indexes are excluded, for opposite reasons that land in the same
    place. A partial index does not serve an unfiltered lookup, so calling it coverage
    silently withholds a real proposal. An expression index's `columns` tuple understates it
    — the expression positions contribute no name — so a prefix match against it is not a
    match at all. Neither can be *proven* irrelevant either, which is why `propose_indexes`
    discloses them instead of dropping them on the floor.
    """
    for index in existing:
        if index.is_partial or index.has_expressions:
            continue
        if _is_prefix(candidate, index.columns):
            return index.name
    return None


#: A period followed by whitespace, i.e. a sentence boundary in this module's own generated
#: rationale prose — never an abbreviation, since none of the rationale text this module
#: writes uses one.
_SENTENCE_BOUNDARY = re.compile(r"(?<=\.)\s+")


def _sentences(text: str) -> list[str]:
    """Split rationale prose into its component sentences.

    Used to detect whole-sentence duplication when folding one proposal's rationale into
    another's — not a general-purpose sentence splitter, just good enough for text this
    same module generates and controls the wording of.
    """
    return [s for s in (chunk.strip() for chunk in _SENTENCE_BOUNDARY.split(text.strip())) if s]


def _qualified(schema: str, name: str) -> str:
    """`"schema"."name"` — both parts quoted.

    Generated DDL is read and run by an operator whose `search_path` we do not control.
    An unqualified `DROP INDEX "idx_cold"` drops whichever `idx_cold` their search_path
    resolves first, which is not necessarily the one we introspected.
    """
    return f"{_quote_ident(schema)}.{_quote_ident(name)}"


def propose_indexes(
    usage: Sequence[ColumnUsage],
    facts: Mapping[Relation, TableFacts],
    existing: Mapping[Relation, Sequence[PgIndex]],
    *,
    min_cost_share: float,
    min_rows: int = MIN_ROWS_FOR_INDEX,
    max_arity: int = MAX_INDEX_ARITY,
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

    Extending the composite requires *joint* support — every added column sharing a query
    group with every column already chosen, tracked as a running intersection of
    ``fingerprint_ids`` — exactly as ADV008 does. Without it this rule welded together the
    hottest equality columns and the hottest range column from a relation whether or not any
    single query used them together, and `cost_share` could not filter that out because it is
    the *max* over the chosen columns, not the min. Measured case: a `DECLARE ... CURSOR FOR
    SELECT ... WHERE tenant_id = $1` read costing 0.003% of the window put `tenant_id` into
    position 2 of an otherwise correct `(customer_id, created_at)`, and
    `(customer_id, tenant_id, created_at)` cannot satisfy the hot query's `ORDER BY
    created_at` that `(customer_id, created_at)` serves — a strictly worse index, emitted at
    HIGH, for the query carrying most of the workload's cost.

    Evidence carries `co_occurring_fingerprints` (the size of that intersection) and
    deliberately no plain `fingerprints` count, matching ADV004 and ADV008 — see
    `propose_grouping_indexes` for the principle. A rule whose claim is about columns
    appearing *together* must not also report a per-column count: reports render evidence as
    bare `k=v` pairs with no per-rule text, so `fingerprints: 1` beside a three-column index
    read as "one query group uses all three" when zero did.
    """
    proposals: list[Proposal] = []
    for relation, items in sorted(_by_relation(usage).items()):
        table_facts = facts.get(relation)
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
        #
        # `shared` is the running intersection of the chosen columns' query groups, so a
        # column only joins the composite when some single query group filters on it
        # *together with* everything already chosen — the same guard, for the same reason,
        # as `propose_grouping_indexes`. A candidate rejected for lack of joint support does
        # not narrow `shared`, so a later candidate that does co-occur with the chosen set
        # can still join it.
        chosen: list[ColumnUsage] = []
        picked: set[str] = set()
        shared: frozenset[str] = frozenset()
        for item in candidate:
            if item.column in picked:
                continue
            if not chosen:
                chosen.append(item)
                picked.add(item.column)
                shared = item.fingerprint_ids
                continue
            overlap = shared & item.fingerprint_ids
            if not overlap:
                continue
            chosen.append(item)
            picked.add(item.column)
            shared = overlap
        if not chosen:
            continue

        cost_share = max(i.cost_share for i in chosen)
        if cost_share < min_cost_share:
            continue

        columns = tuple(i.column for i in chosen)
        covered_by = _covered(columns, existing.get(relation, ()))
        if covered_by is not None:
            continue

        table_indexes = existing.get(relation, ())
        partial_skipped = tuple(
            index.name
            for index in table_indexes
            if index.is_partial and _is_prefix(columns, index.columns)
        )
        # Only expression indexes whose definition mentions the leading column are worth
        # naming. Proving `lower(status)` equivalent to `status` would need the expression
        # parsed and matched; naming it lets the operator make that call in one glance.
        #
        # Whole-identifier matching, not a substring test. `columns[0] in definition` reports
        # a candidate on `id` against an index on `lower(guid)`, and the rationale then tells
        # the operator an index "mentions id" when it does not — a claim the tool cannot
        # support, in the string someone reads while deciding whether to run DDL. Verified
        # that `\b` keeps the true positives: `lower(status)`, `lower(customer_id::text)`
        # and `(id::text)` all still match, because Postgres separates identifiers with
        # parens, commas, dots and `::`, none of which are word characters.
        expression_indexes = tuple(
            index.name
            for index in table_indexes
            if index.has_expressions and mentions_identifier(columns[0], index.definition or "")
        )

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
        # Same caveat as ADV007's, word for word: a low leading NDV is why this proposal is
        # LOW rather than HIGH, and until this line existed ADV001 downgraded silently while
        # ADV007 explained itself for the identical reason — an asymmetry an operator should
        # not have to notice depends on which rule happened to propose the index.
        if leading_ndv is not None and leading_ndv < SELECTIVE_NDV:
            rationale += (
                f" Only about {leading_ndv:.0f} distinct values, so the index may not be "
                "selective enough to be worth its write cost."
            )
        # The MEDIUM rung used to say nothing at all, which reads as a considered judgement
        # when it is really an absence of evidence: the statistics for the leading column
        # were not available, so the selectivity check simply did not run. The whole rule set
        # turns on disclosing a check that could not run rather than assuming its answer, and
        # this was the one confidence level that stayed silent about why.
        elif leading_ndv is None and rows is not None and have_index_data:
            rationale += (
                f" No distinct-value statistics for {columns[0]}, so how selective this index "
                "would be could not be checked — run ANALYZE on the table for a firmer answer."
            )
        if partial_skipped:
            rationale += (
                f" A partial index ({', '.join(partial_skipped)}) leads with these columns "
                "but carries a WHERE predicate, so it does not serve an unfiltered lookup — "
                "it is not treated as covering this proposal."
            )
        if expression_indexes:
            rationale += (
                f" An expression index ({', '.join(expression_indexes)}) mentions "
                f"{columns[0]}; sqlquality cannot tell whether it already serves this "
                "lookup, so confirm before applying."
            )

        proposals.append(
            Proposal(
                code="ADV001",
                title=f"Add index on {relation}({', '.join(columns)})",
                rationale=rationale,
                evidence={
                    "schema": relation.schema,
                    "table": relation.table,
                    "columns": columns,
                    "roles": tuple(i.role.value for i in chosen),
                    "cost_share": cost_share,
                    "calls": max(i.calls for i in chosen),
                    #: How many query groups filter on *every* chosen column together — the
                    #: running intersection, not a per-column count. Same name and same
                    #: meaning as ADV004's and ADV008's identical field, and deliberately no
                    #: plain `fingerprints` key beside it: this proposal is justified by the
                    #: columns appearing together, so a per-column count rendered as a bare
                    #: `k=v` pair next to the joint one can only read as support that is not
                    #: there.
                    "co_occurring_fingerprints": len(shared),
                    #: The query groups behind the count above, digested (not the whole
                    #: canonical SQL `shared` actually holds — `ColumnUsage.fingerprint_ids`'
                    #: name is a known wart, see `fingerprint_digests`'s own docstring) and
                    #: sorted, so `sqlquality verify` (a later task) can name which groups
                    #: back this proposal and diff two runs' evidence without reordering
                    #: noise.
                    "fingerprint_digests": fingerprint_digests(shared),
                    "row_estimate": rows,
                    "leading_ndv": leading_ndv,
                    "partial_indexes_skipped": partial_skipped,
                    "expression_indexes": expression_indexes,
                },
                confidence=confidence,
                ddl=(
                    f"CREATE INDEX ON {_qualified(relation.schema, relation.table)} "
                    f"({', '.join(_quote_ident(c) for c in columns)});"
                ),
            )
        )
    return proposals


def propose_join_keys(
    usage: Sequence[ColumnUsage],
    facts: Mapping[Relation, TableFacts],
    existing: Mapping[Relation, Sequence[PgIndex]],
    *,
    min_cost_share: float,
    min_rows: int = MIN_ROWS_FOR_INDEX,
    have_index_data: bool = True,
) -> list[Proposal]:
    """ADV007 — a hot join key with no index leading with it.

    Deliberately one proposal per join column rather than a composite: two joins against the
    same table want two indexes, and a composite `(a, b)` serves only probes on `a`.

    Not folded into ADV001. That rule's rationale is the B-tree ordering argument —
    "equality columns first so the range column can be scanned last" — and a join key is not
    a filter predicate: it is probed once per outer row. Adding JOIN to ADV001's candidate
    list would have left that sentence in the report while making it false of the index it
    describes. Postgres does not index the referencing side of a foreign key either, so this
    gap is both common and expensive.
    """
    proposals: list[Proposal] = []
    for relation, items in sorted(_by_relation(usage).items()):
        table_facts = facts.get(relation)
        rows = table_facts.row_estimate if table_facts else None
        if rows is not None and rows < min_rows:
            continue
        ndv = table_facts.ndv if table_facts else {}
        joins = sorted(
            (i for i in items if i.role is ColumnRole.JOIN),
            key=lambda i: (-i.cost_ms, i.column),
        )
        table_indexes = existing.get(relation, ())
        for item in joins:
            if item.cost_share < min_cost_share:
                continue
            if _covered((item.column,), table_indexes) is not None:
                continue
            # Same guards as ADV001, same reasons: a partial index leading with this column
            # does not serve an unfiltered lookup, and an expression index's `columns` tuple
            # understates it, so `_covered` already treats neither as coverage. But silence
            # on both would let this rule say "No existing index leads with it" at HIGH next
            # to an index that, in plain English, does lead with it — just partially, or
            # under an expression. Naming them is what ADV001 does for the same reason.
            partial_skipped = tuple(
                index.name
                for index in table_indexes
                if index.is_partial and _is_prefix((item.column,), index.columns)
            )
            expression_indexes = tuple(
                index.name
                for index in table_indexes
                if index.has_expressions
                and mentions_identifier(item.column, index.definition or "")
            )
            column_ndv = ndv.get(item.column)
            if rows is None or not have_index_data:
                confidence = Confidence.LOW
            elif column_ndv is None:
                confidence = Confidence.MEDIUM
            elif column_ndv >= SELECTIVE_NDV:
                confidence = Confidence.HIGH
            else:
                confidence = Confidence.LOW

            rationale = (
                "This column carries the table's hottest join predicate. A join key is "
                "probed once per outer row, so without an index leading with it every probe "
                "is a scan."
            )
            if have_index_data:
                rationale += " No existing index leads with it."
            else:
                rationale += (
                    " The existing-index list could not be read, so whether an index "
                    "already leads with it is unknown — check before applying."
                )
            if rows is None:
                rationale += _UNKNOWN_ROWS_NOTE
            if column_ndv is not None and column_ndv < SELECTIVE_NDV:
                rationale += (
                    f" Only about {column_ndv:.0f} distinct values, so the index may not be "
                    "selective enough to be worth its write cost."
                )
            # Same disclosure ADV001 makes at the same rung, in the same words: MEDIUM here
            # means the selectivity check could not run, not that it ran and was middling.
            elif column_ndv is None and rows is not None and have_index_data:
                rationale += (
                    f" No distinct-value statistics for {item.column}, so how selective this "
                    "index would be could not be checked — run ANALYZE on the table for a "
                    "firmer answer."
                )
            if partial_skipped:
                rationale += (
                    f" A partial index ({', '.join(partial_skipped)}) leads with these "
                    "columns but carries a WHERE predicate, so it does not serve an "
                    "unfiltered lookup — it is not treated as covering this proposal."
                )
            if expression_indexes:
                rationale += (
                    f" An expression index ({', '.join(expression_indexes)}) mentions "
                    f"{item.column}; sqlquality cannot tell whether it already serves this "
                    "lookup, so confirm before applying."
                )

            proposals.append(
                Proposal(
                    code="ADV007",
                    title=f"Add index on join key {relation}({item.column})",
                    rationale=rationale,
                    evidence={
                        "schema": relation.schema,
                        "table": relation.table,
                        "columns": (item.column,),
                        "roles": (item.role.value,),
                        "cost_share": item.cost_share,
                        "calls": item.calls,
                        "fingerprints": item.fingerprints,
                        #: The query groups behind `fingerprints`, digested and sorted —
                        #: see ADV001's identical field for why.
                        "fingerprint_digests": fingerprint_digests(item.fingerprint_ids),
                        "row_estimate": rows,
                        "leading_ndv": column_ndv,
                        "partial_indexes_skipped": partial_skipped,
                        "expression_indexes": expression_indexes,
                    },
                    confidence=confidence,
                    ddl=(
                        f"CREATE INDEX ON {_qualified(relation.schema, relation.table)} "
                        f"({_quote_ident(item.column)});"
                    ),
                )
            )
    return proposals


def propose_grouping_indexes(
    usage: Sequence[ColumnUsage],
    facts: Mapping[Relation, TableFacts],
    existing: Mapping[Relation, Sequence[PgIndex]],
    *,
    min_cost_share: float,
    min_rows: int = MIN_ROWS_FOR_INDEX,
    max_arity: int = MAX_INDEX_ARITY,
    have_index_data: bool = True,
) -> list[Proposal]:
    """ADV008 — an index that can feed a hot GROUP BY already sorted.

    Confidence is capped at MEDIUM and there is deliberately no HIGH branch. Whether
    Postgres uses such an index depends on its choice between `GroupAggregate` (which wants
    sorted input, and is what the index provides) and `HashAggregate` (which does not) — a
    decision driven by `work_mem`, the number of groups and the aggregates involved, none of
    which this tool can see. Claiming HIGH would be asserting something about the planner
    rather than about the catalog. Do not add a HIGH branch here for symmetry with ADV001.

    One composite rather than several single-column indexes, unlike ADV007: `GROUP BY a, b`
    wants input ordered by `(a, b)`, which two separate indexes cannot provide. The column
    *order* is inferred from cost, not read from the query — redaction and fingerprinting do
    not preserve each column's position in the GROUP BY clause — and the rationale says so,
    because getting the order wrong makes the index serve only its leading column.

    Extension requires *joint* support — every chosen column sharing a fingerprint with every
    other chosen column, via a running intersection — not merely pairwise support with the
    seed. Checking only against the seed lets a transitive chain through: `a` grouped with `b`
    in one query and with `c` in another welds `(a, b, c)` into one composite that no query
    groups by, even though `a` alone passes both pairwise checks. That composite would still
    report cost and fingerprint evidence that reads as support, which is worse than proposing
    nothing.

    Evidence carries `co_occurring_fingerprints` (the joint overlap size) rather than a plain
    `fingerprints` count — unlike ADV007 and ADV005, which each speak for a single column, so a
    per-column count is the whole truth about them. This rule, ADV004 and ADV001 all propose an
    index justified by columns appearing *together*, where the joint overlap is the only number
    that actually supports the proposal; a per-column count sitting beside it in a report that
    renders evidence as bare `k=v` pairs would read as corroborating support that is not there.
    The split is by what the rule claims, not by which rule came first — ADV001 joined this side
    of it once it too required joint co-occurrence.
    """
    proposals: list[Proposal] = []
    for relation, items in sorted(_by_relation(usage).items()):
        table_facts = facts.get(relation)
        rows = table_facts.row_estimate if table_facts else None
        if rows is not None and rows < min_rows:
            continue
        grouping = sorted(
            (i for i in items if i.role is ColumnRole.GROUP),
            key=lambda i: (-i.cost_ms, i.column),
        )
        if not grouping:
            continue
        seed = grouping[0]
        # Extend the composite only with columns that share a fingerprint with *every*
        # column already chosen — not merely with the seed. Pairwise-with-seed is not
        # enough: given a in {fp1, fp2}, b in {fp1} and c in {fp2}, checking each candidate
        # against the seed alone welds (a, b, c) into one composite even though no query
        # groups by all three — fp1 groups by (a, b), fp2 groups by (a, c). Tracking the
        # running intersection catches this: the moment a candidate's overlap does not
        # include every fingerprint the chosen set already agrees on, its joint support for
        # the *whole* composite is a query that does not exist.
        chosen = [seed]
        shared = seed.fingerprint_ids
        for candidate in grouping[1:]:
            if len(chosen) >= max_arity:
                break
            overlap = shared & candidate.fingerprint_ids
            if overlap:
                chosen.append(candidate)
                shared = overlap

        cost_share = max(i.cost_share for i in chosen)
        if cost_share < min_cost_share:
            continue
        columns = tuple(i.column for i in chosen)
        if _covered(columns, existing.get(relation, ())) is not None:
            continue

        # Same guards as ADV001 and ADV007, same reasons: a partial index leading with
        # these columns does not serve an unfiltered GROUP BY, and an expression index's
        # `columns` tuple understates it, so `_covered` already treats neither as coverage.
        # But silence on both would let this rule say nothing next to an index that, in
        # plain English, does lead with these columns — just partially, or under an
        # expression. Naming them is what ADV001 and ADV007 do for the same gap.
        table_indexes = existing.get(relation, ())
        partial_skipped = tuple(
            index.name
            for index in table_indexes
            if index.is_partial and _is_prefix(columns, index.columns)
        )
        expression_indexes = tuple(
            index.name
            for index in table_indexes
            if index.has_expressions and mentions_identifier(columns[0], index.definition or "")
        )

        rationale = (
            "This grouping carries a hot share of workload cost. An index on these columns "
            "lets the planner read the rows already ordered and group them without a sort. "
            "The column order here is inferred from cost, not read from the query — "
            "redaction does not preserve each column's position in the GROUP BY — so check "
            "it against the actual grouping before applying, since a composite index only "
            "serves the grouping it leads with."
        )
        if not have_index_data:
            rationale += (
                " The existing-index list could not be read, so whether an index already "
                "leads with these columns is unknown — check before applying."
            )
        if rows is None:
            rationale += _UNKNOWN_ROWS_NOTE
        if partial_skipped:
            rationale += (
                f" A partial index ({', '.join(partial_skipped)}) leads with these columns "
                "but carries a WHERE predicate, so it does not serve an unfiltered lookup — "
                "it is not treated as covering this proposal."
            )
        if expression_indexes:
            rationale += (
                f" An expression index ({', '.join(expression_indexes)}) mentions "
                f"{columns[0]}; sqlquality cannot tell whether it already serves this "
                "lookup, so confirm before applying."
            )

        proposals.append(
            Proposal(
                code="ADV008",
                title=f"Add index for GROUP BY on {relation}({', '.join(columns)})",
                rationale=rationale,
                evidence={
                    "schema": relation.schema,
                    "table": relation.table,
                    "columns": columns,
                    "roles": tuple(i.role.value for i in chosen),
                    "cost_share": cost_share,
                    "calls": max(i.calls for i in chosen),
                    #: How many query groups actually group by *every* chosen column
                    #: together — the running intersection, not a per-column count. Same
                    #: name and same meaning as ADV004's and ADV001's identical field.
                    #: Deliberately no plain `fingerprints` key here (unlike ADV007, which
                    #: speaks for a single column and so has only a per-column count to give):
                    #: this proposal is justified by columns appearing *together*, so a
                    #: per-column count sitting beside the joint one in a report that
                    #: renders evidence as bare `k=v` pairs, with no per-rule text, would
                    #: read as more support than actually exists.
                    "co_occurring_fingerprints": len(shared),
                    #: The query groups behind the count above, digested and sorted — see
                    #: ADV001's identical field for why.
                    "fingerprint_digests": fingerprint_digests(shared),
                    "row_estimate": rows,
                    "partial_indexes_skipped": partial_skipped,
                    "expression_indexes": expression_indexes,
                },
                confidence=(
                    Confidence.LOW if rows is None or not have_index_data else Confidence.MEDIUM
                ),
                ddl=(
                    f"CREATE INDEX ON {_qualified(relation.schema, relation.table)} "
                    f"({', '.join(_quote_ident(c) for c in columns)});"
                ),
            )
        )
    return proposals


def propose_unused_indexes(
    existing: Mapping[Relation, Sequence[PgIndex]],
    *,
    hot_tables: frozenset[Relation],
) -> list[Proposal]:
    """ADV002 — indexes with zero recorded scans, excluding constraint-backing indexes.

    Confidence is capped at MEDIUM: idx_scan accumulates only since the last statistics
    reset, so zero scans cannot prove an index is unused across a full business cycle.

    No `fingerprint_digests` in this proposal's evidence, deliberately: its evidence is
    `idx_scan`, a catalog measurement of the index itself, not a claim about which query
    groups use (or fail to use) it — there is no workload query group to name. Omitted
    entirely rather than emitted as `[]`, matching Redshift's ADV104/ADV105: an empty list
    would read as "zero query groups back this", which is a different and false claim from
    "this rule is not workload-derived".
    """
    proposals: list[Proposal] = []
    for relation in sorted(hot_tables):
        for index in existing.get(relation, ()):
            if index.scans != 0 or index.is_unique or index.is_primary:
                continue
            proposals.append(
                Proposal(
                    code="ADV002",
                    title=f"Drop unused index {index.name} on {relation}",
                    rationale=(
                        "No recorded scans since the last statistics reset. Verify the "
                        "reset time covers a full business cycle before dropping."
                    ),
                    evidence={
                        "schema": relation.schema,
                        "table": relation.table,
                        "index": index.name,
                        "columns": index.columns,
                        "scans": index.scans,
                        "size_bytes": index.size_bytes,
                    },
                    confidence=Confidence.MEDIUM,
                    ddl=f"DROP INDEX {_qualified(relation.schema, index.name)};",
                )
            )
    return proposals


def propose_redundant_indexes(
    existing: Mapping[Relation, Sequence[PgIndex]],
    *,
    hot_tables: frozenset[Relation],
) -> list[Proposal]:
    """ADV003 — an index whose column list is a strict prefix of another's is redundant.

    HIGH when both indexes are plain: prefix redundancy is then provable from the column
    lists alone. A partial index (``WHERE shipped_at IS NULL``) or an expression index is
    skipped entirely rather than downgraded, on either side of the pair — ``PgIndex`` now
    carries ``is_partial``/``has_expressions``/``predicate``, and for a partial index
    "serves the same lookups" is simply false, since the partial index exists precisely to
    serve a subset the wider index serves differently. Emitting that at MEDIUM would still
    be advising a `DROP INDEX` with no basis: "probably wrong" is not a confidence level.

    Scoped to ``hot_tables`` — the relations the workload was actually observed using —
    exactly as ADV002 is, and not to every key in ``existing``. ``fetch_indexes`` filters
    tables by *bare* name, so with two requested schemas holding a same-named table it
    returns rows for relations the run never analysed; iterating all of ``existing`` made
    whether a schema got `DROP INDEX` hygiene depend on whether one of its tables happened to
    collide by name with a hot table in another requested schema. Prefix redundancy is
    provable from the catalog alone, so the advice was not *wrong* — but arbitrary scope for
    a rule that emits `DROP` is not a scope, and this rule should be able to say which
    workload its recommendation came from.

    No `fingerprint_digests` in this proposal's evidence, for the same reason ADV002 has
    none: prefix redundancy is proven from two `PgIndex` column lists, a catalog fact, not
    from any query group. See ADV002's docstring for why the key is omitted rather than
    emitted empty.
    """
    proposals: list[Proposal] = []
    for relation in sorted(hot_tables):
        indexes = existing.get(relation, ())
        for narrow in indexes:
            # A partial or expression index is not comparable on column lists alone: the
            # predicate or the expression is the whole point of it. Skipping the pair is
            # the honest answer, because "probably wrong" is not a confidence level.
            if narrow.is_unique or narrow.is_primary or narrow.is_partial:
                continue
            if narrow.has_expressions:
                continue
            wider = next(
                (
                    other
                    for other in indexes
                    if other.name != narrow.name
                    and not other.is_partial
                    and not other.has_expressions
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
                    title=f"Drop redundant index {narrow.name} on {relation}",
                    rationale=(
                        f"Its columns are a leading prefix of {wider.name}, which can serve "
                        "the same lookups. Both indexes are plain — neither carries a WHERE "
                        "predicate nor an indexed expression — so the column lists are the "
                        "whole comparison."
                    ),
                    evidence={
                        "schema": relation.schema,
                        "table": relation.table,
                        "index": narrow.name,
                        "columns": narrow.columns,
                        "superseded_by": wider.name,
                        "superseding_columns": wider.columns,
                        "size_bytes": narrow.size_bytes,
                    },
                    confidence=Confidence.HIGH,
                    ddl=f"DROP INDEX {_qualified(relation.schema, narrow.name)};",
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
    facts: Mapping[Relation, TableFacts],
    existing: Mapping[Relation, Sequence[PgIndex]],
    *,
    min_cost_share: float,
    min_rows: int = MIN_ROWS_FOR_INDEX,
    have_index_data: bool = True,
) -> list[Proposal]:
    """ADV004 — index the hot equality column, restricted by a hot null-check predicate.

    Only structural predicates qualify. Literal-valued partial indexes would need the
    literals retained, which default redaction deliberately discards.

    Gated on table size exactly as ADV001 is: both rules create an index, and below the
    floor a sequential scan is the right plan whichever rule proposed it. An unknown row
    count caps confidence at LOW rather than being assumed large.

    **This rule was the only index-creating rule that never consulted the existing-index list
    at all**, so it could propose an index an existing one already serves and its rationale
    said nothing about the gap — a check that silently did not run, which is the one thing
    this rule set is built to refuse. It now runs `_covered` like ADV001, ADV007 and ADV008,
    with two differences that follow from the proposal itself being *partial*:

    * **A plain index leading with the guarded column suppresses this proposal.** A partial
      index's only advantage over such an index is size — the access path is already there,
      since `WHERE leading = $1 AND guard IS NULL` can be served by a plain index on
      `(leading)` with the null check applied as a filter. Size is exactly what this tool
      cannot measure: nothing here knows what fraction of the table satisfies the guard, so
      "smaller" is an assertion, not evidence, and it would be traded against a second
      index's write cost on every insert and update. Worse, nothing downstream would catch
      the pair: ADV003's redundant-prefix check is restricted to plain indexes, so a partial
      index shadowed by a plain one is never flagged on a later run. Suppressing is therefore
      the honest call, not merely the conservative one.
    * **A *partial* index that leads with the same column is disclosed, not treated as
      coverage.** `_covered` skips partial and expression indexes deliberately (see its
      docstring), and for this rule that exclusion cuts the other way than it does for
      ADV001: an existing partial index on the same column may be *precisely* this proposal
      already applied. Nothing here compares predicates — the existing index's `WHERE` clause
      is not parsed, and this proposal's guard is reconstructed from redacted usage — so
      whether it is the same index is genuinely unknown and is stated as unknown, naming the
      index so an operator can settle it in one glance.

    `have_index_data` is False when the existing-index catalog query was denied, and is
    handled exactly as the other three rules handle it: the cost evidence is real so the
    proposal survives, but confidence is capped at LOW and the rationale says which check
    could not run. `existing` being empty cannot distinguish "no such index" from "could not
    look", which is why the flag is separate from the mapping.
    """
    proposals: list[Proposal] = []
    for relation, items in sorted(_by_relation(usage).items()):
        table_facts = facts.get(relation)
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
        table_indexes = existing.get(relation, ())
        # A plain index leading with this column already provides the access path; only the
        # size differs, and this rule cannot measure that. See the docstring.
        if _covered((leading.column,), table_indexes) is not None:
            continue
        # Not coverage, and not the same gap ADV001 discloses: an existing *partial* index
        # leading with this column may be this very proposal, already applied. Predicates are
        # not compared, so that is unknown rather than either answer.
        partial_indexes = tuple(
            index.name
            for index in table_indexes
            if index.is_partial and _is_prefix((leading.column,), index.columns)
        )
        # Same whole-identifier matching as ADV001/ADV007/ADV008, for the same reason: a
        # substring test reports an index on `lower(guid)` as "mentions id".
        expression_indexes = tuple(
            index.name
            for index in table_indexes
            if index.has_expressions and mentions_identifier(leading.column, index.definition or "")
        )
        predicate = _NULL_ROLE_PREDICATE[guard.role]
        rationale = (
            "The hot predicates always pair this lookup with the same null check, "
            "so a partial index covers them at a fraction of the size."
        )
        if not have_index_data:
            rationale += (
                " The existing-index list could not be read, so whether an index already "
                "serves this lookup is unknown — check before applying."
            )
        if rows is None:
            rationale += _UNKNOWN_ROWS_NOTE
        if partial_indexes:
            rationale += (
                f" A partial index ({', '.join(partial_indexes)}) already leads with this "
                "column; sqlquality does not compare its WHERE predicate to this proposal's, "
                "so it cannot tell whether this index already exists — check before applying."
            )
        if expression_indexes:
            rationale += (
                f" An expression index ({', '.join(expression_indexes)}) mentions "
                f"{leading.column}; sqlquality cannot tell whether it already serves this "
                "lookup, so confirm before applying."
            )
        proposals.append(
            Proposal(
                code="ADV004",
                title=(
                    f"Partial index on {relation}({leading.column}) "
                    f"WHERE {guard.column} {predicate}"
                ),
                rationale=rationale,
                evidence={
                    "schema": relation.schema,
                    "table": relation.table,
                    "columns": (leading.column,),
                    "guard_column": guard.column,
                    "guard_predicate": predicate,
                    "cost_share": cost_share,
                    "calls": max(leading.calls, guard.calls),
                    "row_estimate": rows,
                    #: How many query groups filter on both columns together. This is what
                    #: makes the proposal supported rather than a guess.
                    "co_occurring_fingerprints": len(shared),
                    #: The query groups behind the count above, digested and sorted — see
                    #: ADV001's identical field for why.
                    "fingerprint_digests": fingerprint_digests(shared),
                    #: Named `partial_indexes_not_compared`, not ADV001's
                    #: `partial_indexes_skipped`: there the partial index is known not to
                    #: cover an unfiltered lookup, here it may be this exact proposal already
                    #: applied and the difference is that nobody compared the predicates. A
                    #: shared key name would have made two different facts indistinguishable
                    #: in `--json`, which renders evidence as bare `k=v` pairs.
                    "partial_indexes_not_compared": partial_indexes,
                    "expression_indexes": expression_indexes,
                },
                confidence=(
                    Confidence.LOW if rows is None or not have_index_data else Confidence.MEDIUM
                ),
                ddl=(
                    f"CREATE INDEX ON {_qualified(relation.schema, relation.table)} "
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
                title=f"Non-sargable predicate on {item.relation}.{item.column}",
                rationale=(
                    "The column is wrapped in a cast or function inside a predicate, so a "
                    "plain B-tree index cannot be used. Rewrite the predicate to leave the "
                    "column bare, or add a matching expression index."
                ),
                evidence={
                    "schema": item.relation.schema,
                    "table": item.relation.table,
                    "column": item.column,
                    "cost_share": item.cost_share,
                    "calls": item.calls,
                    "fingerprints": item.fingerprints,
                    #: The query groups behind `fingerprints`, digested and sorted — see
                    #: ADV001's identical field for why.
                    "fingerprint_digests": fingerprint_digests(item.fingerprint_ids),
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
                    "fingerprint": fingerprint_id(stat.fingerprint),
                    "sql": stat.sql,
                    "cost_share": share,
                    "calls": stat.calls,
                    #: Additional to the singular `fingerprint` above, not a replacement for
                    #: it: this proposal is backed by exactly one query group — itself — so
                    #: the list holds that one digest. Same key, same meaning, as every other
                    #: rule's `fingerprint_digests`, which is what lets `report.py` derive
                    #: `query_groups` uniformly across every proposal code.
                    "fingerprint_digests": (fingerprint_id(stat.fingerprint),),
                },
                confidence=Confidence.MEDIUM,
                ddl=None,
            )
        )
    return proposals


def _wide_relations_touched(
    sql: str, dialect: str, wide: Mapping[Relation, TableFacts]
) -> tuple[Relation, ...]:
    """Which of the *wide* relations this one statement provably references.

    Bare-name text matching was the original approach — a `mentions_identifier` test of
    `relation.table` against the statement text, since deleted along with its `mentions_table`
    alias — and it is exactly right for an *unqualified* reference: real SQL says `from
    orders`, not `from public.orders`. Its failure mode is the same one Task 2 already fixed
    once on this branch for `star_tables`: text matching cannot see a schema qualifier, so
    `select * from public.orders` matched *both* `public.orders` and `staging.orders`
    whenever both were wide — an evidence block naming a table the statement never
    referenced.

    So this parses the statement (with the adapter's own dialect — the same one `aggregate`
    used to build `facts` in the first place) and resolves each `exp.Table` node against
    `wide` specifically, not the full introspected schema: a schema-qualified reference
    (`table.db` set) is matched only against that same relation; a bare reference is
    attributed only when exactly one wide relation shares its name. An ambiguous bare name
    — two wide relations sharing it — is dropped rather than guessed at, the same
    cannot-prove-it policy `resolve_relation`/`star_tables` already apply, just restricted
    here to the (usually much smaller) wide set, since that is all this rule can ever report
    on.

    No parse-failure fallback. The premise is *not* "the same text parsed twice": `sql` is
    `stat.sql`, which is sqlglot's own re-serialisation of the **redacted** tree, not the
    `row.sql` that `ingest()` parsed — different text, so "it already parsed once" would not
    be an argument. The real premise is that sqlglot re-parses its own generated SQL under
    the dialect that generated it, which was measured rather than assumed: 0 reparse failures
    across the 14-statement adversarial corpus, redaction included. `ingest()` also
    guarantees the *original* row parsed (an unparseable row is counted as
    `skipped_unparseable` and never becomes a `QueryStat`), so the input to redaction was
    always a real tree. A bare-name text-match fallback used to sit here for "just in case",
    but it reintroduced the exact over-attribution this function exists to fix —
    `select * from public.orders` matching both `public.orders` and `staging.orders` — for a
    branch nothing can reach. Same reasoning as `_dedupe_by_ddl`'s deleted tie-break: a
    fallback that cannot be reached is worse than none, since it reads as evidence the case
    is handled when the real handling is "it cannot happen". If a round-trip ever did fail,
    it raises here instead of silently mis-attributing evidence — the direction to fail in.
    """
    by_bare_name: dict[str, list[Relation]] = {}
    for relation in wide:
        by_bare_name.setdefault(relation.table, []).append(relation)
    tree = parse(sql, dialect)
    touched: set[Relation] = set()
    for table in tree.find_all(exp.Table):
        if table.db:
            candidate = Relation(schema=table.db, table=table.name)
            if candidate in wide:
                touched.add(candidate)
            continue
        owners = by_bare_name.get(table.name, ())
        if len(owners) == 1:
            touched.add(owners[0])
    return tuple(sorted(touched))


def propose_select_star(
    workload: Workload,
    facts: Mapping[Relation, TableFacts],
    *,
    min_cost_share: float,
    min_columns: int = WIDE_TABLE_COLUMNS,
    dialect: str = "postgres",
) -> list[Proposal]:
    """ADV006 — hot query groups projecting a star from a wide table."""
    wide = {relation: fact for relation, fact in facts.items() if len(fact.columns) >= min_columns}
    if not wide:
        return []
    total = workload.total_cost_ms
    proposals: list[Proposal] = []
    for stat in workload.stats:
        if FLAG_SELECT_STAR not in stat.flags:
            continue
        touched = sorted(_wide_relations_touched(stat.sql, dialect, wide))
        if not touched:
            continue
        share = (stat.total_time_ms / total) if total else 0.0
        if share < min_cost_share:
            continue
        proposals.append(
            Proposal(
                code="ADV006",
                title=(
                    f"Hot SELECT * over wide table(s): "
                    f"{', '.join(str(relation) for relation in touched)}"
                ),
                rationale=(
                    "Projecting every column of a wide table moves data no consumer asked "
                    "for. List the columns the query actually needs."
                ),
                evidence={
                    "tables": tuple(str(relation) for relation in touched),
                    "column_counts": {
                        str(relation): len(facts[relation].columns) for relation in touched
                    },
                    "cost_share": share,
                    "calls": stat.calls,
                    "fingerprint": fingerprint_id(stat.fingerprint),
                    # The identity above is a digest, so the readable text has to be
                    # carried explicitly — ADV005 already does this. Redacted by
                    # default, like every other query text in the report.
                    "sql": stat.sql,
                    # Additional to the singular `fingerprint` above, not a replacement —
                    # see ADV005's identical comment.
                    "fingerprint_digests": (fingerprint_id(stat.fingerprint),),
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


def _has_line_break(text: str) -> bool:
    """True when `text` would occupy more than one physical line in the rendered script.

    Defined as "`str.splitlines` disagrees that this is exactly one line" rather than as a
    membership test against a list of characters, and that is the whole point of the function.
    Both `render_ddl` implementations used to guard with `"\\n" in ddl or "\\r" in ddl`, while
    every place that actually splits the text — `_comment_lines`, `_is_fully_commented`, and
    the tests asserting no bare line is ever emitted — uses `splitlines()`, which also splits
    on `\\v`, `\\f`, `\\x1c`, `\\x1d`, `\\x1e`, `\\x85`, `\\u2028` and `\\u2029`. An identifier
    carrying any of those eight produced a second physical line in the file that the guard
    never examined, so `render_ddl` skipped the NOT-RENDERED fallback and emitted something a
    reader sees as a bare, statement-shaped line — exactly the invariant this script format
    states unconditionally. Postgres permits every one of them inside a quoted identifier.

    Deriving the answer from `splitlines` rather than restating its character set keeps the
    guard and the splitting in lockstep by construction: a future CPython that recognises one
    more line boundary cannot reopen the hole, and there is no second list to forget to update.

    Empty text is not a line break — it has no lines at all — and is answered False rather
    than by the raw `splitlines() != [text]` comparison, which would say True for `""`.
    """
    return bool(text) and text.splitlines() != [text]


def _is_fully_commented(ddl: str) -> bool:
    """True when every physical line of `ddl` already begins with `--`.

    `render_ddl`'s line-break guard exists to catch an *identifier* whose embedded newline
    would otherwise leave part of a raw statement looking like a bare, executable line. A
    `ddl` value that is already a `--`-commented disclosure on every line — for instance, a
    config-block proposal something upstream of this adapter generated instead of raw
    DDL — is categorically not that hazard: it is already inert on every line, so it can be
    emitted verbatim (with the usual code/confidence header) rather than routed through the
    NOT-RENDERED fallback, which would double-comment every line and print a reason ("an
    identifier contains a line break") that is simply false for this kind of proposal. This
    check is about the *shape* of the text alone, so it names nothing about dbt or any
    other specific caller — any adapter-agnostic multi-line, pre-commented `ddl` gets the
    same treatment.
    """
    lines = ddl.splitlines()
    return bool(lines) and all(line.startswith("--") for line in lines)


class PostgresWorkloadAdapter(WorkloadAdapter):
    engine = "postgres"

    SQL: dict[str, str] = {
        # `s.rows` is selected but currently discarded. It is kept deliberately: rows per
        # call is the natural selectivity signal for a future confidence refinement
        # ("returns 3 rows from 8M — an excellent index candidate"), and fetching it costs
        # nothing. Task 8 unpacks it as `_rows`.
        # `total_exec_time` requires PostgreSQL 13+; it was `total_time` on 12 and older,
        # both long past end-of-life. The privilege hint states the floor.
        #
        # Deliberately NOT filtered on `s.toplevel`, and the reason is a *cost*, not an
        # impossibility. Under `pg_stat_statements.track = all` a `COPY (SELECT ...) TO ...`
        # produces two rows for one execution — the verbatim top-level utility statement and
        # its normalised nested query — which `unwrap`/redaction give different fingerprints,
        # so the same execution is counted as two query groups at roughly twice its true
        # cost (documented in the README's "Prerequisites and limits").
        #
        # A blanket `AND s.toplevel` is not the answer: `toplevel = false` is the *only* way
        # Postgres ever exposes the SQL inside a PL/pgSQL function body, and verified live
        # the blanket filter made a genuinely hot, function-wrapped query (3x the cost of the
        # next candidate) vanish from evidence entirely while the surrounding
        # `SELECT my_function()` call sites stayed counted as zero-column-usage cost with no
        # signal that anything was dropped — a confidently wrong proposal, worse than an
        # inflated cost_share.
        #
        # A *narrow* predicate, however, does exist and does work. Measured on PostgreSQL 16
        # under `track = all`, the two nested forms are textually distinguishable: a COPY's
        # nested row KEEPS its wrapper (`COPY (SELECT ... $1) TO STDOUT`) while a PL/pgSQL
        # body is recorded bare (`SELECT count(*) FROM ... WHERE status = $1`), so
        # `NOT (s.toplevel = false AND s.query ~* '^\\s*COPY\\s*\\(')` removed exactly the
        # duplicate (4 rows -> 3) and left the function body untouched. It is declined for a
        # stated price rather than because nothing could work: *naming* `s.toplevel` at all
        # requires PostgreSQL 14 (the column does not exist on 13), and the documented floor
        # is 13+, so a PG13 user would lose the entire workload capability — one missing
        # column costing the whole run — to remove a 2x over-count of one statement form
        # under a non-default setting. That trade is why the filter is absent; if the floor
        # ever rises to 14, this is the predicate to add.
        #
        # Not fixable by any predicate, and documented alongside the COPY case: under
        # `track = all` every PL/pgSQL call is counted twice — measured, `SELECT lc.hot()` at
        # 68.21 ms plus its body at 67.67 ms for one execution — which roughly halves every
        # `cost_share` in the run. The call carries the cost while the body carries the
        # predicates, so excluding either row loses something real.
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
            SELECT c.table_schema, c.table_name, c.column_name, c.data_type
            FROM information_schema.columns c
            WHERE c.table_schema = ANY(%s)
        """,
        CAP_TABLE_FACTS: """
            SELECT n.nspname, c.relname, c.reltuples::bigint, pg_total_relation_size(c.oid)
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE c.relkind = 'r' AND n.nspname = ANY(%s) AND c.relname = ANY(%s)
        """,
        CAP_NDV: """
            SELECT s.schemaname, s.tablename, s.attname, s.n_distinct
            FROM pg_stats s
            WHERE s.schemaname = ANY(%s) AND s.tablename = ANY(%s)
        """,
        # LEFT JOIN, not JOIN: Postgres stores 0 in indkey for an expression column and no
        # pg_attribute row has attnum 0, so an inner join silently discarded every expression
        # index's columns — they arrived with an empty tuple. The NULL attname a LEFT JOIN
        # yields is what tells us the position was an expression.
        #
        # indpred / indexprs are selected as booleans plus the rendered predicate, because a
        # partial index does not serve an unfiltered lookup and an expression index does not
        # serve its bare column — both of which the coverage and redundancy rules previously
        # had to guess at.
        CAP_INDEXES: """
            SELECT n.nspname, t.relname, i.relname, a.attname, k.ordinality,
                   ix.indisunique, ix.indisprimary,
                   COALESCE(psui.idx_scan, 0), pg_relation_size(i.oid),
                   ix.indpred IS NOT NULL,
                   pg_get_expr(ix.indpred, ix.indrelid),
                   ix.indexprs IS NOT NULL,
                   pg_get_indexdef(ix.indexrelid)
            FROM pg_index ix
            JOIN pg_class i ON i.oid = ix.indexrelid
            JOIN pg_class t ON t.oid = ix.indrelid
            JOIN pg_namespace n ON n.oid = t.relnamespace
            JOIN LATERAL unnest(ix.indkey) WITH ORDINALITY AS k(attnum, ordinality) ON TRUE
            LEFT JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = k.attnum
            LEFT JOIN pg_stat_user_indexes psui ON psui.indexrelid = i.oid
            WHERE n.nspname = ANY(%s) AND t.relname = ANY(%s)
            ORDER BY n.nspname, t.relname, i.relname, k.ordinality
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
        #: Recorded by `fetch_workload` for `window_facts()` to report without re-querying —
        #: `window_facts()` must not issue SQL, since it is read for the payload after the
        #: whole analysis (including `--dry-run`) has already run. `None` until
        #: `fetch_workload` runs, or if `pg_stat_database.stats_reset` came back SQL NULL or
        #: denied — both indistinguishable from "unknown" here.
        self._stats_reset_at: str | None = None
        #: The `limit` passed to the most recent `fetch_workload`, for the same reason.
        self._window_limit: int | None = None
        #: Relations that appeared in the last `fetch_table_facts` (CAP_TABLE_FACTS) result
        #: — an ordinary table, since that statement filters `WHERE c.relkind = 'r'`. A
        #: relation absent here is not an ordinary table (a view, a foreign table, or a
        #: partitioned parent) — the same signal `physical_state` reports as
        #: `is_ordinary_table`. Populated as a side effect of `fetch_table_facts` so
        #: `physical_state` can read it back without a second catalog round trip.
        self._ordinary_tables: set[Relation] = set()
        #: Existing indexes per relation, from the last `fetch_indexes` (CAP_INDEXES) call
        #: — kept for the identical reason: `physical_state` reads this instead of
        #: re-querying `pg_index` for a payload field.
        self._indexes_cache: dict[Relation, tuple[PgIndex, ...]] = {}
        #: Every relation ever *asked about* in a `fetch_table_facts` call — the full
        #: `relations` argument, not merely the ones `CAP_TABLE_FACTS` returned a row for.
        #: `_ordinary_tables` alone cannot tell "asked, and it is not an ordinary table"
        #: apart from "never asked at all" — both leave the relation absent from it — and
        #: conflating the two is exactly how `physical_state` used to report a relation
        #: outside `aggregation.tables` (e.g. a dbt-enriched ADV303 proposal) as a hard
        #: `is_ordinary_table: False` when the truth is "this run never looked."
        self._table_facts_requested: set[Relation] = set()
        #: The same tracking for `fetch_indexes`, for the identical reason applied to
        #: `_indexes_cache`: that cache only ever receives relations `CAP_INDEXES`
        #: returned at least one row for, so "genuinely has no indexes" and "never asked"
        #: are otherwise indistinguishable.
        self._indexes_requested: set[Relation] = set()

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
        psycopg = import_psycopg("Postgres", "postgres")

        # Silence is the failure mode being fixed here: a dropped `sslmode` downgrades the
        # connection with no signal at all. Key names only — see `dropped_libpq_fields`.
        dropped = dropped_libpq_fields(params.fields, LIBPQ_FIELD_MAP, LIBPQ_PASSTHROUGH_FIELDS)
        if dropped:
            print(
                f"warning: ignoring connection setting(s) not supported by the Postgres "
                f"adapter: {', '.join(dropped)}. Pass --dsn if you need them.",
                file=sys.stderr,
            )

        # Everything we know to be secret, so a driver exception can be proven clean rather
        # than trusted.
        secrets = secrets_for(params)

        # `read_only_required=True`: on Postgres `SET default_transaction_read_only` is
        # expected to succeed unconditionally, so its failure is indistinguishable from
        # any other setup failure and aborts the connection exactly like one. See
        # `open_session` and `RedshiftWorkloadAdapter.connect` for the engine where a
        # refusal is instead a recorded degradation.
        query, _degradation = open_session(
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
            read_only_required=True,
        )
        self._query = query

    def fetch_workload(self, since: timedelta | None, limit: int) -> WorkloadFetch:
        rows = self._run(CAP_WORKLOAD, (limit,))
        reset = self._run(CAP_STATS_RESET, ())
        # Two different unknowns, one fallback. The statement can be denied (no row at all),
        # or it can succeed and report SQL NULL — which is the *default* state of
        # `pg_stat_database.stats_reset` for any database whose statistics have never been
        # reset. The row is then `(None,)`: non-empty, hence truthy, so testing the row's
        # emptiness printed "since stats reset at None". The value's nullness is what matters.
        reset_value: object | None = None
        if reset and reset[0] and reset[0][0] is not None:
            reset_value = reset[0][0]
        reset_at: object = "an unknown time" if reset_value is None else reset_value
        # pg_stat_statements is cumulative since reset and carries no per-statement
        # timestamps before PG 17, so --since cannot be honored. Say so rather than
        # implying the requested window was applied.
        window = f"since stats reset at {reset_at}"
        if since is not None:
            window += " (--since is not supported by pg_stat_statements)"
        # Recorded for `window_facts()`, which must not issue SQL of its own — see that
        # method's docstring. `since` is deliberately not recorded here at all: Postgres
        # never honors it, and `window_facts()` reports `None` unconditionally rather than
        # echoing back what the caller asked for.
        self._stats_reset_at = str(reset_value) if reset_value is not None else None
        self._window_limit = limit
        return WorkloadFetch(
            rows=tuple(
                RawQueryRow(sql=str(sql), calls=_as_int(calls), total_time_ms=_as_float(total_ms))
                for sql, calls, total_ms, _rows in rows
            ),
            window_description=window,
        )

    def window_facts(self) -> dict[str, object]:
        """What `fetch_workload` already read, recorded rather than re-queried.

        `since` is always `None`: `pg_stat_statements` carries no per-statement
        timestamps, so `--since` is never actually applied here, whatever the caller
        passed. Echoing it back would make a baseline/verification pair that is really
        *nested* — the follow-up's cumulative counters contain the baseline's — look
        *comparable*, which is worse than reporting nothing.
        """
        return {
            "stats_reset_at": self._stats_reset_at,
            "since": None,
            "limit": self._window_limit,
        }

    def _schema_rows(self, schemas: tuple[str, ...]) -> list[tuple[object, ...]]:
        """CAP_SCHEMA rows, fetched at most once per schema tuple. See `_schema_cache`."""
        if schemas not in self._schema_cache:
            self._schema_cache[schemas] = self._run(CAP_SCHEMA, (list(schemas),))
        return self._schema_cache[schemas]

    def fetch_schema(self, schemas: tuple[str, ...]) -> dict:
        """Nested schema mapping for sqlglot qualify(): {schema: {table: {column: type}}}.

        Nested rather than flat because `qualify()` needs to be able to *tell* two
        same-named tables apart — a flat map resolves a column against the union of both
        column sets, which is how a filter on a column that exists in only one of them was
        silently accepted.
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
        # The `= ANY(%s)` table parameter stays a list of bare names: Postgres filters on
        # `relname`/`tablename`, and narrowing per-schema would need one statement per
        # schema. `n.nspname = ANY(%s)` still restricts rows to `schemas`, so a table in a
        # schema we were not asked to introspect at all never comes back. What can
        # over-fetch is a same-named table in a *different requested* schema that is not
        # itself in `relations` — `schemas=("sales", "staging")` with `relations` naming
        # only `sales.orders` still returns `staging.orders`, because the bare-name filter
        # cannot distinguish the two. That row's relation key then simply has no consumer,
        # since only the relations in `relations` are ever assembled into the result below.
        # Recorded before the statement even runs: `physical_state` must be able to tell
        # "asked about, and CAP_TABLE_FACTS said no" apart from "never asked" — see
        # `_table_facts_requested`'s docstring — and that distinction is about what this
        # method was *called with*, not about what came back.
        self._table_facts_requested.update(relations)
        wanted = sorted({relation.table for relation in relations})
        sizes = {
            Relation(schema=str(schema_name), table=str(name)): (
                _row_estimate(rows),
                _as_int(size) if size is not None else None,
            )
            for schema_name, name, rows, size in self._run(CAP_TABLE_FACTS, (list(schemas), wanted))
        }
        # `sizes`'s keys are exactly the relations CAP_TABLE_FACTS returned a row for —
        # i.e. an ordinary table, since that statement filters `relkind = 'r'`. Recorded
        # for `physical_state` to read back rather than re-querying; see `_ordinary_tables`.
        self._ordinary_tables.update(sizes)
        columns: dict[Relation, list[str]] = {}
        for schema_name, table, column, _type in self._schema_rows(schemas):
            relation = Relation(schema=str(schema_name), table=str(table))
            if relation in relations:
                columns.setdefault(relation, []).append(str(column))

        ndv: dict[Relation, dict[str, float]] = {}
        for schema_name, table, column, n_distinct in self._run(CAP_NDV, (list(schemas), wanted)):
            if n_distinct is None:
                continue
            value = _as_float(n_distinct)
            relation = Relation(schema=str(schema_name), table=str(table))
            if value < 0:
                # Postgres encodes "distinct as a fraction of row count" as a negative
                # value, which is meaningless without the row count. If the row-count query
                # returned nothing for this table — different statement, so different
                # privileges can hide it — omit the column so it reads as *unknown*.
                # Defaulting the row estimate to 0 would fabricate "zero distinct values"
                # and hand every proposal on this table a false LOW-confidence rating.
                row_estimate = sizes.get(relation, (None, None))[0]
                if row_estimate is None:
                    continue
                resolved = -value * row_estimate
            else:
                resolved = value
            ndv.setdefault(relation, {})[str(column)] = resolved

        facts: dict[Relation, TableFacts] = {}
        for relation in sorted(relations):
            rows, size = sizes.get(relation, (None, None))
            facts[relation] = TableFacts(
                relation=relation,
                row_estimate=rows,
                size_bytes=size,
                columns=tuple(columns.get(relation, ())),
                ndv=ndv.get(relation, {}),
            )
        return facts

    def fetch_indexes(
        self, schemas: tuple[str, ...], relations: frozenset[Relation]
    ) -> dict[Relation, tuple[PgIndex, ...]]:
        """Existing indexes per relation, columns in ordinal order."""
        # See the identical note in fetch_table_facts: the table parameter is bare names,
        # so a same-named table in a *different requested* schema not itself in
        # `relations` can come back too — `n.nspname = ANY(%s)` still excludes a schema we
        # were not asked to introspect at all.
        #
        # Unlike fetch_table_facts, this method does NOT drop those rows: `_covered` needs
        # only the relations it is asked about, but the returned mapping is also handed
        # whole to ADV002 and ADV003, which iterate it. Both are therefore scoped to
        # `aggregation.tables` by their callers rather than to `existing`'s key set — an
        # earlier version of this comment claimed the over-fetched rows had "no consumer",
        # and ADV003 was that consumer, emitting `DROP INDEX` for relations the workload
        # never touched whenever a bare name collided across two requested schemas.
        #
        # Recorded before the statement runs, same reasoning as `fetch_table_facts`'s
        # identical line: `physical_state` needs to know this relation was asked about at
        # all, distinct from whatever `_indexes_cache` ends up holding for it — see
        # `_indexes_requested`'s docstring.
        self._indexes_requested.update(relations)
        wanted = sorted({relation.table for relation in relations})
        grouped: dict[tuple[Relation, str], _IndexRows] = {}
        for row in self._run(CAP_INDEXES, (list(schemas), wanted)):
            (
                schema_name,
                table,
                index,
                column,
                ordinality,
                unique,
                primary,
                scans,
                size,
                is_partial,
                predicate,
                has_expressions,
                definition,
            ) = row
            relation = Relation(schema=str(schema_name), table=str(table))
            entry = grouped.setdefault(
                (relation, str(index)),
                _IndexRows(
                    is_unique=bool(unique),
                    is_primary=bool(primary),
                    scans=_as_int(scans),
                    size_bytes=_as_int(size) if size is not None else 0,
                    is_partial=bool(is_partial),
                    predicate=str(predicate) if predicate is not None else None,
                    has_expressions=bool(has_expressions),
                    definition=str(definition) if definition is not None else None,
                ),
            )
            # A NULL attname is an expression position: it has no column name to record, and
            # `has_expressions` already marks the index, so skip it rather than storing "None".
            # Keyed by ordinality and sorted below rather than trusting arrival order. The
            # statement does ORDER BY k.ordinality, but composite-index column order decides
            # whether a proposal is right, and a fixture test that pre-sorts its canned rows
            # cannot notice the difference. Cheap defence in depth.
            if column is not None:
                entry.columns.append((_as_int(ordinality), str(column)))

        result: dict[Relation, list[PgIndex]] = {}
        for (relation, index), entry in grouped.items():
            result.setdefault(relation, []).append(
                PgIndex(
                    name=index,
                    columns=tuple(column for _ordinality, column in sorted(entry.columns)),
                    is_unique=entry.is_unique,
                    is_primary=entry.is_primary,
                    scans=entry.scans,
                    size_bytes=entry.size_bytes,
                    is_partial=entry.is_partial,
                    predicate=entry.predicate,
                    has_expressions=entry.has_expressions,
                    definition=entry.definition,
                )
            )
        built = {relation: tuple(indexes) for relation, indexes in result.items()}
        # Recorded for `physical_state` to read back rather than re-querying `pg_index`;
        # see `_indexes_cache`.
        self._indexes_cache.update(built)
        return built

    def physical_state(self, relations: frozenset[Relation]) -> dict[str, dict[str, object]]:
        """See `WorkloadAdapter.physical_state`. Reads `_ordinary_tables`,
        `_indexes_cache`, `_table_facts_requested` and `_indexes_requested` — all
        populated as a side effect of `fetch_table_facts` and `fetch_indexes` during
        `propose()` — no statement is issued here.

        `is_ordinary_table` is `relation in self._ordinary_tables` **only when this run
        actually asked CAP_TABLE_FACTS about `relation` and that capability was not
        denied** — `relation in self._table_facts_requested and CAP_TABLE_FACTS not in
        self.degraded`. Under that condition, `False` means the relation is a view, a
        foreign table, or a partitioned parent — exactly the ADV301 "the relation became
        a table" signal `verify` needs, with no new catalog query. Otherwise the field is
        `None`: a relation this run never fetched facts for at all (e.g. a dbt-enriched
        ADV303 proposal for a relation outside `aggregation.tables`, which
        `fetch_table_facts` is simply never called with) and a relation whose
        `CAP_TABLE_FACTS` read was denied both leave `_ordinary_tables` looking identical
        — empty for that relation — so only tracking "was it asked about, and did the
        capability survive" tells them apart from a genuine "asked, and it is not an
        ordinary table." Reporting `False` for either would read to `verify` as a
        measurement, not as "this run could not tell you," and a later run that *does*
        observe the relation would then look like the relation was just created.

        `indexes` follows the identical discipline against `_indexes_requested` and
        `CAP_INDEXES`: `None` unless this run asked about `relation` and the read
        succeeded, in which case it is the real (possibly empty) list — `[]` is a
        measurement ("fetched, and it genuinely has none"), never a stand-in for "did not
        look." Each index's `columns` is its `PgIndex.columns` tuple exactly as read
        elsewhere in this module (`_covered`, ADV002, ADV003) — for an expression index
        this understates it, since an expression position contributes no column name (see
        `PgIndex.has_expressions`'s docstring), and is recorded as-is here rather than
        inventing a second, unused-elsewhere convention for it.
        """
        facts_degraded = any(cap == CAP_TABLE_FACTS for cap, _ in self.degraded)
        indexes_degraded = any(cap == CAP_INDEXES for cap, _ in self.degraded)
        state: dict[str, dict[str, object]] = {}
        for relation in sorted(relations):
            have_facts = relation in self._table_facts_requested and not facts_degraded
            have_indexes = relation in self._indexes_requested and not indexes_degraded
            indexes: list[dict[str, object]] | None = None
            if have_indexes:
                indexes = [
                    {
                        "name": index.name,
                        "columns": list(index.columns),
                        "is_partial": index.is_partial,
                        "is_unique": index.is_unique,
                    }
                    for index in self._indexes_cache.get(relation, ())
                ]
            state[str(relation)] = {
                "is_ordinary_table": (relation in self._ordinary_tables) if have_facts else None,
                "indexes": indexes,
            }
        return state

    #: Which rule's rationale to keep when two rules propose byte-identical DDL at equal
    #: confidence. Lower wins. The order is by how directly the evidence supports *this*
    #: index: a filter predicate (ADV001) is the most direct reason to build a B-tree, a
    #: join key (ADV007) next, a partial index (ADV004) next since its own `WHERE` clause is
    #: already a stronger claim than a plain composite's, and a grouping (ADV008) last, since
    #: whether the planner uses an index for grouping depends on choices this tool cannot
    #: see. The DROP rules are ranked below them so a CREATE never loses to a DROP that
    #: happens to render the same text — which it cannot today, but this map is the place
    #: that would have to change. `.get(code, len(...))` at every call site rather than
    #: `[code]`, so a code missing from this map sorts last instead of raising `KeyError`
    #: after the whole analysis has run.
    _CODE_PREFERENCE = {
        "ADV001": 0,
        "ADV007": 1,
        "ADV004": 2,
        "ADV008": 3,
        "ADV003": 4,
        "ADV002": 5,
    }

    @classmethod
    def _attribution(cls, discarded: Proposal, *, same_index: bool) -> str:
        """How the folded text introduces a discarded proposal — one phrasing per collapse
        kind, because the two collapses did different things to it.

        `_dedupe_by_ddl` collapses byte-identical DDL, so "reached the same index" is true by
        construction. `_collapse_index_prefixes` collapses a *narrower* proposal into a wider
        one, where it is never true: the operator was told "ADV007 reached the same index at
        high confidence" under `Add index on sales.orders(customer_id, tenant_id,
        created_at)` when ADV007 proposed `(customer_id)` — an endorsement of a three-column
        index that no rule ever made, in the paragraph someone reads before running DDL. The
        second symptom is the same sentence: an ADV008 survivor carried ADV001's "Equality
        columns come first so the range column can be scanned last" with no equality columns
        anywhere in it. Naming the narrower column list gives the borrowed sentences the
        subject they are actually about.
        """
        if same_index:
            return f"{discarded.code} reached the same index"
        key = cls._index_creation_columns(discarded)
        # `None` is unreachable from `_collapse_index_prefixes`, which only ever absorbs
        # proposals this same function accepted — but the fallback keeps the sentence
        # grammatical rather than raising in a report renderer if a future caller differs.
        narrower = f"({', '.join(key[1])})" if key else "a narrower index"
        return (
            f"{discarded.code} proposed the narrower {narrower} on the same table, which this "
            f"index's leading columns already serve, and said of it"
        )

    @classmethod
    def _fold_discarded(
        cls, survivor: Proposal, discarded: Sequence[Proposal], *, same_index: bool
    ) -> Proposal:
        """Attach every discarded proposal's distinguishing rationale, attributed, to the
        survivor's.

        ``same_index`` distinguishes the two callers — see `_attribution`. It is required
        rather than defaulted: a new collapse rule that forgets to say which kind it is would
        otherwise silently claim an endorsement it did not get.

        `_dedupe_by_ddl` and `_collapse_index_prefixes` each throw a whole `Proposal` away
        and keep only one rationale where two, or more, existed. Diffing the texts to keep
        "only the part that's new" at the paragraph level would need to guess which
        sentence is the caveat worth keeping — fragile, and silently wrong the moment a
        rule's wording changes elsewhere. Instead, every discarded rationale is split into
        whole sentences and folded in verbatim, in order, *skipping a sentence only if that
        exact sentence already appears* — in the survivor's rationale or in an
        already-folded discarded one. ADV001, ADV007 and ADV008 share verbatim wording for
        the partial-index and expression-index disclosures, so a real three-way collision
        would otherwise repeat the same sentence up to three times in one paragraph — a
        report an operator reads to decide whether to run the DDL should not look broken
        that way. A sentence unique to one discarded proposal always survives: only exact
        repeats are dropped, never trimmed or summarised. The discarded proposal's
        confidence is stated regardless of whether any of its sentences are new, since a
        reader comparing two proposals for the same index needs to know they disagreed on
        how sure to be, not just what each said.
        """
        if not discarded:
            return survivor
        seen = set(_sentences(survivor.rationale))
        notes: list[str] = []
        for p in discarded:
            fresh = [s for s in _sentences(p.rationale) if s not in seen]
            seen.update(fresh)
            attribution = cls._attribution(p, same_index=same_index)
            if fresh:
                notes.append(
                    f" {attribution} at {p.confidence.value} confidence: {' '.join(fresh)}"
                )
            else:
                notes.append(
                    f" {attribution} at {p.confidence.value} confidence, stating nothing "
                    "beyond what is already covered above."
                )
        return replace(survivor, rationale=survivor.rationale + "".join(notes))

    @classmethod
    def _dedupe_by_ddl(cls, proposals: list[Proposal]) -> list[Proposal]:
        """Collapse proposals that would run identical DDL, keeping the strongest evidence.

        Two rules can genuinely reach the same index from different evidence — a filter
        predicate, a join key and a grouping on the same column all render the same
        `CREATE INDEX` — and an unused index that is also a prefix of a wider one is flagged
        by both ADV002 and ADV003 as the same `DROP INDEX`. They do not contradict each
        other, but a reader should not have to notice they are the same object twice.

        Confidence decides first. When it ties, `_CODE_PREFERENCE` decides, because
        something has to and list order must not: which rationale leads should not be
        "whichever `propose()` happened to append first". The proposal that does not lead
        is not thrown away, though — its rationale is folded into the survivor's via
        `_fold_discarded`, so a caveat the winner never states does not vanish with it.

        There was a window where no tie was reachable — ADV002 is hardcoded MEDIUM and
        ADV003 HIGH, the only colliding pair at the time — and the tie-break was removed as
        unreachable code, on the reasoning that a tie-break nothing can reach is worse than
        none. That reasoning stopped holding the moment two more index-creating rules
        existed: ADV001 at MEDIUM (NDV unknown) and ADV008 at MEDIUM (row count known, which
        is all ADV008 ever checks) can produce byte-identical DDL at the same confidence, and
        list order is not a rule — it is a coincidence of which call happens to come first in
        `propose()`, and deciding which rationale the operator reads is too important to
        leave to that.
        """

        def rank(proposal: Proposal) -> tuple[int, int]:
            return (
                cls._CONFIDENCE_ORDER[proposal.confidence],
                cls._CODE_PREFERENCE.get(proposal.code, len(cls._CODE_PREFERENCE)),
            )

        groups: dict[str, list[Proposal]] = {}
        for proposal in proposals:
            if proposal.ddl:
                groups.setdefault(proposal.ddl, []).append(proposal)

        merged: dict[str, Proposal] = {}
        for ddl, group in groups.items():
            if len(group) == 1:
                merged[ddl] = group[0]
                continue
            # `sorted` is stable, so a true tie in `rank` (both confidence and code
            # preference equal — only possible today if the same code proposes the same
            # DDL twice) keeps whichever proposal `propose()` happened to append first.
            # That residual is accepted rather than papered over with a further tie-break
            # key. Note it is not that such proposals are *identical* — `rationale` and
            # `title` can still differ, and `_fold_discarded` preserves the loser's
            # rationale either way, so nothing a reader needs is lost. What they no longer
            # differ in is the thing a tie-break could act on: same code, same confidence,
            # same DDL leaves no principled basis for preferring one, whereas which *code*
            # wins is a real editorial choice and `_CODE_PREFERENCE` makes it.
            ranked = sorted(group, key=rank)
            winner, *losers = ranked
            merged[ddl] = cls._fold_discarded(winner, losers, same_index=True)

        result: list[Proposal] = []
        emitted: set[str] = set()
        for proposal in proposals:
            if not proposal.ddl:
                result.append(proposal)
                continue
            if proposal.ddl in emitted:
                continue
            emitted.add(proposal.ddl)
            result.append(merged[proposal.ddl])
        return result

    @classmethod
    def _index_creation_columns(cls, proposal: Proposal) -> tuple[Relation, tuple[str, ...]] | None:
        """`(relation, columns)` for a plain `CREATE INDEX` proposal, or `None` if the
        proposal cannot participate in prefix collapsing or overlap disclosure.

        Restricted to plain composite/single-column indexes: a `WHERE` predicate (ADV004's
        partial indexes) makes the index a different object even when its column list is a
        prefix of a plain index's — the same reasoning `_covered` and
        `propose_redundant_indexes` already apply to *existing* indexes, applied here to
        proposals that do not exist as catalog rows yet. `DROP INDEX` proposals (ADV002,
        ADV003) are excluded by the `CREATE INDEX` check: a prefix relationship between
        something being created and something being dropped is not a meaningful comparison.
        """
        if (
            not proposal.ddl
            or not proposal.ddl.startswith("CREATE INDEX")
            or "WHERE" in proposal.ddl
        ):
            return None
        schema = proposal.evidence.get("schema")
        table = proposal.evidence.get("table")
        columns = proposal.evidence.get("columns")
        if not isinstance(schema, str) or not isinstance(table, str):
            return None
        if not isinstance(columns, tuple) or not columns:
            return None
        return Relation(schema=schema, table=table), columns

    @classmethod
    def _collapse_index_prefixes(cls, proposals: list[Proposal]) -> list[Proposal]:
        """Collapse a `CREATE INDEX` proposal whose columns are a strict prefix of
        another's, within the same relation.

        ADV001, ADV007 and ADV008 can each reach a plain index from different evidence, and
        nothing stopped one proposing `(customer_id, created_at)` while another proposed
        `(customer_id)` in the same report — confirmed end-to-end through `propose()` from a
        single ordinary query, both at HIGH. An operator who creates both then holds a pair
        where the narrower is a strict prefix of the wider: exactly what
        `propose_redundant_indexes` (ADV003) flags as redundant on the *next* run. Shipping
        both here would be advising a CREATE today and a DROP tomorrow for the same index.

        The wider proposal always survives — it serves every lookup the narrower one does —
        and the narrower one is folded into it via `_fold_discarded`, so its rationale is
        never silently dropped. When a narrower proposal is a prefix of more than one
        *incomparable* wider proposal (say `(a)` under both `(a, b)` and `(a, c)`, neither a
        prefix of the other), it is folded into all of them: there is no principled way to
        prefer one over the other, and folding into both costs nothing but a repeated
        sentence.

        Two proposals that cover the same column *set* in a different order are not a
        prefix pair — same length, unequal tuples, so `_is_prefix` is false in both
        directions — and are deliberately left alone here; `_disclose_column_set_overlaps`
        handles that case by disclosure instead of collapse, since neither is redundant with
        the other.

        Only within one relation: a prefix relationship across two different tables is
        meaningless. Only plain proposals participate — see `_index_creation_columns`.

        **An absorbed proposal's `evidence` — including its own `fingerprint_digests` — is
        discarded along with it, not merged into the survivor's.** `_fold_discarded` carries
        forward the absorbed proposal's distinguishing *rationale* sentences, but nothing
        merges the two proposals' `evidence` dicts, so the survivor's `fingerprint_digests`
        stays exactly what it already was — the query groups that motivated *it*, not the
        union of both. This is pre-existing behaviour (`_dedupe_by_ddl` throws evidence away
        identically), not a regression, and it produces no dangling reference: every digest
        the survivor cites still resolves to a real query group. But it does mean a
        surviving proposal's `fingerprint_digests` can be a strict subset of every query
        group that actually motivated *some* proposal now folded into it — worth knowing for
        whoever builds `sqlquality verify`'s "did this proposal's evidence hold up" check.
        """
        eligible: dict[int, tuple[Relation, tuple[str, ...]]] = {}
        for i, proposal in enumerate(proposals):
            key = cls._index_creation_columns(proposal)
            if key is not None:
                eligible[i] = key

        # `covers[i]` collects every j whose columns strictly contain i's as a leading
        # prefix — direct parents and transitive ancestors alike, since the prefix relation
        # on tuples is transitive: checking every pair once already finds them all.
        covers: dict[int, list[int]] = {}
        for i, (relation_i, columns_i) in eligible.items():
            for j, (relation_j, columns_j) in eligible.items():
                if (
                    i != j
                    and relation_i == relation_j
                    and len(columns_i) < len(columns_j)
                    and _is_prefix(columns_i, columns_j)
                ):
                    covers.setdefault(i, []).append(j)

        # Maximal: nothing wider exists for it, so it is never removed.
        maximal = {i for i in eligible if i not in covers}

        # Group each absorbed proposal under every maximal proposal it is a prefix of, then
        # sort each group by a key with no dependency on `proposals`' incoming order — the
        # collapse must not depend on which order `propose()` happened to append rules in.
        absorbed_into: dict[int, list[Proposal]] = {}
        for i, targets in covers.items():
            for j in targets:
                if j in maximal:
                    absorbed_into.setdefault(j, []).append(proposals[i])

        result = list(proposals)
        for j, absorbed in absorbed_into.items():
            ordered = sorted(absorbed, key=lambda p: (p.code, p.title, p.ddl or ""))
            result[j] = cls._fold_discarded(proposals[j], ordered, same_index=False)

        drop = set(covers)
        return [p for idx, p in enumerate(result) if idx not in drop]

    @classmethod
    def _disclose_column_set_overlaps(cls, proposals: list[Proposal]) -> list[Proposal]:
        """When two surviving `CREATE INDEX` proposals cover the same column *set* for the
        same relation in a different order, say so in both, naming the other.

        `(status, region)` and `(region, status)` are not redundant — different leading
        columns genuinely serve different probes — so `_collapse_index_prefixes` correctly
        leaves both standing, and no future ADV003 pass will ever reconcile them either:
        prefix redundancy is the only structural overlap it can prove, and neither is a
        prefix of the other. Silence here would recommend two overlapping indexes with no
        acknowledgement that they overlap, leaving the operator to notice on their own that
        creating both means indexing the same columns twice.

        With only two proposals sharing a column set this is symmetric and order cannot
        matter. With three or more (today latent: ADV001 and ADV008 each propose at most one
        composite per relation and ADV007 is single-column only, so three same-set
        proposals cannot occur yet), a given proposal names *every* other member sharing its
        set, and those names are sorted by the same canonical key
        `_collapse_index_prefixes` uses — rather than by the order pairs happened to be
        discovered in — so which proposal appended first cannot change the resulting text.
        """
        pairs: list[tuple[int, tuple[Relation, tuple[str, ...]]]] = []
        for i, proposal in enumerate(proposals):
            key = cls._index_creation_columns(proposal)
            if key is not None:
                pairs.append((i, key))

        def sort_key(proposal: Proposal) -> tuple[str, str, str]:
            return (proposal.code, proposal.title, proposal.ddl or "")

        notes: dict[int, list[tuple[tuple[str, str, str], str]]] = {}
        for a in range(len(pairs)):
            i, (relation_i, columns_i) = pairs[a]
            for b in range(a + 1, len(pairs)):
                j, (relation_j, columns_j) = pairs[b]
                if relation_i != relation_j or columns_i == columns_j:
                    continue
                if set(columns_i) != set(columns_j):
                    continue
                notes.setdefault(i, []).append(
                    (
                        sort_key(proposals[j]),
                        f"{proposals[j].code} proposes an index on the same columns in a "
                        f"different order ({', '.join(columns_j)}) for the same table. "
                        "Neither is redundant — different leading columns serve different "
                        "probes — but creating both means two overlapping indexes; confirm "
                        "the workload needs both orderings before applying both.",
                    )
                )
                notes.setdefault(j, []).append(
                    (
                        sort_key(proposals[i]),
                        f"{proposals[i].code} proposes an index on the same columns in a "
                        f"different order ({', '.join(columns_i)}) for the same table. "
                        "Neither is redundant — different leading columns serve different "
                        "probes — but creating both means two overlapping indexes; confirm "
                        "the workload needs both orderings before applying both.",
                    )
                )

        if not notes:
            return proposals
        result = list(proposals)
        for idx, entries in notes.items():
            messages = [message for _key, message in sorted(entries, key=lambda e: e[0])]
            result[idx] = replace(
                result[idx], rationale=result[idx].rationale + " " + " ".join(messages)
            )
        return result

    def propose(
        self,
        aggregation: Aggregation,
        facts: dict[Relation, TableFacts],
        workload: Workload,
        *,
        min_cost_share: float,
    ) -> list[Proposal]:
        existing = self.fetch_indexes(self.schemas, aggregation.tables)
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
                have_index_data=have_index_data,
            ),
            *propose_join_keys(
                aggregation.usage,
                facts,
                existing,
                min_cost_share=min_cost_share,
                have_index_data=have_index_data,
            ),
            *propose_grouping_indexes(
                aggregation.usage,
                facts,
                existing,
                min_cost_share=min_cost_share,
                have_index_data=have_index_data,
            ),
            *propose_partial_indexes(
                aggregation.usage,
                facts,
                existing,
                min_cost_share=min_cost_share,
                have_index_data=have_index_data,
            ),
            *propose_sargability(aggregation.usage, workload, min_cost_share=min_cost_share),
            *propose_select_star(
                workload, facts, min_cost_share=min_cost_share, dialect=self.engine
            ),
            *propose_unused_indexes(existing, hot_tables=aggregation.tables),
            *propose_redundant_indexes(existing, hot_tables=aggregation.tables),
        ]
        proposals = self._dedupe_by_ddl(proposals)
        proposals = self._collapse_index_prefixes(proposals)
        proposals = self._disclose_column_set_overlaps(proposals)
        return sorted(proposals, key=self.ranking_key)

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
            if _has_line_break(proposal.ddl) and not _is_fully_commented(proposal.ddl):
                # An identifier containing a line break cannot be emitted as a single-line
                # statement. Quoting already makes it *semantically* safe — psql parses the
                # whole thing as one quoted identifier, so nothing extra executes — but the
                # file would still show a line reading like a statement to anyone skimming,
                # which is the property this script exists to provide. Truncating the name
                # instead would emit DDL targeting an object that does not exist. So it is
                # commented out in full with the reason, rather than rendered wrong or
                # silently dropped.
                #
                # This is skipped when `_is_fully_commented` already holds: a `ddl` that is
                # every-line-`--`-commented is not an identifier smuggling a line break, it
                # is an intentional multi-line disclosure, and running it through this
                # fallback would double-comment it and print a reason that is false for it.
                body.append("-- NOT RENDERED: an identifier in this proposal contains a line")
                body.append("-- break, so it cannot be emitted as a single-line statement.")
                body.append("-- Verify the name and apply this by hand:")
                if proposal.note:
                    body.extend(_comment_lines(proposal.note))
                body.extend(_comment_lines(proposal.ddl))
                body.append("")
                continue
            share = cost_share_of(proposal.evidence)
            share_text = f", {share:.1%} of workload cost" if share is not None else ""
            body.append(f"-- {proposal.code} [{proposal.confidence.value}{share_text}]")
            body.extend(_comment_lines(proposal.title))
            # `note` before the statement, not after: this script's whole purpose is to be
            # read top-to-bottom before anything is run, and `rationale` — where every other
            # caveat lives — never reaches this file at all. A caveat printed below the
            # statement it qualifies is a caveat an operator reads after pasting it.
            if proposal.note:
                body.extend(_comment_lines(proposal.note))
            body.append(proposal.ddl)
            body.append("")
        if not body:
            body = ["-- No DDL proposals — every finding is advisory-only.", ""]
        return "\n".join(header + body)
