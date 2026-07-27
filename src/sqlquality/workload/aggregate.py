"""Roll per-query column roles up into a cost-weighted usage index."""

from __future__ import annotations

import re
from collections import defaultdict
from functools import lru_cache

from sqlglot import exp

from sqlquality.models import Aggregation, ColumnRole, ColumnUsage, Relation, Workload
from sqlquality.sqlast import SqlParseError, parse
from sqlquality.workload.extract import (
    AmbiguousRelation,
    UnqualifiableQuery,
    extract_usage,
    resolve_relation,
)
from sqlquality.workload.fingerprint import FLAG_SELECT_STAR

_Key = tuple[Relation, str, ColumnRole]


@lru_cache(maxsize=4096)
def _identifier_pattern(name: str) -> re.Pattern[str]:
    """Compiled whole-identifier matcher for one name, compiled once per name.

    Callers such as ADV006's wide-table detection and the expression-index disclosure in
    `postgres.py` test one name against many statements (or vice versa), and a schema with
    many tables was recompiling the same handful of name patterns over and over, thrashing
    `re`'s own pattern cache. Caching by name here means each identifier is compiled once
    regardless of how many times it is checked.
    """
    return re.compile(rf"\b{re.escape(name)}\b")


def mentions_identifier(name: str, text: str) -> bool:
    """True if ``name`` appears in ``text`` as a whole identifier, not merely a substring.

    A plain `name in text` test would false-positive three ways: a table `order` inside a
    query on `orders`, a table `cart` inside `shopping_cart`, and a table `orders` that only
    appears as part of a column alias like `orders_total`. `\\b` already treats `_` as a word
    character in Python's `re`, so it rejects all three without a custom boundary class,
    while still matching across the punctuation SQL puts around identifiers: parens, commas,
    dots and `::` are all non-word characters.
    """
    return _identifier_pattern(name).search(text) is not None


def mentions_table(name: str, sql: str) -> bool:
    """True if a query mentions this table. See :func:`mentions_identifier`."""
    return mentions_identifier(name, sql)


def star_tables(workload: Workload, schema: dict, dialect: str = "postgres") -> frozenset[Relation]:
    """Relations a `SELECT *` query group merely *mentions*, matched against ``schema``.

    A bare `select * from wide_t` filters nothing, so it contributes no column usage and
    the relation never appears in ``Aggregation.tables``. Introspecting only the relations
    that produced usage therefore left the star rule with no column counts to test — inert
    for precisely the workload it exists to catch. These relations are unioned in before
    catalog facts are fetched.

    Resolved by parsing each starred statement and running its actual `exp.Table` nodes
    through `resolve_relation` — not by text-matching the schema's table names against the
    raw SQL. Text matching cannot see a schema qualifier at all, and that blindness cuts
    both ways: `select * from nosuch.items` would resolve through a bare-name collision
    with an unrelated schema (a phantom `resolve_relation`'s `table.db` guard exists
    specifically to refuse), while `select * from sales.orders` naming one side of a
    same-table-name collision would be dropped even though it is not actually ambiguous.
    Resolving through the same function `extract_usage` uses makes the two agree by
    construction; a second, hand-rolled ambiguity policy here previously did not.

    A parse failure is not counted here: the same statement was already counted
    unparseable at ingest (`Workload.skipped_unparseable`), so re-counting it under a
    different name would make the two counters disagree about what "unparseable" means.

    Deliberately *not* added to ``Aggregation.tables``: that set means "relations with
    recorded column usage" and feeds the unused-index rule's notion of a hot table.

    ``dialect`` defaults to `"postgres"`, the only workload adapter registered today
    (see `sqlquality.workload.get_workload_adapter`); a caller wiring in a second engine
    must pass its dialect explicitly rather than rely on the default.
    """
    found: set[Relation] = set()
    for stat in workload.stats:
        if FLAG_SELECT_STAR not in stat.flags:
            continue
        try:
            tree = parse(stat.sql, dialect)
        except SqlParseError:
            continue
        for table in tree.find_all(exp.Table):
            relation = resolve_relation(table, schema)
            if relation is not None:
                found.add(relation)
    return frozenset(found)


def aggregate(workload: Workload, schema: dict, dialect: str) -> Aggregation:
    """Weight every (relation, column, role) by the cost of the queries that use it."""
    calls: dict[_Key, int] = defaultdict(int)
    cost: dict[_Key, float] = defaultdict(float)
    #: Which query groups contributed each usage, so downstream rules can ask whether two
    #: usages co-occur in a single query rather than merely both being hot on the table.
    contributors: dict[_Key, set[str]] = defaultdict(set)
    tables: set[Relation] = set()
    skipped_unqualifiable = 0
    skipped_ambiguous = 0

    for stat in workload.stats:
        try:
            tree = parse(stat.sql, dialect)
            triples = extract_usage(tree, dialect, schema)
        except AmbiguousRelation:
            # Counted before the broader handler below, because AmbiguousRelation *is* an
            # UnqualifiableQuery — ordering these the other way round makes the specific
            # counter unreachable and the specific remedy unreportable.
            skipped_ambiguous += 1
            continue
        except (SqlParseError, UnqualifiableQuery):
            skipped_unqualifiable += 1
            continue
        for key in triples:
            calls[key] += stat.calls
            cost[key] += stat.total_time_ms
            contributors[key].add(stat.fingerprint)
            tables.add(key[0])

    # The denominator is the whole window's cost, including stats we could not analyze.
    # That keeps the number honest — "this column is involved in 12% of everything the
    # database did" — rather than flattering it to "12% of the sliver we understood". The
    # trade-off is that poor schema coverage dilutes every share, so a --min-cost-share
    # threshold silently gets stricter as coverage drops; the report's skipped counts are
    # what make that visible. See ColumnUsage.cost_share for the full semantics.
    total = workload.total_cost_ms
    usage = tuple(
        sorted(
            (
                ColumnUsage(
                    relation=relation,
                    column=column,
                    role=role,
                    calls=calls[(relation, column, role)],
                    cost_ms=cost[(relation, column, role)],
                    cost_share=(cost[(relation, column, role)] / total) if total else 0.0,
                    fingerprint_ids=frozenset(contributors[(relation, column, role)]),
                )
                for (relation, column, role) in calls
            ),
            # Descending cost with a canonical tiebreak. Without the trailing keys, two
            # logically identical workloads that happened to arrive in a different order
            # produce different output order, and downstream tasks' tests depend on it.
            # `Relation` is `order=True`, so it sorts directly with no key function.
            key=lambda u: (-u.cost_ms, u.relation, u.column, u.role.value),
        )
    )
    return Aggregation(
        usage=usage,
        total_cost_ms=total,
        skipped_unqualifiable=skipped_unqualifiable,
        tables=frozenset(tables),
        skipped_ambiguous=skipped_ambiguous,
    )
