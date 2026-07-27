"""Roll per-query column roles up into a cost-weighted usage index."""

from __future__ import annotations

import re
from collections import defaultdict
from functools import lru_cache

from sqlquality.models import Aggregation, ColumnRole, ColumnUsage, Relation, Workload
from sqlquality.sqlast import SqlParseError, parse
from sqlquality.workload.extract import AmbiguousRelation, UnqualifiableQuery, extract_usage
from sqlquality.workload.fingerprint import FLAG_SELECT_STAR

_Key = tuple[Relation, str, ColumnRole]


@lru_cache(maxsize=4096)
def _identifier_pattern(name: str) -> re.Pattern[str]:
    """Compiled whole-identifier matcher for one name, compiled once per name.

    ``star_tables`` tests every (star-stat, table) pair, and a schema with many tables was
    recompiling the same handful of table-name patterns over and over, thrashing `re`'s own
    pattern cache. Caching by name here means each identifier is compiled once regardless of
    how many stats or tables it is checked against.
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


def star_tables(workload: Workload, schema: dict) -> frozenset[Relation]:
    """Relations a `SELECT *` query group merely *mentions*, matched against ``schema``.

    A bare `select * from wide_t` filters nothing, so it contributes no column usage and
    the relation never appears in ``Aggregation.tables``. Introspecting only the relations
    that produced usage therefore left the star rule with no column counts to test — inert
    for precisely the workload it exists to catch. These names are unioned in before catalog
    facts are fetched.

    A name held by two introspected schemas is skipped rather than attributed to either or
    to both: over-reporting would put a wide-table warning on a table the query never
    touched. Consistent with `resolve_relation`, which declines the same guess.

    Deliberately *not* added to ``Aggregation.tables``: that set means "relations with
    recorded column usage" and feeds the unused-index rule's notion of a hot table.
    """
    found: set[Relation] = set()
    for stat in workload.stats:
        if FLAG_SELECT_STAR not in stat.flags:
            continue
        for table in _table_names(schema):
            if not mentions_table(table, stat.sql):
                continue
            owners = [name for name, tables in schema.items() if table in tables]
            if len(owners) == 1:
                found.add(Relation(schema=owners[0], table=table))
    return frozenset(found)


def _table_names(schema: dict) -> frozenset[str]:
    """Every bare table name in a nested schema map, deduplicated across schemas."""
    return frozenset(table for tables in schema.values() for table in tables)


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
