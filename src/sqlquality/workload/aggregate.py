"""Roll per-query column roles up into a cost-weighted usage index."""

from __future__ import annotations

from collections import defaultdict

from sqlquality.models import Aggregation, ColumnRole, ColumnUsage, Workload
from sqlquality.sqlast import SqlParseError, parse
from sqlquality.workload.extract import UnqualifiableQuery, extract_usage

_Key = tuple[str, str, ColumnRole]


def aggregate(workload: Workload, schema: dict, dialect: str) -> Aggregation:
    """Weight every (table, column, role) by the cost of the queries that use it."""
    calls: dict[_Key, int] = defaultdict(int)
    cost: dict[_Key, float] = defaultdict(float)
    fingerprints: dict[_Key, int] = defaultdict(int)
    tables: set[str] = set()
    skipped_unqualifiable = 0

    for stat in workload.stats:
        try:
            tree = parse(stat.sql, dialect)
            triples = extract_usage(tree, dialect, schema)
        except (SqlParseError, UnqualifiableQuery):
            skipped_unqualifiable += 1
            continue
        for key in triples:
            calls[key] += stat.calls
            cost[key] += stat.total_time_ms
            fingerprints[key] += 1
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
                    table=table,
                    column=column,
                    role=role,
                    calls=calls[(table, column, role)],
                    cost_ms=cost[(table, column, role)],
                    cost_share=(cost[(table, column, role)] / total) if total else 0.0,
                    fingerprints=fingerprints[(table, column, role)],
                )
                for (table, column, role) in calls
            ),
            # Descending cost with a canonical tiebreak. Without the trailing keys, two
            # logically identical workloads that happened to arrive in a different order
            # produce different output order, and downstream tasks' tests depend on it.
            key=lambda u: (-u.cost_ms, u.table, u.column, u.role.value),
        )
    )
    return Aggregation(
        usage=usage,
        total_cost_ms=total,
        skipped_unqualifiable=skipped_unqualifiable,
        tables=frozenset(tables),
    )
