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
            key=lambda u: u.cost_ms,
            reverse=True,
        )
    )
    return Aggregation(
        usage=usage,
        total_cost_ms=total,
        skipped_unqualifiable=skipped_unqualifiable,
        tables=frozenset(tables),
    )
