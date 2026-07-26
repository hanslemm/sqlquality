from sqlquality.models import ColumnRole, QueryStat, Workload
from sqlquality.workload.aggregate import aggregate

SCHEMA = {"orders": {"id": "INT", "status": "TEXT", "created_at": "TIMESTAMP"}}


def _workload(*pairs):
    return Workload(
        stats=tuple(
            QueryStat(fingerprint=f"fp{i}", sql=sql, calls=calls, total_time_ms=cost)
            for i, (sql, calls, cost) in enumerate(pairs)
        ),
        window_description="w",
    )


def _find(agg, column, role):
    return next(u for u in agg.usage if u.column == column and u.role is role)


def test_cost_share_is_fraction_of_analyzed_total():
    agg = aggregate(
        _workload(
            ("select id from orders where status = $1", 1, 75.0),
            ("select id from orders where created_at > $1", 1, 25.0),
        ),
        SCHEMA,
        "postgres",
    )
    assert agg.total_cost_ms == 100.0
    assert _find(agg, "status", ColumnRole.EQUALITY).cost_share == 0.75
    assert _find(agg, "created_at", ColumnRole.RANGE).cost_share == 0.25


def test_same_column_and_role_accumulates_across_fingerprints():
    agg = aggregate(
        _workload(
            ("select id from orders where status = $1", 2, 10.0),
            ("select created_at from orders where status = $1", 3, 30.0),
        ),
        SCHEMA,
        "postgres",
    )
    usage = _find(agg, "status", ColumnRole.EQUALITY)
    assert usage.cost_ms == 40.0
    assert usage.calls == 5
    assert usage.fingerprints == 2


def test_unqualifiable_queries_are_counted_not_raised():
    agg = aggregate(
        _workload(
            ("select mystery from unknown_table", 1, 10.0),
            ("select id from orders where status = $1", 1, 10.0),
        ),
        SCHEMA,
        "postgres",
    )
    assert agg.skipped_unqualifiable == 1
    assert agg.tables == frozenset({"orders"})


def test_usage_is_sorted_by_cost_descending():
    agg = aggregate(
        _workload(
            ("select id from orders where status = $1", 1, 5.0),
            ("select id from orders where created_at > $1", 1, 50.0),
        ),
        SCHEMA,
        "postgres",
    )
    assert agg.usage[0].column == "created_at"


def test_empty_workload_yields_empty_aggregation_and_no_division_error():
    agg = aggregate(Workload(stats=(), window_description="w"), SCHEMA, "postgres")
    assert agg.usage == ()
    assert agg.total_cost_ms == 0.0
    assert agg.tables == frozenset()
