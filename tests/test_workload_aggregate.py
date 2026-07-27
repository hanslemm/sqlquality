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


def test_equal_cost_entries_are_ordered_canonically_not_by_arrival():
    """Two workloads differing only in arrival order must aggregate identically.

    Sorting on cost alone leaves ties to Python's stable sort, which preserves insertion
    order — so the same logical workload yields different output depending on the order
    the engine happened to return rows in, and downstream tests become order-dependent.
    """
    forward = aggregate(
        _workload(
            ("select id from orders where status = $1", 1, 10.0),
            ("select id from orders where created_at > $1", 1, 10.0),
        ),
        SCHEMA,
        "postgres",
    )
    reverse = aggregate(
        _workload(
            ("select id from orders where created_at > $1", 1, 10.0),
            ("select id from orders where status = $1", 1, 10.0),
        ),
        SCHEMA,
        "postgres",
    )
    assert [(u.table, u.column, u.role) for u in forward.usage] == [
        (u.table, u.column, u.role) for u in reverse.usage
    ]


def test_skipped_stats_still_count_toward_the_denominator():
    """cost_share is a fraction of the whole window, not of the analyzable part.

    The 90ms query cannot be qualified, so it contributes no usage — but its cost stays in
    the denominator, leaving the surviving 10ms query at 0.1 rather than 1.0. This keeps
    the number honest about how much of the database's work we actually explained.
    """
    agg = aggregate(
        _workload(
            ("select mystery from unknown_table", 1, 90.0),
            ("select id from orders where status = $1", 1, 10.0),
        ),
        SCHEMA,
        "postgres",
    )
    assert agg.skipped_unqualifiable == 1
    assert agg.total_cost_ms == 100.0
    assert _find(agg, "status", ColumnRole.EQUALITY).cost_share == 0.1


def test_a_multi_predicate_query_credits_its_full_cost_to_each_column():
    """Shares deliberately do not sum to 1: both predicates are involved in the same cost.

    Proposals therefore take the max cost_share over their columns, never the sum.
    """
    agg = aggregate(
        _workload(("select id from orders where status = $1 and created_at > $2", 1, 100.0)),
        SCHEMA,
        "postgres",
    )
    assert _find(agg, "status", ColumnRole.EQUALITY).cost_share == 1.0
    assert _find(agg, "created_at", ColumnRole.RANGE).cost_share == 1.0


def test_role_breaks_ties_when_table_and_column_match():
    """The fourth sort key. Same column, same cost, two roles — order must be canonical."""
    forward = aggregate(
        _workload(
            ("select id from orders where status = $1", 1, 10.0),
            ("select id from orders group by status", 1, 10.0),
        ),
        SCHEMA,
        "postgres",
    )
    reverse = aggregate(
        _workload(
            ("select id from orders group by status", 1, 10.0),
            ("select id from orders where status = $1", 1, 10.0),
        ),
        SCHEMA,
        "postgres",
    )
    assert [u.role for u in forward.usage] == [u.role for u in reverse.usage]


def test_empty_workload_yields_empty_aggregation_and_no_division_error():
    agg = aggregate(Workload(stats=(), window_description="w"), SCHEMA, "postgres")
    assert agg.usage == ()
    assert agg.total_cost_ms == 0.0
    assert agg.tables == frozenset()
