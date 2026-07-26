from sqlquality.models import ColumnRole, ColumnUsage, Confidence, TableFacts
from sqlquality.workload.postgres import (
    PgIndex,
    propose_indexes,
    propose_redundant_indexes,
    propose_unused_indexes,
)


def usage(column, role, cost_share=0.5, cost_ms=50.0, table="orders"):
    return ColumnUsage(
        table=table,
        column=column,
        role=role,
        calls=10,
        cost_ms=cost_ms,
        cost_share=cost_share,
        fingerprints=1,
    )


def facts(rows=1_000_000, ndv=None, columns=("id", "status", "created_at", "customer_id")):
    return {
        "orders": TableFacts(
            name="orders", row_estimate=rows, size_bytes=10**8, columns=columns, ndv=ndv or {}
        )
    }


def codes(proposals):
    return [p.code for p in proposals]


def test_equality_then_range_ordering_in_the_candidate_index():
    proposals = propose_indexes(
        [
            usage("status", ColumnRole.EQUALITY, cost_ms=90.0),
            usage("created_at", ColumnRole.RANGE, cost_ms=80.0),
        ],
        facts(),
        {},
        min_cost_share=0.01,
    )
    assert codes(proposals) == ["ADV001"]
    assert proposals[0].evidence["columns"] == ("status", "created_at")


def test_only_one_range_column_is_included():
    proposals = propose_indexes(
        [
            usage("status", ColumnRole.EQUALITY, cost_ms=90.0),
            usage("created_at", ColumnRole.RANGE, cost_ms=80.0),
            usage("shipped_at", ColumnRole.RANGE, cost_ms=70.0),
        ],
        facts(columns=("status", "created_at", "shipped_at")),
        {},
        min_cost_share=0.01,
    )
    assert proposals[0].evidence["columns"] == ("status", "created_at")


def test_arity_is_capped():
    proposals = propose_indexes(
        [
            usage("a", ColumnRole.EQUALITY, cost_ms=99.0),
            usage("b", ColumnRole.EQUALITY, cost_ms=98.0),
            usage("c", ColumnRole.EQUALITY, cost_ms=97.0),
            usage("d", ColumnRole.EQUALITY, cost_ms=96.0),
        ],
        facts(columns=("a", "b", "c", "d")),
        {},
        min_cost_share=0.01,
    )
    assert len(proposals[0].evidence["columns"]) == 3


def test_small_tables_are_suppressed_entirely():
    proposals = propose_indexes(
        [usage("status", ColumnRole.EQUALITY)],
        facts(rows=500),
        {},
        min_cost_share=0.01,
    )
    assert proposals == []


def test_below_min_cost_share_is_suppressed():
    proposals = propose_indexes(
        [usage("status", ColumnRole.EQUALITY, cost_share=0.001)],
        facts(),
        {},
        min_cost_share=0.01,
    )
    assert proposals == []


def test_existing_index_with_the_same_leading_prefix_is_not_reproposed():
    existing = {"orders": (PgIndex("idx", ("status", "created_at"), False, False, 10, 8192),)}
    proposals = propose_indexes(
        [
            usage("status", ColumnRole.EQUALITY, cost_ms=90.0),
            usage("created_at", ColumnRole.RANGE, cost_ms=80.0),
        ],
        facts(),
        existing,
        min_cost_share=0.01,
    )
    assert proposals == []


def test_a_wider_existing_index_still_covers_a_narrower_candidate():
    existing = {"orders": (PgIndex("idx", ("status", "created_at", "id"), False, False, 5, 1),)}
    proposals = propose_indexes(
        [usage("status", ColumnRole.EQUALITY, cost_ms=90.0)],
        facts(),
        existing,
        min_cost_share=0.01,
    )
    assert proposals == []


def test_a_narrower_existing_index_does_not_cover_a_wider_candidate():
    """The reverse of the coverage rule, and the direction that inverting the slice breaks.

    An index on (status) does not serve a candidate (status, created_at). Only the
    wider-covers-narrower test existed, so `candidate[:len(existing)] == existing` would
    have shipped silently.
    """
    existing = {"orders": (PgIndex("idx_status", ("status",), False, False, 5, 1),)}
    proposals = propose_indexes(
        [
            usage("status", ColumnRole.EQUALITY, cost_ms=90.0),
            usage("created_at", ColumnRole.RANGE, cost_ms=80.0),
        ],
        facts(),
        existing,
        min_cost_share=0.01,
    )
    assert codes(proposals) == ["ADV001"]
    assert proposals[0].evidence["columns"] == ("status", "created_at")


def test_arity_cap_keeps_the_range_column_last_when_it_bites():
    """The interaction of the two most important ordering rules, previously untested.

    Four equality columns plus a range column with max_arity 3 must drop the weakest
    equality column, not the range column, and the range column must still come last —
    once a range predicate is used, later columns cannot be probed by equality.
    """
    proposals = propose_indexes(
        [
            usage("a", ColumnRole.EQUALITY, cost_ms=99.0),
            usage("b", ColumnRole.EQUALITY, cost_ms=98.0),
            usage("c", ColumnRole.EQUALITY, cost_ms=97.0),
            usage("d", ColumnRole.EQUALITY, cost_ms=96.0),
            usage("e", ColumnRole.RANGE, cost_ms=50.0),
        ],
        facts(columns=("a", "b", "c", "d", "e")),
        {},
        min_cost_share=0.01,
    )
    columns = proposals[0].evidence["columns"]
    assert columns == ("a", "b", "e")
    assert len(columns) == 3


def test_unknown_row_count_is_low_confidence_and_says_why():
    """The small-table gate cannot run without a row count.

    Suppressing entirely would deny advice to anyone whose row-count grant is missing; the
    cost evidence is real. But reporting MEDIUM would present an unverified proposal as
    ordinarily-trustworthy, so it is LOW and the rationale states the gap.
    """
    proposals = propose_indexes(
        [usage("status", ColumnRole.EQUALITY)],
        facts(rows=None, ndv={"status": 9999.0}),
        {},
        min_cost_share=0.01,
    )
    assert codes(proposals) == ["ADV001"]
    assert proposals[0].confidence is Confidence.LOW
    assert proposals[0].evidence["row_estimate"] is None
    assert "unknown" in proposals[0].rationale.lower()


def test_cost_share_is_the_max_never_the_sum():
    proposals = propose_indexes(
        [
            usage("status", ColumnRole.EQUALITY, cost_share=0.6, cost_ms=90.0),
            usage("created_at", ColumnRole.RANGE, cost_share=0.6, cost_ms=80.0),
        ],
        facts(),
        {},
        min_cost_share=0.01,
    )
    assert proposals[0].evidence["cost_share"] == 0.6


def test_confidence_is_high_only_with_stats_and_a_selective_leading_column():
    high = propose_indexes(
        [usage("status", ColumnRole.EQUALITY)],
        facts(ndv={"status": 5000.0}),
        {},
        min_cost_share=0.01,
    )
    assert high[0].confidence is Confidence.HIGH

    no_stats = propose_indexes(
        [usage("status", ColumnRole.EQUALITY)],
        facts(ndv={}),
        {},
        min_cost_share=0.01,
    )
    assert no_stats[0].confidence is Confidence.MEDIUM

    unselective = propose_indexes(
        [usage("status", ColumnRole.EQUALITY)],
        facts(ndv={"status": 3.0}),
        {},
        min_cost_share=0.01,
    )
    assert unselective[0].confidence is Confidence.LOW


def test_unused_index_proposed_for_drop_but_never_a_constraint_index():
    existing = {
        "orders": (
            PgIndex("idx_cold", ("note",), False, False, 0, 4096),
            PgIndex("orders_pkey", ("id",), True, True, 0, 4096),
            PgIndex("uq_email", ("email",), True, False, 0, 4096),
            PgIndex("idx_warm", ("status",), False, False, 42, 4096),
        )
    }
    proposals = propose_unused_indexes(existing, hot_tables=frozenset({"orders"}))
    assert codes(proposals) == ["ADV002"]
    assert proposals[0].evidence["index"] == "idx_cold"
    assert proposals[0].confidence is Confidence.MEDIUM


def test_unused_index_rule_ignores_tables_outside_the_workload():
    existing = {"archive": (PgIndex("idx_cold", ("a",), False, False, 0, 1),)}
    assert propose_unused_indexes(existing, hot_tables=frozenset({"orders"})) == []


def test_redundant_prefix_index_proposed_for_drop():
    existing = {
        "orders": (
            PgIndex("idx_narrow", ("status",), False, False, 5, 1),
            PgIndex("idx_wide", ("status", "created_at"), False, False, 5, 1),
        )
    }
    proposals = propose_redundant_indexes(existing)
    assert codes(proposals) == ["ADV003"]
    assert proposals[0].evidence["index"] == "idx_narrow"
    assert proposals[0].confidence is Confidence.HIGH


def test_a_unique_prefix_index_is_never_called_redundant():
    existing = {
        "orders": (
            PgIndex("uq_status", ("status",), True, False, 5, 1),
            PgIndex("idx_wide", ("status", "created_at"), False, False, 5, 1),
        )
    }
    assert propose_redundant_indexes(existing) == []
