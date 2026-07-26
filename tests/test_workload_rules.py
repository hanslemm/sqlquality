from sqlquality.models import (
    Aggregation,
    ColumnRole,
    ColumnUsage,
    Confidence,
    QueryStat,
    TableFacts,
    Workload,
)
from sqlquality.workload.fingerprint import FLAG_LEADING_WILDCARD_LIKE, FLAG_SELECT_STAR
from sqlquality.workload.postgres import (
    PgIndex,
    PostgresWorkloadAdapter,
    propose_indexes,
    propose_partial_indexes,
    propose_redundant_indexes,
    propose_sargability,
    propose_select_star,
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


def _workload(*stats):
    return Workload(stats=tuple(stats), window_description="w")


def test_partial_index_proposed_for_a_hot_not_null_check():
    proposals = propose_partial_indexes(
        [
            usage("status", ColumnRole.EQUALITY, cost_ms=90.0),
            usage("shipped_at", ColumnRole.NOT_NULL_CHECK, cost_share=0.4, cost_ms=40.0),
        ],
        facts(),
        min_cost_share=0.01,
    )
    assert codes(proposals) == ["ADV004"]
    assert "IS NOT NULL" in proposals[0].ddl


def test_partial_index_polarity_follows_the_predicate():
    proposals = propose_partial_indexes(
        [
            usage("status", ColumnRole.EQUALITY, cost_ms=90.0),
            usage("shipped_at", ColumnRole.NULL_CHECK, cost_share=0.4, cost_ms=40.0),
        ],
        facts(),
        min_cost_share=0.01,
    )
    assert "IS NULL" in proposals[0].ddl
    assert "IS NOT NULL" not in proposals[0].ddl


def test_no_partial_index_without_an_equality_column_to_index():
    proposals = propose_partial_indexes(
        [usage("shipped_at", ColumnRole.NOT_NULL_CHECK, cost_share=0.4)],
        facts(),
        min_cost_share=0.01,
    )
    assert proposals == []


def test_non_sargable_column_gets_an_attributed_proposal():
    proposals = propose_sargability(
        [usage("status", ColumnRole.NON_SARGABLE, cost_share=0.3)],
        _workload(),
        min_cost_share=0.01,
    )
    assert codes(proposals) == ["ADV005"]
    assert proposals[0].evidence["column"] == "status"
    assert proposals[0].confidence is Confidence.HIGH


def test_leading_wildcard_reports_without_column_attribution():
    stat = QueryStat(
        fingerprint="fp",
        sql="select id from orders where note like $1",
        calls=5,
        total_time_ms=100.0,
        flags=frozenset({FLAG_LEADING_WILDCARD_LIKE}),
    )
    proposals = propose_sargability([], _workload(stat), min_cost_share=0.01)
    assert codes(proposals) == ["ADV005"]
    assert proposals[0].evidence.get("column") is None
    # Redaction erased the pattern, so we can name the query group but not the column.
    assert proposals[0].confidence is Confidence.MEDIUM


def test_hot_select_star_on_a_wide_table():
    stat = QueryStat(
        fingerprint="fp",
        sql="select * from orders",
        calls=5,
        total_time_ms=100.0,
        flags=frozenset({FLAG_SELECT_STAR}),
    )
    wide = {
        "orders": TableFacts(
            name="orders",
            row_estimate=10**6,
            size_bytes=10**8,
            columns=tuple(f"c{i}" for i in range(30)),
        )
    }
    proposals = propose_select_star(_workload(stat), wide, min_cost_share=0.01)
    assert codes(proposals) == ["ADV006"]


def test_select_star_ignored_on_a_narrow_table():
    stat = QueryStat(
        fingerprint="fp",
        sql="select * from orders",
        calls=5,
        total_time_ms=100.0,
        flags=frozenset({FLAG_SELECT_STAR}),
    )
    narrow = {
        "orders": TableFacts(name="orders", row_estimate=10**6, size_bytes=1, columns=("a", "b"))
    }
    assert propose_select_star(_workload(stat), narrow, min_cost_share=0.01) == []


def test_select_star_table_matching_ignores_substring_false_positives():
    """A plain `name in sql` test would misfire three ways this locks out.

    A wide table `order` must not match a query on `orders` (name is a substring of a
    different table); a wide table `orders` must not match only because it appears inside
    a column alias `orders_total`; and a wide table `cart` must not match a query that only
    touches `shopping_cart`.
    """
    stat = QueryStat(
        fingerprint="fp",
        sql="select id, count(*) as orders_total from shopping_cart",
        calls=5,
        total_time_ms=100.0,
        flags=frozenset({FLAG_SELECT_STAR}),
    )
    wide = {
        "order": TableFacts(
            name="order",
            row_estimate=10**6,
            size_bytes=1,
            columns=tuple(f"c{i}" for i in range(30)),
        ),
        "orders": TableFacts(
            name="orders",
            row_estimate=10**6,
            size_bytes=1,
            columns=tuple(f"c{i}" for i in range(30)),
        ),
        "cart": TableFacts(
            name="cart", row_estimate=10**6, size_bytes=1, columns=tuple(f"c{i}" for i in range(30))
        ),
    }
    assert propose_select_star(_workload(stat), wide, min_cost_share=0.01) == []


def test_propose_collapses_an_index_flagged_both_unused_and_redundant():
    """ADV002 and ADV003 can both fire on one index, emitting the same DROP twice.

    ADV003 must survive: prefix redundancy is provable from the column lists alone, while
    ADV002 rests on a scan counter covering only the window since the last stats reset.
    """
    existing = {
        "orders": (
            PgIndex("idx_narrow", ("status",), False, False, 0, 4096),
            PgIndex("idx_wide", ("status", "created_at"), False, False, 5, 8192),
        )
    }
    adapter = PostgresWorkloadAdapter(querier=lambda sql, params: [])
    aggregation = Aggregation(
        usage=(), total_cost_ms=100.0, skipped_unqualifiable=0, tables=frozenset({"orders"})
    )
    adapter.fetch_indexes = lambda schemas, tables: existing  # type: ignore[method-assign]
    proposals = adapter.propose(aggregation, {}, _workload(), min_cost_share=0.01)

    drops = [p for p in proposals if p.ddl == "DROP INDEX idx_narrow;"]
    assert len(drops) == 1
    assert drops[0].code == "ADV003"


def test_propose_composes_all_rules_and_ranks_high_confidence_first():
    aggregation = Aggregation(
        usage=(
            usage("status", ColumnRole.EQUALITY, cost_share=0.6, cost_ms=60.0),
            usage("note", ColumnRole.NON_SARGABLE, cost_share=0.2, cost_ms=20.0),
        ),
        total_cost_ms=100.0,
        skipped_unqualifiable=0,
        tables=frozenset({"orders"}),
    )
    adapter = PostgresWorkloadAdapter(querier=lambda sql, params: [])
    proposals = adapter.propose(
        aggregation,
        facts(ndv={"status": 9999.0}),
        _workload(),
        min_cost_share=0.01,
    )
    assert {p.code for p in proposals} >= {"ADV001", "ADV005"}
    assert proposals[0].confidence is Confidence.HIGH
