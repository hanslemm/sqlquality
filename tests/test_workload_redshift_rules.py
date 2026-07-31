"""ADV101 (SORTKEY) and ADV102 (DISTKEY): Redshift's own physical-design proposals.

Both recommend DDL that rewrites the whole table, and Redshift exposes no per-column NDV
to predict distribution skew or predicate selectivity — see the plan's "why the rules are
not the Postgres rules renamed" section and each `propose_*` function's own docstring in
`redshift.py`. That is why each of the two tests its own MEDIUM cap independently: a
mutant that quietly added a HIGH branch must fail here, not just in prose.

ADV103 (DISTSTYLE ALL), ADV104 (VACUUM/ANALYZE), ADV105 (Redshift Advisor) and the
`propose()` dispatcher that wires all five together are a later task's addition to this
file.
"""

from __future__ import annotations

import pytest

from sqlquality.models import ColumnRole, ColumnUsage, Confidence, Relation, TableFacts
from sqlquality.workload.redshift import (
    RedshiftTableFacts,
    _diststyle_is_all,
    _diststyle_key_column,
    propose_distkey,
    propose_sortkey,
)

R = Relation(schema="public", table="orders")
R2 = Relation(schema="public", table="customers")


def _usage(
    relation: Relation,
    column: str,
    role: ColumnRole,
    *,
    cost_ms: float = 100.0,
    cost_share: float = 0.5,
    calls: int = 10,
) -> ColumnUsage:
    return ColumnUsage(
        relation=relation,
        column=column,
        role=role,
        calls=calls,
        cost_ms=cost_ms,
        cost_share=cost_share,
    )


def _facts(relation: Relation, *, row_estimate: int | None = 1_000) -> dict[Relation, TableFacts]:
    return {
        relation: TableFacts(
            relation=relation, row_estimate=row_estimate, size_bytes=1000, columns=()
        )
    }


def _phys(
    *,
    unsorted: float | None = None,
    stats_off: float | None = None,
    diststyle: str | None = None,
    sortkey1: str | None = None,
    skew_rows: float | None = None,
) -> RedshiftTableFacts:
    return RedshiftTableFacts(
        unsorted=unsorted,
        stats_off=stats_off,
        diststyle=diststyle,
        sortkey1=sortkey1,
        skew_rows=skew_rows,
    )


# ---------------------------------------------------------------------------
# ADV101 — propose_sortkey
# ---------------------------------------------------------------------------


def test_sortkey_proposes_at_medium_when_the_table_is_sorted_on_a_different_column():
    usage = [_usage(R, "created_at", ColumnRole.RANGE, cost_share=0.5)]
    physical = {R: _phys(sortkey1="status")}
    proposals = propose_sortkey(usage, _facts(R), physical, min_cost_share=0.1)
    assert len(proposals) == 1
    p = proposals[0]
    assert p.code == "ADV101"
    assert p.confidence is Confidence.MEDIUM
    assert p.ddl == 'ALTER TABLE "public"."orders" ALTER SORTKEY ("created_at");'
    assert "created_at" in p.title
    assert "'status'" in p.rationale


def test_sortkey_is_suppressed_when_already_the_sort_key():
    usage = [_usage(R, "created_at", ColumnRole.RANGE, cost_share=0.5)]
    physical = {R: _phys(sortkey1="created_at")}
    assert propose_sortkey(usage, _facts(R), physical, min_cost_share=0.1) == []


def test_sortkey_drops_to_low_when_the_existing_sort_key_could_not_be_read():
    usage = [_usage(R, "created_at", ColumnRole.RANGE, cost_share=0.5)]
    physical = {R: _phys(sortkey1=None)}
    proposals = propose_sortkey(usage, _facts(R), physical, min_cost_share=0.1)
    assert len(proposals) == 1
    assert proposals[0].confidence is Confidence.LOW
    assert "could not be read" in proposals[0].rationale


def test_sortkey_never_reaches_high_even_with_a_dominant_cost_share():
    """Pins the deliberate MEDIUM cap: no input, however hot, should ever produce HIGH."""
    usage = [_usage(R, "created_at", ColumnRole.RANGE, cost_share=0.99)]
    physical = {R: _phys(sortkey1="status")}
    proposals = propose_sortkey(usage, _facts(R), physical, min_cost_share=0.01)
    assert proposals[0].confidence is Confidence.MEDIUM


def test_sortkey_no_proposal_when_relation_absent_from_physical_facts():
    """Cannot tell a Spectrum table from a genuinely empty one, so no proposal at all —
    see `propose_sortkey`'s docstring. Not a silent skip: the reasoning is documented,
    but the runtime outcome is that this relation gets nothing from this rule.
    """
    usage = [_usage(R, "created_at", ColumnRole.RANGE, cost_share=0.5)]
    assert propose_sortkey(usage, _facts(R), {}, min_cost_share=0.1) == []


def test_sortkey_suppressed_below_min_cost_share():
    usage = [_usage(R, "created_at", ColumnRole.RANGE, cost_share=0.05)]
    physical = {R: _phys(sortkey1="status")}
    assert propose_sortkey(usage, _facts(R), physical, min_cost_share=0.1) == []


def test_sortkey_candidate_is_the_hottest_range_or_equality_column_not_join_or_group():
    usage = [
        _usage(R, "join_col", ColumnRole.JOIN, cost_ms=999.0, cost_share=0.9),
        _usage(R, "group_col", ColumnRole.GROUP, cost_ms=999.0, cost_share=0.9),
        _usage(R, "cold_range", ColumnRole.RANGE, cost_ms=10.0, cost_share=0.2),
        _usage(R, "hot_equality", ColumnRole.EQUALITY, cost_ms=500.0, cost_share=0.6),
    ]
    physical = {R: _phys(sortkey1=None)}
    proposals = propose_sortkey(usage, _facts(R), physical, min_cost_share=0.1)
    assert len(proposals) == 1
    assert proposals[0].evidence["column"] == "hot_equality"


def test_sortkey_discloses_stats_off_as_a_caveat_when_present():
    usage = [_usage(R, "created_at", ColumnRole.RANGE, cost_share=0.5)]
    physical = {R: _phys(sortkey1="status", stats_off=42.0)}
    proposals = propose_sortkey(usage, _facts(R), physical, min_cost_share=0.1)
    assert "42%" in proposals[0].rationale
    assert "stats_off" in proposals[0].rationale


def test_sortkey_omits_stats_off_caveat_when_it_is_zero():
    usage = [_usage(R, "created_at", ColumnRole.RANGE, cost_share=0.5)]
    physical = {R: _phys(sortkey1="status", stats_off=0.0)}
    proposals = propose_sortkey(usage, _facts(R), physical, min_cost_share=0.1)
    assert "stats_off" not in proposals[0].rationale


def test_sortkey_note_discloses_the_full_table_rewrite():
    usage = [_usage(R, "created_at", ColumnRole.RANGE, cost_share=0.5)]
    physical = {R: _phys(sortkey1="status")}
    proposals = propose_sortkey(usage, _facts(R), physical, min_cost_share=0.1)
    note = proposals[0].note or ""
    assert "rewrites the entire table" in note
    assert "CONCURRENTLY" in note


# ---------------------------------------------------------------------------
# ADV102 — propose_distkey
# ---------------------------------------------------------------------------


def test_distkey_proposes_at_medium_when_distributed_on_a_different_column():
    usage = [_usage(R, "customer_id", ColumnRole.JOIN, cost_share=0.5)]
    physical = {R: _phys(diststyle="KEY(order_id)")}
    proposals = propose_distkey(usage, _facts(R), physical, min_cost_share=0.1)
    assert len(proposals) == 1
    p = proposals[0]
    assert p.code == "ADV102"
    assert p.confidence is Confidence.MEDIUM
    assert p.ddl == 'ALTER TABLE "public"."orders" ALTER DISTKEY "customer_id";'


def test_distkey_suppressed_when_already_the_distribution_key():
    usage = [_usage(R, "customer_id", ColumnRole.JOIN, cost_share=0.5)]
    physical = {R: _phys(diststyle="KEY(customer_id)")}
    assert propose_distkey(usage, _facts(R), physical, min_cost_share=0.1) == []


def test_distkey_suppressed_when_already_distributed_on_the_key_inside_auto():
    usage = [_usage(R, "customer_id", ColumnRole.JOIN, cost_share=0.5)]
    physical = {R: _phys(diststyle="AUTO(KEY(customer_id))")}
    assert propose_distkey(usage, _facts(R), physical, min_cost_share=0.1) == []


def test_distkey_suppressed_when_diststyle_is_already_all():
    usage = [_usage(R, "customer_id", ColumnRole.JOIN, cost_share=0.5)]
    physical = {R: _phys(diststyle="ALL")}
    assert propose_distkey(usage, _facts(R), physical, min_cost_share=0.1) == []


def test_distkey_suppressed_when_diststyle_is_auto_all():
    usage = [_usage(R, "customer_id", ColumnRole.JOIN, cost_share=0.5)]
    physical = {R: _phys(diststyle="AUTO(ALL)")}
    assert propose_distkey(usage, _facts(R), physical, min_cost_share=0.1) == []


def test_distkey_not_suppressed_by_even_diststyle():
    usage = [_usage(R, "customer_id", ColumnRole.JOIN, cost_share=0.5)]
    physical = {R: _phys(diststyle="EVEN")}
    proposals = propose_distkey(usage, _facts(R), physical, min_cost_share=0.1)
    assert len(proposals) == 1
    assert proposals[0].confidence is Confidence.MEDIUM


def test_distkey_drops_to_low_when_diststyle_could_not_be_read():
    usage = [_usage(R, "customer_id", ColumnRole.JOIN, cost_share=0.5)]
    physical = {R: _phys(diststyle=None)}
    proposals = propose_distkey(usage, _facts(R), physical, min_cost_share=0.1)
    assert len(proposals) == 1
    assert proposals[0].confidence is Confidence.LOW
    assert "could not be read" in proposals[0].rationale


def test_distkey_never_reaches_high_even_with_a_dominant_cost_share():
    usage = [_usage(R, "customer_id", ColumnRole.JOIN, cost_share=0.99)]
    physical = {R: _phys(diststyle="EVEN")}
    proposals = propose_distkey(usage, _facts(R), physical, min_cost_share=0.01)
    assert proposals[0].confidence is Confidence.MEDIUM


def test_distkey_no_proposal_when_relation_absent_from_physical_facts():
    usage = [_usage(R, "customer_id", ColumnRole.JOIN, cost_share=0.5)]
    assert propose_distkey(usage, _facts(R), {}, min_cost_share=0.1) == []


def test_distkey_suppressed_below_min_cost_share():
    usage = [_usage(R, "customer_id", ColumnRole.JOIN, cost_share=0.05)]
    physical = {R: _phys(diststyle="EVEN")}
    assert propose_distkey(usage, _facts(R), physical, min_cost_share=0.1) == []


def test_distkey_candidate_is_the_hottest_join_column_not_range_or_equality():
    usage = [
        _usage(R, "range_col", ColumnRole.RANGE, cost_ms=999.0, cost_share=0.9),
        _usage(R, "cold_join", ColumnRole.JOIN, cost_ms=10.0, cost_share=0.2),
        _usage(R, "hot_join", ColumnRole.JOIN, cost_ms=500.0, cost_share=0.6),
    ]
    physical = {R: _phys(diststyle="EVEN")}
    proposals = propose_distkey(usage, _facts(R), physical, min_cost_share=0.1)
    assert len(proposals) == 1
    assert proposals[0].evidence["column"] == "hot_join"


def test_distkey_discloses_skew_rows_as_a_caveat_when_present():
    usage = [_usage(R, "customer_id", ColumnRole.JOIN, cost_share=0.5)]
    physical = {R: _phys(diststyle="EVEN", skew_rows=3.5)}
    proposals = propose_distkey(usage, _facts(R), physical, min_cost_share=0.1)
    assert "3.50" in proposals[0].rationale
    assert "current" in proposals[0].rationale.lower()


def test_distkey_note_discloses_the_full_table_rewrite():
    usage = [_usage(R, "customer_id", ColumnRole.JOIN, cost_share=0.5)]
    physical = {R: _phys(diststyle="EVEN")}
    proposals = propose_distkey(usage, _facts(R), physical, min_cost_share=0.1)
    note = proposals[0].note or ""
    assert "rewrites the entire table" in note
    assert "CONCURRENTLY" in note


def test_distkey_omits_skew_rows_caveat_when_absent():
    usage = [_usage(R, "customer_id", ColumnRole.JOIN, cost_share=0.5)]
    physical = {R: _phys(diststyle="EVEN", skew_rows=None)}
    proposals = propose_distkey(usage, _facts(R), physical, min_cost_share=0.1)
    assert "skew_rows" not in proposals[0].rationale


def test_distkey_omits_stats_off_caveat_when_it_is_zero():
    usage = [_usage(R, "customer_id", ColumnRole.JOIN, cost_share=0.5)]
    physical = {R: _phys(diststyle="EVEN", stats_off=0.0)}
    proposals = propose_distkey(usage, _facts(R), physical, min_cost_share=0.1)
    assert "stats_off" not in proposals[0].rationale


# ---------------------------------------------------------------------------
# diststyle parsing helpers (used by propose_distkey)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("diststyle", "expected"),
    [
        ("KEY(customer_id)", "customer_id"),
        ("AUTO(KEY(customer_id))", "customer_id"),
        ("EVEN", None),
        ("ALL", None),
        ("AUTO(ALL)", None),
    ],
)
def test_diststyle_key_column_parsing(diststyle, expected):
    assert _diststyle_key_column(diststyle) == expected


@pytest.mark.parametrize(
    ("diststyle", "expected"),
    [
        ("ALL", True),
        ("AUTO(ALL)", True),
        ("EVEN", False),
        ("KEY(customer_id)", False),
        ("AUTO(KEY(customer_id))", False),
    ],
)
def test_diststyle_is_all_parsing(diststyle, expected):
    assert _diststyle_is_all(diststyle) is expected
