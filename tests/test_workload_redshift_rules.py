"""ADV101-105: Redshift's own physical-design proposals (SORTKEY, DISTKEY, DISTSTYLE ALL,
VACUUM/ANALYZE, and Redshift Advisor's own recommendations).

Every one of ADV101-103 recommends DDL that rewrites the whole table, and Redshift exposes
no per-column NDV to predict distribution skew or predicate selectivity — see the plan's
"why the rules are not the Postgres rules renamed" section and each `propose_*` function's
own docstring in `redshift.py`. That is why each of the three tests its own MEDIUM cap
independently: a mutant that quietly added a HIGH branch must fail here, not just in prose.
"""

from __future__ import annotations

import pytest

from sqlquality.dbtproject import ModelNode
from sqlquality.models import (
    Aggregation,
    ColumnRole,
    ColumnUsage,
    Confidence,
    Proposal,
    Relation,
    TableFacts,
    Workload,
)
from sqlquality.workload.dbt import DbtContext, enrich_proposals
from sqlquality.workload.redshift import (
    CAP_ADVISOR,
    DEGRADATION_PHYSICAL_FACTS_GAP,
    MAX_ROWS_FOR_DISTSTYLE_ALL,
    RedshiftAdvisorRow,
    RedshiftTableFacts,
    RedshiftWorkloadAdapter,
    _advisor_category,
    _collapse_diststyle_all_over_distkey,
    _diststyle_is_all,
    _diststyle_key_column,
    _disclose_advisor_agreement,
    _quote_ident,
    _skipped_for_physical_gap,
    propose_advisor,
    propose_diststyle_all,
    propose_distkey,
    propose_maintenance,
    propose_sortkey,
)

# Sibling test module, not a package import: `tests/` has no `__init__.py`, so pytest puts it
# on `sys.path` and its modules are importable by bare name. Imported rather than copied
# because two hand-maintained lists of invisible control characters are two lists free to
# drift apart, and the reason this file asserts the same property again is that the *call site*
# differs (each adapter's `render_ddl` calls the shared guard separately), not the character set.
from test_workload_rules import EXTRA_LINE_BREAKS  # noqa: E402

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


def test_sortkey_medium_cap_rationale_states_the_reason_not_just_the_word():
    """The MEDIUM rung's explanation is the operator's only reason a whole-table rewrite
    is offered at less than HIGH, and it is also what stops a later reader adding the
    missing HIGH branch "for symmetry" — pin the substance, not just that the word
    "MEDIUM" appears somewhere in the rationale.
    """
    usage = [_usage(R, "created_at", ColumnRole.RANGE, cost_share=0.5)]
    physical = {R: _phys(sortkey1="status")}
    proposals = propose_sortkey(usage, _facts(R), physical, min_cost_share=0.1)
    rationale = proposals[0].rationale
    assert "a SORTKEY change only repays the rewrite if this predicate is selective" in rationale
    assert "Redshift exposes no per-column" in rationale
    assert "distinct-value statistics" in rationale


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


def test_distkey_medium_cap_rationale_states_the_reason_not_just_the_word():
    usage = [_usage(R, "customer_id", ColumnRole.JOIN, cost_share=0.5)]
    physical = {R: _phys(diststyle="EVEN")}
    proposals = propose_distkey(usage, _facts(R), physical, min_cost_share=0.1)
    rationale = proposals[0].rationale
    assert "distribution skew is what makes a DISTKEY choice good or catastrophic" in rationale
    assert "Redshift exposes no per-column" in rationale
    assert "distinct-value statistics" in rationale


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
# ADV103 — propose_diststyle_all
# ---------------------------------------------------------------------------


def test_diststyle_all_proposes_at_medium_when_small_and_joined_and_not_already_all():
    usage = [_usage(R2, "id", ColumnRole.JOIN, cost_share=0.5)]
    physical = {R2: _phys(diststyle="EVEN")}
    proposals = propose_diststyle_all(
        usage, _facts(R2, row_estimate=1_000), physical, min_cost_share=0.1
    )
    assert len(proposals) == 1
    p = proposals[0]
    assert p.code == "ADV103"
    assert p.confidence is Confidence.MEDIUM
    assert p.ddl == 'ALTER TABLE "public"."customers" ALTER DISTSTYLE ALL;'


def test_diststyle_all_suppressed_when_already_all():
    usage = [_usage(R2, "id", ColumnRole.JOIN, cost_share=0.5)]
    physical = {R2: _phys(diststyle="ALL")}
    proposals = propose_diststyle_all(
        usage, _facts(R2, row_estimate=1_000), physical, min_cost_share=0.1
    )
    assert proposals == []


def test_diststyle_all_suppressed_when_over_the_row_ceiling():
    usage = [_usage(R2, "id", ColumnRole.JOIN, cost_share=0.5)]
    physical = {R2: _phys(diststyle="EVEN")}
    proposals = propose_diststyle_all(
        usage, _facts(R2, row_estimate=10_000_000), physical, min_cost_share=0.1
    )
    assert proposals == []


def test_diststyle_all_not_suppressed_at_exactly_the_ceiling():
    """The gate is `rows > max_rows`, not `>=` — a table exactly at the ceiling is still a
    candidate. A mutant flipping the comparison direction or operator must fail this."""
    usage = [_usage(R2, "id", ColumnRole.JOIN, cost_share=0.5)]
    physical = {R2: _phys(diststyle="EVEN")}
    proposals = propose_diststyle_all(
        usage, _facts(R2, row_estimate=1_000_000), physical, min_cost_share=0.1, max_rows=1_000_000
    )
    assert len(proposals) == 1


def test_diststyle_all_requires_the_table_to_be_joined_in_the_workload():
    usage = [_usage(R2, "status", ColumnRole.EQUALITY, cost_share=0.9)]
    physical = {R2: _phys(diststyle="EVEN")}
    proposals = propose_diststyle_all(
        usage, _facts(R2, row_estimate=1_000), physical, min_cost_share=0.1
    )
    assert proposals == []


def test_diststyle_all_suppressed_below_min_cost_share():
    usage = [_usage(R2, "id", ColumnRole.JOIN, cost_share=0.05)]
    physical = {R2: _phys(diststyle="EVEN")}
    proposals = propose_diststyle_all(
        usage, _facts(R2, row_estimate=1_000), physical, min_cost_share=0.1
    )
    assert proposals == []


def test_diststyle_all_no_proposal_when_relation_absent_from_physical_facts():
    usage = [_usage(R2, "id", ColumnRole.JOIN, cost_share=0.5)]
    proposals = propose_diststyle_all(usage, _facts(R2, row_estimate=1_000), {}, min_cost_share=0.1)
    assert proposals == []


def test_diststyle_all_low_when_both_row_count_and_diststyle_are_unreadable():
    usage = [_usage(R2, "id", ColumnRole.JOIN, cost_share=0.5)]
    physical = {R2: _phys(diststyle=None)}
    proposals = propose_diststyle_all(
        usage, _facts(R2, row_estimate=None), physical, min_cost_share=0.1
    )
    assert len(proposals) == 1
    assert proposals[0].confidence is Confidence.LOW
    assert "row count could not be verified" in proposals[0].rationale
    assert "distribution style could not be read" in proposals[0].rationale


def test_diststyle_all_low_when_only_row_count_is_unreadable():
    usage = [_usage(R2, "id", ColumnRole.JOIN, cost_share=0.5)]
    physical = {R2: _phys(diststyle="EVEN")}
    proposals = propose_diststyle_all(
        usage, _facts(R2, row_estimate=None), physical, min_cost_share=0.1
    )
    assert len(proposals) == 1
    assert proposals[0].confidence is Confidence.LOW
    assert "row count could not be verified" in proposals[0].rationale
    assert "EVEN" in proposals[0].rationale


def test_diststyle_all_low_when_only_diststyle_is_unreadable():
    usage = [_usage(R2, "id", ColumnRole.JOIN, cost_share=0.5)]
    physical = {R2: _phys(diststyle=None)}
    proposals = propose_diststyle_all(
        usage, _facts(R2, row_estimate=1_000), physical, min_cost_share=0.1
    )
    assert len(proposals) == 1
    assert proposals[0].confidence is Confidence.LOW
    assert "distribution style could not be read" in proposals[0].rationale
    assert "1,000 rows" in proposals[0].rationale


def test_diststyle_all_never_reaches_high():
    usage = [_usage(R2, "id", ColumnRole.JOIN, cost_share=0.99)]
    physical = {R2: _phys(diststyle="EVEN")}
    proposals = propose_diststyle_all(
        usage, _facts(R2, row_estimate=100), physical, min_cost_share=0.01
    )
    assert proposals[0].confidence is Confidence.MEDIUM


def test_diststyle_all_medium_cap_rationale_states_the_reason_not_just_the_word():
    usage = [_usage(R2, "id", ColumnRole.JOIN, cost_share=0.5)]
    physical = {R2: _phys(diststyle="EVEN")}
    proposals = propose_diststyle_all(
        usage, _facts(R2, row_estimate=1_000), physical, min_cost_share=0.1
    )
    rationale = proposals[0].rationale
    assert "this rule's row-count ceiling is a heuristic" in rationale
    assert (
        "not a measurement of the storage and write cost this table will actually incur "
        "once replicated" in rationale
    )


def test_diststyle_all_discloses_write_amplification_and_storage_cost():
    usage = [_usage(R2, "id", ColumnRole.JOIN, cost_share=0.5)]
    physical = {R2: _phys(diststyle="EVEN")}
    proposals = propose_diststyle_all(
        usage, _facts(R2, row_estimate=1_000), physical, min_cost_share=0.1
    )
    rationale = proposals[0].rationale
    assert "storage" in rationale.lower()
    assert "node count" in rationale
    assert "replicated to every node" in rationale


def test_diststyle_all_note_discloses_the_full_table_rewrite():
    usage = [_usage(R2, "id", ColumnRole.JOIN, cost_share=0.5)]
    physical = {R2: _phys(diststyle="EVEN")}
    proposals = propose_diststyle_all(
        usage, _facts(R2, row_estimate=1_000), physical, min_cost_share=0.1
    )
    note = proposals[0].note or ""
    assert "rewrites the entire table" in note
    assert "CONCURRENTLY" in note


def test_diststyle_all_discloses_stats_off_as_a_caveat_when_present():
    usage = [_usage(R2, "id", ColumnRole.JOIN, cost_share=0.5)]
    physical = {R2: _phys(diststyle="EVEN", stats_off=33.0)}
    proposals = propose_diststyle_all(
        usage, _facts(R2, row_estimate=1_000), physical, min_cost_share=0.1
    )
    assert "33%" in proposals[0].rationale
    assert "stats_off" in proposals[0].rationale


def test_diststyle_all_omits_stats_off_caveat_when_it_is_zero():
    usage = [_usage(R2, "id", ColumnRole.JOIN, cost_share=0.5)]
    physical = {R2: _phys(diststyle="EVEN", stats_off=0.0)}
    proposals = propose_diststyle_all(
        usage, _facts(R2, row_estimate=1_000), physical, min_cost_share=0.1
    )
    assert "stats_off" not in proposals[0].rationale


# ---------------------------------------------------------------------------
# ADV104 — propose_maintenance
# ---------------------------------------------------------------------------


def test_maintenance_proposes_vacuum_at_high_when_unsorted_meets_the_threshold():
    physical = {R: _phys(unsorted=20.0)}
    proposals = propose_maintenance(physical, _facts(R))
    assert len(proposals) == 1
    p = proposals[0]
    assert p.code == "ADV104"
    assert p.confidence is Confidence.HIGH
    assert p.ddl == 'VACUUM "public"."orders";'
    assert "VACUUM" in p.title


def test_maintenance_no_vacuum_below_the_unsorted_threshold():
    physical = {R: _phys(unsorted=19.99)}
    assert propose_maintenance(physical, _facts(R)) == []


def test_maintenance_no_vacuum_when_unsorted_is_unmeasured():
    physical = {R: _phys(unsorted=None)}
    assert propose_maintenance(physical, _facts(R)) == []


def test_maintenance_proposes_analyze_at_high_when_stats_off_meets_the_threshold():
    physical = {R: _phys(stats_off=20.0)}
    proposals = propose_maintenance(physical, _facts(R))
    assert len(proposals) == 1
    p = proposals[0]
    assert p.code == "ADV104"
    assert p.confidence is Confidence.HIGH
    assert p.ddl == 'ANALYZE "public"."orders";'
    assert "ANALYZE" in p.title


def test_maintenance_no_analyze_below_the_stats_off_threshold():
    physical = {R: _phys(stats_off=19.99)}
    assert propose_maintenance(physical, _facts(R)) == []


def test_maintenance_no_analyze_when_stats_off_is_unmeasured():
    physical = {R: _phys(stats_off=None)}
    assert propose_maintenance(physical, _facts(R)) == []


def test_maintenance_proposes_both_independently_for_the_same_relation():
    physical = {R: _phys(unsorted=50.0, stats_off=60.0)}
    proposals = propose_maintenance(physical, _facts(R))
    assert len(proposals) == 2
    ddls = {p.ddl for p in proposals}
    assert ddls == {'VACUUM "public"."orders";', 'ANALYZE "public"."orders";'}
    assert all(p.confidence is Confidence.HIGH for p in proposals)


def test_maintenance_ignores_relations_with_neither_measurement_stale():
    physical = {R: _phys(unsorted=1.0, stats_off=1.0)}
    assert propose_maintenance(physical, _facts(R)) == []


def test_maintenance_not_gated_by_cost_share_or_workload_usage():
    """ADV104's evidence is a catalog measurement, not workload cost — there is no
    `min_cost_share` parameter at all, and it must fire from `physical` facts alone even
    with no usage/workload data in view."""
    physical = {R: _phys(unsorted=99.0)}
    proposals = propose_maintenance(physical, {})
    assert len(proposals) == 1
    assert proposals[0].evidence["row_estimate"] is None


# ---------------------------------------------------------------------------
# ADV105 — propose_advisor
# ---------------------------------------------------------------------------


def test_advisor_proposal_is_always_high_confidence():
    row = RedshiftAdvisorRow(
        relation=R, rec_type="sort key", current_ddl=None, recommended_ddl="ALTER TABLE x;"
    )
    proposals = propose_advisor([row])
    assert proposals[0].confidence is Confidence.HIGH


def test_advisor_proposal_carries_advisors_own_ddl_verbatim():
    row = RedshiftAdvisorRow(
        relation=R,
        rec_type="sort key",
        current_ddl="ALTER TABLE public.orders ALTER SORTKEY NONE;",
        recommended_ddl='ALTER TABLE public.orders ALTER SORTKEY ("created_at");',
    )
    proposals = propose_advisor([row])
    p = proposals[0]
    assert p.ddl == 'ALTER TABLE public.orders ALTER SORTKEY ("created_at");'
    assert "Current:" in p.rationale
    assert "Recommended:" in p.rationale


def test_advisor_proposal_note_attributes_the_ddl_to_redshift_not_sqlquality():
    row = RedshiftAdvisorRow(
        relation=R, rec_type="sort key", current_ddl=None, recommended_ddl="ALTER TABLE x;"
    )
    proposals = propose_advisor([row])
    note = proposals[0].note or ""
    assert "Amazon Redshift Advisor" in note
    assert "not sqlquality's own analysis" in note


def test_advisor_proposal_title_attributes_to_advisor_not_sqlquality():
    """`title` is the one field the terminal table always renders — pin it directly,
    since `note` alone (already covered above) does not stop `title` or `rationale` from
    independently being rewritten to claim this recommendation as sqlquality's own."""
    row = RedshiftAdvisorRow(
        relation=R, rec_type="sort key", current_ddl=None, recommended_ddl="ALTER TABLE x;"
    )
    proposals = propose_advisor([row])
    title = proposals[0].title
    assert "Amazon Redshift Advisor" in title
    assert "recommends" in title
    assert "sqlquality recommends" not in title.lower()


def test_advisor_proposal_rationale_attributes_to_advisor_not_sqlquality():
    row = RedshiftAdvisorRow(
        relation=R, rec_type="sort key", current_ddl=None, recommended_ddl="ALTER TABLE x;"
    )
    proposals = propose_advisor([row])
    rationale = proposals[0].rationale
    assert rationale.startswith("Amazon Redshift Advisor")
    assert "sqlquality did not generate or verify this recommendation" in rationale
    assert "sqlquality recommends" not in rationale.lower()


def test_advisor_proposal_evidence_carries_a_machine_readable_source():
    """A caller rendering evidence as bare `k=v` pairs (this report's own discipline) still
    needs a signal that this proposal's source is Advisor, not this adapter's inference —
    prose alone (`title`/`rationale`/`note`) is invisible to that renderer."""
    row = RedshiftAdvisorRow(
        relation=R, rec_type="sort key", current_ddl=None, recommended_ddl="ALTER TABLE x;"
    )
    proposals = propose_advisor([row])
    assert proposals[0].evidence["source"] == "amazon_redshift_advisor"


def test_advisor_proposal_handles_missing_current_and_recommended_ddl():
    row = RedshiftAdvisorRow(
        relation=R, rec_type="sort key", current_ddl=None, recommended_ddl=None
    )
    proposals = propose_advisor([row])
    p = proposals[0]
    assert p.ddl is None
    assert "Current:" not in p.rationale
    assert "Recommended:" not in p.rationale


def test_advisor_produces_one_proposal_per_row_in_deterministic_order():
    """Rows are fed in the *opposite* of sorted order — "orders" before "customers" — so
    a mutant that deleted the `sorted(...)` call in `propose_advisor` and simply iterated
    `rows` as given would still fail this assertion. Feeding already-sorted input here
    previously let that mutant survive."""
    rows = [
        RedshiftAdvisorRow(relation=R, rec_type="a", current_ddl=None, recommended_ddl=None),
        RedshiftAdvisorRow(relation=R2, rec_type="b", current_ddl=None, recommended_ddl=None),
    ]
    proposals = propose_advisor(rows)
    # Sorted by (schema, table, rec_type) — both share schema "public", so table order
    # alone decides: "customers" sorts before "orders".
    assert [p.evidence["table"] for p in proposals] == ["customers", "orders"]


# ---------------------------------------------------------------------------
# `_advisor_category` and diststyle parsing helpers
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


def test_diststyle_key_column_parsing_is_case_insensitive():
    """Pins `_DISTSTYLE_KEY_RE`'s `re.IGNORECASE` flag: nothing in this adapter has ever
    observed a live cluster's actual casing for this text (see the module docstring's
    provenance warning), so the parser must not silently assume upper case.
    """
    assert _diststyle_key_column("key(customer_id)") == "customer_id"
    assert _diststyle_key_column("Auto(Key(customer_id))") == "customer_id"


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


def test_diststyle_is_all_requires_no_key_column_even_when_all_is_a_substring():
    """Pins the second conjunct of `_diststyle_is_all` — `and _diststyle_key_column(...)
    is None` — which a real `svv_table_info.diststyle` value never exercises (`ALL` and
    `KEY(...)` are mutually exclusive shapes), so nothing in the ordinary parametrize table
    above can tell a version missing this conjunct apart from one that has it. A synthetic
    diststyle whose key column name itself contains the substring "ALL" forces the two
    checks apart: the naive `"ALL" in diststyle.upper()` alone would say True here, and
    only the second conjunct correctly says this is a keyed style, not DISTSTYLE ALL.
    """
    assert _diststyle_is_all("KEY(all_customers)") is False


@pytest.mark.parametrize(
    ("rec_type", "recommended_ddl", "expected"),
    [
        ("sort key", None, "sortkey"),
        ("Sort Key", None, "sortkey"),
        ("distribution style", "ALTER TABLE x ALTER DISTKEY y;", "distkey"),
        ("distribution style", "ALTER TABLE x ALTER DISTSTYLE ALL;", "diststyle_all"),
        ("something else entirely", None, None),
    ],
)
def test_advisor_category_classification(rec_type, recommended_ddl, expected):
    row = RedshiftAdvisorRow(
        relation=R, rec_type=rec_type, current_ddl=None, recommended_ddl=recommended_ddl
    )
    assert _advisor_category(row) == expected


# ---------------------------------------------------------------------------
# `_quote_ident` — identifier quoting used by every ADV101/102/103/104 statement
# ---------------------------------------------------------------------------


def test_quote_ident_doubles_an_embedded_double_quote():
    """Unquoted, or naively quoted, this would either produce invalid DDL or let an
    identifier break out of its quoting — the same reasoning `postgres.py`'s identical
    helper documents. Nothing in this adapter's own proposal tests happened to exercise a
    quote-containing identifier, so the doubling itself was unpinned here."""
    assert _quote_ident('weird"col') == '"weird""col"'
    assert _quote_ident("plain") == '"plain"'


# ---------------------------------------------------------------------------
# _disclose_advisor_agreement
# ---------------------------------------------------------------------------


def test_agreement_is_disclosed_on_the_matching_sortkey_proposal():
    usage = [_usage(R, "created_at", ColumnRole.RANGE, cost_share=0.5)]
    physical = {R: _phys(sortkey1="status")}
    proposals = propose_sortkey(usage, _facts(R), physical, min_cost_share=0.1)
    advisor_rows = [
        RedshiftAdvisorRow(relation=R, rec_type="sort key", current_ddl=None, recommended_ddl=None)
    ]
    updated = _disclose_advisor_agreement(proposals, advisor_rows)
    assert len(updated) == 1
    assert "Advisor independently recommends" in updated[0].rationale
    # Agreement must not raise confidence past the documented cap.
    assert updated[0].confidence is Confidence.MEDIUM


def test_agreement_does_not_fire_for_an_unrelated_relation():
    usage = [_usage(R, "created_at", ColumnRole.RANGE, cost_share=0.5)]
    physical = {R: _phys(sortkey1="status")}
    proposals = propose_sortkey(usage, _facts(R), physical, min_cost_share=0.1)
    advisor_rows = [
        RedshiftAdvisorRow(relation=R2, rec_type="sort key", current_ddl=None, recommended_ddl=None)
    ]
    updated = _disclose_advisor_agreement(proposals, advisor_rows)
    assert "Advisor independently recommends" not in updated[0].rationale


def test_agreement_does_not_fire_for_a_different_category():
    """A distribution-style Advisor row must not be read as agreeing with a SORTKEY
    proposal on the same relation."""
    usage = [_usage(R, "created_at", ColumnRole.RANGE, cost_share=0.5)]
    physical = {R: _phys(sortkey1="status")}
    proposals = propose_sortkey(usage, _facts(R), physical, min_cost_share=0.1)
    advisor_rows = [
        RedshiftAdvisorRow(
            relation=R,
            rec_type="distribution style",
            current_ddl=None,
            recommended_ddl="ALTER TABLE x ALTER DISTKEY y;",
        )
    ]
    updated = _disclose_advisor_agreement(proposals, advisor_rows)
    assert "Advisor independently recommends" not in updated[0].rationale


def test_agreement_fires_for_diststyle_all_but_not_distkey_proposal():
    usage = [_usage(R2, "id", ColumnRole.JOIN, cost_share=0.5)]
    physical = {R2: _phys(diststyle="EVEN")}
    distkey_proposals = propose_distkey(usage, _facts(R2), physical, min_cost_share=0.1)
    diststyle_proposals = propose_diststyle_all(
        usage, _facts(R2, row_estimate=1_000), physical, min_cost_share=0.1
    )
    advisor_rows = [
        RedshiftAdvisorRow(
            relation=R2,
            rec_type="distribution style",
            current_ddl=None,
            recommended_ddl="ALTER TABLE x ALTER DISTSTYLE ALL;",
        )
    ]
    updated_distkey = _disclose_advisor_agreement(distkey_proposals, advisor_rows)
    updated_diststyle = _disclose_advisor_agreement(diststyle_proposals, advisor_rows)
    assert "Advisor independently recommends" not in updated_distkey[0].rationale
    assert "Advisor independently recommends" in updated_diststyle[0].rationale


def test_agreement_is_a_noop_with_no_advisor_rows():
    usage = [_usage(R, "created_at", ColumnRole.RANGE, cost_share=0.5)]
    physical = {R: _phys(sortkey1="status")}
    proposals = propose_sortkey(usage, _facts(R), physical, min_cost_share=0.1)
    updated = _disclose_advisor_agreement(proposals, [])
    assert updated == proposals


def test_agreement_does_not_touch_advisor_proposals_or_maintenance_proposals():
    """Agreement disclosure only rewrites ADV101/102/103 rationale — ADV104 and ADV105
    proposals must pass through unmodified, since they carry no `_PROPOSAL_CATEGORY`."""
    maintenance = propose_maintenance({R: _phys(unsorted=50.0)}, _facts(R))
    advisor = propose_advisor(
        [
            RedshiftAdvisorRow(
                relation=R, rec_type="sort key", current_ddl=None, recommended_ddl=None
            )
        ]
    )
    advisor_rows = [
        RedshiftAdvisorRow(relation=R, rec_type="sort key", current_ddl=None, recommended_ddl=None)
    ]
    updated = _disclose_advisor_agreement(maintenance + advisor, advisor_rows)
    assert updated == maintenance + advisor


# ---------------------------------------------------------------------------
# _collapse_diststyle_all_over_distkey — ADV103 subsumes ADV102 for the same relation
# ---------------------------------------------------------------------------


def _distkey_proposal(relation=R2, column="id", confidence=Confidence.MEDIUM):
    usage = [_usage(relation, column, ColumnRole.JOIN, cost_share=0.5)]
    physical = {relation: _phys(diststyle="EVEN")}
    proposals = propose_distkey(usage, _facts(relation), physical, min_cost_share=0.1)
    assert len(proposals) == 1
    assert proposals[0].confidence is confidence
    return proposals[0]


def _diststyle_all_proposal(relation=R2):
    usage = [_usage(relation, "id", ColumnRole.JOIN, cost_share=0.5)]
    physical = {relation: _phys(diststyle="EVEN")}
    proposals = propose_diststyle_all(
        usage, _facts(relation, row_estimate=1_000), physical, min_cost_share=0.1
    )
    assert len(proposals) == 1
    return proposals[0]


def test_collapse_drops_distkey_when_diststyle_all_fires_for_the_same_relation():
    distkey = _distkey_proposal()
    diststyle_all = _diststyle_all_proposal()
    collapsed = _collapse_diststyle_all_over_distkey([distkey, diststyle_all])
    codes = [p.code for p in collapsed]
    assert codes == ["ADV103"]


def test_collapse_discloses_the_withheld_distkey_in_the_survivors_rationale():
    distkey = _distkey_proposal(column="customer_id")
    diststyle_all = _diststyle_all_proposal()
    collapsed = _collapse_diststyle_all_over_distkey([distkey, diststyle_all])
    rationale = collapsed[0].rationale
    assert "ADV102 also proposed a DISTKEY on customer_id" in rationale
    assert "withheld" in rationale
    assert "strictly subsumes any single-column DISTKEY choice" in rationale


def test_collapse_folds_the_withheld_distkeys_own_caveats_into_the_survivor():
    """The fold — `if fresh: addition += ...` — is the substance of the collapse, not a
    flourish, and deleting it left the whole suite green because the existing disclosure
    test asserts only the fixed `addition` string.

    ADV102 is the *only* proposal that carries the `skew_rows` caveat and its own
    DISTKEY-specific MEDIUM-cap explanation. Once it is withheld, the surviving ADV103 is
    the sole remaining proposal for a full-table rewrite of this relation — so without the
    fold the operator loses both the reason the withheld strategy was withheld and the one
    measurement that speaks to how risky a distribution change on this table is.
    """
    usage = [_usage(R2, "id", ColumnRole.JOIN, cost_share=0.5)]
    physical = {R2: _phys(diststyle="EVEN", skew_rows=3.75)}
    (distkey,) = propose_distkey(usage, _facts(R2), physical, min_cost_share=0.1)
    (diststyle_all,) = propose_diststyle_all(
        usage, _facts(R2, row_estimate=1_000), physical, min_cost_share=0.1
    )
    # Control: these two sentences exist on ADV102 and on nothing else, so finding them on
    # the survivor can only be the fold's doing.
    assert "skew_rows is 3.75" in distkey.rationale
    assert "skew_rows" not in diststyle_all.rationale
    assert "good or catastrophic" not in diststyle_all.rationale

    (survivor,) = _collapse_diststyle_all_over_distkey([distkey, diststyle_all])
    assert survivor.code == "ADV103"
    assert "skew_rows is 3.75" in survivor.rationale
    assert "not the skew this DISTKEY would produce" in survivor.rationale
    assert "good or catastrophic" in survivor.rationale


def test_collapse_leaves_distkey_alone_when_diststyle_all_does_not_fire_for_it():
    """A relation with only ADV102 (e.g. it failed ADV103's row-count ceiling) must be
    untouched — this function only ever removes an ADV102 that has a matching ADV103 for
    the *same* relation."""
    distkey_r = _distkey_proposal(relation=R)
    diststyle_all_r2 = _diststyle_all_proposal(relation=R2)
    collapsed = _collapse_diststyle_all_over_distkey([distkey_r, diststyle_all_r2])
    codes = {p.code for p in collapsed}
    assert codes == {"ADV102", "ADV103"}


def test_collapse_is_a_noop_with_no_diststyle_all_proposals():
    distkey = _distkey_proposal()
    collapsed = _collapse_diststyle_all_over_distkey([distkey])
    assert collapsed == [distkey]


def test_collapse_leaves_diststyle_all_alone_with_no_matching_distkey():
    diststyle_all = _diststyle_all_proposal()
    collapsed = _collapse_diststyle_all_over_distkey([diststyle_all])
    assert collapsed == [diststyle_all]


# ---------------------------------------------------------------------------
# _skipped_for_physical_gap — Spectrum/empty-table ambiguity, counted rather than silent
# ---------------------------------------------------------------------------


def test_skipped_for_physical_gap_counts_a_relation_with_a_candidate_role_and_no_facts():
    usage = [_usage(R, "created_at", ColumnRole.RANGE, cost_share=0.5)]
    assert _skipped_for_physical_gap(usage, {}) == 1


def test_skipped_for_physical_gap_counts_join_and_equality_roles_too():
    usage = [
        _usage(R, "customer_id", ColumnRole.JOIN, cost_share=0.5),
        _usage(R2, "status", ColumnRole.EQUALITY, cost_share=0.5),
    ]
    assert _skipped_for_physical_gap(usage, {}) == 2


def test_skipped_for_physical_gap_ignores_a_relation_with_facts_present():
    usage = [_usage(R, "created_at", ColumnRole.RANGE, cost_share=0.5)]
    physical = {R: _phys(sortkey1="status")}
    assert _skipped_for_physical_gap(usage, physical) == 0


def test_skipped_for_physical_gap_ignores_a_relation_with_no_candidate_role():
    """A relation with only GROUP/SORT usage was never going to get a SORTKEY/DISTKEY/
    DISTSTYLE ALL proposal even with facts present, so it must not inflate this count."""
    usage = [_usage(R, "category", ColumnRole.GROUP, cost_share=0.5)]
    assert _skipped_for_physical_gap(usage, {}) == 0


# ---------------------------------------------------------------------------
# RedshiftWorkloadAdapter.propose() — dispatcher wiring
# ---------------------------------------------------------------------------


class _AdvisorQuerier:
    """A minimal `Querier` that answers only CAP_ADVISOR, for testing `propose()`'s own
    wiring rather than the pure rule functions above (already covered directly)."""

    def __init__(self, rows=(), *, fail=False):
        self._rows = rows
        self._fail = fail
        self.calls = []

    def __call__(self, sql, params):
        self.calls.append((sql, params))
        if self._fail:
            raise RuntimeError("permission denied for svv_alter_table_recommendations")
        return self._rows


def _bare_aggregation(usage, tables) -> Aggregation:
    return Aggregation(
        usage=tuple(usage), total_cost_ms=0.0, skipped_unqualifiable=0, tables=frozenset(tables)
    )


def test_propose_no_longer_raises_not_implemented_error():
    adapter = RedshiftWorkloadAdapter(querier=_AdvisorQuerier())
    proposals = adapter.propose(
        _bare_aggregation([], []),
        {},
        Workload(stats=(), window_description="w"),
        min_cost_share=0.1,
    )
    assert proposals == []


def test_propose_wires_sortkey_and_maintenance_and_sorts_by_confidence():
    adapter = RedshiftWorkloadAdapter(querier=_AdvisorQuerier())
    adapter.physical_facts = {
        R: _phys(sortkey1="status", unsorted=50.0),
    }
    usage = [_usage(R, "created_at", ColumnRole.RANGE, cost_share=0.5)]
    proposals = adapter.propose(
        _bare_aggregation(usage, [R]),
        _facts(R),
        Workload(stats=(), window_description="w"),
        min_cost_share=0.1,
    )
    codes = [p.code for p in proposals]
    assert "ADV101" in codes
    assert "ADV104" in codes
    # HIGH-confidence ADV104 (VACUUM) must sort ahead of MEDIUM-confidence ADV101.
    assert codes.index("ADV104") < codes.index("ADV101")


def test_propose_fetches_advisor_rows_scoped_to_the_requested_relations():
    querier = _AdvisorQuerier(rows=[("db", "public", "orders", "sort key", None, "ALTER TABLE x;")])
    adapter = RedshiftWorkloadAdapter(querier=querier)
    adapter.schemas = ("public",)
    proposals = adapter.propose(
        _bare_aggregation([], [R]),
        {},
        Workload(stats=(), window_description="w"),
        min_cost_share=0.1,
    )
    assert len(proposals) == 1
    assert proposals[0].code == "ADV105"
    assert len(querier.calls) == 1
    _sql, params = querier.calls[0]
    assert params == (["public"], ["orders"])


def test_propose_drops_advisor_rows_for_relations_not_in_scope():
    """Over-fetch guard: the statement filters on bare table names, so a same-named table
    in a schema that was not requested must not leak into the returned proposals."""
    querier = _AdvisorQuerier(
        rows=[("db", "other_schema", "orders", "sort key", None, "ALTER TABLE x;")]
    )
    adapter = RedshiftWorkloadAdapter(querier=querier)
    proposals = adapter.propose(
        _bare_aggregation([], [R]),
        {},
        Workload(stats=(), window_description="w"),
        min_cost_share=0.1,
    )
    assert proposals == []


def test_propose_records_a_denied_advisor_capability_in_degraded_rather_than_raising():
    adapter = RedshiftWorkloadAdapter(querier=_AdvisorQuerier(fail=True))
    proposals = adapter.propose(
        _bare_aggregation([], [R]),
        {},
        Workload(stats=(), window_description="w"),
        min_cost_share=0.1,
    )
    assert proposals == []
    assert len(adapter.degraded) == 1
    assert adapter.degraded[0][0] == CAP_ADVISOR


def test_propose_disclosed_agreement_survives_the_full_dispatcher():
    querier = _AdvisorQuerier(rows=[("db", "public", "orders", "sort key", None, "ALTER TABLE x;")])
    adapter = RedshiftWorkloadAdapter(querier=querier)
    adapter.schemas = ("public",)
    adapter.physical_facts = {R: _phys(sortkey1="status")}
    usage = [_usage(R, "created_at", ColumnRole.RANGE, cost_share=0.5)]
    proposals = adapter.propose(
        _bare_aggregation(usage, [R]),
        _facts(R),
        Workload(stats=(), window_description="w"),
        min_cost_share=0.1,
    )
    by_code = {p.code: p for p in proposals}
    assert "ADV101" in by_code and "ADV105" in by_code
    assert "Advisor independently recommends" in by_code["ADV101"].rationale
    # The Advisor row must still stand as its own, separate, unmodified proposal.
    assert by_code["ADV105"].confidence is Confidence.HIGH


def test_propose_wires_distkey_when_the_table_is_too_large_for_diststyle_all():
    """Isolates ADV102 in the full dispatcher: the table is over ADV103's row-count
    ceiling, so ADV103 does not fire and cannot mask ADV102's own wiring — removing
    `propose_distkey(...)` from `propose()` must redden this test on its own."""
    adapter = RedshiftWorkloadAdapter(querier=_AdvisorQuerier())
    adapter.physical_facts = {R: _phys(diststyle="EVEN")}
    usage = [_usage(R, "customer_id", ColumnRole.JOIN, cost_share=0.5)]
    proposals = adapter.propose(
        _bare_aggregation(usage, [R]),
        _facts(R, row_estimate=MAX_ROWS_FOR_DISTSTYLE_ALL + 1),
        Workload(stats=(), window_description="w"),
        min_cost_share=0.1,
    )
    codes = [p.code for p in proposals]
    assert codes == ["ADV102"]


def test_propose_wires_diststyle_all_and_suppresses_distkey_for_the_same_relation():
    """Isolates ADV103 in the full dispatcher, and proves the collapse (finding 4) runs
    end to end: a small, hot-join dimension makes both ADV102 and ADV103 fire from their
    own rule functions, but only ADV103 must survive `propose()`. Removing
    `propose_diststyle_all(...)` from `propose()` reddens this test two ways — "ADV103" no
    longer appears, and "ADV102" reappears because nothing is left to suppress it."""
    adapter = RedshiftWorkloadAdapter(querier=_AdvisorQuerier())
    adapter.physical_facts = {R2: _phys(diststyle="EVEN")}
    usage = [_usage(R2, "id", ColumnRole.JOIN, cost_share=0.5)]
    proposals = adapter.propose(
        _bare_aggregation(usage, [R2]),
        _facts(R2, row_estimate=1_000),
        Workload(stats=(), window_description="w"),
        min_cost_share=0.1,
    )
    codes = [p.code for p in proposals]
    assert codes == ["ADV103"]
    assert "ADV102 also proposed a DISTKEY" in proposals[0].rationale


def test_propose_collapses_before_disclosing_agreement():
    """`propose()`'s documented pass order is load-bearing: reversing it fabricates an
    Advisor agreement Advisor never made.

    The fixture is the worst case for the reversal — a small, hot-join dimension, so ADV102
    and ADV103 both fire for the same relation and the collapse withholds ADV102, plus one
    Advisor row whose category is `distkey` (a distribution-key recommendation, not
    `DISTSTYLE ALL`). Shipped order: the collapse runs first, ADV102 is gone, and the
    agreement finds no ADV102 to attach to — ADV103's category is `diststyle_all`, which
    Advisor did *not* recommend, so no agreement is claimed. Reversed: the agreement
    sentence lands on ADV102, the collapse then folds ADV102's fresh sentences (including
    that one) into the surviving ADV103, and ADV103 claims Advisor agrees with a `DISTSTYLE
    ALL` change Advisor never recommended. Presenting Advisor's opinion as ours is the one
    thing ADV105's whole design exists to prevent, and it is invisible to every other test
    because both orders produce the same *set* of codes.
    """
    querier = _AdvisorQuerier(
        rows=[
            (
                "db",
                R2.schema,
                R2.table,
                "distribution key",
                None,
                f'ALTER TABLE "{R2.schema}"."{R2.table}" ALTER DISTKEY "id";',
            )
        ]
    )
    adapter = RedshiftWorkloadAdapter(querier=querier)
    adapter.schemas = (R2.schema,)
    adapter.physical_facts = {R2: _phys(diststyle="EVEN")}
    usage = [_usage(R2, "id", ColumnRole.JOIN, cost_share=0.5)]
    proposals = adapter.propose(
        _bare_aggregation(usage, [R2]),
        _facts(R2, row_estimate=1_000),
        Workload(stats=(), window_description="w"),
        min_cost_share=0.1,
    )
    by_code = {p.code: p for p in proposals}
    # The collapse ran: ADV102 is withheld, ADV103 survives, ADV105 relays Advisor's row.
    assert set(by_code) == {"ADV103", "ADV105"}
    assert "ADV102 also proposed a DISTKEY" in by_code["ADV103"].rationale
    # And the surviving proposal claims no agreement, because there is none to claim.
    assert "independently recommends the same kind of change" not in by_code["ADV103"].rationale


def test_propose_fetches_advisor_rows_for_relations_reached_only_through_table_facts():
    """ADV105's scope must not be narrower than ADV104's.

    `cli.py` fetches table facts for `aggregation.tables | star_tables(...)`, so a relation
    reached only by a `SELECT *` arrives in `facts`/`physical_facts` without ever appearing
    in `aggregation.tables` — and ADV104 (which iterates `physical_facts`) will happily
    propose VACUUM/ANALYZE for it. Scoping the Advisor query to `aggregation.tables` alone
    meant that same relation could never get an ADV105 relay or an agreement disclosure, so
    the cluster's own opinion about it was dropped on the floor. The two rules now see the
    same set.
    """
    querier = _AdvisorQuerier(
        rows=[("db", R2.schema, R2.table, "sort key", None, "ALTER TABLE x;")]
    )
    adapter = RedshiftWorkloadAdapter(querier=querier)
    adapter.schemas = (R2.schema,)
    adapter.physical_facts = {R2: _phys(unsorted=50.0)}
    proposals = adapter.propose(
        # R2 is deliberately absent from `aggregation.tables` — it is only in `facts`.
        _bare_aggregation([], []),
        _facts(R2),
        Workload(stats=(), window_description="w"),
        min_cost_share=0.1,
    )
    codes = {p.code for p in proposals}
    assert "ADV104" in codes, "control: ADV104 already reaches a star-only relation"
    assert "ADV105" in codes
    _sql, params = querier.calls[0]
    assert params == ([R2.schema], [R2.table])


def test_propose_discloses_the_physical_facts_gap_in_degraded():
    """Finding 5: a relation with a hot predicate but no `physical_facts` entry must not
    simply vanish — `propose()` must count it and disclose the count through the same
    `self.degraded` channel a denied capability uses, so it reaches the coverage line, the
    JSON payload and the markdown report."""
    adapter = RedshiftWorkloadAdapter(querier=_AdvisorQuerier())
    adapter.physical_facts = {}
    usage = [
        _usage(R, "created_at", ColumnRole.RANGE, cost_share=0.5),
        _usage(R2, "id", ColumnRole.JOIN, cost_share=0.5),
    ]
    adapter.propose(
        _bare_aggregation(usage, [R, R2]),
        {},
        Workload(stats=(), window_description="w"),
        min_cost_share=0.1,
    )
    gaps = [reason for cap, reason in adapter.degraded if cap == DEGRADATION_PHYSICAL_FACTS_GAP]
    assert len(gaps) == 1
    assert "2 relation(s)" in gaps[0]
    assert "Spectrum" in gaps[0]


def test_propose_does_not_disclose_a_physical_facts_gap_when_there_is_none():
    """Guards the test above: with `physical_facts` covering every relation involved, no
    `DEGRADATION_PHYSICAL_FACTS_GAP` entry should appear at all."""
    adapter = RedshiftWorkloadAdapter(querier=_AdvisorQuerier())
    adapter.physical_facts = {R: _phys(sortkey1="status")}
    usage = [_usage(R, "created_at", ColumnRole.RANGE, cost_share=0.5)]
    adapter.propose(
        _bare_aggregation(usage, [R]),
        _facts(R),
        Workload(stats=(), window_description="w"),
        min_cost_share=0.1,
    )
    assert not any(cap == DEGRADATION_PHYSICAL_FACTS_GAP for cap, _ in adapter.degraded)


# ---------------------------------------------------------------------------
# render_ddl (Task 7) — a far louder header than Postgres's, ADV105 attribution in the
# file itself, and the same hostile-identifier protection Postgres's render_ddl gives.
# ---------------------------------------------------------------------------


def _rewrite_proposal(code="ADV101", ddl='ALTER TABLE "public"."orders" ALTER SORTKEY ("c");'):
    return Proposal(
        code=code,
        title=f"Consider a change on {R}",
        rationale="hot predicate",
        evidence={"schema": R.schema, "table": R.table, "cost_share": 0.5},
        confidence=Confidence.MEDIUM,
        ddl=ddl,
        note="ALTER SORTKEY rewrites the entire table: ... There is no CONCURRENTLY equivalent.",
    )


def _advisor_render_proposal():
    return Proposal(
        code="ADV105",
        title="Amazon Redshift Advisor recommends a sort key change",
        rationale="Amazon Redshift Advisor recommends this change based on its own analysis.",
        evidence={"schema": R.schema, "table": R.table, "source": "amazon_redshift_advisor"},
        confidence=Confidence.HIGH,
        ddl='ALTER TABLE "public"."orders" ALTER SORTKEY ("created_at");',
        note=(
            "Source: Amazon Redshift Advisor, not sqlquality's own analysis — this "
            "statement was generated by Redshift itself and is relayed verbatim."
        ),
    )


def _maintenance_render_proposal():
    return Proposal(
        code="ADV104",
        title=f"Run VACUUM on {R}",
        rationale="unsorted region is large",
        evidence={"schema": R.schema, "table": R.table},
        confidence=Confidence.HIGH,
        ddl='VACUUM "public"."orders";',
        note="VACUUM is heavy ... it does not need a maintenance-window lock.",
    )


def test_render_ddl_header_calls_out_adv101_102_103_as_full_table_rewrites():
    script = RedshiftWorkloadAdapter().render_ddl([])
    assert "ADV101" in script
    assert "ADV102" in script
    assert "ADV103" in script
    assert "REWRITE THE ENTIRE TABLE" in script


def test_render_ddl_header_says_there_is_no_concurrently_equivalent():
    """Postgres's header can recommend CONCURRENTLY for exactly this hazard; Redshift's
    header must say plainly that no such escape exists here."""
    script = RedshiftWorkloadAdapter().render_ddl([])
    assert "CONCURRENTLY" in script
    assert "NO CONCURRENTLY EQUIVALENT" in script


def test_render_ddl_header_says_schedule_a_maintenance_window():
    """Pinned as one contiguous phrase, not merely "the word 'maintenance' occurs
    somewhere" — the ADV104 sentence a few lines down also says "maintenance window" for
    the opposite reason (it does *not* need one), so a weaker check would still pass with
    the DANGER section's own "schedule a maintenance window" instruction deleted."""
    script = RedshiftWorkloadAdapter().render_ddl([])
    assert "SCHEDULE each one for a maintenance" in script
    assert "window; do not run it ad hoc" in script


def test_render_ddl_header_names_adv104_as_the_one_non_rewrite():
    script = RedshiftWorkloadAdapter().render_ddl([])
    assert "ADV104" in script
    # The header's ADV104 sentence must appear near language distinguishing it from a
    # rewrite, not merely mention the code in passing.
    header = script.split("-- ADV101")[0] if "-- ADV101" in script else script
    assert (
        "not a" in header.lower()
        or "not the" in header.lower()
        or "only statement" in header.lower()
    )


def test_render_ddl_header_is_louder_than_postgres_header():
    """The one property this whole task exists to deliver: the Redshift header must say
    materially more than Postgres's does about the danger of the DDL below it."""
    from sqlquality.workload.postgres import PostgresWorkloadAdapter

    postgres_header = PostgresWorkloadAdapter().render_ddl([])
    redshift_header = RedshiftWorkloadAdapter().render_ddl([])
    for word in ("REWRITE", "HOURS", "DANGER", "SCHEDULE"):
        assert word not in postgres_header
        assert word in redshift_header


def test_render_ddl_emits_the_statement_with_code_confidence_and_cost_share():
    script = RedshiftWorkloadAdapter().render_ddl([_rewrite_proposal()])
    assert 'ALTER TABLE "public"."orders" ALTER SORTKEY ("c");' in script
    assert "-- ADV101 [medium, 50.0% of workload cost]" in script


def test_render_ddl_prints_the_proposals_note_above_its_statement():
    proposal = _rewrite_proposal()
    script = RedshiftWorkloadAdapter().render_ddl([proposal])
    assert script.index("-- ALTER SORTKEY rewrites the entire table") < script.index(proposal.ddl)


def test_render_ddl_skips_a_proposal_with_no_ddl():
    proposal = Proposal(
        code="ADV104",
        title="advisory only",
        rationale="r",
        evidence={},
        confidence=Confidence.HIGH,
        ddl=None,
    )
    script = RedshiftWorkloadAdapter().render_ddl([proposal])
    assert "advisory only" not in script


def test_render_ddl_with_no_proposals_still_explains_itself():
    script = RedshiftWorkloadAdapter().render_ddl([])
    assert "No DDL proposals" in script


def test_render_ddl_tolerates_a_missing_or_non_numeric_cost_share():
    for evidence in ({}, {"cost_share": "nope"}, {"cost_share": True}):
        proposal = Proposal(
            code="ADV104",
            title="t",
            rationale="r",
            evidence=evidence,
            confidence=Confidence.HIGH,
            ddl='VACUUM "public"."orders";',
        )
        script = RedshiftWorkloadAdapter().render_ddl([proposal])
        assert "-- ADV104 [high]" in script
        assert "%" not in script.split("-- ADV104")[1].split("\n")[0]


def _uncommented(script: str) -> list[str]:
    """Every non-blank line that is not a `--` comment. See the identical helper in
    `tests/test_workload_rules.py` — the one property this whole file exists to prove."""
    return [line for line in script.splitlines() if line.strip() and not line.startswith("--")]


def test_render_ddl_never_emits_a_bare_uncommented_line_for_a_hostile_identifier():
    """The same protection `PostgresWorkloadAdapter.render_ddl` gives, reused rather than
    reimplemented: a title or ddl carrying an embedded newline, `\\r`, `--` or `;` must never
    leave a bare, executable-looking line in a Redshift script either."""
    hostile = 'ALTER TABLE "public"."orders"\nDROP TABLE users; --\n ALTER SORTKEY ("c");'
    proposal = Proposal(
        code="ADV101",
        title="line1\nline2 -- injected; DROP TABLE users;\r",
        rationale="r",
        evidence={"schema": "public", "table": "orders", "cost_share": 0.4},
        confidence=Confidence.MEDIUM,
        ddl=hostile,
    )
    script = RedshiftWorkloadAdapter().render_ddl([proposal])
    assert "NOT RENDERED" in script
    assert _uncommented(script) == [], script


@pytest.mark.parametrize(
    ("char", "name"), EXTRA_LINE_BREAKS, ids=[name for _char, name in EXTRA_LINE_BREAKS]
)
def test_every_splitlines_line_break_in_an_identifier_reaches_the_not_rendered_fallback(char, name):
    """The same hole, closed on this adapter too, and pinned here rather than only on the
    Postgres side: the guard is shared code (`_has_line_break`, imported from `postgres.py`),
    but the two `render_ddl` implementations call it in two separate places, so a fix applied
    to one is not a fix applied to both. The codepoint list is imported from the Postgres
    renderer's tests deliberately — two copies would be free to drift apart, and this file's
    whole reason for asserting it again is that the *call site* is different, not the set.
    """
    ddl = f'ALTER TABLE "public"."or{char}ders" ALTER SORTKEY ("created_at");'
    assert len(ddl.splitlines()) == 2, f"{name} is not a splitlines boundary"
    proposal = Proposal(
        code="ADV101",
        title="Consider SORTKEY on public.orders(created_at)",
        rationale="r",
        evidence={"schema": "public", "table": "orders", "cost_share": 0.4},
        confidence=Confidence.MEDIUM,
        ddl=ddl,
    )
    script = RedshiftWorkloadAdapter().render_ddl([proposal])
    assert "NOT RENDERED" in script, name
    assert _uncommented(script) == [], script


def test_render_ddl_emits_a_pre_commented_multiline_block_verbatim():
    """`_is_fully_commented`, reused rather than reimplemented: a `ddl` that already reads
    as a `--`-commented, multi-line disclosure on every line is not the identifier-with-a-
    line-break hazard and must be emitted as-is, with its usual header."""
    block = "-- ADV999: some future multi-line disclosure\n--   line two\n--   line three"
    proposal = Proposal(
        code="ADV999",
        title="hypothetical config-shaped proposal",
        rationale="r",
        evidence={"cost_share": 0.3},
        confidence=Confidence.HIGH,
        ddl=block,
    )
    script = RedshiftWorkloadAdapter().render_ddl([proposal])
    assert "NOT RENDERED" not in script
    assert "-- -- ADV999" not in script, "must not be double-commented"
    assert script.count("some future multi-line disclosure") == 1


def test_render_ddl_marks_adv105_in_its_own_header_line_not_only_in_note():
    """ADV105's DDL is Redshift Advisor's, not sqlquality's, and that must be visible in the
    per-statement header line itself — someone skimming only header lines, never the full
    `note` text, must still be able to tell an Advisor statement apart from an ADV101/102/103
    statement sqlquality inferred."""
    script = RedshiftWorkloadAdapter().render_ddl([_advisor_render_proposal()])
    header_line = next(line for line in script.splitlines() if line.startswith("-- ADV105 ["))
    assert "Advisor" in header_line
    assert "not sqlquality" in header_line


def test_render_ddl_does_not_mark_non_advisor_codes_with_the_advisor_attribution():
    """Guards the test above: the attribution marker must be specific to ADV105, not a
    blanket suffix every statement gets regardless of its actual source."""
    script = RedshiftWorkloadAdapter().render_ddl([_rewrite_proposal(code="ADV101")])
    header_line = next(line for line in script.splitlines() if line.startswith("-- ADV101 ["))
    assert "Advisor" not in header_line


def test_render_ddl_full_script_with_all_five_codes_marks_only_advisor():
    """An end-to-end sanity check with one proposal per code: the rewrite trio, the
    maintenance rule, and Advisor together in one script, each still individually correct."""
    proposals = [
        _rewrite_proposal(code="ADV101"),
        _rewrite_proposal(code="ADV102", ddl='ALTER TABLE "public"."orders" ALTER DISTKEY "c";'),
        _rewrite_proposal(code="ADV103", ddl='ALTER TABLE "public"."orders" ALTER DISTSTYLE ALL;'),
        _maintenance_render_proposal(),
        _advisor_render_proposal(),
    ]
    script = RedshiftWorkloadAdapter().render_ddl(proposals)
    for code in ("ADV101", "ADV102", "ADV103", "ADV104", "ADV105"):
        assert f"-- {code} [" in script
    advisor_lines = [line for line in script.splitlines() if "Advisor" in line and "[" in line]
    assert len(advisor_lines) == 1
    assert advisor_lines[0].startswith("-- ADV105")


# ---------------------------------------------------------------------------
# dbt enrichment interaction (Task 7) — a Redshift table-rewrite proposal on a dbt-managed
# relation. Batch 3a's `enrich_proposals` keys its rewrite-to-config path on a `CREATE
# INDEX` DDL prefix, which never matches Redshift's `ALTER TABLE` statements — but its
# generic fallback branch (built, per its own docstring, for exactly "some other statement
# for a relation dbt owns") already recognises anything that is neither `CREATE INDEX` nor
# `DROP INDEX` and attaches a dbt warning to both `rationale` and `note`. That branch had
# zero test coverage before this task — no Postgres rule's DDL ever reached it — so these
# tests are the first proof it actually behaves as its docstring claims for Redshift's
# proposals, and a regression test for the note-clobbering bug this task found and fixed in
# `sqlquality/workload/dbt.py` (`_prepend_note`).
# ---------------------------------------------------------------------------


def _dbt_context_for(relation: Relation, *, materialized: str = "table") -> DbtContext:
    model = ModelNode(
        unique_id=f"model.proj.{relation.table}",
        name=relation.table,
        resource_type="model",
        materialized=materialized,
        compiled_code=None,
        relation_name=f"{relation.schema}.{relation.table}",
        depends_on=[],
        config={},
    )
    return DbtContext(models={relation: model}, dropped_collisions=0)


def test_dbt_warning_is_attached_to_a_redshift_rewrite_proposal_on_a_dbt_managed_relation():
    """A dbt-managed Redshift model must not get a silent table-rewrite proposal: the next
    `dbt run` can undo hours of rewrite work with no warning at all if this path is broken.
    """
    proposal = _rewrite_proposal(code="ADV101")
    context = _dbt_context_for(R)
    (enriched,) = enrich_proposals([proposal], context)
    assert enriched.ddl == proposal.ddl  # the statement itself is untouched, only disclosed
    assert "dbt model" in enriched.rationale
    assert "may not outlive the next rebuild" in enriched.rationale
    assert "dbt WARNING" in (enriched.note or "")
    assert "reapply" in (enriched.note or "").lower()


def test_dbt_enrichment_leaves_a_non_dbt_relation_proposal_untouched():
    """Control for the test above: a relation dbt does not manage must not be warned about
    at all — this is not a blanket suffix, it depends on `DbtContext.model_for` actually
    finding a match."""
    proposal = _rewrite_proposal(code="ADV101")
    context = _dbt_context_for(R2)  # a *different* relation
    (enriched,) = enrich_proposals([proposal], context)
    assert enriched.rationale == proposal.rationale
    assert enriched.note == proposal.note


def test_dbt_enrichment_preserves_the_advisor_attribution_note_for_a_dbt_managed_relation():
    """Regression test for a note-clobbering bug this task found: `enrich_proposals`'s
    generic fallback used to build `note` as only the dbt warning, discarding ADV105's own
    Amazon-Redshift-Advisor attribution outright. Since `render_ddl` never emits `rationale`,
    that attribution existed *only* in `note` — so a dbt-managed relation that Advisor also
    has a recommendation for would have shipped a DDL script whose only remaining trace of
    the statement's source was a per-statement header marker, with the descriptive sentence
    silently gone. Both must survive: the dbt warning is additive, not a replacement.
    """
    proposal = _advisor_render_proposal()
    context = _dbt_context_for(R)
    (enriched,) = enrich_proposals([proposal], context)
    assert "Amazon Redshift Advisor" in (enriched.note or "")
    assert "not sqlquality's own analysis" in (enriched.note or "")
    assert "dbt WARNING" in (enriched.note or "")


def test_dbt_enrichment_preserved_note_still_renders_safely_in_the_ddl_script():
    """The two notes concatenated by `_prepend_note` must still pass through `render_ddl`'s
    hostile-identifier guard cleanly — this is a integration check that the fix does not
    itself introduce a bare line."""
    proposal = _advisor_render_proposal()
    context = _dbt_context_for(R)
    (enriched,) = enrich_proposals([proposal], context)
    script = RedshiftWorkloadAdapter().render_ddl([enriched])
    assert _uncommented(script) == [enriched.ddl]


def test_dbt_enrichment_preserves_an_existing_note_on_a_drop_index_proposal_too():
    """`_prepend_note` fixes the same note-clobbering shape on the `DROP INDEX` branch of
    `dbt.py`'s `_classify`, not only the generic fallback ADV104/ADV105 land in. No Redshift
    rule emits `DROP INDEX` and no Postgres rule sets `note` before reaching that branch
    today, so this uses a synthetic, engine-agnostic `Proposal` to prove the fix directly
    rather than leaving that branch's preservation unverified — a fix with no test pinning
    it is exactly the kind of thing this feature's mutation-testing discipline exists to
    catch."""
    proposal = Proposal(
        code="ADV002",
        title="Unused index",
        rationale="never scanned",
        evidence={"schema": R.schema, "table": R.table},
        confidence=Confidence.HIGH,
        ddl='DROP INDEX "public"."idx_unused";',
        note="a pre-existing caveat that must not be discarded",
    )
    context = _dbt_context_for(R)
    (enriched,) = enrich_proposals([proposal], context)
    assert "a pre-existing caveat that must not be discarded" in (enriched.note or "")
    assert "dbt WARNING" in (enriched.note or "")
