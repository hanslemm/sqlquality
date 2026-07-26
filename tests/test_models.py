from dataclasses import FrozenInstanceError

import pytest

from sqlquality.models import ComplexityMetrics, ComplexityScore, DagFacts
from sqlquality.models import (
    Aggregation,
    ColumnRole,
    ColumnUsage,
    Confidence,
    Proposal,
    QueryStat,
    RawQueryRow,
    TableFacts,
    Workload,
)


def test_complexity_metrics_holds_counts():
    m = ComplexityMetrics(
        join_count=1,
        cte_count=1,
        subquery_count=0,
        window_count=1,
        case_count=0,
        union_count=0,
        distinct_count=0,
        select_count=2,
        max_select_depth=2,
        projected_columns=3,
    )
    assert m.join_count == 1
    assert m.max_select_depth == 2
    with pytest.raises(FrozenInstanceError):
        m.join_count = 5  # type: ignore[misc]


def test_dag_facts_defaults_zero():
    assert DagFacts() == DagFacts(fan_in=0, fan_out=0, lineage_depth=0)


def test_complexity_score_fields():
    m = ComplexityMetrics(0, 0, 0, 0, 0, 0, 0, 1, 1, 2)
    s = ComplexityScore(composite=5.4, components={"max_select_depth": 5.0}, metrics=m)
    assert s.composite == 5.4
    assert s.dag is None


def test_query_stat_is_frozen_and_defaults_engine_optionals():
    stat = QueryStat(fingerprint="fp", sql="SELECT 1", calls=3, total_time_ms=12.5)
    assert stat.bytes_scanned is None
    assert stat.flags == frozenset()
    with pytest.raises(Exception):
        stat.calls = 4  # type: ignore[misc]


def test_workload_cost_totals_only_its_own_stats():
    workload = Workload(
        stats=(
            QueryStat(fingerprint="a", sql="SELECT 1", calls=1, total_time_ms=10.0),
            QueryStat(fingerprint="b", sql="SELECT 2", calls=1, total_time_ms=30.0),
        ),
        window_description="since stats reset",
        skipped_unparseable=1,
        skipped_noise=2,
    )
    assert workload.total_cost_ms == 40.0


def test_table_facts_ndv_defaults_empty():
    facts = TableFacts(name="orders", row_estimate=100, size_bytes=None, columns=("id",))
    assert facts.ndv == {}


def test_proposal_and_aggregation_construct():
    usage = ColumnUsage(
        table="orders",
        column="status",
        role=ColumnRole.EQUALITY,
        calls=5,
        cost_ms=50.0,
        cost_share=0.5,
        fingerprints=2,
    )
    agg = Aggregation(usage=(usage,), total_cost_ms=100.0, skipped_unqualifiable=0,
                      tables=frozenset({"orders"}))
    assert agg.usage[0].role is ColumnRole.EQUALITY
    proposal = Proposal(
        code="ADV001",
        title="Index orders(status)",
        rationale="hot equality predicate",
        evidence={"cost_share": 0.5},
        confidence=Confidence.MEDIUM,
        ddl="CREATE INDEX ...",
    )
    assert proposal.confidence.value == "medium"


def test_raw_query_row_requires_only_sql_calls_and_time():
    row = RawQueryRow(sql="SELECT 1", calls=1, total_time_ms=1.0)
    assert row.bytes_scanned is None
