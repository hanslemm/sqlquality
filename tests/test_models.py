from dataclasses import FrozenInstanceError

import pytest

from sqlquality.models import ComplexityMetrics, ComplexityScore, DagFacts


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
