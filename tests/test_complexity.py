import pytest

from sqlquality.complexity import METRIC_WEIGHTS, ComplexityEngine
from sqlquality.models import ComplexityMetrics, DagFacts

# metrics matching SQL_A / SQL_B / SQL_C from Task 3
M_A = ComplexityMetrics(0, 0, 0, 0, 0, 0, 0, 1, 1, 2)
M_B = ComplexityMetrics(1, 1, 0, 1, 0, 0, 0, 2, 2, 3)
M_C = ComplexityMetrics(0, 0, 3, 0, 0, 0, 0, 4, 4, 1)


def test_score_simple():
    s = ComplexityEngine().score(M_A)
    assert s.composite == pytest.approx(5.4)


def test_score_cte_join_window():
    s = ComplexityEngine().score(M_B)
    assert s.composite == pytest.approx(22.6)
    assert s.components["join_count"] == pytest.approx(6.0)
    assert s.components["max_select_depth"] == pytest.approx(10.0)


def test_score_nested_subqueries():
    s = ComplexityEngine().score(M_C)
    assert s.composite == pytest.approx(35.2)


def test_dag_facts_increase_score():
    s = ComplexityEngine().score(M_A, DagFacts(fan_out=10, lineage_depth=3))
    assert s.composite == pytest.approx(26.4)
    assert s.components["dag.fan_out"] == pytest.approx(15.0)
    assert s.dag is not None


def test_score_is_uncapped():
    # Previously capped at 100.0; the composite is now open-ended so the delta
    # gate stays sensitive for already-very-complex models.
    huge = ComplexityMetrics(50, 50, 50, 50, 50, 50, 50, 50, 50, 50)
    # select_count carries no weight; every other metric contributes weight * 50.
    expected = round(sum(weight * 50 for weight in METRIC_WEIGHTS.values()), 1)
    score = ComplexityEngine().score(huge).composite
    assert score == pytest.approx(expected)
    assert score > 100.0
