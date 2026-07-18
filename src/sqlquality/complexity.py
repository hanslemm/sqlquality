"""Turn structural metrics (+ optional DAG facts) into a 0-100 complexity score."""

from __future__ import annotations

from sqlquality.models import ComplexityMetrics, ComplexityScore, DagFacts

MAX_SCORE = 100.0

METRIC_WEIGHTS: dict[str, float] = {
    "join_count": 6.0,
    "cte_count": 2.0,
    "subquery_count": 5.0,
    "window_count": 4.0,
    "case_count": 2.0,
    "union_count": 3.0,
    "distinct_count": 2.0,
    "max_select_depth": 5.0,
    "projected_columns": 0.2,
}

DAG_WEIGHTS: dict[str, float] = {
    "fan_out": 1.5,
    "lineage_depth": 2.0,
}


class ComplexityEngine:
    """Compute a weighted composite complexity score."""

    def score(self, metrics: ComplexityMetrics, dag: DagFacts | None = None) -> ComplexityScore:
        components: dict[str, float] = {}
        for name, weight in METRIC_WEIGHTS.items():
            components[name] = round(weight * getattr(metrics, name), 2)
        if dag is not None:
            for name, weight in DAG_WEIGHTS.items():
                components[f"dag.{name}"] = round(weight * getattr(dag, name), 2)
        composite = round(min(MAX_SCORE, sum(components.values())), 1)
        return ComplexityScore(composite=composite, components=components, metrics=metrics, dag=dag)
