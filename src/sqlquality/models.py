"""Shared data models for sqlquality."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ComplexityMetrics:
    """Raw structural complexity counts for one SQL statement."""

    join_count: int
    cte_count: int
    subquery_count: int
    window_count: int
    case_count: int
    union_count: int
    distinct_count: int
    select_count: int
    max_select_depth: int
    projected_columns: int


@dataclass(frozen=True)
class DagFacts:
    """A model's position in the dbt DAG (0 when unknown/offline)."""

    fan_in: int = 0
    fan_out: int = 0
    lineage_depth: int = 0


@dataclass(frozen=True)
class ComplexityScore:
    """A weighted 0-100 complexity score with per-component contributions."""

    composite: float
    components: dict[str, float]
    metrics: ComplexityMetrics
    dag: DagFacts | None = None
