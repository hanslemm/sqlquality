"""Shared data models for sqlquality."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


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
    """A weighted, open-ended complexity score with per-component contributions."""

    composite: float
    components: dict[str, float]
    metrics: ComplexityMetrics
    dag: DagFacts | None = None


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class Finding:
    code: str
    message: str
    line: int
    severity: Severity
    fixable: bool
