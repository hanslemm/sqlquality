"""Shared data models for sqlquality."""

from __future__ import annotations

from dataclasses import dataclass, field
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


@dataclass(frozen=True)
class RawQueryRow:
    """One row of query history as read from an engine, literals still present."""

    sql: str
    calls: int
    total_time_ms: float
    bytes_scanned: int | None = None
    partitions_scanned: int | None = None
    partitions_total: int | None = None


@dataclass(frozen=True)
class WorkloadFetch:
    """What an adapter returns from fetch_workload: raw rows plus an honest window label."""

    rows: tuple[RawQueryRow, ...]
    window_description: str


@dataclass(frozen=True)
class QueryStat:
    """One redacted, fingerprinted query group with its aggregated cost."""

    fingerprint: str
    sql: str
    calls: int
    total_time_ms: float
    bytes_scanned: int | None = None
    partitions_scanned: int | None = None
    partitions_total: int | None = None
    #: Literal-derived signals captured before redaction (see fingerprint.FLAG_*).
    flags: frozenset[str] = frozenset()


@dataclass(frozen=True)
class Workload:
    stats: tuple[QueryStat, ...]
    window_description: str
    skipped_unparseable: int = 0
    skipped_noise: int = 0

    @property
    def total_cost_ms(self) -> float:
        return sum(s.total_time_ms for s in self.stats)


class ColumnRole(str, Enum):
    EQUALITY = "equality"
    RANGE = "range"
    JOIN = "join"
    SORT = "sort"
    GROUP = "group"
    NULL_CHECK = "null_check"
    NOT_NULL_CHECK = "not_null_check"
    NON_SARGABLE = "non_sargable"


@dataclass(frozen=True)
class ColumnUsage:
    table: str
    column: str
    role: ColumnRole
    calls: int
    cost_ms: float
    cost_share: float
    fingerprints: int


@dataclass(frozen=True)
class Aggregation:
    usage: tuple[ColumnUsage, ...]
    total_cost_ms: float
    skipped_unqualifiable: int
    tables: frozenset[str]


@dataclass(frozen=True)
class TableFacts:
    """Engine-neutral catalog facts. Engine-specific physical design stays in the adapter."""

    name: str
    row_estimate: int | None
    size_bytes: int | None
    columns: tuple[str, ...]
    #: column -> distinct-value estimate. Empty when the engine gave us no statistics.
    ndv: dict[str, float] = field(default_factory=dict)


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass(frozen=True)
class Proposal:
    code: str
    title: str
    rationale: str
    evidence: dict[str, object]
    confidence: Confidence
    ddl: str | None = None


@dataclass(frozen=True)
class ConnectionParams:
    engine: str
    dsn: str | None
    fields: dict[str, str]
    #: Where the credentials came from — printed to stderr, like check's dialect resolution.
    source: str
