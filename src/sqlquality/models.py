"""Shared data models for sqlquality."""

from __future__ import annotations

from collections.abc import Mapping
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


@dataclass(frozen=True, order=True)
class Relation:
    """A schema-qualified relation — the key every catalog fact is stored under.

    Bare table names were the key until multi-schema support landed, and they aliased: two
    schemas each holding an `orders` merged into one entry, so the last catalog row won the
    row estimate while `qualify()` resolved columns against the union of both column sets.

    ``order=True`` because the rules sort their output for canonical, run-to-run stable
    report ordering, and a bare `sorted()` over relation keys has to work. Field order is
    (schema, table) so that ordering groups a schema's tables together.
    """

    schema: str
    table: str

    def __str__(self) -> str:
        """`schema.table` — how the relation appears in a proposal title or JSON key."""
        return f"{self.schema}.{self.table}"


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
    relation: Relation
    column: str
    role: ColumnRole
    calls: int
    cost_ms: float
    #: Fraction of the *whole* analyzed window's cost carried by queries using this column
    #: in this role. Deliberately **not** a partition — the shares do not sum to 1:
    #:
    #: * A query filtering on two columns credits its full cost to *both* entries, because
    #:   both predicates really are involved in that cost. (This is why proposals take the
    #:   max cost_share over their columns rather than the sum — summing double-counts.)
    #: * The denominator includes queries that were skipped as unparseable or
    #:   unqualifiable, so poor schema coverage dilutes every surviving share rather than
    #:   silently inflating it. Read it alongside the report's skipped counts.
    cost_share: float
    #: Fingerprints of the query groups that contributed this usage. Needed to ask whether
    #: two usages *co-occur* — a partial-index proposal is only supported if some single
    #: query actually filters on the indexed column and the guard column together.
    fingerprint_ids: frozenset[str] = frozenset()

    @property
    def fingerprints(self) -> int:
        """How many query groups contributed this usage.

        Derived rather than stored: it and `fingerprint_ids` were two fields carrying one
        fact, kept in step only by convention.
        """
        return len(self.fingerprint_ids)


@dataclass(frozen=True)
class Aggregation:
    usage: tuple[ColumnUsage, ...]
    total_cost_ms: float
    skipped_unqualifiable: int
    tables: frozenset[Relation]
    #: Statements dropped because a bare table name is held by two introspected schemas.
    #: Separate from `skipped_unqualifiable` because the remedy differs: qualify the query
    #: or run once per schema, rather than widen the schema.
    #:
    #: Counts two distinct discovery situations, both the same underlying fact. Most
    #: statements reference a column by name, so `qualify()` (or the DML sole-target check)
    #: raises trying to resolve it and `aggregate()` counts the exception directly. A
    #: statement that names the ambiguous table but references none of its columns by name
    #: — `select * from orders`, `select count(*) from orders`, `select 1 from orders` —
    #: gives `qualify()` nothing to validate, so it raises nothing either; `aggregate()`
    #: instead recognizes this case directly (zero usage extracted, plus a bare table name
    #: two introspected schemas both hold) and counts it the same way.
    skipped_ambiguous: int = 0


@dataclass(frozen=True)
class TableFacts:
    """Engine-neutral catalog facts. Engine-specific physical design stays in the adapter."""

    #: The schema-qualified key this table is stored under — not a display name. Two
    #: same-named tables in different schemas each get their own `TableFacts`, keyed by
    #: their own `Relation`; a bare name here would alias them back together.
    relation: Relation
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


def cost_share_of(evidence: Mapping[str, object]) -> float | None:
    """A proposal's cost share as a number, or None when it is absent or not one.

    ``evidence`` is ``dict[str, object]``, so neither presence nor type is guaranteed —
    and ``bool`` is an ``int`` subclass, so a plain ``isinstance(value, (int, float))``
    accepts ``True`` and renders it as "100.0%": the most prominent number in the report,
    fabricated. That guard existed in the DDL renderer and was documented there, but the
    markdown renderer, the terminal table and the proposal sort key each re-derived the
    check and each omitted it. One helper, four call sites, no way to omit it again.
    """
    value = evidence.get("cost_share")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


@dataclass(frozen=True)
class ConnectionParams:
    engine: str
    dsn: str | None
    fields: dict[str, str]
    #: Where the credentials came from — printed to stderr, like check's dialect resolution.
    source: str
