"""WorkloadAdapter interface — one per engine.

An adapter owns exactly four things: its driver, its introspection statements, its
proposal rules, and its DDL syntax. Parsing, redaction and the usage rollup are shared,
engine-agnostic code so they are implemented and audited once.

This is deliberately *not* an extension of PerfAdapter: PerfAdapter analyzes one SQL
string offline, whereas a WorkloadAdapter analyzes a corpus against a live catalog.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from datetime import timedelta

from sqlquality.models import (
    Aggregation,
    ConnectionParams,
    Proposal,
    Relation,
    TableFacts,
    Workload,
    WorkloadFetch,
)

#: Executes one parameterized introspection statement and returns its rows.
#: Injectable so every fetch path is testable without a live database.
Querier = Callable[[str, tuple[object, ...]], list[tuple[object, ...]]]

#: Accepted `--timeout` range, in seconds. Defined here, once, because two layers need it:
#: the CLI rejects an out-of-range value up front (silently altering a number the user
#: typed is worse than telling them), and the adapter clamps to the same bounds as a safety
#: net for any other caller. As two independent constant pairs they could drift, and then
#: the CLI would reject what the adapter accepts, or the adapter would clamp past the range
#: the CLI's own error message promises. 0 in particular means "no limit" to Postgres — the
#: opposite of a timeout — and an absurd upper value is more likely a typo than an intent.
MIN_TIMEOUT_S = 1
MAX_TIMEOUT_S = 3600


@dataclass(frozen=True)
class IntrospectionStatement:
    """One statement an adapter may run, with what to tell the user if it is denied."""

    capability: str
    sql: str
    privilege_hint: str


class WorkloadAdapter(ABC):
    """Reads workload and catalog metadata for one engine and proposes optimizations."""

    engine: str

    def __init__(self) -> None:
        #: (capability, reason) for each introspection statement that failed. The command
        #: reports these and continues rather than aborting on a single missing grant.
        self.degraded: list[tuple[str, str]] = []
        #: Schema(s) to introspect. The CLI overwrites this from --schema before connect().
        self.schemas: tuple[str, ...] = ("public",)

    @abstractmethod
    def introspection_sql(self) -> list[IntrospectionStatement]:
        """Every statement this adapter can run. Backs --dry-run."""

    @abstractmethod
    def connect(self, params: ConnectionParams, timeout_s: int) -> None:
        """Open a read-only session with a statement timeout."""

    @abstractmethod
    def fetch_workload(self, since: timedelta | None, limit: int) -> WorkloadFetch:
        """Raw query-history rows plus an honest description of the window they cover."""

    @abstractmethod
    def fetch_schema(self, schemas: tuple[str, ...]) -> dict:
        """Schema mapping for sqlglot qualify(): {schema: {table: {column: type}}}.

        Nested rather than flat: a flat `{table: {column: type}}` map cannot tell two
        same-named tables in different schemas apart, so a column that exists in only one
        of them resolves against the union of both — the exact aliasing this task exists
        to remove.
        """

    @abstractmethod
    def fetch_table_facts(
        self, schemas: tuple[str, ...], relations: frozenset[Relation]
    ) -> dict[Relation, TableFacts]:
        """Row estimates, sizes, columns and per-column NDV for the given relations."""

    @abstractmethod
    def propose(
        self,
        aggregation: Aggregation,
        facts: dict[Relation, TableFacts],
        workload: Workload,
        *,
        min_cost_share: float,
    ) -> list[Proposal]:
        """Engine-specific proposal rules."""

    @abstractmethod
    def render_ddl(self, proposals: list[Proposal]) -> str:
        """A reviewable DDL script for the proposals that carry DDL. Never executed."""
