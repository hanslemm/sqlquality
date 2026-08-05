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
    Confidence,
    ConnectionParams,
    Proposal,
    Relation,
    TableFacts,
    Workload,
    WorkloadFetch,
    cost_share_of,
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

    #: Highest confidence first, then largest cost share — the reading order a human wants.
    _CONFIDENCE_ORDER = {Confidence.HIGH: 0, Confidence.MEDIUM: 1, Confidence.LOW: 2}

    @classmethod
    def ranking_key(cls, proposal: Proposal) -> tuple[int, float, str, str]:
        """Canonical presentation order for proposals *this* adapter produced.

        Public, and on the ABC, because ordering is each adapter's own responsibility and a
        caller outside the adapter legitimately needs it: `cli.advise` re-sorts after the
        optional dbt enrichment layer appends ADV301/ADV303 and downgrades some proposals,
        and without a hook it reached into one specific adapter's private classmethod — so a
        future engine would silently have got Postgres's ordering on the dbt path only, while
        keeping its own everywhere else. An adapter whose proposals want a different reading
        order overrides this; the default is the ordering every adapter has wanted so far.

        Highest confidence first, then largest cost share, then a canonical tiebreak so
        equal-confidence equal-cost proposals do not reorder between runs and make the CLI's
        tests flaky.

        `cost_share_of` rather than `float(evidence.get(...))`: bool is an int subclass, so a
        stray True became -1.0 and sorted a fabricated share ahead of a genuinely hot
        proposal, at the top of the list the CLI presents as "read this first".
        """
        return (
            cls._CONFIDENCE_ORDER[proposal.confidence],
            -(cost_share_of(proposal.evidence) or 0.0),
            proposal.code,
            proposal.title,
        )

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

    def window_facts(self) -> dict[str, object]:
        """Structured facts about the window `fetch_workload` just read, for the payload.

        Not abstract, and returns `{}` by default, because an adapter that knows none of
        these is a legitimate state rather than an unfinished one — the payload fills the
        gaps with `None`. What each field is for: `stats_reset_at` tells a later
        comparison whether cumulative counters were cleared between two runs (the
        difference between two independent samples and one containing the other), `since`
        whether a duration filter was genuinely applied, and `limit` whether the window
        was truncated. Reporting a `since` an engine did not apply would make an
        incomparable pair look comparable, which is worse than reporting nothing.
        """
        return {}

    def physical_state(self, relations: frozenset[Relation]) -> dict[str, dict[str, object]]:
        """Physical-design facts behind each of `relations`, for the payload.

        Not abstract, and returns `{}` by default, because an adapter with no physical
        levers to report — or a caller asking about no relations at all — is a legitimate
        state rather than an unfinished one. `sqlquality verify` (a later task) diffs this
        field between two `advise --json` artifacts to *observe* whether a proposal was
        applied, rather than trust the assertion, so what is recorded here must be the
        physical facts a later run can compare against, not a restatement of the proposal.

        Keyed by `str(relation)` (`"schema.table"`), never by `Relation` itself: a
        `Relation` is not JSON-serializable, and this dict flows straight into the
        `--json` payload — a `TypeError` raised here would fire only after the whole
        analysis has already run, which this project has shipped once before.

        Must not issue any SQL of its own. This is called after `propose()` has already
        fetched everything the run needed, so an implementation reads back what that
        fetch cached on the instance rather than querying the catalog a second time for a
        payload field.

        **A relation's key is always present when asked about, but its fields must be
        `None` — present-but-null, the same idiom `window_facts()` uses — for either of
        two distinct reasons a fetch left no evidence, both of which mean "this run could
        not tell you," never a measurement:**

        * the relation's facts were simply never fetched at all, because the analysis
          never needed them (e.g. a dbt-enriched proposal for a relation outside
          `aggregation.tables`, which `fetch_table_facts`/`fetch_indexes` were never asked
          about); or
        * the relevant capability is in `self.degraded` (a denied grant), so whatever the
          cache holds for this run is empty regardless of which relations were asked
          about.

        Neither case may report `False` or `[]` — a **measurement** that the relation is
        not an ordinary table, or genuinely has no indexes — because a later run that
        *does* observe the relation would then read a manufactured transition
        (`None → False` reads as nothing happened, but `False → True` or `[] → [...]`
        reads as a table or its indexes having just been created) where nothing physical
        actually changed. An implementation must therefore track, per relation, both
        "was this genuinely looked up this run" and "did the lookup's capability
        degrade" — not merely whether a cache happens to hold an entry, since an empty
        cache is what *both* an unfetched relation and a degraded one look like.
        """
        return {}
