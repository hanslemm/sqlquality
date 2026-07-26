# `sqlquality advise` (Postgres) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `sqlquality advise` for Postgres — read `pg_stat_statements` and catalog metadata over a read-only connection, aggregate column usage weighted by query cost, and emit ranked index proposals as a report plus a reviewable DDL file.

**Architecture:** An engine-agnostic core (`ingest` → `extract` → `aggregate`) does all parsing, literal redaction and column-usage rollup with no database involved; a `WorkloadAdapter` per engine owns only its driver, introspection SQL, proposal rules and DDL syntax. Redaction lives in the shared core so there is exactly one place to audit. The Postgres adapter takes an injectable `Querier` callable, so every fetch path is fixture-testable without a live warehouse.

**Tech Stack:** Python 3.11+, sqlglot 30.12 (`parse_one`, `optimizer.qualify`), typer, rich, psycopg 3 (new optional extra), pytest.

Spec: `docs/superpowers/specs/2026-07-26-advise-workload-analysis-design.md`. This plan covers spec build steps 1–4 (engine-agnostic core, connection resolution, Postgres adapter, README rewrite). Redshift, Snowflake and dbt enrichment are follow-up plans.

## Global Constraints

- Python `>=3.11`. Every new module starts with `from __future__ import annotations`.
- `sqlglot>=30.12,<31`. Verified available: `sqlglot.optimizer.qualify.qualify(expression, dialect=, schema=, expand_stars=)`, raising `sqlglot.errors.OptimizeError` on unresolvable columns.
- Ruff line length 100. Mypy runs with `python_version = "3.11"`; new third-party imports without stubs need a `[[tool.mypy.overrides]]` entry with `ignore_missing_imports = true`.
- Exit-code contract, unchanged from the README: **0** = success (proposals are advisory and never gate), **2** = usage/config/input/connection error. `advise` never exits 1.
- `advise` never issues DDL or DML, and never executes user-supplied SQL. Only the statements returned by `introspection_sql()` are ever run.
- Literals are redacted at ingest by default. `--keep-literals` opts back in. Any literal-derived signal must be captured as a boolean flag *before* redaction, never by retaining the literal.
- All new dataclasses are `@dataclass(frozen=True)`, matching `models.py`.
- Tests use `pytest` + `typer.testing.CliRunner`, following `tests/test_perf_cli.py`.
- Commit after every task. Branch: `feat/advise-postgres`.

## Deviations from the spec (agreed while planning)

These are refinements found while verifying the sqlglot API. The spec is updated in Task 15.

1. `fetch_workload()` returns a `WorkloadFetch` (raw rows + window description), not a `Workload`. Redaction then happens in the engine-agnostic `ingest()`. Rationale: keeps redaction in exactly one auditable place instead of once per adapter.
2. `ColumnRole` gains `NULL_CHECK`, `NOT_NULL_CHECK` and `NON_SARGABLE` beyond the spec's five. ADV004 and ADV005 need them for table/column attribution.
3. `skipped_unqualifiable` moves from `Workload` to `Aggregation`, because qualification happens during aggregation, not ingest.
4. New `--schema` option (repeatable, default `public`). Without it, bare table names are ambiguous across schemas and the sqlglot schema dict cannot be built reliably.
5. ADV005's leading-wildcard rule is driven by a pre-redaction boolean flag on `QueryStat`, since `like '%x'` becomes `like $1` after redaction. It reports at query-group level without column attribution; the function/cast rules keep full attribution.

## File Structure

**Create:**

| File | Responsibility |
|---|---|
| `src/sqlquality/workload/__init__.py` | Adapter registry — `get_workload_adapter(engine)` |
| `src/sqlquality/workload/base.py` | `WorkloadAdapter` ABC, `IntrospectionStatement`, `Querier` type |
| `src/sqlquality/workload/fingerprint.py` | Parse, capture literal-derived flags, redact, fingerprint, `ingest()` |
| `src/sqlquality/workload/extract.py` | `qualify()` + column-role extraction |
| `src/sqlquality/workload/aggregate.py` | Rollup to `ColumnUsage` with cost shares |
| `src/sqlquality/workload/connection.py` | Credential resolution: flag > env > profiles.yml |
| `src/sqlquality/workload/profiles.py` | Optional dbt `profiles.yml` reader |
| `src/sqlquality/workload/postgres.py` | Postgres adapter: introspection SQL, fetch, ADV001–006, DDL |
| `tests/test_workload_fingerprint.py`, `tests/test_workload_extract.py`, `tests/test_workload_aggregate.py`, `tests/test_workload_connection.py`, `tests/test_workload_postgres.py`, `tests/test_workload_rules.py`, `tests/test_advise_cli.py`, `tests/test_workload_redaction.py` | One test module per unit |

**Modify:** `src/sqlquality/models.py` (new dataclasses), `src/sqlquality/antipatterns.py` (expose `has_select_star`), `src/sqlquality/report.py` (advise renderers), `src/sqlquality/cli.py` (`advise` command), `pyproject.toml` (extras, mypy override), `README.md`, `CHANGELOG.md`.

The three core modules (`fingerprint`, `extract`, `aggregate`) hold the logic and are pure functions over SQL strings and dataclasses. That is deliberate: it is where the tests live, and none of them need a database.

---

### Task 1: Workload dataclasses

**Files:**
- Modify: `src/sqlquality/models.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `RawQueryRow`, `QueryStat`, `Workload`, `WorkloadFetch`, `ColumnRole`, `ColumnUsage`, `Aggregation`, `TableFacts`, `Confidence`, `Proposal`, `ConnectionParams`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_models.py`:

```python
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
```

Ensure `import pytest` is present at the top of the file.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_models.py -v`
Expected: FAIL with `ImportError: cannot import name 'RawQueryRow'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/sqlquality/models.py`:

```python
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
```

`models.py` currently imports only `dataclass` and `Enum`; add `field` to the dataclasses import.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_models.py -v && uv run mypy src/sqlquality/models.py && uv run ruff check src/sqlquality/models.py`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/sqlquality/models.py tests/test_models.py
git commit -m "feat(workload): add workload analysis dataclasses"
```

---

### Task 2: Ingest — literal flags, redaction, fingerprinting

**Files:**
- Create: `src/sqlquality/workload/__init__.py` (empty for now), `src/sqlquality/workload/fingerprint.py`
- Modify: `src/sqlquality/antipatterns.py` (rename `_has_select_star` to `has_select_star`)
- Test: `tests/test_workload_fingerprint.py`

**Interfaces:**
- Consumes: `RawQueryRow`, `QueryStat`, `Workload` (Task 1); `sqlquality.sqlast.parse`, `SqlParseError`.
- Produces:
  - `FLAG_LEADING_WILDCARD_LIKE: str`, `FLAG_SELECT_STAR: str`
  - `literal_flags(tree: exp.Expression) -> frozenset[str]`
  - `redact_tree(tree: exp.Expression) -> exp.Expression`
  - `is_noise(sql: str) -> bool`
  - `ingest(fetch: WorkloadFetch, dialect: str, *, keep_literals: bool = False) -> Workload`
  - `sqlquality.antipatterns.has_select_star(tree) -> bool` (was `_has_select_star`)

Verified behaviour this task relies on: `exp.Literal` nodes replaced with `exp.Placeholder()` render as `%s` in the postgres dialect; `pg_stat_statements`-style `$1` parses to `exp.Parameter` and survives `qualify()` untouched.

- [ ] **Step 1: Write the failing test**

Create `tests/test_workload_fingerprint.py`:

```python
import sqlglot

from sqlquality.models import RawQueryRow, WorkloadFetch
from sqlquality.workload.fingerprint import (
    FLAG_LEADING_WILDCARD_LIKE,
    FLAG_SELECT_STAR,
    ingest,
    is_noise,
    literal_flags,
    redact_tree,
)


def _tree(sql):
    return sqlglot.parse_one(sql, dialect="postgres")


def test_redact_removes_every_literal():
    redacted = redact_tree(_tree("select id from t where email = 'a@b.de' and n > 42"))
    rendered = redacted.sql("postgres")
    assert "a@b.de" not in rendered
    assert "42" not in rendered


def test_redact_does_not_mutate_the_input_tree():
    tree = _tree("select id from t where email = 'a@b.de'")
    redact_tree(tree)
    assert "a@b.de" in tree.sql("postgres")


def test_literal_flags_capture_leading_wildcard_before_redaction():
    assert FLAG_LEADING_WILDCARD_LIKE in literal_flags(_tree("select 1 from t where a like '%x'"))
    assert FLAG_LEADING_WILDCARD_LIKE not in literal_flags(
        _tree("select 1 from t where a like 'x%'")
    )


def test_literal_flags_capture_select_star():
    assert FLAG_SELECT_STAR in literal_flags(_tree("select * from t"))
    assert FLAG_SELECT_STAR not in literal_flags(_tree("select id from t"))


def test_is_noise_filters_our_own_introspection_and_ddl():
    assert is_noise("SELECT * FROM pg_stat_statements")
    assert is_noise("select column_name from information_schema.columns")
    assert is_noise("CREATE INDEX idx ON t (a)")
    assert not is_noise("select id from orders where status = $1")


def test_ingest_groups_by_fingerprint_and_sums_cost():
    fetch = WorkloadFetch(
        rows=(
            RawQueryRow(sql="select id from t where a = 1", calls=2, total_time_ms=10.0),
            RawQueryRow(sql="select id from t where a = 999", calls=3, total_time_ms=20.0),
        ),
        window_description="since stats reset",
    )
    workload = ingest(fetch, "postgres")
    assert len(workload.stats) == 1
    assert workload.stats[0].calls == 5
    assert workload.stats[0].total_time_ms == 30.0
    assert "999" not in workload.stats[0].sql


def test_ingest_counts_unparseable_and_noise_without_raising():
    fetch = WorkloadFetch(
        rows=(
            RawQueryRow(sql="select from where", calls=1, total_time_ms=1.0),
            RawQueryRow(sql="select * from pg_stat_statements", calls=1, total_time_ms=1.0),
            RawQueryRow(sql="select id from t", calls=1, total_time_ms=1.0),
        ),
        window_description="w",
    )
    workload = ingest(fetch, "postgres")
    assert workload.skipped_unparseable == 1
    assert workload.skipped_noise == 1
    assert len(workload.stats) == 1


def test_ingest_keep_literals_preserves_values():
    fetch = WorkloadFetch(
        rows=(RawQueryRow(sql="select id from t where a = 999", calls=1, total_time_ms=1.0),),
        window_description="w",
    )
    workload = ingest(fetch, "postgres", keep_literals=True)
    assert "999" in workload.stats[0].sql


def test_ingest_stats_are_sorted_by_cost_descending():
    fetch = WorkloadFetch(
        rows=(
            RawQueryRow(sql="select a from t", calls=1, total_time_ms=1.0),
            RawQueryRow(sql="select b from t", calls=1, total_time_ms=99.0),
        ),
        window_description="w",
    )
    workload = ingest(fetch, "postgres")
    assert [s.total_time_ms for s in workload.stats] == [99.0, 1.0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_workload_fingerprint.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sqlquality.workload'`

- [ ] **Step 3: Write minimal implementation**

Create `src/sqlquality/workload/__init__.py`:

```python
"""Workload analysis: query-history ingestion, column-usage rollup, per-engine adapters."""
```

In `src/sqlquality/antipatterns.py`, rename `_has_select_star` to `has_select_star` (definition and its single call site in `antipattern_findings`), and update its docstring first line to `"""True if any SELECT projects a star, ignoring EXISTS probes and dbt CTE closers."""`. The two exemptions it already implements — `EXISTS (SELECT *)` and the idiomatic `select * from <local cte>` closer — are exactly right for workload analysis too, which is why this is reused rather than reimplemented.

Create `src/sqlquality/workload/fingerprint.py`:

```python
"""Ingest raw query history: capture literal-derived flags, redact, fingerprint, group.

Redaction lives here — in engine-agnostic code — so there is exactly one place to audit
for literal leakage, rather than one per adapter.
"""

from __future__ import annotations

import re
from collections import defaultdict

from sqlglot import exp

from sqlquality.antipatterns import has_select_star
from sqlquality.models import QueryStat, RawQueryRow, Workload, WorkloadFetch
from sqlquality.sqlast import SqlParseError, parse

#: A LIKE/ILIKE pattern starting with '%' is non-sargable. Detectable only before
#: redaction, so it is captured as a flag and the literal is then discarded.
FLAG_LEADING_WILDCARD_LIKE = "leading_wildcard_like"
#: The query group projects a star. Captured pre-qualify (qualify runs expand_stars=False).
FLAG_SELECT_STAR = "select_star"

#: Statements that are our own introspection, dbt's metadata, or DDL — advising on these
#: would be advising on our own noise.
_NOISE = re.compile(
    r"\b(pg_stat_statements|pg_stat_user_indexes|pg_stats|pg_class|pg_index|pg_namespace"
    r"|information_schema|svv_table_info|svl_statementtext|sys_query_history"
    r"|account_usage|create\s+index|drop\s+index|create\s+table|alter\s+table"
    r"|vacuum|analyze|begin|commit|rollback|set\s+)\b",
    re.IGNORECASE,
)


def is_noise(sql: str) -> bool:
    """True if a statement is introspection, session management, or DDL."""
    return _NOISE.search(sql) is not None


def literal_flags(tree: exp.Expression) -> frozenset[str]:
    """Signals that can only be read while literals are still present."""
    flags: set[str] = set()
    for node in tree.find_all(exp.Like, exp.ILike):
        pattern = node.args.get("expression")
        if isinstance(pattern, exp.Literal) and pattern.is_string and pattern.this.startswith("%"):
            flags.add(FLAG_LEADING_WILDCARD_LIKE)
            break
    if has_select_star(tree):
        flags.add(FLAG_SELECT_STAR)
    return frozenset(flags)


def redact_tree(tree: exp.Expression) -> exp.Expression:
    """Return a copy of ``tree`` with every literal replaced by a bind placeholder."""
    copy = tree.copy()
    for literal in list(copy.find_all(exp.Literal)):
        literal.replace(exp.Placeholder())
    return copy


def ingest(
    fetch: WorkloadFetch, dialect: str, *, keep_literals: bool = False
) -> Workload:
    """Parse, flag, redact, fingerprint and group raw history rows into a Workload.

    Unparseable and noise statements are counted, never raised and never silently
    dropped. Raw rows are not retained past this function.
    """
    calls: dict[str, int] = defaultdict(int)
    cost: dict[str, float] = defaultdict(float)
    bytes_scanned: dict[str, int] = defaultdict(int)
    flags: dict[str, set[str]] = defaultdict(set)
    display: dict[str, str] = {}
    skipped_unparseable = 0
    skipped_noise = 0

    for row in fetch.rows:
        if is_noise(row.sql):
            skipped_noise += 1
            continue
        try:
            tree = parse(row.sql, dialect)
        except SqlParseError:
            skipped_unparseable += 1
            continue

        row_flags = literal_flags(tree)
        analyzed = tree if keep_literals else redact_tree(tree)
        # identify=True quotes every identifier, giving a stable canonical key regardless
        # of how the engine happened to spell the query.
        key = analyzed.sql(dialect, identify=True)
        calls[key] += row.calls
        cost[key] += row.total_time_ms
        if row.bytes_scanned is not None:
            bytes_scanned[key] += row.bytes_scanned
        flags[key] |= set(row_flags)
        display.setdefault(key, analyzed.sql(dialect))

    stats = tuple(
        sorted(
            (
                QueryStat(
                    fingerprint=key,
                    sql=display[key],
                    calls=calls[key],
                    total_time_ms=cost[key],
                    bytes_scanned=bytes_scanned.get(key) or None,
                    flags=frozenset(flags[key]),
                )
                for key in calls
            ),
            key=lambda s: s.total_time_ms,
            reverse=True,
        )
    )
    return Workload(
        stats=stats,
        window_description=fetch.window_description,
        skipped_unparseable=skipped_unparseable,
        skipped_noise=skipped_noise,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_workload_fingerprint.py tests/test_antipatterns.py -v`
Expected: all PASS (the existing antipatterns tests confirm the rename broke nothing)

- [ ] **Step 5: Commit**

```bash
git add src/sqlquality/workload/ src/sqlquality/antipatterns.py tests/test_workload_fingerprint.py
git commit -m "feat(workload): ingest query history with literal redaction"
```

---

### Task 3: Extract column roles

**Files:**
- Create: `src/sqlquality/workload/extract.py`
- Test: `tests/test_workload_extract.py`

**Interfaces:**
- Consumes: `ColumnRole` (Task 1).
- Produces:
  - `class UnqualifiableQuery(ValueError)`
  - `extract_usage(tree: exp.Expression, dialect: str, schema: dict) -> tuple[tuple[str, str, ColumnRole], ...]` returning `(table, column, role)` triples with duplicates removed.

Verified AST shapes this relies on: `qualify()` qualifies columns with the table **alias** (`o`, `c`), so an alias→table map is built from `exp.Table` nodes via `alias_or_name` → `name`. Ancestor chains: WHERE equality → `[EQ, Where]`; join key → `[EQ, Join]`; `ORDER BY` → `[Ordered, Order]`; `a is null` → `[Is, Where]`; `a is not null` → `[Is, Not, Where]`; `lower(a) = x` → `[Lower, EQ, Where]`; `a::text = x` → `[Cast, EQ, Where]`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_workload_extract.py`:

```python
import pytest
import sqlglot

from sqlquality.models import ColumnRole
from sqlquality.workload.extract import UnqualifiableQuery, extract_usage

SCHEMA = {
    "orders": {"id": "INT", "customer_id": "INT", "status": "TEXT", "created_at": "TIMESTAMP",
               "note": "TEXT", "shipped_at": "TIMESTAMP"},
    "customers": {"id": "INT", "email": "TEXT"},
}


def _usage(sql):
    tree = sqlglot.parse_one(sql, dialect="postgres")
    return set(extract_usage(tree, "postgres", SCHEMA))


def test_where_equality_is_equality_role():
    assert ("orders", "status", ColumnRole.EQUALITY) in _usage(
        "select id from orders where status = $1"
    )


def test_in_predicate_is_equality_role():
    assert ("orders", "status", ColumnRole.EQUALITY) in _usage(
        "select id from orders where status in ($1, $2)"
    )


def test_comparison_is_range_role():
    assert ("orders", "created_at", ColumnRole.RANGE) in _usage(
        "select id from orders where created_at > $1"
    )


def test_between_is_range_role():
    assert ("orders", "created_at", ColumnRole.RANGE) in _usage(
        "select id from orders where created_at between $1 and $2"
    )


def test_join_key_is_join_role_not_equality():
    usage = _usage(
        "select o.id from orders o join customers c on c.id = o.customer_id"
    )
    assert ("orders", "customer_id", ColumnRole.JOIN) in usage
    assert ("customers", "id", ColumnRole.JOIN) in usage
    assert ("orders", "customer_id", ColumnRole.EQUALITY) not in usage


def test_order_by_is_sort_role():
    assert ("orders", "created_at", ColumnRole.SORT) in _usage(
        "select id from orders order by created_at desc"
    )


def test_group_by_is_group_role():
    assert ("orders", "status", ColumnRole.GROUP) in _usage(
        "select status, count(*) from orders group by status"
    )


def test_window_order_by_is_not_a_query_sort_key():
    usage = _usage(
        "select id, row_number() over (partition by status order by created_at) from orders"
    )
    assert ("orders", "created_at", ColumnRole.SORT) not in usage


def test_null_checks_carry_polarity():
    assert ("orders", "shipped_at", ColumnRole.NULL_CHECK) in _usage(
        "select id from orders where shipped_at is null"
    )
    assert ("orders", "shipped_at", ColumnRole.NOT_NULL_CHECK) in _usage(
        "select id from orders where shipped_at is not null"
    )


def test_function_wrapped_predicate_is_non_sargable():
    usage = _usage("select id from orders where lower(status) = $1")
    assert ("orders", "status", ColumnRole.NON_SARGABLE) in usage
    assert ("orders", "status", ColumnRole.EQUALITY) not in usage


def test_cast_wrapped_predicate_is_non_sargable():
    assert ("orders", "id", ColumnRole.NON_SARGABLE) in _usage(
        "select id from orders where id::text = $1"
    )


def test_projected_columns_produce_no_usage():
    assert _usage("select note from orders") == set()


def test_unresolvable_column_raises_unqualifiable():
    tree = sqlglot.parse_one("select nope from mystery_table", dialect="postgres")
    with pytest.raises(UnqualifiableQuery):
        extract_usage(tree, "postgres", SCHEMA)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_workload_extract.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sqlquality.workload.extract'`

- [ ] **Step 3: Write minimal implementation**

Create `src/sqlquality/workload/extract.py`:

```python
"""Resolve a query's columns to their tables and classify how each column is used."""

from __future__ import annotations

from sqlglot import exp
from sqlglot.errors import OptimizeError
from sqlglot.optimizer.qualify import qualify

from sqlquality.models import ColumnRole

#: Comparison nodes that an index can satisfy with an equality probe.
_EQUALITY_NODES = (exp.EQ, exp.In)
#: Comparison nodes that an index can satisfy with a range scan.
_RANGE_NODES = (exp.GT, exp.GTE, exp.LT, exp.LTE, exp.Between)
#: Wrapping a column in one of these defeats a plain B-tree index on that column.
_SARGABILITY_BREAKERS = (exp.Cast, exp.Func)


class UnqualifiableQuery(ValueError):
    """Raised when a query's columns cannot be resolved against the supplied schema."""


def _within(node: exp.Expression, *types: type[exp.Expression]) -> bool:
    """True if any ancestor of ``node`` is one of ``types``. Mirrors antipatterns._within_exists."""
    parent = node.parent
    while parent is not None:
        if isinstance(parent, types):
            return True
        parent = parent.parent
    return False


def _role(column: exp.Column) -> ColumnRole | None:
    """Classify how a column is used, or None if it is merely projected.

    A window's PARTITION BY / ORDER BY is not a query-level sort key, so anything under
    an exp.Window is discarded before the ancestor walk begins.
    """
    if _within(column, exp.Window):
        return None

    # A column wrapped in a cast or function inside a predicate cannot use a plain index.
    if isinstance(column.parent, _SARGABILITY_BREAKERS) and _within(
        column, exp.Where, exp.Join, exp.Having
    ):
        return ColumnRole.NON_SARGABLE

    comparison: ColumnRole | None = None
    node: exp.Expression | None = column.parent
    while node is not None:
        if isinstance(node, exp.Is):
            null_side = isinstance(node.expression, exp.Null)
            if null_side:
                # `is not null` parses as Not(Is(...)), so polarity comes from the parent.
                return (
                    ColumnRole.NOT_NULL_CHECK
                    if isinstance(node.parent, exp.Not)
                    else ColumnRole.NULL_CHECK
                )
        if comparison is None:
            if isinstance(node, _EQUALITY_NODES):
                comparison = ColumnRole.EQUALITY
            elif isinstance(node, _RANGE_NODES):
                comparison = ColumnRole.RANGE
        # A join predicate is a join key, not a filter — Join must win over the EQ below it.
        if isinstance(node, exp.Join):
            return ColumnRole.JOIN
        if isinstance(node, exp.Order):
            return ColumnRole.SORT
        if isinstance(node, exp.Group):
            return ColumnRole.GROUP
        if isinstance(node, exp.Select):
            break
        node = node.parent
    return comparison


def extract_usage(
    tree: exp.Expression, dialect: str, schema: dict
) -> tuple[tuple[str, str, ColumnRole], ...]:
    """(table, column, role) triples for one query, deduplicated.

    Stars are not expanded: a projected star tells us nothing about which columns are
    filtered, and expanding it would drown the rollup in projection noise.
    """
    try:
        qualified = qualify(tree.copy(), dialect=dialect, schema=schema, expand_stars=False)
    except OptimizeError as exc:
        raise UnqualifiableQuery(str(exc)) from exc

    alias_to_table = {t.alias_or_name: t.name for t in qualified.find_all(exp.Table)}

    seen: set[tuple[str, str, ColumnRole]] = set()
    for column in qualified.find_all(exp.Column):
        table = alias_to_table.get(column.table)
        if not table or not column.name:
            continue
        role = _role(column)
        if role is None:
            continue
        seen.add((table, column.name, role))
    return tuple(sorted(seen, key=lambda triple: (triple[0], triple[1], triple[2].value)))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_workload_extract.py -v && uv run mypy src/sqlquality/workload/extract.py`
Expected: all PASS

If `test_function_wrapped_predicate_is_non_sargable` fails because `exp.Lower` is not an `exp.Func` subclass in this sqlglot version, verify with `uv run python -c "from sqlglot import exp; print(exp.Lower.__mro__)"` and widen `_SARGABILITY_BREAKERS` to match what it actually inherits from.

- [ ] **Step 5: Commit**

```bash
git add src/sqlquality/workload/extract.py tests/test_workload_extract.py
git commit -m "feat(workload): extract column roles via sqlglot qualify"
```

---

### Task 4: Aggregate usage into cost-weighted rollup

**Files:**
- Create: `src/sqlquality/workload/aggregate.py`
- Test: `tests/test_workload_aggregate.py`

**Interfaces:**
- Consumes: `Workload`, `QueryStat`, `Aggregation`, `ColumnUsage`, `ColumnRole` (Task 1); `extract_usage`, `UnqualifiableQuery` (Task 3).
- Produces: `aggregate(workload: Workload, schema: dict, dialect: str) -> Aggregation`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_workload_aggregate.py`:

```python
from sqlquality.models import ColumnRole, QueryStat, Workload
from sqlquality.workload.aggregate import aggregate

SCHEMA = {"orders": {"id": "INT", "status": "TEXT", "created_at": "TIMESTAMP"}}


def _workload(*pairs):
    return Workload(
        stats=tuple(
            QueryStat(fingerprint=f"fp{i}", sql=sql, calls=calls, total_time_ms=cost)
            for i, (sql, calls, cost) in enumerate(pairs)
        ),
        window_description="w",
    )


def _find(agg, column, role):
    return next(u for u in agg.usage if u.column == column and u.role is role)


def test_cost_share_is_fraction_of_analyzed_total():
    agg = aggregate(
        _workload(
            ("select id from orders where status = $1", 1, 75.0),
            ("select id from orders where created_at > $1", 1, 25.0),
        ),
        SCHEMA,
        "postgres",
    )
    assert agg.total_cost_ms == 100.0
    assert _find(agg, "status", ColumnRole.EQUALITY).cost_share == 0.75
    assert _find(agg, "created_at", ColumnRole.RANGE).cost_share == 0.25


def test_same_column_and_role_accumulates_across_fingerprints():
    agg = aggregate(
        _workload(
            ("select id from orders where status = $1", 2, 10.0),
            ("select created_at from orders where status = $1", 3, 30.0),
        ),
        SCHEMA,
        "postgres",
    )
    usage = _find(agg, "status", ColumnRole.EQUALITY)
    assert usage.cost_ms == 40.0
    assert usage.calls == 5
    assert usage.fingerprints == 2


def test_unqualifiable_queries_are_counted_not_raised():
    agg = aggregate(
        _workload(
            ("select mystery from unknown_table", 1, 10.0),
            ("select id from orders where status = $1", 1, 10.0),
        ),
        SCHEMA,
        "postgres",
    )
    assert agg.skipped_unqualifiable == 1
    assert agg.tables == frozenset({"orders"})


def test_usage_is_sorted_by_cost_descending():
    agg = aggregate(
        _workload(
            ("select id from orders where status = $1", 1, 5.0),
            ("select id from orders where created_at > $1", 1, 50.0),
        ),
        SCHEMA,
        "postgres",
    )
    assert agg.usage[0].column == "created_at"


def test_empty_workload_yields_empty_aggregation_and_no_division_error():
    agg = aggregate(Workload(stats=(), window_description="w"), SCHEMA, "postgres")
    assert agg.usage == ()
    assert agg.total_cost_ms == 0.0
    assert agg.tables == frozenset()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_workload_aggregate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sqlquality.workload.aggregate'`

- [ ] **Step 3: Write minimal implementation**

Create `src/sqlquality/workload/aggregate.py`:

```python
"""Roll per-query column roles up into a cost-weighted usage index."""

from __future__ import annotations

from collections import defaultdict

from sqlquality.models import Aggregation, ColumnRole, ColumnUsage, Workload
from sqlquality.sqlast import SqlParseError, parse
from sqlquality.workload.extract import UnqualifiableQuery, extract_usage

_Key = tuple[str, str, ColumnRole]


def aggregate(workload: Workload, schema: dict, dialect: str) -> Aggregation:
    """Weight every (table, column, role) by the cost of the queries that use it."""
    calls: dict[_Key, int] = defaultdict(int)
    cost: dict[_Key, float] = defaultdict(float)
    fingerprints: dict[_Key, int] = defaultdict(int)
    tables: set[str] = set()
    skipped_unqualifiable = 0

    for stat in workload.stats:
        try:
            tree = parse(stat.sql, dialect)
            triples = extract_usage(tree, dialect, schema)
        except (SqlParseError, UnqualifiableQuery):
            skipped_unqualifiable += 1
            continue
        for key in triples:
            calls[key] += stat.calls
            cost[key] += stat.total_time_ms
            fingerprints[key] += 1
            tables.add(key[0])

    total = workload.total_cost_ms
    usage = tuple(
        sorted(
            (
                ColumnUsage(
                    table=table,
                    column=column,
                    role=role,
                    calls=calls[(table, column, role)],
                    cost_ms=cost[(table, column, role)],
                    cost_share=(cost[(table, column, role)] / total) if total else 0.0,
                    fingerprints=fingerprints[(table, column, role)],
                )
                for (table, column, role) in calls
            ),
            key=lambda u: u.cost_ms,
            reverse=True,
        )
    )
    return Aggregation(
        usage=usage,
        total_cost_ms=total,
        skipped_unqualifiable=skipped_unqualifiable,
        tables=frozenset(tables),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_workload_aggregate.py -v && uv run mypy src/sqlquality/workload/`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/sqlquality/workload/aggregate.py tests/test_workload_aggregate.py
git commit -m "feat(workload): aggregate column usage weighted by query cost"
```

---

### Task 5: WorkloadAdapter interface and registry

**Files:**
- Create: `src/sqlquality/workload/base.py`
- Modify: `src/sqlquality/workload/__init__.py`
- Test: `tests/test_workload_postgres.py` (registry cases only; adapter cases land in Task 8)

**Interfaces:**
- Consumes: `ConnectionParams`, `WorkloadFetch`, `TableFacts`, `Proposal`, `Aggregation`, `Workload` (Task 1).
- Produces:
  - `Querier = Callable[[str, tuple[object, ...]], list[tuple[object, ...]]]`
  - `@dataclass(frozen=True) IntrospectionStatement(capability: str, sql: str, privilege_hint: str)`
  - `class WorkloadAdapter(ABC)` with `engine`, `introspection_sql()`, `connect()`, `fetch_workload()`, `fetch_schema()`, `fetch_table_facts()`, `propose()`, `render_ddl()`, and a concrete `degraded` list.
  - `get_workload_adapter(engine: str) -> WorkloadAdapter` (raises `ValueError`).

- [ ] **Step 1: Write the failing test**

Create `tests/test_workload_postgres.py`:

```python
import pytest

from sqlquality.workload import get_workload_adapter


def test_registry_returns_postgres_adapter():
    adapter = get_workload_adapter("postgres")
    assert adapter.engine == "postgres"


def test_registry_rejects_unsupported_engine_with_a_useful_message():
    with pytest.raises(ValueError) as exc:
        get_workload_adapter("duckdb")
    assert "duckdb" in str(exc.value)


def test_introspection_statements_are_named_and_carry_privilege_hints():
    statements = get_workload_adapter("postgres").introspection_sql()
    assert statements, "adapter must declare its introspection statements for --dry-run"
    capabilities = {s.capability for s in statements}
    assert "workload" in capabilities
    assert all(s.privilege_hint for s in statements)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_workload_postgres.py -v`
Expected: FAIL with `ImportError: cannot import name 'get_workload_adapter'`

- [ ] **Step 3: Write minimal implementation**

Create `src/sqlquality/workload/base.py`:

```python
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
    TableFacts,
    Workload,
    WorkloadFetch,
)

#: Executes one parameterized introspection statement and returns its rows.
#: Injectable so every fetch path is testable without a live database.
Querier = Callable[[str, tuple[object, ...]], list[tuple[object, ...]]]


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
        """Schema mapping for sqlglot qualify(): {table: {column: type}}."""

    @abstractmethod
    def fetch_table_facts(
        self, schemas: tuple[str, ...], tables: frozenset[str]
    ) -> dict[str, TableFacts]:
        """Row estimates, sizes, columns and per-column NDV for the given tables."""

    @abstractmethod
    def propose(
        self,
        aggregation: Aggregation,
        facts: dict[str, TableFacts],
        workload: Workload,
        *,
        min_cost_share: float,
    ) -> list[Proposal]:
        """Engine-specific proposal rules."""

    @abstractmethod
    def render_ddl(self, proposals: list[Proposal]) -> str:
        """A reviewable DDL script for the proposals that carry DDL. Never executed."""
```

Replace `src/sqlquality/workload/__init__.py` with:

```python
"""Workload analysis: query-history ingestion, column-usage rollup, per-engine adapters."""

from __future__ import annotations

from sqlquality.workload.base import WorkloadAdapter
from sqlquality.workload.postgres import PostgresWorkloadAdapter

_ADAPTERS: dict[str, type[WorkloadAdapter]] = {
    "postgres": PostgresWorkloadAdapter,
}


def get_workload_adapter(engine: str) -> WorkloadAdapter:
    """Return the workload adapter for an engine, or raise ValueError."""
    try:
        return _ADAPTERS[engine]()
    except KeyError:
        raise ValueError(
            f"No workload adapter for engine '{engine}'. "
            f"Supported: {', '.join(sorted(_ADAPTERS))}."
        )
```

This import will fail until Task 7 creates `postgres.py`. Create a minimal placeholder now so the registry test passes:

```python
# src/sqlquality/workload/postgres.py — expanded in Tasks 7-11
"""Postgres workload adapter."""

from __future__ import annotations

from datetime import timedelta

from sqlquality.models import (
    Aggregation, ConnectionParams, Proposal, TableFacts, Workload, WorkloadFetch,
)
from sqlquality.workload.base import IntrospectionStatement, WorkloadAdapter


class PostgresWorkloadAdapter(WorkloadAdapter):
    engine = "postgres"

    def introspection_sql(self) -> list[IntrospectionStatement]:
        return [
            IntrospectionStatement(
                capability="workload",
                sql="SELECT 1",
                privilege_hint="requires pg_read_all_stats or superuser",
            )
        ]

    def connect(self, params: ConnectionParams, timeout_s: int) -> None:
        raise NotImplementedError

    def fetch_workload(self, since: timedelta | None, limit: int) -> WorkloadFetch:
        raise NotImplementedError

    def fetch_schema(self, schemas: tuple[str, ...]) -> dict:
        raise NotImplementedError

    def fetch_table_facts(
        self, schemas: tuple[str, ...], tables: frozenset[str]
    ) -> dict[str, TableFacts]:
        raise NotImplementedError

    def propose(
        self, aggregation: Aggregation, facts: dict[str, TableFacts], workload: Workload,
        *, min_cost_share: float,
    ) -> list[Proposal]:
        raise NotImplementedError

    def render_ddl(self, proposals: list[Proposal]) -> str:
        raise NotImplementedError
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_workload_postgres.py -v && uv run mypy src/sqlquality/workload/`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/sqlquality/workload/ tests/test_workload_postgres.py
git commit -m "feat(workload): add WorkloadAdapter interface and registry"
```

---

### Task 6: Connection resolution

**Files:**
- Create: `src/sqlquality/workload/connection.py`, `src/sqlquality/workload/profiles.py`
- Test: `tests/test_workload_connection.py`

**Interfaces:**
- Consumes: `ConnectionParams` (Task 1).
- Produces:
  - `class ConnectionResolutionError(ValueError)`
  - `class ProfileError(ValueError)` (in `profiles.py`, re-raised as `ConnectionResolutionError`)
  - `resolve_connection(*, dsn: str | None, engine: str | None, profile: str | None, target: str | None, profiles_dir: Path | None, env: Mapping[str, str]) -> ConnectionParams`
  - `read_profile(profiles_dir: Path, profile: str, target: str | None, env: Mapping[str, str]) -> tuple[str, dict[str, str]]` returning `(engine, fields)`
  - `ENGINE_BY_SCHEME: dict[str, str]`

Precedence, per the spec: explicit `--dsn` > environment variable > `profiles.yml`. Engine is inferred from the DSN scheme or the dbt `type`, and an explicit `--engine` overrides.

- [ ] **Step 1: Write the failing test**

Create `tests/test_workload_connection.py`:

```python
from pathlib import Path

import pytest

from sqlquality.workload.connection import (
    ConnectionResolutionError,
    read_profile,
    resolve_connection,
)


def test_dsn_wins_over_env_and_profiles():
    params = resolve_connection(
        dsn="postgresql://u@h/db", engine=None, profile=None, target=None,
        profiles_dir=None, env={"SQLQUALITY_DSN": "postgresql://other@h/db"},
    )
    assert params.dsn == "postgresql://u@h/db"
    assert params.source == "--dsn"


def test_engine_inferred_from_dsn_scheme():
    for scheme, engine in [("postgresql", "postgres"), ("postgres", "postgres"),
                           ("redshift", "redshift"), ("snowflake", "snowflake")]:
        params = resolve_connection(
            dsn=f"{scheme}://u@h/db", engine=None, profile=None, target=None,
            profiles_dir=None, env={},
        )
        assert params.engine == engine


def test_explicit_engine_overrides_the_scheme():
    params = resolve_connection(
        dsn="postgresql://u@h/db", engine="redshift", profile=None, target=None,
        profiles_dir=None, env={},
    )
    assert params.engine == "redshift"


def test_env_dsn_used_when_no_flag():
    params = resolve_connection(
        dsn=None, engine=None, profile=None, target=None,
        profiles_dir=None, env={"SQLQUALITY_DSN": "postgresql://u@h/db"},
    )
    assert params.source == "env"
    assert params.engine == "postgres"


def test_unknown_dsn_scheme_is_an_error():
    with pytest.raises(ConnectionResolutionError) as exc:
        resolve_connection(dsn="mysql://u@h/db", engine=None, profile=None, target=None,
                           profiles_dir=None, env={})
    assert "mysql" in str(exc.value)


def test_nothing_supplied_is_an_error_naming_all_three_options():
    with pytest.raises(ConnectionResolutionError) as exc:
        resolve_connection(dsn=None, engine=None, profile=None, target=None,
                           profiles_dir=None, env={})
    message = str(exc.value)
    assert "--dsn" in message and "SQLQUALITY_DSN" in message and "--profile" in message


def test_read_profile_resolves_target_and_env_var(tmp_path):
    (tmp_path / "profiles.yml").write_text(
        """
jaffle:
  target: dev
  outputs:
    dev:
      type: postgres
      host: localhost
      port: 5432
      user: "{{ env_var('PGUSER') }}"
      dbname: analytics
      schema: public
"""
    )
    engine, fields = read_profile(tmp_path, "jaffle", None, {"PGUSER": "hans"})
    assert engine == "postgres"
    assert fields["user"] == "hans"
    assert fields["dbname"] == "analytics"


def test_read_profile_rejects_unknown_profile(tmp_path):
    (tmp_path / "profiles.yml").write_text("jaffle:\n  target: dev\n  outputs:\n    dev:\n      type: postgres\n")
    with pytest.raises(ConnectionResolutionError) as exc:
        read_profile(tmp_path, "nope", None, {})
    assert "nope" in str(exc.value)


def test_read_profile_missing_env_var_reports_the_variable_name(tmp_path):
    (tmp_path / "profiles.yml").write_text(
        "jaffle:\n  target: dev\n  outputs:\n    dev:\n      type: postgres\n"
        "      user: \"{{ env_var('PGUSER') }}\"\n"
    )
    with pytest.raises(ConnectionResolutionError) as exc:
        read_profile(tmp_path, "jaffle", None, {})
    assert "PGUSER" in str(exc.value)


def test_profiles_path_used_when_no_dsn(tmp_path):
    (tmp_path / "profiles.yml").write_text(
        "jaffle:\n  target: dev\n  outputs:\n    dev:\n      type: postgres\n      dbname: a\n"
    )
    params = resolve_connection(
        dsn=None, engine=None, profile="jaffle", target=None,
        profiles_dir=Path(tmp_path), env={},
    )
    assert params.source == "profiles.yml"
    assert params.engine == "postgres"
    assert params.dsn is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_workload_connection.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sqlquality.workload.connection'`

- [ ] **Step 3: Write minimal implementation**

Create `src/sqlquality/workload/profiles.py`:

```python
"""Read a dbt profiles.yml — a convenience for dbt users, never a requirement.

sqlquality is not a dbt tool: `advise` works against any database via --dsn or
SQLQUALITY_DSN. This module exists only so dbt users need not restate connection
details they already have.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

import yaml

#: dbt's env_var() Jinja call, the one templating form that appears in real profiles.
_ENV_VAR = re.compile(r"\{\{\s*env_var\(\s*['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]\s*\)\s*\}\}")

#: dbt adapter type -> sqlquality engine name.
ENGINE_BY_DBT_TYPE = {"postgres": "postgres", "redshift": "redshift", "snowflake": "snowflake"}


class ProfileError(ValueError):
    """Raised when profiles.yml is missing, malformed, or references an unset env var."""


def _interpolate(value: object, env: Mapping[str, str]) -> str:
    """Substitute env_var() references, or raise naming the missing variable."""
    text = str(value)

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in env:
            raise ProfileError(
                f"profiles.yml references env_var('{name}') but {name} is not set"
            )
        return env[name]

    return _ENV_VAR.sub(replace, text)


def read_profiles_file(profiles_dir: Path) -> dict:
    """Load profiles.yml from a directory, or raise ProfileError."""
    path = Path(profiles_dir) / "profiles.yml"
    try:
        raw = yaml.safe_load(path.read_text())
    except FileNotFoundError:
        raise ProfileError(f"No profiles.yml in {profiles_dir}")
    except OSError as exc:
        raise ProfileError(f"Could not read {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ProfileError(f"Malformed YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ProfileError(f"top-level of {path} must be a mapping")
    return raw


def read_output(
    profiles_dir: Path, profile: str, target: str | None, env: Mapping[str, str]
) -> tuple[str, dict[str, str]]:
    """Return (engine, connection fields) for one profile/target."""
    profiles = read_profiles_file(profiles_dir)
    block = profiles.get(profile)
    if not isinstance(block, dict):
        available = ", ".join(k for k in profiles if k != "config") or "none"
        raise ProfileError(f"No profile '{profile}' in profiles.yml (found: {available})")

    chosen = target or block.get("target")
    outputs = block.get("outputs")
    if not isinstance(outputs, dict) or chosen not in outputs:
        available = ", ".join(outputs) if isinstance(outputs, dict) else "none"
        raise ProfileError(
            f"No target '{chosen}' in profile '{profile}' (found: {available})"
        )
    output = outputs[chosen]
    if not isinstance(output, dict):
        raise ProfileError(f"Target '{chosen}' in profile '{profile}' must be a mapping")

    dbt_type = str(output.get("type", "")).lower()
    engine = ENGINE_BY_DBT_TYPE.get(dbt_type)
    if engine is None:
        raise ProfileError(
            f"dbt adapter type '{dbt_type}' has no workload adapter. "
            f"Supported: {', '.join(sorted(ENGINE_BY_DBT_TYPE))}."
        )
    fields = {k: _interpolate(v, env) for k, v in output.items() if k != "type"}
    return engine, fields
```

Create `src/sqlquality/workload/connection.py`:

```python
"""Resolve connection details: explicit flag > environment variable > profiles.yml."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from urllib.parse import urlparse

from sqlquality.models import ConnectionParams
from sqlquality.workload.profiles import ProfileError, read_output

#: Environment variable holding a full DSN.
DSN_ENV_VAR = "SQLQUALITY_DSN"

#: DSN URL scheme -> sqlquality engine name.
ENGINE_BY_SCHEME = {
    "postgresql": "postgres",
    "postgres": "postgres",
    "redshift": "redshift",
    "redshift+psycopg2": "redshift",
    "snowflake": "snowflake",
}


class ConnectionResolutionError(ValueError):
    """Raised when connection details cannot be resolved from any source."""


def _engine_from_dsn(dsn: str) -> str:
    scheme = urlparse(dsn).scheme.lower()
    engine = ENGINE_BY_SCHEME.get(scheme)
    if engine is None:
        raise ConnectionResolutionError(
            f"Unsupported DSN scheme {scheme or '(none)'!r}. "
            f"Supported: {', '.join(sorted(ENGINE_BY_SCHEME))}."
        )
    return engine


def read_profile(
    profiles_dir: Path, profile: str, target: str | None, env: Mapping[str, str]
) -> tuple[str, dict[str, str]]:
    """Read (engine, fields) from profiles.yml, re-raising as ConnectionResolutionError."""
    try:
        return read_output(profiles_dir, profile, target, env)
    except ProfileError as exc:
        raise ConnectionResolutionError(str(exc)) from exc


def resolve_connection(
    *,
    dsn: str | None,
    engine: str | None,
    profile: str | None,
    target: str | None,
    profiles_dir: Path | None,
    env: Mapping[str, str],
) -> ConnectionParams:
    """Resolve connection details, honoring flag > env > profiles.yml precedence."""
    if dsn:
        return ConnectionParams(
            engine=engine or _engine_from_dsn(dsn), dsn=dsn, fields={}, source="--dsn"
        )

    env_dsn = env.get(DSN_ENV_VAR)
    if env_dsn:
        return ConnectionParams(
            engine=engine or _engine_from_dsn(env_dsn), dsn=env_dsn, fields={}, source="env"
        )

    if profile:
        directory = profiles_dir or Path.home() / ".dbt"
        profile_engine, fields = read_profile(directory, profile, target, env)
        return ConnectionParams(
            engine=engine or profile_engine, dsn=None, fields=fields, source="profiles.yml"
        )

    raise ConnectionResolutionError(
        "No connection details. Pass --dsn, set SQLQUALITY_DSN, "
        "or pass --profile (with an optional --target) to read from profiles.yml."
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_workload_connection.py -v && uv run mypy src/sqlquality/workload/`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/sqlquality/workload/connection.py src/sqlquality/workload/profiles.py tests/test_workload_connection.py
git commit -m "feat(workload): resolve connections from dsn, env, or profiles.yml"
```

---

### Task 7: Postgres introspection statements

**Files:**
- Modify: `src/sqlquality/workload/postgres.py` (replaces the Task 5 placeholder)
- Test: `tests/test_workload_postgres.py`

**Interfaces:**
- Consumes: `IntrospectionStatement`, `WorkloadAdapter`, `Querier` (Task 5).
- Produces on `PostgresWorkloadAdapter`: `CAP_WORKLOAD`, `CAP_STATS_RESET`, `CAP_SCHEMA`, `CAP_TABLE_FACTS`, `CAP_NDV`, `CAP_INDEXES` module constants; `SQL: dict[str, str]` keyed by capability; `introspection_sql()` returning one `IntrospectionStatement` per capability.

The SQL text here is exercised by `--dry-run` snapshot tests, which prove it does not drift but cannot prove it is semantically valid against a real server. A live smoke test is the deferred docker-compose integration test noted in the spec.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_workload_postgres.py`:

```python
from sqlquality.workload.postgres import (
    CAP_INDEXES,
    CAP_NDV,
    CAP_SCHEMA,
    CAP_STATS_RESET,
    CAP_TABLE_FACTS,
    CAP_WORKLOAD,
    PostgresWorkloadAdapter,
)

EXPECTED_CAPABILITIES = {
    CAP_WORKLOAD, CAP_STATS_RESET, CAP_SCHEMA, CAP_TABLE_FACTS, CAP_NDV, CAP_INDEXES,
}


def test_every_capability_has_a_statement_and_a_hint():
    statements = PostgresWorkloadAdapter().introspection_sql()
    assert {s.capability for s in statements} == EXPECTED_CAPABILITIES
    for statement in statements:
        assert statement.sql.strip()
        assert statement.privilege_hint.strip()


def test_no_introspection_statement_writes():
    forbidden = ("insert", "update", "delete", "create", "drop", "alter", "truncate", "grant")
    for statement in PostgresWorkloadAdapter().introspection_sql():
        lowered = statement.sql.lower()
        for word in forbidden:
            assert f" {word} " not in f" {lowered} ", f"{statement.capability} contains {word}"


def test_workload_statement_is_scoped_to_the_current_database():
    sql = PostgresWorkloadAdapter().SQL[CAP_WORKLOAD].lower()
    assert "pg_stat_statements" in sql
    assert "current_database()" in sql
    assert "order by" in sql and "limit" in sql
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_workload_postgres.py -v`
Expected: FAIL with `ImportError: cannot import name 'CAP_INDEXES'`

- [ ] **Step 3: Write minimal implementation**

Replace `src/sqlquality/workload/postgres.py` with:

```python
"""Postgres workload adapter: pg_stat_statements + catalog introspection, index rules."""

from __future__ import annotations

from datetime import timedelta

from sqlquality.models import (
    Aggregation,
    ConnectionParams,
    Proposal,
    TableFacts,
    Workload,
    WorkloadFetch,
)
from sqlquality.workload.base import IntrospectionStatement, Querier, WorkloadAdapter

CAP_WORKLOAD = "workload"
CAP_STATS_RESET = "stats_reset"
CAP_SCHEMA = "schema"
CAP_TABLE_FACTS = "table_facts"
CAP_NDV = "ndv"
CAP_INDEXES = "indexes"

_HINTS = {
    CAP_WORKLOAD: (
        "requires the pg_stat_statements extension and pg_read_all_stats "
        "(or superuser); enable via shared_preload_libraries then CREATE EXTENSION"
    ),
    CAP_STATS_RESET: "requires read access to pg_stat_database",
    CAP_SCHEMA: "requires read access to information_schema.columns",
    CAP_TABLE_FACTS: "requires read access to pg_class and pg_namespace",
    CAP_NDV: "requires read access to pg_stats (per-column statistics)",
    CAP_INDEXES: "requires read access to pg_index and pg_stat_user_indexes",
}


class PostgresWorkloadAdapter(WorkloadAdapter):
    engine = "postgres"

    SQL: dict[str, str] = {
        CAP_WORKLOAD: """
            SELECT s.query, s.calls, s.total_exec_time, s.rows
            FROM pg_stat_statements s
            JOIN pg_database d ON d.oid = s.dbid
            WHERE d.datname = current_database()
            ORDER BY s.total_exec_time DESC
            LIMIT %s
        """,
        CAP_STATS_RESET: """
            SELECT stats_reset
            FROM pg_stat_database
            WHERE datname = current_database()
        """,
        CAP_SCHEMA: """
            SELECT c.table_name, c.column_name, c.data_type
            FROM information_schema.columns c
            WHERE c.table_schema = ANY(%s)
        """,
        CAP_TABLE_FACTS: """
            SELECT c.relname, c.reltuples::bigint, pg_total_relation_size(c.oid)
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE c.relkind = 'r' AND n.nspname = ANY(%s) AND c.relname = ANY(%s)
        """,
        CAP_NDV: """
            SELECT s.tablename, s.attname, s.n_distinct
            FROM pg_stats s
            WHERE s.schemaname = ANY(%s) AND s.tablename = ANY(%s)
        """,
        CAP_INDEXES: """
            SELECT t.relname, i.relname, a.attname, k.ordinality,
                   ix.indisunique, ix.indisprimary,
                   COALESCE(psui.idx_scan, 0), pg_relation_size(i.oid)
            FROM pg_index ix
            JOIN pg_class i ON i.oid = ix.indexrelid
            JOIN pg_class t ON t.oid = ix.indrelid
            JOIN pg_namespace n ON n.oid = t.relnamespace
            JOIN LATERAL unnest(ix.indkey) WITH ORDINALITY AS k(attnum, ordinality) ON TRUE
            JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = k.attnum
            LEFT JOIN pg_stat_user_indexes psui ON psui.indexrelid = i.oid
            WHERE n.nspname = ANY(%s) AND t.relname = ANY(%s)
            ORDER BY t.relname, i.relname, k.ordinality
        """,
    }

    def __init__(self, querier: Querier | None = None) -> None:
        super().__init__()
        self._query = querier

    def introspection_sql(self) -> list[IntrospectionStatement]:
        return [
            IntrospectionStatement(capability=cap, sql=sql.strip(), privilege_hint=_HINTS[cap])
            for cap, sql in self.SQL.items()
        ]

    def connect(self, params: ConnectionParams, timeout_s: int) -> None:
        raise NotImplementedError  # Task 8

    def fetch_workload(self, since: timedelta | None, limit: int) -> WorkloadFetch:
        raise NotImplementedError  # Task 8

    def fetch_schema(self, schemas: tuple[str, ...]) -> dict:
        raise NotImplementedError  # Task 8

    def fetch_table_facts(
        self, schemas: tuple[str, ...], tables: frozenset[str]
    ) -> dict[str, TableFacts]:
        raise NotImplementedError  # Task 8

    def propose(
        self,
        aggregation: Aggregation,
        facts: dict[str, TableFacts],
        workload: Workload,
        *,
        min_cost_share: float,
    ) -> list[Proposal]:
        raise NotImplementedError  # Tasks 9-10

    def render_ddl(self, proposals: list[Proposal]) -> str:
        raise NotImplementedError  # Task 11
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_workload_postgres.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/sqlquality/workload/postgres.py tests/test_workload_postgres.py
git commit -m "feat(workload): declare postgres introspection statements"
```

---

### Task 8: Postgres fetch layer

**Files:**
- Modify: `src/sqlquality/workload/postgres.py`, `pyproject.toml`
- Test: `tests/test_workload_postgres.py`

**Interfaces:**
- Consumes: `SQL`, capability constants (Task 7); `TableFacts`, `WorkloadFetch`, `RawQueryRow` (Task 1).
- Produces: working `connect()`, `fetch_workload()`, `fetch_schema()`, `fetch_table_facts()`, plus `PgIndex` and `fetch_indexes(schemas, tables) -> dict[str, tuple[PgIndex, ...]]` used by Tasks 9–10, and a `_run(capability, params)` helper that records degradation instead of raising.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_workload_postgres.py`:

```python
from sqlquality.models import ConnectionParams


class FakeQuerier:
    """Returns canned rows per capability, keyed by a distinctive SQL substring."""

    def __init__(self, rows_by_marker, fail_markers=()):
        self.rows_by_marker = rows_by_marker
        self.fail_markers = fail_markers
        self.calls = []

    def __call__(self, sql, params):
        self.calls.append((sql, params))
        for marker in self.fail_markers:
            if marker in sql:
                raise RuntimeError(f"permission denied for {marker}")
        for marker, rows in self.rows_by_marker.items():
            if marker in sql:
                return rows
        return []


def test_fetch_workload_maps_rows_and_reports_the_window():
    querier = FakeQuerier({
        "pg_stat_statements": [("select id from orders where status = $1", 10, 250.0, 100)],
        "pg_stat_database": [("2026-07-01 00:00:00",)],
    })
    fetch = PostgresWorkloadAdapter(querier=querier).fetch_workload(None, 500)
    assert fetch.rows[0].sql == "select id from orders where status = $1"
    assert fetch.rows[0].calls == 10
    assert fetch.rows[0].total_time_ms == 250.0
    assert "2026-07-01" in fetch.window_description


def test_fetch_workload_window_is_honest_that_since_is_not_supported():
    querier = FakeQuerier({
        "pg_stat_statements": [],
        "pg_stat_database": [("2026-07-01 00:00:00",)],
    })
    fetch = PostgresWorkloadAdapter(querier=querier).fetch_workload(timedelta(days=7), 500)
    assert "since stats reset" in fetch.window_description.lower()


def test_fetch_schema_builds_a_sqlglot_schema_mapping():
    querier = FakeQuerier({"information_schema.columns": [
        ("orders", "id", "integer"), ("orders", "status", "text"), ("customers", "id", "integer"),
    ]})
    schema = PostgresWorkloadAdapter(querier=querier).fetch_schema(("public",))
    assert schema == {
        "orders": {"id": "integer", "status": "text"},
        "customers": {"id": "integer"},
    }


def test_fetch_table_facts_resolves_negative_n_distinct_as_a_row_fraction():
    querier = FakeQuerier({
        "pg_total_relation_size": [("orders", 1000, 8192)],
        "information_schema.columns": [("orders", "id", "integer"), ("orders", "s", "text")],
        "pg_stats": [("orders", "id", 500.0), ("orders", "s", -0.25)],
    })
    facts = PostgresWorkloadAdapter(querier=querier).fetch_table_facts(
        ("public",), frozenset({"orders"})
    )
    assert facts["orders"].row_estimate == 1000
    assert facts["orders"].ndv["id"] == 500.0
    # -0.25 means "a quarter of the rows are distinct"
    assert facts["orders"].ndv["s"] == 250.0


def test_fetch_indexes_groups_columns_in_ordinal_order():
    querier = FakeQuerier({"pg_index": [
        ("orders", "orders_pkey", "id", 1, True, True, 900, 4096),
        ("orders", "idx_status_created", "status", 1, False, False, 0, 8192),
        ("orders", "idx_status_created", "created_at", 2, False, False, 0, 8192),
    ]})
    indexes = PostgresWorkloadAdapter(querier=querier).fetch_indexes(
        ("public",), frozenset({"orders"})
    )
    by_name = {i.name: i for i in indexes["orders"]}
    assert by_name["idx_status_created"].columns == ("status", "created_at")
    assert by_name["orders_pkey"].is_primary is True
    assert by_name["idx_status_created"].scans == 0


def test_a_denied_statement_degrades_and_names_the_privilege():
    querier = FakeQuerier({"information_schema.columns": []}, fail_markers=("pg_stats",))
    adapter = PostgresWorkloadAdapter(querier=querier)
    facts = adapter.fetch_table_facts(("public",), frozenset({"orders"}))
    assert facts == {} or facts["orders"].ndv == {}
    assert any(cap == CAP_NDV for cap, _ in adapter.degraded)
    assert any("pg_stats" in reason for _, reason in adapter.degraded)


def test_connect_without_psycopg_installed_raises_a_helpful_import_error(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_psycopg(name, *args, **kwargs):
        if name == "psycopg":
            raise ImportError("No module named 'psycopg'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_psycopg)
    adapter = PostgresWorkloadAdapter()
    params = ConnectionParams(engine="postgres", dsn="postgresql://u@h/db", fields={},
                              source="--dsn")
    with pytest.raises(ImportError) as exc:
        adapter.connect(params, 30)
    assert "sqlquality[postgres]" in str(exc.value)
```

Add `from datetime import timedelta` to the test module imports.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_workload_postgres.py -v`
Expected: FAIL with `NotImplementedError`

- [ ] **Step 3: Write minimal implementation**

In `src/sqlquality/workload/postgres.py`, add the `PgIndex` dataclass and replace the five `NotImplementedError` fetch methods:

```python
@dataclass(frozen=True)
class PgIndex:
    """One existing Postgres index, with its ordered column list and usage counter."""

    name: str
    columns: tuple[str, ...]
    is_unique: bool
    is_primary: bool
    scans: int
    size_bytes: int
```

Add `from dataclasses import dataclass` and `from sqlquality.models import RawQueryRow` to the imports. Then, on `PostgresWorkloadAdapter`:

```python
    def _run(self, capability: str, params: tuple[object, ...]) -> list[tuple[object, ...]]:
        """Run one introspection statement, recording degradation rather than raising.

        A single missing grant must cost only that capability — never the whole run.
        """
        if self._query is None:
            raise RuntimeError("connect() must be called before fetching")
        try:
            return self._query(self.SQL[capability], params)
        except Exception as exc:  # driver-specific; we only need the message
            self.degraded.append((capability, f"{exc} — {_HINTS[capability]}"))
            return []

    def connect(self, params: ConnectionParams, timeout_s: int) -> None:
        try:
            import psycopg
        except ImportError as exc:
            raise ImportError(
                "Postgres support requires psycopg. "
                "Install it with: pip install 'sqlquality[postgres]'"
            ) from exc

        conninfo = params.dsn or psycopg.conninfo.make_conninfo(**_pg_fields(params.fields))
        connection = psycopg.connect(conninfo, autocommit=True)
        with connection.cursor() as cursor:
            # Belt and braces: the session cannot write even if a statement tried to.
            cursor.execute("SET default_transaction_read_only = on")
            cursor.execute(f"SET statement_timeout = '{int(timeout_s)}s'")

        def query(sql: str, bind: tuple[object, ...]) -> list[tuple[object, ...]]:
            with connection.cursor() as cur:
                cur.execute(sql, bind)
                return list(cur.fetchall())

        self._query = query

    def fetch_workload(self, since: timedelta | None, limit: int) -> WorkloadFetch:
        rows = self._run(CAP_WORKLOAD, (limit,))
        reset = self._run(CAP_STATS_RESET, ())
        reset_at = reset[0][0] if reset and reset[0] else "an unknown time"
        # pg_stat_statements is cumulative since reset and carries no per-statement
        # timestamps before PG 17, so --since cannot be honored. Say so rather than
        # implying the requested window was applied.
        window = f"since stats reset at {reset_at}"
        if since is not None:
            window += " (--since is not supported by pg_stat_statements)"
        return WorkloadFetch(
            rows=tuple(
                RawQueryRow(sql=str(sql), calls=int(calls), total_time_ms=float(total_ms))
                for sql, calls, total_ms, _rows in rows
            ),
            window_description=window,
        )

    def fetch_schema(self, schemas: tuple[str, ...]) -> dict:
        schema: dict[str, dict[str, str]] = {}
        for table, column, data_type in self._run(CAP_SCHEMA, (list(schemas),)):
            schema.setdefault(str(table), {})[str(column)] = str(data_type)
        return schema

    def fetch_table_facts(
        self, schemas: tuple[str, ...], tables: frozenset[str]
    ) -> dict[str, TableFacts]:
        wanted = sorted(tables)
        sizes = {
            str(name): (int(rows), int(size) if size is not None else None)
            for name, rows, size in self._run(CAP_TABLE_FACTS, (list(schemas), wanted))
        }
        columns: dict[str, list[str]] = {}
        for table, column, _type in self._run(CAP_SCHEMA, (list(schemas),)):
            if str(table) in tables:
                columns.setdefault(str(table), []).append(str(column))

        ndv: dict[str, dict[str, float]] = {}
        for table, column, n_distinct in self._run(CAP_NDV, (list(schemas), wanted)):
            if n_distinct is None:
                continue
            value = float(n_distinct)
            row_estimate = sizes.get(str(table), (0, None))[0]
            # Postgres encodes "distinct as a fraction of row count" as a negative value.
            resolved = -value * row_estimate if value < 0 else value
            ndv.setdefault(str(table), {})[str(column)] = resolved

        facts: dict[str, TableFacts] = {}
        for table in wanted:
            rows, size = sizes.get(table, (None, None))
            facts[table] = TableFacts(
                name=table,
                row_estimate=rows,
                size_bytes=size,
                columns=tuple(columns.get(table, ())),
                ndv=ndv.get(table, {}),
            )
        return facts

    def fetch_indexes(
        self, schemas: tuple[str, ...], tables: frozenset[str]
    ) -> dict[str, tuple[PgIndex, ...]]:
        """Existing indexes per table, columns in ordinal order."""
        grouped: dict[tuple[str, str], dict[str, object]] = {}
        for row in self._run(CAP_INDEXES, (list(schemas), sorted(tables))):
            table, index, column, _ordinality, unique, primary, scans, size = row
            entry = grouped.setdefault(
                (str(table), str(index)),
                {
                    "columns": [],
                    "is_unique": bool(unique),
                    "is_primary": bool(primary),
                    "scans": int(scans),
                    "size_bytes": int(size) if size is not None else 0,
                },
            )
            columns = entry["columns"]
            assert isinstance(columns, list)
            columns.append(str(column))

        result: dict[str, list[PgIndex]] = {}
        for (table, index), entry in grouped.items():
            result.setdefault(table, []).append(
                PgIndex(
                    name=index,
                    columns=tuple(entry["columns"]),  # type: ignore[arg-type]
                    is_unique=bool(entry["is_unique"]),
                    is_primary=bool(entry["is_primary"]),
                    scans=int(entry["scans"]),  # type: ignore[call-overload]
                    size_bytes=int(entry["size_bytes"]),  # type: ignore[call-overload]
                )
            )
        return {table: tuple(indexes) for table, indexes in result.items()}
```

Add the field-mapping helper at module level:

```python
#: dbt profiles.yml field names -> libpq connection keywords.
_PG_FIELD_MAP = {"dbname": "dbname", "database": "dbname", "host": "host", "port": "port",
                 "user": "user", "username": "user", "password": "password"}


def _pg_fields(fields: dict[str, str]) -> dict[str, str]:
    """Translate profiles.yml keys to libpq keywords, dropping anything unrecognized."""
    return {_PG_FIELD_MAP[k]: v for k, v in fields.items() if k in _PG_FIELD_MAP}
```

In `pyproject.toml`, add to `[project.optional-dependencies]`:

```toml
postgres = ["psycopg[binary]>=3.1"]
warehouse = ["psycopg[binary]>=3.1"]
```

and add a mypy override:

```toml
[[tool.mypy.overrides]]
module = ["psycopg.*"]
ignore_missing_imports = true
```

`warehouse` currently duplicates `postgres`; Redshift and Snowflake drivers join it in their own plans.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_workload_postgres.py -v && uv run mypy src/sqlquality/workload/ && uv run ruff check src/`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/sqlquality/workload/postgres.py pyproject.toml tests/test_workload_postgres.py
git commit -m "feat(workload): postgres fetch layer with graceful privilege degradation"
```

---

### Task 9: Index proposals — ADV001, ADV002, ADV003

**Files:**
- Modify: `src/sqlquality/workload/postgres.py`
- Test: `tests/test_workload_rules.py`

**Interfaces:**
- Consumes: `ColumnUsage`, `ColumnRole`, `TableFacts`, `Proposal`, `Confidence` (Task 1); `PgIndex` (Task 8).
- Produces module-level pure functions:
  - `propose_indexes(usage, facts, existing, *, min_cost_share, min_rows=10_000, max_arity=3) -> list[Proposal]`
  - `propose_unused_indexes(existing, *, hot_tables) -> list[Proposal]`
  - `propose_redundant_indexes(existing) -> list[Proposal]`
  - `MIN_ROWS_FOR_INDEX = 10_000`, `MAX_INDEX_ARITY = 3`, `SELECTIVE_NDV = 100.0`

Two rules that must be stated because they are the difference between advice and noise:

- **Cost share for a multi-column proposal is the maximum over its contributing columns, never the sum.** Roles drawn from the same query overlap, so summing double-counts and manufactures shares above 100%.
- **Small tables are suppressed entirely.** Below `MIN_ROWS_FOR_INDEX` a sequential scan is the correct plan and an index is pure write overhead.

- [ ] **Step 1: Write the failing test**

Create `tests/test_workload_rules.py`:

```python
from sqlquality.models import ColumnRole, ColumnUsage, Confidence, TableFacts
from sqlquality.workload.postgres import (
    PgIndex,
    propose_indexes,
    propose_redundant_indexes,
    propose_unused_indexes,
)


def usage(column, role, cost_share=0.5, cost_ms=50.0, table="orders"):
    return ColumnUsage(table=table, column=column, role=role, calls=10, cost_ms=cost_ms,
                       cost_share=cost_share, fingerprints=1)


def facts(rows=1_000_000, ndv=None, columns=("id", "status", "created_at", "customer_id")):
    return {"orders": TableFacts(name="orders", row_estimate=rows, size_bytes=10**8,
                                 columns=columns, ndv=ndv or {})}


def codes(proposals):
    return [p.code for p in proposals]


def test_equality_then_range_ordering_in_the_candidate_index():
    proposals = propose_indexes(
        [usage("status", ColumnRole.EQUALITY, cost_ms=90.0),
         usage("created_at", ColumnRole.RANGE, cost_ms=80.0)],
        facts(), {}, min_cost_share=0.01,
    )
    assert codes(proposals) == ["ADV001"]
    assert proposals[0].evidence["columns"] == ("status", "created_at")


def test_only_one_range_column_is_included():
    proposals = propose_indexes(
        [usage("status", ColumnRole.EQUALITY, cost_ms=90.0),
         usage("created_at", ColumnRole.RANGE, cost_ms=80.0),
         usage("shipped_at", ColumnRole.RANGE, cost_ms=70.0)],
        facts(columns=("status", "created_at", "shipped_at")), {}, min_cost_share=0.01,
    )
    assert proposals[0].evidence["columns"] == ("status", "created_at")


def test_arity_is_capped():
    proposals = propose_indexes(
        [usage("a", ColumnRole.EQUALITY, cost_ms=99.0),
         usage("b", ColumnRole.EQUALITY, cost_ms=98.0),
         usage("c", ColumnRole.EQUALITY, cost_ms=97.0),
         usage("d", ColumnRole.EQUALITY, cost_ms=96.0)],
        facts(columns=("a", "b", "c", "d")), {}, min_cost_share=0.01,
    )
    assert len(proposals[0].evidence["columns"]) == 3


def test_small_tables_are_suppressed_entirely():
    proposals = propose_indexes(
        [usage("status", ColumnRole.EQUALITY)], facts(rows=500), {}, min_cost_share=0.01,
    )
    assert proposals == []


def test_below_min_cost_share_is_suppressed():
    proposals = propose_indexes(
        [usage("status", ColumnRole.EQUALITY, cost_share=0.001)],
        facts(), {}, min_cost_share=0.01,
    )
    assert proposals == []


def test_existing_index_with_the_same_leading_prefix_is_not_reproposed():
    existing = {"orders": (PgIndex("idx", ("status", "created_at"), False, False, 10, 8192),)}
    proposals = propose_indexes(
        [usage("status", ColumnRole.EQUALITY, cost_ms=90.0),
         usage("created_at", ColumnRole.RANGE, cost_ms=80.0)],
        facts(), existing, min_cost_share=0.01,
    )
    assert proposals == []


def test_a_wider_existing_index_still_covers_a_narrower_candidate():
    existing = {"orders": (PgIndex("idx", ("status", "created_at", "id"), False, False, 5, 1),)}
    proposals = propose_indexes(
        [usage("status", ColumnRole.EQUALITY, cost_ms=90.0)], facts(), existing,
        min_cost_share=0.01,
    )
    assert proposals == []


def test_cost_share_is_the_max_never_the_sum():
    proposals = propose_indexes(
        [usage("status", ColumnRole.EQUALITY, cost_share=0.6, cost_ms=90.0),
         usage("created_at", ColumnRole.RANGE, cost_share=0.6, cost_ms=80.0)],
        facts(), {}, min_cost_share=0.01,
    )
    assert proposals[0].evidence["cost_share"] == 0.6


def test_confidence_is_high_only_with_stats_and_a_selective_leading_column():
    high = propose_indexes(
        [usage("status", ColumnRole.EQUALITY)], facts(ndv={"status": 5000.0}), {},
        min_cost_share=0.01,
    )
    assert high[0].confidence is Confidence.HIGH

    no_stats = propose_indexes(
        [usage("status", ColumnRole.EQUALITY)], facts(ndv={}), {}, min_cost_share=0.01,
    )
    assert no_stats[0].confidence is Confidence.MEDIUM

    unselective = propose_indexes(
        [usage("status", ColumnRole.EQUALITY)], facts(ndv={"status": 3.0}), {},
        min_cost_share=0.01,
    )
    assert unselective[0].confidence is Confidence.LOW


def test_unused_index_proposed_for_drop_but_never_a_constraint_index():
    existing = {"orders": (
        PgIndex("idx_cold", ("note",), False, False, 0, 4096),
        PgIndex("orders_pkey", ("id",), True, True, 0, 4096),
        PgIndex("uq_email", ("email",), True, False, 0, 4096),
        PgIndex("idx_warm", ("status",), False, False, 42, 4096),
    )}
    proposals = propose_unused_indexes(existing, hot_tables=frozenset({"orders"}))
    assert codes(proposals) == ["ADV002"]
    assert proposals[0].evidence["index"] == "idx_cold"
    assert proposals[0].confidence is Confidence.MEDIUM


def test_unused_index_rule_ignores_tables_outside_the_workload():
    existing = {"archive": (PgIndex("idx_cold", ("a",), False, False, 0, 1),)}
    assert propose_unused_indexes(existing, hot_tables=frozenset({"orders"})) == []


def test_redundant_prefix_index_proposed_for_drop():
    existing = {"orders": (
        PgIndex("idx_narrow", ("status",), False, False, 5, 1),
        PgIndex("idx_wide", ("status", "created_at"), False, False, 5, 1),
    )}
    proposals = propose_redundant_indexes(existing)
    assert codes(proposals) == ["ADV003"]
    assert proposals[0].evidence["index"] == "idx_narrow"
    assert proposals[0].confidence is Confidence.HIGH


def test_a_unique_prefix_index_is_never_called_redundant():
    existing = {"orders": (
        PgIndex("uq_status", ("status",), True, False, 5, 1),
        PgIndex("idx_wide", ("status", "created_at"), False, False, 5, 1),
    )}
    assert propose_redundant_indexes(existing) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_workload_rules.py -v`
Expected: FAIL with `ImportError: cannot import name 'propose_indexes'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/sqlquality/workload/postgres.py` (module level, after `PgIndex`):

```python
#: Below this row estimate a sequential scan is the right plan; an index is pure overhead.
MIN_ROWS_FOR_INDEX = 10_000
#: Wider composite indexes cost more to maintain than they repay in practice.
MAX_INDEX_ARITY = 3
#: An equality column with fewer distinct values than this is not selective enough to
#: justify HIGH confidence on its own.
SELECTIVE_NDV = 100.0


def _by_table(usage: Sequence[ColumnUsage]) -> dict[str, list[ColumnUsage]]:
    grouped: dict[str, list[ColumnUsage]] = {}
    for item in usage:
        grouped.setdefault(item.table, []).append(item)
    return grouped


def _covered(candidate: tuple[str, ...], existing: Sequence[PgIndex]) -> str | None:
    """Name of an existing index whose leading columns already cover ``candidate``."""
    for index in existing:
        if index.columns[: len(candidate)] == candidate:
            return index.name
    return None


def propose_indexes(
    usage: Sequence[ColumnUsage],
    facts: Mapping[str, TableFacts],
    existing: Mapping[str, Sequence[PgIndex]],
    *,
    min_cost_share: float,
    min_rows: int = MIN_ROWS_FOR_INDEX,
    max_arity: int = MAX_INDEX_ARITY,
) -> list[Proposal]:
    """ADV001 — composite index candidates: equality columns first, one range column last.

    Equality-then-range is the standard B-tree ordering: once a range predicate is used,
    later columns can no longer be probed by equality.
    """
    proposals: list[Proposal] = []
    for table, items in sorted(_by_table(usage).items()):
        table_facts = facts.get(table)
        rows = table_facts.row_estimate if table_facts else None
        if rows is not None and rows < min_rows:
            continue

        equality = sorted(
            (i for i in items if i.role is ColumnRole.EQUALITY),
            key=lambda i: i.cost_ms,
            reverse=True,
        )
        ranges = sorted(
            (i for i in items if i.role in (ColumnRole.RANGE, ColumnRole.SORT)),
            key=lambda i: i.cost_ms,
            reverse=True,
        )
        if ranges:
            chosen = equality[: max_arity - 1] + ranges[:1]
        else:
            chosen = equality[:max_arity]
        if not chosen:
            continue

        cost_share = max(i.cost_share for i in chosen)
        if cost_share < min_cost_share:
            continue

        columns = tuple(i.column for i in chosen)
        covered_by = _covered(columns, existing.get(table, ()))
        if covered_by is not None:
            continue

        ndv = table_facts.ndv if table_facts else {}
        leading_ndv = ndv.get(columns[0])
        if rows is None or leading_ndv is None:
            confidence = Confidence.MEDIUM
        elif leading_ndv >= SELECTIVE_NDV:
            confidence = Confidence.HIGH
        else:
            confidence = Confidence.LOW

        proposals.append(
            Proposal(
                code="ADV001",
                title=f"Add index on {table}({', '.join(columns)})",
                rationale=(
                    "These columns carry the table's hottest predicates and no existing "
                    "index leads with them. Equality columns come first so the range "
                    "column can be scanned last."
                ),
                evidence={
                    "table": table,
                    "columns": columns,
                    "roles": tuple(i.role.value for i in chosen),
                    "cost_share": cost_share,
                    "calls": max(i.calls for i in chosen),
                    "fingerprints": max(i.fingerprints for i in chosen),
                    "row_estimate": rows,
                    "leading_ndv": leading_ndv,
                },
                confidence=confidence,
                ddl=f"CREATE INDEX ON {table} ({', '.join(columns)});",
            )
        )
    return proposals


def propose_unused_indexes(
    existing: Mapping[str, Sequence[PgIndex]], *, hot_tables: frozenset[str]
) -> list[Proposal]:
    """ADV002 — indexes with zero recorded scans, excluding constraint-backing indexes.

    Confidence is capped at MEDIUM: idx_scan accumulates only since the last statistics
    reset, so zero scans cannot prove an index is unused across a full business cycle.
    """
    proposals: list[Proposal] = []
    for table in sorted(hot_tables):
        for index in existing.get(table, ()):
            if index.scans != 0 or index.is_unique or index.is_primary:
                continue
            proposals.append(
                Proposal(
                    code="ADV002",
                    title=f"Drop unused index {index.name} on {table}",
                    rationale=(
                        "No recorded scans since the last statistics reset. Verify the "
                        "reset time covers a full business cycle before dropping."
                    ),
                    evidence={
                        "table": table,
                        "index": index.name,
                        "columns": index.columns,
                        "scans": index.scans,
                        "size_bytes": index.size_bytes,
                    },
                    confidence=Confidence.MEDIUM,
                    ddl=f"DROP INDEX {index.name};",
                )
            )
    return proposals


def propose_redundant_indexes(
    existing: Mapping[str, Sequence[PgIndex]],
) -> list[Proposal]:
    """ADV003 — an index whose column list is a strict prefix of another's is redundant."""
    proposals: list[Proposal] = []
    for table, indexes in sorted(existing.items()):
        for narrow in indexes:
            if narrow.is_unique or narrow.is_primary:
                continue
            wider = next(
                (
                    other
                    for other in indexes
                    if other.name != narrow.name
                    and len(other.columns) > len(narrow.columns)
                    and other.columns[: len(narrow.columns)] == narrow.columns
                ),
                None,
            )
            if wider is None:
                continue
            proposals.append(
                Proposal(
                    code="ADV003",
                    title=f"Drop redundant index {narrow.name} on {table}",
                    rationale=(
                        f"Its columns are a leading prefix of {wider.name}, which can "
                        "serve the same lookups."
                    ),
                    evidence={
                        "table": table,
                        "index": narrow.name,
                        "columns": narrow.columns,
                        "superseded_by": wider.name,
                        "superseding_columns": wider.columns,
                        "size_bytes": narrow.size_bytes,
                    },
                    confidence=Confidence.HIGH,
                    ddl=f"DROP INDEX {narrow.name};",
                )
            )
    return proposals
```

Add `from collections.abc import Mapping, Sequence` and extend the `sqlquality.models` import with `ColumnRole`, `ColumnUsage`, `Confidence`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_workload_rules.py -v && uv run mypy src/sqlquality/workload/`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/sqlquality/workload/postgres.py tests/test_workload_rules.py
git commit -m "feat(workload): propose, drop and dedupe postgres indexes (ADV001-003)"
```

---

### Task 10: Remaining rules — ADV004, ADV005, ADV006, and `propose()`

**Files:**
- Modify: `src/sqlquality/workload/postgres.py`
- Test: `tests/test_workload_rules.py`

**Interfaces:**
- Consumes: everything from Task 9; `FLAG_LEADING_WILDCARD_LIKE`, `FLAG_SELECT_STAR` (Task 2); `QueryStat`, `Workload`, `Aggregation` (Task 1).
- Produces:
  - `propose_partial_indexes(usage, facts, *, min_cost_share) -> list[Proposal]`
  - `propose_sargability(usage, workload, *, min_cost_share) -> list[Proposal]`
  - `propose_select_star(workload, facts, *, min_cost_share, min_columns=15) -> list[Proposal]`
  - `WIDE_TABLE_COLUMNS = 15`
  - `PostgresWorkloadAdapter.propose()` composing all six rules, sorted by confidence then cost share.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_workload_rules.py`:

```python
from sqlquality.models import Aggregation, QueryStat, Workload
from sqlquality.workload.fingerprint import FLAG_LEADING_WILDCARD_LIKE, FLAG_SELECT_STAR
from sqlquality.workload.postgres import (
    PostgresWorkloadAdapter,
    propose_partial_indexes,
    propose_sargability,
    propose_select_star,
)


def _workload(*stats):
    return Workload(stats=tuple(stats), window_description="w")


def test_partial_index_proposed_for_a_hot_not_null_check():
    proposals = propose_partial_indexes(
        [usage("status", ColumnRole.EQUALITY, cost_ms=90.0),
         usage("shipped_at", ColumnRole.NOT_NULL_CHECK, cost_share=0.4, cost_ms=40.0)],
        facts(), min_cost_share=0.01,
    )
    assert codes(proposals) == ["ADV004"]
    assert "IS NOT NULL" in proposals[0].ddl


def test_partial_index_polarity_follows_the_predicate():
    proposals = propose_partial_indexes(
        [usage("status", ColumnRole.EQUALITY, cost_ms=90.0),
         usage("shipped_at", ColumnRole.NULL_CHECK, cost_share=0.4, cost_ms=40.0)],
        facts(), min_cost_share=0.01,
    )
    assert "IS NULL" in proposals[0].ddl
    assert "IS NOT NULL" not in proposals[0].ddl


def test_no_partial_index_without_an_equality_column_to_index():
    proposals = propose_partial_indexes(
        [usage("shipped_at", ColumnRole.NOT_NULL_CHECK, cost_share=0.4)],
        facts(), min_cost_share=0.01,
    )
    assert proposals == []


def test_non_sargable_column_gets_an_attributed_proposal():
    proposals = propose_sargability(
        [usage("status", ColumnRole.NON_SARGABLE, cost_share=0.3)],
        _workload(), min_cost_share=0.01,
    )
    assert codes(proposals) == ["ADV005"]
    assert proposals[0].evidence["column"] == "status"
    assert proposals[0].confidence is Confidence.HIGH


def test_leading_wildcard_reports_without_column_attribution():
    stat = QueryStat(fingerprint="fp", sql="select id from orders where note like $1",
                     calls=5, total_time_ms=100.0,
                     flags=frozenset({FLAG_LEADING_WILDCARD_LIKE}))
    proposals = propose_sargability([], _workload(stat), min_cost_share=0.01)
    assert codes(proposals) == ["ADV005"]
    assert proposals[0].evidence.get("column") is None
    # Redaction erased the pattern, so we can name the query group but not the column.
    assert proposals[0].confidence is Confidence.MEDIUM


def test_hot_select_star_on_a_wide_table():
    stat = QueryStat(fingerprint="fp", sql="select * from orders", calls=5,
                     total_time_ms=100.0, flags=frozenset({FLAG_SELECT_STAR}))
    wide = {"orders": TableFacts(name="orders", row_estimate=10**6, size_bytes=10**8,
                                 columns=tuple(f"c{i}" for i in range(30)))}
    proposals = propose_select_star(_workload(stat), wide, min_cost_share=0.01)
    assert codes(proposals) == ["ADV006"]


def test_select_star_ignored_on_a_narrow_table():
    stat = QueryStat(fingerprint="fp", sql="select * from orders", calls=5,
                     total_time_ms=100.0, flags=frozenset({FLAG_SELECT_STAR}))
    narrow = {"orders": TableFacts(name="orders", row_estimate=10**6, size_bytes=1,
                                   columns=("a", "b"))}
    assert propose_select_star(_workload(stat), narrow, min_cost_share=0.01) == []


def test_propose_composes_all_rules_and_ranks_high_confidence_first():
    aggregation = Aggregation(
        usage=(usage("status", ColumnRole.EQUALITY, cost_share=0.6, cost_ms=60.0),
               usage("note", ColumnRole.NON_SARGABLE, cost_share=0.2, cost_ms=20.0)),
        total_cost_ms=100.0, skipped_unqualifiable=0, tables=frozenset({"orders"}),
    )
    adapter = PostgresWorkloadAdapter(querier=lambda sql, params: [])
    proposals = adapter.propose(
        aggregation, facts(ndv={"status": 9999.0}), _workload(), min_cost_share=0.01,
    )
    assert {p.code for p in proposals} >= {"ADV001", "ADV005"}
    assert proposals[0].confidence is Confidence.HIGH
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_workload_rules.py -v`
Expected: FAIL with `ImportError: cannot import name 'propose_partial_indexes'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/sqlquality/workload/postgres.py`:

```python
#: A table with at least this many columns makes `SELECT *` materially wasteful.
WIDE_TABLE_COLUMNS = 15

_NULL_ROLE_PREDICATE = {
    ColumnRole.NULL_CHECK: "IS NULL",
    ColumnRole.NOT_NULL_CHECK: "IS NOT NULL",
}


def propose_partial_indexes(
    usage: Sequence[ColumnUsage],
    facts: Mapping[str, TableFacts],
    *,
    min_cost_share: float,
) -> list[Proposal]:
    """ADV004 — index the hot equality column, restricted by a hot null-check predicate.

    Only structural predicates qualify. Literal-valued partial indexes would need the
    literals retained, which default redaction deliberately discards.
    """
    proposals: list[Proposal] = []
    for table, items in sorted(_by_table(usage).items()):
        equality = sorted(
            (i for i in items if i.role is ColumnRole.EQUALITY),
            key=lambda i: i.cost_ms,
            reverse=True,
        )
        null_checks = sorted(
            (i for i in items if i.role in _NULL_ROLE_PREDICATE),
            key=lambda i: i.cost_ms,
            reverse=True,
        )
        if not equality or not null_checks:
            continue
        leading, guard = equality[0], null_checks[0]
        cost_share = max(leading.cost_share, guard.cost_share)
        if cost_share < min_cost_share:
            continue
        predicate = _NULL_ROLE_PREDICATE[guard.role]
        table_facts = facts.get(table)
        proposals.append(
            Proposal(
                code="ADV004",
                title=(
                    f"Partial index on {table}({leading.column}) "
                    f"WHERE {guard.column} {predicate}"
                ),
                rationale=(
                    "The hot predicates always pair this lookup with the same null check, "
                    "so a partial index covers them at a fraction of the size."
                ),
                evidence={
                    "table": table,
                    "columns": (leading.column,),
                    "guard_column": guard.column,
                    "guard_predicate": predicate,
                    "cost_share": cost_share,
                    "calls": max(leading.calls, guard.calls),
                    "row_estimate": table_facts.row_estimate if table_facts else None,
                },
                confidence=Confidence.MEDIUM,
                ddl=(
                    f"CREATE INDEX ON {table} ({leading.column}) "
                    f"WHERE {guard.column} {predicate};"
                ),
            )
        )
    return proposals


def propose_sargability(
    usage: Sequence[ColumnUsage],
    workload: Workload,
    *,
    min_cost_share: float,
) -> list[Proposal]:
    """ADV005 — predicates an index cannot serve, ranked by the cost they carry."""
    proposals: list[Proposal] = []
    for item in sorted(usage, key=lambda i: i.cost_ms, reverse=True):
        if item.role is not ColumnRole.NON_SARGABLE or item.cost_share < min_cost_share:
            continue
        proposals.append(
            Proposal(
                code="ADV005",
                title=f"Non-sargable predicate on {item.table}.{item.column}",
                rationale=(
                    "The column is wrapped in a cast or function inside a predicate, so a "
                    "plain B-tree index cannot be used. Rewrite the predicate to leave the "
                    "column bare, or add a matching expression index."
                ),
                evidence={
                    "table": item.table,
                    "column": item.column,
                    "cost_share": item.cost_share,
                    "calls": item.calls,
                    "fingerprints": item.fingerprints,
                },
                confidence=Confidence.HIGH,
                ddl=None,
            )
        )

    total = workload.total_cost_ms
    for stat in workload.stats:
        if FLAG_LEADING_WILDCARD_LIKE not in stat.flags:
            continue
        share = (stat.total_time_ms / total) if total else 0.0
        if share < min_cost_share:
            continue
        proposals.append(
            Proposal(
                code="ADV005",
                title="Leading-wildcard LIKE in a hot query group",
                rationale=(
                    "A LIKE pattern beginning with '%' cannot use a B-tree index. Consider "
                    "a trigram index or full-text search. The pattern itself was redacted, "
                    "so the specific column is not attributed here."
                ),
                evidence={
                    "fingerprint": stat.fingerprint,
                    "sql": stat.sql,
                    "cost_share": share,
                    "calls": stat.calls,
                },
                confidence=Confidence.MEDIUM,
                ddl=None,
            )
        )
    return proposals


def propose_select_star(
    workload: Workload,
    facts: Mapping[str, TableFacts],
    *,
    min_cost_share: float,
    min_columns: int = WIDE_TABLE_COLUMNS,
) -> list[Proposal]:
    """ADV006 — hot query groups projecting a star from a wide table."""
    wide = {name for name, fact in facts.items() if len(fact.columns) >= min_columns}
    if not wide:
        return []
    total = workload.total_cost_ms
    proposals: list[Proposal] = []
    for stat in workload.stats:
        if FLAG_SELECT_STAR not in stat.flags:
            continue
        touched = sorted(name for name in wide if name in stat.sql)
        if not touched:
            continue
        share = (stat.total_time_ms / total) if total else 0.0
        if share < min_cost_share:
            continue
        proposals.append(
            Proposal(
                code="ADV006",
                title=f"Hot SELECT * over wide table(s): {', '.join(touched)}",
                rationale=(
                    "Projecting every column of a wide table moves data no consumer asked "
                    "for. List the columns the query actually needs."
                ),
                evidence={
                    "tables": tuple(touched),
                    "column_counts": {name: len(facts[name].columns) for name in touched},
                    "cost_share": share,
                    "calls": stat.calls,
                    "fingerprint": stat.fingerprint,
                },
                confidence=Confidence.MEDIUM,
                ddl=None,
            )
        )
    return proposals
```

Replace `PostgresWorkloadAdapter.propose`:

```python
    #: Highest confidence first, then largest cost share — the reading order a human wants.
    _CONFIDENCE_ORDER = {Confidence.HIGH: 0, Confidence.MEDIUM: 1, Confidence.LOW: 2}

    def propose(
        self,
        aggregation: Aggregation,
        facts: dict[str, TableFacts],
        workload: Workload,
        *,
        min_cost_share: float,
    ) -> list[Proposal]:
        existing = self.fetch_indexes(self.schemas, aggregation.tables)
        proposals = [
            *propose_indexes(
                aggregation.usage, facts, existing, min_cost_share=min_cost_share
            ),
            *propose_partial_indexes(aggregation.usage, facts, min_cost_share=min_cost_share),
            *propose_sargability(aggregation.usage, workload, min_cost_share=min_cost_share),
            *propose_select_star(workload, facts, min_cost_share=min_cost_share),
            *propose_unused_indexes(existing, hot_tables=aggregation.tables),
            *propose_redundant_indexes(existing),
        ]
        return sorted(
            proposals,
            key=lambda p: (
                self._CONFIDENCE_ORDER[p.confidence],
                -float(p.evidence.get("cost_share", 0.0)),  # type: ignore[arg-type]
            ),
        )
```

`propose()` reads `self.schemas`, which `WorkloadAdapter.__init__` defaults to `("public",)` (Task 5). The CLI overwrites it from `--schema` in Task 13. Tests in this task rely on the default, so they stay self-contained.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_workload_rules.py -v && uv run mypy src/sqlquality/workload/`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/sqlquality/workload/postgres.py tests/test_workload_rules.py
git commit -m "feat(workload): partial-index, sargability and select-star rules (ADV004-006)"
```

---

### Task 11: DDL rendering

**Files:**
- Modify: `src/sqlquality/workload/postgres.py`
- Test: `tests/test_workload_rules.py`

**Interfaces:**
- Consumes: `Proposal` (Task 1).
- Produces: `PostgresWorkloadAdapter.render_ddl(proposals) -> str`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_workload_rules.py`:

```python
from sqlquality.models import Proposal


def test_render_ddl_emits_a_reviewable_commented_script():
    proposals = [
        Proposal(code="ADV001", title="Add index on orders(status)", rationale="hot",
                 evidence={"cost_share": 0.5}, confidence=Confidence.HIGH,
                 ddl="CREATE INDEX ON orders (status);"),
        Proposal(code="ADV005", title="Non-sargable predicate", rationale="cast",
                 evidence={"cost_share": 0.2}, confidence=Confidence.HIGH, ddl=None),
    ]
    script = PostgresWorkloadAdapter().render_ddl(proposals)
    assert "CREATE INDEX ON orders (status);" in script
    assert "-- ADV001" in script
    assert "high" in script
    assert "Non-sargable" not in script  # no DDL, so it belongs in the report only
    assert "review" in script.lower()


def test_render_ddl_with_no_ddl_proposals_still_explains_itself():
    script = PostgresWorkloadAdapter().render_ddl([])
    assert script.strip().startswith("--")
    assert "no ddl" in script.lower()


def test_render_ddl_recommends_concurrently_for_index_creation():
    proposals = [
        Proposal(code="ADV001", title="t", rationale="r", evidence={},
                 confidence=Confidence.HIGH, ddl="CREATE INDEX ON orders (status);"),
    ]
    script = PostgresWorkloadAdapter().render_ddl(proposals)
    assert "CONCURRENTLY" in script
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_workload_rules.py -k render_ddl -v`
Expected: FAIL with `NotImplementedError`

- [ ] **Step 3: Write minimal implementation**

Replace `PostgresWorkloadAdapter.render_ddl`:

```python
    def render_ddl(self, proposals: list[Proposal]) -> str:
        """A commented, reviewable script. sqlquality never executes this."""
        header = [
            "-- Generated by `sqlquality advise` — REVIEW BEFORE RUNNING.",
            "-- sqlquality does not execute this script and has not validated it against",
            "-- your workload's write patterns. Each statement is advisory.",
            "--",
            "-- On a live table prefer CREATE INDEX CONCURRENTLY / DROP INDEX CONCURRENTLY:",
            "-- the plain forms below take a lock that blocks writes for the duration.",
            "",
        ]
        body: list[str] = []
        for proposal in proposals:
            if not proposal.ddl:
                continue
            share = proposal.evidence.get("cost_share")
            share_text = f", {float(share):.1%} of workload cost" if share is not None else ""
            body.append(f"-- {proposal.code} [{proposal.confidence.value}{share_text}]")
            body.append(f"-- {proposal.title}")
            body.append(proposal.ddl)
            body.append("")
        if not body:
            body = ["-- No DDL proposals — every finding is advisory-only.", ""]
        return "\n".join(header + body)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_workload_rules.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/sqlquality/workload/postgres.py tests/test_workload_rules.py
git commit -m "feat(workload): render reviewable DDL scripts"
```

---

### Task 12: Report rendering

**Files:**
- Modify: `src/sqlquality/report.py`
- Test: `tests/test_report_markdown.py`

**Interfaces:**
- Consumes: `Proposal`, `Workload`, `Aggregation`, `Confidence` (Task 1).
- Produces:
  - `advise_payload(proposals, workload, aggregation, *, engine, redacted, degraded) -> dict`
  - `render_advise_markdown(proposals, workload, aggregation, *, engine, redacted, degraded) -> str`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_report_markdown.py`:

```python
from sqlquality.models import Aggregation, Confidence, Proposal, QueryStat, Workload
from sqlquality.report import advise_payload, render_advise_markdown

PROPOSALS = [
    Proposal(code="ADV001", title="Add index on orders(status)", rationale="hot predicate",
             evidence={"cost_share": 0.42, "calls": 100, "table": "orders"},
             confidence=Confidence.HIGH, ddl="CREATE INDEX ON orders (status);"),
]
WORKLOAD = Workload(
    stats=(QueryStat(fingerprint="fp", sql="select id from orders where status = $1",
                     calls=100, total_time_ms=500.0),),
    window_description="since stats reset at 2026-07-01",
    skipped_unparseable=2,
    skipped_noise=7,
)
AGGREGATION = Aggregation(usage=(), total_cost_ms=500.0, skipped_unqualifiable=3,
                          tables=frozenset({"orders"}))


def _payload():
    return advise_payload(PROPOSALS, WORKLOAD, AGGREGATION, engine="postgres",
                          redacted=True, degraded=[("ndv", "permission denied")])


def test_payload_reports_proposals_window_and_skips():
    payload = _payload()
    assert payload["engine"] == "postgres"
    assert payload["redacted"] is True
    assert payload["window"] == "since stats reset at 2026-07-01"
    assert payload["proposals"][0]["code"] == "ADV001"
    assert payload["skipped"] == {"unparseable": 2, "noise": 7, "unqualifiable": 3}
    assert payload["degraded"] == [{"capability": "ndv", "reason": "permission denied"}]


def test_payload_is_json_serializable():
    import json

    json.dumps(_payload())


def test_markdown_shows_confidence_and_cost_share():
    md = render_advise_markdown(PROPOSALS, WORKLOAD, AGGREGATION, engine="postgres",
                                redacted=True, degraded=[])
    assert "ADV001" in md
    assert "high" in md
    assert "42.0%" in md


def test_markdown_discloses_the_window_and_the_skips():
    md = render_advise_markdown(PROPOSALS, WORKLOAD, AGGREGATION, engine="postgres",
                                redacted=True, degraded=[])
    assert "since stats reset at 2026-07-01" in md
    assert "2" in md and "7" in md and "3" in md


def test_markdown_escapes_pipes_from_query_text():
    hostile = [
        Proposal(code="ADV005", title="a | b", rationale="r", evidence={"cost_share": 0.1},
                 confidence=Confidence.LOW, ddl=None),
    ]
    md = render_advise_markdown(hostile, WORKLOAD, AGGREGATION, engine="postgres",
                                redacted=True, degraded=[])
    assert "a \\| b" in md


def test_markdown_states_when_no_proposals_were_produced():
    md = render_advise_markdown([], WORKLOAD, AGGREGATION, engine="postgres",
                                redacted=True, degraded=[])
    assert "no proposals" in md.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_report_markdown.py -v`
Expected: FAIL with `ImportError: cannot import name 'advise_payload'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/sqlquality/report.py` (and extend the imports with `Aggregation`, `Proposal`, `Workload` from `sqlquality.models`):

```python
def advise_payload(
    proposals: list[Proposal],
    workload: Workload,
    aggregation: Aggregation,
    *,
    engine: str,
    redacted: bool,
    degraded: list[tuple[str, str]],
) -> dict:
    """JSON-serializable summary of an advise run."""
    return {
        "engine": engine,
        "redacted": redacted,
        "window": workload.window_description,
        "analyzed": {
            "query_groups": len(workload.stats),
            "total_cost_ms": workload.total_cost_ms,
            "tables": sorted(aggregation.tables),
        },
        "skipped": {
            "unparseable": workload.skipped_unparseable,
            "noise": workload.skipped_noise,
            "unqualifiable": aggregation.skipped_unqualifiable,
        },
        "degraded": [{"capability": cap, "reason": reason} for cap, reason in degraded],
        "proposals": [
            {
                "code": p.code,
                "title": p.title,
                "rationale": p.rationale,
                "confidence": p.confidence.value,
                # Evidence values include tuples and frozensets; normalize for JSON.
                "evidence": {k: _jsonable(v) for k, v in p.evidence.items()},
                "ddl": p.ddl,
            }
            for p in proposals
        ],
    }


def _jsonable(value: object) -> object:
    """Coerce evidence values (tuples, sets) into JSON-friendly types."""
    if isinstance(value, (tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def render_advise_markdown(
    proposals: list[Proposal],
    workload: Workload,
    aggregation: Aggregation,
    *,
    engine: str,
    redacted: bool,
    degraded: list[tuple[str, str]],
) -> str:
    """Render advise proposals as markdown (suitable for a ticket or PR comment)."""
    lines = [
        f"# sqlquality advise — {_md_escape(engine)}",
        "",
        f"**Window:** {_md_escape(workload.window_description)}",
        f"**Query groups analyzed:** {len(workload.stats)}  ",
        f"**Literals:** {'redacted' if redacted else 'retained (--keep-literals)'}",
        "",
        (
            f"Skipped: {workload.skipped_unparseable} unparseable, "
            f"{workload.skipped_noise} introspection/DDL, "
            f"{aggregation.skipped_unqualifiable} unresolvable against the schema."
        ),
        "",
    ]
    if degraded:
        lines.append("## Reduced coverage")
        lines.append("")
        for capability, reason in degraded:
            lines.append(f"- `{_md_escape(capability)}`: {_md_escape(reason)}")
        lines.append("")

    if not proposals:
        lines.append("No proposals — nothing in the analyzed workload met the thresholds.")
        return "\n".join(lines) + "\n"

    lines += [
        "| code | confidence | cost share | proposal |",
        "|---|---|---:|---|",
    ]
    for p in proposals:
        share = p.evidence.get("cost_share")
        share_text = f"{float(share):.1%}" if isinstance(share, (int, float)) else "—"
        lines.append(
            f"| {_md_escape(p.code)} | {p.confidence.value} | {share_text} "
            f"| {_md_escape(p.title)} |"
        )
    lines.append("")
    lines.append("## Detail")
    lines.append("")
    for p in proposals:
        lines.append(f"### {_md_escape(p.code)} — {_md_escape(p.title)}")
        lines.append("")
        lines.append(_md_escape(p.rationale))
        lines.append("")
        evidence = ", ".join(f"{k}={_md_escape(v)}" for k, v in sorted(p.evidence.items()))
        lines.append(f"Evidence: {evidence}")
        lines.append("")
        if p.ddl:
            lines.append("```sql")
            lines.append(p.ddl)
            lines.append("```")
            lines.append("")
    return "\n".join(lines) + "\n"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_report_markdown.py tests/test_report.py -v && uv run mypy src/sqlquality/report.py`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/sqlquality/report.py tests/test_report_markdown.py
git commit -m "feat(report): render advise proposals as JSON and markdown"
```

---

### Task 13: Wire the `advise` CLI command

**Files:**
- Modify: `src/sqlquality/cli.py`
- Test: `tests/test_advise_cli.py`

**Interfaces:**
- Consumes: everything from Tasks 1–12.
- Produces: `sqlquality advise` with the spec's full flag set, assigning `adapter.schemas` from `--schema` before `connect()` (the attribute is declared on the ABC in Task 5).

- [ ] **Step 1: Write the failing test**

Create `tests/test_advise_cli.py`:

```python
import json

import pytest
from typer.testing import CliRunner

from sqlquality.cli import app

runner = CliRunner()


def test_dry_run_prints_statements_and_never_connects(monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("--dry-run must not connect")

    monkeypatch.setattr(
        "sqlquality.workload.postgres.PostgresWorkloadAdapter.connect", explode
    )
    result = runner.invoke(app, ["advise", "--engine", "postgres", "--dry-run"])
    assert result.exit_code == 0
    assert "pg_stat_statements" in result.stdout


def test_dry_run_needs_no_credentials():
    result = runner.invoke(app, ["advise", "--engine", "postgres", "--dry-run"])
    assert result.exit_code == 0


def test_missing_credentials_exit_2():
    result = runner.invoke(app, ["advise"])
    assert result.exit_code == 2
    assert "--dsn" in result.output


def test_unsupported_engine_exit_2():
    result = runner.invoke(app, ["advise", "--dsn", "postgresql://u@h/db",
                                 "--engine", "duckdb"])
    assert result.exit_code == 2


def test_bad_dsn_scheme_exit_2():
    result = runner.invoke(app, ["advise", "--dsn", "mysql://u@h/db"])
    assert result.exit_code == 2


def test_malformed_since_exit_2():
    result = runner.invoke(app, ["advise", "--dsn", "postgresql://u@h/db",
                                 "--since", "banana"])
    assert result.exit_code == 2


def _stub_adapter(monkeypatch, rows):
    """Replace connect() with an injected fake querier."""
    from sqlquality.workload.postgres import PostgresWorkloadAdapter

    def fake_connect(self, params, timeout_s):
        self.schemas = ("public",)

        def query(sql, bind):
            for marker, result in rows.items():
                if marker in sql:
                    return result
            return []

        self._query = query

    monkeypatch.setattr(PostgresWorkloadAdapter, "connect", fake_connect)


WIDE_COLUMNS = [("orders", "id", "integer"), ("orders", "status", "text"),
                ("orders", "created_at", "timestamp")]


def test_successful_run_exits_0_and_emits_json(monkeypatch):
    _stub_adapter(monkeypatch, {
        "pg_stat_statements": [
            ("select id from orders where status = $1 and created_at > $2", 100, 5000.0, 10),
        ],
        "pg_stat_database": [("2026-07-01",)],
        "information_schema.columns": WIDE_COLUMNS,
        "pg_total_relation_size": [("orders", 5_000_000, 10**8)],
        "pg_stats": [("orders", "status", 5000.0)],
        "pg_index": [],
    })
    result = runner.invoke(app, ["advise", "--dsn", "postgresql://u@h/db", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["engine"] == "postgres"
    assert payload["redacted"] is True
    assert any(p["code"] == "ADV001" for p in payload["proposals"])


def test_empty_workload_exits_0(monkeypatch):
    _stub_adapter(monkeypatch, {"pg_stat_statements": [], "pg_stat_database": [("2026-07-01",)]})
    result = runner.invoke(app, ["advise", "--dsn", "postgresql://u@h/db", "--json"])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["proposals"] == []


def test_ddl_and_markdown_files_are_written(monkeypatch, tmp_path):
    _stub_adapter(monkeypatch, {
        "pg_stat_statements": [
            ("select id from orders where status = $1 and created_at > $2", 100, 5000.0, 10),
        ],
        "pg_stat_database": [("2026-07-01",)],
        "information_schema.columns": WIDE_COLUMNS,
        "pg_total_relation_size": [("orders", 5_000_000, 10**8)],
        "pg_stats": [("orders", "status", 5000.0)],
        "pg_index": [],
    })
    ddl = tmp_path / "out.sql"
    md = tmp_path / "out.md"
    result = runner.invoke(app, ["advise", "--dsn", "postgresql://u@h/db",
                                 "--ddl", str(ddl), "--markdown", str(md)])
    assert result.exit_code == 0
    assert "CREATE INDEX" in ddl.read_text()
    assert "sqlquality advise" in md.read_text()
    assert "REVIEW BEFORE RUNNING" in ddl.read_text()


def test_connection_failure_exits_2(monkeypatch):
    from sqlquality.workload.postgres import PostgresWorkloadAdapter

    def boom(self, params, timeout_s):
        raise RuntimeError("could not connect to server")

    monkeypatch.setattr(PostgresWorkloadAdapter, "connect", boom)
    result = runner.invoke(app, ["advise", "--dsn", "postgresql://u@h/db"])
    assert result.exit_code == 2
    assert "could not connect" in result.output


def test_missing_driver_exits_2_with_install_hint(monkeypatch):
    from sqlquality.workload.postgres import PostgresWorkloadAdapter

    def no_driver(self, params, timeout_s):
        raise ImportError("Install it with: pip install 'sqlquality[postgres]'")

    monkeypatch.setattr(PostgresWorkloadAdapter, "connect", no_driver)
    result = runner.invoke(app, ["advise", "--dsn", "postgresql://u@h/db"])
    assert result.exit_code == 2
    assert "sqlquality[postgres]" in result.output
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_advise_cli.py -v`
Expected: FAIL — `advise` is not a command (exit code 2 with "No such command")

- [ ] **Step 3: Write minimal implementation**

`adapter.schemas` already exists (declared on the ABC in Task 5, read by `propose()` in Task 10), so no adapter change is needed here — the CLI just assigns it before calling `connect()`.

Add to `src/sqlquality/cli.py`:

```python
_SINCE_UNITS = {"h": "hours", "d": "days", "w": "weeks"}


def _parse_since(value: str | None) -> timedelta | None:
    """Parse a '7d' / '24h' / '2w' duration, or exit 2."""
    if value is None:
        return None
    match = re.fullmatch(r"(\d+)([hdw])", value.strip().lower())
    if match is None:
        typer.echo(
            f"Could not parse --since {value!r}. Use a count and a unit, e.g. 24h, 7d, 2w.",
            err=True,
        )
        raise typer.Exit(code=2)
    return timedelta(**{_SINCE_UNITS[match.group(2)]: int(match.group(1))})


@app.command()
def advise(
    engine: str | None = typer.Option(
        None, "--engine", help="postgres | redshift | snowflake. Inferred from the DSN if unset."
    ),
    dsn: str | None = typer.Option(None, "--dsn", help="Database URL. Overrides SQLQUALITY_DSN."),
    profile: str | None = typer.Option(None, "--profile", help="dbt profile name (optional)."),
    target: str | None = typer.Option(None, "--target", help="dbt target within the profile."),
    profiles_dir: Path | None = typer.Option(
        None, "--profiles-dir", help="Directory holding profiles.yml (default: ~/.dbt)."
    ),
    schema: list[str] = typer.Option(
        ["public"], "--schema", help="Schema(s) to introspect. Repeatable."
    ),
    since: str | None = typer.Option(
        None, "--since", help="Window, e.g. 7d. Not supported by pg_stat_statements."
    ),
    limit: int = typer.Option(500, "--limit", help="Max query-history rows to read."),
    min_cost_share: float = typer.Option(
        0.01, "--min-cost-share", help="Suppress proposals below this share of workload cost."
    ),
    keep_literals: bool = typer.Option(
        False, "--keep-literals", help="Do NOT redact literal values from query text."
    ),
    timeout: int = typer.Option(30, "--timeout", help="Statement timeout in seconds."),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print the statements that would run, then exit without connecting."
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
    markdown: Path | None = typer.Option(None, "--markdown", help="Write a markdown report."),
    ddl: Path | None = typer.Option(None, "--ddl", help="Write proposed DDL for review."),
) -> None:
    """Propose database optimizations from query history and catalog metadata."""
    since_delta = _parse_since(since)

    # --dry-run must work with no credentials at all: it is how you audit what we would run.
    if dry_run:
        try:
            adapter = get_workload_adapter(engine or "postgres")
        except ValueError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=2)
        for statement in adapter.introspection_sql():
            typer.echo(f"-- {statement.capability}: {statement.privilege_hint}")
            typer.echo(statement.sql)
            typer.echo("")
        raise typer.Exit(code=0)

    try:
        params = resolve_connection(
            dsn=dsn, engine=engine, profile=profile, target=target,
            profiles_dir=profiles_dir, env=os.environ,
        )
        adapter = get_workload_adapter(params.engine)
    except (ConnectionResolutionError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2)

    typer.echo(f"engine: {params.engine} (credentials from {params.source})", err=True)

    schemas = tuple(schema)
    adapter.schemas = schemas
    try:
        adapter.connect(params, timeout)
    except ImportError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2)
    except Exception as exc:  # driver-specific connection failures
        typer.echo(f"Could not connect: {exc}", err=True)
        raise typer.Exit(code=2)

    fetch = adapter.fetch_workload(since_delta, limit)
    workload = ingest(fetch, params.engine, keep_literals=keep_literals)
    db_schema = adapter.fetch_schema(schemas)
    aggregation = aggregate(workload, db_schema, params.engine)
    facts = adapter.fetch_table_facts(schemas, aggregation.tables)
    proposals = adapter.propose(aggregation, facts, workload, min_cost_share=min_cost_share)

    payload = advise_payload(
        proposals, workload, aggregation,
        engine=params.engine, redacted=not keep_literals, degraded=adapter.degraded,
    )
    if markdown is not None:
        markdown.write_text(
            render_advise_markdown(
                proposals, workload, aggregation,
                engine=params.engine, redacted=not keep_literals, degraded=adapter.degraded,
            )
        )
    if ddl is not None:
        ddl.write_text(adapter.render_ddl(proposals))

    if json_out:
        typer.echo(json.dumps(payload, indent=2, sort_keys=True))
        raise typer.Exit(code=0)

    for capability, reason in adapter.degraded:
        typer.echo(f"reduced coverage — {capability}: {reason}", err=True)
    typer.echo(f"window: {workload.window_description}", err=True)

    table = Table(
        title=(
            f"Advise — {params.engine} "
            f"({len(proposals)} proposals, {len(workload.stats)} query groups)"
        )
    )
    table.add_column("code")
    table.add_column("conf")
    table.add_column("cost share", justify="right")
    table.add_column("proposal")
    for proposal in proposals:
        share = proposal.evidence.get("cost_share")
        share_text = f"{float(share):.1%}" if isinstance(share, (int, float)) else "—"
        table.add_row(proposal.code, proposal.confidence.value, share_text, proposal.title)
    console.print(table)
    # Proposals are advisory: advise never gates.
    raise typer.Exit(code=0)
```

Add to the `cli.py` imports: `import os`, `import re`, `from datetime import timedelta`, `from sqlquality.report import advise_payload, render_advise_markdown`, `from sqlquality.workload import get_workload_adapter`, `from sqlquality.workload.aggregate import aggregate`, `from sqlquality.workload.connection import ConnectionResolutionError, resolve_connection`, `from sqlquality.workload.fingerprint import ingest`.

Update the `app` help text to `"Measure dbt model performance and complexity, and advise on database optimizations."`

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest -v && uv run mypy src/ && uv run ruff check src/ tests/`
Expected: all PASS — the full suite, since `cli.py` changed

- [ ] **Step 5: Commit**

```bash
git add src/sqlquality/cli.py src/sqlquality/workload/postgres.py tests/test_advise_cli.py
git commit -m "feat(cli): add advise command"
```

---

### Task 14: End-to-end redaction guarantee

**Files:**
- Create: `tests/test_workload_redaction.py`

**Interfaces:**
- Consumes: the whole pipeline (Tasks 1–13).
- Produces: no source changes — this task is a guarantee, and it must fail loudly if any later change weakens redaction.

This is the compliance-critical test. Every other test checks a unit; this one asserts the end-to-end property the design promises: **with default settings, no literal from the query log reaches any output.**

- [ ] **Step 1: Write the failing test**

Create `tests/test_workload_redaction.py`:

```python
"""Guarantee: with default settings, no literal from the query log reaches any output.

Query history can contain personal data in predicates. This is the test that fails if a
future change lets a literal escape into a report, a DDL script, or the JSON payload.
"""

import json

import pytest
from typer.testing import CliRunner

from sqlquality.cli import app

runner = CliRunner()

#: Distinctive strings that must never appear downstream. If any of these leaks, the
#: redaction path is broken.
SECRETS = ("patient-4711", "hans@betterdoc.de", "DE89370400440532013000", "1990-04-17")

HOT_QUERY = (
    "select id, note from orders "
    "where status = 'shipped' "
    "and customer_email = 'hans@betterdoc.de' "
    "and reference = 'patient-4711' "
    "and iban = 'DE89370400440532013000' "
    "and created_at > '1990-04-17' "
    "order by created_at desc"
)

COLUMNS = [
    ("orders", "id", "integer"), ("orders", "note", "text"), ("orders", "status", "text"),
    ("orders", "customer_email", "text"), ("orders", "reference", "text"),
    ("orders", "iban", "text"), ("orders", "created_at", "timestamp"),
]

ROWS = {
    "pg_stat_statements": [(HOT_QUERY, 500, 90_000.0, 10)],
    "pg_stat_database": [("2026-07-01",)],
    "information_schema.columns": COLUMNS,
    "pg_total_relation_size": [("orders", 8_000_000, 10**9)],
    "pg_stats": [("orders", "status", 4.0), ("orders", "customer_email", 900_000.0)],
    "pg_index": [],
}


@pytest.fixture
def stubbed(monkeypatch):
    from sqlquality.workload.postgres import PostgresWorkloadAdapter

    def fake_connect(self, params, timeout_s):
        def query(sql, bind):
            for marker, result in ROWS.items():
                if marker in sql:
                    return result
            return []

        self._query = query

    monkeypatch.setattr(PostgresWorkloadAdapter, "connect", fake_connect)


def test_no_literal_reaches_json_markdown_ddl_or_stdout(stubbed, tmp_path):
    md = tmp_path / "report.md"
    ddl = tmp_path / "proposals.sql"
    result = runner.invoke(app, ["advise", "--dsn", "postgresql://u@h/db",
                                 "--markdown", str(md), "--ddl", str(ddl), "--json"])
    assert result.exit_code == 0

    surfaces = {
        "stdout": result.stdout,
        "markdown": md.read_text(),
        "ddl": ddl.read_text(),
        "json": json.dumps(json.loads(result.stdout)),
    }
    for name, content in surfaces.items():
        for secret in SECRETS:
            assert secret not in content, f"{secret!r} leaked into {name}"


def test_analysis_still_works_on_redacted_sql(stubbed):
    """Redaction must not cost us the advice — the point of flags-before-redaction."""
    result = runner.invoke(app, ["advise", "--dsn", "postgresql://u@h/db", "--json"])
    payload = json.loads(result.stdout)
    assert payload["proposals"], "redaction must not silence the analysis"
    assert payload["redacted"] is True


def test_keep_literals_is_the_only_way_to_retain_values(stubbed):
    result = runner.invoke(app, ["advise", "--dsn", "postgresql://u@h/db",
                                 "--keep-literals", "--json"])
    payload = json.loads(result.stdout)
    assert payload["redacted"] is False
    # With the opt-in, values are retained — proving the default was doing real work.
    assert "hans@betterdoc.de" in json.dumps(payload)
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `uv run pytest tests/test_workload_redaction.py -v`
Expected: PASS. Unlike other tasks this test should pass immediately — it asserts a property Tasks 1–13 already built. If it FAILS, redaction has a real hole; fix the pipeline, not the test.

- [ ] **Step 3: Deliberately break redaction and confirm the test catches it**

Temporarily change `redact_tree` in `src/sqlquality/workload/fingerprint.py` to `return tree.copy()` without replacing literals.

Run: `uv run pytest tests/test_workload_redaction.py -v`
Expected: FAIL on `test_no_literal_reaches_json_markdown_ddl_or_stdout`. A guarantee test that cannot fail is worthless — this step proves it can.

Revert the change and re-run to confirm PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_workload_redaction.py
git commit -m "test(workload): guarantee no literal escapes into advise output"
```

---

### Task 15: Documentation

**Files:**
- Modify: `README.md`, `CHANGELOG.md`, `docs/superpowers/specs/2026-07-26-advise-workload-analysis-design.md`

**Interfaces:**
- Consumes: the shipped behavior of Tasks 1–14.
- Produces: no code.

The README's central claim is currently false once `advise` ships. Correcting it is part of the feature, not a follow-up.

- [ ] **Step 1: Correct the static-tool claim**

In `README.md`, replace the current opening paragraph:

> It is a static tool. It does not connect to your warehouse or execute queries:

with:

```markdown
sqlquality **never executes your SQL**. `complexity`, `lint`, `perf` and `check` are
fully offline and never open a connection. `advise` is the one exception: it opens a
**read-only** session to read query history and catalog metadata, using only a fixed set
of built-in introspection statements. Run `sqlquality advise --dry-run` to print every
statement it can issue, without connecting.
```

Keep the three existing bullets (complexity from the AST, performance as static analysis plus captured `EXPLAIN`, neighbors reported not scored) and add a fourth:

```markdown
- **Advice** is derived from your query history and catalog statistics, and is emitted as
  a report plus a DDL file for you to review and apply. sqlquality never writes to your
  database.
```

- [ ] **Step 2: Document the command**

Add an `advise` section to the Commands list and a full subsection after `perf`, covering: the flag table, the three credential sources and their precedence, `--dry-run`, the redaction default and `--keep-literals` (including the data-protection rationale), the `pg_stat_statements` prerequisite and the `--since` limitation, the ADV001–ADV006 code table, the confidence model, and the `sqlquality[postgres]` extra. Add `advise` to the Exit codes section noting it always exits 0 on success. Update the Limitations section: replace "Performance is static" with wording that scopes the claim to `perf`, and add that `advise` proposals are ranked by evidence rather than proven, that index write cost is not modeled, and that conclusions are only as representative as the log window.

- [ ] **Step 3: Update the CHANGELOG**

Add an `## [Unreleased]` section following the existing format:

```markdown
### Added
- `sqlquality advise` — reads Postgres query history (`pg_stat_statements`) and catalog
  metadata over a read-only connection and proposes indexes, index removals, partial
  indexes, sargability fixes and `SELECT *` cleanups (ADV001–ADV006), with a `--json`
  and `--markdown` report and a reviewable `--ddl` script.
- Connections resolve from `--dsn`, `SQLQUALITY_DSN`, or a dbt `profiles.yml`, in that
  order. dbt is optional throughout.
- `advise --dry-run` prints every introspection statement without connecting.
- Optional extras `sqlquality[postgres]` and `sqlquality[warehouse]`.

### Changed
- The "static tool, never connects" claim is now scoped: sqlquality never executes your
  SQL, and only `advise` opens a (read-only, metadata-only) connection.
- Query literals are redacted at ingest by default; `--keep-literals` opts back in.
```

- [ ] **Step 4: Reconcile the spec with what shipped**

Apply the five deviations listed in this plan's "Deviations from the spec" section to the spec document, so it describes the built system: `fetch_workload` returning `WorkloadFetch`, the three additional `ColumnRole` members, `skipped_unqualifiable` living on `Aggregation`, the `--schema` option, and ADV005's flag-driven leading-wildcard path.

- [ ] **Step 5: Verify and commit**

Run: `uv run pytest -v && uv run mypy src/ && uv run ruff check src/ tests/`
Expected: all PASS

```bash
git add README.md CHANGELOG.md docs/
git commit -m "docs: document advise and scope the static-tool claim"
```

---

## Self-Review

**Spec coverage.** Every section of the spec maps to a task: module layout → Tasks 2–8; `WorkloadAdapter` interface → Task 5; dataclasses → Task 1; data flow steps 1–8 → Tasks 6, 2, 2, 3, 4, 8, 9–10, 12; safety model → Tasks 5 (`degraded`), 7 (`introspection_sql`), 8 (`_run`, read-only session), 13 (`--dry-run`, `--timeout`); redaction → Tasks 2 and 14; CLI surface and exit codes → Task 13; Postgres rules ADV001–ADV006 → Tasks 9–10; confidence model → Task 9; error handling table → Tasks 8 and 13; testing → every task, with the fixture and redaction disciplines in Tasks 8 and 14; README rewrite → Task 15.

Deliberately out of this plan, per the spec's staged build order: Redshift (ADV101–105), Snowflake (ADV201–204), dbt enrichment (ADV301–303), and the opt-in docker-compose integration test. Each gets its own plan.

**Type consistency.** `WorkloadFetch` is produced by `fetch_workload` (Task 8) and consumed by `ingest` (Task 2). `Workload` is produced by `ingest` and consumed by `aggregate` (Task 4), `propose` (Tasks 9–10) and the renderers (Task 12). `Aggregation` is produced by `aggregate` and consumed by `propose` and the renderers. `PgIndex` is produced by `fetch_indexes` (Task 8) and consumed by all three index rules (Task 9). `Proposal` flows from Tasks 9–10 into Tasks 11–13. `ConnectionParams` is produced by `resolve_connection` (Task 6) and consumed by `connect` (Task 8). Capability constants are defined in Task 7 and used in Tasks 7, 8 and 13. `adapter.schemas` is declared on the `WorkloadAdapter` ABC (Task 5), read by `propose()` (Task 10), and assigned from `--schema` by the CLI (Task 13).

**Known plan-level risks.** Two places where an implementer may need to adapt:

1. Task 3's `_SARGABILITY_BREAKERS` assumes `exp.Lower` inherits from `exp.Func`. Step 4 of that task includes the exact command to verify and the fix if not.
2. Task 7's introspection SQL is verified only for drift, not semantics. The `pg_index`/`unnest ... WITH ORDINALITY` join in particular deserves a manual run against a real Postgres before release; the deferred integration test is where that belongs.
