# Design: `sqlquality advise` — workload-driven database optimization

Date: 2026-07-26
Status: approved design, not yet implemented

## Summary

Add a command that reads a database's **query history** and **catalog metadata** over a
read-only connection, aggregates which columns are actually filtered, joined, sorted and
grouped — weighted by what those queries cost — and proposes concrete optimizations:
indexes on Postgres, DISTKEY/SORTKEY/VACUUM on Redshift, clustering on Snowflake, plus
schema-redesign advice. Output is an advisory report and a reviewable DDL file.

This is the first sqlquality command that connects to a database.

## Motivation

Every static SQL linter can say "this predicate is non-sargable." None of them can say
"this predicate is non-sargable **and it accounts for 12% of your warehouse's execution
time**." Workload evidence is what turns a list of style nits into a ranked work queue,
and it is the thing sqlquality cannot get from a dbt manifest.

Connecting also unlocks catalog statistics, which is what separates a heuristic guess
from a defensible proposal:

| Without a connection | With a connection |
|---|---|
| "this column is filtered often" | "…has 40k distinct values over 8M rows, and no index" |
| propose indexes | also propose *dropping* indexes with zero scans |
| infer DISTKEY statically | compare against the table's actual current DISTKEY and skew |
| guess at scan cost | read `partitions_scanned / partitions_total` |

## The invariant, restated

The current README claims sqlquality "is a static tool. It does not connect to your
warehouse or execute queries." That claim narrows to:

> sqlquality never executes your SQL. `complexity`, `lint`, `perf` and `check` are fully
> offline. `advise` opens a **read-only** session to read workload and catalog metadata,
> using only a fixed set of built-in introspection statements.

This restatement must land in the README in the same commit that ships `advise`.

## Positioning

`advise` is **not** dbt-specific. It is a database performance advisor that works against
a bare database, with dbt as optional enrichment.

- Connection comes from a DSN or environment variables by default; `profiles.yml` is a
  convenience for users who happen to have a dbt project.
- The **live catalog** is the schema source for column resolution. dbt's `catalog.json` is
  not used — the connection already has better data.
- `--manifest` adds exactly three proposal classes (ADV301–ADV303). Without dbt you lose
  those and nothing else.

`check` remains the dbt-specific gate. `advise` sits beside `perf` as engine-facing.

## Architecture

### Module layout

```
src/sqlquality/workload/
  __init__.py      adapter registry (mirrors adapters/__init__.py)
  base.py          WorkloadAdapter ABC + shared dataclasses
  connection.py    resolve flag > env > profiles.yml -> ConnectionParams
  profiles.py      optional dbt profiles.yml reader, with env_var() interpolation
  fingerprint.py   literal erasure + canonical fingerprint
  extract.py       predicate/join/sort/group role extraction (engine-agnostic)
  aggregate.py     rollup: table -> column -> role -> weighted cost
  postgres.py      introspection SQL + index rules + DDL rendering
  redshift.py      introspection SQL + table-design rules + DDL rendering
  snowflake.py     introspection SQL + clustering rules + DDL rendering
  dbtenrich.py     ADV301-303, active only when --manifest is supplied
```

`extract.py`, `aggregate.py` and `fingerprint.py` are engine-agnostic and hold the bulk of
the logic. Each adapter owns only four things: its driver, its introspection statements,
its proposal rules, and its DDL syntax. That boundary is what makes the analysis testable
without a database.

`WorkloadAdapter` is a **new** interface, not an extension of `PerfAdapter`. `PerfAdapter`
analyzes one SQL string offline; `WorkloadAdapter` analyzes a corpus plus a live catalog.
Merging them would give every perf adapter methods it cannot implement.

### Interface

```python
class WorkloadAdapter(ABC):
    engine: str

    def introspection_sql(self) -> list[tuple[str, str]]:
        """(capability_name, sql) for every statement this adapter can run.
        Used by --dry-run, and by the degradation path to name what failed."""

    def connect(self, params: ConnectionParams, timeout_s: int) -> None:
        """Open a read-only session with a statement timeout."""

    def fetch_workload(self, since: timedelta | None, limit: int) -> Workload:
        """Query history rows, engine-normalized. Literals NOT yet redacted."""

    def fetch_schema(self, tables: set[str]) -> dict:
        """Schema mapping for sqlglot qualify()."""

    def fetch_table_facts(self, tables: set[str]) -> dict[str, TableFacts]:
        """Row estimates, sizes, per-column NDV, current physical design."""

    def propose(
        self,
        usage: tuple[ColumnUsage, ...],
        facts: dict[str, TableFacts],
        workload: Workload,
    ) -> list[Proposal]:
        """Engine-specific proposal rules."""

    def render_ddl(self, proposals: list[Proposal]) -> str:
        """Engine-specific DDL for the proposals that have any."""
```

### Core dataclasses

Added to `models.py` alongside the existing `Finding` / `ComplexityScore`:

```python
@dataclass(frozen=True)
class ConnectionParams:
    engine: str               # postgres | redshift | snowflake
    dsn: str | None           # when supplied directly
    fields: dict[str, str]    # host/port/user/database/schema/warehouse/role, as resolved
    source: str               # "--dsn" | "env" | "profiles.yml" — printed to stderr

@dataclass(frozen=True)
class QueryStat:
    fingerprint: str          # canonical, literal-free
    sql: str                  # redacted unless --keep-literals
    calls: int
    total_time_ms: float
    bytes_scanned: int | None      # None where the engine doesn't report it
    partitions_scanned: int | None  # Snowflake
    partitions_total: int | None    # Snowflake

@dataclass(frozen=True)
class Workload:
    stats: tuple[QueryStat, ...]
    window_description: str   # human-readable, honest about what the engine gave us
    skipped_unparseable: int
    skipped_unqualifiable: int

class ColumnRole(str, Enum):
    EQUALITY = "equality"
    RANGE = "range"
    JOIN = "join"
    SORT = "sort"
    GROUP = "group"

@dataclass(frozen=True)
class ColumnUsage:
    table: str
    column: str
    role: ColumnRole
    calls: int
    cost_ms: float
    cost_share: float         # fraction of total analyzed workload cost
    fingerprints: int

@dataclass(frozen=True)
class TableFacts:
    name: str
    row_estimate: int | None
    size_bytes: int | None
    columns: tuple[str, ...]
    ndv: dict[str, float]     # column -> distinct-value estimate; may be empty

class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass(frozen=True)
class Proposal:
    code: str                 # ADV001, ADV101, ...
    title: str
    rationale: str
    evidence: dict[str, object]   # cost_share, calls, fingerprints, rows, ndv, current state
    confidence: Confidence
    ddl: str | None           # None for redesign-class advice
```

`TableFacts` stays engine-neutral. Engine-specific physical facts (Postgres index lists,
Redshift `diststyle`/`sortkey`/`unsorted`, Snowflake `clustering_key`) are held privately
by each adapter rather than accumulating as optional fields on a shared god-object.

### Data flow

1. **Resolve connection.** Precedence: explicit flag > environment variable >
   `profiles.yml`. Engine is inferred from the DSN scheme or the dbt adapter type;
   `--engine` overrides. This mirrors how `check` resolves dialect from
   `manifest.adapter_type` and prints the resolution to stderr.
2. **Fetch workload, then parse once and redact.** A single parse per statement yields
   both the redacted SQL text retained for reporting and the canonical fingerprint.
   Redaction is the first transform applied — before aggregation, before any file is
   written, before any log line. Nothing downstream sees a literal unless
   `--keep-literals` was passed. Raw rows are not retained past this step.
3. **Roll up by fingerprint.** Exclude our own introspection statements, DDL,
   `information_schema` reads, and dbt's metadata queries so the tool does not advise on
   its own noise.
4. **Fetch schema and `qualify()`.** `sqlglot.optimizer.qualify.qualify` resolves
   unqualified columns to their tables. Queries that fail to qualify increment
   `skipped_unqualifiable` and appear in the report — never silently dropped, matching
   `check`'s existing `skipped` discipline.
5. **Extract column roles** and weight each by the cost share of the queries containing
   it. Cost weight is `total_time_ms`, or `bytes_scanned` on Snowflake.
6. **Fetch table facts only for tables above `--min-cost-share`.** Catalog round-trips
   stay proportional to what we will actually advise on.
7. **Propose**, collapse prefix-redundant candidates, assign confidence.
8. **Render** — rich table, `--json`, `--markdown`, `--ddl`.

### Safety model

- Only built-in, parameterized introspection statements are ever executed. No
  user-supplied SQL reaches the connection.
- Read-only transaction where the engine supports it (`SET TRANSACTION READ ONLY` on
  Postgres and Redshift).
- Statement timeout, `--timeout`, default 30 seconds.
- `--dry-run` prints the exact statements the adapter would run and exits 0 **without
  connecting**.
- Each introspection statement is wrapped individually. A missing grant degrades that one
  capability with the exact privilege named, and the command continues with whatever it
  did get.
- `advise` never issues DDL or DML. Generated DDL goes to a file for human review.

### Literal redaction

Redaction happens in `fingerprint.py` at ingest, by walking the parsed tree and replacing
every `exp.Literal` with a placeholder. Consequences, stated plainly:

- Index proposals are unaffected — fingerprinting erases literals anyway.
- ADV004 (partial indexes) can only key off structural predicates that survive redaction,
  such as `IS NULL` and boolean columns. Literal-valued partial indexes require
  `--keep-literals`.
- Hot-literal skew detection is unavailable by default. This is an accepted loss.

`--keep-literals` opts back in. The report header states which mode produced it.

## CLI surface

```
sqlquality advise [--engine postgres|redshift|snowflake]
                  [--dsn URL | --profile NAME --target NAME --profiles-dir DIR]
                  [--since 7d] [--limit 500] [--min-cost-share 0.01]
                  [--manifest target/manifest.json]
                  [--keep-literals] [--timeout 30] [--dry-run]
                  [--json] [--markdown FILE] [--ddl FILE]
```

Exit codes follow the existing project contract:

- **0** — analysis succeeded, whether or not proposals were produced. Proposals are
  advisory and never gate, matching `perf`'s treatment of anti-pattern warnings.
- **2** — usage, connection, or configuration error: unknown engine, unresolvable
  credentials, connection failure, missing driver, `pg_stat_statements` not installed,
  malformed `--since`.

Optional extras: `sqlquality[postgres]`, `sqlquality[redshift]`, `sqlquality[snowflake]`,
and `sqlquality[warehouse]` for all three. A missing driver degrades with an install hint,
exactly as the `llm` extra does today.

## Proposal rules

### Postgres

Workload from `pg_stat_statements` (`queryid`, `query`, `calls`, `total_exec_time`,
`rows`, `shared_blks_read`). Facts from `pg_class.reltuples`,
`pg_total_relation_size()`, `pg_stats` (`n_distinct`, `null_frac`, `correlation`),
`pg_index` joined to `pg_attribute` for ordered index column lists, and
`pg_stat_user_indexes.idx_scan`.

| Code | Proposal | Evidence |
|---|---|---|
| ADV001 | Composite index: equality columns ordered by weighted cost, then one range/sort column last, arity ≤ 3 | NDV, row estimate, absence of an existing index with that leading prefix |
| ADV002 | Drop unused index (`idx_scan = 0`), excluding unique, primary-key and FK-backing indexes | scans since stats reset, size reclaimed |
| ADV003 | Drop redundant index whose column list is a prefix of another | both column lists |
| ADV004 | Partial index for a hot fingerprint carrying a constant structural predicate | fingerprint count, cost share |
| ADV005 | Non-sargable hot predicate (`lower(col) =`, casts, leading wildcard) → rewrite or expression index | cost share |
| ADV006 | Hot `SELECT *` on a wide table | column count, cost share |

Proposals are suppressed entirely for tables below a row-count floor, where a sequential
scan is the correct plan and an index would be pure overhead.

**Window caveat.** `pg_stat_statements` is cumulative since the last reset and carries no
per-statement timestamps before PostgreSQL 17 added `stats_since`. On earlier versions
`--since` cannot be honored: the report states the window as "since stats reset at
`<timestamp>`" rather than implying the requested window was applied.

### Redshift

Workload from `SYS_QUERY_HISTORY` when available, falling back to `STL_QUERY` plus
`SVL_STATEMENTTEXT` reassembled in `sequence` order. The fallback path is subject to
`STL_QUERY.querytxt` truncation, which the report discloses. Facts from `SVV_TABLE_INFO`
(`diststyle`, `sortkey1`, `skew_rows`, `unsorted`, `stats_off`, `tbl_rows`, `size`),
`SVV_REDSHIFT_COLUMNS`, and `SVL_QUERY_SUMMARY` for disk-based spill.

Redshift has no indexes, so every proposal is physical design or maintenance:

| Code | Proposal |
|---|---|
| ADV101 | `ALTER DISTKEY` to the hottest join key on a large `DISTSTYLE EVEN` table, guarded against skew |
| ADV102 | `ALTER SORTKEY` to the hottest range/filter column |
| ADV103 | `VACUUM` when `unsorted` is high, `ANALYZE` when `stats_off` is high, on hot tables only |
| ADV104 | Attribute disk-based spill to the query group and the join or sort that caused it |
| ADV105 | `DISTSTYLE ALL` for a small dimension joined by many hot queries |

`SVV_ALTER_TABLE_RECOMMENDATIONS` is also read: agreement with Redshift Advisor raises
confidence, and disagreement is surfaced rather than hidden.

### Snowflake

Workload from `SNOWFLAKE.ACCOUNT_USAGE.QUERY_HISTORY`, falling back to the
`INFORMATION_SCHEMA.QUERY_HISTORY()` table function when the role lacks `IMPORTED
PRIVILEGES` on the `SNOWFLAKE` database. The fallback has a shorter retention window,
which the report discloses. `ACCOUNT_USAGE` latency means very recent queries may be
absent. Cost weight is `bytes_scanned`. Facts from `ACCOUNT_USAGE.TABLES`
(`row_count`, `bytes`, `clustering_key`).

| Code | Proposal |
|---|---|
| ADV201 | `CLUSTER BY` where `partitions_scanned / partitions_total` approaches 1 on a large table with a recurring predicate |
| ADV202 | Pruning-hostile predicate: function or cast applied to the clustering column |
| ADV203 | Search optimization candidate: highly selective point lookups on a large table |
| ADV204 | Hot `SELECT *` in a columnar store |

Clustering carries ongoing credit cost, so ADV201 and ADV203 must state that the
recommendation itself has a price — unlike an index, it is not a one-time cost.
`SYSTEM$CLUSTERING_INFORMATION` consumes compute and is therefore not called by default.

### dbt enrichment (`--manifest`)

| Code | Proposal | Note |
|---|---|---|
| ADV301 | Hot table maps to a model materialized as `view` → propose `table` or `incremental` | cost share attributed to the model |
| ADV302 | Model never referenced in the window → dead-model candidate | permanently LOW confidence: BI tools, longer windows and downstream-only models all hide usage |
| ADV303 | Recurring join path across a large cost share, all tables mapping to models → propose a mart | fingerprint count |

## Confidence model

Mechanical, derived from inputs rather than judgment:

- **HIGH** — cost share above threshold, **and** supporting catalog stats present, **and**
  the current physical state confirmed to lack the proposal.
- **MEDIUM** — cost evidence solid, but a catalog input is missing or stale.
- **LOW** — absence-based (ADV302) or thin evidence.

Every proposal renders its inputs inline: cost share, calls, distinct fingerprints, row
estimate, NDV, and current index/DISTKEY/clustering state. A reader must be able to
overrule any proposal from the report alone, without re-querying.

## Error handling

| Situation | Behavior |
|---|---|
| Driver extra not installed | Exit 2 with an install hint, mirroring the `llm` extra |
| Connection failure | Exit 2 with the engine's message, no stack trace |
| `pg_stat_statements` absent | Exit 2 explaining `shared_preload_libraries` + `CREATE EXTENSION` |
| Missing grant for one statement | That capability degrades, names the required privilege, analysis continues |
| Query fails to parse or qualify | Counted in `skipped_*` and reported; never silently dropped |
| Empty workload | Report "no workload matched", exit 0 |
| Malformed `--since` | Exit 2 |

## Testing

The engine-agnostic core carries the bulk of the tests and needs no database:

- `fingerprint`, `extract`, `aggregate` — unit tests over SQL strings.
- Proposal rules — pure functions of `ColumnUsage` + `TableFacts`, tested with
  hand-built fixtures. Includes the suppression cases: small tables, already-indexed
  prefixes, prefix-redundant candidate collapse.
- **Redaction test**: for a fixture containing distinctive literals, assert that no
  literal appears anywhere in the table, JSON, markdown, or DDL output. This is the
  compliance-critical test and gets an explicit, named case per engine.
- Adapters — fixture-driven. Recorded rows from each system view stored as JSON, fed
  through a fake connection object. This tests the introspection→dataclass mapping
  without a live warehouse, extending the existing `tests/fixtures/` convention.
- `--dry-run` snapshot tests, so introspection SQL cannot drift unnoticed.
- CLI — exit codes 0 and 2, and each degradation path, mirroring `tests/test_perf_cli.py`.

Live-database tests are out of scope for CI. An opt-in integration test against a
docker-compose Postgres with `pg_stat_statements` enabled is worth adding later, marked so
it never runs by default.

## Build order

Postgres end-to-end first, so the `WorkloadAdapter` interface is validated against a real
engine before Redshift and Snowflake are written against it. All three are in scope; only
the sequencing is staged.

1. `fingerprint`, `extract`, `aggregate`, dataclasses, redaction — no database.
2. `connection` / `profiles` resolution, `--dry-run`.
3. Postgres adapter, rules ADV001–ADV006, report and DDL rendering, CLI wiring.
4. README rewrite of the static-tool invariant.
5. Redshift adapter, ADV101–ADV105.
6. Snowflake adapter, ADV201–ADV204.
7. dbt enrichment, ADV301–ADV303.

Steps 1–4 are one implementation plan and deliver a shippable `advise` for Postgres.
Steps 5, 6 and 7 are separate follow-up plans against the interface step 3 validated.

## Out of scope

- Executing the generated DDL. `advise` never writes to a database; `--apply` is
  deliberately not built. Index creation can lock tables or run for hours, and that does
  not belong inside a linting tool.
- Gating on proposals. `advise` always exits 0 on success. A gate needs a stable
  confidence model to avoid flapping in CI, and that can only be judged after the
  confidence model has seen real workloads.
- BigQuery. The design accommodates it (`INFORMATION_SCHEMA.JOBS`, partition pruning
  advice) but it is not in this scope.
- Sending workload data to an LLM. `perf --suggest` already carries a data-egress warning
  for single files; a whole query history is a materially larger exposure and needs its
  own decision.
