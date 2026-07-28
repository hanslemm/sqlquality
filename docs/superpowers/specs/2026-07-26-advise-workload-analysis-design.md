# Design: `sqlquality advise` — workload-driven database optimization

Date: 2026-07-26
Status: Postgres (steps 1–4 below) shipped in `sqlquality advise`, now including ADV007
(join keys), ADV008 (`GROUP BY`), multi-schema `(schema, table)` keying, and
`DECLARE`/`COPY` unwrapping (Batch 2, 2026-07-27); optional dbt enrichment (ADV301–ADV303,
`--project-dir`/`--manifest`) shipped Batch 3a, 2026-07-28, with a code reassignment from
this document's original dbt section — see "Deviations from the spec (Batch 3a: dbt
enrichment)" below. Redshift and Snowflake (steps 5–6) remain design-only, not yet
implemented. See `docs/superpowers/plans/2026-07-26-advise-postgres.md` for the Postgres
implementation plan and its own "Deviations from the spec" section, reconciled into this
document below; `docs/superpowers/plans/2026-07-27-advise-dbt-enrichment.md` for the dbt
enrichment plan; "Deviations from the spec (Batch 2)" further down for what changed after
the initial Postgres ship; and "Deviations from the spec (Batch 3a: dbt enrichment)" for
what changed while building dbt enrichment.

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
  dbt.py           ADV301-303, active only when --project-dir/--manifest is supplied
                   (shipped filename; see Batch 3a deviation #2 below for why this module
                   is imported from cli.py only, never from an adapter)
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

    def fetch_workload(self, since: timedelta | None, limit: int) -> WorkloadFetch:
        """Raw query-history rows plus an honest window description. Literals NOT yet
        redacted — redaction happens once, in the engine-agnostic `ingest()`, so there is
        exactly one place to audit for literal leakage instead of one per adapter."""

    def fetch_schema(self, schemas: tuple[str, ...]) -> dict:
        """Nested schema mapping for sqlglot qualify(): {schema: {table: {column: type}}}.
        Nested, not flat, so qualify() can tell two same-named tables in different
        schemas apart — see "Deviations from the spec (Batch 2)" below."""

    def fetch_table_facts(
        self, schemas: tuple[str, ...], relations: frozenset[Relation]
    ) -> dict[Relation, TableFacts]:
        """Row estimates, sizes, per-column NDV, current physical design, keyed by the
        schema-qualified relation each row belongs to."""

    def propose(
        self,
        aggregation: Aggregation,
        facts: dict[Relation, TableFacts],
        workload: Workload,
        *,
        min_cost_share: float,
    ) -> list[Proposal]:
        """Engine-specific proposal rules. `aggregation.usage` carries the weighted
        column-role index; `aggregation.tables` and `.skipped_unqualifiable` drive the
        coverage disclosure."""

    #: Schema(s) to introspect (repeatable `--schema`, default `("public",)`). The CLI
    #: assigns this from the flag before connect(); read by fetch_schema/fetch_table_facts
    #: and by propose() wherever an adapter needs it (e.g. to re-fetch existing indexes).
    schemas: tuple[str, ...]

    def render_ddl(self, proposals: list[Proposal]) -> str:
        """Engine-specific DDL for the proposals that have any."""
```

### Core dataclasses

Added to `models.py` alongside the existing `Finding` / `ComplexityScore`:

```python
@dataclass(frozen=True, order=True)
class Relation:
    """A schema-qualified relation — the key every catalog fact is stored under.
    Bare table names aliased: two schemas each holding an `orders` merged into one entry,
    so the last catalog row read won the row estimate. `order=True` so rules can sort
    their output for stable report ordering."""
    schema: str
    table: str

    def __str__(self) -> str:
        return f"{self.schema}.{self.table}"

@dataclass(frozen=True)
class ConnectionParams:
    engine: str               # postgres | redshift | snowflake
    dsn: str | None           # when supplied directly
    fields: dict[str, str]    # host/port/user/database/schema/warehouse/role, as resolved
    source: str               # "--dsn" | "env" | "profiles.yml" — printed to stderr

@dataclass(frozen=True)
class RawQueryRow:
    """One row of query history as read from an engine, literals still present."""
    sql: str
    calls: int
    total_time_ms: float
    bytes_scanned: int | None = None       # None where the engine doesn't report it
    partitions_scanned: int | None = None  # Snowflake
    partitions_total: int | None = None    # Snowflake

@dataclass(frozen=True)
class WorkloadFetch:
    """What an adapter's fetch_workload() returns: raw rows plus an honest window label.
    Redaction has NOT happened yet — that is `ingest()`'s job, in the engine-agnostic
    core, so there is exactly one place to audit for literal leakage."""
    rows: tuple[RawQueryRow, ...]
    window_description: str

@dataclass(frozen=True)
class QueryStat:
    fingerprint: str          # canonical, literal-free
    sql: str                  # redacted unless --keep-literals
    calls: int
    total_time_ms: float
    bytes_scanned: int | None = None
    partitions_scanned: int | None = None  # Snowflake
    partitions_total: int | None = None    # Snowflake
    flags: frozenset[str] = frozenset()    # literal-derived signals captured pre-redaction

@dataclass(frozen=True)
class Workload:
    """Produced by `ingest()` from a `WorkloadFetch`: parsed, flagged, redacted, grouped."""
    stats: tuple[QueryStat, ...]
    window_description: str   # human-readable, honest about what the engine gave us
    skipped_unparseable: int = 0
    skipped_noise: int = 0    # our own introspection / DDL / session control, filtered out

class ColumnRole(str, Enum):
    EQUALITY = "equality"
    RANGE = "range"
    JOIN = "join"
    SORT = "sort"
    GROUP = "group"
    NULL_CHECK = "null_check"          # IS NULL
    NOT_NULL_CHECK = "not_null_check"  # IS NOT NULL
    NON_SARGABLE = "non_sargable"      # column wrapped in a cast or function inside a predicate

@dataclass(frozen=True)
class ColumnUsage:
    relation: Relation
    column: str
    role: ColumnRole
    calls: int
    cost_ms: float
    cost_share: float         # fraction of total analyzed workload cost — NOT a partition,
                              # see the "cost_share is not a partition" note in the README
    fingerprint_ids: frozenset[str] = frozenset()  # which query groups contributed this
                                                    # usage, so rules can test co-occurrence

    @property
    def fingerprints(self) -> int:
        return len(self.fingerprint_ids)

@dataclass(frozen=True)
class Aggregation:
    """Produced by `aggregate()`: the rolled-up usage index plus what could not be used."""
    usage: tuple[ColumnUsage, ...]
    total_cost_ms: float
    skipped_unqualifiable: int  # queries that failed qualify() — lives here, not on
                                # Workload, because qualification happens during
                                # aggregation, not ingest
    tables: frozenset[Relation]
    skipped_ambiguous: int = 0  # a bare table name held by 2+ introspected schemas,
                                # named without qualification — attributing it would be a
                                # coin flip, so it is counted and dropped instead

@dataclass(frozen=True)
class TableFacts:
    relation: Relation        # the schema-qualified key this table is stored under — not
                              # a display name; two same-named tables in different
                              # schemas each get their own TableFacts
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
2. **Fetch workload as a `WorkloadFetch`** (raw rows plus an honest window description),
   **then parse once and redact.** `ingest()` — engine-agnostic, in `fingerprint.py` —
   does this: a single parse per statement yields both the redacted SQL text retained
   for reporting and the canonical fingerprint. Redaction is the first transform applied
   — before aggregation, before any file is written, before any log line. Nothing
   downstream sees a literal unless `--keep-literals` was passed. Raw rows are not
   retained past this step. `ingest()` returns a `Workload`.
3. **Roll up by fingerprint.** Exclude our own introspection statements, DDL, and
   `information_schema`/dbt metadata reads so the tool does not advise on its own noise.
   **All DML is kept as workload** — `INSERT`, `UPDATE` and `DELETE`, not only `SELECT`:
   a write's `WHERE` clause benefits from an index exactly as a read's does, and write
   volume is what makes an index expensive to maintain, so excluding DML would both hide
   index candidates and bias the cost picture toward reads. `qualify()` leaves DML
   columns bare (Postgres resolves them to a single-table statement's sole target table),
   and comparison roles are gated on being inside a `WHERE`/`JOIN`/`HAVING` so a DML
   `SET` assignment is never mistaken for a predicate.
4. **Fetch schema and `qualify()`.** `sqlglot.optimizer.qualify.qualify` resolves
   unqualified columns to their tables. This step is `aggregate()`, which produces an
   `Aggregation` from a `Workload`: queries that fail to qualify increment
   `Aggregation.skipped_unqualifiable` and appear in the report — never silently dropped,
   matching `check`'s existing `skipped` discipline. (`skipped_unqualifiable` lives on
   `Aggregation`, not `Workload`, because qualification happens during aggregation, not
   ingest.)
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
- ADV005's leading-wildcard `LIKE` check (`'%x'`) is a genuine casualty of ingest-time
  redaction: `like '%x'` becomes `like $1` once redacted, and the pattern is gone by the
  time proposal rules run. As shipped, this is handled with a pre-redaction boolean flag
  (`ColumnRole`'s `NON_SARGABLE` plus a `QueryStat.flags` entry, `FLAG_LEADING_WILDCARD_LIKE`)
  captured in `fingerprint.py` before the literal is discarded. Because the flag is
  query-group-level rather than column-level, this branch of ADV005 reports at the query
  level with no column attribution ("Leading-wildcard LIKE in a hot query group") — unlike
  the function/cast branch of ADV005, which keeps full table/column attribution.

`--keep-literals` opts back in. The report header states which mode produced it.

## CLI surface

```
sqlquality advise [--engine postgres|redshift|snowflake]
                  [--dsn URL | --profile NAME --target NAME --profiles-dir DIR]
                  [--schema public ...] [--since 7d] [--limit 500] [--min-cost-share 0.01]
                  [--manifest target/manifest.json]
                  [--keep-literals] [--timeout 30] [--dry-run]
                  [--json] [--markdown FILE] [--ddl FILE]
```

`--schema` is repeatable (default `public`). Without it, bare table names are ambiguous
across schemas and the sqlglot schema dict handed to `qualify()` cannot be built
reliably — this option was added while implementing, not originally specified.

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
| ADV007 | Add index on a hot join key with no existing index leading with it | cost share, NDV, row estimate, absence of a covering index |
| ADV008 | Composite index for a hot `GROUP BY`, column order inferred from cost, capped at MEDIUM | cost share, row estimate, absence of a covering index |

Proposals are suppressed entirely for tables below a row-count floor, where a sequential
scan is the correct plan and an index would be pure overhead.

**Window caveat.** `pg_stat_statements` is cumulative since the last reset and carries no
per-statement timestamps before PostgreSQL 17 added `stats_since`. On earlier versions
`--since` cannot be honored: the report states the window as "since stats reset at
`<timestamp>`" rather than implying the requested window was applied.

## Deviations from the spec (Batch 2)

Found and agreed while implementing ADV007/ADV008, multi-schema support and wrapped-read
handling, after the initial ship this document otherwise describes. The dataclass shapes
in "Core dataclasses" and "Interface" above already reflect what shipped as a result —
`ColumnUsage.relation: Relation`, `Aggregation.tables: frozenset[Relation]`,
`TableFacts.relation`, `fetch_schema`/`fetch_table_facts`/`fetch_indexes` keyed and
parameterized by `Relation` — not the bare-string shapes (`ColumnUsage.table: str`,
`Aggregation.tables: frozenset[str]`, `TableFacts.name`) that shipped first. This section
records why they changed.

1. **Every catalog fact is keyed by `Relation(schema, table)`, not a bare table name.** A
   bare-name key aliased two same-named tables in different schemas: whichever catalog row
   was read last won the row estimate, while `qualify()` resolved columns against the
   union of both tables' columns. `ColumnUsage.table` becomes `ColumnUsage.relation`,
   `Aggregation.tables` becomes `frozenset[Relation]`, `TableFacts.name` becomes
   `TableFacts.relation`, and `fetch_schema`/`fetch_table_facts`/`fetch_indexes` are all
   keyed and parameterized by `Relation`. `fetch_schema` also changes shape, from a flat
   `{table: {column: type}}` to a nested `{schema: {table: {column: type}}}` — the nesting
   is what lets `qualify()` tell two same-named tables apart at all; a flat map resolves a
   column against the union of both column sets.
2. **Schema resolution reads the introspected schema map, not `Table.db`.** The obvious
   implementation attributes a query's table to `table.db`, the schema sqlglot's `qualify()`
   already resolved. That is wrong for the common case: `qualify()` leaves `db` **empty**
   for a bare table reference — `SELECT * FROM orders`, not `SELECT * FROM public.orders`
   — because production SQL relies on `search_path`, not full qualification, and
   `qualify()` has no schema to fill `db` in with. A `Table.db`-only implementation would
   therefore key every search_path-reliant workload under `Relation(schema="", ...)`,
   silently suppressing every proposal for it — exactly the shape-of-real-data failure
   this batch's live suite exists to catch. The shipped resolver (`resolve_relation` in
   `workload/extract.py`) instead trusts `table.db` only once it is checked against the
   introspected schema map, and falls back to looking the bare table name up in that map:
   exactly one introspected schema holding the name resolves unambiguously; more than one
   is genuinely ambiguous and is counted (`Aggregation.skipped_ambiguous`) rather than
   guessed at; none means the table lives outside the introspected schemas and the column
   is dropped.
3. **`advise` accepts repeated `--schema`.** Multiple schemas used to be rejected outright
   because the bare-name key could not tell two schemas' same-named tables apart; the
   `Relation` keying above is what makes accepting more than one safe.
4. **`DECLARE ... CURSOR FOR` and `COPY (...) TO` reads are unwrapped to their inner query**
   before the noise filter runs, rather than filtered as maintenance statements. Both are
   ordinary reads with real predicates — `DECLARE` is what every psycopg2 server-side
   cursor emits — but both begin with a keyword the noise filter otherwise drops.
5. **The rules are evaluated independently but not *reported* independently.** The spec above
   describes eight rules that each report their own findings; in the shipped code a
   reconciliation pass runs over the assembled proposal list before the report is written.
   Proposals with identical DDL collapse into one, and a proposed index whose columns are a
   leading prefix of another proposed index for the same table collapses into the wider one —
   without this, ADV001 and ADV007 shipped a `CREATE INDEX` pair on the same table that
   ADV003 would flag as redundant on the following run, i.e. the tool contradicting itself
   across runs. The absorbed proposal's rationale and confidence are folded into the
   survivor's, attributed by code; its `evidence` is discarded. Same column *set* in a
   different order is not a prefix relationship and both are kept, each disclosing the other.
   Consequence a consumer must know: a rule can fire and contribute no entry to `proposals`.
6. **A composite index proposal requires *joint* support.** ADV001, ADV004 and ADV008 only
   combine columns that some single query group uses together, tracked as a running
   intersection of contributing fingerprints, and report that joint count as
   `co_occurring_fingerprints` instead of a per-column `fingerprints`. Cost weighting alone
   is not enough: a proposal's `cost_share` is the *max* over its columns, so a column
   carrying ~0% of workload cost could not be filtered out by it, and a near-free query
   contributing one column to the middle of a composite produced an index no query could use.
   ADV002 and ADV003, the two `DROP INDEX` rules, are likewise both scoped to the relations
   the workload was observed using rather than to every relation the catalog query returned.

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

### dbt enrichment (`--project-dir` / `--manifest`)

Shipped in Batch 3a (2026-07-28), with a code reassignment from what this section
originally specified — see "Deviations from the spec (Batch 3a: dbt enrichment)" below for
why.

| Code | Proposal | Note |
|---|---|---|
| ADV301 | Hot table maps to a model materialized as `view` → propose `table` or `incremental` | cost share attributed to the model; capped at MEDIUM |
| ADV302 | An index-creating proposal for a `table`/`incremental`/`materialized_view` dbt model is rewritten into a config block instead of DDL that a normal (or `--full-refresh`) `dbt run` would destroy; on a `view` the *DDL* is dropped and explained and the proposal survives at LOW. Not a proposal code: the original rule keeps its own code — see deviation 5 | dbt materialization, columns |
| ADV303 | Model never referenced in the analyzed window, and no other model, snapshot or dbt exposure declares it as a consumer → dead-model candidate. Only *immediate* consumers count, so a dead chain unwinds one model per run from its leaf | permanently LOW confidence: BI tools, longer windows, `--limit` truncation and downstream-only models all hide usage |

The originally-specified "recurring join path across models → propose a mart" rule is out
of scope for Batch 3a; nothing in the shipped code claims that code or that behavior.

## Deviations from the spec (Batch 3a: dbt enrichment)

Found and agreed while implementing the dbt enrichment this document's "dbt enrichment"
subsection above originally specified.

1. **Code reassignment: ADV302 is the DDL-correctness rewrite, not "dead-model
   candidate."** The spec as written gave ADV301 the hot-view-materialization proposal,
   ADV302 the dead-model proposal, and ADV303 a join-path/mart proposal. While scoping the
   implementation it became clear the highest-value rule — the one that justified doing
   dbt enrichment *before* Redshift/Snowflake in this batch — is neither of those: it is
   recognizing that a raw `CREATE INDEX` proposal for a dbt-managed `table` (or
   `incremental`, or `materialized_view`) relation is *actively wrong* advice, because
   `dbt run` drops and recreates that relation (or, for `incremental`, `--full-refresh`
   does), silently destroying the index the next time the pipeline runs. Every other
   dbt-enrichment behavior is *additive* (a proposal that would not otherwise exist);
   this one is *corrective* (a proposal the tool already made, made safe). That
   asymmetry — correctness fix vs. new proposal — is why it took the lower, more
   prominent number: **ADV302** is the rewrite/config-block rule, and the dead-model
   rule this section originally called ADV302 shipped as **ADV303** instead. ADV301
   (materialize a hot view) is unchanged from the original spec. The join-path/mart rule
   originally slotted at ADV303 was dropped from this batch's scope entirely (see the
   table above) rather than renumbered again, since a fourth code with no implementation
   behind it would just be a dangling promise.
2. **dbt is layered strictly on top of the engine-agnostic core — no adapter imports
   it.** `sqlquality.workload.dbt` is imported from exactly one place, `cli.py`, which
   calls `enrich_proposals`/`propose_materialization`/`propose_unused_models` once, after
   `adapter.propose()` has already returned and been re-sorted with the adapter's own
   ranking key — `adapter.ranking_key`, a public hook on the `WorkloadAdapter` ABC that each
   adapter may override, resolved off the instance the CLI resolved. (It first shipped as
   `PostgresWorkloadAdapter._ranking_key` reached into directly from `cli.py`, which made
   this claim false for any second engine: it would have got Postgres's ordering on the dbt
   path and its own everywhere else. Corrected before merge; the CLI now imports no adapter
   at all.) `PostgresWorkloadAdapter` (and, when it ships, the Redshift adapter) has
   no knowledge that dbt enrichment exists. This was a design goal restated in the
   architecture section above, not a deviation from it — recorded here because it was
   verified by grep (`sqlquality.workload.dbt` appears nowhere under `workload/postgres.py`
   or any other adapter) at the end of every task in this batch, not merely assumed.
3. **Matching a dbt model to a relation is on the qualified `(schema, table)` pair, with
   deliberately no bare-table-name fallback.** `DbtContext.model_for` looks up
   `self.models.get(relation)` and nothing else. A dbt project's target schema — `dev`,
   `main`, a CI schema, whatever `profiles.yml` names — routinely differs from the schema
   `advise` introspects in production. A name-only match (ignore the manifest's schema,
   match on table name alone) would therefore attribute a production table's proposal to
   whatever development-schema model happens to share its table name, and ADV302 would
   then rewrite that production table's DDL into a config block on the strength of a
   guess about the wrong model's materialization — the exact class of silent
   misattribution this whole batch exists to avoid, not introduce. The cost of this
   strictness: a relation that two *different* models both build (legitimate when a
   project targets more than one database, since dbt's `relation_name` carries a
   database segment this project drops) cannot be disambiguated from the manifest alone,
   so it is dropped from the index entirely rather than resolved by dict-insertion-order
   luck, and counted in `DbtContext.dropped_collisions` — surfaced in both the CLI's `dbt
   enrichment from ...` disclosure line and the JSON payload's `dbt.dropped_collisions`.
4. **The no-manifest path is byte-identical to a build with no dbt support at all, proven
   by measurement, not asserted.** Neither `--project-dir` nor `--manifest` given means
   `load_dbt_context` returns `(None, None)` and none of the enrichment functions run, so
   `advise` without a manifest is, by construction, the same code path as before this
   batch. This was verified, not just argued: `stdout` (`--json`), the markdown report,
   the `--ddl` file and `stderr` were each diffed byte-for-byte against `main` twice —
   once with a stubbed adapter (deterministic, no live DB) and once against a live,
   seeded Postgres — and all four came back an empty diff both times. Getting to a
   literal empty diff required one interface decision: `advise_payload`'s `dbt` key is
   *omitted from the JSON payload entirely* when no manifest loaded, rather than emitted
   as `"dbt": null` — the latter is a schema addition relative to `main` that would have
   made "byte-identical" true only with an asterisk. A consumer that wants the key
   unconditionally still has `payload.get("dbt")`. Guarded by a test that renders all four
   artifacts from a fixed, non-vacuous workload and asserts every dbt-conditional element is
   absent from each — a diff against another commit is not something the suite can do, but
   the absence property is, and the byte-identity claim was otherwise unprotected against
   regression.
5. **ADV302 is a rewrite, not a proposal `code`.** No `Proposal` ever carries
   `code="ADV302"`: `_enrich_one` replaces `ddl` and amends `rationale`, and the original
   rule (ADV001/ADV004/ADV007/ADV008) keeps its code, confidence and cost share, since the
   evidence for the index is unchanged — only the delivery mechanism is. Consequences,
   recorded because they surprise a consumer: a `--json` filter on `code == "ADV302"` matches
   nothing on every run (filter on `evidence.dbt_index_config` instead), and the terminal
   table row is identical to the same proposal from a dbt-free run, so `advise` prints a
   stderr line naming how many proposals were rewritten.
6. **One model gets one `indexes:` block.** dbt reads a single `indexes` key per model
   config, so two standalone blocks pasted into one config are a duplicate YAML mapping key
   and PyYAML — dbt's parser — silently keeps one, discarding the other recommended index.
   Multiple index proposals per relation is the ordinary case (the collapse layer never folds
   non-prefix column lists, and deliberately preserves same-set-different-order pairs), so
   `enrich_proposals` merges every index for one model into the block carried by the
   highest-ranked of those proposals; the others point at it by code.
7. **ADV302's `indexes:` config shape is postgres/redshift-specific, and is disclosed rather
   than suppressed elsewhere.** dbt implements the `indexes` model config only on those two
   adapters. `advise` warns on stderr when the manifest's `adapter_type` is something else,
   and when the manifest is not a v12 schema — the same two checks `check` makes on the same
   file. It still emits the rewrite: the alternative is raw DDL the same rebuild destroys, so
   declining would leave the operator less informed, and `advise` connects only to Postgres
   today, so this configuration is already a mismatch worth naming rather than working
   around.
8. **A caveat that qualifies an executable statement is written into the `--ddl` script.**
   `render_ddl` emits only the code/confidence header, the title and the DDL — never
   `rationale` — so ADV302's decline paths (partial index, unrecognised materialization, no
   plain column list, non-btree access method) produced a file holding a config block that
   explained raw DDL does not survive `dbt run` and, below it, a bare `CREATE INDEX` on that
   same dbt-managed table. `Proposal` grew an optional `note` field that `render_ddl` emits as
   comment lines above the statement; it is deliberately absent from the JSON payload and the
   markdown report, both of which already carry `rationale`, so the pre-dbt payload shape is
   unchanged.
9. **A `DROP INDEX` for a dbt-managed relation is disclosed, not exempted.** The rewrite
   branch originally exempted drops outright, reasoning that "dbt never created this index, so
   dropping it is ordinary." That is false in exactly the case ADV302 exists for: if the index
   *is* declared in the model's `indexes:` config, the next `dbt run` recreates it — the
   operator drops it, dbt puts it back, and ADV002/ADV003 propose the same drop again next run.
   The same silently-reverting advice, pointing the other way, and reachable through the
   ordinary rules rather than only in principle. The proposal is kept (dropping a genuinely
   unused index is still right, and dbt's `indexes` config cannot express a removal) and gains
   the warning in both `rationale` and `note`, so the operator gets both halves of the
   instruction. The property the DDL script now holds is stated statement-wise, not
   relation-wise: no executable line for a dbt-managed relation without an adjacent warning.
   The test that first checked this filtered blocks on the *table* name and so skipped every
   drop, since a `DROP INDEX` names an index.
10. **A manifest inconsistent with the connection is warned about, and "absent" warns too.**
    `advise` connects to Postgres; an `adapter_type` outside `{postgres, redshift}` means dbt
    is not building the relations just introspected at all, so every `(schema, table)` match is
    a name coincidence and ADV301/ADV302/ADV303 are all wrong — a stronger statement than
    "ADV302's `indexes` config key may not exist there," which was the first wording. A
    manifest recording *no* `adapter_type` warns as well, deliberately: `dbt compile` always
    writes one, and warning on "different" while staying silent on "unknown" would make silence
    mean either "consistent" or "unchecked". Warned rather than suppressed, since the fix is in
    the user's invocation and dropping all dbt output would hide it.

## Confidence model

Mechanical, derived from inputs rather than judgment:

- **HIGH** — cost share above threshold, **and** supporting catalog stats present, **and**
  the current physical state confirmed to lack the proposal.
- **MEDIUM** — cost evidence solid, but a catalog input is missing or stale.
- **LOW** — absence-based (ADV303) or thin evidence.

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
