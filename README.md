# sqlquality

Measure the **structural complexity** of dbt models' SQL and **gate pull requests
on the complexity delta** between two dbt manifests. Alongside the gate, `sqlquality`
runs per-engine static **performance anti-pattern** checks (with optional
captured-`EXPLAIN` analysis), sqlfluff-backed **linting**, and optional, advisory
**LLM suggestions**.

sqlquality **never executes your SQL**. `complexity`, `lint`, `perf` and `check` are
fully offline and never open a connection. `advise` is the one exception: it opens a
**read-only** session to read query history and catalog metadata, using only a fixed set
of built-in introspection statements. Run `sqlquality advise --dry-run` to print every
statement it can issue, without connecting.

- **Complexity** is computed from the SQL AST (via [sqlglot](https://github.com/tobymao/sqlglot)).
- **Performance** is static anti-pattern detection plus ingestion of an `EXPLAIN`
  plan you captured yourself — no query is ever run.
- **Neighbors** (a changed model's direct upstream/downstream models) are *reported*
  for context; they are not scored or gated.
- **Advice** is derived from your query history and catalog statistics, and is emitted as
  a report plus a DDL file for you to review and apply. sqlquality never writes to your
  database.

Requires Python 3.11+.

## Contents

- [Install](#install)
- [Commands](#commands)
  - [complexity](#complexity)
  - [lint](#lint)
  - [perf](#perf)
  - [advise](#advise)
  - [check](#check-the-ci-gate)
- [Configuration](#configuration-sqlqualityyml)
- [Exit codes](#exit-codes)
- [CI recipe (a gate that actually gates)](#ci-recipe-a-gate-that-actually-gates)
- [Pre-commit hook](#pre-commit-hook)
- [LLM suggestions](#llm-suggestions-optional-advisory)
- [Limitations](#limitations)

## Install

Once published to PyPI:

```bash
pip install sqlquality
# or
uv add sqlquality
```

Until then, install from git:

```bash
pip install "sqlquality @ git+https://github.com/hanslemm/sqlquality"
# or
uv add "git+https://github.com/hanslemm/sqlquality"
```

The optional LLM suggestions feature needs the `llm` extra (pulls in the Anthropic
SDK):

```bash
pip install "sqlquality[llm]"
```

`advise` needs a database driver. For Postgres:

```bash
pip install "sqlquality[postgres]"
# or, for the driver bundle covering every engine advise targets over time:
pip install "sqlquality[warehouse]"
```

Today `[warehouse]` pulls in the same driver as `[postgres]` (psycopg) — Redshift and
Snowflake support is designed but not yet implemented; see
[Limitations](#limitations). Without the extra, `advise` degrades with an install hint
instead of a traceback.

`--version` prints the installed version:

```console
$ sqlquality --version
0.2.0
```

## Commands

```
sqlquality complexity   Score the structural complexity of a single SQL file.
sqlquality check        Gate a dbt change on the complexity delta of its changed models.
sqlquality lint         Lint SQL files for best-practice violations (SQLFluff); --fix rewrites them.
sqlquality perf         Analyze a SQL file for performance anti-patterns (+ optional EXPLAIN plan).
sqlquality advise       Propose database optimizations from query history and catalog metadata.
```

The `--dialect` / `-d` flag is validated against sqlglot's dialect registry on every
command; an unknown value fails fast with exit 2 and a suggestion. `complexity` and
`lint` also accept `-` to read SQL from stdin.

### complexity

Scores one SQL file and prints a per-metric contribution breakdown plus a composite.

```console
$ sqlquality complexity model.sql
  Complexity — model.sql  (composite 18.4)
┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━━┓
┃ metric            ┃ value ┃ contribution ┃
┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━━┩
│ join_count        │     0 │          0.0 │
│ cte_count         │     2 │          4.0 │
│ subquery_count    │     0 │          0.0 │
│ window_count      │     1 │          4.0 │
│ case_count        │     0 │          0.0 │
│ union_count       │     0 │          0.0 │
│ distinct_count    │     0 │          0.0 │
│ max_select_depth  │     2 │         10.0 │
│ projected_columns │     2 │          0.4 │
└───────────────────┴───────┴──────────────┘
```

Read SQL from stdin with `-`:

```bash
cat model.sql | sqlquality complexity -
```

`--json` emits a machine-readable payload (composite, per-metric contributions, and
the raw metrics):

```console
$ sqlquality complexity model.sql --json
{
  "components": {
    "case_count": 0.0,
    "cte_count": 4.0,
    "distinct_count": 0.0,
    "join_count": 0.0,
    "max_select_depth": 10.0,
    "projected_columns": 0.4,
    "subquery_count": 0.0,
    "union_count": 0.0,
    "window_count": 4.0
  },
  "composite": 18.4,
  "dialect": "postgres",
  "metrics": {
    "case_count": 0,
    "cte_count": 2,
    "distinct_count": 0,
    "join_count": 0,
    "max_select_depth": 2,
    "projected_columns": 2,
    "select_count": 3,
    "subquery_count": 0,
    "union_count": 0,
    "window_count": 1
  },
  "path": "model.sql"
}
```

**dbt / Jinja models:** if the file contains Jinja (`{{ ... }}`, `{% ... %}`),
`sqlquality` first tries to parse it as-is; on failure it retries with Jinja
markers stripped to placeholders and prints a notice to **stderr**:

```
analyzed with Jinja placeholders — results are approximate; prefer compiled SQL from target/compiled/
```

For accurate scores, point `complexity` at compiled SQL from `target/compiled/`
after `dbt compile`. The composite is a real, comparable score in both cases — but
placeholder-stripped results are approximate.

### lint

Lints SQL with sqlfluff and prints findings per file.

```console
$ sqlquality lint messy.sql
                                     Lint —
                            messy.sql (5 findings)
┏━━━━━━┳━━━━━━┳━━━━━━━━━━┳━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ line ┃ code ┃ severity ┃ fix? ┃ message                                      ┃
┡━━━━━━╇━━━━━━╇━━━━━━━━━━╇━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│    1 │ AM04 │ warning  │      │ Query produces an unknown number of result   │
│      │      │          │      │ columns.                                     │
│    1 │ RF02 │ warning  │      │ Unqualified reference '*' found in select …   │
│    2 │ AL01 │ warning  │  ✓   │ Implicit/explicit aliasing of table.         │
│    2 │ AL01 │ warning  │  ✓   │ Implicit/explicit aliasing of table.         │
│    2 │ AL05 │ warning  │  ✓   │ Alias 'o' is never used in SELECT statement.  │
└──────┴──────┴──────────┴──────┴──────────────────────────────────────────────┘
```

**Exit-code semantics:** `lint` exits **1** when any `WARNING`/`ERROR` finding is
present, so it gates CI and pre-commit by default. `--warn-only` prints/emits
findings but always exits 0. Findings from unresolved Jinja are demoted to `info`
severity and **never** gate.

Useful flags:

| Flag | Effect |
|---|---|
| `--fix` | Rewrite the file with auto-fixes. The exit code still reflects *pre-fix* findings (a fully-fixed file still exits 1). Cannot rewrite stdin. |
| `--warn-only` | Always exit 0. |
| `--sqlfluff-config <file>` | Apply a custom sqlfluff config (e.g. `.sqlfluff`). |
| `--exclude-rules <codes>` | Comma-separated rule codes to skip. |
| `--json` | Emit machine-readable JSON. |

`lint` accepts multiple files (and `-` for stdin), which is what the pre-commit hook
relies on.

### perf

Detects static performance anti-patterns for a given engine, and optionally folds in
findings parsed from a captured `EXPLAIN` plan.

```console
$ sqlquality perf messy.sql
                         Perf — messy.sql (postgres, 3 findings)
┏━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ code  ┃ severity ┃ message                                                             ┃
┡━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ SQ001 │ warning  │ SELECT * projects an unknown/wide column set; list columns          │
│       │          │ explicitly.                                                         │
│ SQ002 │ warning  │ Cartesian/cross join without an ON/USING condition.                 │
│ SQ003 │ warning  │ Leading-wildcard LIKE ('%...') is non-sargable and cannot use an    │
│       │          │ index.                                                              │
└───────┴──────────┴─────────────────────────────────────────────────────────────────────┘
```

Supported engines: **`postgres`** and **`redshift`** (Redshift additionally infers
`DISTKEY`/`SORTKEY` advice). Any other valid sqlglot dialect is accepted for
`complexity`/`lint` but has no perf adapter, so `perf` exits 2 for it.

**Exit code:** `perf` exits **1 only when a finding is `ERROR` severity** — which in
practice means the SQL was unparseable (`SQ000`). Anti-pattern findings are `warning`
severity and exit **0**, so `perf` surfaces advice without blocking a build. Bad
input (missing file, unreadable `--explain`) exits 2.

**Captured EXPLAIN.** `--explain <file>` takes a plan you captured yourself:

- **Postgres:** `EXPLAIN (FORMAT JSON) <query>` output (JSON).
- **Redshift:** the plan text from `EXPLAIN <query>`.

```console
$ sqlquality perf messy.sql --explain plan.json
                         Perf — messy.sql (postgres, 4 findings)
┏━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ code  ┃ severity ┃ message                                                             ┃
┡━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ SQ001 │ warning  │ SELECT * projects an unknown/wide column set; list columns …        │
│ SQ002 │ warning  │ Cartesian/cross join without an ON/USING condition.                 │
│ SQ003 │ warning  │ Leading-wildcard LIKE ('%...') is non-sargable …                    │
│ PG001 │ warning  │ Seq Scan on orders — consider an index if the filter is selective.  │
└───────┴──────────┴─────────────────────────────────────────────────────────────────────┘
```

`--json` emits findings and any LLM suggestions. `--suggest` enriches findings with
advisory LLM suggestions — see [LLM suggestions](#llm-suggestions-optional-advisory).

### advise

Reads a database's query history and catalog metadata over a **read-only** connection,
weights column usage by the cost of the queries that use it, and proposes concrete
optimizations — indexes to add, indexes to drop, partial indexes, non-sargable
predicates, and hot `SELECT *`. Output is an advisory report plus a DDL file for you to
review. **`advise` never writes to your database and never executes DDL.**

Only **Postgres** is implemented today; Redshift, Snowflake and dbt enrichment are
designed but not built — see [Limitations](#limitations).

```console
$ sqlquality advise --dsn postgresql://readonly@db.internal/analytics
engine: postgres (credentials from --dsn)
window: since stats reset at 2026-07-19 03:00:00+00
analyzed 3 of 3 query group(s); skipped 0 unparseable, 0 filtered, 0 unresolvable
                Advise — postgres (5 proposals, 3 query groups)
┏━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ code   ┃ conf   ┃ cost share ┃ proposal                                      ┃
┡━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ ADV001 │ high   │      67.0% │ Add index on orders(status)                   │
│ ADV005 │ high   │      22.7% │ Non-sargable predicate on orders.email        │
│ ADV006 │ medium │      22.7% │ Hot SELECT * over wide table(s): orders       │
│ ADV005 │ medium │      10.3% │ Leading-wildcard LIKE in a hot query group    │
│ ADV002 │ medium │          — │ Drop unused index idx_orders_customer_ref on  │
│        │        │            │ orders                                        │
└────────┴────────┴────────────┴───────────────────────────────────────────────┘
```

**Credentials**, resolved in precedence order:

1. `--dsn` — a full database URL.
2. `SQLQUALITY_DSN` — the same, from the environment.
3. `--profile` (with optional `--target` and `--profiles-dir`, default `~/.dbt`) — reads a
   dbt `profiles.yml`. `advise` is not dbt-specific; this is a convenience for projects
   that happen to have one, not a requirement.

The engine is inferred from the DSN scheme (`postgresql://` → `postgres`) or the resolved
dbt adapter type; `--engine` overrides both. The resolved source is always printed to
stderr — `engine: postgres (credentials from --dsn)` — the same discipline `check` uses
for its dialect resolution. Requires the `sqlquality[postgres]` extra (`psycopg`); a
missing driver degrades with an install hint instead of a traceback.

**Flags:**

| Flag | Default | Effect |
|---|---|---|
| `--engine` | inferred | `postgres` is the only engine implemented today. |
| `--dsn` | — | Database URL. Overrides `SQLQUALITY_DSN`. |
| `--profile` | — | dbt profile name, read from `profiles.yml`. |
| `--target` | — | dbt target within the profile. |
| `--profiles-dir` | `~/.dbt` | Directory holding `profiles.yml`. |
| `--schema` | `public` | Schema to introspect. **One at a time** — passing two exits 2, see Limitations. |
| `--since` | — | Window, e.g. `7d`. **Not honored on Postgres** — see Prerequisites below. |
| `--limit` | `500` | Max query-history rows to read. |
| `--min-cost-share` | `0.01` | Suppress proposals below this share of workload cost. |
| `--keep-literals` | off | Do **not** redact literal values from query text. |
| `--timeout` | `30` | Statement timeout in seconds (rejected outside 1–3600). |
| `--dry-run` | off | Print every statement the adapter would issue, then exit 0 **without connecting**. |
| `--json` | off | Emit a machine-readable payload. |
| `--markdown <path>` | — | Write a markdown report. |
| `--ddl <path>` | — | Write proposed DDL for review — sqlquality never executes it. |

**`--dry-run`** is how you verify the read-only claim before `advise` ever touches your
database — it prints the complete, fixed set of statements the adapter can issue and
exits 0 without connecting:

```console
$ sqlquality advise --engine postgres --dry-run
-- workload: requires the pg_stat_statements extension (PostgreSQL 13+) and pg_read_all_stats or superuser; enable via shared_preload_libraries then CREATE EXTENSION. On PostgreSQL 12 and older the view lacks total_exec_time and this will fail.
SELECT s.query, s.calls, s.total_exec_time, s.rows
            FROM pg_stat_statements s
            JOIN pg_database d ON d.oid = s.dbid
            WHERE d.datname = current_database()
            ORDER BY s.total_exec_time DESC
            LIMIT %s

-- stats_reset: reads pg_stat_database; world-readable unless explicitly revoked
SELECT stats_reset
            FROM pg_stat_database
            WHERE datname = current_database()
```

(truncated here; the real output also lists the `schema`, `table_facts`, `ndv` and
`indexes` capabilities). `--json` is honored on `--dry-run` too, so the statement list can
be diffed or fed into review tooling.

**Data protection.** Query history routinely contains personal data inside predicates
(`WHERE email = 'name@example.com'`). Literal values are **redacted at ingest, by
default**: every literal in the parsed query is replaced with a placeholder before
aggregation, before any file is written, before any log line. `--keep-literals` is the
only way to retain them, and the report states which mode produced it. `advise` never
writes to your database — proposed DDL only ever goes to a file (`--ddl`) for you to
review and apply by hand.

**Prerequisites and limits:**

- **`pg_stat_statements`** must be installed (`shared_preload_libraries` +
  `CREATE EXTENSION`), and the connecting role needs `pg_read_all_stats` or superuser to
  see queries run by other users.
- **PostgreSQL 13+.** `pg_stat_statements.total_exec_time` did not exist before version 13
  (it was `total_time`); older servers fail the workload read outright.
- **`--since` cannot be honored on Postgres.** `pg_stat_statements` is cumulative since the
  last statistics reset and carries no per-statement timestamps before PostgreSQL 17
  (which added `stats_since`). Passing `--since` does not narrow the query — the report
  states the real window instead: `since stats reset at <timestamp>`.

**Proposal codes** (Postgres):

| Code | Proposal | Evidence |
|---|---|---|
| ADV001 | Composite index candidate: hot equality columns, then one range/sort column, arity ≤ 3 | cost share, NDV, row estimate, absence of a covering index |
| ADV002 | Drop an index with zero recorded scans (excludes unique/primary-key indexes) | scans since last stats reset, size |
| ADV003 | Drop an index whose column list is a strict prefix of a wider index | both column lists |
| ADV004 | Partial index: a hot equality column guarded by a hot, co-occurring `IS [NOT] NULL` check | cost share, co-occurring fingerprint count |
| ADV005 | Non-sargable predicate — a cast/function on a column, or a leading-wildcard `LIKE` | cost share |
| ADV006 | Hot `SELECT *` on a wide table (≥15 columns) | cost share, column count |

**Confidence model**, mechanical rather than judgment-based:

- **HIGH** — cost share above `--min-cost-share`, **and** supporting catalog stats present
  (e.g. NDV), **and** confirmation that the proposed index does not already exist.
- **MEDIUM** — cost evidence is solid but a catalog input is missing or stale. ADV002 is
  capped at MEDIUM unconditionally: `idx_scan` only accumulates since the last statistics
  reset, so zero scans can never prove an index is unused across a full business cycle.
- **LOW** — thin evidence, e.g. the row count is unknown so the small-table floor could
  not be checked.

Every proposal's evidence renders inline (cost share, calls, fingerprints, row estimate,
NDV, existing index state) so it can be judged from the report alone.

`--ddl` writes a standalone, commented script — never executed by sqlquality:

```sql
-- Generated by `sqlquality advise` — REVIEW BEFORE RUNNING.
-- sqlquality does not execute this script and has not validated it against
-- your workload's write patterns. Each statement is advisory.
--
-- On a live table prefer CREATE INDEX CONCURRENTLY / DROP INDEX CONCURRENTLY:
-- the plain forms below take a lock that blocks writes for the duration.
-- Note that CONCURRENTLY cannot run inside a transaction block, so apply those
-- statements individually rather than piping this whole file into one.

-- ADV001 [high, 67.0% of workload cost]
-- Add index on orders(status)
CREATE INDEX ON "orders" ("status");

-- ADV002 [medium]
-- Drop unused index idx_orders_customer_ref on orders
DROP INDEX "idx_orders_customer_ref";
```

`--json` emits the same evidence as a structured payload (`analyzed`, `degraded`,
`engine`, `proposals`, `redacted`, `skipped`, `window`). This is the first proposal from
the run above — the real payload lists all five under `proposals`:

```console
$ sqlquality advise --dsn postgresql://readonly@db.internal/analytics --json
{
  "analyzed": {
    "query_groups": 3,
    "tables": [
      "orders"
    ],
    "total_cost_ms": 925000.0
  },
  "degraded": [],
  "engine": "postgres",
  "proposals": [
    {
      "code": "ADV001",
      "confidence": "high",
      "ddl": "CREATE INDEX ON \"orders\" (\"status\");",
      "evidence": {
        "calls": 15000,
        "columns": [
          "status"
        ],
        "cost_share": 0.6702702702702703,
        "fingerprints": 1,
        "leading_ndv": 500.0,
        "roles": [
          "equality"
        ],
        "row_estimate": 5200000,
        "table": "orders"
      },
      "rationale": "These columns carry the table's hottest predicates and no existing index leads with them. Equality columns come first so the range column can be scanned last.",
      "title": "Add index on orders(status)"
    }
    /* … 4 more proposal objects, same shape … */
  ],
  "redacted": true,
  "skipped": {
    "noise": 0,
    "unparseable": 0,
    "unqualifiable": 0
  },
  "window": "since stats reset at 2026-07-19 03:00:00+00"
}
```

**Coverage is always disclosed**, not just when it is bad — the terminal, markdown and
JSON paths all print how many query groups were actually understood:

```console
analyzed 2 of 3 query group(s); skipped 0 unparseable, 0 filtered, 1 unresolvable
low coverage: 33% of candidate statements could not be analyzed (0 unparseable, 1
unresolvable against the schema). Cost shares are computed against the whole window, so
they are diluted and --min-cost-share is effectively stricter — few or no proposals may
reflect coverage rather than a healthy workload.
```

This matters because `cost_share` is **not** a partition of the workload: a query
filtering two columns credits its full cost to *both* entries (proposals take the max
over their columns rather than the sum, since summing would double-count), and the
denominator always includes queries that could not be parsed or resolved against the
schema. Poor coverage silently dilutes every proposal's share rather than inflating it —
read the skip counts alongside every proposal.

A missing grant degrades one capability at a time rather than aborting the whole run:

```console
reduced coverage — ndv: permission denied for table pg_stats — reads pg_stats, which
exposes only rows for tables the current role owns or can select from — a role without
table access silently sees no statistics
```

### check (the CI gate)

Scores each changed model on both a candidate and a baseline dbt manifest, and gates
the change on the per-model **complexity delta**.

Requirements:

- **dbt >= 1.5 on `PATH`** (override the executable with `--dbt`). `check` shells out
  to `dbt ls --select state:modified` to discover changed models.
- A **compiled candidate manifest** at `<project-dir>/target/manifest.json` — run
  `dbt compile` first (the gate scores compiled SQL; uncompiled models are skipped).
- A **baseline artifacts directory** (`--state`) containing the prior
  `manifest.json` to diff against.

The dialect is auto-resolved from the manifest's `adapter_type` (falling back to
`postgres`), and printed to stderr; pass `--dialect` to override. `--state` and
`--project-dir` are resolved to absolute paths, so `check` works from a monorepo root.

```console
$ sqlquality check --project-dir . --state prod-artifacts/
dialect: postgres (from manifest adapter_type)
          sqlquality: ❌ FAIL  (changed 1, neighbors 2)
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━┳━━━━┓
┃ model                      ┃ baseline ┃ candidate ┃ delta ┃    ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━╇━━━━┩
│ model.demo.customer_orders │     11.2 │      17.6 │  +6.4 │ ⚠️ │
└────────────────────────────┴──────────┴───────────┴───────┴────┘
```

Whether a regression **fails** the build depends on `gate.mode` (see
[Configuration](#configuration-sqlqualityyml)). In the default `warn` mode the same
change reports the regression but exits 0:

```console
$ sqlquality check --project-dir . --state prod-artifacts/
 sqlquality: ⚠️ WARN (1 regression, gate mode: warn)  (changed 1, neighbors 2)
...
```

`--json` emits the full gate report (verdict, per-model deltas, neighbors, skipped
models):

```console
$ sqlquality check --project-dir . --state prod-artifacts/ --json
{
  "mode": "fail",
  "models": [
    {
      "baseline": 11.2,
      "candidate": 17.6,
      "delta": 6.4,
      "is_new": false,
      "unique_id": "model.demo.customer_orders"
    }
  ],
  "neighbors": [
    "model.demo.orders",
    "model.demo.stg_orders"
  ],
  "passed": false,
  "regressions": [
    "model.demo.customer_orders"
  ],
  "skipped": [],
  "warned": false
}
```

`--markdown <path>` writes a report suitable for a PR comment, and `--html <path>`
writes a self-contained HTML report. The markdown looks like:

```markdown
# sqlquality: ❌ FAIL

| model | baseline | candidate | delta | |
|---|---:|---:|---:|:--:|
| model.demo.customer_orders | 11.2 | 17.6 | +6.4 | ⚠️ |
```

## Configuration (`sqlquality.yml`)

`check` reads `<project-dir>/sqlquality.yml` by default, or the path given to
`--config`. All keys are optional; absent files use the defaults below.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `gate.mode` | `warn` \| `fail` | `warn` | `warn` reports regressions but exits 0; `fail` exits 1 on any regression. **The default `warn` does not fail CI** — set `fail` to actually gate. An invalid value is rejected with exit 2. |
| `gate.max_complexity_increase` | float | `10.0` | A model is a regression when its delta exceeds this threshold. New models (no baseline) are never counted as regressions. |
| `waivers` | list of strings | `[]` | Model `unique_id`s exempt from the gate. |

A complete example:

```yaml
gate:
  mode: fail
  max_complexity_increase: 10.0
waivers:
  - model.my_project.legacy_wide_fact
  - model.my_project.known_gnarly_rollup
```

## Exit codes

Every command follows the same contract:

| Code | Meaning |
|---|---|
| `0` | Pass / no findings. |
| `1` | Findings present, or the gate failed. |
| `2` | Usage, config, or input error (bad flag, unknown dialect, unparseable SQL, unreadable file, malformed `sqlquality.yml`, dbt invocation failure). |

Per-command nuances of code `1`:

- **`complexity`** never gates — it always exits 0 unless the input errors (2).
- **`lint`** exits 1 on any `WARNING`/`ERROR` finding; `info`-level (unresolved-Jinja)
  findings never gate; `--warn-only` forces 0.
- **`perf`** exits 1 only on an `ERROR`-severity finding (unparseable SQL). Anti-pattern
  warnings exit 0.
- **`check`** exits 1 only when `gate.mode: fail` and a regression is present; `warn`
  mode exits 0 even with regressions.
- **`advise`** exits 0 on any successful analysis, whether or not proposals were
  produced — proposals are advisory and never gate. It exits 2 on a usage, config,
  connection or input error (unresolvable credentials, connection failure, missing
  driver, malformed `--since`, out-of-range `--timeout`). It never exits 1.

## CI recipe (a gate that actually gates)

To make CI fail on a complexity regression you must (1) set `gate.mode: fail` in
`sqlquality.yml`, (2) `dbt compile` so the candidate manifest exists, and (3) provide
a baseline (`--state`) produced from your production `dbt compile` artifacts.

```yaml
# sqlquality.yml (committed to the repo)
gate:
  mode: fail
  max_complexity_increase: 10.0
```

```yaml
# .github/workflows/sqlquality.yml
name: sqlquality
on: pull_request

jobs:
  gate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5

      # Fetch the baseline artifacts your production run published.
      # These must come from `dbt compile` (a compiled manifest.json), not a bare parse.
      - name: Download baseline artifacts
        run: ./scripts/download-prod-artifacts.sh prod-artifacts/

      # Produce the candidate manifest for the PR.
      - name: dbt compile
        run: uv run dbt compile

      - name: sqlquality gate
        run: >
          uv run sqlquality check
          --project-dir .
          --state prod-artifacts/
          --markdown report.md

      - name: Comment report on the PR
        uses: actions/github-script@v7
        if: always()   # post the report even when the gate fails
        with:
          script: |
            const body = require('fs').readFileSync('report.md', 'utf8')
            github.rest.issues.createComment({
              ...context.repo,
              issue_number: context.issue.number,
              body,
            })
```

Baseline hygiene: the baseline `manifest.json` must be a **compiled** artifact
(`dbt compile` output). A parse-only manifest lacks `compiled_code`, so those models
are skipped rather than scored.

## Pre-commit hook

`sqlquality` ships a [pre-commit](https://pre-commit.com) hook that lints staged SQL:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/hanslemm/sqlquality
    rev: v0.2.0
    hooks:
      - id: sqlquality-lint
```

The hook runs `sqlquality lint` on staged `.sql` files and excludes `target/`. It
lints **raw model files** (not compiled SQL), so unresolved-Jinja findings are demoted
to `info` and don't block the commit — only real `WARNING`/`ERROR` findings do.

To make the hook non-blocking (report only), pass `--warn-only`:

```yaml
      - id: sqlquality-lint
        args: [--warn-only]
```

## LLM suggestions (optional, advisory)

`perf --suggest` can attach a short, concrete rewrite suggestion to each finding using
an LLM. It is **off by default** and **advisory only** — suggestions never change
findings, severities, exit codes, or the gate.

Setup:

1. Install the extra: `pip install "sqlquality[llm]"`.
2. Set `SQLQUALITY_LLM=anthropic` (also accepts `1` or `true`).
3. Provide `ANTHROPIC_API_KEY` (read by the Anthropic SDK).
4. Optionally set `SQLQUALITY_LLM_MODEL` to override the model (the built-in default is
   `claude-opus-4-8`).

```bash
export SQLQUALITY_LLM=anthropic
export ANTHROPIC_API_KEY=sk-ant-...
sqlquality perf model.sql --suggest
```

If `--suggest` is passed without `SQLQUALITY_LLM` set, `perf` prints a note to stderr
and continues without suggestions. If the extra or credentials are missing, it
degrades gracefully (findings still print, exit code unchanged):

```
LLM suggestions unavailable: The 'anthropic' package is required for AnthropicProvider. Install it with: pip install 'sqlquality[llm]'
```

> **⚠️ Data egress warning.** `perf --suggest` sends the analyzed SQL (up to 20,000
> characters per finding) to the Anthropic API. Do **not** enable it on proprietary or
> sensitive SQL without clearance. API cost scales with the number of findings (one
> call per finding).

## Limitations

- **Complexity is structural.** The composite is an open-ended, weighted score of
  AST features (joins, CTEs, subqueries, windows, select depth, …); it is not capped,
  so a large model can exceed 100. As a rough guide, ~100 is very complex. It measures
  shape, not runtime cost or correctness.
- **`perf` is static.** Anti-patterns and captured-`EXPLAIN` ingestion only —
  `sqlquality` never runs your queries in this command. Perf adapters exist for
  **postgres** and **redshift** only today.
- **Jinja analysis is approximate.** Raw dbt models are analyzed by stripping Jinja to
  placeholders (with a stderr notice). Prefer compiled SQL from `target/compiled/` for
  accurate results.
- **Neighbors are reported, not scored.** A changed model's direct upstream/downstream
  models are surfaced for context; the gate only evaluates the changed models
  themselves.
- **`advise` proposals are ranked by evidence, not proven.** A HIGH-confidence proposal
  is well-supported, not guaranteed correct — it is still advice to review, not a
  decision already made.
- **Index write cost is not modeled.** Proposals weigh read-side benefit (cost share,
  selectivity) against the fact that an index exists; they do not estimate the ongoing
  cost of maintaining it on every write. A hot write path with many proposed indexes
  needs a human judgment call `advise` does not make.
- **Conclusions are only as representative as the log window.** `pg_stat_statements` is
  cumulative since the last statistics reset, with no per-statement timestamps before
  PostgreSQL 17. A reset an hour ago produces a confident-looking report over an hour of
  traffic; the report's `window:` line is the only way to know which you have.
- **`cost_share` is not a partition.** A query filtering two columns credits its full
  cost to *both* — summing across columns double-counts. The denominator also includes
  statements that could not be parsed or resolved against the schema, so poor coverage
  dilutes every share and makes `--min-cost-share` effectively stricter; the CLI warns
  when coverage is poor, and the report always prints the skip counts.
- **Join keys and grouping columns are measured and then ignored.** `advise` classifies
  eight column roles and cost-weights all of them, but only five are read by the proposal
  rules. A hot unindexed foreign-key join produces `orders.customer_id join cost_share
  1.0` and **no proposal at all** — arguably the most valuable index recommendation in a
  relational workload, measured and discarded. Same for `GROUP BY` columns. Compounding
  it: any column under a `JOIN` is classified as a join key, so a predicate you placed in
  an `ON` clause (as `LEFT JOIN` semantics require) is not treated as a filter and drops
  out of ADV001's reach too. Proposing on join keys is a follow-up, not a bug fix.
- **`DECLARE` and `COPY` statements are discarded whole.** The noise filter matches on the
  leading keyword, so `DECLARE cur CURSOR FOR SELECT ... WHERE ...` and
  `COPY (SELECT ... WHERE ...) TO STDOUT` are dropped along with the session-control and
  DDL traffic the filter is for — predicates and all. Django's `QuerySet.iterator()` and
  every psycopg2 server-side cursor emit the first form, so on a Django codebase this can
  be most of your hot reads. They are counted, and the skip line calls them `filtered`
  rather than pretending they were introspection or DDL, but they are not analyzed.
  Unwrapping to the inner `SELECT` is a follow-up.
- **Expression indexes are invisible to the catalog query.** Postgres's `pg_index.indkey`
  holds `0` for an expression column, which matches no `pg_attribute` row, so ADV001 may
  propose a plain-column index whose `lower(col)` expression-index equivalent already
  exists.
- **ADV003 cannot see partial-index predicates.** It compares column lists only, so it
  could recommend dropping a partial index in favor of a wider full index that does not
  actually cover the same rows. Its confidence is capped at MEDIUM for that reason, and
  the caveat is repeated in the proposal's own rationale — a README does not travel
  inside the `.sql` file you run. `advise` cannot tell you which of the two indexes is
  partial or expression-based; you have to check.
- **One schema per run.** Every catalog fact is keyed on the bare relation name — table
  sizes, NDV statistics, index lists and the `qualify()` schema all merge across schemas —
  so `orders` in two schemas would alias into one another and the last catalog row read
  would silently win the row estimate. Rather than report that quietly, `advise` rejects
  more than one `--schema` with exit 2. Run it once per schema. Generated DDL is qualified
  with the schema you passed, so it does not depend on the applying session's
  `search_path`.
- **Redshift, Snowflake and dbt enrichment are designed but not implemented.** `advise`
  supports Postgres only today; passing another `--engine` fails with a clear error
  rather than silently degrading.
