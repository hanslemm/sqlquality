# sqlquality

Measure the **structural complexity** of dbt models' SQL and **gate pull requests
on the complexity delta** between two dbt manifests. Alongside the gate, `sqlquality`
runs per-engine static **performance anti-pattern** checks (with optional
captured-`EXPLAIN` analysis), sqlfluff-backed **linting**, and optional, advisory
**LLM suggestions**.

It is a static tool. It does not connect to your warehouse or execute queries:

- **Complexity** is computed from the SQL AST (via [sqlglot](https://github.com/tobymao/sqlglot)).
- **Performance** is static anti-pattern detection plus ingestion of an `EXPLAIN`
  plan you captured yourself — no query is ever run.
- **Neighbors** (a changed model's direct upstream/downstream models) are *reported*
  for context; they are not scored or gated.

Requires Python 3.11+.

## Contents

- [Install](#install)
- [Commands](#commands)
  - [complexity](#complexity)
  - [lint](#lint)
  - [perf](#perf)
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
- **Performance is static.** Anti-patterns and captured-`EXPLAIN` ingestion only —
  `sqlquality` never runs your queries. Perf adapters exist for **postgres** and
  **redshift** only today.
- **Jinja analysis is approximate.** Raw dbt models are analyzed by stripping Jinja to
  placeholders (with a stderr notice). Prefer compiled SQL from `target/compiled/` for
  accurate results.
- **Neighbors are reported, not scored.** A changed model's direct upstream/downstream
  models are surfaced for context; the gate only evaluates the changed models
  themselves.
