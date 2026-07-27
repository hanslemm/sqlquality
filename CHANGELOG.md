# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `sqlquality advise` — reads Postgres query history (`pg_stat_statements`) and catalog
  metadata over a read-only connection and proposes indexes, index removals, partial
  indexes, sargability fixes and `SELECT *` cleanups (ADV001–ADV008), with a `--json`
  and `--markdown` report and a reviewable `--ddl` script.
- `advise` supports multiple `--schema` flags: every catalog fact (table sizes, NDV,
  index lists, generated DDL) is keyed by `schema.table`, so same-named tables in
  different introspected schemas no longer alias into one another.
- ADV007 proposes an index on a hot, unindexed join key; ADV008 proposes a composite
  index for a hot `GROUP BY`.
- Overlapping proposals are reconciled before the report is written, so the eight rules
  cannot contradict each other: two rules reaching identical DDL collapse into one entry,
  and a proposed index whose columns are a leading prefix of another proposed index for the
  same table collapses into the wider one (creating both would have produced a pair ADV003
  flags as redundant on the next run). The absorbed proposal's rationale and confidence are
  folded into the survivor's, attributed by rule code; its `evidence` block is **not**
  merged and is discarded. Two proposals covering the same columns in a different order are
  both kept, each naming the other. Consequence for `--json` consumers: a rule can fire and
  contribute no entry of its own to `proposals`, so counting entries by `code` is not a
  count of which rules matched.
- ADV001 now requires the columns of a composite candidate to co-occur in at least one query
  group, and reports that joint count as `co_occurring_fingerprints` in place of the former
  per-column `fingerprints`. Previously a near-free query could contribute a column to the
  middle of an otherwise correct composite, producing an index that no query used and that
  could no longer satisfy the hot query's `ORDER BY`.
- ADV003 is scoped to the tables the workload was observed using, like ADV002 — it no longer
  proposes `DROP INDEX` for a relation the run never analysed.
- `advise` unwraps `DECLARE ... CURSOR FOR` and `COPY (...) TO` reads to their inner
  query before filtering, so server-side-cursor and `COPY`-based workloads (what
  psycopg2, Django and SQLAlchemy emit for large result sets) reach the analysis
  instead of being discarded as maintenance statements.
- Connections resolve from `--dsn`, `SQLQUALITY_DSN`, or a dbt `profiles.yml`, in that
  order. dbt is optional throughout.
- `advise --dry-run` prints every introspection statement without connecting.
- Optional extras `sqlquality[postgres]` and `sqlquality[warehouse]`.

### Changed

- The "static tool, never connects" claim is now scoped: sqlquality never executes your
  SQL, and only `advise` opens a (read-only, metadata-only) connection.
- Query literals are redacted at ingest by default; `--keep-literals` opts back in.

## [0.2.0] - 2026-07-20

### Added

- **Gate**: config validation with a dedicated `ConfigError`; schema-version
  warning when the dbt manifest schema is newer than expected; cycle-safe
  lineage depth computation.
- **Lint**: `--warn-only` mode; `--sqlfluff-config` option to pass a custom
  SQLFluff config; pre-commit hook now excludes `target/`.
- **CLI**: unknown `--dialect` now fails with a friendly, suggestion-bearing
  error (exit 2) on every command; `complexity` and `lint` accept `-` to read
  from stdin; `complexity`/`perf` retry Jinja models with `strip_jinja`
  placeholders (approximate results); `--help` documents the exit-code contract
  (0 = pass, 1 = findings/gate failure, 2 = usage/config/input error).
- **Perf / LLM**: `perf --suggest` LLM enrichment with per-finding isolation
  and prompt truncation.
- **Scoring**: `EXISTS` now counted as a subquery; single-column `DISTKEY`
  advice; `IN`-list `SORTKEY` detection; `strip_jinja` helper;
  `dialects.validate_dialect`.
- **Packaging**: `py.typed` marker (PEP 561); full project metadata (authors,
  keywords, classifiers, URLs); CI test matrix across Python 3.11–3.14;
  automated PyPI + GitHub release workflow.

### Changed

- **Gate**: absolute `--state` path resolution; complexity composite is no
  longer capped; warn-mode now renders `WARN` explicitly; friendly dbt error
  messages with a timeout; markdown output is escaped.
- **Lint**: Jinja/`TMP` findings demoted to `info` severity.
- **Check**: `--dialect` now resolves from the manifest's `adapter_type` when not
  passed explicitly (falling back to `postgres`), so non-postgres projects may see
  different scores/gate outcomes than in `0.1.0`; pass `--dialect` to override.
- **Scoring**: `SQ001` gains `EXISTS`/CTE-closer exemptions; `SQ002` gains
  comma-join scoping and constant-true `ON` detection.

### Fixed

- **Gate**: `dbt ls --no-write-json` no longer clobbers the project manifest;
  a baseline model that is unscoreable is no longer reported as new.
- **Perf / LLM**: `perf --suggest` degrades gracefully when the LLM provider
  fails to construct; `PG002` fixed to match the real `EXPLAIN` JSON shape.

### BREAKING

- **Lint**: `lint` now exits with status `1` when findings are present
  (previously always exited `0`).
- **Scoring**: the complexity composite score is no longer capped, so absolute
  scores and deltas may be larger than in `0.1.0`.

## [0.1.0]

### Added

- Initial development state: dbt model SQL complexity scoring, performance
  anti-pattern detection, linting, and CI gating on score deltas.
