# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `sqlquality advise --engine redshift` — reads Redshift query history
  (`sys_query_history`) and catalog metadata (`svv_columns`, `svv_table_info`,
  `svv_alter_table_recommendations`) over a read-only connection and proposes SORTKEY
  (ADV101), DISTKEY (ADV102) and DISTSTYLE ALL (ADV103) changes, VACUUM/ANALYZE
  maintenance (ADV104), and relays Amazon Redshift Advisor's own recommendations verbatim
  (ADV105), attributed as Advisor's rather than sqlquality's. Redshift has no indexes, so
  none of ADV001–ADV008 apply; ADV101/102/103 each rewrite the entire table (no
  `CONCURRENTLY` escape exists on Redshift), which the generated `--ddl` script's header
  says loudly, and which caps those three rules at MEDIUM confidence — Redshift exposes no
  per-column NDV to judge predicate selectivity or distribution skew. ADV104 is the
  exception: its evidence is a direct catalog measurement and its remediation does not
  rewrite anything, so it is the only Redshift rule that reaches HIGH. A dbt-managed
  Redshift model's table-rewrite proposal is disclosed (not silently applied) via the same
  dbt-enrichment path Postgres's index proposals use. **Redshift's introspection SQL has
  not been executed against a live cluster** — see the README's `advise` section for what
  is and is not verified, and run `--dry-run` before pointing this at production.
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
- Optional dbt enrichment for `advise`: `--project-dir` (reads
  `<project-dir>/target/manifest.json`) or `--manifest <path>` layers dbt model metadata
  onto the same analysis, and every manifest-free `advise` invocation is proven
  byte-identical (stdout, markdown, DDL and stderr) to a run with no dbt support at all.
  **ADV302** rewrites an index-creating proposal for a `table`-, `incremental`- or
  `materialized_view`-materialized dbt model into a commented `indexes:` config block
  instead of raw DDL, because a normal `dbt run` (or `--full-refresh`) drops and rebuilds
  those relations and the DDL would not survive it; on a `view` the *DDL* is dropped and
  explained while the proposal stays at LOW confidence, since a view has no storage to index
  but "this index cannot apply here" is still the finding. ADV302 is a rewrite, not a
  proposal code: the original rule keeps its code, confidence and cost share, so no proposal
  ever carries `code: "ADV302"` — filter on `evidence.dbt_index_config` instead — and
  `advise` prints a stderr line saying how many proposals it rewrote, since the terminal
  table row is otherwise unchanged. Several index proposals for one model merge into a single
  `indexes:` block, because dbt reads one `indexes` key per config and two blocks pasted into
  one config are a duplicate YAML key whose loser is silently discarded. Wherever ADV302
  declines and leaves executable DDL in place (a partial index, an unrecognised
  materialization, no plain column list, a non-btree access method), the warning is written
  into the `--ddl` script above the statement, not only into the rationale. A `DROP INDEX`
  proposal (ADV002, ADV003) on a dbt-managed relation is the same hazard pointing the other
  way — if the index is declared in the model's `indexes:` config, `dbt run` recreates it and
  the drop silently reverts — so those keep their DDL and gain a warning, in the rationale and
  in the DDL script, that the config entry has to be removed too. A manifest whose
  `adapter_type` is neither postgres nor redshift (or is absent), or whose schema is not v12,
  is disclosed on stderr: a foreign adapter means dbt is not building the relations `advise`
  introspected at all, so every match is a name coincidence. **ADV301** proposes
  materializing a `view`-backed model that carries a hot share of workload cost, capped at
  MEDIUM. **ADV303** flags a dbt model within reach of the manifest that the analyzed
  workload never touched and that no other model, snapshot or dbt exposure declares as a
  consumer, capped at LOW. Matching a model to a relation is on the exact `(schema, table)`
  pair with no bare-name fallback, since a dbt project's target schema routinely differs
  from the schema being introspected; a relation two different models both claim is
  dropped from matching (not guessed at) and counted in the CLI disclosure and the JSON
  payload's `dbt.dropped_collisions`.

### Fixed

- `--ddl`'s guarantee that every line of the generated script is either an intended statement
  or a `--` comment now holds for all ten codepoints `str.splitlines()` treats as a line
  boundary, on both the Postgres and Redshift renderers. The guard tested only `\n` and `\r`,
  while everything that splits the text uses `splitlines()`, so an introspected identifier
  containing `\v`, `\f`, `\x1c`, `\x1d`, `\x1e`, `\x85`, `U+2028` or `U+2029` — all legal
  inside a quoted Postgres identifier — produced a second physical line the guard never
  examined, and the tail of the statement was emitted looking like a bare statement of its own.
- ADV004 (partial index) now consults the existing-index list, which it was alone among the
  index-creating rules in never doing. A plain index leading with the guarded column now
  suppresses the proposal — that index already serves the lookup, and the partial index's only
  advantage is a size saving this tool cannot measure against a second index's write cost (and
  which ADV003 would never flag as redundant, since its prefix check is restricted to plain
  indexes). Where a check genuinely could not run it is now stated rather than skipped: an
  unreadable existing-index list caps confidence at LOW and says so, and an existing *partial*
  or expression index that leads with the same column is named, since sqlquality does not
  compare index predicates and so cannot tell whether the proposal is already applied. New
  evidence keys `partial_indexes_not_compared` and `expression_indexes`; deliberately not
  ADV001's `partial_indexes_skipped`, which records a different fact.
- dbt enrichment now discloses itself in the terminal on **every** engine. The stderr
  disclosure line counted only ADV302's config-block rewrite, which no Redshift proposal can
  reach (nothing Redshift emits is a `CREATE INDEX`), so a `--project-dir` run on Redshift
  warned in `rationale` and in the `--ddl` note that `dbt run` may undo an hours-long
  full-table rewrite while the terminal row stayed byte-identical to a dbt-free run. Any
  proposal whose DDL cannot be expressed as dbt config is now counted and reported too, as is
  an index *drop* (ADV002, ADV003) on a dbt-managed relation — the case reachable on Postgres,
  where a run proposing only drops enriched every one of them and still said nothing, so an
  operator applied a drop that the next `dbt run` recreated from the model's `indexes:` config.
  Each of the three outcomes is reported as its own clause, because each calls for a different
  action: paste a config block, expect a runnable statement not to survive the next rebuild, or
  delete a config entry as well as running the drop.
- `IS NOT NULL` predicates were classified as `IS NULL` when sqlglot 30.13 or newer was
  installed, because that release moved the negation from a wrapping `Not` node onto a
  `negate` flag on the `Is` node itself. Both encodings are now read. This was not cosmetic:
  ADV004 turns these roles directly into a partial index's `WHERE` clause, so it proposed
  `WHERE col IS NULL` for a workload filtering `WHERE col IS NOT NULL` — an index over exactly
  the complement of the intended rows. `uv.lock` pins 30.12, so development and CI never saw
  it while any fresh `pip install sqlquality` resolved a newer 30.x and did. CI now also runs
  the suite against the highest versions the declared dependency ranges allow.
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
