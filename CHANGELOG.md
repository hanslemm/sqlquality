# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-08-06

### Added

- **`sqlquality verify BEFORE AFTER`** — a new, **fully offline** command that diffs two
  `advise --json` artifacts and reports, per proposal, whether the advice was **applied**
  (observed from the two runs' recorded `physical_state`, never declared or asked) and
  whether the query groups that justified it actually got **faster per call**. It reads two
  files, connects to nothing and needs no credentials, so anyone reviewing a change can run
  it. `--json` and `--markdown PATH` mirror `advise`'s options; it exits 0 whenever a
  comparison was reported and 2 when it was refused, never 1 — `verify` reports and never
  gates, and there is deliberately no `--gate` flag.

  Each proposal gets an outcome (`improved`, `unchanged`, `regressed`, `disappeared`,
  `not_applied`, `unobservable`) and a confidence taken from how comparable the two
  **windows** are. The valuable one is **applied but unchanged**: the work was done and it
  did not help.

  **Mean time per call is the metric; `cost_share` rides along only as context.** On
  Postgres `pg_stat_statements` is cumulative, so a group's share of the window falls simply
  because a week of other traffic accumulated — `cost_share` measures whether a finding
  still *matters*, never whether it got *better*. The workload-context line prints both
  runs' total window cost and query-group count so a global workload shift is visible
  rather than deduced.

  **On the common Postgres path (counters baselined once and never reset) the two windows
  are *nested*: the later one contains the earlier one, so a real improvement is
  understated** — sometimes badly enough that a proposal which genuinely helped reads as
  `unchanged`. `verify` says so in the caveat it prints on every such run, caps those
  verdicts at medium confidence, and names the remedy (call `pg_stat_statements_reset()`
  yourself right after applying a change and take the after artifact from there —
  sqlquality never writes to your database, this reset included).

  **`verify` refuses rather than guesses** (exit 2, with the reason): an unreadable,
  non-UTF-8, malformed or non-object JSON file; an artifact missing the keys this release
  added, i.e. one produced by 0.3.0 or an intermediate build — regenerate it rather than let
  a verdict rest on absent data; the same artifact twice (identical path, or byte-identical
  content, which two genuinely distinct runs cannot produce since the counters accumulate);
  two artifacts from **different engines**; two runs that **disagree about literal
  redaction** (`--keep-literals` changes the canonical query text every query-group digest
  is computed from, so the same query group is recorded under different identifiers in the
  two artifacts — an absence there is not a measurement); and a pair whose artifacts are
  demonstrably **swapped**, which is detectable only when both runs report a
  `stats_reset_at` and the after run's is earlier, since a server's statistics-reset instant
  cannot move backwards.

  Everything else that limits what the comparison can claim is **disclosed rather than
  folded silently into a verdict**: the window relation and what it does to confidence; that
  run order is otherwise taken from the argument order (an `advise` artifact carries no run
  timestamp, deliberately, so its bytes stay reproducible); a `--limit` mismatch, which
  makes a group missing from one artifact possibly a sampling artifact rather than a real
  disappearance; each run's `degraded` capabilities, since a read that could not run
  produces the same emptiness a real change would; recommendations whose key matched more
  than one proposal within their *own* artifact, which are reported as unmatched rather than
  as disappeared or new; and proposals only the after run makes, which carry no verdict
  because there is nothing to compare them against.

  **Whether an after-only proposal is a *new* finding is itself a claim `verify` will not make
  unless both runs' coverage supports it.** Three conditions withhold it, and in each the
  proposals are still listed with the reason stated rather than being called new — exactly the
  treatment a query group's absence from the *after* run already gets before `disappeared` may
  be graded:

  - the **before** run's reads were degraded, so it may never have had the evidence to make
    that recommendation — the absence is a fact about that run, not about your database;
  - the **before** run's window sampled fewer (or an unknown number of) query groups, for the
    same reason;
  - the **after** run's reads were degraded in a way that can *relax* a rule rather than
    silence it. A rule that cannot evaluate a threshold proposes anyway at reduced confidence —
    correct for `advise`, whose job is to disclose rather than withhold — so a run that could
    not read table sizes, or could not see an existing index, can make a recommendation a
    fully-observed run would not have made at all.

- `advise --json` now carries `physical_state`, keyed by `"schema.table"` for each
  relation some proposal targets *or* the run's workload analysis touched
  (`aggregation.tables`): Postgres records `is_ordinary_table` and each existing index's
  `name`/`columns`/`is_partial`/`is_unique`; Redshift records `is_ordinary_table`,
  `sortkey1`, `diststyle`, `unsorted` and `stats_off`. This is the physical evidence
  `sqlquality verify` diffs between two artifacts to tell whether a
  proposal was actually applied — observed from what the run's own catalog reads already
  returned, never a second round trip. Scoped to the union, not to proposal relations alone: a
  relation whose proposal got resolved between two runs (the recommended index now
  exists, so the rule stops firing) would otherwise have no entry at all in the very run
  `verify` needs it in. The key itself is always present, even when empty: an *absent*
  `physical_state` key (an artifact from before this feature) and an *empty* `{}` one
  (this run found no relation to report on at all) are different facts, and `verify`
  needs to tell them apart.

  **Each field within an entry is a genuine three-way signal, not just present-vs-absent:**
  `null` means *this run could not tell you* — either the relation's catalog facts were
  never fetched at all (e.g. a dbt-enriched proposal for a relation outside the analyzed
  workload), or the relevant catalog read was denied (see `degraded`) — while `false` /
  `[]` is a real measurement (Postgres's `is_ordinary_table: false` means a view, a
  foreign table, or a partitioned parent; `indexes: []` means the relation was fetched and
  genuinely has none). Treating `null` as `false`/`[]`, or the reverse, fabricates a
  transition across two runs where nothing physically changed — only what each run
  happened to fetch did.

  **`is_ordinary_table` is not at parity between the two engines, and deliberately so.** On
  Postgres it comes from a catalog read filtered `relkind = 'r'`, so `false` unambiguously
  means "not an ordinary table". On Redshift it comes from the relation's presence in
  `svv_table_info`, which omits external (Spectrum) tables — those cannot carry a SORTKEY,
  DISTKEY or DISTSTYLE at all — *and* genuinely **empty** local tables, which can. Nothing
  available to the adapter tells those two apart, so a Redshift `false` conflates "not a
  table" with "a table nobody has written to yet", and any `verify` verdict resting on it is
  correspondingly weaker than the identical field name suggests. `null` still means "this run
  could not tell you" on both engines; it is only `false` whose meaning differs.

- `advise --json` now carries `query_groups`, a top-level `list[dict]` of **every** workload
  query group this run analysed, each with `digest`
  (the same 12-character id as `evidence["fingerprint"]`/`evidence["fingerprint_digests"]`),
  `calls`, `total_time_ms`, and `mean_ms` (`total_time_ms / calls`, or `null` — never
  `0.0` — when `calls` is `0`, since a group with no recorded calls has no meaningful mean
  to report). This is the workload evidence `sqlquality verify`
  compares against a later run's timings for the same query groups. The key is always
  present, `[]` when the run analysed no query groups at all, for the same reason
  `physical_state` is: an absent key marks a pre-this-feature artifact, distinct from a
  genuinely-empty run.

  **Not scoped to what some proposal currently cites**, for the same reason
  `physical_state` is not scoped to proposal relations alone: a group whose proposal got
  resolved between two runs (the recommended index now exists, so the rule stops firing and
  stops citing the group) would otherwise vanish from the later artifact even though the
  query is still running, and `verify` would read that as the query having disappeared
  rather than as the improvement it is. The list is bounded by the same `--limit` that
  bounds the workload itself, and `analyzed.query_groups_in_window` reports its size, so it
  scales with the workload rather than with the schema.

  **This is a different key from, and coexists with, the existing
  `payload["analyzed"]["query_groups"]`** — that one is (and remains) an integer count of
  how many query groups this run understood; the new one is the list of those groups with
  their timings. JSON nesting disambiguates the two (one is nested under
  `"analyzed"`, the other sits at the payload's root) — renaming the older, already-shipped
  key to avoid the collision would itself be a breaking change to existing consumers, and
  is out of scope here.

  Every index- and physical-design-rule proposal (ADV001, ADV004, ADV005, ADV006, ADV007,
  ADV008, ADV101, ADV102, ADV103) now also carries `evidence["fingerprint_digests"]`: a
  sorted tuple of the same 12-character digests, naming which of `query_groups` back that
  specific proposal — beside the existing `fingerprints` / `co_occurring_fingerprints`
  counts, not replacing them. Every other rule omits the key entirely rather than emit `[]`,
  since an empty list would misread as "zero query groups support this" rather than the true
  "this rule does not name individual query groups": Postgres's ADV002/ADV003 (catalog
  measurements — index scan counts, index-prefix comparisons), Redshift's ADV104/ADV105 (a
  direct catalog measurement and a verbatim Amazon Redshift Advisor relay, respectively), and
  the dbt rules ADV301/ADV303 — the first proposes on a relation's *aggregate* share of
  workload cost rather than on any one group's timings, the second on a model the workload
  never touched at all. `sqlquality verify` consequently never grades those rules
  `improved`, `unchanged` or `regressed` — there is no per-group timing to compare — only
  `not_applied` when the change demonstrably was not made, and `unobservable` otherwise, with
  the applied signal reported beside it.

  `evidence["fingerprint_digests"]` is a `--json`-only key: `--markdown`'s evidence line
  omits it, since a list of opaque 12-character hashes (one per backing query group) is a
  machine correlation key for `verify` with no meaning to a human reader — the
  human-relevant number is already right beside it as `fingerprints` /
  `co_occurring_fingerprints`. `--markdown` output is otherwise unchanged by this release.

### Changed

- **Breaking:** `advise --json`'s `"window"` key is now an object —
  `{"description": str, "engine": str, "stats_reset_at": str | None, "since": str | None,
  "since_duration_seconds": float | None, "limit": int | None}` — rather than the bare
  prose string it was in 0.3.0. A sentence cannot be compared across two runs, and
  `sqlquality verify` needs to tell whether a baseline and a follow-up
  windows are disjoint, nested (Postgres's cumulative `pg_stat_statements` counters were
  never reset between the two, so the baseline is really a subset of the follow-up) or
  otherwise comparable before it can grade its own confidence. The old `description`
  string is unchanged and still present, under that key. **Artifacts produced by 0.3.0
  are not accepted by `verify`** — this key's shape is exactly how it tells an old
  artifact apart from one that genuinely has nothing to report for a field, so
  regenerate any saved baseline with the current version before verifying it.

  `"since_duration_seconds"` is the *requested* `--since` duration (e.g. `7d` is
  `604800.0`), separate from `"since"`'s *absolute* cutoff: `verify` grades two windows
  comparable on equal *durations*, not equal cutoffs, because two runs a week apart with
  the identical `--since 7d` bind two different absolute cutoffs by construction but
  request the same duration — grading on the cutoff alone would have made Redshift's
  `COMPARABLE` case unreachable from any real pair of runs. Postgres always reports it as
  `null`, the same as `"since"`, since it cannot apply `--since` at all.

## [0.3.0] - 2026-08-03

`advise` is new in this release. Everything below describes it as it ships; it has no
previously-released behaviour, so there is no "Fixed" section — bugs found and fixed while
building it are in the git history, not here.

### Added

- `sqlquality advise` — reads Postgres query history (`pg_stat_statements`) and catalog
  metadata over a read-only connection and proposes indexes, index removals, partial
  indexes, sargability fixes and `SELECT *` cleanups (ADV001–ADV008), with a `--json`
  and `--markdown` report and a reviewable `--ddl` script.

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

- `advise` supports multiple `--schema` flags: every catalog fact (table sizes, NDV,
  index lists, generated DDL) is keyed by `schema.table`, so same-named tables in
  different introspected schemas no longer alias into one another.

- ADV007 proposes an index on a hot, unindexed join key; ADV008 proposes a composite
  index for a hot `GROUP BY`.

- Overlapping proposals are reconciled before the report is written, so no two rules can
  advise contradictory work on one relation. On Postgres: two rules reaching identical DDL
  collapse into one entry, and a proposed index whose columns are a leading prefix of another
  proposed index for the same table collapses into the wider one (creating both would have
  produced a pair ADV003 flags as redundant on the next run). On Redshift, where the conflict
  is between whole strategies rather than prefixes, an ADV103 `DISTSTYLE ALL` withholds the
  ADV102 `DISTKEY` proposal for the same relation, since `ALL` subsumes it — otherwise an
  operator would run one hours-long table rewrite and then a second undoing it. The absorbed
  proposal's rationale and confidence are folded into the survivor's, attributed by rule code;
  its `evidence` block is **not** merged and is discarded. Two proposals covering the same
  columns in a different order are both kept, each naming the other. Consequence for `--json`
  consumers: a rule can fire and contribute no entry of its own to `proposals`, so counting
  entries by `code` is not a count of which rules matched.

- A composite index proposal requires its columns to co-occur in at least one query group, and
  reports that joint count as `co_occurring_fingerprints` rather than a per-column
  `fingerprints` — so the number beside a multi-column proposal is the support for *that
  combination*, not the highest support any one column has. ADV001, ADV004 and ADV008 all
  report it; the single-candidate rules (ADV005, ADV007) keep a per-column count, which for
  them is the whole truth.

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

- `advise` resolves connections from `--dsn`, the `SQLQUALITY_DSN` environment variable, or a
  dbt `profiles.yml`, in that order. dbt is optional throughout: every `advise` invocation
  behaves identically without a manifest, which is verified by comparing stdout, JSON, markdown
  and the DDL file byte-for-byte.
- `advise --dry-run` prints every introspection statement the chosen engine can issue, and
  exits without connecting — so the exact reads can be audited, or handed to a DBA, before any
  credential is used.
- `advise` unwraps `DECLARE ... CURSOR FOR` and `COPY (...) TO` reads to their inner query
  before filtering, so server-side-cursor and `COPY`-based workloads — what psycopg2, Django
  and SQLAlchemy emit for large result sets — reach the analysis instead of being discarded as
  maintenance traffic. Note that on Postgres a cursor's cost is attributed to its `FETCH`
  statements, so such a read contributes its predicate columns but little of its cost.
- Optional extras `sqlquality[postgres]` and `sqlquality[warehouse]`, both currently psycopg.
  Without one, `advise` exits with an install hint rather than a traceback.
- Every line of the `--ddl` script is either a statement `advise` intends or a `--` comment,
  including when an introspected identifier contains a character `str.splitlines()` treats as a
  line boundary. sqlquality never executes the script.

### Changed

- The "static tool, never connects" claim is now scoped: sqlquality never executes your SQL,
  and only `advise` opens a connection — read-only, metadata-only, with a statement timeout.
- Query literals are redacted at ingest by default; `--keep-literals` opts back in.

### Known limitations

- **Redshift's introspection SQL has never been executed against a live cluster.** Its
  connection path is verified against a real server (Redshift speaks the PostgreSQL wire
  protocol) and every statement is syntax-checked and proven bindable, but the column names and
  proposal semantics come from AWS documentation. Run `advise --engine redshift --dry-run` and
  check the statements against your cluster before relying on the output — and please report
  what you find.
- Snowflake is designed but not implemented; `advise` supports `postgres` and `redshift`.
- `advise`'s remaining engine-specific caveats — cursor cost attribution, `COPY` and PL/pgSQL
  double-counting under `pg_stat_statements.track = all`, Redshift's `--limit` semantics, and
  what `svv_table_info` absence cannot distinguish — are documented in the README's Limitations
  section rather than repeated here.

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
