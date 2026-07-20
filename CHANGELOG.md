# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - Unreleased

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
