# Graph Report - sqlquality  (2026-07-18)

## Corpus Check
- 37 files · ~6,601 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 268 nodes · 496 edges · 14 communities (11 shown, 3 thin omitted)
- Extraction: 75% EXTRACTED · 25% INFERRED · 0% AMBIGUOUS · INFERRED: 123 edges (avg confidence: 0.75)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `d9a71b8a`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- DbtProject
- cli.py
- analyze_sql
- Finding
- load_config
- models.py
- changeset.py
- antipattern_findings
- lint_sql
- check
- test_check_gate.py
- sqlquality
- __init__.py
- sqlquality

## God Nodes (most connected - your core abstractions)
1. `DbtProject` - 26 edges
2. `ComplexityEngine` - 13 edges
3. `load_config()` - 12 edges
4. `evaluate_gate()` - 12 edges
5. `Finding` - 12 edges
6. `analyze_sql()` - 12 edges
7. `antipattern_findings()` - 11 edges
8. `check()` - 11 edges
9. `Config` - 11 edges
10. `DbtProjectError` - 11 edges

## Surprising Connections (you probably didn't know these)
- `test_dag_facts()` --calls--> `DagFacts`  [INFERRED]
  tests/test_dbtproject.py → src/sqlquality/models.py
- `test_adapter_static_findings()` --calls--> `get_adapter()`  [INFERRED]
  tests/test_adapters_postgres.py → src/sqlquality/adapters/__init__.py
- `test_get_adapter_unknown_raises()` --calls--> `get_adapter()`  [INFERRED]
  tests/test_adapters_postgres.py → src/sqlquality/adapters/__init__.py
- `test_index_scan_clean()` --calls--> `parse_pg_plan()`  [INFERRED]
  tests/test_adapters_postgres.py → src/sqlquality/adapters/postgres.py
- `test_seq_scan_flagged()` --calls--> `parse_pg_plan()`  [INFERRED]
  tests/test_adapters_postgres.py → src/sqlquality/adapters/postgres.py

## Import Cycles
- None detected.

## Communities (14 total, 3 thin omitted)

### Community 0 - "DbtProject"
Cohesion: 0.08
Nodes (22): DbtProject, DbtProjectError, ModelNode, Path, ValueError, Read a dbt project's manifest.json (schema v12) as a model graph., Raised when the manifest is malformed or a node is missing/uncompiled., A loaded dbt manifest, indexed for model-graph queries. (+14 more)

### Community 1 - "cli.py"
Cohesion: 0.07
Nodes (10): complexity(), main(), perf(), Path, Command-line interface for sqlquality., Analyze a SQL file for performance anti-patterns (+ optional EXPLAIN plan)., Console-script entry point., sqlquality — measure dbt model performance and complexity. (+2 more)

### Community 2 - "analyze_sql"
Cohesion: 0.11
Nodes (24): dist_sort_findings(), _filter_columns(), _join_key_columns(), Expression, Redshift DISTKEY/SORTKEY inference from a model's JOIN/FILTER columns., Suggest a DISTKEY (join keys) and SORTKEY (filter columns) for the model., analyze_sql(), parse() (+16 more)

### Community 3 - "Finding"
Cohesion: 0.11
Nodes (20): ABC, PerfAdapter, PerfAdapter interface — one per SQL dialect/engine., Per-engine performance analyzer (static + EXPLAIN-plan)., Static anti-pattern findings from the SQL text., Findings from a captured EXPLAIN output (raw file text; format per engine)., get_adapter(), Perf adapter registry. (+12 more)

### Community 4 - "load_config"
Cohesion: 0.18
Nodes (23): Config, GateConfig, load_config(), Path, Load sqlquality.yml into typed config (with defaults)., Load config from YAML, or return defaults if absent., ModelDelta, evaluate_gate() (+15 more)

### Community 5 - "models.py"
Cohesion: 0.14
Nodes (21): Enum, ComplexityEngine, Turn structural metrics (+ optional DAG facts) into a 0-100 complexity score., Compute a weighted composite complexity score., ComplexityMetrics, ComplexityScore, DagFacts, Shared data models for sqlquality. (+13 more)

### Community 6 - "changeset.py"
Cohesion: 0.14
Nodes (18): RuntimeError, ChangeSet, ChangeSetError, compute_changeset(), parse_state_modified(), Path, Turn dbt `state:modified` selection into a changed-model ChangeSet., Raised when the `dbt ls` invocation fails. (+10 more)

### Community 7 - "antipattern_findings"
Cohesion: 0.20
Nodes (16): antipattern_findings(), _has_cartesian_join(), _has_leading_wildcard_like(), _has_select_star(), Expression, Dialect-agnostic static SQL anti-pattern detectors (SQLGlot)., Static anti-pattern findings for one SQL statement., _codes() (+8 more)

### Community 8 - "lint_sql"
Cohesion: 0.20
Nodes (13): lint(), Lint a SQL file for best-practice violations (SQLFluff); --fix rewrites it., fix_sql(), lint_sql(), Best-practice linting + auto-fix via SQLFluff's programmatic API., Lint one SQL string; return findings (parse errors included as PRS)., Return SQL with SQLFluff auto-fixes applied (unchanged if unparseable)., _to_finding() (+5 more)

### Community 9 - "check"
Cohesion: 0.22
Nodes (11): check(), Gate a dbt change on the complexity delta of its changed models., gate_payload(), Render a GateReport to a JSON payload and a self-contained HTML document., JSON-serializable summary of a gate report., A self-contained HTML report (no external assets)., render_html(), test_gate_payload_includes_skipped() (+3 more)

### Community 10 - "test_check_gate.py"
Cohesion: 0.47
Nodes (8): _mock_changed(), _project_with_baseline(), A project dir (candidate) + a state dir (baseline with simpler orders)., test_check_corrupt_baseline_exit_2(), test_check_dbt_failure_exit_2(), test_check_fail_mode_exit_1(), test_check_html_written(), test_check_json_reports_delta()

## Knowledge Gaps
- **2 isolated node(s):** `sqlquality`, `Install (dev)`
  These have ≤1 connection - possible missing edges or undocumented components.
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `DbtProject` connect `DbtProject` to `load_config`, `models.py`, `changeset.py`?**
  _High betweenness centrality (0.127) - this node is a cross-community bridge._
- **Why does `Finding` connect `Finding` to `lint_sql`, `analyze_sql`, `models.py`, `antipattern_findings`?**
  _High betweenness centrality (0.096) - this node is a cross-community bridge._
- **Why does `check()` connect `check` to `DbtProject`, `cli.py`, `load_config`, `changeset.py`?**
  _High betweenness centrality (0.095) - this node is a cross-community bridge._
- **Are the 4 inferred relationships involving `DbtProject` (e.g. with `ChangeSet` and `ChangeSetError`) actually correct?**
  _`DbtProject` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Are the 10 inferred relationships involving `ComplexityEngine` (e.g. with `complexity()` and `ComplexityMetrics`) actually correct?**
  _`ComplexityEngine` has 10 INFERRED edges - model-reasoned connections that need verification._
- **Are the 7 inferred relationships involving `load_config()` (e.g. with `check()` and `test_defaults_when_missing_file()`) actually correct?**
  _`load_config()` has 7 INFERRED edges - model-reasoned connections that need verification._
- **Are the 7 inferred relationships involving `evaluate_gate()` (e.g. with `check()` and `test_delta_equal_to_threshold_is_not_regression()`) actually correct?**
  _`evaluate_gate()` has 7 INFERRED edges - model-reasoned connections that need verification._