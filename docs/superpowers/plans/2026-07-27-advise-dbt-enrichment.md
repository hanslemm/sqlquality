# Advise dbt Enrichment Implementation Plan (Batch 3a)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** When a dbt manifest is available, `advise` stops proposing DDL that dbt will destroy, and adds three proposals that are only computable by joining workload cost to the model graph.

**Architecture:** dbt stays **optional enrichment layered on top of an engine-agnostic core** — no adapter learns anything about dbt. `PostgresWorkloadAdapter.propose()` returns proposals as it does today; a new engine-neutral `workload/dbt.py` then transforms them and appends ADV301–ADV303. The seam is one call in `cli.py` between `propose()` and the renderers, so the same enrichment will apply to the Redshift adapter in Batch 3b without change.

**Tech Stack:** Python 3.11+, existing `DbtProject` (manifest v12 reader), sqlglot 30.x, typer, rich, pytest.

## Global Constraints

Every task's requirements implicitly include this section.

- **All four CI gates must pass before every commit:** `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy src/sqlquality`, `uv run pytest -q`. There are now also `no-extras` and `highest-deps` CI jobs; don't break either.
- **`uv run pytest` must report `N passed, M deselected`, never `skipped`.** Integration tests are marked `integration`.
- **dbt is optional, never required.** Every existing invocation of `advise` must behave **identically** when no manifest is supplied. This is a hard requirement, not a nicety: the project's positioning is that the dbt-free path is first-class.
- **No adapter may import from `workload/dbt.py`.** Enrichment is applied above the adapter layer. If you find yourself needing adapter knowledge inside the enrichment, stop and report it.
- **sqlquality never executes user SQL.** DDL is written to a file for human review.
- **No credential and no user literal may reach stdout, stderr, a report, or an exception message.**
- **Confidence never overstates evidence.** A check that could not run is disclosed, not assumed.
- **A test that passes with the production change reverted is not a test.** Every test must be observed to FAIL against a deliberate mutation of the line it claims to pin, and the mutation reported. Use `PYTHONDONTWRITEBYTECODE=1` and purge `__pycache__` around each.
- **Where a test asserts over a set, it must discriminate for every member.** Batch 2 produced **seven** findings of the shape "asserts over a set, checks one member". Do not add an eighth.
- Public identifiers get a docstring saying *why*, matching the surrounding module's density.

## Facts established by probing before this plan was written

Do not re-derive these; do not contradict them.

- `ModelNode` already carries `unique_id`, `name`, `resource_type`, `materialized`, `compiled_code`, `relation_name`, `depends_on`, `config`. No new manifest parsing is needed for this plan.
- `relation_name` is a **quoted three-part** string: `'"dev"."main"."stg_orders"'`. The `schema` field on the raw node is `None` in real fixtures, so `relation_name` is the only reliable source of the schema.
- `tests/fixtures/manifest_v12.json` already contains what this plan needs: `model.demo.stg_orders` materialized as `view`, `model.demo.orders` materialized as `table` **with an `indexes` key already in its config**, and non-model resources (`seed`, `test`) that must not be treated as models.
- `original_file_path` and `patch_path` are `None` in that fixture, so a model's source file is **not** always knowable. Every message must degrade gracefully.
- `advise` has **no** manifest option today; `check` derives one from `--project-dir` as `project_dir / "target" / "manifest.json"`.
- `resolve_connection` already recognises the `redshift` and `snowflake` engines, but `get_workload_adapter` only registers `postgres`, so `--engine redshift` fails today with a clear `ValueError`. That is Batch 3b's problem, not this plan's.

## Why ADV302 is the point of this plan

dbt's `table` materialization **drops and recreates** the relation on every `dbt run`. So a `CREATE INDEX` that `advise` currently emits for a dbt-managed table is destroyed by the next build — the tool is confidently telling an operator to do something that silently reverts. dbt's postgres adapter accepts an `indexes` config instead, which it applies after each build. The fixture already has a model using it.

`incremental` differs and the difference matters: dbt does not recreate the relation on a normal run, so an index **survives** until someone runs `--full-refresh`. And a `view` cannot carry an index at all, which makes any index proposal against one meaningless rather than merely fragile.

---

### Task 1: load a manifest for `advise`, and map relations to models

**Files:**
- Create: `src/sqlquality/workload/dbt.py`
- Create: `tests/test_workload_dbt.py`
- Modify: `src/sqlquality/cli.py` (`advise` gains `--project-dir` / `--manifest`)

**Interfaces:**
- Produces:
  - `sqlquality.workload.dbt.parse_relation_name(relation_name: str) -> Relation | None`
  - `sqlquality.workload.dbt.DbtContext` — `@dataclass(frozen=True)` holding
    `models: dict[Relation, ModelNode]`, with:
    - `classmethod from_project(project: DbtProject) -> DbtContext`
    - `model_for(relation: Relation) -> ModelNode | None`
  - `sqlquality.workload.dbt.load_dbt_context(project_dir: Path | None, manifest: Path | None) -> tuple[DbtContext | None, str | None]`
    returning `(context, disclosure)` — `(None, None)` when neither option was given.
- Consumes: `DbtProject`, `ModelNode`, `Relation`.

**The matching rule, and why it declines rather than guesses.** `relation_name` gives
`database.schema.table`; a `Relation` has only `(schema, table)`. Match on the last two parts and
ignore the database, because `advise` connects to one database at a time. Do **not** fall back to
matching on the bare table name when the schemas differ: dbt's `main`/`dev` targets routinely
differ from a production schema, and a bare-name match would attribute a production table to an
unrelated dev model and then rewrite its DDL. An unmatched relation simply gets no enrichment.

- [ ] **Step 1: Write the failing tests**

```python
import json
from pathlib import Path

import pytest

from sqlquality.dbtproject import DbtProject
from sqlquality.models import Relation
from sqlquality.workload.dbt import DbtContext, load_dbt_context, parse_relation_name

FIXTURE = Path(__file__).parent / "fixtures" / "manifest_v12.json"


def _project() -> DbtProject:
    return DbtProject.from_path(FIXTURE)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('"dev"."main"."stg_orders"', Relation("main", "stg_orders")),
        ('"main"."orders"', Relation("main", "orders")),
        ("dev.main.orders", Relation("main", "orders")),
        ('"dev"."main"."Weird.Name"', Relation("main", "Weird.Name")),
    ],
)
def test_parse_relation_name_takes_the_last_two_parts(raw, expected):
    assert parse_relation_name(raw) == expected


@pytest.mark.parametrize("raw", ["", "orders", '"orders"', "   "])
def test_parse_relation_name_declines_what_it_cannot_qualify(raw):
    """A one-part name has no schema, and inventing one would mis-attribute."""
    assert parse_relation_name(raw) is None


def test_context_indexes_models_by_relation():
    context = DbtContext.from_project(_project())
    node = context.model_for(Relation("main", "stg_orders"))
    assert node is not None
    assert node.unique_id == "model.demo.stg_orders"
    assert node.materialized == "view"


def test_context_excludes_non_model_resources():
    """A seed and a test are not models: proposing a materialization change for a dbt
    test, or rewriting DDL because a seed shares a name, would both be nonsense."""
    context = DbtContext.from_project(_project())
    assert context.model_for(Relation("main", "raw_orders")) is None
    for node in context.models.values():
        assert node.resource_type == "model"


def test_context_does_not_match_on_a_bare_table_name():
    """dbt's target schema routinely differs from the introspected one. Matching `orders`
    in schema `public` to a model in schema `main` would attribute a production table to an
    unrelated dev model and then rewrite its DDL."""
    context = DbtContext.from_project(_project())
    assert context.model_for(Relation("public", "orders")) is None
    assert context.model_for(Relation("main", "orders")) is not None


def test_load_returns_nothing_when_no_option_is_given():
    assert load_dbt_context(None, None) == (None, None)


def test_load_reads_an_explicit_manifest_and_discloses_the_source():
    context, disclosure = load_dbt_context(None, FIXTURE)
    assert context is not None
    assert disclosure is not None and str(FIXTURE) in disclosure


def test_load_reports_a_missing_manifest_without_raising(tmp_path):
    """A bad manifest path must degrade to 'no enrichment', not abort a run that already
    did all the catalog work — the same reasoning as the report-write failure path."""
    context, disclosure = load_dbt_context(None, tmp_path / "nope.json")
    assert context is None
    assert disclosure is not None and "nope.json" in disclosure


def test_load_reports_unparseable_json_without_raising(tmp_path):
    bad = tmp_path / "manifest.json"
    bad.write_text("{not json", encoding="utf-8")
    context, disclosure = load_dbt_context(None, bad)
    assert context is None
    assert disclosure is not None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_workload_dbt.py -x -q`

Expected: FAIL — `ModuleNotFoundError: No module named 'sqlquality.workload.dbt'`.

- [ ] **Step 3: Implement the module**

```python
"""Optional dbt enrichment for `advise`.

dbt is *layered on top of* the engine-agnostic core, never underneath it: no workload adapter
imports this module, and every `advise` run behaves identically without a manifest. The
project's positioning is that the dbt-free path is first-class, so enrichment has to be
additive by construction rather than by discipline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from sqlquality.dbtproject import DbtProject, DbtProjectError, ModelNode
from sqlquality.models import Relation

#: Splits a dbt `relation_name` on unquoted dots. dbt quotes each part, so a dot *inside* a
#: quoted identifier (`"Weird.Name"`) must not split — hence matching quoted segments first.
_PART = re.compile(r'"((?:[^"]|"")*)"|([^.]+)')


def parse_relation_name(relation_name: str) -> Relation | None:
    """`(schema, table)` from a dbt `relation_name`, or None if it cannot be qualified.

    dbt writes a quoted three-part name — `'"dev"."main"."stg_orders"'` — and the raw node's
    own `schema` field is `None` in practice, so this string is the only reliable source. The
    database part is dropped because `advise` connects to one database at a time.

    A name with fewer than two parts returns None rather than a guess: a `Relation` needs a
    schema, and inventing one is how a production table gets attributed to an unrelated model.
    """
    parts = [
        (quoted if quoted is not None else bare).replace('""', '"')
        for quoted, bare in _PART.findall(relation_name.strip())
    ]
    parts = [p for p in parts if p]
    if len(parts) < 2:
        return None
    return Relation(schema=parts[-2], table=parts[-1])


@dataclass(frozen=True)
class DbtContext:
    """dbt models indexed by the relation they build, for joining against workload facts."""

    models: dict[Relation, ModelNode]

    @classmethod
    def from_project(cls, project: DbtProject) -> DbtContext:
        """Index every *model* by its relation.

        Only `resource_type == "model"` is indexed. Seeds, tests and snapshots also occupy
        relations, but "materialize this dbt test as a table" and "express this seed's index
        as dbt config" are both nonsense, and a seed sharing a name with a model's relation
        would otherwise silently win the mapping.
        """
        models: dict[Relation, ModelNode] = {}
        for uid in project.model_ids():
            node = project.node(uid)
            if node.resource_type != "model" or not node.relation_name:
                continue
            relation = parse_relation_name(node.relation_name)
            if relation is not None:
                models[relation] = node
        return cls(models=models)

    def model_for(self, relation: Relation) -> ModelNode | None:
        """The model building this exact relation, matching schema *and* table.

        Deliberately no bare-table-name fallback: dbt's `main`/`dev` target schemas routinely
        differ from the schema `advise` introspects, so a name-only match would attribute a
        production table to an unrelated development model — and then, via ADV302, rewrite
        that table's DDL on the strength of it.
        """
        return self.models.get(relation)


def load_dbt_context(
    project_dir: Path | None, manifest: Path | None
) -> tuple[DbtContext | None, str | None]:
    """Load a manifest if one was requested, returning `(context, disclosure)`.

    Never raises. A manifest that is missing, unreadable or malformed degrades to "no
    enrichment" plus a line for the user, because by the time this runs the whole catalog
    analysis has already happened — aborting would throw away real work over an optional
    input. Same reasoning as the report-write failure path in `cli.py`.
    """
    if manifest is None and project_dir is None:
        return None, None
    path = manifest if manifest is not None else (project_dir or Path()) / "target" / "manifest.json"
    try:
        project = DbtProject.from_path(path)
    except (OSError, ValueError, DbtProjectError) as exc:
        # `ValueError` covers `json.JSONDecodeError`, and `DbtProjectError` is a ValueError
        # subclass — both listed so the intent survives a refactor of either.
        return None, f"dbt enrichment unavailable: could not read {path}: {exc}"
    context = DbtContext.from_project(project)
    return context, f"dbt enrichment from {path} ({len(context.models)} model(s))"
```

- [ ] **Step 4: Wire the CLI options**

Add to `advise`, alongside the existing dbt-flavoured options:

```python
    project_dir: Path | None = typer.Option(
        None,
        "--project-dir",
        help="dbt project dir; reads target/manifest.json to enrich proposals (optional).",
    ),
    manifest: Path | None = typer.Option(
        None, "--manifest", help="Path to a dbt manifest.json. Overrides --project-dir."
    ),
```

In the body, after `proposals = adapter.propose(...)`, load the context and echo the
disclosure to stderr. Do not apply enrichment yet — Tasks 2-4 add the rules and Task 5 wires
them. For this task it is enough that the options parse, the manifest loads, and the
disclosure prints.

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_workload_dbt.py -q` then `uv run pytest -q`

Expected: PASS. The full suite must be unchanged in count except for the new tests.

- [ ] **Step 6: Prove the no-guess rule discriminates**

Add a bare-name fallback to `model_for` — `self.models.get(relation) or next((n for r, n in
self.models.items() if r.table == relation.table), None)` — and run
`tests/test_workload_dbt.py`. Expected: `test_context_does_not_match_on_a_bare_table_name`
FAILS. Restore. Report the mutation.

- [ ] **Step 7: Commit**

```bash
git add src/sqlquality/workload/dbt.py src/sqlquality/cli.py tests/test_workload_dbt.py
git commit -m "feat(advise): load an optional dbt manifest and index models by relation"
```

---

### Task 2: ADV302 — stop proposing DDL that dbt will destroy

**Files:**
- Modify: `src/sqlquality/workload/dbt.py`
- Test: `tests/test_workload_dbt.py`

**Interfaces:**
- Consumes: `DbtContext` (Task 1), `Proposal`, `Relation`.
- Produces:
  `enrich_proposals(proposals: list[Proposal], context: DbtContext) -> list[Proposal]` —
  rewrites index-creating proposals whose relation is a dbt model, and emits ADV302 notes.
  Returns the list unchanged when `context` has no matching model.

**This is the correctness fix.** A `CREATE INDEX` on a `table`-materialized dbt model is
destroyed by the next `dbt run`. The three materializations differ and the difference is the
whole rule:

| materialized | what happens to a raw `CREATE INDEX` | what to say |
|---|---|---|
| `table` | dropped on **every** `dbt run` | express as an `indexes` config entry; raw DDL will not survive |
| `incremental` | survives a normal run, lost on `--full-refresh` | express as config so a full refresh keeps it |
| `view` | cannot exist at all | the proposal is not applicable; a view has no storage to index |
| anything else / absent | unknown | say the materialization is unrecognised and leave the DDL alone |

- [ ] **Step 1: Write the failing tests**

```python
from sqlquality.models import Confidence, Proposal
from sqlquality.workload.dbt import enrich_proposals


def _index_proposal(relation, columns=("status",), code="ADV001"):
    quoted = ", ".join(f'"{c}"' for c in columns)
    return Proposal(
        code=code,
        title=f"Add index on {relation}({', '.join(columns)})",
        rationale="hot predicate.",
        evidence={
            "schema": relation.schema,
            "table": relation.table,
            "columns": tuple(columns),
            "cost_share": 0.5,
        },
        confidence=Confidence.HIGH,
        ddl=f'CREATE INDEX ON "{relation.schema}"."{relation.table}" ({quoted});',
    )


def test_adv302_replaces_raw_ddl_for_a_table_model_with_a_dbt_config_block():
    context = DbtContext.from_project(_project())
    relation = Relation("main", "orders")  # materialized: table
    [out] = enrich_proposals([_index_proposal(relation)], context)
    assert out.ddl is not None
    assert "CREATE INDEX" not in out.ddl
    assert "indexes" in out.ddl
    assert "columns" in out.ddl and "status" in out.ddl
    assert "dbt run" in out.rationale


def test_adv302_keeps_the_relation_and_columns_it_was_given():
    context = DbtContext.from_project(_project())
    relation = Relation("main", "orders")
    [out] = enrich_proposals([_index_proposal(relation, ("status", "created_at"))], context)
    assert "status" in out.ddl and "created_at" in out.ddl
    assert out.evidence["dbt_model"] == "model.demo.orders"
    assert out.evidence["dbt_materialized"] == "table"


def test_adv302_says_a_view_cannot_be_indexed_at_all():
    context = DbtContext.from_project(_project())
    relation = Relation("main", "stg_orders")  # materialized: view
    [out] = enrich_proposals([_index_proposal(relation)], context)
    assert out.ddl is None, "a view has no storage to index, so there is no DDL to run"
    assert "view" in out.rationale
    assert out.confidence is Confidence.LOW


def test_adv302_distinguishes_incremental_from_table():
    """An index survives a normal incremental run and is lost on --full-refresh. Saying
    'every dbt run drops it' would be false, and false in the direction that makes an
    operator distrust a correct proposal."""
    project = _project_with_materialization("model.demo.orders", "incremental")
    context = DbtContext.from_project(project)
    [out] = enrich_proposals([_index_proposal(Relation("main", "orders"))], context)
    assert "full-refresh" in out.rationale or "full refresh" in out.rationale
    assert "every dbt run" not in out.rationale


def test_adv302_leaves_an_unrecognised_materialization_alone_and_says_so():
    project = _project_with_materialization("model.demo.orders", "exotic")
    context = DbtContext.from_project(project)
    original = _index_proposal(Relation("main", "orders"))
    [out] = enrich_proposals([original], context)
    assert out.ddl == original.ddl, "unknown materialization must not have its DDL rewritten"
    assert "exotic" in out.rationale


def test_adv302_does_not_touch_a_relation_dbt_does_not_manage():
    context = DbtContext.from_project(_project())
    original = _index_proposal(Relation("public", "orders"))
    assert enrich_proposals([original], context) == [original]


def test_adv302_does_not_rewrite_a_drop_index_proposal():
    """Dropping an index dbt never created is a perfectly ordinary thing to do, and there is
    no `indexes` config that expresses a removal."""
    context = DbtContext.from_project(_project())
    drop = Proposal(
        code="ADV002",
        title="Drop unused index idx_cold on main.orders",
        rationale="no scans.",
        evidence={"schema": "main", "table": "orders", "index": "idx_cold"},
        confidence=Confidence.MEDIUM,
        ddl='DROP INDEX "main"."idx_cold";',
    )
    [out] = enrich_proposals([drop], context)
    assert out.ddl == drop.ddl


def test_adv302_does_not_rewrite_an_advisory_proposal_with_no_ddl():
    context = DbtContext.from_project(_project())
    advisory = Proposal(
        code="ADV005",
        title="Non-sargable predicate on main.orders.status",
        rationale="wrapped in a function.",
        evidence={"schema": "main", "table": "orders", "column": "status"},
        confidence=Confidence.HIGH,
        ddl=None,
    )
    [out] = enrich_proposals([advisory], context)
    assert out.ddl is None
    # It should still be attributed to the model, so the reader knows where to fix it.
    assert out.evidence["dbt_model"] == "model.demo.orders"
```

Add the helper the two materialization tests need, next to `_project`:

```python
def _project_with_materialization(uid: str, materialized: str) -> DbtProject:
    """The fixture manifest with one model's materialization changed.

    Edits a deep copy rather than a second fixture file: the point of variation is one field,
    and a whole extra manifest would drift from the real one.
    """
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    raw["nodes"][uid]["config"]["materialized"] = materialized
    return DbtProject.from_manifest(raw)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_workload_dbt.py -x -q`

Expected: FAIL — `ImportError: cannot import name 'enrich_proposals'`.

- [ ] **Step 3: Implement `enrich_proposals`**

Key points to get right, stated because each is a way to get it wrong:

- Detect an index-creating proposal by its **DDL prefix** (`CREATE INDEX`), not by rule code:
  ADV001, ADV007 and ADV008 all create indexes and Batch 3b will add more, so a code list
  would silently miss the next one.
- Preserve a `WHERE` clause if present. ADV004's partial index cannot be expressed by dbt's
  `indexes` config, which has no predicate field — so that proposal must be **disclosed as
  not expressible**, keeping its DDL and saying dbt will drop it. Do not silently emit a
  config block that loses the predicate.
- The generated config block is YAML for a human to paste, so it goes in `ddl` (the reviewable
  script) commented as configuration, not as an executable statement. `render_ddl` comments
  every non-DDL line already; make sure what you emit survives that renderer unchanged.

```python
#: dbt materializations whose relation is rebuilt, and what that does to a raw index.
_REBUILD = {
    "table": "every `dbt run` drops and recreates this relation, so a raw CREATE INDEX is lost",
    "incremental": (
        "a normal `dbt run` keeps this relation, but `dbt run --full-refresh` rebuilds it and a "
        "raw CREATE INDEX is lost"
    ),
}
```

Emit, for a rebuilt materialization, a `ddl` value of the form:

```
-- ADV302: express this as dbt config, not DDL. Add to the model's config block:
--   indexes:
--     - columns: ['status', 'created_at']
--       type: btree
```

and append to the rationale which materialization it is and what happens. Add
`"dbt_model"`, `"dbt_materialized"` and `"dbt_index_config"` to `evidence`.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_workload_dbt.py -q` then `uv run pytest -q`

Expected: PASS.

- [ ] **Step 5: Prove the materialization branches discriminate individually**

Three separate mutations, three results — this rule's whole value is that it distinguishes
cases, so a test that passes for two of three is the shape this project has been bitten by
seven times:

1. Make `incremental` share `table`'s wording. Expected: the incremental test FAILS.
2. Make `view` fall through to the rebuild branch. Expected: the view test FAILS.
3. Make an unrecognised materialization rewrite the DDL anyway. Expected: the exotic test FAILS.

Restore each. Report all three.

- [ ] **Step 6: Commit**

```bash
git add src/sqlquality/workload/dbt.py tests/test_workload_dbt.py
git commit -m "feat(advise): ADV302 -- express index proposals as dbt config, not doomed DDL"
```

---

### Task 3: ADV301 — a hot model materialized as a view

**Files:**
- Modify: `src/sqlquality/workload/dbt.py`
- Test: `tests/test_workload_dbt.py`

**Interfaces:**
- Produces:
  `propose_materialization(aggregation: Aggregation, context: DbtContext, *, min_cost_share: float) -> list[Proposal]`
  emitting code `"ADV301"`.
- Consumes: `Aggregation` (`usage`, `tables`), `DbtContext`.

A `view` re-executes its query on every read. When the workload shows a view-materialized model
carrying a hot share of cost, materializing it as a `table` or `incremental` trades build time
for read time. This is only computable by joining cost to the model graph, which is the point.

Confidence ceiling is **MEDIUM**, never HIGH, and there is deliberately no HIGH branch: the
trade depends on how often the model is rebuilt versus read, and on freshness requirements —
neither visible from query history. Follow ADV008's precedent and say so in the docstring so a
later reader does not add the missing rung for symmetry.

- [ ] **Step 1: Write the failing tests**

```python
def _aggregation(*usages, total=1000.0):
    return Aggregation(
        usage=tuple(usages),
        total_cost_ms=total,
        skipped_unqualifiable=0,
        tables=frozenset(u.relation for u in usages),
    )


def test_adv301_proposes_materializing_a_hot_view():
    context = DbtContext.from_project(_project())
    relation = Relation("main", "stg_orders")  # view
    usage = _usage(relation, "status", ColumnRole.EQUALITY, cost_share=0.4)
    proposals = propose_materialization(_aggregation(usage), context, min_cost_share=0.01)
    assert [p.code for p in proposals] == ["ADV301"]
    assert proposals[0].confidence is Confidence.MEDIUM
    assert proposals[0].evidence["dbt_model"] == "model.demo.stg_orders"
    assert proposals[0].ddl is None, "changing a materialization is a config edit, not DDL"


def test_adv301_is_silent_for_a_model_already_materialized_as_a_table():
    context = DbtContext.from_project(_project())
    usage = _usage(Relation("main", "orders"), "status", ColumnRole.EQUALITY, cost_share=0.9)
    assert propose_materialization(_aggregation(usage), context, min_cost_share=0.01) == []


def test_adv301_respects_the_cost_share_threshold():
    context = DbtContext.from_project(_project())
    usage = _usage(Relation("main", "stg_orders"), "status", ColumnRole.EQUALITY, cost_share=0.001)
    assert propose_materialization(_aggregation(usage), context, min_cost_share=0.01) == []


def test_adv301_never_reaches_high_confidence():
    """The build-vs-read trade is not visible from query history, so HIGH would be a claim
    about a schedule this tool cannot see."""
    context = DbtContext.from_project(_project())
    usage = _usage(Relation("main", "stg_orders"), "status", ColumnRole.EQUALITY, cost_share=0.99)
    [out] = propose_materialization(_aggregation(usage), context, min_cost_share=0.01)
    assert out.confidence is Confidence.MEDIUM


def test_adv301_ignores_relations_dbt_does_not_manage():
    context = DbtContext.from_project(_project())
    usage = _usage(Relation("public", "orders"), "status", ColumnRole.EQUALITY, cost_share=0.9)
    assert propose_materialization(_aggregation(usage), context, min_cost_share=0.01) == []


def test_adv301_reports_one_proposal_per_relation_not_per_column():
    """Two hot columns on one view are one materialization decision."""
    context = DbtContext.from_project(_project())
    relation = Relation("main", "stg_orders")
    proposals = propose_materialization(
        _aggregation(
            _usage(relation, "status", ColumnRole.EQUALITY, cost_share=0.4),
            _usage(relation, "created_at", ColumnRole.RANGE, cost_share=0.3),
        ),
        context,
        min_cost_share=0.01,
    )
    assert len(proposals) == 1
    assert proposals[0].evidence["cost_share"] == 0.4, "the max, not the sum"
```

- [ ] **Step 2: Run to verify they fail.** Expected: `NameError: propose_materialization`.
- [ ] **Step 3: Implement it.** One proposal per relation, `cost_share` as the max over that
      relation's usage (summing double-counts — see `ColumnUsage.cost_share`), sorted by
      relation for canonical output.
- [ ] **Step 4: Run the tests.** Expected: PASS.
- [ ] **Step 5: Prove the max-not-sum choice discriminates.** Change `max` to `sum` and confirm
      `test_adv301_reports_one_proposal_per_relation_not_per_column` FAILS. Restore, report.
- [ ] **Step 6: Commit**

```bash
git commit -am "feat(advise): ADV301 -- materialize a hot view-backed dbt model"
```

---

### Task 4: ADV303 — a dbt model the workload never touched

**Files:**
- Modify: `src/sqlquality/workload/dbt.py`
- Test: `tests/test_workload_dbt.py`

**Interfaces:**
- Produces:
  `propose_unused_models(aggregation: Aggregation, context: DbtContext, workload: Workload) -> list[Proposal]`
  emitting `"ADV303"`.

A model that costs a build every night and that nothing queries is worth knowing about. But the
evidence here is **absence**, which is much weaker than presence, so this rule needs the loudest
caveat in the codebase and a hard confidence cap of **LOW**:

- the window may simply not cover the reader (a monthly report, a BI tool with its own cache);
- `--limit` truncates query history, so a cold-but-used model can look unused;
- a model consumed only by *other models* is used, just not by ad-hoc queries — so a model with
  dbt children must be excluded outright, not merely downgraded.

The last point is a correctness gate, not a caveat: excluding models with children is what stops
this rule proposing the deletion of every staging model in a project.

- [ ] **Step 1: Write the failing tests**

```python
def test_adv303_flags_a_model_no_query_touched():
    context = DbtContext.from_project(_project())
    usage = _usage(Relation("main", "orders"), "status", ColumnRole.EQUALITY, cost_share=0.5)
    proposals = propose_unused_models(_aggregation(usage), context, _workload())
    codes = {p.code for p in proposals}
    assert codes == {"ADV303"}
    flagged = {p.evidence["dbt_model"] for p in proposals}
    assert "model.demo.customer_orders" in flagged


def test_adv303_excludes_a_model_that_other_models_depend_on():
    """A staging model consumed by a downstream model is used. Without this gate the rule
    proposes deleting every staging model in the project."""
    context = DbtContext.from_project(_project())
    usage = _usage(Relation("main", "orders"), "status", ColumnRole.EQUALITY, cost_share=0.5)
    proposals = propose_unused_models(_aggregation(usage), context, _workload())
    flagged = {p.evidence["dbt_model"] for p in proposals}
    assert "model.demo.stg_orders" not in flagged, "stg_orders feeds orders"


def test_adv303_is_capped_at_low_confidence():
    context = DbtContext.from_project(_project())
    usage = _usage(Relation("main", "orders"), "status", ColumnRole.EQUALITY, cost_share=0.5)
    for proposal in propose_unused_models(_aggregation(usage), context, _workload()):
        assert proposal.confidence is Confidence.LOW


def test_adv303_states_the_window_caveat_and_the_limit_caveat():
    context = DbtContext.from_project(_project())
    usage = _usage(Relation("main", "orders"), "status", ColumnRole.EQUALITY, cost_share=0.5)
    [first, *_] = propose_unused_models(_aggregation(usage), context, _workload())
    assert "window" in first.rationale
    assert "--limit" in first.rationale


def test_adv303_emits_nothing_when_every_model_was_touched():
    context = DbtContext.from_project(_project())
    usages = [
        _usage(relation, "status", ColumnRole.EQUALITY, cost_share=0.1)
        for relation in context.models
    ]
    assert propose_unused_models(_aggregation(*usages), context, _workload()) == []


def test_adv303_carries_no_ddl():
    """Deleting a model is a repository change with review implications; the tool must not
    hand over a statement that does it."""
    context = DbtContext.from_project(_project())
    usage = _usage(Relation("main", "orders"), "status", ColumnRole.EQUALITY, cost_share=0.5)
    for proposal in propose_unused_models(_aggregation(usage), context, _workload()):
        assert proposal.ddl is None
```

`_workload()` is a minimal `Workload`; reuse the helper style already in
`tests/test_workload_aggregate.py`.

- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Implement.** Use `DbtProject.model_children` — the `DbtContext` will need to
      carry the child relation, so extend it to keep `child_count: dict[Relation, int]` or hold
      the `DbtProject`. Prefer holding what you need rather than the whole project, and say why
      in the docstring.
- [ ] **Step 4: Run the tests.**
- [ ] **Step 5: Prove the children gate discriminates.** Remove it and confirm
      `test_adv303_excludes_a_model_that_other_models_depend_on` FAILS. Restore, report.
- [ ] **Step 6: Commit**

```bash
git commit -am "feat(advise): ADV303 -- a dbt model the analysed workload never touched"
```

---

### Task 5: wire enrichment into the command, and keep the dbt-free path identical

**Files:**
- Modify: `src/sqlquality/cli.py`
- Modify: `src/sqlquality/report.py`
- Test: `tests/test_advise_cli.py`

**Interfaces:**
- Produces: `advise` applying `enrich_proposals` and appending ADV301/ADV303 when a context
  loaded; `advise_payload` and `render_advise_markdown` carrying the dbt disclosure.

**The constraint that matters most in this task.** Every `advise` invocation without a manifest
must produce **byte-identical** output to `main` before this branch. Prove it, do not assert it:
capture a run's full stdout, JSON, markdown and DDL on `main`, then on the branch with no dbt
options, and diff them.

- [ ] **Step 1: Write the failing tests**

```python
def test_no_manifest_means_no_behaviour_change(tmp_path, monkeypatch):
    """The dbt-free path is first-class, so enrichment must be additive by construction."""
    without = _run_advise(tmp_path, extra=[])
    assert "dbt" not in without.stdout.lower()
    payload = json.loads(without.stdout)
    for proposal in payload["proposals"]:
        assert "dbt_model" not in proposal["evidence"]
    assert payload["dbt"] is None


def test_a_manifest_is_disclosed_on_stderr():
    result = runner.invoke(app, ["advise", ..., "--manifest", str(FIXTURE), "--json"])
    assert "dbt enrichment from" in result.stderr


def test_an_unreadable_manifest_does_not_fail_the_run(tmp_path):
    """Exit 0 with a disclosure — the catalog work already happened, and dbt is optional."""
    result = runner.invoke(app, ["advise", ..., "--manifest", str(tmp_path / "no.json")])
    assert result.exit_code == 0
    assert "dbt enrichment unavailable" in result.stderr


def test_the_payload_records_which_manifest_was_used():
    payload = ...  # a --json run with --manifest
    assert payload["dbt"]["manifest"].endswith("manifest_v12.json")
    assert payload["dbt"]["models"] >= 1


def test_adv301_and_adv303_only_appear_with_a_manifest():
    with_dbt = {p["code"] for p in _payload(extra=["--manifest", str(FIXTURE)])["proposals"]}
    without = {p["code"] for p in _payload(extra=[])["proposals"]}
    assert not ({"ADV301", "ADV303"} & without)
    # And at least one of them appears with the manifest, or this test proves nothing.
    assert {"ADV301", "ADV303"} & with_dbt
```

- [ ] **Step 2: Run to verify they fail.**
- [ ] **Step 3: Wire it.** Load the context, `enrich_proposals` the adapter's output, extend
      with ADV301/ADV303, then re-sort with the adapter's ranking key so ordering stays
      canonical. Pass the disclosure into both renderers and add a `"dbt"` key to the payload
      (`None` when absent).
- [ ] **Step 4: Prove the no-manifest path is byte-identical**

```bash
git stash && git checkout main
# run advise against the integration fixture, capturing stdout/json/markdown/ddl
git checkout - && git stash pop
# run the same invocation with no dbt options, and diff every artifact
```

Record the diff (which must be empty) in your report. If it is not empty, that is a defect in
this task, not an acceptable change.

- [ ] **Step 5: Run the whole suite and all four gates.**
- [ ] **Step 6: Commit**

---

### Task 6: prove it against a real project, and document it

**Files:**
- Modify: `tests/integration/` (a live run with a manifest)
- Modify: `README.md`, `CHANGELOG.md`
- Modify: `docs/superpowers/specs/2026-07-26-advise-workload-analysis-design.md`

- [ ] **Step 1: Add a live test** that runs `advise` against the seeded Postgres **with** a
      manifest whose `relation_name` schemas match the seeded schemas, and asserts that an
      index proposal for a dbt-managed table comes out as a config block rather than
      `CREATE INDEX`. Include a non-vacuity guard: assert the un-enriched run *did* produce a
      `CREATE INDEX` for that relation first, or the test proves nothing.
- [ ] **Step 2: Run the integration suite.**

Note: host port 55432 collides with an unrelated container on at least one dev machine, in
which case `docker compose up` does not bind and the suite silently talks to whatever else is
listening. If the seeded assertions behave oddly, check `docker ps --filter publish=55432`
before assuming a code defect.

- [ ] **Step 3: Confirm `uv run pytest` still reports zero skips**, and that the `no-extras`
      and `highest-deps` jobs would still pass (the new module must not import psycopg).
- [ ] **Step 4: Document.** README section on dbt enrichment including the ADV302 rationale
      (raw DDL on a dbt-managed table does not survive `dbt run`); ADV301/302/303 in the rule
      table; `--min-cost-share`'s help text if ADV301 is cost-weighted; CHANGELOG entries; and
      a spec deviation recording that matching is on the qualified `(schema, table)` pair with
      no bare-name fallback, and why.
- [ ] **Step 5: All four gates, then commit.**

---

## Self-Review

**Spec coverage.** The three approved dbt deliverables map to Tasks 2, 3 and 4; Task 1 is the
shared foundation, Task 5 the wiring, Task 6 the proof and docs. Snowflake is deliberately out
of scope (deferred pending an account); Redshift is Batch 3b.

**Placeholder scan.** Task 5's test bodies use `...` for the connection arguments of an
`advise` invocation, because that harness already exists in `tests/test_advise_cli.py` and
copying it here would drift from it. Every other step carries real code. The implementer must
read that file and match its existing stub-adapter fixture rather than invent a second one —
note that fixture's `fake_connect` was recently fixed to stop hard-coding `schemas`, so it is
the current one to follow.

**Type consistency.** `Relation`, `Proposal`, `Aggregation`, `Workload`, `ModelNode` and
`DbtProject` are all pre-existing and used with their current field names. `DbtContext` gains a
child-count map in Task 4; that is the one shape that changes mid-plan, and Task 4 says so.

**Known risk this plan accepts.** ADV302 rewrites DDL based on a manifest that may be stale —
someone can change a model's materialization without re-running `dbt compile`. The rule
discloses the materialization it read, so a wrong rewrite is traceable to a stale manifest
rather than invisible. Guarding harder would mean verifying the live relation against the
manifest, which is Batch 3b territory at best.
