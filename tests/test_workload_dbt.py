import json
from pathlib import Path

import pytest

from sqlquality.dbtproject import DbtProject
from sqlquality.models import Aggregation, ColumnRole, ColumnUsage, Confidence, Proposal, Relation
from sqlquality.workload.dbt import (
    DbtContext,
    _comment_block,
    enrich_proposals,
    load_dbt_context,
    parse_relation_name,
    propose_materialization,
)

FIXTURE = Path(__file__).parent / "fixtures" / "manifest_v12.json"


def _project() -> DbtProject:
    return DbtProject.from_path(FIXTURE)


def _project_with_materialization(uid: str, materialized: str) -> DbtProject:
    """The fixture manifest with one model's materialization changed.

    Edits a deep copy rather than a second fixture file: the point of variation is one field,
    and a whole extra manifest would drift from the real one.
    """
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    raw["nodes"][uid]["config"]["materialized"] = materialized
    return DbtProject.from_manifest(raw)


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


def test_parse_relation_name_strips_surrounding_whitespace():
    assert parse_relation_name('  "main"."orders"  ') == Relation("main", "orders")


def test_parse_relation_name_unescapes_a_doubled_quote():
    """`""` inside a quoted identifier is SQL's escape for a literal `"`, not a delimiter —
    `"ord""ers"` names the table `ord"ers`, not two segments."""
    assert parse_relation_name('"main"."ord""ers"') == Relation("main", 'ord"ers')


@pytest.mark.parametrize(
    "raw",
    [
        '"db".""."t"',  # an empty quoted segment can't be a schema
        '"db"."sch".',  # trailing dot: something was supposed to follow and didn't
        '"a"."b',  # unterminated quote on the last segment
    ],
)
def test_parse_relation_name_declines_a_malformed_segment_rather_than_shifting(raw):
    """A regex `findall` used to silently *skip* a character it couldn't match — so each of
    these had a part disappear instead of being rejected, and the remaining parts shifted
    one slot, misreading a `catalog.schema` pair as `schema.table`. Declining is correct:
    inventing a schema from a name we couldn't fully tokenize is how a production table
    gets attributed to an unrelated model."""
    assert parse_relation_name(raw) is None


def test_context_indexes_models_by_relation():
    context = DbtContext.from_project(_project())
    node = context.model_for(Relation("main", "stg_orders"))
    assert node is not None
    assert node.unique_id == "model.demo.stg_orders"
    assert node.materialized == "view"


def test_context_excludes_non_model_resources():
    """A seed and a test are not models: proposing a materialization change for a dbt
    test, or rewriting DDL because a seed shares a name, would both be nonsense.

    `DbtProject.model_ids()` is what actually guarantees this — it already filters to
    `resource_type == "model"` before `DbtContext.from_project` ever sees a unique_id, so
    there is no reachable guard left in this module to pin. This asserts the guarantee
    itself: no seed or test unique_id ever reaches `models`.
    """
    context = DbtContext.from_project(_project())
    assert context.model_for(Relation("main", "raw_orders")) is None  # the seed's relation
    unique_ids = {node.unique_id for node in context.models.values()}
    assert not any(uid.startswith(("seed.", "test.")) for uid in unique_ids)


def test_context_skips_a_model_with_no_relation_name():
    """An ephemeral materialization is inlined as a CTE and never occupies a physical
    relation, so dbt leaves its `relation_name` unset — unlike the resource-type check,
    this guard *is* reachable: removing it would call `parse_relation_name(None)` and
    raise, rather than just index one extra model."""
    manifest = {
        "nodes": {
            "model.demo.ephemeral_thing": {
                "resource_type": "model",
                "config": {"materialized": "ephemeral"},
                "relation_name": None,
            },
        },
    }
    context = DbtContext.from_project(DbtProject.from_manifest(manifest))
    assert context.models == {}


def test_context_does_not_match_on_a_bare_table_name():
    """dbt's target schema routinely differs from the introspected one. Matching `orders`
    in schema `public` to a model in schema `main` would attribute a production table to an
    unrelated dev model and then rewrite its DDL."""
    context = DbtContext.from_project(_project())
    assert context.model_for(Relation("public", "orders")) is None
    assert context.model_for(Relation("main", "orders")) is not None


def test_context_declines_a_cross_database_collision():
    """Two different databases can each have a `main.orders` — `'"prod"."main"."orders"'`
    and `'"stage"."main"."orders"'` both key `Relation("main", "orders")` once the database
    is dropped. `advise` connects to one database at a time, so there is no way to tell
    which model actually built the relation it introspected — guessing is exactly the
    failure `model_for`'s no-bare-name-fallback rule exists to prevent, applied to a
    different source of ambiguity. The relation must be dropped, not resolved to whichever
    unique_id happened to sort last."""
    manifest = {
        "nodes": {
            "model.demo.orders_prod": {
                "resource_type": "model",
                "config": {"materialized": "table"},
                "relation_name": '"prod"."main"."orders"',
            },
            "model.demo.orders_stage": {
                "resource_type": "model",
                "config": {"materialized": "table"},
                "relation_name": '"stage"."main"."orders"',
            },
        },
    }
    context = DbtContext.from_project(DbtProject.from_manifest(manifest))
    assert context.model_for(Relation("main", "orders")) is None
    assert context.dropped_collisions == 1


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


@pytest.mark.parametrize(
    "manifest_json",
    [
        "[]",
        '"s"',
        "42",
        "null",
        '{"nodes": "abc"}',
        '{"nodes": null}',
        '{"nodes": {"model.x": "abc"}}',
        '{"nodes": {"model.x": {"resource_type": "model", "relation_name": 42, "config": {}}}}',
    ],
    ids=[
        "top-level-list",
        "top-level-string",
        "top-level-int",
        "top-level-null",
        "nodes-is-a-string",
        "nodes-is-null",
        "node-value-is-not-an-object",
        "relation_name-is-not-a-string",
    ],
)
def test_load_survives_a_wrong_shaped_manifest_without_raising(tmp_path, manifest_json):
    """Each of these is *valid JSON* but the wrong shape for a manifest, and each raises a
    different AttributeError/TypeError deep inside DbtProject or DbtContext — reproduced
    end-to-end, `advise --manifest ...` used to exit 1 with a traceback for every one of
    them, *after* the whole catalog analysis had already run. A narrow except list missed
    all of these; each case must independently stay green."""
    bad = tmp_path / "manifest.json"
    bad.write_text(manifest_json, encoding="utf-8")
    context, disclosure = load_dbt_context(None, bad)
    assert context is None
    assert disclosure is not None


def test_adv302_replaces_raw_ddl_for_a_table_model_with_a_dbt_config_block():
    context = DbtContext.from_project(_project())
    relation = Relation("main", "orders")  # materialized: table
    [out] = enrich_proposals([_index_proposal(relation)], context)
    assert out.ddl is not None
    assert "CREATE INDEX" not in out.ddl
    assert "indexes" in out.ddl
    assert "columns" in out.ddl and "status" in out.ddl
    assert "type: btree" in out.ddl
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
    # The substantive claim, not just the word "view" — the generic attribution string
    # `(materialized as `view`)` alone would already satisfy a bare `"view" in rationale`
    # even if this sentence were stripped or replaced with something generic.
    assert "has no storage of its own to index" in out.rationale
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
    no `indexes` config that expresses a removal.

    `evidence` deliberately includes a `columns` tuple, the same shape ADV001/007/008
    carry: without it, a broken `_is_index_creating` that wrongly called this DROP
    "index-creating" would still slip through the *separate* "no columns to express as
    config" bail-out and leave `ddl` untouched by accident — passing this test for the
    wrong reason instead of actually exercising the DDL-prefix guard it claims to pin.
    """
    context = DbtContext.from_project(_project())
    drop = Proposal(
        code="ADV002",
        title="Drop unused index idx_cold on main.orders",
        rationale="no scans.",
        evidence={
            "schema": "main",
            "table": "orders",
            "index": "idx_cold",
            "columns": ("status",),
        },
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


def test_adv302_discloses_a_partial_index_as_not_expressible_rather_than_dropping_the_where():
    """ADV004's partial index has a WHERE clause dbt's `indexes` config has no field for.
    Silently emitting a config block would lose the predicate and turn a correct proposal
    into a wrong one, so it must keep its raw DDL and say dbt will drop it instead."""
    context = DbtContext.from_project(_project())
    relation = Relation("main", "orders")  # materialized: table
    partial = Proposal(
        code="ADV004",
        title=f"Partial index on {relation}(status) WHERE deleted_at IS NULL",
        rationale="hot predicate, restricted by a null check.",
        evidence={
            "schema": relation.schema,
            "table": relation.table,
            "columns": ("status",),
            "guard_column": "deleted_at",
            "guard_predicate": "IS NULL",
            "cost_share": 0.5,
        },
        confidence=Confidence.MEDIUM,
        ddl=(
            f'CREATE INDEX ON "{relation.schema}"."{relation.table}" '
            '("status") WHERE "deleted_at" IS NULL;'
        ),
    )
    [out] = enrich_proposals([partial], context)
    assert out.ddl == partial.ddl, "the WHERE clause must survive, not be silently dropped"
    assert "WHERE" in out.ddl
    assert "dbt" in out.rationale and "drop" in out.rationale.lower()


def test_adv302_does_not_mistake_a_column_named_where_for_a_partial_index():
    """A substring search for "WHERE" in the DDL is foolable by a column literally named
    `WHERE`: quoted, it is a perfectly ordinary identifier, but `CREATE INDEX ON t
    ("WHERE")` contains the substring anyway. Detection keys on ADV004's own
    `guard_column`/`guard_predicate` evidence instead, which this proposal does not carry,
    so it must be rewritten normally rather than disclosed as an inexpressible partial
    index."""
    context = DbtContext.from_project(_project())
    relation = Relation("main", "orders")  # materialized: table
    [out] = enrich_proposals([_index_proposal(relation, ("WHERE",))], context)
    assert out.ddl is not None
    assert "CREATE INDEX" not in out.ddl, "a column named WHERE must not block the rewrite"
    assert "indexes" in out.ddl


def test_adv302_detects_a_lowercase_where_partial_index_via_evidence_not_text():
    """Every rule in this codebase emits an uppercase `WHERE` today, but detection must not
    depend on that: keying on ADV004's `guard_column`/`guard_predicate` evidence catches a
    lowercase `where` (plausible from a future engine's rule) exactly the same as an
    uppercase one, whereas a text search for `"WHERE"` would silently miss it and drop the
    predicate into a config block that has nowhere to put it."""
    context = DbtContext.from_project(_project())
    relation = Relation("main", "orders")  # materialized: table
    partial = Proposal(
        code="ADV004",
        title=f"Partial index on {relation}(status) where deleted_at is null",
        rationale="hot predicate, restricted by a null check.",
        evidence={
            "schema": relation.schema,
            "table": relation.table,
            "columns": ("status",),
            "guard_column": "deleted_at",
            "guard_predicate": "is null",
            "cost_share": 0.5,
        },
        confidence=Confidence.MEDIUM,
        ddl=(
            f'CREATE INDEX ON "{relation.schema}"."{relation.table}" '
            '("status") where "deleted_at" is null;'
        ),
    )
    [out] = enrich_proposals([partial], context)
    assert out.ddl == partial.ddl, "a lowercase predicate must still be disclosed, not dropped"


def test_adv302_expresses_a_unique_index_with_dbts_unique_config_field():
    """`CREATE UNIQUE INDEX` is still index-creating DDL a table rebuild destroys, and
    dbt's `indexes` config has a `unique` field for exactly this — so it must be rewritten
    (not silently skipped) and the uniqueness preserved rather than dropped."""
    context = DbtContext.from_project(_project())
    relation = Relation("main", "orders")  # materialized: table
    unique = Proposal(
        code="ADV001",
        title=f"Add unique index on {relation}(email)",
        rationale="hot predicate.",
        evidence={"schema": relation.schema, "table": relation.table, "columns": ("email",)},
        confidence=Confidence.HIGH,
        ddl=f'CREATE UNIQUE INDEX ON "{relation.schema}"."{relation.table}" ("email");',
    )
    [out] = enrich_proposals([unique], context)
    assert out.ddl is not None
    assert "CREATE UNIQUE INDEX" not in out.ddl
    assert "unique: true" in out.ddl


def test_adv302_recognises_create_index_concurrently_as_index_creating():
    """The DDL script's own header recommends CONCURRENTLY for a live table, so a
    proposal that used it must not silently stop being detected as index-creating —
    that is exactly the silent-skip the DDL-prefix requirement exists to prevent."""
    context = DbtContext.from_project(_project())
    relation = Relation("main", "orders")  # materialized: table
    concurrent = Proposal(
        code="ADV001",
        title=f"Add index on {relation}(status)",
        rationale="hot predicate.",
        evidence={"schema": relation.schema, "table": relation.table, "columns": ("status",)},
        confidence=Confidence.HIGH,
        ddl=f'CREATE INDEX CONCURRENTLY ON "{relation.schema}"."{relation.table}" ("status");',
    )
    [out] = enrich_proposals([concurrent], context)
    assert out.ddl is not None
    assert "CREATE INDEX" not in out.ddl
    assert "indexes" in out.ddl


def test_adv302_rewrites_a_materialized_view_instead_of_calling_it_unrecognised():
    """`materialized_view` has been a real dbt materialization since dbt-core 1.6 and
    supports an `indexes` config exactly like `table`/`incremental` — calling it
    "unrecognised" is safe but wrong, since the operator is left with DDL a rebuild or
    full refresh destroys."""
    project = _project_with_materialization("model.demo.orders", "materialized_view")
    context = DbtContext.from_project(project)
    [out] = enrich_proposals([_index_proposal(Relation("main", "orders"))], context)
    assert out.ddl is not None
    assert "CREATE INDEX" not in out.ddl
    assert "indexes" in out.ddl
    assert "unrecognised" not in out.rationale
    assert "materialized view" in out.rationale


def test_adv302_config_block_survives_a_newline_in_a_column_name():
    """A newline inside a quoted identifier parses successfully (parse_relation_name
    accepts one in a relation name), and a column introspected from a live catalog can
    carry the same thing. The generated config block goes straight into the --ddl file as
    `--`-commented lines, so an embedded raw newline there must not let the second half of
    the line break out of the comment — the same hazard render_ddl already defends against
    for raw DDL, reproduced for enrich_proposals' own generated block.
    """
    context = DbtContext.from_project(_project())
    relation = Relation("main", "orders")  # materialized: table
    hostile_columns = ("sta\ntus -- DROP TABLE users;",)
    [out] = enrich_proposals([_index_proposal(relation, hostile_columns)], context)
    assert out.ddl is not None
    for line in out.ddl.splitlines():
        assert line.startswith("--"), f"bare line outside comment mode: {line!r}"
    # The hostile text must not appear as a live, uncommented statement anywhere.
    executable = [ln for ln in out.ddl.splitlines() if not ln.startswith("--")]
    assert not any("DROP TABLE users" in ln for ln in executable)
    # Pins the specific mechanism: `list(columns)` formatting escapes the embedded
    # newline into the two literal characters `\` and `n` — this is what actually
    # neutralizes the hazard here, not `_comment_block`'s resplitting (see
    # `test_comment_block_defends_a_raw_newline_smuggled_past_the_repr_escaping` for
    # that defense pinned in isolation).
    assert "\\n" in out.ddl


def test_comment_block_defends_a_raw_newline_smuggled_past_the_repr_escaping():
    """`_comment_block` is `enrich_proposals`' own equivalent of `render_ddl`'s per-line
    comment guard. The end-to-end hazard test above never actually exercises this
    function's own resplitting, because `list(columns)` formatting already escapes an
    embedded newline before `_comment_block` ever sees it — so this pins the second,
    independent defense directly: a raw `\\n` inside one logical line, bypassing any
    repr-based escaping entirely, must still not produce a bare physical line."""
    rendered = _comment_block(["safe line", "unsafe\nline -- DROP TABLE users;", "also safe"])
    lines = rendered.splitlines()
    assert len(lines) == 4  # three logical lines, one of which splits into two physical ones
    for line in lines:
        assert line.startswith("--"), f"bare line outside comment mode: {line!r}"


def _usage(relation, column, role, cost_share=0.5, cost_ms=50.0, fps=("fp1",)):
    """Mirrors `tests/test_workload_rules.py`'s helper of the same name — not imported
    across test modules, because that file's helper is private to it. `fps` defaults to a
    single shared fingerprint, so usages co-occur unless a test deliberately gives them
    disjoint sets; irrelevant to the dbt rules below (neither checks co-occurrence) but kept
    for parity with the original."""
    return ColumnUsage(
        relation=relation,
        column=column,
        role=role,
        calls=10,
        cost_ms=cost_ms,
        cost_share=cost_share,
        fingerprint_ids=frozenset(fps),
    )


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
