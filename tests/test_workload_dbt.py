import dataclasses
import json
from pathlib import Path

import pytest

from sqlquality.dbtproject import DbtProject
from sqlquality.models import (
    Aggregation,
    ColumnRole,
    ColumnUsage,
    Confidence,
    Proposal,
    Relation,
    Workload,
)
from sqlquality.workload.dbt import (
    DbtContext,
    _comment_block,
    _manifest_warnings,
    describe_rewrites,
    enrich_proposals,
    load_dbt_context,
    parse_relation_name,
    propose_materialization,
    propose_unused_models,
    resolve_manifest_path,
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
        # Garbage immediately after a closing quote, with no dot between. This is the shape
        # the scanner's own docstring is about and the one the parametrization was missing:
        # without the "the char after a quoted segment must be a dot" reject, this parses as
        # three parts `a`, `b`, `y` -- the `x` vanishes and every later part shifts one slot,
        # so `Relation("b", "y")` is returned for a name that names neither.
        '"a"."b"xy',
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

    The exact set, not just the two negatives: with only negative assertions this test passed
    against an empty model index, so it could not distinguish "seeds and tests are excluded"
    from "nothing is indexed at all" — and it read as coverage for the former.
    """
    context = DbtContext.from_project(_project())
    unique_ids = {node.unique_id for node in context.models.values()}
    assert unique_ids == {
        "model.demo.stg_orders",
        "model.demo.orders",
        "model.demo.customer_orders",
    }, "the three models, and only them — the fixture also carries a seed and a test"
    assert context.model_for(Relation("main", "raw_orders")) is None  # the seed's relation


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
    assert "indexes:" not in out.ddl


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


def _workload() -> Workload:
    """A minimal Workload, in the style of `tests/test_workload_aggregate.py`'s `_workload`
    helper (not imported across test modules — see that file's own helper). ADV303 only
    reads `window_description` off it, so an empty `stats` tuple is enough."""
    return Workload(stats=(), window_description="the last 7 days")


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
    proposals = propose_unused_models(_aggregation(usage), context, _workload())
    assert proposals, "the loop below is vacuous otherwise — this must pin a non-empty result"
    for proposal in proposals:
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
    proposals = propose_unused_models(_aggregation(usage), context, _workload())
    assert proposals, "the loop below is vacuous otherwise — this must pin a non-empty result"
    for proposal in proposals:
        assert proposal.ddl is None


def _unrelated_usage() -> ColumnUsage:
    """A usage on a relation outside every fixture project used here, just enough to keep
    `Aggregation.usage` non-empty so the "nothing was analysed" bail-out (see
    `test_adv303_emits_nothing_when_no_usage_was_extracted`) does not swallow a test that
    is not testing that guard."""
    return _usage(Relation("other", "noise"), "id", ColumnRole.EQUALITY, cost_share=0.01)


def test_adv303_emits_nothing_when_no_usage_was_extracted():
    """An `Aggregation` with no usage at all means nothing was analysed — every relation in
    `context.models` would trivially look "untouched" by definition (none of them can be in
    `aggregation.tables`, which is built only from usage), so an empty workload or a fully
    unparseable one must not read as evidence that every childless model is unused."""
    context = DbtContext.from_project(_project())
    assert propose_unused_models(_aggregation(), context, _workload()) == []


def test_adv303_excludes_a_model_with_exactly_one_model_child():
    """`> 0` and `> 1` both leave every other test green if the only fixture with children
    happens to have two of them (`stg_orders` feeds both `orders` and `customer_orders`).
    This is the commonest real shape — one staging model feeding one downstream model — so
    it needs its own fixture to be pinned at all.

    `orphan` is in the fixture so the assertion discriminates: with only `parent` and `child`
    the expected result is the empty set, which is also what a rule that flagged *nothing*
    produces — so the test passed without the exclusion it claims to pin ever being reached.
    """
    manifest = {
        "nodes": {
            "model.demo.parent": {
                "resource_type": "model",
                "config": {"materialized": "table"},
                "relation_name": '"dev"."main"."parent"',
            },
            "model.demo.child": {
                "resource_type": "model",
                "config": {"materialized": "table"},
                "relation_name": '"dev"."main"."child"',
            },
            "model.demo.orphan": {
                "resource_type": "model",
                "config": {"materialized": "table"},
                "relation_name": '"dev"."main"."orphan"',
            },
        },
        "child_map": {
            "model.demo.parent": ["model.demo.child"],
            "model.demo.child": [],
            "model.demo.orphan": [],
        },
    }
    context = DbtContext.from_project(DbtProject.from_manifest(manifest))
    usage = _usage(Relation("main", "child"), "status", ColumnRole.EQUALITY, cost_share=0.5)
    proposals = propose_unused_models(_aggregation(usage), context, _workload())
    flagged = {p.evidence["dbt_model"] for p in proposals}
    assert flagged == {"model.demo.orphan"}, "parent has exactly one model child; orphan has none"


def test_adv303_excludes_a_model_whose_only_child_is_a_snapshot():
    """An exposure/snapshot is a real, dbt-declared consumer that `model_children` cannot
    see because it filters to `resource_type == 'model'`. ADV303 must read the manifest's
    raw child_map (via `DbtProject.child_ids`) instead, or it would propose deleting a model
    dbt itself documents as being snapshotted.

    `orphan` is in the fixture for the same reason as in the one-model-child test: without a
    model this rule *does* flag, the expected result is the empty set and the test passes
    against a rule that flags nothing at all.
    """
    manifest = {
        "nodes": {
            "model.demo.raw": {
                "resource_type": "model",
                "config": {"materialized": "table"},
                "relation_name": '"dev"."main"."raw"',
            },
            "model.demo.orphan": {
                "resource_type": "model",
                "config": {"materialized": "table"},
                "relation_name": '"dev"."main"."orphan"',
            },
            "snapshot.demo.raw_snapshot": {
                "resource_type": "snapshot",
                "config": {"materialized": "snapshot"},
                "relation_name": '"dev"."main"."raw_snapshot"',
            },
        },
        "child_map": {
            "model.demo.raw": ["snapshot.demo.raw_snapshot"],
            "model.demo.orphan": [],
            "snapshot.demo.raw_snapshot": [],
        },
    }
    context = DbtContext.from_project(DbtProject.from_manifest(manifest))
    proposals = propose_unused_models(_aggregation(_unrelated_usage()), context, _workload())
    flagged = {p.evidence["dbt_model"] for p in proposals}
    assert flagged == {"model.demo.orphan"}, "a snapshot is a declared consumer; orphan has none"


def test_adv303_excludes_a_model_whose_only_child_is_an_exposure():
    """An exposure exists in dbt specifically to declare "a BI dashboard / a downstream
    tool reads this" — a mart whose only declared consumer is an exposure is exactly the
    case this rule must not flag. Exposures live outside `nodes` in a real manifest, so
    this only works because `child_ids` reads `child_map` directly rather than resolving
    each child through `DbtProject.node`.

    `orphan` is in the fixture so the expected result is a non-empty set: as an
    absence-only assertion this passed against a rule that flagged nothing.
    """
    manifest = {
        "nodes": {
            "model.demo.mart": {
                "resource_type": "model",
                "config": {"materialized": "table"},
                "relation_name": '"dev"."main"."mart"',
            },
            "model.demo.orphan": {
                "resource_type": "model",
                "config": {"materialized": "table"},
                "relation_name": '"dev"."main"."orphan"',
            },
        },
        "child_map": {
            "model.demo.mart": ["exposure.demo.dashboard"],
            "model.demo.orphan": [],
        },
    }
    context = DbtContext.from_project(DbtProject.from_manifest(manifest))
    proposals = propose_unused_models(_aggregation(_unrelated_usage()), context, _workload())
    flagged = {p.evidence["dbt_model"] for p in proposals}
    assert flagged == {"model.demo.orphan"}, "an exposure is a declared consumer; orphan has none"


def test_adv303_does_not_count_a_test_as_a_consumer():
    """A `not_null` test is an assertion about a model, not a consumer of it: it does not
    read the model's output for any purpose downstream would recognise, so a model whose
    only child is a test is still unused and must be flagged, unlike a snapshot or an
    exposure."""
    manifest = {
        "nodes": {
            "model.demo.mart": {
                "resource_type": "model",
                "config": {"materialized": "table"},
                "relation_name": '"dev"."main"."mart"',
            },
            "test.demo.not_null_mart_id.abc123": {
                "resource_type": "test",
                "config": {"materialized": "test"},
                "relation_name": None,
            },
        },
        "child_map": {"model.demo.mart": ["test.demo.not_null_mart_id.abc123"]},
    }
    context = DbtContext.from_project(DbtProject.from_manifest(manifest))
    proposals = propose_unused_models(_aggregation(_unrelated_usage()), context, _workload())
    flagged = {p.evidence["dbt_model"] for p in proposals}
    assert flagged == {"model.demo.mart"}


def test_adv303_orders_proposals_canonically_by_relation():
    """Canonical output order is by relation, not by dbt unique_id or manifest insertion
    order — chosen so the two orders disagree: `DbtProject.model_ids()` already sorts by
    unique_id, so removing `propose_unused_models`'s own `sorted(context.models)` would
    still pass by accident unless a model's unique_id order disagrees with its relation's
    (schema, table) order, as it deliberately does here."""
    manifest = {
        "nodes": {
            "model.demo.a_second": {
                "resource_type": "model",
                "config": {"materialized": "table"},
                "relation_name": '"dev"."main"."zzz_relation"',
            },
            "model.demo.b_first": {
                "resource_type": "model",
                "config": {"materialized": "table"},
                "relation_name": '"dev"."main"."aaa_relation"',
            },
        },
    }
    context = DbtContext.from_project(DbtProject.from_manifest(manifest))
    proposals = propose_unused_models(_aggregation(_unrelated_usage()), context, _workload())
    assert [p.evidence["table"] for p in proposals] == ["aaa_relation", "zzz_relation"]


def test_adv301_orders_proposals_canonically_by_relation():
    """Same canonical-order requirement as ADV303, pinned independently for
    `propose_materialization`: without its own `sorted(by_relation)`, output would follow
    the order usages were supplied in, not relation order."""
    manifest = {
        "nodes": {
            "model.demo.z_model": {
                "resource_type": "model",
                "config": {"materialized": "view"},
                "relation_name": '"dev"."main"."z_model"',
            },
            "model.demo.a_model": {
                "resource_type": "model",
                "config": {"materialized": "view"},
                "relation_name": '"dev"."main"."a_model"',
            },
        },
    }
    context = DbtContext.from_project(DbtProject.from_manifest(manifest))
    proposals = propose_materialization(
        _aggregation(
            _usage(Relation("main", "z_model"), "status", ColumnRole.EQUALITY, cost_share=0.5),
            _usage(Relation("main", "a_model"), "status", ColumnRole.EQUALITY, cost_share=0.5),
        ),
        context,
        min_cost_share=0.01,
    )
    assert [p.evidence["table"] for p in proposals] == ["a_model", "z_model"]


# --- ADV302: the generated config block -----------------------------------------------


def _config_mapping(ddl: str) -> dict:
    """The `indexes:` mapping inside a generated ADV302 block, parsed as YAML.

    The block exists to be pasted into a dbt model's `.yml`, and nothing asserted it was
    valid YAML at all — so `list(columns)` → `tuple(columns)` survived mutation, emitting
    `columns: ('status',)`, which PyYAML (dbt's own parser) rejects outright. Stripping the
    `--` comment prefixes and dropping the prose above `indexes:` is exactly what a human
    copying the block into a model config does.
    """
    import yaml

    lines = [ln[3:] if ln.startswith("-- ") else ln[2:] for ln in ddl.splitlines()]
    start = next(i for i, ln in enumerate(lines) if ln.strip() == "indexes:")
    parsed = yaml.safe_load("\n".join(lines[start:]))
    assert isinstance(parsed, dict), parsed
    return parsed


def _indexes_key_lines(proposals) -> list[str]:
    """Every emitted line that opens an `indexes:` mapping, across a whole run.

    Counting this is the direct form of the property: dbt reads one `indexes` key per model
    config, so a second one for the same model is a duplicate YAML key and is silently
    dropped.
    """
    return [
        ln
        for p in proposals
        for ln in (p.ddl or "").splitlines()
        if ln.removeprefix("--").strip() == "indexes:"
    ]


def test_adv302_config_block_is_valid_yaml():
    """The block's only purpose is to be pasted into a `.yml`, and no test parsed it."""
    context = DbtContext.from_project(_project())
    [out] = enrich_proposals([_index_proposal(Relation("main", "orders"), ("status",))], context)
    assert _config_mapping(out.ddl) == {"indexes": [{"columns": ["status"], "type": "btree"}]}


def test_adv302_merges_every_index_for_one_model_into_a_single_config_block():
    """Two proposals on one dbt model must produce ONE `indexes:` block, not two.

    Two standalone blocks pasted under a single model's `config:` are a duplicate YAML
    mapping key, and PyYAML — dbt's parser — keeps only the last, silently discarding the
    other recommended index with no error. Two survivors per relation is the *normal* case:
    the adapter's collapse layer never folds non-prefix column lists, and deliberately
    preserves same-set-different-order pairs.
    """
    context = DbtContext.from_project(_project())
    relation = Relation("main", "orders")
    first = _index_proposal(relation, ("status", "created_at"), code="ADV001")
    second = _index_proposal(relation, ("customer_id",), code="ADV007")

    out = enrich_proposals([first, second], context)

    assert [p.code for p in out] == ["ADV001", "ADV007"], "input order must be preserved"
    assert len(_indexes_key_lines(out)) == 1, (
        "one model, one `indexes:` block — a second one is a duplicate YAML key that dbt "
        "silently resolves by keeping only one of them"
    )
    [owner] = [p for p in out if p.evidence.get("dbt_index_config") is True]
    assert _config_mapping(owner.ddl) == {
        "indexes": [
            {"columns": ["status", "created_at"], "type": "btree"},
            {"columns": ["customer_id"], "type": "btree"},
        ]
    }, "both recommended indexes, in input (ranked) order, in one block"
    [deferred] = [p for p in out if "dbt_index_config_reported_with" in p.evidence]
    assert deferred.code == "ADV007"
    assert deferred.evidence["dbt_index_config_reported_with"] == "ADV001"
    assert deferred.ddl is not None
    assert "CREATE INDEX" not in deferred.ddl, "still not doomed DDL"
    assert "customer_id" in deferred.ddl, "must say which index it is"
    assert "ADV001" in deferred.ddl, "and where the block carrying it is"
    assert "ADV001" in deferred.rationale


def test_adv302_merges_three_indexes_and_keeps_a_unique_one_distinct():
    """A merged block must not flatten the per-entry fields: three entries, one unique."""
    context = DbtContext.from_project(_project())
    relation = Relation("main", "orders")
    plain_a = _index_proposal(relation, ("status",), code="ADV001")
    plain_b = _index_proposal(relation, ("customer_id",), code="ADV007")
    unique = _index_proposal(relation, ("order_key",), code="ADV008")
    unique = dataclasses.replace(
        unique, ddl='CREATE UNIQUE INDEX ON "main"."orders" ("order_key");'
    )

    out = enrich_proposals([plain_a, plain_b, unique], context)

    assert len(_indexes_key_lines(out)) == 1
    [owner] = [p for p in out if p.evidence.get("dbt_index_config") is True]
    assert _config_mapping(owner.ddl) == {
        "indexes": [
            {"columns": ["status"], "type": "btree"},
            {"columns": ["customer_id"], "type": "btree"},
            {"columns": ["order_key"], "type": "btree", "unique": True},
        ]
    }


def test_adv302_does_not_repeat_an_identical_entry_in_the_merged_block():
    """Two proposals reducing to the same index contribute one entry, not two identical ones."""
    context = DbtContext.from_project(_project())
    relation = Relation("main", "orders")
    out = enrich_proposals(
        [
            _index_proposal(relation, ("status",), code="ADV001"),
            _index_proposal(relation, ("status",), code="ADV008"),
        ],
        context,
    )
    [owner] = [p for p in out if p.evidence.get("dbt_index_config") is True]
    assert _config_mapping(owner.ddl) == {"indexes": [{"columns": ["status"], "type": "btree"}]}


def test_adv302_keeps_one_block_per_model_when_two_models_are_involved():
    """Merging is per model, not per run: two dbt models get one block each."""
    context = DbtContext.from_project(_project())
    out = enrich_proposals(
        [
            _index_proposal(Relation("main", "orders"), ("status",), code="ADV001"),
            _index_proposal(Relation("main", "customer_orders"), ("status",), code="ADV007"),
        ],
        context,
    )
    assert len(_indexes_key_lines(out)) == 2, "two models, two blocks — neither collides"
    assert all(p.evidence.get("dbt_index_config") is True for p in out)


def test_adv302_config_block_names_its_model_so_two_relations_never_render_the_same_text():
    """Two relations recommending the same column list used to render byte-identical `ddl`
    *and* identical evidence — distinguishable in the DDL file only by the title comment
    above them, and in the JSON payload by nothing at all. The block says which model to
    paste it into, so it has to name that model."""
    context = DbtContext.from_project(_project())
    [orders] = enrich_proposals(
        [_index_proposal(Relation("main", "orders"), ("customer_id",))], context
    )
    [customer_orders] = enrich_proposals(
        [_index_proposal(Relation("main", "customer_orders"), ("customer_id",))], context
    )
    assert orders.ddl != customer_orders.ddl
    assert "model.demo.orders" in orders.ddl
    assert "model.demo.customer_orders" in customer_orders.ddl


def test_adv302_never_claims_the_config_block_is_above_anything():
    """ "Add the config block above" was wrong in every surface: in markdown the fenced DDL is
    *below* the rationale, in the DDL file the rationale is absent entirely, and in JSON there
    is no spatial relation at all. The block replaces the DDL, so it is never "above" it."""
    context = DbtContext.from_project(_project())
    [out] = enrich_proposals([_index_proposal(Relation("main", "orders"))], context)
    assert "block above" not in out.rationale
    assert "above" not in out.ddl


def test_adv302_evidence_carries_a_flag_not_a_copy_of_the_ddl():
    """`dbt_index_config` used to hold a byte-identical copy of `ddl`. It earned nothing and
    smeared the markdown Evidence line, which renders evidence as flat `k=v` pairs — so a
    multi-line value landed inline with literal `\\n` escapes mid-sentence. As a flag it is
    the one thing a consumer cannot get elsewhere: ADV302 is never a proposal `code`, so
    `code == "ADV302"` matches nothing and this is the only way to filter for it."""
    context = DbtContext.from_project(_project())
    [out] = enrich_proposals([_index_proposal(Relation("main", "orders"))], context)
    assert out.evidence["dbt_index_config"] is True
    assert "\n" not in str(out.evidence["dbt_index_config"])


def test_adv302_does_not_mark_a_plain_index_as_unique():
    """`_is_unique_index` → `return True` survived: every block would gain `unique: true`,
    recommending a uniqueness *constraint* on a column with no evidence of being unique.
    `"unique: true"` was asserted once, only for a genuinely unique index; nothing asserted
    its absence."""
    context = DbtContext.from_project(_project())
    [out] = enrich_proposals([_index_proposal(Relation("main", "orders"))], context)
    assert "unique" not in out.ddl
    assert "unique" not in _config_mapping(out.ddl)["indexes"][0]


@pytest.mark.parametrize(
    "ddl",
    [
        # A future `CREATE TABLE`/`CREATE VIEW` rule must not be rewritten into an
        # `indexes:` block: narrowing the regex to `^CREATE\\b` survived mutation, and the
        # only negatives tested were `DROP INDEX` and `ddl=None`.
        'CREATE TABLE "main"."orders_new" AS SELECT * FROM "main"."orders";',
        'CREATE VIEW "main"."orders_v" AS SELECT * FROM "main"."orders";',
        # `INDEX` must be a whole word: dropping the `\\b` accepted this.
        'CREATE INDEXES ON "main"."orders" ("status");',
        # `.match()` → `.search()` survived: an index-creating phrase anywhere in a
        # statement that is not itself index-creating must not qualify it.
        'CREATE TABLE "main"."t" AS SELECT 1; -- next step: CREATE INDEX ON "main"."t" (a)',
    ],
)
def test_adv302_only_rewrites_a_statement_that_starts_by_creating_an_index(ddl):
    """`evidence` deliberately carries a valid `columns` tuple, so a broken prefix check
    cannot be caught by the separate "no columns to express" bail-out instead — the same
    reasoning as the DROP INDEX test above."""
    context = DbtContext.from_project(_project())
    proposal = Proposal(
        code="ADV999",
        title="something other than an index",
        rationale="r.",
        evidence={"schema": "main", "table": "orders", "columns": ("status",)},
        confidence=Confidence.MEDIUM,
        ddl=ddl,
    )
    [out] = enrich_proposals([proposal], context)
    assert out.ddl == ddl, "only index creation becomes an `indexes:` config block"
    assert "indexes" not in (out.ddl or "")
    assert out.evidence["dbt_model"] == "model.demo.orders", "still attributed, just not rewritten"


@pytest.mark.parametrize("guard", ["guard_column", "guard_predicate"])
def test_adv302_treats_either_guard_fact_alone_as_a_partial_index(guard):
    """`_is_partial_index` is a disjunction and every fixture supplied *both* facts, so
    dropping either alternative survived. A partial index rewritten into a config block that
    has no predicate field silently drops the WHERE clause and turns a correct proposal into
    a wrong one, so each alternative has to hold on its own."""
    context = DbtContext.from_project(_project())
    proposal = _index_proposal(Relation("main", "orders"), ("region",), code="ADV004")
    proposal = dataclasses.replace(
        proposal,
        evidence={**proposal.evidence, guard: "deleted" if guard == "guard_column" else "IS NULL"},
        ddl='CREATE INDEX ON "main"."orders" ("region") WHERE "deleted" IS NULL;',
    )
    [out] = enrich_proposals([proposal], context)
    assert out.ddl == proposal.ddl, "a partial index must keep its DDL, not lose its WHERE"
    assert "no predicate field" in out.rationale
    assert out.note is not None and "dbt WARNING" in out.note


@pytest.mark.parametrize(
    "columns",
    [
        None,  # the key is absent entirely
        (),  # present but empty
        ("status", 3),  # present but not all strings
        "status",  # a bare string is iterable, and `list("status")` would emit characters
    ],
)
def test_adv302_declines_and_discloses_when_there_is_no_plain_column_list(columns):
    """The whole `columns` validation was unpinned — replacing it with `if False:` survived.

    Unreachable from today's rules, all of which populate `columns`; reachable *by design*,
    because `_is_index_creating` matches on the DDL prefix precisely so a future
    index-creating rule is covered without being enumerated. It used to decline in complete
    silence: DDL left executable, rationale entirely unamended, only `evidence` quietly
    gaining the dbt keys — the one outcome this module exists to prevent.
    """
    context = DbtContext.from_project(_project())
    proposal = _index_proposal(Relation("main", "orders"))
    evidence = dict(proposal.evidence)
    if columns is None:
        del evidence["columns"]
    else:
        evidence["columns"] = columns
    proposal = dataclasses.replace(proposal, evidence=evidence)

    [out] = enrich_proposals([proposal], context)

    assert out.ddl == proposal.ddl, "no invented column list"
    assert "indexes" not in out.ddl
    assert "no plain column list" in out.rationale, "the decline must be disclosed"
    assert out.note is not None and "dbt WARNING" in out.note


def test_adv302_declines_a_non_btree_access_method_rather_than_calling_it_btree():
    """The block is rebuilt from `evidence["columns"]` and hardcodes `type: btree`, throwing
    the DDL away. Faithful for every rule today, all of which emit a plain btree over a
    column list — but a `USING gin` proposal rewritten to `type: btree` hands back a
    different index than the evidence justified."""
    context = DbtContext.from_project(_project())
    proposal = dataclasses.replace(
        _index_proposal(Relation("main", "orders"), ("payload",)),
        ddl='CREATE INDEX ON "main"."orders" USING gin ("payload");',
    )
    [out] = enrich_proposals([proposal], context)
    assert out.ddl == proposal.ddl
    assert "non-btree access method" in out.rationale
    assert out.note is not None and "dbt WARNING" in out.note


def test_adv302_still_rewrites_an_explicit_using_btree():
    """`USING btree` *is* btree, so it must not trip the non-btree guard. Two spaces
    deliberately: a lookahead without the trailing `\\S` lets `\\s+` backtrack one space in and
    refuse a plain btree index."""
    context = DbtContext.from_project(_project())
    proposal = dataclasses.replace(
        _index_proposal(Relation("main", "orders"), ("status",)),
        ddl='CREATE INDEX ON "main"."orders" USING  btree ("status");',
    )
    [out] = enrich_proposals([proposal], context)
    assert _config_mapping(out.ddl) == {"indexes": [{"columns": ["status"], "type": "btree"}]}


def test_enrich_proposals_preserves_input_order():
    """`return out[::-1]` survived: every test passed a one-element list. The caller re-sorts
    by the adapter's ranking key afterwards, and this function must not pre-empt that."""
    context = DbtContext.from_project(_project())
    proposals = [
        _index_proposal(Relation("main", "orders"), ("status",), code="ADV001"),
        _index_proposal(Relation("public", "unmanaged"), ("id",), code="ADV007"),
        _index_proposal(Relation("main", "stg_orders"), ("id",), code="ADV008"),
    ]
    out = enrich_proposals(proposals, context)
    assert [p.code for p in out] == ["ADV001", "ADV007", "ADV008"]


def test_enrich_proposals_does_not_mutate_the_evidence_it_was_given():
    """`dict(proposal.evidence)` → `proposal.evidence` survived: nothing pinned that the
    input proposals come back unchanged. `Proposal` is frozen but `evidence` is a plain dict,
    so the freeze does not cover this."""
    context = DbtContext.from_project(_project())
    original = _index_proposal(Relation("main", "orders"))
    before = dict(original.evidence)
    out = enrich_proposals([original], context)
    assert original.evidence == before, "the caller's proposal must be untouched"
    assert "dbt_model" not in original.evidence
    assert out[0].evidence["dbt_model"] == "model.demo.orders"


def test_adv302_view_branch_keeps_the_proposal_and_only_drops_the_ddl():
    """Documented for a while as "the proposal is dropped", which it is not: the finding
    ("this index cannot apply here") is worth reporting, so the proposal survives at LOW with
    its title and cost share and only its DDL goes."""
    context = DbtContext.from_project(_project())
    original = _index_proposal(Relation("main", "stg_orders"))
    [out] = enrich_proposals([original], context)
    assert out.ddl is None
    assert out.title == original.title
    assert out.evidence["cost_share"] == original.evidence["cost_share"]
    assert out.confidence is Confidence.LOW
    assert out.note is None, "no statement is emitted, so there is nothing to warn beside"


@pytest.mark.parametrize("absent", [None, ""])
def test_adv302_states_an_absent_materialization_once_and_without_a_python_literal(absent):
    """It used to say both "materialized as `None`" — a Python literal in an operator-facing
    string — and "materialization '(absent)'", two spellings of one fact; `materialized=""`
    rendered "materialized as ``", an empty code span, beside the same duplicate."""
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    raw["nodes"]["model.demo.orders"]["config"]["materialized"] = absent
    context = DbtContext.from_project(DbtProject.from_manifest(raw))
    original = _index_proposal(Relation("main", "orders"))
    [out] = enrich_proposals([original], context)

    assert out.ddl == original.ddl, "unknown is not known-safe: the DDL is not rewritten"
    assert "None" not in out.rationale
    assert "``" not in out.rationale
    assert "(absent)" not in out.rationale
    assert out.rationale.count("materializ") == 2, (
        f"one statement of the fact plus one reason, not two spellings: {out.rationale}"
    )
    assert "no materialization recorded in the manifest" in out.rationale
    assert out.note is not None and "dbt WARNING" in out.note


def test_adv303_discloses_that_dead_chains_unwind_one_model_per_run():
    """The transitive-deadness caveat landed as a docstring only, reaching no user: a fully
    dead chain surfaces one model per run and nothing explained why the parent was not
    flagged the first time."""
    context = DbtContext.from_project(_project())
    usage = _usage(Relation("main", "orders"), "status", ColumnRole.EQUALITY, cost_share=0.5)
    [first, *_] = propose_unused_models(_aggregation(usage), context, _workload())
    assert "not transitive" in first.rationale
    assert "leaf" in first.rationale


def test_describe_rewrites_is_silent_when_nothing_was_rewritten():
    """The terminal line must not appear on a run where ADV302 did not fire — including a
    run with a manifest that simply matched nothing."""
    assert describe_rewrites([]) is None
    assert describe_rewrites([_index_proposal(Relation("public", "unmanaged"))]) is None


def test_describe_rewrites_reports_both_rewritten_and_folded_proposals():
    """ADV302 is never a proposal `code`, and the terminal never prints `rationale`, so an
    enriched row is byte-identical to the same proposal from a dbt-free run. This line is the
    only signal a terminal-only user gets that enrichment fired."""
    context = DbtContext.from_project(_project())
    relation = Relation("main", "orders")
    out = enrich_proposals(
        [
            _index_proposal(relation, ("status",), code="ADV001"),
            _index_proposal(relation, ("customer_id",), code="ADV007"),
        ],
        context,
    )
    line = describe_rewrites(out)
    assert line is not None
    assert "ADV302" in line
    assert "2 index proposal(s)" in line
    assert "1 folded" in line
    assert "\n" not in line, "one stderr line"


def _non_index_proposal(relation, *, code="ADV101", note=None):
    """A proposal whose DDL is neither `CREATE INDEX` nor `DROP INDEX` — the shape every
    Redshift rule emits (`ALTER TABLE ... ALTER SORTKEY`, `VACUUM`, Advisor's own DDL), and
    the only shape that reaches `_classify`'s generic path."""
    return Proposal(
        code=code,
        title=f"Consider SORTKEY on {relation}(created_at)",
        rationale="hot range predicate.",
        evidence={"schema": relation.schema, "table": relation.table, "cost_share": 0.5},
        confidence=Confidence.MEDIUM,
        ddl=f'ALTER TABLE "{relation.schema}"."{relation.table}" ALTER SORTKEY ("created_at");',
        note=note,
    )


def test_describe_rewrites_reports_a_statement_that_cannot_be_expressed_as_dbt_config():
    """Counting only ADV302's config-block rewrite made this function return `None` for every
    Redshift run — nothing Redshift emits is a `CREATE INDEX`, so all of ADV101-105 take
    `_classify`'s generic path instead. Enrichment fired, the warning went into `rationale`
    and `note`, and the terminal row stayed byte-identical to a dbt-free run: exactly the
    failure mode this line exists to prevent, on the engine where the undone work is hours
    of full-table rewrite rather than seconds of index build.
    """
    context = DbtContext.from_project(_project())
    out = enrich_proposals([_non_index_proposal(Relation("main", "orders"))], context)
    line = describe_rewrites(out)
    assert line is not None
    assert "1 proposal(s)" in line
    assert "cannot be expressed as dbt config" in line
    assert "next `dbt run` may undo it" in line
    assert "\n" not in line, "one stderr line"


def test_describe_rewrites_is_still_silent_for_an_unmanaged_non_index_proposal():
    """Control for the test above: the new count must depend on `DbtContext.model_for`
    actually matching, not merely on a proposal whose DDL is not an index."""
    context = DbtContext.from_project(_project())
    out = enrich_proposals([_non_index_proposal(Relation("public", "unmanaged"))], context)
    assert describe_rewrites(out) is None


def test_describe_rewrites_reports_both_kinds_in_one_line():
    """A mixed run (a Postgres index proposal and a non-index one for the same dbt model)
    must disclose both, since they call for different actions: one DDL is config to paste,
    the other is runnable SQL that may not survive the next rebuild."""
    context = DbtContext.from_project(_project())
    relation = Relation("main", "orders")
    out = enrich_proposals([_index_proposal(relation), _non_index_proposal(relation)], context)
    line = describe_rewrites(out)
    assert line is not None
    assert "ADV302 expressed 1 index proposal(s)" in line
    assert "1 proposal(s) target a dbt-managed relation" in line


def _drop_proposal(relation, *, code="ADV002", index="idx_cold"):
    """An index-drop proposal — the shape ADV002 (unused) and ADV003 (redundant) emit, and the
    only shape that reaches `_classify`'s `DROP INDEX` branch."""
    return Proposal(
        code=code,
        title=f"Drop unused index {index} on {relation}",
        rationale="no scans.",
        evidence={
            "schema": relation.schema,
            "table": relation.table,
            "index": index,
            "columns": ("status",),
        },
        confidence=Confidence.MEDIUM,
        ddl=f'DROP INDEX "{relation.schema}"."{index}";',
    )


def test_describe_rewrites_reports_an_index_drop_that_a_dbt_run_may_recreate():
    """The third enrichment outcome, and the one reachable on the adapter this module was
    written for. A Postgres run whose only proposals for dbt-managed relations are ADV002 /
    ADV003 drops enriches every one of them — rationale *and* note — and the terminal said
    nothing at all, because only the config-block rewrite and the generic non-index warning
    were counted. The operator then applies a drop that the next `dbt run` puts straight back
    from the model's `indexes:` config, and this tool proposes the same drop again next run.
    """
    context = DbtContext.from_project(_project())
    relation = Relation("main", "orders")
    out = enrich_proposals(
        [
            _drop_proposal(relation, code="ADV002", index="idx_cold"),
            _drop_proposal(relation, code="ADV003", index="idx_narrow"),
        ],
        context,
    )
    assert [p.ddl for p in out] == [
        'DROP INDEX "main"."idx_cold";',
        'DROP INDEX "main"."idx_narrow";',
    ], "the drops themselves are unchanged; only the disclosure is new"
    line = describe_rewrites(out)
    assert line is not None
    assert "2 index drop(s)" in line
    assert "`indexes:` config" in line
    assert "does not stick" in line
    assert "\n" not in line, "one stderr line"


def test_describe_rewrites_is_still_silent_for_an_unmanaged_index_drop():
    """Control for the test above: the new count must depend on `DbtContext.model_for`
    actually matching, not merely on a proposal whose DDL is a `DROP INDEX`."""
    context = DbtContext.from_project(_project())
    out = enrich_proposals([_drop_proposal(Relation("public", "unmanaged"))], context)
    assert describe_rewrites(out) is None


def test_describe_rewrites_counts_the_drop_branch_separately_from_the_other_two():
    """The count has to discriminate *this* branch, not merely be non-zero when enrichment
    fired. A flag set in `_classify`'s generic non-index path (or read off the same key as the
    config rewrite) would satisfy a bare "the line mentions drops" assertion while reporting
    the wrong number for a mixed run, and would report drops on a run that had none.

    All three outcomes call for different actions, which is why they are three clauses: paste
    a config block, expect a runnable statement not to last, or delete a config entry as well.
    """
    context = DbtContext.from_project(_project())
    relation = Relation("main", "orders")

    only_drops = enrich_proposals([_drop_proposal(relation)], context)
    line = describe_rewrites(only_drops)
    assert line is not None
    assert "1 index drop(s)" in line
    assert "ADV302" not in line, "a drop is not expressed as an `indexes` config block"
    assert "cannot be expressed as dbt config" not in line, "that is the generic branch"

    # The other two branches must not be counted as drops.
    for other in (
        enrich_proposals([_index_proposal(relation)], context),
        enrich_proposals([_non_index_proposal(relation)], context),
    ):
        other_line = describe_rewrites(other)
        assert other_line is not None
        assert "index drop(s)" not in other_line, other_line

    all_three = enrich_proposals(
        [_index_proposal(relation), _non_index_proposal(relation), _drop_proposal(relation)],
        context,
    )
    mixed = describe_rewrites(all_three)
    assert mixed is not None
    assert "ADV302 expressed 1 index proposal(s)" in mixed
    assert "1 proposal(s) target a dbt-managed relation" in mixed
    assert "1 index drop(s)" in mixed


def test_prepend_note_keeps_the_existing_note_first_and_emits_the_dbt_note_once():
    """Order and non-duplication, neither of which was pinned: losing the existing note was
    caught by two tests, but swapping the concatenation order and emitting the dbt warning
    twice both left the whole suite green. The existing note is the proposal's own statement
    about its own DDL — ADV105's "this DDL came from Redshift, not sqlquality" — and the dbt
    warning is a caveat on it, so it belongs after, once.
    """
    context = DbtContext.from_project(_project())
    existing = "Source: Amazon Redshift Advisor, not sqlquality's own analysis."
    (out,) = enrich_proposals(
        [_non_index_proposal(Relation("main", "orders"), note=existing)], context
    )
    note = out.note or ""
    assert note.startswith(existing)
    assert "dbt WARNING" in note
    assert note.index(existing) < note.index("dbt WARNING")
    assert note.count("dbt WARNING") == 1


def test_prepend_note_is_idempotent():
    """Enriching an already-enriched proposal must not stack a second copy of the same dbt
    warning onto its note. Not reachable from `cli.py`, which enriches once — pinned because
    the fix's own name and docstring claim it cannot duplicate a note, and because a note
    that grows on every pass is the kind of thing only a test notices."""
    context = DbtContext.from_project(_project())
    (once,) = enrich_proposals([_non_index_proposal(Relation("main", "orders"))], context)
    (twice,) = enrich_proposals([once], context)
    assert twice.note == once.note
    assert (twice.note or "").count("dbt WARNING") == 1


def test_resolve_manifest_path_prefers_an_explicit_manifest_over_a_project_dir():
    """One function, because this precedence used to exist twice — in `load_dbt_context` and
    again in the CLI's payload builder — and swapping it in either copy alone left the whole
    suite green, so the payload could name a manifest that was never read."""
    project_dir = Path("/tmp/proj")
    explicit = Path("/tmp/elsewhere/manifest.json")
    assert resolve_manifest_path(project_dir, explicit) == explicit
    assert resolve_manifest_path(project_dir, None) == project_dir / "target" / "manifest.json"
    assert resolve_manifest_path(None, explicit) == explicit
    assert resolve_manifest_path(None, None) is None


def test_load_warns_when_the_manifest_is_not_a_v12_schema(tmp_path):
    """`check` warns on this and the dbt `advise` path checked nothing, so it silently
    accepted a v10/v11 manifest whose node shapes it reads as if they were v12."""
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    raw["metadata"]["dbt_schema_version"] = "https://schemas.getdbt.com/dbt/manifest/v10.json"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    context, disclosure = load_dbt_context(None, path)
    assert context is not None, "a wrong schema version degrades to a warning, not a refusal"
    assert "dbt_schema_version" in disclosure
    assert "v10" in disclosure


def test_load_warns_that_the_indexes_config_is_postgres_specific_on_another_adapter(tmp_path):
    """ADV302 rewrites index DDL into dbt's `indexes` model config, which only the postgres
    and redshift adapters implement. Against a Snowflake or BigQuery project the rewrite is
    advice for a config key that does not exist, presented as a correctness fix — and no
    document said so."""
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    raw["metadata"]["adapter_type"] = "snowflake"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    context, disclosure = load_dbt_context(None, path)
    assert context is not None
    assert "snowflake" in disclosure
    assert "ADV302" in disclosure


@pytest.mark.parametrize("adapter_type", ["postgres", "redshift"])
def test_load_is_quiet_for_an_adapter_that_has_the_indexes_config(adapter_type):
    """The warning must discriminate: firing for postgres too would make it noise, and a
    warning nobody can act on is worse than none."""
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    raw["metadata"]["adapter_type"] = adapter_type
    project = DbtProject.from_manifest(raw)
    assert _manifest_warnings(project) == []


@pytest.mark.parametrize("metadata", [{}, {"adapter_type": 12, "dbt_schema_version": None}])
def test_load_survives_metadata_of_the_wrong_shape(tmp_path, metadata):
    """`metadata` is a section some other tool wrote, and this runs after the whole catalog
    analysis: a non-string version raising from an `in` test would discard real work over an
    optional input."""
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    raw["metadata"] = metadata
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    context, disclosure = load_dbt_context(None, path)
    assert context is not None
    assert disclosure is not None
    assert "3 model(s)" in disclosure


def test_the_merged_block_and_its_cross_reference_stay_fully_commented():
    """Invariant 1, over the two texts this rewrite newly interpolates raw values into.

    The deferred block interpolates the *owner proposal's title* and the model's `unique_id`,
    both of which can carry a raw newline — a title is built from live catalog identifiers,
    and dbt's `relation_name`/`unique_id` permit one too. Neither may produce an output line
    without a leading `--`, or the DDL script shows something that reads like a statement.
    """
    hostile = 'evil";\nDROP TABLE users; --'
    # A newline in the *unique_id* too: a manifest is a file some other tool wrote, this key
    # is interpolated raw into both texts, and unlike a column list it does not pass through
    # `repr()` escaping on the way. `_comment_block`'s own re-split is the only thing
    # standing between it and an uncommented output line.
    uid = f"model.demo.{hostile}"
    manifest = {
        "nodes": {
            uid: {
                "resource_type": "model",
                "config": {"materialized": "table"},
                "relation_name": '"dev"."main"."orders"',
            }
        }
    }
    context = DbtContext.from_project(DbtProject.from_manifest(manifest))
    relation = Relation("main", "orders")
    first = _index_proposal(relation, (hostile,), code="ADV001")
    second = _index_proposal(relation, ("customer_id",), code="ADV007")

    out = enrich_proposals([first, second], context)

    assert out[0].evidence["dbt_index_config"] is True, "the owner block must be exercised"
    assert "dbt_index_config_reported_with" in out[1].evidence, "and the deferred one too"
    for proposal in out:
        lines = [ln for ln in (proposal.ddl or "").splitlines() if ln.strip()]
        assert lines, proposal
        assert all(ln.startswith("--") for ln in lines), lines


def test_a_drop_index_on_a_dbt_relation_warns_that_the_config_entry_must_go_too():
    """The mirror image of the bug ADV302 exists to fix, and it is reachable through the
    ordinary rules — ADV002 and ADV003 read the catalog, not the manifest.

    "Dropping an index dbt never created is ordinary" — the original justification for
    exempting drops entirely — is false in exactly the case this module cares about. If the
    index is declared in the model's `indexes:` config, the next `dbt run` recreates it: the
    operator drops it, dbt puts it back, and this tool proposes the same drop again next run.
    That is the same silently-reverting advice ADV302 was built to eliminate. The proposal is
    *not* suppressed — dropping a genuinely unused index is still right — so the operator has
    to be given both halves of the instruction, in the rationale and beside the statement.
    """
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

    assert out.ddl == drop.ddl, "a genuinely unused index is still worth dropping"
    assert out.confidence is drop.confidence
    # The DDL script never carries a rationale, so the warning has to be a note as well.
    assert out.note is not None
    assert "dbt WARNING" in out.note
    assert "model.demo.orders" in out.note
    assert "indexes" in out.note and "recreates it" in out.note
    assert "remove the config entry" in out.note
    assert "`indexes:` config" in out.rationale
    assert "will not stick" in out.rationale


def test_a_drop_index_on_a_relation_dbt_does_not_manage_gets_no_note():
    """Discriminating: a note on every drop would satisfy the test above and mean nothing."""
    context = DbtContext.from_project(_project())
    drop = Proposal(
        code="ADV002",
        title="Drop unused index idx_cold on public.orders",
        rationale="no scans.",
        evidence={"schema": "public", "table": "orders", "index": "idx_cold"},
        confidence=Confidence.MEDIUM,
        ddl='DROP INDEX "public"."idx_cold";',
    )
    assert enrich_proposals([drop], context) == [drop]


def test_an_advisory_proposal_for_a_dbt_relation_gets_no_note():
    """`note` renders only beside a statement in the DDL script. A proposal with no DDL
    contributes no line to that file, so a note would appear nowhere at all — its `rationale`
    already reaches every surface that shows it."""
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
    assert out.note is None
    assert out.evidence["dbt_model"] == "model.demo.orders"


def test_any_other_kept_statement_for_a_dbt_relation_also_warns():
    """No rule emits one today, and `_is_index_creating` matches by DDL prefix specifically so
    a future one stays covered. The constraint is about the *file*: no executable statement for
    a dbt-managed relation without an adjacent warning, whatever the statement is."""
    context = DbtContext.from_project(_project())
    proposal = Proposal(
        code="ADV999",
        title="Cluster main.orders",
        rationale="r.",
        evidence={"schema": "main", "table": "orders", "columns": ("status",)},
        confidence=Confidence.MEDIUM,
        ddl='CLUSTER "main"."orders" USING "idx_status";',
    )
    [out] = enrich_proposals([proposal], context)
    assert out.ddl == proposal.ddl
    assert out.note is not None and "dbt WARNING" in out.note
    assert "which dbt rebuilds on its own schedule" in out.rationale


def test_load_warns_when_the_manifest_targets_a_different_warehouse(tmp_path):
    """A foreign `adapter_type` is not merely "ADV302 might emit a config key that adapter
    lacks" — it means dbt is not building the Postgres relations `advise` just introspected at
    all, so every `(schema, table)` match is a name coincidence and all three dbt rules are
    wrong, ADV302's rebuild premise included."""
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    raw["metadata"]["adapter_type"] = "snowflake"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    _context, disclosure = load_dbt_context(None, path)
    assert "snowflake" in disclosure
    assert "do not build the relations being introspected" in disclosure


def test_load_warns_when_the_manifest_records_no_adapter_type(tmp_path):
    """Deliberate: `dbt compile` always writes an `adapter_type`, so absence means a
    hand-written or truncated manifest. Warning on "different" while staying silent on
    "unknown" would make silence mean two things — enrichment consistent with the connection,
    or unchecked. `check` draws the same distinction rather than assuming."""
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    del raw["metadata"]["adapter_type"]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    context, disclosure = load_dbt_context(None, path)
    assert context is not None, "an unverifiable pairing still enriches, with a warning"
    assert "records no adapter_type" in disclosure
    # And it must not claim a *schema version* problem the manifest does not have.
    assert "dbt_schema_version" not in disclosure
