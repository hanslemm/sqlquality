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
