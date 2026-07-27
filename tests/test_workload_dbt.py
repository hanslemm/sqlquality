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
