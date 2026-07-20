from pathlib import Path

import pytest

from sqlquality.dbtproject import DbtProject, DbtProjectError
from sqlquality.models import DagFacts

FIXTURE = Path(__file__).parent / "fixtures" / "manifest_v12.json"


@pytest.fixture
def project() -> DbtProject:
    return DbtProject.from_path(FIXTURE)


def test_model_ids_excludes_seeds_and_tests(project):
    assert project.model_ids() == [
        "model.demo.customer_orders",
        "model.demo.orders",
        "model.demo.stg_orders",
    ]


def test_adapter_and_schema(project):
    assert project.adapter_type() == "postgres"
    assert "v12" in project.schema_version()


def test_node_fields(project):
    node = project.node("model.demo.orders")
    assert node.resource_type == "model"
    assert node.materialized == "table"
    assert node.depends_on == ["model.demo.stg_orders"]
    assert node.config["indexes"] == [{"columns": ["customer_id"], "unique": False}]


def test_model_neighbors_are_model_filtered(project):
    # stg_orders' parent is a seed -> excluded; children are two models
    assert project.model_parents("model.demo.stg_orders") == []
    assert project.model_children("model.demo.stg_orders") == [
        "model.demo.customer_orders",
        "model.demo.orders",
    ]
    # orders' child test.* is excluded, leaving one model child
    assert project.model_children("model.demo.orders") == ["model.demo.customer_orders"]
    assert project.model_parents("model.demo.customer_orders") == [
        "model.demo.orders",
        "model.demo.stg_orders",
    ]


def test_compiled_sql(project):
    sql = project.compiled_sql("model.demo.orders")
    assert "group by customer_id" in sql


def test_compiled_sql_missing_raises(project):
    with pytest.raises(DbtProjectError):
        project.compiled_sql("seed.demo.raw_orders")


def test_dag_facts(project):
    assert project.dag_facts("model.demo.stg_orders") == DagFacts(
        fan_in=0, fan_out=2, lineage_depth=1
    )
    assert project.dag_facts("model.demo.orders") == DagFacts(fan_in=1, fan_out=1, lineage_depth=2)
    assert project.dag_facts("model.demo.customer_orders") == DagFacts(
        fan_in=2, fan_out=0, lineage_depth=3
    )


def test_unknown_node_raises(project):
    with pytest.raises(DbtProjectError):
        project.node("model.demo.nope")


def _manifest_from_parent_map(parent_map: dict[str, list[str]]) -> dict:
    nodes = {
        uid: {"resource_type": "model", "name": uid, "depends_on": {"nodes": parents}}
        for uid, parents in parent_map.items()
    }
    return {"nodes": nodes, "parent_map": parent_map, "child_map": {}}


def test_lineage_depth_survives_cycle():
    # a <-> b: a recursive walk loops forever; iteratively the depth stays finite.
    project = DbtProject.from_manifest(
        _manifest_from_parent_map({"model.a": ["model.b"], "model.b": ["model.a"]})
    )
    for uid in ("model.a", "model.b"):
        depth = project.dag_facts(uid).lineage_depth
        assert 1 <= depth <= 2  # finite, not a hang / RecursionError


def test_lineage_depth_handles_deep_chain():
    # 5000-deep chain would blow the recursion limit; iterative version handles it.
    chain = {f"model.n{i}": [f"model.n{i + 1}"] for i in range(4999)}
    chain["model.n4999"] = []
    project = DbtProject.from_manifest(_manifest_from_parent_map(chain))
    assert project.dag_facts("model.n0").lineage_depth == 5000
