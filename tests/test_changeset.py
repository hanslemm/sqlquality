from pathlib import Path
from unittest import mock

import pytest

from sqlquality.changeset import (
    ChangeSet,
    ChangeSetError,
    compute_changeset,
    parse_state_modified,
    run_state_modified,
)
from sqlquality.dbtproject import DbtProject

FIXTURE = Path(__file__).parent / "fixtures" / "manifest_v12.json"


@pytest.fixture
def project() -> DbtProject:
    return DbtProject.from_path(FIXTURE)


def test_parse_state_modified_extracts_model_ids():
    stdout = (
        '{"resource_type": "model", "unique_id": "model.demo.orders", "name": "orders"}\n'
        '{"resource_type": "test", "unique_id": "test.demo.x.abc", "name": "x"}\n'
        "12:00:00  Some log line that is not JSON\n"
        "\n"
    )
    assert parse_state_modified(stdout) == ["model.demo.orders"]


def test_compute_changeset_adds_model_neighbors(project):
    stdout = '{"resource_type": "model", "unique_id": "model.demo.orders"}\n'
    cs = compute_changeset(project, stdout)
    assert cs.changed == ["model.demo.orders"]
    # orders' 1-hop model neighbors: parent stg_orders + child customer_orders
    assert cs.neighbors == ["model.demo.customer_orders", "model.demo.stg_orders"]
    assert cs.analysis_set == [
        "model.demo.customer_orders",
        "model.demo.orders",
        "model.demo.stg_orders",
    ]


def test_compute_changeset_empty(project):
    cs = compute_changeset(project, "")
    assert cs.changed == []
    assert cs.neighbors == []
    assert cs.analysis_set == []


def test_run_state_modified_builds_command_and_returns_stdout(tmp_path):
    fake = mock.Mock(returncode=0, stdout='{"unique_id": "model.demo.orders"}\n', stderr="")
    with mock.patch("sqlquality.changeset.subprocess.run", return_value=fake) as run:
        out = run_state_modified(tmp_path / "proj", tmp_path / "base", dbt="dbt")
    assert "model.demo.orders" in out
    args, kwargs = run.call_args
    cmd = args[0]
    assert cmd[:2] == ["dbt", "ls"]
    assert "state:modified" in cmd
    assert "--output" in cmd and "json" in cmd
    assert "--resource-type" in cmd and "model" in cmd
    assert kwargs["cwd"] == tmp_path / "proj"


def test_run_state_modified_raises_on_failure(tmp_path):
    fake = mock.Mock(returncode=1, stdout="", stderr="boom")
    with mock.patch("sqlquality.changeset.subprocess.run", return_value=fake):
        with pytest.raises(ChangeSetError):
            run_state_modified(tmp_path / "proj", tmp_path / "base")
