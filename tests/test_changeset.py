from pathlib import Path
from unittest import mock

import pytest

from sqlquality.changeset import (
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
    assert cmd == [
        "dbt",
        "ls",
        "--no-write-json",
        "--select",
        "state:modified",
        "--state",
        str((tmp_path / "base").resolve()),
        "--resource-type",
        "model",
        "--output",
        "json",
    ]
    assert kwargs["cwd"] == tmp_path / "proj"
    assert kwargs["timeout"] == 600


def test_run_state_modified_resolves_relative_state_to_absolute(tmp_path, monkeypatch):
    # The subprocess runs with cwd=project_dir, so a relative --state must be
    # resolved against the real cwd *before* being passed to dbt.
    monkeypatch.chdir(tmp_path)
    fake = mock.Mock(returncode=0, stdout="", stderr="")
    with mock.patch("sqlquality.changeset.subprocess.run", return_value=fake) as run:
        run_state_modified(tmp_path / "proj", "base", dbt="dbt")
    cmd = run.call_args[0][0]
    state_arg = cmd[cmd.index("--state") + 1]
    assert Path(state_arg).is_absolute()
    assert state_arg == str(Path("base").resolve())


def test_run_state_modified_raises_on_failure(tmp_path):
    fake = mock.Mock(returncode=1, stdout="", stderr="boom")
    with mock.patch("sqlquality.changeset.subprocess.run", return_value=fake):
        with pytest.raises(ChangeSetError):
            run_state_modified(tmp_path / "proj", tmp_path / "base")


def test_run_state_modified_missing_dbt_raises_friendly(tmp_path):
    with mock.patch("sqlquality.changeset.subprocess.run", side_effect=FileNotFoundError()):
        with pytest.raises(ChangeSetError, match="not found on PATH"):
            run_state_modified(tmp_path / "proj", tmp_path / "base", dbt="dbt")


def test_run_state_modified_empty_stderr_includes_stdout_tail(tmp_path):
    fake = mock.Mock(
        returncode=1,
        stdout="line1\nCompilation Error in model foo\n",
        stderr="",
    )
    with mock.patch("sqlquality.changeset.subprocess.run", return_value=fake):
        with pytest.raises(ChangeSetError, match="Compilation Error in model foo"):
            run_state_modified(tmp_path / "proj", tmp_path / "base")


def test_run_state_modified_timeout_raises(tmp_path):
    import subprocess

    with mock.patch(
        "sqlquality.changeset.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="dbt", timeout=600),
    ):
        with pytest.raises(ChangeSetError, match="timed out"):
            run_state_modified(tmp_path / "proj", tmp_path / "base")
