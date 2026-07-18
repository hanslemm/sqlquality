import json
from pathlib import Path
from unittest import mock

from typer.testing import CliRunner

from sqlquality.checkcmd import CheckResult, run_check
from sqlquality.cli import app
from sqlquality.dbtproject import DbtProject
from sqlquality.models import DagFacts

runner = CliRunner()
FIXTURE = Path(__file__).parent / "fixtures" / "manifest_v12.json"


def test_run_check_scores_changed_models():
    project = DbtProject.from_path(FIXTURE)
    results, skipped = run_check(project, ["model.demo.orders"], "postgres")
    assert skipped == []
    assert len(results) == 1
    r = results[0]
    assert isinstance(r, CheckResult)
    assert r.unique_id == "model.demo.orders"
    assert r.dag == DagFacts(fan_in=1, fan_out=1, lineage_depth=2)
    assert r.composite > 0


def test_run_check_skips_uncompiled():
    project = DbtProject.from_path(FIXTURE)
    results, skipped = run_check(project, ["seed.demo.raw_orders"], "postgres")
    assert results == []
    assert skipped and skipped[0][0] == "seed.demo.raw_orders"


def test_check_command_json(tmp_path):
    # a project dir whose target/manifest.json is our fixture
    proj = tmp_path / "proj"
    (proj / "target").mkdir(parents=True)
    (proj / "target" / "manifest.json").write_text(FIXTURE.read_text())
    base = tmp_path / "baseline"
    base.mkdir()

    stdout = '{"resource_type": "model", "unique_id": "model.demo.orders"}\n'
    with mock.patch("sqlquality.cli.run_state_modified", return_value=stdout):
        result = runner.invoke(
            app,
            ["check", "--project-dir", str(proj), "--state", str(base), "--json"],
        )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["changed"] == ["model.demo.orders"]
    assert payload["neighbors"] == [
        "model.demo.customer_orders",
        "model.demo.stg_orders",
    ]
    assert payload["results"][0]["unique_id"] == "model.demo.orders"


def test_check_command_dbt_failure_exit_2(tmp_path):
    from sqlquality.changeset import ChangeSetError

    proj = tmp_path / "proj"
    (proj / "target").mkdir(parents=True)
    (proj / "target" / "manifest.json").write_text(FIXTURE.read_text())
    base = tmp_path / "baseline"
    base.mkdir()

    with mock.patch(
        "sqlquality.cli.run_state_modified",
        side_effect=ChangeSetError("dbt not found"),
    ):
        result = runner.invoke(
            app, ["check", "--project-dir", str(proj), "--state", str(base)]
        )
    assert result.exit_code == 2
