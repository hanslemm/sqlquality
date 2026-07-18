import copy
import json
from pathlib import Path
from unittest import mock

from typer.testing import CliRunner

from sqlquality.cli import app

runner = CliRunner()
FIXTURE = Path(__file__).parent / "fixtures" / "manifest_v12.json"


def _project_with_baseline(tmp_path):
    """A project dir (candidate) + a state dir (baseline with simpler orders)."""
    proj = tmp_path / "proj"
    (proj / "target").mkdir(parents=True)
    (proj / "target" / "manifest.json").write_text(FIXTURE.read_text())

    baseline = json.loads(FIXTURE.read_text())
    baseline = copy.deepcopy(baseline)
    baseline["nodes"]["model.demo.orders"]["compiled_code"] = (
        'select customer_id from "dev"."main"."stg_orders" group by customer_id'
    )
    state = tmp_path / "state"
    state.mkdir()
    (state / "manifest.json").write_text(json.dumps(baseline))
    return proj, state


def _mock_changed():
    stdout = '{"resource_type": "model", "unique_id": "model.demo.orders"}\n'
    return mock.patch("sqlquality.cli.run_state_modified", return_value=stdout)


def test_check_json_reports_delta(tmp_path):
    proj, state = _project_with_baseline(tmp_path)
    with _mock_changed():
        result = runner.invoke(
            app, ["check", "--project-dir", str(proj), "--state", str(state), "--json"]
        )
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["passed"] is True
    assert payload["models"][0]["unique_id"] == "model.demo.orders"
    assert payload["models"][0]["delta"] == 0.4


def test_check_fail_mode_exit_1(tmp_path):
    proj, state = _project_with_baseline(tmp_path)
    cfg = proj / "sqlquality.yml"
    cfg.write_text("gate:\n  mode: fail\n  max_complexity_increase: 0.1\n")
    with _mock_changed():
        result = runner.invoke(app, ["check", "--project-dir", str(proj), "--state", str(state)])
    # delta 0.4 > 0.1 threshold, fail mode -> exit 1
    assert result.exit_code == 1


def test_check_html_written(tmp_path):
    proj, state = _project_with_baseline(tmp_path)
    out = tmp_path / "report.html"
    with _mock_changed():
        result = runner.invoke(
            app,
            ["check", "--project-dir", str(proj), "--state", str(state), "--html", str(out)],
        )
    assert result.exit_code == 0
    assert out.exists()
    assert "<!doctype html>" in out.read_text()


def test_check_dbt_failure_exit_2(tmp_path):
    from sqlquality.changeset import ChangeSetError

    proj, state = _project_with_baseline(tmp_path)
    with mock.patch(
        "sqlquality.cli.run_state_modified",
        side_effect=ChangeSetError("dbt not found"),
    ):
        result = runner.invoke(app, ["check", "--project-dir", str(proj), "--state", str(state)])
    assert result.exit_code == 2


def test_check_corrupt_baseline_exit_2(tmp_path):
    proj, state = _project_with_baseline(tmp_path)
    (state / "manifest.json").write_text("{ not valid json")
    with _mock_changed():
        result = runner.invoke(app, ["check", "--project-dir", str(proj), "--state", str(state)])
    assert result.exit_code == 2
