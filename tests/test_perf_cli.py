import json

from typer.testing import CliRunner

from sqlquality.cli import app

runner = CliRunner()


def test_perf_static_json(tmp_path):
    f = tmp_path / "m.sql"
    f.write_text("select * from a, b")
    result = runner.invoke(app, ["perf", str(f), "--json"])
    assert result.exit_code == 0
    codes = {x["code"] for x in json.loads(result.stdout)["findings"]}
    assert "SQ001" in codes and "SQ002" in codes


def test_perf_with_explain_json(tmp_path):
    f = tmp_path / "m.sql"
    f.write_text("select id from orders")
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps([{"Plan": {"Node Type": "Seq Scan", "Relation Name": "orders", "Plans": []}}]))
    result = runner.invoke(app, ["perf", str(f), "--explain", str(plan), "--json"])
    assert result.exit_code == 0
    codes = {x["code"] for x in json.loads(result.stdout)["findings"]}
    assert "PG001" in codes


def test_perf_parse_error_exit_1(tmp_path):
    f = tmp_path / "bad.sql"
    f.write_text("select from where")
    result = runner.invoke(app, ["perf", str(f)])
    assert result.exit_code == 1


def test_perf_unknown_dialect_exit_2(tmp_path):
    f = tmp_path / "m.sql"
    f.write_text("select 1")
    result = runner.invoke(app, ["perf", str(f), "--dialect", "oracle"])
    assert result.exit_code == 2


def test_perf_human_output(tmp_path):
    f = tmp_path / "m.sql"
    f.write_text("select * from t")
    result = runner.invoke(app, ["perf", str(f)])
    assert result.exit_code == 0
    assert "SQ001" in result.stdout


def test_perf_missing_explain_json_exit_2(tmp_path):
    f = tmp_path / "m.sql"
    f.write_text("select 1")
    result = runner.invoke(
        app, ["perf", str(f), "--explain", str(tmp_path / "nope.json")]
    )
    assert result.exit_code == 2


def test_perf_malformed_explain_json_exit_2(tmp_path):
    f = tmp_path / "m.sql"
    f.write_text("select 1")
    plan = tmp_path / "plan.json"
    plan.write_text("{ not json")
    result = runner.invoke(app, ["perf", str(f), "--explain", str(plan)])
    assert result.exit_code == 2
