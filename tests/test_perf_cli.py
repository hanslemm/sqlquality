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
    plan.write_text(
        json.dumps([{"Plan": {"Node Type": "Seq Scan", "Relation Name": "orders", "Plans": []}}])
    )
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


def test_perf_truly_unknown_dialect_has_suggestions(tmp_path):
    f = tmp_path / "m.sql"
    f.write_text("select 1")
    result = runner.invoke(app, ["perf", str(f), "--dialect", "oracle9000"])
    assert result.exit_code == 2
    assert "oracle9000" in result.stderr
    assert "postgres" in result.stderr  # suggestions listed
    assert "Traceback" not in result.stderr


def test_perf_jinja_model_real_findings(tmp_path):
    f = tmp_path / "model.sql"
    f.write_text(
        "{{ config(materialized='table') }}\nselect * from {{ ref('stg') }}, other where x = 1"
    )
    result = runner.invoke(app, ["perf", str(f), "--json"])
    assert result.exit_code == 0  # WARNING findings only -> exit 0
    assert "Jinja placeholders" in result.stderr
    codes = {x["code"] for x in json.loads(result.stdout)["findings"]}
    assert "SQ000" not in codes  # retried against stripped SQL, not left unparseable
    assert "SQ001" in codes  # SELECT * on stripped SQL is a real finding


def test_perf_non_utf8_file_exit_2(tmp_path):
    f = tmp_path / "latin1.sql"
    f.write_bytes(b"select caf\xe9 from t")
    result = runner.invoke(app, ["perf", str(f)])
    assert result.exit_code == 2
    assert "UTF-8" in result.stderr


def test_perf_human_output(tmp_path):
    f = tmp_path / "m.sql"
    f.write_text("select * from t")
    result = runner.invoke(app, ["perf", str(f)])
    assert result.exit_code == 0
    assert "SQ001" in result.stdout


def test_perf_missing_explain_json_exit_2(tmp_path):
    f = tmp_path / "m.sql"
    f.write_text("select 1")
    result = runner.invoke(app, ["perf", str(f), "--explain", str(tmp_path / "nope.json")])
    assert result.exit_code == 2


def test_perf_malformed_explain_json_exit_2(tmp_path):
    f = tmp_path / "m.sql"
    f.write_text("select 1")
    plan = tmp_path / "plan.json"
    plan.write_text("{ not json")
    result = runner.invoke(app, ["perf", str(f), "--explain", str(plan)])
    assert result.exit_code == 2


def test_perf_redshift_with_explain(tmp_path):
    f = tmp_path / "m.sql"
    f.write_text("select * from a join b on a.id = b.id where a.d > '2024-01-01'")
    plan = tmp_path / "plan.txt"
    plan.write_text("XN Hash Join DS_BCAST_INNER  (cost=0.00..1.00)\n")
    result = runner.invoke(
        app,
        ["perf", str(f), "--dialect", "redshift", "--explain", str(plan), "--json"],
    )
    assert result.exit_code == 0
    codes = {x["code"] for x in json.loads(result.stdout)["findings"]}
    assert "RS010" in codes  # from the EXPLAIN plan text
    assert "RS001" in codes  # DISTKEY suggestion (join key)
    assert "RS002" in codes  # SORTKEY suggestion (filter column)
