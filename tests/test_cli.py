import json

from typer.testing import CliRunner

from sqlquality import __version__
from sqlquality.cli import app

runner = CliRunner()


def test_version_flag_prints_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_complexity_json_output(tmp_path):
    sql_file = tmp_path / "m.sql"
    sql_file.write_text(
        "with recent as (select * from orders where created_at > current_date - 7) "
        "select u.id, count(*) as n, "
        "row_number() over (partition by u.id order by max(o.created_at)) "
        "from users u join recent o on o.user_id = u.id group by u.id"
    )
    result = runner.invoke(app, ["complexity", str(sql_file), "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["composite"] == 22.6
    assert payload["metrics"]["join_count"] == 1
    assert payload["dialect"] == "postgres"


def test_complexity_human_output(tmp_path):
    sql_file = tmp_path / "m.sql"
    sql_file.write_text("select id, name from users where active")
    result = runner.invoke(app, ["complexity", str(sql_file)])
    assert result.exit_code == 0
    assert "5.4" in result.stdout


def test_complexity_parse_error_exit_2(tmp_path):
    sql_file = tmp_path / "bad.sql"
    sql_file.write_text("select from where")
    result = runner.invoke(app, ["complexity", str(sql_file)])
    assert result.exit_code == 2


def test_complexity_unknown_dialect_exit_2(tmp_path):
    sql_file = tmp_path / "m.sql"
    sql_file.write_text("select 1 from t")
    result = runner.invoke(app, ["complexity", str(sql_file), "--dialect", "oracle9000"])
    assert result.exit_code == 2
    assert "oracle9000" in result.stderr
    assert "postgres" in result.stderr  # suggestions listed
    assert "Traceback" not in result.stderr


def test_complexity_jinja_model_stripped(tmp_path):
    sql_file = tmp_path / "model.sql"
    sql_file.write_text(
        "{{ config(materialized='table') }}\nselect a, b from {{ ref('stg') }} where x = 1"
    )
    result = runner.invoke(app, ["complexity", str(sql_file), "--json"])
    assert result.exit_code == 0
    assert "Jinja placeholders" in result.stderr
    payload = json.loads(result.stdout)  # stdout stays parseable
    assert payload["metrics"]["projected_columns"] == 2


def test_complexity_non_utf8_file_exit_2(tmp_path):
    sql_file = tmp_path / "latin1.sql"
    sql_file.write_bytes(b"select caf\xe9 from t")
    result = runner.invoke(app, ["complexity", str(sql_file)])
    assert result.exit_code == 2
    assert "UTF-8" in result.stderr


def test_complexity_stdin(tmp_path):
    result = runner.invoke(app, ["complexity", "-", "--json"], input="select 1 from t\n")
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["path"] == "<stdin>"


def test_complexity_stdin_non_utf8_exit_2(tmp_path):
    # A non-UTF-8 pipe must fail like a non-UTF-8 file (exit 2), never a traceback.
    result = runner.invoke(app, ["complexity", "-"], input=b"select caf\xe9 from t")
    assert result.exit_code == 2
    assert "UTF-8" in result.stderr
    assert "<stdin>" in result.stderr
    assert "Traceback" not in result.stderr


def test_complexity_jinja_unstrippable_exit_2(tmp_path):
    # Jinja markers present but the stripped SQL still won't parse -> exit 2 with a
    # compiled-SQL hint.
    sql_file = tmp_path / "model.sql"
    sql_file.write_text("{{ config() }}\nselect ((( from {{ ref('t') }}")
    result = runner.invoke(app, ["complexity", str(sql_file)])
    assert result.exit_code == 2
    assert "target/compiled/" in result.stderr


def test_complexity_missing_file_exit_2(tmp_path):
    result = runner.invoke(app, ["complexity", str(tmp_path / "nope.sql")])
    assert result.exit_code == 2


def test_help_shows_exit_code_contract():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Exit codes" in result.stdout
    assert "gate failure" in result.stdout
