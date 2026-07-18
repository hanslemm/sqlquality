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
