import json

from typer.testing import CliRunner

from sqlquality.cli import app

runner = CliRunner()


def test_lint_json(tmp_path):
    f = tmp_path / "m.sql"
    f.write_text("SELECT *  from users where id=1")
    result = runner.invoke(app, ["lint", str(f), "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)["files"][0]
    assert payload["path"] == str(f)
    assert payload["fixed"] is False
    assert any(item["code"] == "AM04" for item in payload["findings"])


def test_lint_multiple_files(tmp_path):
    a = tmp_path / "a.sql"
    a.write_text("SELECT *  from users")
    b = tmp_path / "b.sql"
    b.write_text("select 1")
    result = runner.invoke(app, ["lint", str(a), str(b), "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert {item["path"] for item in payload["files"]} == {str(a), str(b)}


def test_lint_parse_error_exit_1(tmp_path):
    f = tmp_path / "bad.sql"
    f.write_text("select from where")
    result = runner.invoke(app, ["lint", str(f)])
    assert result.exit_code == 1  # PRS is ERROR severity


def test_lint_fix_rewrites_file(tmp_path):
    f = tmp_path / "m.sql"
    f.write_text("select   a,b from t where  x=1")
    before = f.read_text()
    result = runner.invoke(app, ["lint", str(f), "--fix"])
    assert result.exit_code == 0
    after = f.read_text()
    assert after != before
    assert "where x = 1" in after


def test_lint_human_output(tmp_path):
    f = tmp_path / "m.sql"
    f.write_text("SELECT *  from users where id=1")
    result = runner.invoke(app, ["lint", str(f)])
    assert result.exit_code == 0
    assert "AM04" in result.stdout


def test_lint_fix_no_change_on_unparseable(tmp_path):
    f = tmp_path / "bad.sql"
    f.write_text("select from where")
    result = runner.invoke(app, ["lint", str(f), "--fix"])
    assert result.exit_code == 1  # still a parse error
    assert f.read_text() == "select from where"  # unchanged, not rewritten


def test_lint_multiple_files_exit_1_if_any_error(tmp_path):
    good = tmp_path / "good.sql"
    good.write_text("select 1")
    bad = tmp_path / "bad.sql"
    bad.write_text("select from where")  # PRS -> ERROR severity
    result = runner.invoke(app, ["lint", str(good), str(bad)])
    assert result.exit_code == 1
