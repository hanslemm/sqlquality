import json

import sqlfluff
from typer.testing import CliRunner

from sqlquality.cli import app

runner = CliRunner()


def test_lint_json(tmp_path):
    f = tmp_path / "m.sql"
    f.write_text("SELECT *  from users where id=1")
    result = runner.invoke(app, ["lint", str(f), "--json"])
    assert result.exit_code == 1  # AM04 is a WARNING -> gates the commit
    payload = json.loads(result.stdout)["files"][0]
    assert payload["path"] == str(f)
    assert payload["fixed"] is False
    assert any(item["code"] == "AM04" for item in payload["findings"])


def test_lint_multiple_files(tmp_path):
    a = tmp_path / "a.sql"
    a.write_text("SELECT *  from users")
    b = tmp_path / "b.sql"
    b.write_text("select 1")  # LT12 (missing trailing newline) -> WARNING -> gates
    result = runner.invoke(app, ["lint", str(a), str(b), "--json"])
    assert result.exit_code == 1
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
    # Findings are reported on the pre-fix source (still has WARNING violations).
    assert result.exit_code == 1
    after = f.read_text()
    assert after != before
    assert "where x = 1" in after


def test_lint_human_output(tmp_path):
    f = tmp_path / "m.sql"
    f.write_text("SELECT *  from users where id=1")
    result = runner.invoke(app, ["lint", str(f)])
    assert result.exit_code == 1  # AM04 WARNING gates the commit
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


def test_lint_dbt_macro_is_info_and_does_not_gate(tmp_path):
    # Unresolved dbt/custom macro -> TMP + PRS from the Jinja templater. These are
    # advisory (INFO) and must not fail the commit when no real violations exist.
    f = tmp_path / "model.sql"
    f.write_text("select {{ my_macro() }} as id\n")
    result = runner.invoke(app, ["lint", str(f), "--json"])
    findings = json.loads(result.stdout)["files"][0]["findings"]
    templating = [item for item in findings if item["code"] in ("TMP", "PRS")]
    assert templating, "expected TMP/PRS from the unresolved macro"
    for item in templating:
        assert item["severity"] == "info"
        assert "unresolved Jinja" in item["message"]
    # Only INFO findings here -> exit code driven purely by WARNING+ findings -> 0.
    assert all(item["severity"] == "info" for item in findings)
    assert result.exit_code == 0


def test_lint_dbt_macro_warning_still_gates(tmp_path):
    # Same unresolved-macro noise, but the file also has a genuine WARNING (LT05:
    # line too long) -> the INFO demotion of TMP/PRS must not mask the warning.
    f = tmp_path / "model.sql"
    f.write_text(
        "select {{ dbt_utils.generate_surrogate_key(['a', 'b']) }} as id from {{ ref('users') }}\n"
    )
    result = runner.invoke(app, ["lint", str(f), "--json"])
    findings = json.loads(result.stdout)["files"][0]["findings"]
    for item in findings:
        if item["code"] in ("TMP", "PRS"):
            assert item["severity"] == "info"
    assert any(item["severity"] == "warning" for item in findings)
    assert result.exit_code == 1


def test_lint_plain_parse_error_is_error(tmp_path):
    f = tmp_path / "bad.sql"
    f.write_text("select ((( from")  # plain SQL, no Jinja -> genuine PRS ERROR
    result = runner.invoke(app, ["lint", str(f), "--json"])
    assert result.exit_code == 1
    prs = [i for i in json.loads(result.stdout)["files"][0]["findings"] if i["code"] == "PRS"]
    assert prs and prs[0]["severity"] == "error"
    assert "unresolved Jinja" not in prs[0]["message"]


def test_lint_warn_only_exits_zero_with_findings(tmp_path):
    f = tmp_path / "m.sql"
    f.write_text("SELECT *  from users where id=1")
    result = runner.invoke(app, ["lint", str(f), "--warn-only", "--json"])
    assert result.exit_code == 0
    findings = json.loads(result.stdout)["files"][0]["findings"]
    assert any(item["code"] == "AM04" for item in findings)  # findings still emitted


def test_lint_sqlfluff_config_passthrough(tmp_path, monkeypatch):
    cfg = tmp_path / ".sqlfluff"
    cfg.write_text("[sqlfluff]\ndialect = postgres\n")
    f = tmp_path / "m.sql"
    f.write_text("select 1\n")

    lint_kwargs: dict = {}
    fix_kwargs: dict = {}

    def fake_lint(sql, **kwargs):
        lint_kwargs.update(kwargs)
        return []

    def fake_fix(sql, **kwargs):
        fix_kwargs.update(kwargs)
        return sql

    monkeypatch.setattr(sqlfluff, "lint", fake_lint)
    monkeypatch.setattr(sqlfluff, "fix", fake_fix)
    result = runner.invoke(app, ["lint", str(f), "--fix", "--sqlfluff-config", str(cfg), "--json"])
    assert result.exit_code == 0
    assert lint_kwargs["config_path"] == str(cfg)
    assert fix_kwargs["config_path"] == str(cfg)


def test_lint_jinja_in_comment_parse_error_stays_error(tmp_path):
    # Jinja only in a comment renders fine -> no TMP -> the PRS is a genuine parse
    # error and must stay ERROR (the raw-source scan would have wrongly demoted it).
    f = tmp_path / "bad.sql"
    f.write_text("-- comment mentions {{ ref('x') }}\nselect ((( from t")
    result = runner.invoke(app, ["lint", str(f), "--json"])
    assert result.exit_code == 1
    prs = [i for i in json.loads(result.stdout)["files"][0]["findings"] if i["code"] == "PRS"]
    assert prs and prs[0]["severity"] == "error"
    assert "unresolved Jinja" not in prs[0]["message"]


def test_lint_valid_jinja_broken_sql_parse_error_stays_error(tmp_path):
    # Valid Jinja that renders to broken SQL -> no TMP -> the PRS is genuine ERROR.
    f = tmp_path / "bad.sql"
    f.write_text("{% set n = 1 %}\nselect {{ n }} as x from")
    result = runner.invoke(app, ["lint", str(f), "--json"])
    assert result.exit_code == 1
    prs = [i for i in json.loads(result.stdout)["files"][0]["findings"] if i["code"] == "PRS"]
    assert prs and prs[0]["severity"] == "error"
    assert "unresolved Jinja" not in prs[0]["message"]


def test_lint_unknown_dialect_exit_2(tmp_path):
    f = tmp_path / "m.sql"
    f.write_text("select 1\n")
    result = runner.invoke(app, ["lint", str(f), "--dialect", "oracle9000"])
    assert result.exit_code == 2
    assert "oracle9000" in result.stderr
    assert "postgres" in result.stderr  # suggestions listed
    assert "Traceback" not in result.stderr


def test_lint_stdin(tmp_path):
    result = runner.invoke(app, ["lint", "-", "--json"], input="select 1 from t\n")
    assert result.exit_code == 0  # clean SQL -> no gating findings
    payload = json.loads(result.stdout)["files"][0]
    assert payload["path"] == "<stdin>"


def test_lint_fix_stdin_rejected(tmp_path):
    result = runner.invoke(app, ["lint", "--fix", "-"], input="select 1\n")
    assert result.exit_code == 2
    assert "stdin" in result.stderr


def test_lint_non_utf8_file_exit_2(tmp_path):
    f = tmp_path / "latin1.sql"
    f.write_bytes(b"select caf\xe9 from t")
    result = runner.invoke(app, ["lint", str(f)])
    assert result.exit_code == 2
    assert "UTF-8" in result.stderr


def test_lint_missing_file_exit_2(tmp_path):
    result = runner.invoke(app, ["lint", str(tmp_path / "nope.sql")])
    assert result.exit_code == 2
