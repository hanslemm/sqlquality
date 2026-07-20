from sqlquality.linter import fix_sql, lint_sql
from sqlquality.models import Severity

STAR_SQL = "SELECT *  from users where id=1"
MESSY_SQL = "select   a,b from t where  x=1"
BROKEN_SQL = "select from where"
DBT_MACRO_SQL = "select {{ my_macro() }} as id\n"


def test_lint_returns_findings():
    findings = lint_sql(STAR_SQL, "postgres")
    codes = {f.code for f in findings}
    assert "AM04" in codes  # SELECT * -> ambiguous column count
    assert any(f.fixable for f in findings)  # at least one auto-fixable layout rule
    assert all(f.line >= 1 for f in findings)


def test_lint_parse_error_is_error_severity():
    findings = lint_sql(BROKEN_SQL, "postgres")
    prs = [f for f in findings if f.code == "PRS"]
    assert prs, "expected a PRS parse-error finding"
    assert prs[0].severity is Severity.ERROR
    assert prs[0].fixable is False


def test_exclude_rules():
    all_codes = {f.code for f in lint_sql(STAR_SQL, "postgres")}
    excluded = {f.code for f in lint_sql(STAR_SQL, "postgres", exclude_rules=["LT01"])}
    assert "LT01" in all_codes
    assert "LT01" not in excluded


def test_fix_reformats_sql():
    fixed = fix_sql(MESSY_SQL, "postgres")
    assert fixed != MESSY_SQL
    assert "where x = 1" in fixed  # spacing normalized around '='


def test_redshift_dialect_ok():
    findings = lint_sql(STAR_SQL, "redshift")
    assert any(f.code == "AM04" for f in findings)


def test_unresolved_jinja_demoted_to_info():
    findings = lint_sql(DBT_MACRO_SQL, "postgres")
    templating = [f for f in findings if f.code in ("TMP", "PRS")]
    assert templating, "expected TMP/PRS from the unresolved macro"
    for f in templating:
        assert f.severity is Severity.INFO
        assert "unresolved Jinja" in f.message


def test_genuine_parse_error_not_demoted():
    # No Jinja and no TMP -> a real PRS parse error stays ERROR, no hint appended.
    findings = lint_sql(BROKEN_SQL, "postgres")
    prs = [f for f in findings if f.code == "PRS"]
    assert prs and prs[0].severity is Severity.ERROR
    assert "unresolved Jinja" not in prs[0].message
