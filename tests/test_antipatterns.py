from sqlquality.antipatterns import antipattern_findings
from sqlquality.models import Severity


def _codes(sql: str) -> set[str]:
    return {f.code for f in antipattern_findings(sql, "postgres")}


def test_select_star_flagged():
    assert "SQ001" in _codes("select * from users")
    assert "SQ001" in _codes("select u.* from users u")


def test_count_star_not_flagged():
    assert "SQ001" not in _codes("select count(*) from t")
    assert "SQ001" not in _codes("select count(*) as n, id from t group by id")


def test_select_star_in_exists_not_flagged():
    assert "SQ001" not in _codes(
        "select id from t where exists (select * from t2 where t2.id = t.id)"
    )


def test_select_star_from_cte_closer_not_flagged():
    assert "SQ001" not in _codes("with final as (select a, b from t) select * from final")


def test_select_star_not_sole_projection_flagged():
    assert "SQ001" in _codes("with final as (select a, b from t) select *, id from final")


def test_select_star_from_non_cte_flagged():
    assert "SQ001" in _codes("select * from some_table")


def test_cartesian_flagged():
    assert "SQ002" in _codes("select x from a, b")
    assert "SQ002" in _codes("select x from a cross join b")


def test_comma_join_predicate_in_where_not_flagged():
    assert "SQ002" not in _codes("select x from a, b where a.id = b.id")


def test_constant_true_join_flagged():
    assert "SQ002" in _codes("select x from a join b on true")
    assert "SQ002" in _codes("select x from a join b on 1 = 1")


def test_join_on_not_flagged():
    assert "SQ002" not in _codes("select x from a join b on a.id = b.id")
    assert "SQ002" not in _codes("select x from a inner join b using (id)")


def test_leading_wildcard_flagged():
    assert "SQ003" in _codes("select x from t where name like '%smith'")


def test_trailing_wildcard_not_flagged():
    assert "SQ003" not in _codes("select x from t where name like 'smith%'")


def test_parse_error_is_sq000_error():
    findings = antipattern_findings("select from where", "postgres")
    assert findings[0].code == "SQ000"
    assert findings[0].severity is Severity.ERROR


def test_clean_sql_no_findings():
    assert antipattern_findings("select id, name from users where id = 1", "postgres") == []
