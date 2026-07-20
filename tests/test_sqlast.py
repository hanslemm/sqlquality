import pytest
import sqlglot

from sqlquality.sqlast import SqlParseError, analyze_sql, strip_jinja

SQL_A = "select id, name from users where active"
SQL_B = (
    "with recent as (select * from orders where created_at > current_date - 7) "
    "select u.id, count(*) as n, "
    "row_number() over (partition by u.id order by max(o.created_at)) "
    "from users u join recent o on o.user_id = u.id group by u.id"
)
SQL_C = (
    "select * from (select a, b from "
    "(select a, b, c from t where c in (select c from filt)) x where b > 0) y"
)


def test_analyze_simple_select():
    m = analyze_sql(SQL_A, "postgres")
    assert (m.join_count, m.cte_count, m.subquery_count, m.select_count) == (0, 0, 0, 1)
    assert m.max_select_depth == 1
    assert m.projected_columns == 2


def test_analyze_cte_join_window():
    m = analyze_sql(SQL_B, "postgres")
    assert m.join_count == 1
    assert m.cte_count == 1
    assert m.subquery_count == 0
    assert m.window_count == 1
    assert m.select_count == 2
    assert m.max_select_depth == 2
    assert m.projected_columns == 3


def test_analyze_nested_subqueries():
    m = analyze_sql(SQL_C, "postgres")
    assert m.subquery_count == 3
    assert m.select_count == 4
    assert m.max_select_depth == 4
    assert m.projected_columns == 1


def test_set_operation_counts_except():
    m = analyze_sql("select a from t except select a from u", "postgres")
    assert m.union_count == 1


def test_parse_error_raised_on_garbage():
    with pytest.raises(SqlParseError):
        analyze_sql("select from where", "postgres")


def test_parse_error_on_unterminated_string():
    with pytest.raises(SqlParseError):
        analyze_sql("select 'unterminated", "postgres")


def test_exists_and_in_subqueries_score_symmetrically():
    exists = analyze_sql(
        "select 1 from t where exists (select 1 from t2 where t2.id = t.id)", "postgres"
    )
    in_ = analyze_sql("select 1 from t where id in (select id from t2)", "postgres")
    assert exists.subquery_count == in_.subquery_count == 1


def test_exists_wrapped_subquery_counts_once():
    m = analyze_sql("select 1 from t where exists ((select 1 from t2))", "postgres")
    assert m.subquery_count == 1


def test_strip_jinja_leading_config_and_ref_parse():
    raw = "{{ config(materialized='table') }}\nselect * from {{ ref('stg') }}"
    stripped = strip_jinja(raw)
    assert "{{" not in stripped and "}}" not in stripped
    assert stripped.lstrip().lower().startswith("select")
    sqlglot.parse_one(stripped, dialect="postgres")  # must not raise


def test_strip_jinja_removes_statement_and_comment_blocks():
    raw = (
        "select a from t\n"
        "{% if is_incremental() %} where a > 0 {% endif %}\n"
        "{# a trailing comment #}"
    )
    stripped = strip_jinja(raw)
    assert "{%" not in stripped and "%}" not in stripped
    assert "{#" not in stripped and "#}" not in stripped
    assert "is_incremental" not in stripped
    sqlglot.parse_one(stripped, dialect="postgres")


def test_strip_jinja_multiline_blocks():
    raw = "{% set x = 1\n   y = 2 %}\nselect {{ col }} from t"
    stripped = strip_jinja(raw)
    assert "{%" not in stripped and "{{" not in stripped
    assert stripped.lstrip().lower().startswith("select")
