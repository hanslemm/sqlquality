import pytest

from sqlquality.sqlast import SqlParseError, analyze_sql

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
