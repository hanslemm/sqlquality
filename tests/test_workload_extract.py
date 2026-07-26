import pytest
import sqlglot

from sqlquality.models import ColumnRole
from sqlquality.workload.extract import UnqualifiableQuery, extract_usage

SCHEMA = {
    "orders": {
        "id": "INT",
        "customer_id": "INT",
        "status": "TEXT",
        "created_at": "TIMESTAMP",
        "note": "TEXT",
        "shipped_at": "TIMESTAMP",
    },
    "customers": {"id": "INT", "email": "TEXT", "status": "TEXT"},
}


def _usage(sql):
    tree = sqlglot.parse_one(sql, dialect="postgres")
    return set(extract_usage(tree, "postgres", SCHEMA))


def test_where_equality_is_equality_role():
    assert ("orders", "status", ColumnRole.EQUALITY) in _usage(
        "select id from orders where status = $1"
    )


def test_in_predicate_is_equality_role():
    assert ("orders", "status", ColumnRole.EQUALITY) in _usage(
        "select id from orders where status in ($1, $2)"
    )


def test_comparison_is_range_role():
    assert ("orders", "created_at", ColumnRole.RANGE) in _usage(
        "select id from orders where created_at > $1"
    )


def test_between_is_range_role():
    assert ("orders", "created_at", ColumnRole.RANGE) in _usage(
        "select id from orders where created_at between $1 and $2"
    )


def test_join_key_is_join_role_not_equality():
    usage = _usage("select o.id from orders o join customers c on c.id = o.customer_id")
    assert ("orders", "customer_id", ColumnRole.JOIN) in usage
    assert ("customers", "id", ColumnRole.JOIN) in usage
    assert ("orders", "customer_id", ColumnRole.EQUALITY) not in usage


def test_order_by_is_sort_role():
    assert ("orders", "created_at", ColumnRole.SORT) in _usage(
        "select id from orders order by created_at desc"
    )


def test_group_by_is_group_role():
    assert ("orders", "status", ColumnRole.GROUP) in _usage(
        "select status, count(*) from orders group by status"
    )


def test_window_order_by_is_not_a_query_sort_key():
    usage = _usage(
        "select id, row_number() over (partition by status order by created_at) from orders"
    )
    assert ("orders", "created_at", ColumnRole.SORT) not in usage


def test_null_checks_carry_polarity():
    assert ("orders", "shipped_at", ColumnRole.NULL_CHECK) in _usage(
        "select id from orders where shipped_at is null"
    )
    assert ("orders", "shipped_at", ColumnRole.NOT_NULL_CHECK) in _usage(
        "select id from orders where shipped_at is not null"
    )


def test_function_wrapped_predicate_is_non_sargable():
    usage = _usage("select id from orders where lower(status) = $1")
    assert ("orders", "status", ColumnRole.NON_SARGABLE) in usage
    assert ("orders", "status", ColumnRole.EQUALITY) not in usage


def test_cast_wrapped_predicate_is_non_sargable():
    assert ("orders", "id", ColumnRole.NON_SARGABLE) in _usage(
        "select id from orders where id::text = $1"
    )


def test_projected_columns_produce_no_usage():
    assert _usage("select note from orders") == set()


def test_update_where_predicate_is_attributed_to_the_target_table():
    """qualify() leaves DML columns bare (table == ''), so this needs the sole-table path."""
    assert ("orders", "created_at", ColumnRole.RANGE) in _usage(
        "update orders set status = $1 where created_at < $2"
    )


def test_update_set_clause_column_is_not_a_predicate():
    """`SET status = $1` parses to EQ(status, $1) with no WHERE ancestor.

    Recording it as EQUALITY would make us propose indexing the column being written.
    """
    usage = _usage("update orders set status = $1 where created_at < $2")
    assert ("orders", "status", ColumnRole.EQUALITY) not in usage
    assert not any(column == "status" for _table, column, _role in usage)


def test_delete_where_predicate_is_attributed():
    assert ("orders", "status", ColumnRole.EQUALITY) in _usage(
        "delete from orders where status = $1"
    )


def test_insert_values_produces_no_usage():
    assert _usage("insert into customers (id, email) values ($1, $2)") == set()


def test_insert_from_select_attributes_the_source_table():
    assert ("orders", "status", ColumnRole.EQUALITY) in _usage(
        "insert into customers (id) select customer_id from orders where status = $1"
    )


def test_projected_comparison_is_not_a_predicate():
    assert _usage("select status = $1 as is_shipped from orders") == set()


def test_reused_alias_across_scopes_does_not_corrupt_attribution():
    """The bug a flat alias map causes.

    Both scopes alias their table `o`. A single tree-wide map keeps whichever table
    find_all visited last, so the outer filter on `orders` was attributed to `customers`
    and the `orders` entries vanished entirely — a wrong index recommendation, silently.
    """
    usage = _usage(
        "select o.id from orders o where o.status = $1 "
        "and o.id in (select o.id from customers o where o.status = $2)"
    )
    assert ("orders", "status", ColumnRole.EQUALITY) in usage
    assert ("customers", "status", ColumnRole.EQUALITY) in usage


def test_distinct_aliases_across_scopes_both_resolve():
    usage = _usage(
        "select o.id from orders o where o.status = $1 "
        "and o.id in (select c.id from customers c where c.status = $2)"
    )
    assert ("orders", "status", ColumnRole.EQUALITY) in usage
    assert ("customers", "status", ColumnRole.EQUALITY) in usage


def test_self_join_aliases_both_resolve_to_the_same_table():
    usage = _usage(
        "select a.id from orders a join orders b on b.customer_id = a.id where a.status = $1"
    )
    assert ("orders", "customer_id", ColumnRole.JOIN) in usage
    assert ("orders", "status", ColumnRole.EQUALITY) in usage


def test_cte_predicate_resolves_to_the_underlying_base_table():
    """A CTE name is not a base table, so the outer predicate on it is skipped; the CTE's
    own scope still contributes its real base-table predicate."""
    usage = _usage(
        "with recent as (select id, status from orders where created_at > $1) "
        "select id from recent where status = $2"
    )
    assert ("orders", "created_at", ColumnRole.RANGE) in usage
    assert not any(table == "recent" for table, _column, _role in usage)


def test_unresolvable_column_raises_unqualifiable():
    tree = sqlglot.parse_one("select nope from mystery_table", dialect="postgres")
    with pytest.raises(UnqualifiableQuery):
        extract_usage(tree, "postgres", SCHEMA)
