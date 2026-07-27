import pytest
import sqlglot
from sqlglot import exp

from sqlquality.models import ColumnRole, Relation
from sqlquality.sqlast import parse
from sqlquality.workload.extract import (
    UnqualifiableQuery,
    extract_usage,
    resolve_relation,
)

SCHEMA = {
    "public": {
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
}
ORDERS = Relation("public", "orders")
CUSTOMERS = Relation("public", "customers")


def _usage(sql):
    tree = sqlglot.parse_one(sql, dialect="postgres")
    return set(extract_usage(tree, "postgres", SCHEMA))


def test_where_equality_is_equality_role():
    assert (ORDERS, "status", ColumnRole.EQUALITY) in _usage(
        "select id from orders where status = $1"
    )


def test_in_predicate_is_equality_role():
    assert (ORDERS, "status", ColumnRole.EQUALITY) in _usage(
        "select id from orders where status in ($1, $2)"
    )


def test_comparison_is_range_role():
    assert (ORDERS, "created_at", ColumnRole.RANGE) in _usage(
        "select id from orders where created_at > $1"
    )


def test_between_is_range_role():
    assert (ORDERS, "created_at", ColumnRole.RANGE) in _usage(
        "select id from orders where created_at between $1 and $2"
    )


def test_join_key_is_join_role_not_equality():
    usage = _usage("select o.id from orders o join customers c on c.id = o.customer_id")
    assert (ORDERS, "customer_id", ColumnRole.JOIN) in usage
    assert (CUSTOMERS, "id", ColumnRole.JOIN) in usage
    assert (ORDERS, "customer_id", ColumnRole.EQUALITY) not in usage


def test_order_by_is_sort_role():
    assert (ORDERS, "created_at", ColumnRole.SORT) in _usage(
        "select id from orders order by created_at desc"
    )


def test_group_by_is_group_role():
    assert (ORDERS, "status", ColumnRole.GROUP) in _usage(
        "select status, count(*) from orders group by status"
    )


def test_window_order_by_is_not_a_query_sort_key():
    usage = _usage(
        "select id, row_number() over (partition by status order by created_at) from orders"
    )
    assert (ORDERS, "created_at", ColumnRole.SORT) not in usage


def test_null_checks_carry_polarity():
    assert (ORDERS, "shipped_at", ColumnRole.NULL_CHECK) in _usage(
        "select id from orders where shipped_at is null"
    )
    assert (ORDERS, "shipped_at", ColumnRole.NOT_NULL_CHECK) in _usage(
        "select id from orders where shipped_at is not null"
    )


def test_function_wrapped_predicate_is_non_sargable():
    usage = _usage("select id from orders where lower(status) = $1")
    assert (ORDERS, "status", ColumnRole.NON_SARGABLE) in usage
    assert (ORDERS, "status", ColumnRole.EQUALITY) not in usage


def test_cast_wrapped_predicate_is_non_sargable():
    assert (ORDERS, "id", ColumnRole.NON_SARGABLE) in _usage(
        "select id from orders where id::text = $1"
    )


def test_projected_columns_produce_no_usage():
    assert _usage("select note from orders") == set()


def test_update_where_predicate_is_attributed_to_the_target_table():
    """qualify() leaves DML columns bare (table == ''), so this needs the sole-table path."""
    assert (ORDERS, "created_at", ColumnRole.RANGE) in _usage(
        "update orders set status = $1 where created_at < $2"
    )


def test_update_set_clause_column_is_not_a_predicate():
    """`SET status = $1` parses to EQ(status, $1) with no WHERE ancestor.

    Recording it as EQUALITY would make us propose indexing the column being written.
    """
    usage = _usage("update orders set status = $1 where created_at < $2")
    assert (ORDERS, "status", ColumnRole.EQUALITY) not in usage
    assert not any(column == "status" for _relation, column, _role in usage)


def test_delete_where_predicate_is_attributed():
    assert (ORDERS, "status", ColumnRole.EQUALITY) in _usage("delete from orders where status = $1")


def test_insert_values_produces_no_usage():
    assert _usage("insert into customers (id, email) values ($1, $2)") == set()


def test_insert_from_select_attributes_the_source_table():
    assert (ORDERS, "status", ColumnRole.EQUALITY) in _usage(
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
    assert (ORDERS, "status", ColumnRole.EQUALITY) in usage
    assert (CUSTOMERS, "status", ColumnRole.EQUALITY) in usage


def test_distinct_aliases_across_scopes_both_resolve():
    usage = _usage(
        "select o.id from orders o where o.status = $1 "
        "and o.id in (select c.id from customers c where c.status = $2)"
    )
    assert (ORDERS, "status", ColumnRole.EQUALITY) in usage
    assert (CUSTOMERS, "status", ColumnRole.EQUALITY) in usage


def test_self_join_aliases_both_resolve_to_the_same_table():
    usage = _usage(
        "select a.id from orders a join orders b on b.customer_id = a.id where a.status = $1"
    )
    assert (ORDERS, "customer_id", ColumnRole.JOIN) in usage
    assert (ORDERS, "status", ColumnRole.EQUALITY) in usage


def test_cte_predicate_resolves_to_the_underlying_base_table():
    """A CTE name is not a base table, so the outer predicate on it is skipped; the CTE's
    own scope still contributes its real base-table predicate."""
    usage = _usage(
        "with recent as (select id, status from orders where created_at > $1) "
        "select id from recent where status = $2"
    )
    assert (ORDERS, "created_at", ColumnRole.RANGE) in usage
    assert not any(relation.table == "recent" for relation, _column, _role in usage)


def test_unresolvable_column_raises_unqualifiable():
    tree = sqlglot.parse_one("select nope from mystery_table", dialect="postgres")
    with pytest.raises(UnqualifiableQuery):
        extract_usage(tree, "postgres", SCHEMA)


ONE_SCHEMA = {"public": {"orders": {"id": "int", "status": "text", "shipped_at": "timestamp"}}}
TWO_SCHEMAS = {
    "sales": {"orders": {"id": "int", "status": "text"}},
    "staging": {"items": {"sku": "text", "qty": "int"}},
}
COLLIDING = {
    "sales": {"orders": {"id": "int", "status": "text"}},
    "staging": {"orders": {"id": "int", "status": "text"}},
}


def test_bare_table_resolves_to_its_only_owning_schema():
    """The common case: production SQL relies on search_path and says `from orders`.

    qualify() leaves Table.db empty here, so a `table.db`-only implementation keys this
    under Relation("", "orders") and every catalog lookup misses.
    """
    tree = parse("select id from orders where status = 'x'", "postgres")
    usage = extract_usage(tree, "postgres", ONE_SCHEMA)
    assert {relation for relation, _c, _r in usage} == {Relation("public", "orders")}


def test_explicitly_qualified_table_uses_the_schema_it_names():
    tree = parse("select sku from staging.items where qty > 1", "postgres")
    usage = extract_usage(tree, "postgres", TWO_SCHEMAS)
    assert {relation for relation, _c, _r in usage} == {Relation("staging", "items")}


def test_two_schemas_distinct_names_attribute_to_the_right_one():
    """A join across schemas must not collapse both sides onto one relation."""
    tree = parse("select o.id, i.sku from orders o join items i on i.sku = o.status", "postgres")
    usage = extract_usage(tree, "postgres", TWO_SCHEMAS)
    assert {relation for relation, _c, _r in usage} == {
        Relation("sales", "orders"),
        Relation("staging", "items"),
    }


def test_ambiguous_bare_name_is_unqualifiable_not_a_crash():
    """sqlglot raises SchemaError, which is NOT an OptimizeError subclass."""
    tree = parse("select id from orders where status = 'x'", "postgres")
    with pytest.raises(UnqualifiableQuery):
        extract_usage(tree, "postgres", COLLIDING)


def test_resolve_relation_prefers_an_explicit_db_over_the_map():
    table = exp.Table(this=exp.to_identifier("orders"), db=exp.to_identifier("sales"))
    assert resolve_relation(table, COLLIDING) == Relation("sales", "orders")


def test_resolve_relation_returns_none_when_ambiguous():
    """Two owners is not a guess we are entitled to make."""
    table = exp.Table(this=exp.to_identifier("orders"))
    assert resolve_relation(table, COLLIDING) is None


def test_resolve_relation_returns_none_for_a_table_outside_the_map():
    table = exp.Table(this=exp.to_identifier("nowhere"))
    assert resolve_relation(table, ONE_SCHEMA) is None


def test_dml_columns_attribute_to_the_qualified_target():
    tree = parse("update orders set status = 'y' where id = 1", "postgres")
    usage = extract_usage(tree, "postgres", ONE_SCHEMA)
    assert (Relation("public", "orders"), "id", ColumnRole.EQUALITY) in usage
