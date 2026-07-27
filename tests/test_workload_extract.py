import pytest
import sqlglot
from sqlglot import exp

from sqlquality.models import ColumnRole, Relation
from sqlquality.sqlast import parse
from sqlquality.workload.extract import (
    AmbiguousRelation,
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


def test_update_from_second_table_bare_column_is_dropped_not_guessed():
    """`UPDATE ... FROM` puts two tables in scope; a bare column has no unambiguous target.

    ``customer_id`` is unqualified here (qualify() leaves DML columns bare), so without the
    ``len(tables) == 1`` guard in ``_collect_dml`` it would be attributed to ``tables[0]``
    (``orders``) purely because that table happened to be found first — a guess dressed up
    as a fact, not something the query actually told us.
    """
    usage = _usage("update orders set status = 'x' from customers where customer_id = customers.id")
    assert not any(column == "customer_id" for _relation, column, _role in usage)


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
    """`orders` exists in both schemas of COLLIDING, so the schema-map fallback alone would
    find two owners and refuse to guess. Only the explicit `staging.` qualifier can produce
    the expected answer — deleting `resolve_relation`'s `if table.db:` branch collapses this
    to `set()` because the fallback returns None for an ambiguous bare name.
    """
    tree = parse("select status from staging.orders where id > 1", "postgres")
    usage = extract_usage(tree, "postgres", COLLIDING)
    assert {relation for relation, _c, _r in usage} == {Relation("staging", "orders")}


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


def test_resolve_relation_returns_none_for_an_explicit_schema_never_introspected():
    """An explicit qualifier naming a schema we never introspected must not be trusted
    blindly. `qualify()` does not validate this for UPDATE/DELETE targets (see
    resolve_relation's docstring) — an ungated `table.db` branch would manufacture
    `Relation("other", "orders")`, a phantom that matches no catalog fact.
    """
    table = exp.Table(this=exp.to_identifier("orders"), db=exp.to_identifier("other"))
    assert resolve_relation(table, ONE_SCHEMA) is None


def test_dml_columns_attribute_to_the_qualified_target():
    tree = parse("update orders set status = 'y' where id = 1", "postgres")
    usage = extract_usage(tree, "postgres", ONE_SCHEMA)
    assert (Relation("public", "orders"), "id", ColumnRole.EQUALITY) in usage


def test_dml_with_an_unintrospected_explicit_schema_does_not_leak_a_phantom_relation():
    """The reachable case: `qualify()` leaves UPDATE/DELETE columns and their target table
    unvalidated against the schema, so an explicitly-qualified schema qualify() never saw
    sails through untouched. Confirmed by probing sqlglot 30.12 directly — this statement
    raises nothing. Without the guard in resolve_relation, this attributes `id` to a
    phantom `Relation("other", "orders")` instead of dropping it.
    """
    tree = parse("update other.orders set status = 'x' where id = 1", "postgres")
    usage = extract_usage(tree, "postgres", ONE_SCHEMA)
    assert usage == ()


def test_ambiguous_bare_dml_target_raises_ambiguous_relation():
    """`qualify()` does not validate UPDATE/DELETE targets, so a bare name held by two
    introspected schemas reaches `_collect_dml` with no `SchemaError` ever raised. Left
    silent, the statement would vanish with no usage and no counter moved — reported as
    analysed by `aggregate` when it was not. It must raise the same way the SELECT-path
    ambiguity does.
    """
    tree = parse("update orders set status = 'x' where id = 1", "postgres")
    with pytest.raises(AmbiguousRelation):
        extract_usage(tree, "postgres", COLLIDING)


def test_relation_str_is_schema_dot_table():
    """Proposal titles and JSON keys render a Relation through this — pin the contract."""
    assert str(Relation("public", "orders")) == "public.orders"
