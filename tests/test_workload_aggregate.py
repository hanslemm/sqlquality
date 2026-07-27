from sqlquality.models import ColumnRole, QueryStat, Relation, Workload
from sqlquality.workload.aggregate import aggregate, star_tables
from sqlquality.workload.fingerprint import FLAG_SELECT_STAR

SCHEMA = {"public": {"orders": {"id": "INT", "status": "TEXT", "created_at": "TIMESTAMP"}}}


def _workload(*pairs):
    return Workload(
        stats=tuple(
            QueryStat(fingerprint=f"fp{i}", sql=sql, calls=calls, total_time_ms=cost)
            for i, (sql, calls, cost) in enumerate(pairs)
        ),
        window_description="w",
    )


def _find(agg, column, role):
    return next(u for u in agg.usage if u.column == column and u.role is role)


def test_cost_share_is_fraction_of_analyzed_total():
    agg = aggregate(
        _workload(
            ("select id from orders where status = $1", 1, 75.0),
            ("select id from orders where created_at > $1", 1, 25.0),
        ),
        SCHEMA,
        "postgres",
    )
    assert agg.total_cost_ms == 100.0
    assert _find(agg, "status", ColumnRole.EQUALITY).cost_share == 0.75
    assert _find(agg, "created_at", ColumnRole.RANGE).cost_share == 0.25


def test_same_column_and_role_accumulates_across_fingerprints():
    agg = aggregate(
        _workload(
            ("select id from orders where status = $1", 2, 10.0),
            ("select created_at from orders where status = $1", 3, 30.0),
        ),
        SCHEMA,
        "postgres",
    )
    usage = _find(agg, "status", ColumnRole.EQUALITY)
    assert usage.cost_ms == 40.0
    assert usage.calls == 5
    assert usage.fingerprints == 2


def test_unqualifiable_queries_are_counted_not_raised():
    agg = aggregate(
        _workload(
            ("select mystery from unknown_table", 1, 10.0),
            ("select id from orders where status = $1", 1, 10.0),
        ),
        SCHEMA,
        "postgres",
    )
    assert agg.skipped_unqualifiable == 1
    assert agg.tables == frozenset({Relation("public", "orders")})


def test_usage_is_sorted_by_cost_descending():
    agg = aggregate(
        _workload(
            ("select id from orders where status = $1", 1, 5.0),
            ("select id from orders where created_at > $1", 1, 50.0),
        ),
        SCHEMA,
        "postgres",
    )
    assert agg.usage[0].column == "created_at"


def test_equal_cost_entries_are_ordered_canonically_not_by_arrival():
    """Two workloads differing only in arrival order must aggregate identically.

    Sorting on cost alone leaves ties to Python's stable sort, which preserves insertion
    order — so the same logical workload yields different output depending on the order
    the engine happened to return rows in, and downstream tests become order-dependent.
    """
    forward = aggregate(
        _workload(
            ("select id from orders where status = $1", 1, 10.0),
            ("select id from orders where created_at > $1", 1, 10.0),
        ),
        SCHEMA,
        "postgres",
    )
    reverse = aggregate(
        _workload(
            ("select id from orders where created_at > $1", 1, 10.0),
            ("select id from orders where status = $1", 1, 10.0),
        ),
        SCHEMA,
        "postgres",
    )
    assert [(u.relation, u.column, u.role) for u in forward.usage] == [
        (u.relation, u.column, u.role) for u in reverse.usage
    ]


def test_skipped_stats_still_count_toward_the_denominator():
    """cost_share is a fraction of the whole window, not of the analyzable part.

    The 90ms query cannot be qualified, so it contributes no usage — but its cost stays in
    the denominator, leaving the surviving 10ms query at 0.1 rather than 1.0. This keeps
    the number honest about how much of the database's work we actually explained.
    """
    agg = aggregate(
        _workload(
            ("select mystery from unknown_table", 1, 90.0),
            ("select id from orders where status = $1", 1, 10.0),
        ),
        SCHEMA,
        "postgres",
    )
    assert agg.skipped_unqualifiable == 1
    assert agg.total_cost_ms == 100.0
    assert _find(agg, "status", ColumnRole.EQUALITY).cost_share == 0.1


def test_a_multi_predicate_query_credits_its_full_cost_to_each_column():
    """Shares deliberately do not sum to 1: both predicates are involved in the same cost.

    Proposals therefore take the max cost_share over their columns, never the sum.
    """
    agg = aggregate(
        _workload(("select id from orders where status = $1 and created_at > $2", 1, 100.0)),
        SCHEMA,
        "postgres",
    )
    assert _find(agg, "status", ColumnRole.EQUALITY).cost_share == 1.0
    assert _find(agg, "created_at", ColumnRole.RANGE).cost_share == 1.0


def test_role_breaks_ties_when_table_and_column_match():
    """The fourth sort key. Same column, same cost, two roles — order must be canonical."""
    forward = aggregate(
        _workload(
            ("select id from orders where status = $1", 1, 10.0),
            ("select id from orders group by status", 1, 10.0),
        ),
        SCHEMA,
        "postgres",
    )
    reverse = aggregate(
        _workload(
            ("select id from orders group by status", 1, 10.0),
            ("select id from orders where status = $1", 1, 10.0),
        ),
        SCHEMA,
        "postgres",
    )
    assert [u.role for u in forward.usage] == [u.role for u in reverse.usage]


def test_empty_workload_yields_empty_aggregation_and_no_division_error():
    agg = aggregate(Workload(stats=(), window_description="w"), SCHEMA, "postgres")
    assert agg.usage == ()
    assert agg.total_cost_ms == 0.0
    assert agg.tables == frozenset()


def test_identifier_pattern_is_compiled_once_per_name(monkeypatch):
    """A fresh regex per identifier check thrashes re's own pattern cache.

    ``star_tables`` no longer text-matches (it parses and resolves through
    ``resolve_relation`` instead — see the ``star_tables`` tests below for why), so the
    caching this pins is exercised directly through ``mentions_identifier``, which is what
    the expression-index disclosures in ADV001/ADV007/ADV008 actually call many times over.
    It used to go through a `mentions_table` alias, which by then had no production caller at
    all — a test exercising a wrapper nobody used, of a cache everybody used.
    """
    import re as _re

    from sqlquality.workload import aggregate as agg

    compiles: list[str] = []
    real_compile = _re.compile

    def counting_compile(pattern, *args, **kwargs):
        compiles.append(pattern)
        return real_compile(pattern, *args, **kwargs)

    monkeypatch.setattr(agg._re if hasattr(agg, "_re") else _re, "compile", counting_compile)
    names = [f"t{i}" for i in range(20)] + ["orders"]
    for _ in range(5):
        for name in names:
            agg.mentions_identifier(name, "select * from orders")
    assert len(compiles) <= len(names), (
        f"compiled {len(compiles)} patterns for {len(names)} distinct identifiers across 5 passes"
    )


ONE_SCHEMA = {"public": {"orders": {"id": "int", "status": "text"}}}
TWO_SCHEMAS = {
    "sales": {"orders": {"id": "int", "status": "text"}},
    "staging": {"items": {"sku": "text", "qty": "int"}},
}
COLLIDING = {
    "sales": {"orders": {"id": "int", "status": "text"}},
    "staging": {"orders": {"id": "int", "status": "text"}},
}


def _mixed_workload(*sql: str) -> Workload:
    return Workload(
        stats=tuple(
            QueryStat(fingerprint=f"fp{i}", sql=s, calls=1, total_time_ms=100.0)
            for i, s in enumerate(sql)
        ),
        window_description="test",
    )


def test_usage_is_keyed_by_relation():
    result = aggregate(
        _mixed_workload("select id from orders where status = 'x'"), ONE_SCHEMA, "postgres"
    )
    assert {u.relation for u in result.usage} == {Relation("public", "orders")}
    assert result.tables == frozenset({Relation("public", "orders")})


def test_same_table_name_in_two_schemas_does_not_alias():
    """The bug multi-schema keying exists to fix: two relations, not one merged entry."""
    result = aggregate(
        _mixed_workload(
            "select id from sales.orders where status = 'x'",
            "select id from staging.orders where status = 'y'",
        ),
        COLLIDING,
        "postgres",
    )
    assert result.tables == frozenset({Relation("sales", "orders"), Relation("staging", "orders")})


def test_ambiguous_bare_name_is_counted_not_crashed():
    result = aggregate(
        _mixed_workload("select id from orders where status = 'x'"), COLLIDING, "postgres"
    )
    assert result.skipped_ambiguous == 1
    assert result.usage == ()


def test_bare_select_star_over_a_colliding_name_is_counted_ambiguous():
    """A bare `select * from orders` has no predicate, so `qualify()` has no column
    reference to validate and never raises for it — it just silently produces zero usage,
    the same as it would for an *unambiguous* bare star. Left uncounted, that reads as
    "nothing to see here" when what actually happened is the same unattributable-bare-name
    fact `AmbiguousRelation` reports for a predicated statement (see
    `test_ambiguous_bare_name_is_counted_not_crashed` above) — and the same fact ADV006's
    `_wide_relations_touched` later declines to guess at for exactly this statement shape.
    """
    workload = Workload(
        stats=(
            QueryStat(
                fingerprint="fp",
                sql="select * from orders",
                calls=1,
                total_time_ms=100.0,
                flags=frozenset({FLAG_SELECT_STAR}),
            ),
        ),
        window_description="test",
    )
    result = aggregate(workload, COLLIDING, "postgres")
    assert result.skipped_ambiguous == 1
    assert result.usage == ()
    assert result.tables == frozenset()


def test_bare_select_star_over_an_unambiguous_name_is_not_counted():
    """The new check must not fire just because a statement is a bare star — only when the
    bare name it references is genuinely held by more than one introspected schema."""
    workload = Workload(
        stats=(
            QueryStat(
                fingerprint="fp",
                sql="select * from items",
                calls=1,
                total_time_ms=100.0,
                flags=frozenset({FLAG_SELECT_STAR}),
            ),
        ),
        window_description="test",
    )
    result = aggregate(workload, TWO_SCHEMAS, "postgres")
    assert result.skipped_ambiguous == 0
    assert result.usage == ()


def test_ambiguous_bare_reference_is_counted_even_without_a_literal_star():
    """`select count(*) from orders` and `select 1 from orders` are not flagged
    `FLAG_SELECT_STAR` — that flag only marks a literal `SELECT *` — but neither references
    any column by name either, so `qualify()` never raises for either of them, exactly like
    the bare-star case above. Gating the check on the star flag let these two escape *both*
    counters: parsed fine, zero usage, never raised, never counted.
    """
    result = aggregate(
        _mixed_workload("select count(*) from orders", "select 1 from orders"),
        COLLIDING,
        "postgres",
    )
    assert result.skipped_ambiguous == 2
    assert result.usage == ()


def test_a_qualified_reference_to_a_colliding_name_is_not_counted_ambiguous():
    """`select * from sales.orders` produces no usage — a star has no predicate to attribute
    — but it is not *ambiguous*: the statement says which schema it means.

    `_references_an_ambiguous_bare_table` skips any reference carrying a `.db` qualifier, and
    nothing pinned that skip: removing it left the whole suite green while this statement
    started counting toward `skipped_ambiguous`, which drives both the low-coverage share and
    a warning whose remedy is "qualify the table in the query" — advice already followed. The
    unqualified twin below is asserted in the same test so a guard that silently swallowed
    *both* cases could not pass either.
    """
    qualified = Workload(
        stats=(
            QueryStat(
                fingerprint="fp",
                sql="select * from sales.orders",
                calls=1,
                total_time_ms=100.0,
                flags=frozenset({FLAG_SELECT_STAR}),
            ),
        ),
        window_description="test",
    )
    bare = Workload(
        stats=(
            QueryStat(
                fingerprint="fp",
                sql="select * from orders",
                calls=1,
                total_time_ms=100.0,
                flags=frozenset({FLAG_SELECT_STAR}),
            ),
        ),
        window_description="test",
    )
    assert aggregate(qualified, COLLIDING, "postgres").skipped_ambiguous == 0
    assert aggregate(bare, COLLIDING, "postgres").skipped_ambiguous == 1


def test_a_plain_parse_failure_is_not_counted_as_ambiguous():
    """The two counters must not both fire for the same statement."""
    result = aggregate(_mixed_workload("this is not sql at all"), ONE_SCHEMA, "postgres")
    assert result.skipped_ambiguous == 0
    assert result.skipped_unqualifiable == 1


def test_star_tables_returns_qualified_relations():
    workload = Workload(
        stats=(
            QueryStat(
                fingerprint="fp",
                sql="select * from items",
                calls=1,
                total_time_ms=1.0,
                flags=frozenset({FLAG_SELECT_STAR}),
            ),
        ),
        window_description="test",
    )
    assert star_tables(workload, TWO_SCHEMAS) == frozenset({Relation("staging", "items")})


def test_star_tables_skips_an_ambiguous_name():
    """Attributing a bare `select *` to one of two same-named tables would be a guess."""
    workload = Workload(
        stats=(
            QueryStat(
                fingerprint="fp",
                sql="select * from orders",
                calls=1,
                total_time_ms=1.0,
                flags=frozenset({FLAG_SELECT_STAR}),
            ),
        ),
        window_description="test",
    )
    assert star_tables(workload, COLLIDING) == frozenset()


def test_star_tables_does_not_attribute_an_unintrospected_schema_qualifier():
    """`star_tables` must decline exactly what `resolve_relation` declines.

    Text-matching `nosuch.items` against the schema's table names cannot see the
    qualifier at all, so it would previously resolve through a bare-name collision with
    `staging.items` — the phantom `resolve_relation`'s `table.db` guard exists to refuse.
    """
    workload = Workload(
        stats=(
            QueryStat(
                fingerprint="fp",
                sql="select * from nosuch.items",
                calls=1,
                total_time_ms=1.0,
                flags=frozenset({FLAG_SELECT_STAR}),
            ),
        ),
        window_description="test",
    )
    assert star_tables(workload, TWO_SCHEMAS) == frozenset()


def test_star_tables_resolves_an_explicitly_qualified_colliding_name():
    """`orders` collides across two schemas, but an explicit qualifier is not ambiguous.

    `resolve_relation` resolves `sales.orders` outright; `star_tables`'s old text-match
    path could not see the qualifier and dropped it as if the query had said `orders`
    bare. The two must agree.
    """
    workload = Workload(
        stats=(
            QueryStat(
                fingerprint="fp",
                sql="select * from sales.orders",
                calls=1,
                total_time_ms=1.0,
                flags=frozenset({FLAG_SELECT_STAR}),
            ),
        ),
        window_description="test",
    )
    assert star_tables(workload, COLLIDING) == frozenset({Relation("sales", "orders")})


def test_relation_breaks_ties_when_cost_column_and_role_all_match():
    """The fifth sort key. Two relations, same column/role/cost — order must be
    canonical (by `Relation`), not the order the statements happened to arrive in.

    ``alpha`` sorts before ``zeta``, but the ``zeta`` statement is listed — and therefore
    processed — first, so this only holds if the sort key actually includes `u.relation`.
    """
    schema = {
        "zeta": {"orders": {"id": "int", "status": "text"}},
        "alpha": {"orders": {"id": "int", "status": "text"}},
    }
    agg = aggregate(
        _mixed_workload(
            "select id from zeta.orders where status = 'x'",
            "select id from alpha.orders where status = 'y'",
        ),
        schema,
        "postgres",
    )
    matches = [u for u in agg.usage if u.column == "status"]
    assert [u.relation for u in matches] == [
        Relation("alpha", "orders"),
        Relation("zeta", "orders"),
    ]


def test_ambiguous_dml_target_is_counted_not_silently_dropped():
    """`qualify()` does not validate UPDATE/DELETE targets, so an ambiguous bare DML
    target used to vanish with no usage recorded and neither counter incremented —
    reported as analysed by the coverage line when it was not.
    """
    result = aggregate(
        _mixed_workload("update orders set status = 'x' where id = 1"), COLLIDING, "postgres"
    )
    assert result.skipped_ambiguous == 1
    assert result.usage == ()


def test_ambiguous_statement_cost_stays_in_the_denominator():
    """Same 'not a partition' semantics as an unqualifiable statement: an ambiguous
    statement's cost is not excluded from the denominator merely because it produced no
    usage (see test_skipped_stats_still_count_toward_the_denominator).
    """
    workload = Workload(
        stats=(
            QueryStat(
                fingerprint="fp0",
                sql="select id from orders where status = 'x'",
                calls=1,
                total_time_ms=90.0,
            ),
            QueryStat(
                fingerprint="fp1",
                sql="select id from sales.orders where status = 'y'",
                calls=1,
                total_time_ms=10.0,
            ),
        ),
        window_description="test",
    )
    result = aggregate(workload, COLLIDING, "postgres")
    assert result.skipped_ambiguous == 1
    assert result.total_cost_ms == 100.0
    usage = next(u for u in result.usage if u.column == "status")
    assert usage.cost_share == 0.1
