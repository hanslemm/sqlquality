from pathlib import Path

from sqlquality.models import (
    Aggregation,
    ColumnRole,
    ColumnUsage,
    Confidence,
    Proposal,
    QueryStat,
    Relation,
    TableFacts,
    Workload,
)
from sqlquality.workload.fingerprint import FLAG_LEADING_WILDCARD_LIKE, FLAG_SELECT_STAR
from sqlquality.workload.postgres import (
    PgIndex,
    PostgresWorkloadAdapter,
    _is_fully_commented,
    _quote_ident,
    propose_grouping_indexes,
    propose_indexes,
    propose_join_keys,
    propose_partial_indexes,
    propose_redundant_indexes,
    propose_sargability,
    propose_select_star,
    propose_unused_indexes,
)

#: The relation nearly every test in this file talks about, so each test only has to name
#: a different relation when the point of the test *is* the relation.
_ORDERS = Relation("public", "orders")


def _usage(relation, column, role, cost_share=0.5, cost_ms=50.0, fps=("fp1",)):
    """`fps` defaults to a single shared fingerprint, so usages co-occur unless a test
    deliberately gives them disjoint sets."""
    return ColumnUsage(
        relation=relation,
        column=column,
        role=role,
        calls=10,
        cost_ms=cost_ms,
        cost_share=cost_share,
        fingerprint_ids=frozenset(fps),
    )


def _facts(
    relation, rows=1_000_000, ndv=None, columns=("id", "status", "created_at", "customer_id")
):
    return TableFacts(
        relation=relation, row_estimate=rows, size_bytes=10**8, columns=columns, ndv=ndv or {}
    )


def _facts_map(relation=_ORDERS, **kwargs):
    """The common case: one relation's facts, as the `facts` mapping every rule expects."""
    return {relation: _facts(relation, **kwargs)}


def codes(proposals):
    return [p.code for p in proposals]


def test_equality_then_range_ordering_in_the_candidate_index():
    proposals = propose_indexes(
        [
            _usage(_ORDERS, "status", ColumnRole.EQUALITY, cost_ms=90.0),
            _usage(_ORDERS, "created_at", ColumnRole.RANGE, cost_ms=80.0),
        ],
        _facts_map(),
        {},
        min_cost_share=0.01,
    )
    assert codes(proposals) == ["ADV001"]
    assert proposals[0].evidence["columns"] == ("status", "created_at")


def test_only_one_range_column_is_included():
    proposals = propose_indexes(
        [
            _usage(_ORDERS, "status", ColumnRole.EQUALITY, cost_ms=90.0),
            _usage(_ORDERS, "created_at", ColumnRole.RANGE, cost_ms=80.0),
            _usage(_ORDERS, "shipped_at", ColumnRole.RANGE, cost_ms=70.0),
        ],
        _facts_map(columns=("status", "created_at", "shipped_at")),
        {},
        min_cost_share=0.01,
    )
    assert proposals[0].evidence["columns"] == ("status", "created_at")


def test_arity_is_capped():
    proposals = propose_indexes(
        [
            _usage(_ORDERS, "a", ColumnRole.EQUALITY, cost_ms=99.0),
            _usage(_ORDERS, "b", ColumnRole.EQUALITY, cost_ms=98.0),
            _usage(_ORDERS, "c", ColumnRole.EQUALITY, cost_ms=97.0),
            _usage(_ORDERS, "d", ColumnRole.EQUALITY, cost_ms=96.0),
        ],
        _facts_map(columns=("a", "b", "c", "d")),
        {},
        min_cost_share=0.01,
    )
    assert len(proposals[0].evidence["columns"]) == 3


def test_small_tables_are_suppressed_entirely():
    proposals = propose_indexes(
        [_usage(_ORDERS, "status", ColumnRole.EQUALITY)],
        _facts_map(rows=500),
        {},
        min_cost_share=0.01,
    )
    assert proposals == []


def test_below_min_cost_share_is_suppressed():
    proposals = propose_indexes(
        [_usage(_ORDERS, "status", ColumnRole.EQUALITY, cost_share=0.001)],
        _facts_map(),
        {},
        min_cost_share=0.01,
    )
    assert proposals == []


def test_existing_index_with_the_same_leading_prefix_is_not_reproposed():
    existing = {_ORDERS: (PgIndex("idx", ("status", "created_at"), False, False, 10, 8192),)}
    proposals = propose_indexes(
        [
            _usage(_ORDERS, "status", ColumnRole.EQUALITY, cost_ms=90.0),
            _usage(_ORDERS, "created_at", ColumnRole.RANGE, cost_ms=80.0),
        ],
        _facts_map(),
        existing,
        min_cost_share=0.01,
    )
    assert proposals == []


def test_a_wider_existing_index_still_covers_a_narrower_candidate():
    existing = {_ORDERS: (PgIndex("idx", ("status", "created_at", "id"), False, False, 5, 1),)}
    proposals = propose_indexes(
        [_usage(_ORDERS, "status", ColumnRole.EQUALITY, cost_ms=90.0)],
        _facts_map(),
        existing,
        min_cost_share=0.01,
    )
    assert proposals == []


def test_a_narrower_existing_index_does_not_cover_a_wider_candidate():
    """The reverse of the coverage rule, and the direction that inverting the slice breaks.

    An index on (status) does not serve a candidate (status, created_at). Only the
    wider-covers-narrower test existed, so `candidate[:len(existing)] == existing` would
    have shipped silently.
    """
    existing = {_ORDERS: (PgIndex("idx_status", ("status",), False, False, 5, 1),)}
    proposals = propose_indexes(
        [
            _usage(_ORDERS, "status", ColumnRole.EQUALITY, cost_ms=90.0),
            _usage(_ORDERS, "created_at", ColumnRole.RANGE, cost_ms=80.0),
        ],
        _facts_map(),
        existing,
        min_cost_share=0.01,
    )
    assert codes(proposals) == ["ADV001"]
    assert proposals[0].evidence["columns"] == ("status", "created_at")


def test_a_partial_index_does_not_suppress_a_candidate():
    """`idx ON orders(status) WHERE shipped_at IS NULL` does not serve `WHERE status = $1`.

    Treating it as coverage silently withheld a good proposal — the inverse of the
    confidently-wrong failures, and just as invisible.
    """
    existing = {
        _ORDERS: (
            PgIndex(
                "idx_open",
                ("status",),
                False,
                False,
                5,
                4096,
                is_partial=True,
                predicate="(shipped_at IS NULL)",
            ),
        )
    }
    proposals = propose_indexes(
        [_usage(_ORDERS, "status", ColumnRole.EQUALITY)],
        _facts_map(),
        existing,
        min_cost_share=0.01,
    )
    assert codes(proposals) == ["ADV001"]
    assert proposals[0].evidence["partial_indexes_skipped"] == ("idx_open",)
    assert "partial" in proposals[0].rationale.lower()


def test_a_plain_index_still_suppresses_a_candidate():
    """The control. Task 2's new fields default to False, so this must not have changed."""
    existing = {_ORDERS: (PgIndex("idx_status", ("status",), False, False, 5, 4096),)}
    proposals = propose_indexes(
        [_usage(_ORDERS, "status", ColumnRole.EQUALITY)],
        _facts_map(),
        existing,
        min_cost_share=0.01,
    )
    assert proposals == []


def test_an_expression_index_is_disclosed_not_silently_ignored():
    """We cannot prove `lower(status)` makes an index on `status` redundant — or that it
    doesn't. Saying so beats both suppressing and pretending it isn't there."""
    existing = {
        _ORDERS: (
            PgIndex(
                "idx_lower_status",
                (),
                False,
                False,
                5,
                4096,
                has_expressions=True,
                definition="CREATE INDEX idx_lower_status ON orders (lower(status))",
            ),
        )
    }
    proposals = propose_indexes(
        [_usage(_ORDERS, "status", ColumnRole.EQUALITY)],
        _facts_map(),
        existing,
        min_cost_share=0.01,
    )
    assert codes(proposals) == ["ADV001"]
    assert proposals[0].evidence["expression_indexes"] == ("idx_lower_status",)
    assert "expression" in proposals[0].rationale.lower()


def test_a_mixed_expression_index_sharing_a_prefix_does_not_count_as_coverage():
    """The `has_expressions` half of `_covered`, which every other test leaves unexercised.

    Those tests all build `columns=()`, where `_is_prefix(candidate, ())` is already False —
    so the guard can never be the reason they pass, and deleting `or index.has_expressions`
    left the whole suite green. This is the shape that needs it, and it is ordinary:

        CREATE INDEX idx_mixed ON orders (lower(note), status)

    `indkey` is `[0, status_attnum]`; the expression position at ordinality 1 yields a NULL
    attname and is dropped, so the tuple sqlquality reconstructs is `("status",)` — position
    1 is *lost*. `_is_prefix(("status",), ("status",))` is then True and the index reads as
    coverage, silently withholding a genuine HIGH-confidence ADV001. The real index leads
    with `lower(note)` and cannot serve a bare `status` lookup.
    """
    existing = {
        _ORDERS: (
            PgIndex(
                "idx_mixed",
                # Non-empty on purpose: the reconstructed tuple from a mixed index, with the
                # leading expression position missing. This is what the catalog query yields.
                ("status",),
                False,
                False,
                5,
                4096,
                has_expressions=True,
                definition="CREATE INDEX idx_mixed ON orders (lower(note), status)",
            ),
        )
    }
    proposals = propose_indexes(
        [_usage(_ORDERS, "status", ColumnRole.EQUALITY)],
        _facts_map(ndv={"status": 500.0}),
        existing,
        min_cost_share=0.01,
    )
    assert codes(proposals) == ["ADV001"]
    assert proposals[0].confidence is Confidence.HIGH
    assert proposals[0].evidence["expression_indexes"] == ("idx_mixed",)
    assert "expression index" in proposals[0].rationale.lower()


def test_an_expression_index_not_mentioning_the_column_is_not_disclosed():
    """Only expression indexes that plausibly relate to the candidate are worth naming."""
    existing = {
        _ORDERS: (
            PgIndex(
                "idx_lower_note",
                (),
                False,
                False,
                5,
                4096,
                has_expressions=True,
                definition="CREATE INDEX idx_lower_note ON orders (lower(note))",
            ),
        )
    }
    proposals = propose_indexes(
        [_usage(_ORDERS, "status", ColumnRole.EQUALITY)],
        _facts_map(),
        existing,
        min_cost_share=0.01,
    )
    assert proposals[0].evidence["expression_indexes"] == ()
    assert "expression index" not in proposals[0].rationale.lower()


def test_a_column_name_inside_a_longer_identifier_is_not_disclosed():
    """`id` is a substring of `guid`, and a substring test said so out loud.

    The rationale would have told the operator an index "mentions id" when it indexes
    `lower(guid)` — a false claim in the text someone reads before running DDL.
    """
    existing = {
        _ORDERS: (
            PgIndex(
                "idx_lower_guid",
                (),
                False,
                False,
                5,
                4096,
                has_expressions=True,
                definition="CREATE INDEX idx_lower_guid ON orders (lower(guid))",
            ),
        )
    }
    proposals = propose_indexes(
        [_usage(_ORDERS, "id", ColumnRole.EQUALITY)],
        _facts_map(columns=("id", "guid")),
        existing,
        min_cost_share=0.01,
    )
    assert proposals[0].evidence["expression_indexes"] == ()


def test_an_expression_index_on_a_cast_of_the_column_is_still_disclosed():
    """The control for the fix: word boundaries must not cost a true positive."""
    existing = {
        _ORDERS: (
            PgIndex(
                "idx_status_cast",
                (),
                False,
                False,
                5,
                4096,
                has_expressions=True,
                definition="CREATE INDEX idx_status_cast ON orders ((status::text))",
            ),
        )
    }
    proposals = propose_indexes(
        [_usage(_ORDERS, "status", ColumnRole.EQUALITY)],
        _facts_map(),
        existing,
        min_cost_share=0.01,
    )
    assert proposals[0].evidence["expression_indexes"] == ("idx_status_cast",)


def test_arity_cap_keeps_the_range_column_last_when_it_bites():
    """The interaction of the two most important ordering rules, previously untested.

    Four equality columns plus a range column with max_arity 3 must drop the weakest
    equality column, not the range column, and the range column must still come last —
    once a range predicate is used, later columns cannot be probed by equality.
    """
    proposals = propose_indexes(
        [
            _usage(_ORDERS, "a", ColumnRole.EQUALITY, cost_ms=99.0),
            _usage(_ORDERS, "b", ColumnRole.EQUALITY, cost_ms=98.0),
            _usage(_ORDERS, "c", ColumnRole.EQUALITY, cost_ms=97.0),
            _usage(_ORDERS, "d", ColumnRole.EQUALITY, cost_ms=96.0),
            _usage(_ORDERS, "e", ColumnRole.RANGE, cost_ms=50.0),
        ],
        _facts_map(columns=("a", "b", "c", "d", "e")),
        {},
        min_cost_share=0.01,
    )
    columns = proposals[0].evidence["columns"]
    assert columns == ("a", "b", "e")
    assert len(columns) == 3


def test_a_column_used_in_two_roles_is_not_proposed_twice():
    """`where id = $1` plus `order by id desc` puts `id` in both role lists.

    Concatenating them yielded `(id, id)`: invalid DDL, and unmatchable by _covered —
    `_is_prefix(("id","id"), ("id",))` is False, so even the primary key could not suppress
    it. Two ordinary queries were enough to reproduce it at HIGH confidence.
    """
    proposals = propose_indexes(
        [
            _usage(_ORDERS, "id", ColumnRole.EQUALITY, cost_ms=90.0),
            _usage(_ORDERS, "id", ColumnRole.SORT, cost_ms=80.0),
        ],
        _facts_map(ndv={"id": 5000.0}),
        {},
        min_cost_share=0.01,
    )
    columns = proposals[0].evidence["columns"]
    assert columns == ("id",)
    assert len(set(columns)) == len(columns), f"repeated column in {columns}"
    assert proposals[0].ddl == 'CREATE INDEX ON "public"."orders" ("id");'


def test_the_existing_primary_key_suppresses_the_deduplicated_candidate():
    """The consequence of the dedupe, and the reason the bug mattered.

    Once `(id, id)` collapses to `(id,)`, `orders_pkey(id)` covers it and there is no
    proposal at all — which is the correct answer for this workload.
    """
    existing = {_ORDERS: (PgIndex("orders_pkey", ("id",), True, True, 900, 4096),)}
    proposals = propose_indexes(
        [
            _usage(_ORDERS, "id", ColumnRole.EQUALITY, cost_ms=90.0),
            _usage(_ORDERS, "id", ColumnRole.SORT, cost_ms=80.0),
        ],
        _facts_map(ndv={"id": 5000.0}),
        existing,
        min_cost_share=0.01,
    )
    assert proposals == []


def test_dedupe_prefers_the_equality_occurrence_of_a_two_role_column():
    """Order matters: the surviving entry must be the equality one.

    `roles` is reported as evidence, and an index built for an equality probe on a column
    that is also sorted on is described honestly only if the equality role is the one kept.
    """
    proposals = propose_indexes(
        [
            _usage(_ORDERS, "status", ColumnRole.EQUALITY, cost_ms=99.0),
            _usage(_ORDERS, "created_at", ColumnRole.EQUALITY, cost_ms=90.0),
            _usage(_ORDERS, "created_at", ColumnRole.RANGE, cost_ms=95.0),
        ],
        _facts_map(ndv={"status": 5000.0}),
        {},
        min_cost_share=0.01,
    )
    assert proposals[0].evidence["columns"] == ("status", "created_at")
    assert proposals[0].evidence["roles"] == ("equality", "equality")


def test_unknown_row_count_is_low_confidence_and_says_why():
    """The small-table gate cannot run without a row count.

    Suppressing entirely would deny advice to anyone whose row-count grant is missing; the
    cost evidence is real. But reporting MEDIUM would present an unverified proposal as
    ordinarily-trustworthy, so it is LOW and the rationale states the gap.
    """
    proposals = propose_indexes(
        [_usage(_ORDERS, "status", ColumnRole.EQUALITY)],
        _facts_map(rows=None, ndv={"status": 9999.0}),
        {},
        min_cost_share=0.01,
    )
    assert codes(proposals) == ["ADV001"]
    assert proposals[0].confidence is Confidence.LOW
    assert proposals[0].evidence["row_estimate"] is None
    assert "unknown" in proposals[0].rationale.lower()


def test_a_denied_index_list_caps_confidence_at_low_and_stops_claiming_coverage():
    """ "No existing index leads with them" is unknowable when pg_index was denied.

    Absent evidence is not proof of absence — the same mistake the row_estimate branch
    exists to prevent, three lines away.
    """
    proposals = propose_indexes(
        [_usage(_ORDERS, "status", ColumnRole.EQUALITY)],
        _facts_map(ndv={"status": 5000.0}),
        {},
        min_cost_share=0.01,
        have_index_data=False,
    )
    assert codes(proposals) == ["ADV001"]
    assert proposals[0].confidence is Confidence.LOW
    rationale = proposals[0].rationale
    assert "no existing index leads with them" not in rationale
    assert "could not be read" in rationale


def test_a_readable_index_list_still_reaches_high():
    """The other half of the branch: the cap must not fire when the evidence is there."""
    proposals = propose_indexes(
        [_usage(_ORDERS, "status", ColumnRole.EQUALITY)],
        _facts_map(ndv={"status": 5000.0}),
        {},
        min_cost_share=0.01,
        have_index_data=True,
    )
    assert proposals[0].confidence is Confidence.HIGH
    assert "no existing index leads with them" in proposals[0].rationale


def test_propose_lowers_confidence_when_the_indexes_capability_degraded():
    """The adapter is what knows pg_index was denied, so it must pass that down."""

    def denied(sql, params):
        if "pg_index" in sql:
            raise RuntimeError("permission denied for relation pg_index")
        return []

    aggregation = Aggregation(
        usage=(_usage(_ORDERS, "status", ColumnRole.EQUALITY, cost_share=0.6, cost_ms=60.0),),
        total_cost_ms=100.0,
        skipped_unqualifiable=0,
        tables=frozenset({_ORDERS}),
    )
    adapter = PostgresWorkloadAdapter(querier=denied)
    proposals = adapter.propose(
        aggregation, _facts_map(ndv={"status": 9999.0}), _workload(), min_cost_share=0.01
    )
    adv001 = [p for p in proposals if p.code == "ADV001"]
    assert adv001, "the proposal must survive a missing grant, just at lower confidence"
    assert adv001[0].confidence is Confidence.LOW
    assert "could not be read" in adv001[0].rationale


def test_cost_share_is_the_max_never_the_sum():
    proposals = propose_indexes(
        [
            _usage(_ORDERS, "status", ColumnRole.EQUALITY, cost_share=0.6, cost_ms=90.0),
            _usage(_ORDERS, "created_at", ColumnRole.RANGE, cost_share=0.6, cost_ms=80.0),
        ],
        _facts_map(),
        {},
        min_cost_share=0.01,
    )
    assert proposals[0].evidence["cost_share"] == 0.6


def test_adv001_does_not_weld_in_a_column_no_query_uses_with_the_others():
    """The measured cross-task regression: a near-free cursor read degrading the hot index.

    `DECLARE ... CURSOR FOR SELECT ... WHERE tenant_id = $1` became analysable, and at 0.003%
    of window cost it put `tenant_id` into position 2 of the hot query's
    `(customer_id, created_at)`. `cost_share` cannot filter that out — it is the *max* over
    the chosen columns — and `(customer_id, tenant_id, created_at)` no longer satisfies the
    hot query's `ORDER BY created_at`, so the tool proposed a strictly worse index at HIGH.
    """
    proposals = propose_indexes(
        [
            _usage(_ORDERS, "customer_id", ColumnRole.EQUALITY, cost_ms=5000.0, fps=("hot",)),
            _usage(_ORDERS, "tenant_id", ColumnRole.EQUALITY, cost_ms=0.4, fps=("cursor",)),
            _usage(_ORDERS, "created_at", ColumnRole.SORT, cost_ms=5000.0, fps=("hot",)),
        ],
        _facts_map(columns=("customer_id", "tenant_id", "created_at")),
        {},
        min_cost_share=0.01,
    )
    assert proposals[0].evidence["columns"] == ("customer_id", "created_at")


def test_adv001_requires_joint_support_not_merely_pairwise_with_the_seed():
    """Transitive support is not support: `a` with `b` in one query and with `c` in another
    is no query filtering on all three. Same guard, same reason, as ADV008's."""
    proposals = propose_indexes(
        [
            _usage(_ORDERS, "a", ColumnRole.EQUALITY, cost_ms=99.0, fps=("fp1", "fp2")),
            _usage(_ORDERS, "b", ColumnRole.EQUALITY, cost_ms=98.0, fps=("fp1",)),
            _usage(_ORDERS, "c", ColumnRole.EQUALITY, cost_ms=97.0, fps=("fp2",)),
        ],
        _facts_map(columns=("a", "b", "c")),
        {},
        min_cost_share=0.01,
    )
    assert proposals[0].evidence["columns"] == ("a", "b")


def test_adv001_rejecting_one_column_does_not_block_a_later_co_occurring_one():
    """A rejected candidate must not narrow the running intersection, or the rule would
    silently drop a column that a query really does use alongside the seed."""
    proposals = propose_indexes(
        [
            _usage(_ORDERS, "a", ColumnRole.EQUALITY, cost_ms=99.0, fps=("fp1",)),
            _usage(_ORDERS, "b", ColumnRole.EQUALITY, cost_ms=98.0, fps=("fp2",)),
            _usage(_ORDERS, "c", ColumnRole.EQUALITY, cost_ms=97.0, fps=("fp1",)),
        ],
        _facts_map(columns=("a", "b", "c")),
        {},
        min_cost_share=0.01,
    )
    assert proposals[0].evidence["columns"] == ("a", "c")


def test_adv001_reports_the_honest_joint_support_count_and_omits_the_plain_one():
    """`fingerprints` was `max(per-column)`, so a three-column composite that no single
    query group supports reported `fingerprints: 1` — a number that reads as corroboration.
    Only the joint count is reported now, under ADV004's and ADV008's existing key."""
    proposals = propose_indexes(
        [
            _usage(_ORDERS, "status", ColumnRole.EQUALITY, cost_ms=90.0, fps=("fp1", "fp2")),
            _usage(_ORDERS, "created_at", ColumnRole.RANGE, cost_ms=80.0, fps=("fp1",)),
        ],
        _facts_map(),
        {},
        min_cost_share=0.01,
    )
    evidence = proposals[0].evidence
    assert evidence["columns"] == ("status", "created_at")
    assert evidence["co_occurring_fingerprints"] == 1
    assert "fingerprints" not in evidence


def test_confidence_is_high_only_with_stats_and_a_selective_leading_column():
    high = propose_indexes(
        [_usage(_ORDERS, "status", ColumnRole.EQUALITY)],
        _facts_map(ndv={"status": 5000.0}),
        {},
        min_cost_share=0.01,
    )
    assert high[0].confidence is Confidence.HIGH

    no_stats = propose_indexes(
        [_usage(_ORDERS, "status", ColumnRole.EQUALITY)],
        _facts_map(ndv={}),
        {},
        min_cost_share=0.01,
    )
    assert no_stats[0].confidence is Confidence.MEDIUM

    unselective = propose_indexes(
        [_usage(_ORDERS, "status", ColumnRole.EQUALITY)],
        _facts_map(ndv={"status": 3.0}),
        {},
        min_cost_share=0.01,
    )
    assert unselective[0].confidence is Confidence.LOW


def test_unused_index_proposed_for_drop_but_never_a_constraint_index():
    existing = {
        _ORDERS: (
            PgIndex("idx_cold", ("note",), False, False, 0, 4096),
            PgIndex("orders_pkey", ("id",), True, True, 0, 4096),
            PgIndex("uq_email", ("email",), True, False, 0, 4096),
            PgIndex("idx_warm", ("status",), False, False, 42, 4096),
        )
    }
    proposals = propose_unused_indexes(existing, hot_tables=frozenset({_ORDERS}))
    assert codes(proposals) == ["ADV002"]
    assert proposals[0].evidence["index"] == "idx_cold"
    assert proposals[0].confidence is Confidence.MEDIUM


def test_unused_index_rule_ignores_tables_outside_the_workload():
    existing = {Relation("public", "archive"): (PgIndex("idx_cold", ("a",), False, False, 0, 1),)}
    assert propose_unused_indexes(existing, hot_tables=frozenset({_ORDERS})) == []


def test_adv002_drop_ddl_qualifies_the_index_with_its_relations_schema():
    existing = {
        Relation("staging", "orders"): (
            PgIndex(
                name="idx_cold",
                columns=("note",),
                is_unique=False,
                is_primary=False,
                scans=0,
                size_bytes=1,
            ),
        )
    }
    proposals = propose_unused_indexes(
        existing, hot_tables=frozenset({Relation("staging", "orders")})
    )
    assert proposals[0].ddl == 'DROP INDEX "staging"."idx_cold";'


def test_unused_index_ddl_uses_its_own_relations_schema_not_the_others():
    """`orders_pkey`/any index name can exist identically in two schemas at once (proven
    live: `orders_pkey` exists under both `public` and `staging`). Grouping by relation must
    not let one schema's index bleed into the other's DROP statement."""
    existing = {
        Relation("sales", "orders"): (
            PgIndex(
                name="idx_cold",
                columns=("note",),
                is_unique=False,
                is_primary=False,
                scans=0,
                size_bytes=1,
            ),
        ),
        Relation("staging", "orders"): (
            PgIndex(
                name="idx_cold",
                columns=("note",),
                is_unique=False,
                is_primary=False,
                scans=0,
                size_bytes=1,
            ),
        ),
    }
    proposals = propose_unused_indexes(
        existing,
        hot_tables=frozenset({Relation("sales", "orders"), Relation("staging", "orders")}),
    )
    ddls = {p.ddl for p in proposals}
    assert ddls == {
        'DROP INDEX "sales"."idx_cold";',
        'DROP INDEX "staging"."idx_cold";',
    }


def test_a_plain_redundant_pair_is_high_confidence():
    existing = {
        _ORDERS: (
            PgIndex("idx_narrow", ("status",), False, False, 5, 1),
            PgIndex("idx_wide", ("status", "created_at"), False, False, 5, 1),
        )
    }
    proposals = propose_redundant_indexes(existing, hot_tables=frozenset(existing))
    assert codes(proposals) == ["ADV003"]
    assert proposals[0].confidence is Confidence.HIGH
    assert proposals[0].evidence["index"] == "idx_narrow"
    # Pin the claim, not just the confidence. Deleting the old MEDIUM test removed the only
    # assertion on this rationale's wording, so a future edit could reintroduce a hedge, or
    # drop the "both are plain" claim while leaving HIGH, with nothing failing.
    assert "plain" in proposals[0].rationale
    assert "partial" not in proposals[0].rationale


def test_adv003_ignores_relations_the_workload_never_touched():
    """Scope, for a rule whose output is `DROP INDEX`, must not be an accident.

    `fetch_indexes` filters tables by *bare* name, so `schemas=("sales", "staging")` with
    only `sales.orders` in the workload still returns `staging.orders`'s indexes. Iterating
    every key of `existing` therefore emitted `DROP INDEX "staging"."idx_narrow"` for a
    relation with zero recorded column usage — and whether it did depended on whether a
    bare name happened to collide across two requested schemas. ADV002 was already scoped to
    `hot_tables`; this asserts both members, so scoping that silently dropped the hot
    relation as well would fail here rather than look like a pass.
    """
    sales, staging = Relation("sales", "orders"), Relation("staging", "orders")
    redundant_pair = (
        PgIndex("idx_narrow", ("status",), False, False, 5, 1),
        PgIndex("idx_wide", ("status", "created_at"), False, False, 5, 1),
    )
    proposals = propose_redundant_indexes(
        {sales: redundant_pair, staging: redundant_pair},
        hot_tables=frozenset({sales}),
    )
    assert {p.ddl for p in proposals} == {'DROP INDEX "sales"."idx_narrow";'}


def test_a_partial_narrow_index_is_never_called_redundant():
    """The partial index exists to serve a subset; the wider full index serves it
    differently. Dropping it is not less certain, it is probably wrong."""
    existing = {
        _ORDERS: (
            PgIndex(
                "idx_open",
                ("status",),
                False,
                False,
                5,
                1,
                is_partial=True,
                predicate="(shipped_at IS NULL)",
            ),
            PgIndex("idx_wide", ("status", "created_at"), False, False, 5, 1),
        )
    }
    assert propose_redundant_indexes(existing, hot_tables=frozenset(existing)) == []


def test_a_partial_wider_index_does_not_supersede_a_plain_one():
    existing = {
        _ORDERS: (
            PgIndex("idx_narrow", ("status",), False, False, 5, 1),
            PgIndex(
                "idx_wide_open",
                ("status", "created_at"),
                False,
                False,
                5,
                1,
                is_partial=True,
                predicate="(shipped_at IS NULL)",
            ),
        )
    }
    assert propose_redundant_indexes(existing, hot_tables=frozenset(existing)) == []


def test_a_wider_expression_index_does_not_supersede_a_plain_one():
    """The wider index must be strictly wider, or the length guard skips the pair anyway.

    An earlier version of this test gave both indexes one column, so
    `len(other.columns) > len(narrow.columns)` was already False and it passed whether or
    not the has_expressions filter existed at all.
    """
    existing = {
        _ORDERS: (
            PgIndex("idx_narrow", ("status",), False, False, 5, 1),
            PgIndex(
                "idx_expr",
                ("status", "note"),
                False,
                False,
                5,
                1,
                has_expressions=True,
                definition="CREATE INDEX idx_expr ON orders (status, note, lower(note))",
            ),
        )
    }
    assert propose_redundant_indexes(existing, hot_tables=frozenset(existing)) == []


def test_a_narrow_expression_index_is_never_called_redundant():
    """The other direction, and the reason it matters.

    `columns` understates an expression index — the expression positions contribute no
    name — so a narrow one may index something the wider one does not. Dropping it on a
    column-list comparison would discard an index nothing else provides.
    """
    existing = {
        _ORDERS: (
            PgIndex(
                "idx_narrow_expr",
                ("status",),
                False,
                False,
                5,
                1,
                has_expressions=True,
                definition="CREATE INDEX idx_narrow_expr ON orders (status, lower(note))",
            ),
            PgIndex("idx_wide", ("status", "created_at"), False, False, 5, 1),
        )
    }
    assert propose_redundant_indexes(existing, hot_tables=frozenset(existing)) == []


def test_a_unique_prefix_index_is_never_called_redundant():
    existing = {
        _ORDERS: (
            PgIndex("uq_status", ("status",), True, False, 5, 1),
            PgIndex("idx_wide", ("status", "created_at"), False, False, 5, 1),
        )
    }
    assert propose_redundant_indexes(existing, hot_tables=frozenset(existing)) == []


def test_redundant_index_ddl_uses_its_own_relations_schema_not_the_others():
    """Same collision as ADV002's, for ADV003: `idx_narrow`/`idx_wide` named identically in
    two schemas must each produce a DROP scoped to their own schema."""
    existing = {
        Relation("sales", "orders"): (
            PgIndex("idx_narrow", ("status",), False, False, 5, 1),
            PgIndex("idx_wide", ("status", "created_at"), False, False, 5, 1),
        ),
        Relation("staging", "orders"): (
            PgIndex("idx_narrow", ("status",), False, False, 5, 1),
            PgIndex("idx_wide", ("status", "created_at"), False, False, 5, 1),
        ),
    }
    proposals = propose_redundant_indexes(existing, hot_tables=frozenset(existing))
    ddls = {p.ddl for p in proposals}
    assert ddls == {
        'DROP INDEX "sales"."idx_narrow";',
        'DROP INDEX "staging"."idx_narrow";',
    }


def _workload(*stats):
    return Workload(stats=tuple(stats), window_description="w")


def test_partial_index_proposed_for_a_hot_not_null_check():
    proposals = propose_partial_indexes(
        [
            _usage(_ORDERS, "status", ColumnRole.EQUALITY, cost_ms=90.0),
            _usage(_ORDERS, "shipped_at", ColumnRole.NOT_NULL_CHECK, cost_share=0.4, cost_ms=40.0),
        ],
        _facts_map(),
        min_cost_share=0.01,
    )
    assert codes(proposals) == ["ADV004"]
    assert "IS NOT NULL" in proposals[0].ddl


def test_partial_index_polarity_follows_the_predicate():
    proposals = propose_partial_indexes(
        [
            _usage(_ORDERS, "status", ColumnRole.EQUALITY, cost_ms=90.0),
            _usage(_ORDERS, "shipped_at", ColumnRole.NULL_CHECK, cost_share=0.4, cost_ms=40.0),
        ],
        _facts_map(),
        min_cost_share=0.01,
    )
    assert "IS NULL" in proposals[0].ddl
    assert "IS NOT NULL" not in proposals[0].ddl


def test_no_partial_index_when_the_columns_never_co_occur():
    """Two independently hot columns are not evidence for a partial index.

    If query A filters `status = $1` and query B checks `shipped_at IS NOT NULL`, a partial
    index on status guarded by shipped_at helps neither — A does not satisfy the guard and
    B does not use the indexed column. It only costs writes.
    """
    proposals = propose_partial_indexes(
        [
            _usage(_ORDERS, "status", ColumnRole.EQUALITY, cost_ms=90.0, fps=("fp_a",)),
            _usage(
                _ORDERS,
                "shipped_at",
                ColumnRole.NOT_NULL_CHECK,
                cost_share=0.4,
                cost_ms=40.0,
                fps=("fp_b",),
            ),
        ],
        _facts_map(),
        min_cost_share=0.01,
    )
    assert proposals == []


def test_partial_index_reports_the_co_occurrence_that_justifies_it():
    proposals = propose_partial_indexes(
        [
            _usage(_ORDERS, "status", ColumnRole.EQUALITY, cost_ms=90.0, fps=("fp_a", "fp_b")),
            _usage(
                _ORDERS,
                "shipped_at",
                ColumnRole.NOT_NULL_CHECK,
                cost_share=0.4,
                cost_ms=40.0,
                fps=("fp_b", "fp_c"),
            ),
        ],
        _facts_map(),
        min_cost_share=0.01,
    )
    assert codes(proposals) == ["ADV004"]
    assert proposals[0].evidence["co_occurring_fingerprints"] == 1


def test_partial_index_picks_the_costliest_pair_that_actually_co_occurs():
    """A cheaper pair that co-occurs beats a hotter pair that does not."""
    proposals = propose_partial_indexes(
        [
            _usage(_ORDERS, "status", ColumnRole.EQUALITY, cost_ms=99.0, fps=("fp_lonely",)),
            _usage(_ORDERS, "region", ColumnRole.EQUALITY, cost_ms=50.0, fps=("fp_shared",)),
            _usage(
                _ORDERS,
                "shipped_at",
                ColumnRole.NOT_NULL_CHECK,
                cost_share=0.4,
                cost_ms=40.0,
                fps=("fp_shared",),
            ),
        ],
        _facts_map(columns=("status", "region", "shipped_at")),
        min_cost_share=0.01,
    )
    assert proposals[0].evidence["columns"] == ("region",)


def test_partial_index_is_suppressed_on_a_small_table():
    """Two index-creation rules, one small-table gate — ADV004 had none.

    On a 50-row table it emitted ADV004 medium with `evidence.row_estimate: 50` while
    ADV001 correctly suppressed. An index on a table that fits in a page is pure write
    overhead whichever rule proposes it.
    """
    proposals = propose_partial_indexes(
        [
            _usage(_ORDERS, "status", ColumnRole.EQUALITY, cost_ms=90.0),
            _usage(_ORDERS, "shipped_at", ColumnRole.NOT_NULL_CHECK, cost_share=0.4, cost_ms=40.0),
        ],
        _facts_map(rows=50),
        min_cost_share=0.01,
    )
    assert proposals == []


def test_partial_index_with_an_unknown_row_count_is_low_and_says_why():
    """Same treatment ADV001 gives an unknown row count: keep the advice, lower the claim."""
    proposals = propose_partial_indexes(
        [
            _usage(_ORDERS, "status", ColumnRole.EQUALITY, cost_ms=90.0),
            _usage(_ORDERS, "shipped_at", ColumnRole.NOT_NULL_CHECK, cost_share=0.4, cost_ms=40.0),
        ],
        _facts_map(rows=None),
        min_cost_share=0.01,
    )
    assert codes(proposals) == ["ADV004"]
    assert proposals[0].confidence is Confidence.LOW
    assert proposals[0].evidence["row_estimate"] is None
    assert "unknown" in proposals[0].rationale.lower()


def test_no_partial_index_without_an_equality_column_to_index():
    proposals = propose_partial_indexes(
        [_usage(_ORDERS, "shipped_at", ColumnRole.NOT_NULL_CHECK, cost_share=0.4)],
        _facts_map(),
        min_cost_share=0.01,
    )
    assert proposals == []


def test_non_sargable_column_gets_an_attributed_proposal():
    proposals = propose_sargability(
        [_usage(_ORDERS, "status", ColumnRole.NON_SARGABLE, cost_share=0.3)],
        _workload(),
        min_cost_share=0.01,
    )
    assert codes(proposals) == ["ADV005"]
    assert proposals[0].evidence["column"] == "status"
    assert proposals[0].evidence["schema"] == "public"
    assert proposals[0].evidence["table"] == "orders"
    assert proposals[0].confidence is Confidence.HIGH


def test_leading_wildcard_reports_without_column_attribution():
    stat = QueryStat(
        fingerprint="fp",
        sql="select id from orders where note like $1",
        calls=5,
        total_time_ms=100.0,
        flags=frozenset({FLAG_LEADING_WILDCARD_LIKE}),
    )
    proposals = propose_sargability([], _workload(stat), min_cost_share=0.01)
    assert codes(proposals) == ["ADV005"]
    assert proposals[0].evidence.get("column") is None
    # Redaction erased the pattern, so we can name the query group but not the column.
    assert proposals[0].confidence is Confidence.MEDIUM


def test_hot_select_star_on_a_wide_table():
    stat = QueryStat(
        fingerprint="fp",
        sql="select * from orders",
        calls=5,
        total_time_ms=100.0,
        flags=frozenset({FLAG_SELECT_STAR}),
    )
    wide = {
        _ORDERS: TableFacts(
            relation=_ORDERS,
            row_estimate=10**6,
            size_bytes=10**8,
            columns=tuple(f"c{i}" for i in range(30)),
        )
    }
    proposals = propose_select_star(_workload(stat), wide, min_cost_share=0.01)
    assert codes(proposals) == ["ADV006"]


def test_select_star_ignored_on_a_narrow_table():
    stat = QueryStat(
        fingerprint="fp",
        sql="select * from orders",
        calls=5,
        total_time_ms=100.0,
        flags=frozenset({FLAG_SELECT_STAR}),
    )
    narrow = {
        _ORDERS: TableFacts(relation=_ORDERS, row_estimate=10**6, size_bytes=1, columns=("a", "b"))
    }
    assert propose_select_star(_workload(stat), narrow, min_cost_share=0.01) == []


def test_select_star_table_matching_ignores_substring_false_positives():
    """A plain `name in sql` test would misfire three ways this locks out.

    A wide table `order` must not match a query on `orders` (name is a substring of a
    different table); a wide table `orders` must not match only because it appears inside
    a column alias `orders_total`; and a wide table `cart` must not match a query that only
    touches `shopping_cart`.
    """
    stat = QueryStat(
        fingerprint="fp",
        sql="select id, count(*) as orders_total from shopping_cart",
        calls=5,
        total_time_ms=100.0,
        flags=frozenset({FLAG_SELECT_STAR}),
    )
    order = Relation("public", "order")
    orders = Relation("public", "orders")
    cart = Relation("public", "cart")
    wide = {
        order: TableFacts(
            relation=order,
            row_estimate=10**6,
            size_bytes=1,
            columns=tuple(f"c{i}" for i in range(30)),
        ),
        orders: TableFacts(
            relation=orders,
            row_estimate=10**6,
            size_bytes=1,
            columns=tuple(f"c{i}" for i in range(30)),
        ),
        cart: TableFacts(
            relation=cart,
            row_estimate=10**6,
            size_bytes=1,
            columns=tuple(f"c{i}" for i in range(30)),
        ),
    }
    assert propose_select_star(_workload(stat), wide, min_cost_share=0.01) == []


def test_select_star_evidence_reports_the_qualified_relation():
    """`touched` is matched against the bare name in the SQL text, but the evidence must
    still surface the schema-qualified name — the bare match is an implementation detail of
    text-matching, not what should be shown to an operator."""
    stat = QueryStat(
        fingerprint="fp",
        sql="select * from orders",
        calls=5,
        total_time_ms=100.0,
        flags=frozenset({FLAG_SELECT_STAR}),
    )
    staging_orders = Relation("staging", "orders")
    wide = {
        staging_orders: TableFacts(
            relation=staging_orders,
            row_estimate=10**6,
            size_bytes=10**8,
            columns=tuple(f"c{i}" for i in range(30)),
        )
    }
    proposals = propose_select_star(_workload(stat), wide, min_cost_share=0.01)
    assert proposals[0].evidence["tables"] == ("staging.orders",)
    assert "staging.orders" in proposals[0].title


def test_select_star_does_not_attribute_a_schema_qualified_statement_to_the_wrong_schema():
    """`select * from public.orders` must report only `public.orders`, even when
    `staging.orders` is also wide and shares the bare name `orders`.

    Bare-name text matching cannot see the schema qualifier the statement itself carries —
    the same defect class Task 2 already fixed once on this branch for `star_tables`. Before
    the fix, a bare-name text match of `orders` against `select * from public.orders` hit *both*
    wide relations, so the evidence named a table (`staging.orders`) the statement never
    referenced at all.
    """
    stat = QueryStat(
        fingerprint="fp",
        sql="select * from public.orders",
        calls=5,
        total_time_ms=100.0,
        flags=frozenset({FLAG_SELECT_STAR}),
    )
    public_orders = Relation("public", "orders")
    staging_orders = Relation("staging", "orders")
    wide = {
        public_orders: TableFacts(
            relation=public_orders,
            row_estimate=10**6,
            size_bytes=10**8,
            columns=tuple(f"c{i}" for i in range(30)),
        ),
        staging_orders: TableFacts(
            relation=staging_orders,
            row_estimate=10**6,
            size_bytes=10**8,
            columns=tuple(f"c{i}" for i in range(30)),
        ),
    }
    proposals = propose_select_star(_workload(stat), wide, min_cost_share=0.01)
    assert proposals[0].evidence["tables"] == ("public.orders",)


def test_select_star_attributes_an_ambiguous_bare_reference_to_neither_wide_relation():
    """A *bare* `select * from orders` with both `public.orders` and `staging.orders` wide
    cannot be attributed to either — the statement itself does not say which. Reporting
    either would be a guess; reporting both repeats the original defect. Dropping it entirely
    is the same cannot-prove-it-so-drop-it policy `resolve_relation`/`star_tables` follow."""
    stat = QueryStat(
        fingerprint="fp",
        sql="select * from orders",
        calls=5,
        total_time_ms=100.0,
        flags=frozenset({FLAG_SELECT_STAR}),
    )
    public_orders = Relation("public", "orders")
    staging_orders = Relation("staging", "orders")
    wide = {
        public_orders: TableFacts(
            relation=public_orders,
            row_estimate=10**6,
            size_bytes=10**8,
            columns=tuple(f"c{i}" for i in range(30)),
        ),
        staging_orders: TableFacts(
            relation=staging_orders,
            row_estimate=10**6,
            size_bytes=10**8,
            columns=tuple(f"c{i}" for i in range(30)),
        ),
    }
    assert propose_select_star(_workload(stat), wide, min_cost_share=0.01) == []


def test_identical_ddl_at_equal_confidence_resolves_by_code_preference():
    """ADV001 and ADV008 can emit byte-identical DDL at the same confidence."""
    ddl = 'CREATE INDEX ON "public"."events" ("tenant_id");'
    adv008 = Proposal(
        code="ADV008",
        title="group",
        rationale="g",
        evidence={"cost_share": 0.5},
        confidence=Confidence.MEDIUM,
        ddl=ddl,
    )
    adv001 = Proposal(
        code="ADV001",
        title="filter",
        rationale="f",
        evidence={"cost_share": 0.5},
        confidence=Confidence.MEDIUM,
        ddl=ddl,
    )
    # Both orderings must pick the same winner, or list order is deciding.
    assert [p.code for p in PostgresWorkloadAdapter._dedupe_by_ddl([adv008, adv001])] == ["ADV001"]
    assert [p.code for p in PostgresWorkloadAdapter._dedupe_by_ddl([adv001, adv008])] == ["ADV001"]


def test_confidence_still_beats_code_preference():
    ddl = 'CREATE INDEX ON "public"."events" ("tenant_id");'
    adv001_low = Proposal(
        code="ADV001",
        title="filter",
        rationale="f",
        evidence={"cost_share": 0.5},
        confidence=Confidence.LOW,
        ddl=ddl,
    )
    adv008_med = Proposal(
        code="ADV008",
        title="group",
        rationale="g",
        evidence={"cost_share": 0.5},
        confidence=Confidence.MEDIUM,
        ddl=ddl,
    )
    assert [p.code for p in PostgresWorkloadAdapter._dedupe_by_ddl([adv001_low, adv008_med])] == [
        "ADV008"
    ]


def test_every_ddl_emitting_code_has_a_preference_rank():
    """A code missing from the map would raise KeyError mid-run, after all the analysis."""
    ddl_codes = {"ADV001", "ADV002", "ADV003", "ADV004", "ADV007", "ADV008"}
    assert ddl_codes <= set(PostgresWorkloadAdapter._CODE_PREFERENCE)


def test_dedupe_folds_the_discarded_proposals_rationale_into_the_survivor():
    """Collapsing on identical DDL must not silently drop the loser's rationale.

    ADV008 does not read NDV at all, so it has no selectivity caveat to lose — but ADV007
    does, and simply keeping "the more confident" text on identical DDL would drop it.
    """
    aggregation = Aggregation(
        usage=(
            _usage(_ORDERS, "tenant_id", ColumnRole.JOIN, cost_share=0.6, cost_ms=60.0),
            _usage(_ORDERS, "tenant_id", ColumnRole.GROUP, cost_share=0.5, cost_ms=50.0),
        ),
        total_cost_ms=100.0,
        skipped_unqualifiable=0,
        tables=frozenset({_ORDERS}),
    )
    adapter = PostgresWorkloadAdapter(querier=lambda sql, params: [])
    proposals = adapter.propose(
        aggregation, _facts_map(ndv={"tenant_id": 3.0}), _workload(), min_cost_share=0.01
    )
    ddl = 'CREATE INDEX ON "public"."orders" ("tenant_id");'
    creates = [p for p in proposals if p.ddl == ddl]
    assert len(creates) == 1
    survivor = creates[0]
    # ADV008 has no NDV to read, so it wins on confidence (MEDIUM beats LOW) — but ADV007's
    # rationale, including the caveat ADV008 could never have stated, must survive with it.
    assert survivor.code == "ADV008"
    assert survivor.confidence is Confidence.MEDIUM
    assert "ADV007" in survivor.rationale
    assert "distinct values" in survivor.rationale


def test_adv001_and_adv007_prefix_collision_collapses_instead_of_shipping_a_pair_adv003_would_flag():
    """ADV001 proposing (customer_id, created_at) and ADV007 proposing (customer_id), both
    HIGH, in the same report would advise creating a pair where the second is a strict
    prefix of the first — exactly what ADV003 flags as redundant on the next run. The
    narrower proposal must collapse into the wider one, not ship alongside it.
    """
    aggregation = Aggregation(
        usage=(
            _usage(_ORDERS, "customer_id", ColumnRole.EQUALITY, cost_share=0.6, cost_ms=60.0),
            _usage(_ORDERS, "created_at", ColumnRole.RANGE, cost_share=0.5, cost_ms=50.0),
            _usage(_ORDERS, "customer_id", ColumnRole.JOIN, cost_share=0.55, cost_ms=55.0),
        ),
        total_cost_ms=100.0,
        skipped_unqualifiable=0,
        tables=frozenset({_ORDERS}),
    )
    adapter = PostgresWorkloadAdapter(querier=lambda sql, params: [])
    proposals = adapter.propose(
        aggregation, _facts_map(ndv={"customer_id": 9999.0}), _workload(), min_cost_share=0.01
    )
    creates = [p for p in proposals if p.ddl and p.ddl.startswith("CREATE INDEX")]
    assert codes(creates) == ["ADV001"]
    assert creates[0].evidence["columns"] == ("customer_id", "created_at")
    assert creates[0].confidence is Confidence.HIGH
    assert "ADV007" in creates[0].rationale


def _plain_index_proposal(code, columns, confidence=Confidence.HIGH, relation=_ORDERS):
    """A minimal `CREATE INDEX` proposal eligible for `_collapse_index_prefixes` /
    `_disclose_column_set_overlaps` — built by hand so prefix-collision tests can control
    exactly which proposals collide, instead of tuning real rule inputs to get there."""
    quoted = ", ".join(f'"{c}"' for c in columns)
    return Proposal(
        code=code,
        title=f"{code} on {relation}({', '.join(columns)})",
        rationale=f"{code} rationale for {columns}.",
        evidence={"schema": relation.schema, "table": relation.table, "columns": columns},
        confidence=confidence,
        ddl=f'CREATE INDEX ON "{relation.schema}"."{relation.table}" ({quoted});',
    )


def test_prefix_collision_collapses_regardless_of_which_rule_appended_first():
    """Determinism: `propose()` hardcodes a call order today, but the collapse must not
    depend on it. Two proposals must be absorbed into the same wider one — with only one
    absorbed proposal, `sorted(absorbed, ...)` inside `_collapse_index_prefixes` is a
    no-op and this test cannot tell a real sort from none at all.
    """
    wide = _plain_index_proposal("ADV001", ("customer_id", "created_at", "region"))
    narrow_one = _plain_index_proposal("ADV007", ("customer_id",))
    narrow_two = _plain_index_proposal("ADV008", ("customer_id", "created_at"), Confidence.MEDIUM)

    forward = PostgresWorkloadAdapter._collapse_index_prefixes([wide, narrow_one, narrow_two])
    backward = PostgresWorkloadAdapter._collapse_index_prefixes([narrow_two, narrow_one, wide])
    shuffled = PostgresWorkloadAdapter._collapse_index_prefixes([narrow_one, wide, narrow_two])

    assert codes(forward) == codes(backward) == codes(shuffled) == ["ADV001"]
    assert forward[0].rationale == backward[0].rationale == shuffled[0].rationale
    # Both absorbed proposals' rationale must actually be present — this is what makes the
    # sort's job non-trivial: two notes, and their concatenation order must not vary.
    assert "ADV007" in forward[0].rationale
    assert "ADV008" in forward[0].rationale


#: Repository root, for the documentation claims the collapse layer is pinned against.
_ROOT = Path(__file__).resolve().parents[1]


def test_the_proposal_collapse_layer_is_documented_for_users():
    """The collapse layer changes what `proposals` contains, so it cannot be internal-only.

    It shipped documented nowhere user-facing, including the two facts an operator or a
    `--json` consumer actually has to know: a rule can fire and contribute no proposal at
    all, and the absorbed proposal's `evidence` is discarded while its rationale is kept.
    Each claim is asserted separately, so documenting one and omitting another cannot pass.
    """
    readme = (_ROOT / "README.md").read_text(encoding="utf-8")
    changelog = (_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert "How overlapping proposals are reconciled" in readme
    assert "Identical DDL collapses to one proposal" in readme
    assert "A narrower index collapses into a wider one" in readme
    assert "Same columns in a different order are both kept" in readme
    assert "A rule can fire and still contribute no proposal" in readme
    assert "discarded, not merged" in readme, "the evidence loss is not disclosed"
    assert "Overlapping proposals are reconciled" in changelog


def test_a_prefix_collapse_does_not_claim_the_absorbed_rule_endorsed_this_index():
    """The absorbed proposal reached a *narrower* index, and the text must say which one.

    "ADV007 reached the same index at high confidence" under `(customer_id, tenant_id,
    created_at)` told the operator ADV007 endorsed a three-column index when ADV007 proposed
    `(customer_id)` — and the borrowed sentences that follow ("Equality columns come first so
    the range column can be scanned last", from an ADV001 absorbed into an ADV008 survivor
    with no equality columns at all) then have no subject they are true of.
    """
    wide = _plain_index_proposal("ADV001", ("customer_id", "tenant_id", "created_at"))
    narrow = _plain_index_proposal("ADV007", ("customer_id",))

    survivor = PostgresWorkloadAdapter._collapse_index_prefixes([wide, narrow])[0]

    assert "reached the same index" not in survivor.rationale
    assert "ADV007 proposed the narrower (customer_id) on the same table" in survivor.rationale
    # The absorbed rationale itself is still carried over, attributed — the collapse must
    # change the attribution, not start discarding text.
    assert "ADV007 rationale" in survivor.rationale


def test_identical_ddl_dedupe_still_says_the_two_rules_reached_the_same_index():
    """The other half of the split: for byte-identical DDL "the same index" is true by
    construction, and must not be reworded into the prefix form, which would tell the
    operator a narrower index was proposed when none was."""
    survivor = _plain_index_proposal("ADV001", ("tenant_id",))
    loser = _plain_index_proposal("ADV007", ("tenant_id",), Confidence.MEDIUM)

    merged = PostgresWorkloadAdapter._dedupe_by_ddl([survivor, loser])

    assert len(merged) == 1
    assert "ADV007 reached the same index at medium confidence" in merged[0].rationale
    assert "narrower" not in merged[0].rationale


def test_adv001_and_adv008_same_column_set_different_order_are_disclosed_not_collapsed():
    """(status, region) and (region, status) are not redundant — different leading columns
    serve different probes — so neither `_collapse_index_prefixes` nor a future ADV003 pass
    can reconcile them. Both must survive, and each must name the other.
    """
    aggregation = Aggregation(
        usage=(
            _usage(_ORDERS, "status", ColumnRole.EQUALITY, cost_share=0.6, cost_ms=90.0),
            _usage(_ORDERS, "region", ColumnRole.EQUALITY, cost_share=0.55, cost_ms=50.0),
            _usage(_ORDERS, "region", ColumnRole.GROUP, cost_share=0.5, cost_ms=90.0),
            _usage(_ORDERS, "status", ColumnRole.GROUP, cost_share=0.45, cost_ms=50.0),
        ),
        total_cost_ms=100.0,
        skipped_unqualifiable=0,
        tables=frozenset({_ORDERS}),
    )
    adapter = PostgresWorkloadAdapter(querier=lambda sql, params: [])
    proposals = adapter.propose(aggregation, _facts_map(), _workload(), min_cost_share=0.01)
    creates = {p.code: p for p in proposals if p.ddl and p.ddl.startswith("CREATE INDEX")}
    assert set(creates) == {"ADV001", "ADV008"}
    assert creates["ADV001"].evidence["columns"] == ("status", "region")
    assert creates["ADV008"].evidence["columns"] == ("region", "status")
    assert "ADV008" in creates["ADV001"].rationale
    assert "ADV001" in creates["ADV008"].rationale


def test_a_partial_index_is_never_collapsed_against_a_plain_prefix():
    """ADV004's WHERE predicate makes it a different object than a plain index over the
    same leading column, even when its single column is a textual prefix of a plain
    composite's — the same reasoning `_covered` applies to catalog indexes must hold here
    for proposals that do not exist as catalog rows yet.
    """
    aggregation = Aggregation(
        usage=(
            _usage(_ORDERS, "status", ColumnRole.EQUALITY, cost_share=0.6, cost_ms=90.0),
            _usage(_ORDERS, "created_at", ColumnRole.RANGE, cost_share=0.5, cost_ms=50.0),
            _usage(_ORDERS, "shipped_at", ColumnRole.NOT_NULL_CHECK, cost_share=0.4, cost_ms=40.0),
        ),
        total_cost_ms=100.0,
        skipped_unqualifiable=0,
        tables=frozenset({_ORDERS}),
    )
    adapter = PostgresWorkloadAdapter(querier=lambda sql, params: [])
    proposals = adapter.propose(aggregation, _facts_map(), _workload(), min_cost_share=0.01)
    assert "ADV004" in codes(proposals)
    adv004 = next(p for p in proposals if p.code == "ADV004")
    assert "ADV001" not in adv004.rationale
    adv001 = next(p for p in proposals if p.code == "ADV001")
    assert "ADV004" not in adv001.rationale


def test_disclose_overlaps_orders_notes_deterministically_with_three_same_set_proposals():
    """With only two proposals sharing a column set the note each gets is symmetric and
    order cannot matter — the bug needs at least three, so a given proposal names more
    than one partner and the join order of those names has something to get wrong.
    """
    p1 = _plain_index_proposal("ADV001", ("a", "b", "c"))
    p2 = _plain_index_proposal("ADV007", ("b", "a", "c"))
    p3 = _plain_index_proposal("ADV008", ("c", "b", "a"))

    def rationale_by_code(order):
        result = PostgresWorkloadAdapter._disclose_column_set_overlaps(list(order))
        return {p.code: p.rationale for p in result}

    forward = rationale_by_code([p1, p2, p3])
    backward = rationale_by_code([p3, p2, p1])
    shuffled = rationale_by_code([p2, p3, p1])

    assert forward == backward == shuffled
    # Sanity: each did get both partners named, not merely "no notes at all" everywhere.
    assert "ADV007" in forward["ADV001"] and "ADV008" in forward["ADV001"]


def test_collapse_index_prefixes_never_crosses_relations():
    """A prefix relationship across two different tables is meaningless, even when the
    columns are identical strings."""
    shipments = Relation("public", "shipments")
    wide = _plain_index_proposal("ADV001", ("a", "b"))
    narrow = _plain_index_proposal("ADV007", ("a",), relation=shipments)

    result = PostgresWorkloadAdapter._collapse_index_prefixes([wide, narrow])

    assert codes(result) == ["ADV001", "ADV007"]
    assert "ADV007" not in result[0].rationale
    assert "ADV001" not in result[1].rationale


def test_disclose_overlaps_never_crosses_relations():
    """Same column set, different order, but on two different tables: not an overlap."""
    shipments = Relation("public", "shipments")
    p1 = _plain_index_proposal("ADV001", ("a", "b"))
    p2 = _plain_index_proposal("ADV008", ("b", "a"), relation=shipments)

    result = PostgresWorkloadAdapter._disclose_column_set_overlaps([p1, p2])

    assert result[0].rationale == p1.rationale
    assert result[1].rationale == p2.rationale


def test_index_creation_columns_rejects_a_missing_or_empty_columns_tuple():
    """A proposal whose evidence carries no columns, or an empty tuple of them, must never
    be treated as eligible for prefix collapsing or overlap disclosure — there is nothing
    to compare."""
    missing = Proposal(
        code="ADV001",
        title="t",
        rationale="r",
        evidence={"schema": "public", "table": "orders"},
        confidence=Confidence.HIGH,
        ddl='CREATE INDEX ON "public"."orders" ("a");',
    )
    empty = Proposal(
        code="ADV001",
        title="t",
        rationale="r",
        evidence={"schema": "public", "table": "orders", "columns": ()},
        confidence=Confidence.HIGH,
        ddl='CREATE INDEX ON "public"."orders" ();',
    )
    assert PostgresWorkloadAdapter._index_creation_columns(missing) is None
    assert PostgresWorkloadAdapter._index_creation_columns(empty) is None


def test_adv001_states_the_same_low_ndv_caveat_as_adv007():
    """Task 6 gave ADV007 a caveat for a low leading NDV; ADV001 emitted LOW for the exact
    same reason with no explanation at all. Batch 2's other rules make that asymmetry
    visible in one report, so the wording must now match — confidence is unaffected, only
    the explanation changes.
    """
    low = propose_indexes(
        [_usage(_ORDERS, "status", ColumnRole.EQUALITY)],
        _facts_map(ndv={"status": 3.0}),
        {},
        min_cost_share=0.01,
    )
    assert low[0].confidence is Confidence.LOW
    assert "Only about 3 distinct values" in low[0].rationale
    assert "selective enough to be worth its write cost" in low[0].rationale

    # HIGH and MEDIUM must not gain the sentence — it is specifically the low-NDV caveat.
    high = propose_indexes(
        [_usage(_ORDERS, "status", ColumnRole.EQUALITY)],
        _facts_map(ndv={"status": 5000.0}),
        {},
        min_cost_share=0.01,
    )
    assert high[0].confidence is Confidence.HIGH
    assert "distinct values" not in high[0].rationale

    no_stats = propose_indexes(
        [_usage(_ORDERS, "status", ColumnRole.EQUALITY)],
        _facts_map(ndv={}),
        {},
        min_cost_share=0.01,
    )
    assert no_stats[0].confidence is Confidence.MEDIUM
    assert "distinct values" not in no_stats[0].rationale


def test_fold_discarded_deduplicates_repeated_sentences_across_three_proposals():
    """ADV001, ADV007 and ADV008 share verbatim wording for the partial/expression-index
    disclosures, so a real three-way collision would otherwise repeat the same sentence up
    to three times. A sentence must be dropped only when it exactly repeats one already
    present; a sentence unique to one proposal must always survive.
    """
    shared = "This sentence is shared by all three."
    survivor = Proposal(
        code="ADV001",
        title="wide",
        rationale=f"{shared} Only ADV001 says this.",
        evidence={"cost_share": 0.5},
        confidence=Confidence.HIGH,
        ddl="DDL",
    )
    loser_one = Proposal(
        code="ADV007",
        title="n1",
        rationale=f"{shared} Only ADV007 says this.",
        evidence={"cost_share": 0.5},
        confidence=Confidence.HIGH,
        ddl="DDL",
    )
    loser_two = Proposal(
        code="ADV008",
        title="n2",
        rationale=f"{shared} Only ADV008 says this.",
        evidence={"cost_share": 0.5},
        confidence=Confidence.MEDIUM,
        ddl="DDL",
    )

    merged = PostgresWorkloadAdapter._fold_discarded(
        survivor, [loser_one, loser_two], same_index=True
    )

    assert merged.rationale.count(shared) == 1
    assert "Only ADV001 says this." in merged.rationale
    assert "Only ADV007 says this." in merged.rationale
    assert "Only ADV008 says this." in merged.rationale


def test_propose_collapses_an_index_flagged_both_unused_and_redundant():
    """ADV002 and ADV003 can both fire on one index, emitting the same DROP twice.

    ADV003 must survive: prefix redundancy is provable from the column lists alone, while
    ADV002 rests on a scan counter covering only the window since the last stats reset.
    """
    existing = {
        _ORDERS: (
            PgIndex("idx_narrow", ("status",), False, False, 0, 4096),
            PgIndex("idx_wide", ("status", "created_at"), False, False, 5, 8192),
        )
    }
    adapter = PostgresWorkloadAdapter(querier=lambda sql, params: [])
    aggregation = Aggregation(
        usage=(), total_cost_ms=100.0, skipped_unqualifiable=0, tables=frozenset({_ORDERS})
    )
    adapter.fetch_indexes = lambda schemas, tables: existing  # type: ignore[method-assign]
    proposals = adapter.propose(aggregation, {}, _workload(), min_cost_share=0.01)

    drops = [p for p in proposals if p.ddl == 'DROP INDEX "public"."idx_narrow";']
    assert len(drops) == 1
    assert drops[0].code == "ADV003"


def test_propose_composes_all_rules_and_ranks_high_confidence_first():
    """Pins that every one of the eight rules `propose()` wires in actually fired.

    Deliberately an equality set, not a superset assertion (`>=`): a set comparison that
    only checks two of eight codes would stay green even if a rule's whole block were
    deleted from `propose()` — which is exactly what happened to ADV007 before this test
    was tightened. Each rule below gets its own trigger, on distinct columns/index names so
    none of them suppress or collide with another:
      - ADV001: hot equality column `status`.
      - ADV002: `idx_unused` on an unrelated column, zero scans.
      - ADV003: `idx_narrow_redundant` is a plain prefix of `idx_wide_redundant`.
      - ADV004: `status` (equality) and `shipped_at` (not-null check) share fingerprint fp1.
      - ADV005: `note` is non-sargable.
      - ADV006: a hot `SELECT *` over the wide `orders` table.
      - ADV007: hot join key `customer_id`, unrelated to `status` so it cannot collide with
        ADV001's candidate.
      - ADV008: hot grouping column `region`, unrelated to every other column above so it
        cannot collide with ADV001's or ADV007's candidate.
    """
    aggregation = Aggregation(
        usage=(
            _usage(_ORDERS, "status", ColumnRole.EQUALITY, cost_share=0.6, cost_ms=60.0),
            _usage(_ORDERS, "shipped_at", ColumnRole.NOT_NULL_CHECK, cost_share=0.3, cost_ms=30.0),
            _usage(_ORDERS, "note", ColumnRole.NON_SARGABLE, cost_share=0.2, cost_ms=20.0),
            _usage(_ORDERS, "customer_id", ColumnRole.JOIN, cost_share=0.5, cost_ms=50.0),
            _usage(_ORDERS, "region", ColumnRole.GROUP, cost_share=0.15, cost_ms=15.0),
        ),
        total_cost_ms=100.0,
        skipped_unqualifiable=0,
        tables=frozenset({_ORDERS}),
    )
    existing = {
        _ORDERS: (
            PgIndex("idx_unused", ("zzz",), False, False, 0, 4096),
            PgIndex("idx_narrow_redundant", ("aaa",), False, False, 5, 1),
            PgIndex("idx_wide_redundant", ("aaa", "bbb"), False, False, 5, 1),
        )
    }
    wide_columns = (
        "status",
        "shipped_at",
        "note",
        "customer_id",
        "created_at",
        *(f"c{i}" for i in range(10)),
    )
    select_star_stat = QueryStat(
        fingerprint="fp_star",
        sql="select * from orders",
        calls=5,
        total_time_ms=100.0,
        flags=frozenset({FLAG_SELECT_STAR}),
    )
    adapter = PostgresWorkloadAdapter(querier=lambda sql, params: [])
    adapter.fetch_indexes = lambda schemas, tables: existing  # type: ignore[method-assign]
    proposals = adapter.propose(
        aggregation,
        _facts_map(ndv={"status": 9999.0, "customer_id": 9999.0}, columns=wide_columns),
        _workload(select_star_stat),
        min_cost_share=0.01,
    )
    assert {p.code for p in proposals} == {
        "ADV001",
        "ADV002",
        "ADV003",
        "ADV004",
        "ADV005",
        "ADV006",
        "ADV007",
        "ADV008",
    }
    assert proposals[0].confidence is Confidence.HIGH


def test_render_ddl_emits_a_reviewable_commented_script():
    proposals = [
        Proposal(
            code="ADV001",
            title="Add index on orders(status)",
            rationale="hot",
            evidence={"cost_share": 0.5},
            confidence=Confidence.HIGH,
            ddl="CREATE INDEX ON orders (status);",
        ),
        Proposal(
            code="ADV005",
            title="Non-sargable predicate",
            rationale="cast",
            evidence={"cost_share": 0.2},
            confidence=Confidence.HIGH,
            ddl=None,
        ),
    ]
    script = PostgresWorkloadAdapter().render_ddl(proposals)
    assert "CREATE INDEX ON orders (status);" in script
    assert "-- ADV001" in script
    assert "high" in script
    assert "Non-sargable" not in script  # no DDL, so it belongs in the report only
    assert "review" in script.lower()


def _uncommented(script: str) -> list[str]:
    """Every non-blank line of a rendered DDL script that is not a `--` comment.

    The one property the whole script format exists to provide is that nothing unintended is
    executable, so tests assert over this list *exactly* rather than over a per-line
    "comment or looks like a statement" disjunction — which any injected line ending in a
    semicolon satisfies.
    """
    return [line for line in script.splitlines() if line.strip() and not line.startswith("--")]


def test_render_ddl_never_emits_a_bare_uncommented_line():
    """The one property this file exists to guarantee: safe to skim, nothing unintended.

    A single `f"-- {title}"` comments only the first line, so a newline in a title leaves
    the remainder bare in a file someone may pipe into psql. Titles are built from live
    schema identifiers and Postgres permits newlines in quoted identifiers.
    """
    proposals = [
        Proposal(
            code="ADV001",
            title="line1\nline2 -- injected",
            rationale="r",
            evidence={"cost_share": 0.3},
            confidence=Confidence.HIGH,
            ddl="CREATE INDEX ON t (c);",
        ),
    ]
    script = PostgresWorkloadAdapter().render_ddl(proposals)
    # An exact list, not "comment or ends in a semicolon": that disjunction is satisfied by
    # *any* bare line ending in `;`, which is precisely the thing being smuggled, so it could
    # not distinguish a clean script from one carrying an injected statement.
    assert _uncommented(script) == ["CREATE INDEX ON t (c);"], script
    assert "-- line2 -- injected" in script


def test_render_ddl_tolerates_a_missing_or_non_numeric_cost_share():
    """evidence is dict[str, object], so neither presence nor type is guaranteed."""
    for evidence in ({}, {"cost_share": "not a number"}, {"cost_share": True}):
        proposals = [
            Proposal(
                code="ADV001",
                title="t",
                rationale="r",
                evidence=evidence,
                confidence=Confidence.HIGH,
                ddl="CREATE INDEX ON t (c);",
            ),
        ]
        script = PostgresWorkloadAdapter().render_ddl(proposals)
        assert "-- ADV001 [high]" in script
        assert "%" not in script.split("-- ADV001")[1].split("\n")[0]


def test_render_ddl_with_no_ddl_proposals_still_explains_itself():
    script = PostgresWorkloadAdapter().render_ddl([])
    assert script.strip().startswith("--")
    assert "no ddl" in script.lower()


def test_render_ddl_recommends_concurrently_for_index_creation():
    proposals = [
        Proposal(
            code="ADV001",
            title="t",
            rationale="r",
            evidence={},
            confidence=Confidence.HIGH,
            ddl="CREATE INDEX ON orders (status);",
        ),
    ]
    script = PostgresWorkloadAdapter().render_ddl(proposals)
    assert "CONCURRENTLY" in script


def test_render_ddl_emits_a_pre_commented_multiline_block_verbatim_with_its_header():
    """A proposal can arrive with `ddl` already a `--`-commented, multi-line disclosure
    rather than raw executable DDL — dbt enrichment's ADV302 rewrite is exactly this
    shape. Before this test, the line-break guard treated *any* multi-line `ddl` as the
    identifier-with-an-embedded-break hazard: it double-commented every line, printed a
    false "an identifier ... contains a line break", and — because that whole branch
    `continue`s before the header is appended — dropped the `-- ADV001 [confidence]` line
    a reader needs to know which rule this is and how confident it was. A `ddl` value that
    is already fully commented is not that hazard and must be emitted verbatim, with its
    usual header.
    """
    config_block = (
        "-- ADV302: express this as dbt config, not DDL. Add to the model's config block:\n"
        "--   indexes:\n"
        "--     - columns: ['status']\n"
        "--       type: btree"
    )
    proposals = [
        Proposal(
            code="ADV001",
            title="Add index on orders(status)",
            rationale="hot predicate.",
            evidence={"cost_share": 0.5},
            confidence=Confidence.HIGH,
            ddl=config_block,
        ),
    ]
    script = PostgresWorkloadAdapter().render_ddl(proposals)
    assert "-- ADV001 [high, 50.0% of workload cost]" in script
    assert "NOT RENDERED" not in script
    assert "-- --   indexes:" not in script, "must not be double-commented"
    assert script.count("- columns: ['status']") == 1


def test_generated_ddl_quotes_identifiers():
    """Unquoted identifiers break on anything needing quotes — mixed case, reserved words."""
    relation = Relation("public", "Order")
    proposals = propose_indexes(
        [_usage(relation, "Status", ColumnRole.EQUALITY)],
        {
            relation: TableFacts(
                relation=relation,
                row_estimate=10**6,
                size_bytes=10**8,
                columns=("Status",),
                ndv={"Status": 5000.0},
            )
        },
        {},
        min_cost_share=0.01,
    )
    assert proposals[0].ddl == 'CREATE INDEX ON "public"."Order" ("Status");'


def test_a_newline_in_an_identifier_is_not_rendered_as_a_statement():
    """Two things had to be true here, and quoting alone only gave the first.

    Before quoting, a table named `orders\\nDROP TABLE users; --` produced a line
    `DROP TABLE users; -- (status);` that *passed* the comment-or-semicolon check — a real
    smuggled statement. Quoting fixes the semantics (psql reads it as one identifier) but
    not the file: a line break inside a quoted identifier still splits the physical line,
    leaving something that reads like a statement. Truncating the name would instead emit
    DDL against an object that does not exist. So the statement is commented out whole.
    """
    hostile = "orders\nDROP TABLE users; --"
    relation = Relation("public", hostile)
    proposals = propose_indexes(
        [_usage(relation, "status", ColumnRole.EQUALITY)],
        {
            relation: TableFacts(
                relation=relation,
                row_estimate=10**6,
                size_bytes=10**8,
                columns=("status",),
                ndv={"status": 5000.0},
            )
        },
        {},
        min_cost_share=0.01,
    )
    script = PostgresWorkloadAdapter().render_ddl(proposals)
    assert "NOT RENDERED" in script
    # Nothing at all is executable here: the whole statement was commented out, so the
    # smuggled text survives only as a comment. Asserting the empty list rather than
    # "comment or ends in `;`" — a bare `DROP TABLE users; --` satisfies that disjunction.
    assert _uncommented(script) == [], script
    # The real name is still recoverable, so an operator can see what was anomalous.
    assert "DROP TABLE users; --" in script


def test_quote_ident_doubles_an_embedded_quote():
    assert _quote_ident('we"ird') == '"we""ird"'


def test_adv001_ddl_is_qualified_with_the_relations_own_schema():
    usage = (_usage(Relation("sales", "orders"), "status", ColumnRole.EQUALITY, cost_share=0.5),)
    facts = {Relation("sales", "orders"): _facts(Relation("sales", "orders"), rows=50_000)}
    proposals = propose_indexes(usage, facts, {}, min_cost_share=0.01)
    assert proposals[0].ddl == 'CREATE INDEX ON "sales"."orders" ("status");'
    assert proposals[0].evidence["schema"] == "sales"
    assert proposals[0].evidence["table"] == "orders"
    assert "sales.orders" in proposals[0].title


def test_two_same_named_relations_get_two_independent_proposals():
    """One proposal per relation, each stamped with its own schema."""
    usage = (
        _usage(Relation("sales", "orders"), "status", ColumnRole.EQUALITY, cost_share=0.5),
        _usage(Relation("staging", "orders"), "status", ColumnRole.EQUALITY, cost_share=0.5),
    )
    facts = {
        Relation("sales", "orders"): _facts(Relation("sales", "orders"), rows=50_000),
        Relation("staging", "orders"): _facts(Relation("staging", "orders"), rows=50_000),
    }
    ddls = {p.ddl for p in propose_indexes(usage, facts, {}, min_cost_share=0.01)}
    assert ddls == {
        'CREATE INDEX ON "sales"."orders" ("status");',
        'CREATE INDEX ON "staging"."orders" ("status");',
    }


def test_an_index_in_one_schema_does_not_cover_the_other_schemas_candidate():
    """The coverage check must not reach across schemas."""
    usage = (
        _usage(Relation("sales", "orders"), "status", ColumnRole.EQUALITY, cost_share=0.5),
        _usage(Relation("staging", "orders"), "status", ColumnRole.EQUALITY, cost_share=0.5),
    )
    facts = {
        Relation("sales", "orders"): _facts(Relation("sales", "orders"), rows=50_000),
        Relation("staging", "orders"): _facts(Relation("staging", "orders"), rows=50_000),
    }
    existing = {
        Relation("sales", "orders"): (
            PgIndex(
                name="idx_status",
                columns=("status",),
                is_unique=False,
                is_primary=False,
                scans=1,
                size_bytes=1,
            ),
        )
    }
    proposals = propose_indexes(usage, facts, existing, min_cost_share=0.01)
    assert [p.evidence["schema"] for p in proposals] == ["staging"]


def test_created_index_ddl_is_schema_qualified():
    """An unqualified name resolves against the *operator's* search_path, not ours.

    `CREATE INDEX ON "orders"` run by someone whose search_path is `public` targets the
    wrong table entirely when the relation actually lives in `analytics`.
    """
    relation = Relation("analytics", "orders")
    proposals = propose_indexes(
        [_usage(relation, "status", ColumnRole.EQUALITY)],
        _facts_map(relation, ndv={"status": 5000.0}),
        {},
        min_cost_share=0.01,
    )
    assert proposals[0].ddl == 'CREATE INDEX ON "analytics"."orders" ("status");'


def test_partial_index_ddl_is_schema_qualified():
    relation = Relation("analytics", "orders")
    proposals = propose_partial_indexes(
        [
            _usage(relation, "status", ColumnRole.EQUALITY, cost_ms=90.0),
            _usage(relation, "shipped_at", ColumnRole.NOT_NULL_CHECK, cost_share=0.4, cost_ms=40.0),
        ],
        _facts_map(relation),
        min_cost_share=0.01,
    )
    assert proposals[0].ddl.startswith('CREATE INDEX ON "analytics"."orders" ("status")')


def test_dropped_index_ddl_is_schema_qualified():
    """A bare `DROP INDEX idx_cold` drops whichever idx_cold the search_path finds first."""
    relation = Relation("analytics", "orders")
    unused = {relation: (PgIndex("idx_cold", ("note",), False, False, 0, 4096),)}
    proposals = propose_unused_indexes(unused, hot_tables=frozenset({relation}))
    assert proposals[0].ddl == 'DROP INDEX "analytics"."idx_cold";'

    redundant = {
        relation: (
            PgIndex("idx_narrow", ("status",), False, False, 5, 1),
            PgIndex("idx_wide", ("status", "created_at"), False, False, 5, 1),
        )
    }
    proposals = propose_redundant_indexes(redundant, hot_tables=frozenset(redundant))
    assert proposals[0].ddl == 'DROP INDEX "analytics"."idx_narrow";'


def test_adv002_evidence_reports_the_bare_table_name_and_its_own_schema():
    """The brief's evidence contract — `"schema": relation.schema`, `"table": relation.table`
    (the *bare* name, so existing JSON consumers keep reading the same value from the same
    key) — applies to every rule, not just ADV001. `staging` (not `public`) pins that the
    schema is not a hardcoded default, and the bare `"orders"` (not `"staging.orders"`) pins
    that `evidence["table"]` was not quietly switched to the qualified string."""
    existing = {
        Relation("staging", "orders"): (
            PgIndex(
                name="idx_cold",
                columns=("note",),
                is_unique=False,
                is_primary=False,
                scans=0,
                size_bytes=1,
            ),
        )
    }
    proposals = propose_unused_indexes(
        existing, hot_tables=frozenset({Relation("staging", "orders")})
    )
    assert proposals[0].evidence["schema"] == "staging"
    assert proposals[0].evidence["table"] == "orders"


def test_adv003_evidence_reports_the_bare_table_name_and_its_own_schema():
    existing = {
        Relation("staging", "orders"): (
            PgIndex("idx_narrow", ("status",), False, False, 5, 1),
            PgIndex("idx_wide", ("status", "created_at"), False, False, 5, 1),
        )
    }
    proposals = propose_redundant_indexes(existing, hot_tables=frozenset(existing))
    assert proposals[0].evidence["schema"] == "staging"
    assert proposals[0].evidence["table"] == "orders"


def test_adv004_evidence_reports_the_bare_table_name_and_its_own_schema():
    relation = Relation("staging", "orders")
    proposals = propose_partial_indexes(
        [
            _usage(relation, "status", ColumnRole.EQUALITY, cost_ms=90.0),
            _usage(relation, "shipped_at", ColumnRole.NOT_NULL_CHECK, cost_share=0.4, cost_ms=40.0),
        ],
        _facts_map(relation),
        min_cost_share=0.01,
    )
    assert proposals[0].evidence["schema"] == "staging"
    assert proposals[0].evidence["table"] == "orders"


def test_propose_end_to_end_uses_a_non_public_relations_own_schema():
    """The three adapter-level `propose()` tests elsewhere in this file all use
    `public.orders`, so the end-to-end path was only ever exercised with the default
    schema. A relation living anywhere else must flow through unchanged."""
    relation = Relation("analytics", "orders")
    aggregation = Aggregation(
        usage=(_usage(relation, "status", ColumnRole.EQUALITY, cost_share=0.6, cost_ms=60.0),),
        total_cost_ms=100.0,
        skipped_unqualifiable=0,
        tables=frozenset({relation}),
    )
    adapter = PostgresWorkloadAdapter(querier=lambda sql, params: [])
    proposals = adapter.propose(
        aggregation,
        _facts_map(relation, ndv={"status": 9999.0}),
        _workload(),
        min_cost_share=0.01,
    )
    adv001 = [p for p in proposals if p.code == "ADV001"]
    assert adv001
    assert adv001[0].ddl == 'CREATE INDEX ON "analytics"."orders" ("status");'
    assert adv001[0].evidence["schema"] == "analytics"
    assert adv001[0].evidence["table"] == "orders"


def test_adv005_reports_a_short_fingerprint_id_and_keeps_the_sql_separately():
    """`fingerprint` was the entire canonical SQL, so ADV005 printed the query twice.

    Once as `fingerprint`, once as `sql` — for a long statement that is the bulk of the
    proposal's evidence block, duplicated. `fingerprint` is an identity, so a short stable
    hash is what it should carry.
    """
    canonical = 'SELECT "note" FROM "orders" WHERE "note" LIKE ? AND "status" = ?'
    stat = QueryStat(
        fingerprint=canonical,
        sql="select note from orders where note like $1 and status = $2",
        calls=5,
        total_time_ms=100.0,
        flags=frozenset({FLAG_LEADING_WILDCARD_LIKE}),
    )
    proposals = propose_sargability([], _workload(stat), min_cost_share=0.01)
    fingerprint = proposals[0].evidence["fingerprint"]
    assert fingerprint != canonical
    assert len(fingerprint) <= 16
    # The text is still there — the identity is what got shorter, not the evidence.
    assert proposals[0].evidence["sql"] == stat.sql


def test_the_fingerprint_id_is_stable_and_distinguishing():
    """It is an identity: the same query group must always get the same id, and two
    different groups must not collide."""
    from sqlquality.workload.postgres import _fingerprint_id

    assert _fingerprint_id("select 1") == _fingerprint_id("select 1")
    assert _fingerprint_id("select 1") != _fingerprint_id("select 2")


def test_adv006_also_reports_the_short_id_without_losing_the_query():
    canonical = 'SELECT * FROM "orders"'
    stat = QueryStat(
        fingerprint=canonical,
        sql="select * from orders",
        calls=5,
        total_time_ms=100.0,
        flags=frozenset({FLAG_SELECT_STAR}),
    )
    wide = {
        _ORDERS: TableFacts(
            relation=_ORDERS,
            row_estimate=10**6,
            size_bytes=10**8,
            columns=tuple(f"c{i}" for i in range(30)),
        )
    }
    proposals = propose_select_star(_workload(stat), wide, min_cost_share=0.01)
    assert proposals[0].evidence["fingerprint"] != canonical
    assert proposals[0].evidence["sql"] == stat.sql


def test_adv007_proposes_an_index_on_an_unindexed_hot_join_key():
    relation = Relation("public", "order_items")
    usage = (_usage(relation, "order_id", ColumnRole.JOIN, cost_share=0.4, cost_ms=400.0),)
    facts = {relation: _facts(relation, rows=100_000, ndv={"order_id": 5000.0})}
    proposals = propose_join_keys(usage, facts, {}, min_cost_share=0.01)
    assert [p.code for p in proposals] == ["ADV007"]
    assert proposals[0].ddl == 'CREATE INDEX ON "public"."order_items" ("order_id");'
    assert proposals[0].confidence is Confidence.HIGH


def test_adv007_is_silent_when_an_index_already_leads_with_the_join_key():
    relation = Relation("public", "order_items")
    usage = (_usage(relation, "order_id", ColumnRole.JOIN, cost_share=0.4),)
    facts = {relation: _facts(relation, rows=100_000, ndv={"order_id": 5000.0})}
    existing = {
        relation: (
            PgIndex(
                name="idx_oi_order",
                columns=("order_id", "sku"),
                is_unique=False,
                is_primary=False,
                scans=5,
                size_bytes=1,
            ),
        )
    }
    assert propose_join_keys(usage, facts, existing, min_cost_share=0.01) == []


def test_adv007_respects_the_small_table_floor():
    relation = Relation("public", "tiny")
    usage = (_usage(relation, "order_id", ColumnRole.JOIN, cost_share=0.9),)
    facts = {relation: _facts(relation, rows=10, ndv={"order_id": 5.0})}
    assert propose_join_keys(usage, facts, {}, min_cost_share=0.01) == []


def test_adv007_caps_at_low_when_the_index_list_could_not_be_read():
    relation = Relation("public", "order_items")
    usage = (_usage(relation, "order_id", ColumnRole.JOIN, cost_share=0.4),)
    facts = {relation: _facts(relation, rows=100_000, ndv={"order_id": 5000.0})}
    proposals = propose_join_keys(usage, facts, {}, min_cost_share=0.01, have_index_data=False)
    assert proposals[0].confidence is Confidence.LOW
    assert "could not be read" in proposals[0].rationale


def test_adv007_caps_at_low_and_discloses_an_unknown_row_count():
    relation = Relation("public", "order_items")
    usage = (_usage(relation, "order_id", ColumnRole.JOIN, cost_share=0.4),)
    facts = {relation: _facts(relation, rows=None, ndv={"order_id": 5000.0})}
    proposals = propose_join_keys(usage, facts, {}, min_cost_share=0.01)
    assert proposals[0].confidence is Confidence.LOW
    assert "small-table floor" in proposals[0].rationale


def test_adv007_is_low_for_a_low_cardinality_join_key():
    relation = Relation("public", "order_items")
    usage = (_usage(relation, "kind", ColumnRole.JOIN, cost_share=0.4),)
    facts = {relation: _facts(relation, rows=100_000, ndv={"kind": 3.0})}
    proposals = propose_join_keys(usage, facts, {}, min_cost_share=0.01)
    assert proposals[0].confidence is Confidence.LOW


def test_adv007_is_medium_when_ndv_is_unknown():
    """The middle rung of the confidence ladder: `rows` and `have_index_data` are both
    fine, but the NDV catalog has nothing for this column. Changing that branch to return
    HIGH instead of MEDIUM would overstate a claim with no selectivity evidence behind it —
    exactly the failure mode this rule set exists to avoid — and must fail this test."""
    relation = Relation("public", "order_items")
    usage = (_usage(relation, "order_id", ColumnRole.JOIN, cost_share=0.4),)
    facts = {relation: _facts(relation, rows=100_000, ndv={})}
    proposals = propose_join_keys(usage, facts, {}, min_cost_share=0.01)
    assert proposals[0].confidence is Confidence.MEDIUM


def test_adv007_suppresses_a_join_key_below_the_cost_share_threshold():
    """Pins that `--min-cost-share` actually reaches ADV007, as its own help text now
    claims. Replacing the `cost_share < min_cost_share` guard with `if False` must fail
    this test."""
    relation = Relation("public", "order_items")
    usage = (_usage(relation, "order_id", ColumnRole.JOIN, cost_share=0.005, cost_ms=5.0),)
    facts = {relation: _facts(relation, rows=100_000, ndv={"order_id": 5000.0})}
    assert propose_join_keys(usage, facts, {}, min_cost_share=0.01) == []


def test_adv007_discloses_a_partial_index_leading_with_the_join_key():
    """Mirrors ADV001's `test_a_partial_index_does_not_suppress_a_candidate` exactly.
    `_covered` correctly does not treat a partial index as coverage — a WHERE-guarded index
    does not serve an unfiltered join probe either — but before this test existed, ADV007
    silently said nothing about `idx_open` at all. Naming it in the evidence and rationale,
    the same way ADV001 does for the same gap, is the fix."""
    relation = Relation("public", "order_items")
    existing = {
        relation: (
            PgIndex(
                "idx_open",
                ("order_id",),
                False,
                False,
                5,
                4096,
                is_partial=True,
                predicate="(shipped_at IS NULL)",
            ),
        )
    }
    usage = (_usage(relation, "order_id", ColumnRole.JOIN, cost_share=0.4),)
    facts = {relation: _facts(relation, rows=100_000, ndv={"order_id": 5000.0})}
    proposals = propose_join_keys(usage, facts, existing, min_cost_share=0.01)
    assert codes(proposals) == ["ADV007"]
    assert proposals[0].evidence["partial_indexes_skipped"] == ("idx_open",)
    assert "partial" in proposals[0].rationale.lower()


def test_adv007_discloses_an_expression_index_mentioning_the_join_key():
    """Mirrors ADV001's `test_an_expression_index_is_disclosed_not_silently_ignored`. The
    `columns` tuple of an expression index understates it, so `_covered` cannot see
    `lower(order_id)` leads with `order_id` — naming the index is the only honest option."""
    relation = Relation("public", "order_items")
    existing = {
        relation: (
            PgIndex(
                "idx_lower_order_id",
                (),
                False,
                False,
                5,
                4096,
                has_expressions=True,
                definition="CREATE INDEX idx_lower_order_id ON order_items (lower(order_id))",
            ),
        )
    }
    usage = (_usage(relation, "order_id", ColumnRole.JOIN, cost_share=0.4),)
    facts = {relation: _facts(relation, rows=100_000, ndv={"order_id": 5000.0})}
    proposals = propose_join_keys(usage, facts, existing, min_cost_share=0.01)
    assert codes(proposals) == ["ADV007"]
    assert proposals[0].evidence["expression_indexes"] == ("idx_lower_order_id",)
    assert "expression" in proposals[0].rationale.lower()


def test_adv007_ignores_non_join_roles():
    """The rule must not re-propose what ADV001 already covers."""
    relation = Relation("public", "orders")
    usage = (_usage(relation, "status", ColumnRole.EQUALITY, cost_share=0.9),)
    facts = {relation: _facts(relation, rows=100_000)}
    assert propose_join_keys(usage, facts, {}, min_cost_share=0.01) == []


def test_adv007_reports_the_hottest_join_key_per_relation():
    relation = Relation("public", "order_items")
    usage = (
        _usage(relation, "order_id", ColumnRole.JOIN, cost_share=0.4, cost_ms=400.0),
        _usage(relation, "sku", ColumnRole.JOIN, cost_share=0.1, cost_ms=100.0),
    )
    facts = {relation: _facts(relation, rows=100_000)}
    proposals = propose_join_keys(usage, facts, {}, min_cost_share=0.01)
    assert [p.evidence["columns"] for p in proposals] == [("order_id",), ("sku",)]


def test_adv007_evidence_reports_the_bare_table_name_and_its_own_schema():
    relation = Relation("staging", "order_items")
    usage = (_usage(relation, "order_id", ColumnRole.JOIN, cost_share=0.4),)
    facts = {relation: _facts(relation, rows=100_000, ndv={"order_id": 5000.0})}
    proposals = propose_join_keys(usage, facts, {}, min_cost_share=0.01)
    assert proposals[0].evidence["schema"] == "staging"
    assert proposals[0].evidence["table"] == "order_items"


def test_adv008_proposes_a_composite_index_for_a_hot_group_by():
    relation = Relation("public", "events")
    usage = (
        _usage(
            relation, "tenant_id", ColumnRole.GROUP, cost_share=0.5, cost_ms=500.0, fps=("fp1",)
        ),
        _usage(relation, "day", ColumnRole.GROUP, cost_share=0.5, cost_ms=400.0, fps=("fp1",)),
    )
    facts = {relation: _facts(relation, rows=5_000_000)}
    proposals = propose_grouping_indexes(usage, facts, {}, min_cost_share=0.01)
    assert [p.code for p in proposals] == ["ADV008"]
    assert proposals[0].evidence["columns"] == ("tenant_id", "day")
    assert proposals[0].ddl == 'CREATE INDEX ON "public"."events" ("tenant_id", "day");'


def test_adv008_never_reaches_high_confidence():
    """Whether the planner picks GroupAggregate over HashAggregate is not visible to us."""
    relation = Relation("public", "events")
    usage = (
        _usage(
            relation, "tenant_id", ColumnRole.GROUP, cost_share=0.9, cost_ms=900.0, fps=("fp1",)
        ),
    )
    facts = {relation: _facts(relation, rows=5_000_000, ndv={"tenant_id": 100_000.0})}
    proposals = propose_grouping_indexes(usage, facts, {}, min_cost_share=0.01)
    assert proposals[0].confidence is Confidence.MEDIUM


def test_adv008_is_low_when_the_row_count_is_unknown():
    """The other rung of the confidence ladder: row count unknown, so the small-table gate
    could not run. Changing this branch to MEDIUM would claim a check happened that did not,
    exactly the failure mode the ladder exists to avoid."""
    relation = Relation("public", "events")
    usage = (
        _usage(
            relation, "tenant_id", ColumnRole.GROUP, cost_share=0.9, cost_ms=900.0, fps=("fp1",)
        ),
    )
    facts = {relation: _facts(relation, rows=None)}
    proposals = propose_grouping_indexes(usage, facts, {}, min_cost_share=0.01)
    assert proposals[0].confidence is Confidence.LOW
    assert "small-table floor" in proposals[0].rationale


def test_adv008_is_low_when_the_index_list_could_not_be_read():
    """The other LOW trigger: the existing-index catalog query was denied, so whether an
    index already leads with these columns is unknowable."""
    relation = Relation("public", "events")
    usage = (
        _usage(
            relation, "tenant_id", ColumnRole.GROUP, cost_share=0.9, cost_ms=900.0, fps=("fp1",)
        ),
    )
    facts = {relation: _facts(relation, rows=5_000_000)}
    proposals = propose_grouping_indexes(
        usage, facts, {}, min_cost_share=0.01, have_index_data=False
    )
    assert proposals[0].confidence is Confidence.LOW
    assert "could not be read" in proposals[0].rationale


def test_adv008_groups_only_columns_that_co_occur_in_one_query():
    """Two GROUP BYs in two different queries are not one composite index."""
    relation = Relation("public", "events")
    usage = (
        _usage(
            relation, "tenant_id", ColumnRole.GROUP, cost_share=0.5, cost_ms=500.0, fps=("fp1",)
        ),
        _usage(relation, "day", ColumnRole.GROUP, cost_share=0.5, cost_ms=400.0, fps=("fp2",)),
    )
    facts = {relation: _facts(relation, rows=5_000_000)}
    proposals = propose_grouping_indexes(usage, facts, {}, min_cost_share=0.01)
    assert proposals[0].evidence["columns"] == ("tenant_id",)


def test_adv008_requires_joint_support_not_just_pairwise_with_the_seed():
    """A transitive chain must not be welded into one composite.

    `a` co-occurs with `b` in fp1 and with `c` in fp2, but `b` and `c` never co-occur with
    each other — no query in the workload groups by `a, b, c` together. Checking each
    candidate only against the seed's fingerprints (the old, pairwise rule) let `c` join
    once `b` had already been accepted, on the strength of `a`'s membership in fp2 — even
    though the *composite so far*, `(a, b)`, is never grouped by alongside `c`. Requiring the
    running intersection to stay non-empty catches this: after `b` joins, the shared set
    narrows to fp1 alone, and `c` (only in fp2) can no longer extend it.
    """
    relation = Relation("public", "events")
    usage = (
        _usage(relation, "a", ColumnRole.GROUP, cost_share=0.5, cost_ms=500.0, fps=("fp1", "fp2")),
        _usage(relation, "b", ColumnRole.GROUP, cost_share=0.5, cost_ms=400.0, fps=("fp1",)),
        _usage(relation, "c", ColumnRole.GROUP, cost_share=0.5, cost_ms=300.0, fps=("fp2",)),
    )
    facts = {relation: _facts(relation, rows=5_000_000)}
    proposals = propose_grouping_indexes(usage, facts, {}, min_cost_share=0.01)
    assert proposals[0].evidence["columns"] == ("a", "b")
    assert proposals[0].evidence["columns"] != ("a", "b", "c")


def test_adv008_reports_the_honest_joint_support_count_and_omits_the_plain_one():
    """`co_occurring_fingerprints` must report the running intersection's size, not the
    per-column `fingerprints` max, which would (falsely) read as "two query groups back this
    three-column composite" when the joint support for `(a, b)` is exactly one query group.

    The plain `fingerprints` key must be *absent*, not merely unused: `report.py` renders
    evidence as generic sorted `k=v` pairs with no per-rule text, so a reader sees both
    numbers side by side with nothing to say which one actually supports the proposal.
    Unlike ADV001/ADV007 (one candidate, so a per-column count is the whole truth), this
    rule's claim is about columns appearing *together*, and only the joint count says that.
    """
    relation = Relation("public", "events")
    usage = (
        _usage(relation, "a", ColumnRole.GROUP, cost_share=0.5, cost_ms=500.0, fps=("fp1", "fp2")),
        _usage(relation, "b", ColumnRole.GROUP, cost_share=0.5, cost_ms=400.0, fps=("fp1",)),
        _usage(relation, "c", ColumnRole.GROUP, cost_share=0.5, cost_ms=300.0, fps=("fp2",)),
    )
    facts = {relation: _facts(relation, rows=5_000_000)}
    proposals = propose_grouping_indexes(usage, facts, {}, min_cost_share=0.01)
    assert proposals[0].evidence["co_occurring_fingerprints"] == 1
    assert "fingerprints" not in proposals[0].evidence


def test_adv008_composite_is_just_the_seed_when_it_shares_nothing_with_any_other_column():
    relation = Relation("public", "events")
    usage = (
        _usage(relation, "a", ColumnRole.GROUP, cost_share=0.5, cost_ms=500.0, fps=("fp1",)),
        _usage(relation, "b", ColumnRole.GROUP, cost_share=0.5, cost_ms=400.0, fps=("fp2",)),
        _usage(relation, "c", ColumnRole.GROUP, cost_share=0.5, cost_ms=300.0, fps=("fp3",)),
    )
    facts = {relation: _facts(relation, rows=5_000_000)}
    proposals = propose_grouping_indexes(usage, facts, {}, min_cost_share=0.01)
    assert proposals[0].evidence["columns"] == ("a",)
    assert proposals[0].evidence["co_occurring_fingerprints"] == 1


def test_adv008_is_silent_when_an_index_already_leads_with_the_grouping_columns():
    relation = Relation("public", "events")
    usage = (
        _usage(
            relation, "tenant_id", ColumnRole.GROUP, cost_share=0.5, cost_ms=500.0, fps=("fp1",)
        ),
    )
    facts = {relation: _facts(relation, rows=5_000_000)}
    existing = {
        relation: (
            PgIndex(
                name="idx_events_tenant",
                columns=("tenant_id", "day"),
                is_unique=False,
                is_primary=False,
                scans=3,
                size_bytes=1,
            ),
        )
    }
    assert propose_grouping_indexes(usage, facts, existing, min_cost_share=0.01) == []


def test_adv008_respects_max_arity():
    relation = Relation("public", "events")
    usage = tuple(
        _usage(relation, name, ColumnRole.GROUP, cost_share=0.5, cost_ms=500.0 - i, fps=("fp1",))
        for i, name in enumerate(["a", "b", "c", "d"])
    )
    facts = {relation: _facts(relation, rows=5_000_000)}
    proposals = propose_grouping_indexes(usage, facts, {}, min_cost_share=0.01)
    assert proposals[0].evidence["columns"] == ("a", "b", "c")


def test_adv008_discloses_that_the_column_order_is_inferred():
    relation = Relation("public", "events")
    usage = (
        _usage(
            relation, "tenant_id", ColumnRole.GROUP, cost_share=0.5, cost_ms=500.0, fps=("fp1",)
        ),
        _usage(relation, "day", ColumnRole.GROUP, cost_share=0.5, cost_ms=400.0, fps=("fp1",)),
    )
    facts = {relation: _facts(relation, rows=5_000_000)}
    proposals = propose_grouping_indexes(usage, facts, {}, min_cost_share=0.01)
    assert "inferred" in proposals[0].rationale.lower()


def test_adv008_ignores_non_group_roles():
    relation = Relation("public", "orders")
    usage = (_usage(relation, "status", ColumnRole.EQUALITY, cost_share=0.9),)
    facts = {relation: _facts(relation, rows=5_000_000)}
    assert propose_grouping_indexes(usage, facts, {}, min_cost_share=0.01) == []


def test_adv008_respects_the_small_table_floor():
    relation = Relation("public", "tiny")
    usage = (_usage(relation, "tenant_id", ColumnRole.GROUP, cost_share=0.9),)
    facts = {relation: _facts(relation, rows=10)}
    assert propose_grouping_indexes(usage, facts, {}, min_cost_share=0.01) == []


def test_adv008_suppresses_a_grouping_below_the_cost_share_threshold():
    """Pins the `cost_share < min_cost_share` guard specifically: rows are well above
    MIN_ROWS_FOR_INDEX (10,000) so the small-table floor cannot be why this is suppressed."""
    relation = Relation("public", "events")
    usage = (
        _usage(
            relation, "tenant_id", ColumnRole.GROUP, cost_share=0.005, cost_ms=5.0, fps=("fp1",)
        ),
    )
    facts = {relation: _facts(relation, rows=5_000_000)}
    assert propose_grouping_indexes(usage, facts, {}, min_cost_share=0.01) == []


def test_adv008_discloses_a_partial_index_leading_with_the_grouping_columns():
    """Mirrors ADV001's/ADV007's identical disclosure. `_covered` correctly does not treat a
    partial index as coverage — a WHERE-guarded index does not serve an unfiltered GROUP BY
    either — but silence on that gap would let ADV008 say nothing next to an index that, in
    plain English, does lead with these columns."""
    relation = Relation("public", "events")
    existing = {
        relation: (
            PgIndex(
                "idx_events_open",
                ("tenant_id",),
                False,
                False,
                5,
                4096,
                is_partial=True,
                predicate="(closed_at IS NULL)",
            ),
        )
    }
    usage = (
        _usage(
            relation, "tenant_id", ColumnRole.GROUP, cost_share=0.5, cost_ms=500.0, fps=("fp1",)
        ),
    )
    facts = {relation: _facts(relation, rows=5_000_000)}
    proposals = propose_grouping_indexes(usage, facts, existing, min_cost_share=0.01)
    assert codes(proposals) == ["ADV008"]
    assert proposals[0].evidence["partial_indexes_skipped"] == ("idx_events_open",)
    assert "partial" in proposals[0].rationale.lower()


def test_adv008_discloses_an_expression_index_mentioning_the_leading_grouping_column():
    """Mirrors ADV001's/ADV007's identical disclosure. The `columns` tuple of an expression
    index understates it, so `_covered` cannot see `lower(tenant_id)` leads with `tenant_id`
    — naming the index is the only honest option."""
    relation = Relation("public", "events")
    existing = {
        relation: (
            PgIndex(
                "idx_lower_tenant",
                (),
                False,
                False,
                5,
                4096,
                has_expressions=True,
                definition="CREATE INDEX idx_lower_tenant ON events (lower(tenant_id))",
            ),
        )
    }
    usage = (
        _usage(
            relation, "tenant_id", ColumnRole.GROUP, cost_share=0.5, cost_ms=500.0, fps=("fp1",)
        ),
    )
    facts = {relation: _facts(relation, rows=5_000_000)}
    proposals = propose_grouping_indexes(usage, facts, existing, min_cost_share=0.01)
    assert codes(proposals) == ["ADV008"]
    assert proposals[0].evidence["expression_indexes"] == ("idx_lower_tenant",)
    assert "expression" in proposals[0].rationale.lower()


def test_adv008_evidence_reports_the_bare_table_name_and_its_own_schema():
    relation = Relation("staging", "events")
    usage = (
        _usage(
            relation, "tenant_id", ColumnRole.GROUP, cost_share=0.5, cost_ms=500.0, fps=("fp1",)
        ),
    )
    facts = {relation: _facts(relation, rows=5_000_000)}
    proposals = propose_grouping_indexes(usage, facts, {}, min_cost_share=0.01)
    assert proposals[0].evidence["schema"] == "staging"
    assert proposals[0].evidence["table"] == "events"


def test_adv001_medium_discloses_that_the_selectivity_check_could_not_run():
    """MEDIUM must say *why*, not just be a middling number.

    The rung is reached when the row count and the index list are both known but the NDV
    catalog has nothing for the leading column — so the selectivity check did not run. Saying
    nothing reads as a considered judgement rather than an absence of evidence, and disclosing
    a check that could not run is the discipline the whole rule set turns on. It was the one
    confidence level that stayed silent about its own reason.
    """
    relation = Relation("public", "orders")
    usage = (_usage(relation, "status", ColumnRole.EQUALITY, cost_share=0.5),)
    facts = {relation: _facts(relation, rows=100_000, ndv={})}
    proposals = propose_indexes(usage, facts, {}, min_cost_share=0.01)
    assert proposals[0].confidence is Confidence.MEDIUM
    assert "No distinct-value statistics for status" in proposals[0].rationale
    assert "ANALYZE" in proposals[0].rationale


def test_adv007_medium_discloses_the_same_gap_in_the_same_words():
    """The two index-creating rules must not explain the same rung differently.

    An operator reads the report, not the rule that produced the line; ADV001 explaining a
    MEDIUM while ADV007 stayed silent for the identical reason is the asymmetry that made the
    low-NDV caveat a finding in the first place.
    """
    relation = Relation("public", "order_items")
    usage = (_usage(relation, "order_id", ColumnRole.JOIN, cost_share=0.4),)
    facts = {relation: _facts(relation, rows=100_000, ndv={})}
    adv007 = propose_join_keys(usage, facts, {}, min_cost_share=0.01)[0]

    orders = Relation("public", "orders")
    adv001 = propose_indexes(
        (_usage(orders, "order_id", ColumnRole.EQUALITY, cost_share=0.4),),
        {orders: _facts(orders, rows=100_000, ndv={})},
        {},
        min_cost_share=0.01,
    )[0]

    shared = "so how selective this index would be could not be checked"
    assert shared in adv007.rationale
    assert shared in adv001.rationale


def test_adv001_says_nothing_about_selectivity_when_a_louder_gap_already_applies():
    """The disclosure is for the MEDIUM rung only, not a blanket sentence.

    With an unknown row count the proposal is LOW and already carries the small-table note; a
    second "statistics were missing" sentence there would be noise, and would also be
    misleading — the reason it is LOW is the row count, not the NDV.
    """
    relation = Relation("public", "orders")
    usage = (_usage(relation, "status", ColumnRole.EQUALITY, cost_share=0.5),)
    facts = {relation: _facts(relation, rows=None, ndv={})}
    proposals = propose_indexes(usage, facts, {}, min_cost_share=0.01)
    assert proposals[0].confidence is Confidence.LOW
    assert "No distinct-value statistics" not in proposals[0].rationale


def test_adv007_orders_equal_cost_join_keys_by_column_name():
    """Two join keys tied on cost must come out in a fixed order.

    The sort key is `(-cost_ms, column)`; without the column component two equally hot join
    keys would order by whatever `_by_relation` happened to accumulate, and the report would
    reshuffle between runs on identical input.
    """
    relation = Relation("public", "order_items")
    facts = {relation: _facts(relation, rows=100_000)}
    forward = (
        _usage(relation, "zeta", ColumnRole.JOIN, cost_share=0.4, cost_ms=100.0),
        _usage(relation, "alpha", ColumnRole.JOIN, cost_share=0.4, cost_ms=100.0),
    )
    columns = [
        p.evidence["columns"] for p in propose_join_keys(forward, facts, {}, min_cost_share=0.01)
    ]
    assert columns == [("alpha",), ("zeta",)]
    # Reversed input must give the same order, or the tiebreak is not doing the work.
    reversed_columns = [
        p.evidence["columns"]
        for p in propose_join_keys(tuple(reversed(forward)), facts, {}, min_cost_share=0.01)
    ]
    assert reversed_columns == columns


def _ddl_proposal(
    ddl: str = 'CREATE INDEX ON "main"."orders" ("status");', note: str | None = None
) -> Proposal:
    """A minimal DDL-carrying proposal for the renderer tests below."""
    return Proposal(
        code="ADV001",
        title="Add index on orders(status)",
        rationale="hot predicate.",
        evidence={"cost_share": 0.5},
        confidence=Confidence.HIGH,
        ddl=ddl,
        note=note,
    )


def test_a_proposal_note_is_emitted_as_comment_lines_above_its_statement():
    """`rationale` never reaches this file — only code, confidence, cost share and title do.

    So a caveat that lives only in `rationale` is invisible to the one person acting on the
    statement, which is how a `--ddl` file came to hold a dbt config block explaining that
    raw DDL is destroyed by `dbt run` and, below it, a bare `CREATE INDEX` on that same
    dbt-managed table. `note` is the field that travels with the statement, and it has to be
    *above* it: a caveat printed below is a caveat read after pasting.
    """
    script = PostgresWorkloadAdapter().render_ddl(
        [
            _ddl_proposal(
                note="dbt WARNING: rebuilt by dbt model model.demo.orders.\nReapply by hand."
            )
        ]
    )
    lines = script.splitlines()
    statement = lines.index('CREATE INDEX ON "main"."orders" ("status");')
    assert lines[statement - 2] == "-- dbt WARNING: rebuilt by dbt model model.demo.orders."
    assert lines[statement - 1] == "-- Reapply by hand."
    assert _uncommented(script) == ['CREATE INDEX ON "main"."orders" ("status");'], script


def test_a_multiline_note_cannot_break_out_of_comment_mode():
    """A note is interpolated from a manifest's `unique_id`, which can carry a newline.

    Rendered through `_comment_lines`, every physical line gets its own `--`, so a note is
    subject to the same guarantee as a title.
    """
    script = PostgresWorkloadAdapter().render_ddl(
        [_ddl_proposal(note="warning\nDROP TABLE users; -- injected")]
    )
    assert "-- DROP TABLE users; -- injected" in script
    assert _uncommented(script) == ['CREATE INDEX ON "main"."orders" ("status");'], script


def test_a_note_survives_the_not_rendered_fallback():
    """The fallback tells an operator to apply the statement by hand, so a caveat about
    whether the statement is even durable belongs there too."""
    script = PostgresWorkloadAdapter().render_ddl(
        [_ddl_proposal(ddl='CREATE INDEX ON "main"."or\nders" ("status");', note="dbt WARNING: x")]
    )
    assert "NOT RENDERED" in script
    assert "-- dbt WARNING: x" in script
    assert _uncommented(script) == [], script


def test_a_proposal_without_a_note_renders_exactly_as_before():
    """`note` defaults to None and must add nothing at all when unset — every dbt-free run
    is this case, and the pre-dbt DDL script has to stay byte-identical."""
    script = PostgresWorkloadAdapter().render_ddl([_ddl_proposal()])
    assert script == PostgresWorkloadAdapter().render_ddl([_ddl_proposal(note=None)])
    assert (
        "-- ADV001 [high, 50.0% of workload cost]\n"
        "-- Add index on orders(status)\n"
        'CREATE INDEX ON "main"."orders" ("status");'
    ) in script


def test_a_carriage_return_in_an_identifier_is_not_rendered_as_a_statement():
    """The line-break guard tests `"\\n" in ddl or "\\r" in ddl` and only the `\\n` half was
    pinned: dropping the `\\r` half left every test green while a CR-only break — which
    `str.splitlines()` splits on, so the file really does show two physical lines — bypassed
    the fallback entirely and emitted something reading like a bare statement."""
    script = PostgresWorkloadAdapter().render_ddl(
        [_ddl_proposal(ddl='CREATE INDEX ON "main"."or\rders" ("status");')]
    )
    assert "NOT RENDERED" in script
    assert _uncommented(script) == [], script


def test_is_fully_commented_requires_every_line_to_be_a_comment():
    """The guard that lets a pre-commented multi-line `ddl` skip the NOT-RENDERED fallback.

    It is the check standing between `render_ddl` and its own core promise, and it had no
    direct test: `all(...)` → `any(...)` left the whole suite green while
    `ddl='-- note\\nDROP TABLE users;'` was emitted verbatim, bare and executable. Each case
    below kills a distinct leniency mutation — `any`, `startswith("-")`, tolerating blank
    lines, and `.lstrip()`-ing before the check — so the guard cannot be widened silently.
    """
    assert _is_fully_commented("-- one line") is True
    assert _is_fully_commented("--   indexes:\n--     - columns: ['status']") is True
    # `any` instead of `all`: a first line that is a comment must not vouch for the rest.
    assert _is_fully_commented("-- note\nDROP TABLE users;") is False
    assert _is_fully_commented("DROP TABLE users;\n-- note") is False
    # A single dash is not a SQL comment; `startswith("-")` would accept this.
    assert _is_fully_commented("- note\n- more") is False
    # A blank line is not a comment line. This one is about intent rather than safety — a
    # blank line is inert either way, and a raw statement after one is still rejected (the
    # next case) — but tolerating it silently widens which `ddl` values skip the fallback,
    # and no generated block needs it: `_comment_block` prefixes every line it emits.
    assert _is_fully_commented("-- note\n\n-- more") is False
    assert _is_fully_commented("-- note\n\nDROP TABLE users;") is False
    # An indented comment is not safe to emit verbatim: `psql` is fine with it, but the
    # guard's premise is "already inert on every line as written", and `.lstrip()` would
    # extend it to text this renderer has no other reason to trust.
    assert _is_fully_commented("  -- note") is False
    # Nothing at all is not "every line is a comment".
    assert _is_fully_commented("") is False


def test_a_partially_commented_ddl_is_never_emitted_bare():
    """The invariant `_is_fully_commented` protects, asserted through the renderer.

    A multi-line `ddl` whose first line is a comment and whose later lines are raw
    statements is exactly the hazard the fallback exists for, and widening the guard routes
    it around the fallback and prints it verbatim.
    """
    script = PostgresWorkloadAdapter().render_ddl(
        [_ddl_proposal(ddl="-- a note\nDROP TABLE users;")]
    )
    assert "NOT RENDERED" in script
    assert _uncommented(script) == [], script
    assert "-- DROP TABLE users;" in script
