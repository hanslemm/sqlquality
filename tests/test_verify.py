"""Tests for `sqlquality.verify`'s matching layer: `proposal_key`, `index_proposals`,
`group_index`.

Every evidence shape used here is copied from the real rule that emits it (see
`src/sqlquality/workload/postgres.py` and `src/sqlquality/workload/redshift.py`), not
invented, so a test failure here means the real rule's evidence would be mis-keyed too.
"""

from __future__ import annotations

from sqlquality.verify import ProposalIndex, group_index, index_proposals, proposal_key


# --- relation-scoped, plural `columns` (ADV001, ADV002, ADV003, ADV004, ADV007, ADV008) ---


def test_a_relation_scoped_proposal_keys_on_code_relation_and_columns():
    key = proposal_key(
        {
            "code": "ADV001",
            "evidence": {
                "schema": "public",
                "table": "orders",
                "columns": ["status", "created_at"],
            },
        }
    )
    assert key == ("ADV001", "public", "orders", "status", "created_at")


def test_the_same_recommendation_in_two_runs_produces_the_same_key():
    """Column order is part of the recommendation — `(a, b)` and `(b, a)` are different
    indexes, and `advise` already keeps both when it proposes them."""
    a = {"code": "ADV001", "evidence": {"schema": "s", "table": "t", "columns": ["a", "b"]}}
    b = {"code": "ADV001", "evidence": {"schema": "s", "table": "t", "columns": ["b", "a"]}}
    assert proposal_key(a) != proposal_key(b)

    identical_a = {
        "code": "ADV001",
        "evidence": {"schema": "s", "table": "t", "columns": ["a", "b"]},
    }
    assert proposal_key(a) == proposal_key(identical_a)


# --- relation-scoped, singular `column` (ADV005's sargability branch, ADV101, ADV102) ---
# Correction 1: a `columns`-only key silently discards the column here. Each evidence dict
# below is the real shape from the rule named.


def test_adv005_sargability_keys_on_code_relation_and_singular_column():
    """postgres.py's `propose_sargability`, ~line 1112 — the relation-scoped branch."""
    key = proposal_key(
        {
            "code": "ADV005",
            "evidence": {
                "schema": "public",
                "table": "events",
                "column": "payload",
                "cost_share": 0.4,
                "calls": 12,
                "fingerprints": 3,
            },
        }
    )
    assert key == ("ADV005", "public", "events", "payload")


def test_adv101_sortkey_keys_on_code_relation_and_singular_column():
    """redshift.py's `propose_sortkey`, ~line 431."""
    key = proposal_key(
        {
            "code": "ADV101",
            "evidence": {
                "schema": "public",
                "table": "orders",
                "column": "created_at",
                "role": "range",
                "cost_share": 0.5,
                "calls": 10,
                "current_sortkey1": None,
                "stats_off": None,
            },
        }
    )
    assert key == ("ADV101", "public", "orders", "created_at")


def test_adv102_distkey_keys_on_code_relation_and_singular_column():
    """redshift.py's `propose_distkey`, ~line 560."""
    key = proposal_key(
        {
            "code": "ADV102",
            "evidence": {
                "schema": "public",
                "table": "orders",
                "column": "customer_id",
                "role": "join",
                "cost_share": 0.5,
                "calls": 10,
                "current_diststyle": None,
                "skew_rows": None,
                "stats_off": None,
            },
        }
    )
    assert key == ("ADV102", "public", "orders", "customer_id")


def test_a_changed_sortkey_recommendation_on_the_same_table_gets_a_different_key():
    """If run A proposes SORTKEY(created_at) and run B proposes SORTKEY(tenant_id) on the
    same table, a `columns`-only key would collapse both to the same key and `verify`
    would report "still present, unchanged" for advice that actually changed."""
    run_a = {
        "code": "ADV101",
        "evidence": {"schema": "public", "table": "orders", "column": "created_at"},
    }
    run_b = {
        "code": "ADV101",
        "evidence": {"schema": "public", "table": "orders", "column": "tenant_id"},
    }
    assert proposal_key(run_a) != proposal_key(run_b)


# --- statement-scoped: no relation at all (ADV005's wildcard branch, ADV006) ---


def test_a_statement_scoped_finding_keys_on_code_and_fingerprint():
    """ADV005's wildcard branch and ADV006 carry no schema/table/columns triple. Keying them
    the same way would match nothing and report every one as `disappeared`."""
    assert proposal_key({"code": "ADV005", "evidence": {"fingerprint": "abc123abc123"}}) == (
        "ADV005",
        "abc123abc123",
    )
    assert proposal_key(
        {
            "code": "ADV006",
            "evidence": {"tables": ["public.orders"], "fingerprint": "def456def456"},
        }
    ) == ("ADV006", "def456def456")


def test_a_proposal_with_both_a_relation_and_a_fingerprint_keys_by_the_relation():
    """No rule emits both today (every relation-scoped rule carries `fingerprint_digests`,
    never a singular `fingerprint`), but the branch precedence must not be an accident: a
    proposal naming a table is identified by that table, not by incidentally which single
    query happened to trigger it this run. If this ever flipped, every relation-scoped key
    would silently degrade from a 4+-tuple to a 2-tuple the moment a rule's evidence grew a
    `fingerprint` field."""
    proposal = {
        "code": "ADV001",
        "evidence": {
            "schema": "public",
            "table": "orders",
            "columns": ["status"],
            "fingerprint": "shouldnotwin1",
        },
    }
    assert proposal_key(proposal) == ("ADV001", "public", "orders", "status")


# --- unkeyable ---


def test_a_proposal_with_neither_shape_is_unkeyable_rather_than_mis_keyed():
    """Returning a partial key would silently group unrelated proposals together."""
    assert proposal_key({"code": "ADV999", "evidence": {}}) is None


def test_a_proposal_with_a_malformed_evidence_value_is_unkeyable():
    """`evidence` is `dict[str, object]` in the real payload, so its shape is not
    guaranteed — a proposal whose `evidence` is not even a dict must not raise, and must
    not be mis-keyed either."""
    assert proposal_key({"code": "ADV999", "evidence": "not-a-dict"}) is None
    assert proposal_key({"code": "ADV999"}) is None
    assert proposal_key({"evidence": {"schema": "s", "table": "t"}}) is None


# --- Correction 2: two ADV104 proposals for one relation, no column/index to tell them apart ---


def test_two_adv104_proposals_for_one_relation_get_distinct_keys():
    """redshift.py's `propose_maintenance` — VACUUM and ANALYZE are two independent `if`
    statements (not `if`/`elif`), so a relation whose `unsorted` *and* `stats_off` both
    cross their thresholds emits two ADV104 proposals with identical schema/table and no
    column. Evidence copied verbatim from ~lines 793 and 823. Asserted as full tuples, not
    only `!=` and a shared prefix: a mutant swapping the `ddl` fallback for `proposal["title"]`
    still makes these two keys unequal (the titles differ too), so only pinning the exact
    expected tuple — which encodes `ddl` specifically, not just "some string differs" —
    catches that substitution."""
    vacuum = {
        "code": "ADV104",
        "title": "Run VACUUM on public.orders",
        "evidence": {"schema": "public", "table": "orders", "unsorted": 30.0, "row_estimate": 1000},
        "ddl": "VACUUM public.orders;",
    }
    analyze = {
        "code": "ADV104",
        "title": "Run ANALYZE on public.orders",
        "evidence": {
            "schema": "public",
            "table": "orders",
            "stats_off": 20.0,
            "row_estimate": 1000,
        },
        "ddl": "ANALYZE public.orders;",
    }
    assert proposal_key(vacuum) == ("ADV104", "public", "orders", "VACUUM public.orders;")
    assert proposal_key(analyze) == ("ADV104", "public", "orders", "ANALYZE public.orders;")


def test_two_indexes_on_the_same_columns_get_distinct_keys_by_index_name():
    """Two indexes on the *same columns* is exactly what ADV003 exists to detect, so two
    ADV002 (unused-index) proposals can legitimately share `(code, schema, table, columns)`
    and be distinguished only by `index`. Evidence copied verbatim from
    postgres.py's `propose_unused_indexes`, ~line 801."""
    drop_a = {
        "code": "ADV002",
        "evidence": {
            "schema": "public",
            "table": "orders",
            "index": "orders_customer_id_idx",
            "columns": ["customer_id"],
            "scans": 0,
            "size_bytes": 1000,
        },
    }
    drop_b = {
        "code": "ADV002",
        "evidence": {
            "schema": "public",
            "table": "orders",
            "index": "orders_customer_id_idx2",
            "columns": ["customer_id"],
            "scans": 0,
            "size_bytes": 900,
        },
    }
    key_a = proposal_key(drop_a)
    key_b = proposal_key(drop_b)
    assert key_a is not None
    assert key_b is not None
    assert key_a != key_b
    assert key_a == ("ADV002", "public", "orders", "customer_id", "orders_customer_id_idx")
    assert key_b == ("ADV002", "public", "orders", "customer_id", "orders_customer_id_idx2")


# --- Minor 7 (review round 1): ADV004's guard must be folded into the key ---


def test_adv004_partial_index_keys_include_the_guard_column_and_predicate():
    """`WHERE shipped_at IS NULL` and `WHERE cancelled_at IS NOT NULL`, both restricting a
    partial index on the same leading column, are different proposals. Evidence copied
    verbatim from postgres.py's `propose_partial_indexes`, ~line 1049."""
    shipped_at_guard = {
        "code": "ADV004",
        "evidence": {
            "schema": "public",
            "table": "orders",
            "columns": ["customer_id"],
            "guard_column": "shipped_at",
            "guard_predicate": "IS NULL",
        },
    }
    cancelled_at_guard = {
        "code": "ADV004",
        "evidence": {
            "schema": "public",
            "table": "orders",
            "columns": ["customer_id"],
            "guard_column": "cancelled_at",
            "guard_predicate": "IS NOT NULL",
        },
    }
    assert proposal_key(shipped_at_guard) == (
        "ADV004",
        "public",
        "orders",
        "customer_id",
        "shipped_at",
        "IS NULL",
    )
    assert proposal_key(cancelled_at_guard) == (
        "ADV004",
        "public",
        "orders",
        "customer_id",
        "cancelled_at",
        "IS NOT NULL",
    )
    assert proposal_key(shipped_at_guard) != proposal_key(cancelled_at_guard)


# --- Critical 1 / Important 2 (review round 1): ADV105 must not key on its own `ddl` ---


def test_adv105_two_recommendations_with_a_null_ddl_get_distinct_keys():
    """`propose_advisor` evidence (`redshift.py` ~line 906): `recommended_ddl` is `str |
    None` — Amazon Redshift Advisor's own output, not sqlquality's. Two Advisor rows for
    one relation with different `rec_type` and a NULL `recommended_ddl` both fell back to
    `ddl` (`None`) before this fix and produced the identical key `("ADV105", schema,
    table)`, which made `index_proposals` treat two live recommendations as one ambiguous
    collision. `evidence["recommendation_type"]` must be used instead and take priority
    over the `ddl` fallback."""
    sort_rec = {
        "code": "ADV105",
        "title": "Amazon Redshift Advisor recommends a sort change for public.orders",
        "evidence": {
            "schema": "public",
            "table": "orders",
            "source": "amazon_redshift_advisor",
            "recommendation_type": "sort",
            "current_ddl": None,
            "recommended_ddl": None,
        },
        "ddl": None,
    }
    dist_rec = {
        "code": "ADV105",
        "title": "Amazon Redshift Advisor recommends a dist change for public.orders",
        "evidence": {
            "schema": "public",
            "table": "orders",
            "source": "amazon_redshift_advisor",
            "recommendation_type": "dist",
            "current_ddl": None,
            "recommended_ddl": None,
        },
        "ddl": None,
    }
    assert proposal_key(sort_rec) == ("ADV105", "public", "orders", "sort")
    assert proposal_key(dist_rec) == ("ADV105", "public", "orders", "dist")
    assert proposal_key(sort_rec) != proposal_key(dist_rec)


def test_adv105_key_is_stable_even_if_advisors_relayed_ddl_text_drifts():
    """`recommended_ddl` is Advisor's own verbatim text — whitespace or column-ordering
    drift between two runs of the *same* recommendation must not change the key, or
    `verify` would report one stable Advisor recommendation as both `disappeared` and
    `new`. The key must come from `recommendation_type` alone, never from `ddl`, once a
    discriminator already exists."""
    run_a = {
        "code": "ADV105",
        "evidence": {
            "schema": "public",
            "table": "orders",
            "recommendation_type": "sort",
            "recommended_ddl": "ALTER TABLE public.orders ALTER SORTKEY (a, b);",
        },
        "ddl": "ALTER TABLE public.orders ALTER SORTKEY (a, b);",
    }
    run_b = {
        "code": "ADV105",
        "evidence": {
            "schema": "public",
            "table": "orders",
            "recommendation_type": "sort",
            "recommended_ddl": "ALTER TABLE public.orders ALTER SORTKEY (a,b);",
        },
        "ddl": "ALTER TABLE public.orders ALTER SORTKEY (a,b);",
    }
    assert proposal_key(run_a) == proposal_key(run_b) == ("ADV105", "public", "orders", "sort")


# --- index_proposals ---


def test_index_proposals_builds_one_matched_entry_per_distinct_key():
    payload = {
        "proposals": [
            {
                "code": "ADV104",
                "title": "Run VACUUM on public.orders",
                "evidence": {"schema": "public", "table": "orders", "unsorted": 30.0},
                "ddl": "VACUUM public.orders;",
            },
            {
                "code": "ADV104",
                "title": "Run ANALYZE on public.orders",
                "evidence": {"schema": "public", "table": "orders", "stats_off": 20.0},
                "ddl": "ANALYZE public.orders;",
            },
            # Unkeyable — must be silently excluded, not treated as a collision.
            {"code": "ADV999", "evidence": {}},
        ]
    }
    result = index_proposals(payload)
    assert isinstance(result, ProposalIndex)
    assert len(result.matched) == 2
    assert result.collisions == {}
    titles = {proposal["title"] for proposal in result.matched.values()}
    assert titles == {"Run VACUUM on public.orders", "Run ANALYZE on public.orders"}


def test_index_proposals_discloses_rather_than_silently_dropping_on_a_genuine_key_collision():
    """A dict comprehension that overwrites is the defect this guards against: contrive two
    distinct proposals that share a relation but carry none of the discriminators
    `proposal_key` currently applies (a hypothetical future rule's evidence shape it cannot
    yet tell apart) and confirm neither is lost — both surface in `.collisions`, and the
    ambiguous key is absent from `.matched` rather than pointing at whichever proposal
    happened to be seen last."""
    first = {
        "code": "ADV999",
        "title": "First finding for public.orders",
        "evidence": {"schema": "public", "table": "orders"},
        "ddl": None,
    }
    second = {
        "code": "ADV999",
        "title": "Second, different finding for public.orders",
        "evidence": {"schema": "public", "table": "orders"},
        "ddl": None,
    }
    result = index_proposals({"proposals": [first, second]})
    key = ("ADV999", "public", "orders")
    assert key not in result.matched
    assert key in result.collisions
    assert result.collisions[key] == (first, second)


def test_index_proposals_ignores_a_payload_with_no_proposals_list():
    assert index_proposals({}) == ProposalIndex(matched={}, collisions={})
    assert index_proposals({"proposals": "not-a-list"}) == ProposalIndex(matched={}, collisions={})


# --- group_index ---


def test_group_index_maps_digest_to_its_measurements():
    payload = {
        "query_groups": [
            {"digest": "aaa", "calls": 10, "total_time_ms": 100.0, "mean_ms": 10.0},
        ]
    }
    assert group_index(payload)["aaa"]["mean_ms"] == 10.0


def test_group_index_preserves_a_null_mean_ms_rather_than_coercing_it():
    """`mean_ms` is `None` when `calls` is 0 — `0.0` would read as "instantaneous", the
    opposite of "unknown". `group_index` must not coerce it either."""
    payload = {
        "query_groups": [
            {"digest": "bbb", "calls": 0, "total_time_ms": 0.0, "mean_ms": None},
        ]
    }
    group = group_index(payload)["bbb"]
    assert group["calls"] == 0
    assert group["mean_ms"] is None


def test_group_index_skips_a_group_with_no_usable_digest():
    payload = {"query_groups": [{"calls": 1, "total_time_ms": 1.0, "mean_ms": 1.0}]}
    assert group_index(payload) == {}


def test_group_index_on_a_payload_with_no_query_groups_list():
    assert group_index({}) == {}
