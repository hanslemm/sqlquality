"""Tests for `sqlquality.verify`'s matching layer: `proposal_key`, `index_proposals`,
`group_index`.

Every evidence shape used here is copied from the real rule that emits it (see
`src/sqlquality/workload/postgres.py` and `src/sqlquality/workload/redshift.py`), not
invented, so a test failure here means the real rule's evidence would be mis-keyed too.
"""

from __future__ import annotations

import pytest

from sqlquality.verify import ProposalKeyCollisionError, group_index, index_proposals, proposal_key


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
    column. Evidence copied verbatim from ~lines 793 and 823."""
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
    key_vacuum = proposal_key(vacuum)
    key_analyze = proposal_key(analyze)
    assert key_vacuum is not None
    assert key_analyze is not None
    assert key_vacuum != key_analyze
    # Both still key to the same relation as a prefix — only the trailing discriminator differs.
    assert key_vacuum[:3] == ("ADV104", "public", "orders")
    assert key_analyze[:3] == ("ADV104", "public", "orders")


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


# --- index_proposals ---


def test_index_proposals_builds_one_entry_per_distinct_key():
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
            # Unkeyable — must be silently excluded, not raised on.
            {"code": "ADV999", "evidence": {}},
        ]
    }
    index = index_proposals(payload)
    assert len(index) == 2
    titles = {key: proposal["title"] for key, proposal in index.items()}
    assert "Run VACUUM on public.orders" in titles.values()
    assert "Run ANALYZE on public.orders" in titles.values()


def test_index_proposals_raises_rather_than_silently_dropping_on_a_genuine_key_collision():
    """A dict comprehension that overwrites is the defect this guards against: contrive two
    distinct proposals that share a relation but carry none of `columns`/`column`/`index`
    and no `ddl` either (a hypothetical future rule's evidence shape `proposal_key` cannot
    yet discriminate) and confirm the losing proposal does not simply vanish."""
    payload = {
        "proposals": [
            {
                "code": "ADV999",
                "title": "First finding for public.orders",
                "evidence": {"schema": "public", "table": "orders"},
                "ddl": None,
            },
            {
                "code": "ADV999",
                "title": "Second, different finding for public.orders",
                "evidence": {"schema": "public", "table": "orders"},
                "ddl": None,
            },
        ]
    }
    with pytest.raises(ProposalKeyCollisionError):
        index_proposals(payload)


def test_index_proposals_ignores_a_payload_with_no_proposals_list():
    assert index_proposals({}) == {}
    assert index_proposals({"proposals": "not-a-list"}) == {}


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
