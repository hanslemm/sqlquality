"""Tests for `sqlquality.verify`'s matching layer: `proposal_key`, `index_proposals`,
`group_index`; and its window-classification layer: `WindowRelation`, `classify_windows`,
`confidence_for`, `window_limits`.

Every evidence shape used here is copied from the real rule that emits it (see
`src/sqlquality/workload/postgres.py` and `src/sqlquality/workload/redshift.py`), not
invented, so a test failure here means the real rule's evidence would be mis-keyed too.
The window shapes below are likewise copied from `report.py`'s real `"window"` object
(`description`, `engine`, `stats_reset_at`, `since`, `since_duration_seconds`, `limit`,
all keys always present, values nullable) — see `postgres.py`'s and `redshift.py`'s
`window_facts()` for which fields each engine can and cannot supply.
"""

from __future__ import annotations

from sqlquality.models import Confidence
from sqlquality.verify import (
    ProposalIndex,
    WindowLimits,
    WindowRelation,
    classify_windows,
    confidence_for,
    group_index,
    index_proposals,
    proposal_key,
    window_limits,
)


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


# --- classify_windows / confidence_for / window_limits ---
#
# `classify_windows` takes the two top-level payload dicts (matching
# `classify_windows({"window": w}, {"window": dict(w)})` in the brief), not the `"window"`
# sub-objects directly, and reads `["window"]` out of each itself.
#
# Round-1 review of this task's first pass (Important 1-4) found three cells graded
# higher than their evidence supported, all from the same root cause: `since` treated as
# the field that determines comparability, when it is `since_duration_seconds` (added to
# the payload in this fix round, sanctioned by Hans — see CHANGELOG.md and both adapters'
# `window_facts()`) that actually does. Every test below that touches `since`-based
# comparability now sets `since_duration_seconds` explicitly, and `since` (the absolute
# cutoff) is included in fixtures only for realism — `classify_windows` no longer reads it
# at all.


def test_the_same_stats_reset_means_the_later_window_contains_the_earlier():
    """This is the common Postgres case — baselined last week, never reset. The comparison
    is still worth making, but a real improvement is diluted by pre-change executions, so
    it is graded MEDIUM rather than HIGH and the report says so."""
    w = {"engine": "postgres", "stats_reset_at": "2026-08-01T00:00:00", "since": None, "limit": 500}
    assert classify_windows({"window": w}, {"window": dict(w)}) is WindowRelation.NESTED
    assert confidence_for(WindowRelation.NESTED) is Confidence.MEDIUM


def test_two_engines_are_never_comparable():
    """A Postgres mean and a Redshift mean measure different servers. Grading this at all
    would be inventing a comparison.

    This is the brief's own mandated test, kept verbatim — but its payload is already
    INCOMPARABLE from its `stats_reset_at`/`since` shape alone (null-vs-value reset,
    one-sided `since`), so the engine comparison never actually has to fire to pass it.
    See `test_ruling_3_engine_mismatch_beats_a_would_be_disjoint_pair` below for the test
    that isolates the engine gate as the *only* disqualifier (round-1 review, Important 1)."""
    before = {"window": {"engine": "postgres", "stats_reset_at": "T", "since": None, "limit": 500}}
    after = {"window": {"engine": "redshift", "stats_reset_at": None, "since": "T2", "limit": 500}}
    assert classify_windows(before, after) is WindowRelation.INCOMPARABLE
    assert confidence_for(WindowRelation.INCOMPARABLE) is Confidence.LOW


# --- Ruling 3 (round-1 review, Important 1 / Minor 5): the engine gate, isolated ---


def test_ruling_3_engine_mismatch_beats_a_would_be_disjoint_pair():
    """Round-1 review, Important 1. Every other field here would grade `DISJOINT`/`HIGH`
    on its own (both `stats_reset_at` non-null and different, both `since_duration_seconds`
    null) — the *only* disqualifier is the differing `engine`. This kills both the
    deletion mutation (dropping the `!=` check entirely) and the reordering mutation
    (moving the engine gate below the reset-based check), since either one lets this
    exact pair fall through to `DISJOINT`."""
    before = {
        "window": {
            "engine": "postgres",
            "stats_reset_at": "2026-08-01T00:00:00",
            "since": None,
            "since_duration_seconds": None,
            "limit": 500,
        }
    }
    after = {
        "window": {
            "engine": "redshift",
            "stats_reset_at": "2026-08-04T00:00:00",
            "since": None,
            "since_duration_seconds": None,
            "limit": 500,
        }
    }
    assert classify_windows(before, after) is WindowRelation.INCOMPARABLE


def test_ruling_3_a_missing_engine_on_both_sides_is_incomparable_not_nested():
    """Round-1 review, Minor 5. Dropping the `isinstance(..., str)` checks and keeping
    only `!=` lets two windows with `engine` absent on both sides read as equal (`None ==
    None`), and this pair would otherwise grade `NESTED`/`MEDIUM` (matching non-null
    `stats_reset_at`, no `since_duration_seconds` on either side) — a fully engine-less
    artifact must not out-grade one that at least names its engine."""
    before_window = {
        "stats_reset_at": "2026-08-01T00:00:00",
        "since": None,
        "since_duration_seconds": None,
        "limit": 500,
    }
    assert (
        classify_windows({"window": dict(before_window)}, {"window": dict(before_window)})
        is WindowRelation.INCOMPARABLE
    )


def test_disjoint_requires_both_stats_reset_at_non_null_and_different():
    """Two Postgres runs whose `stats_reset_at` genuinely differ: the counters were
    cleared between them, so they are two independent samples — the strongest relation."""
    before = {
        "window": {
            "engine": "postgres",
            "stats_reset_at": "2026-08-01T00:00:00",
            "since": None,
            "since_duration_seconds": None,
            "limit": 500,
        }
    }
    after = {
        "window": {
            "engine": "postgres",
            "stats_reset_at": "2026-08-04T00:00:00",
            "since": None,
            "since_duration_seconds": None,
            "limit": 500,
        }
    }
    assert classify_windows(before, after) is WindowRelation.DISJOINT
    assert confidence_for(WindowRelation.DISJOINT) is Confidence.HIGH


def test_ruling_1_a_null_stats_reset_at_paired_with_a_timestamp_is_not_disjoint():
    """The brief's own wording ('differs between runs') would fire for a null-versus-
    timestamp pair. That pair is *missing information* on one side, not evidence the
    counters were cleared, so it must not be graded DISJOINT — or anything but
    INCOMPARABLE, since nothing else can be concluded either."""
    before = {
        "window": {
            "engine": "redshift",
            "stats_reset_at": None,
            "since": None,
            "since_duration_seconds": None,
            "limit": 500,
        }
    }
    after = {
        "window": {
            "engine": "redshift",
            "stats_reset_at": "2026-08-04T00:00:00",
            "since": None,
            "since_duration_seconds": None,
            "limit": 500,
        }
    }
    assert classify_windows(before, after) is WindowRelation.INCOMPARABLE


# --- COMPARABLE (round-1 review, Important 3): gated on duration, not the absolute cutoff ---


def test_comparable_requires_equal_since_duration_seconds_not_equal_since_cutoff():
    """Round-1 review, Important 3. Two Redshift runs a week apart with the *same*
    `--since 7d` bind two *different* absolute cutoffs (Redshift's adapter records
    `datetime.now() - since` at microsecond precision) — gating on the cutoff made
    `COMPARABLE` unreachable from any real pair of runs. Gating on
    `since_duration_seconds` instead (both `604800.0`, i.e. 7 days, here) is what actually
    reaches `COMPARABLE`/`HIGH`, regardless of `stats_reset_at` (Redshift never sets it)
    and regardless of the differing `since` cutoffs kept in the fixture for realism."""
    before = {
        "window": {
            "engine": "redshift",
            "stats_reset_at": None,
            "since": "2026-08-01T00:00:00",
            "since_duration_seconds": 604800.0,
            "limit": 500,
        }
    }
    after = {
        "window": {
            "engine": "redshift",
            "stats_reset_at": None,
            "since": "2026-08-08T00:00:00",
            "since_duration_seconds": 604800.0,
            "limit": 200,
        }
    }
    assert classify_windows(before, after) is WindowRelation.COMPARABLE
    assert confidence_for(WindowRelation.COMPARABLE) is Confidence.HIGH


def test_ruling_2_both_stats_reset_at_and_both_since_null_is_incomparable_not_nested():
    """Under the brief's literal wording, two nulls read as 'same `stats_reset_at`, no
    `since`' and would be graded NESTED/MEDIUM. Two nulls mean nothing is known about
    either window's extent — NESTED's MEDIUM must be earned by the demonstrated fact
    that counters were not reset, never handed out for an absence of information."""
    before = {
        "window": {
            "engine": "redshift",
            "stats_reset_at": None,
            "since": None,
            "since_duration_seconds": None,
            "limit": 500,
        }
    }
    after = {
        "window": {
            "engine": "redshift",
            "stats_reset_at": None,
            "since": None,
            "since_duration_seconds": None,
            "limit": 500,
        }
    }
    assert classify_windows(before, after) is WindowRelation.INCOMPARABLE


# --- Ruling 4: since_duration_seconds set on one side only ---


def test_ruling_4_since_duration_set_on_only_one_side_is_incomparable():
    """One run filtered its window with `--since` and the other did not; there is no
    defensible relation between the two, even if `stats_reset_at` happens to match (here,
    both null — realistic for a Redshift pair)."""
    before = {
        "window": {
            "engine": "redshift",
            "stats_reset_at": None,
            "since": "2026-08-01T00:00:00",
            "since_duration_seconds": 604800.0,
            "limit": 500,
        }
    }
    after = {
        "window": {
            "engine": "redshift",
            "stats_reset_at": None,
            "since": None,
            "since_duration_seconds": None,
            "limit": 500,
        }
    }
    assert classify_windows(before, after) is WindowRelation.INCOMPARABLE


def test_ruling_4_one_sided_since_duration_is_incomparable_even_with_matching_reset():
    """Even when `stats_reset_at` matches on both sides (a shape no real engine produces
    alongside a `since_duration_seconds` value today, but a malformed or hand-built
    artifact could), `since_duration_seconds` set on only one side must still read as
    INCOMPARABLE, not NESTED — the duration check is decided in full before
    `stats_reset_at` is ever consulted, so a matching reset cannot rescue a one-sided
    duration."""
    before = {
        "window": {
            "engine": "postgres",
            "stats_reset_at": "2026-08-01T00:00:00",
            "since": "2026-08-01T00:00:00",
            "since_duration_seconds": 604800.0,
            "limit": 500,
        }
    }
    after = {
        "window": {
            "engine": "postgres",
            "stats_reset_at": "2026-08-01T00:00:00",
            "since": None,
            "since_duration_seconds": None,
            "limit": 500,
        }
    }
    assert classify_windows(before, after) is WindowRelation.INCOMPARABLE


def test_ruling_4_one_sided_since_duration_beats_a_would_be_disjoint_reset():
    """Round-1 review, Important 2 — the core fix. `stats_reset_at` differs (which alone
    would grade `DISJOINT`/`HIGH`) *and* `since_duration_seconds` is set on only one side.
    The `since_duration_seconds` mismatch must win: a user who filtered one run with
    `--since` and not the other gets no claim to `HIGH` merely because the counters also
    happen to look cleared. Before this fix, the reset-based `DISJOINT` check ran first
    and dominated every `since` shape, including this one."""
    before = {
        "window": {
            "engine": "postgres",
            "stats_reset_at": "2026-08-01T00:00:00",
            "since": "2026-08-01T00:00:00",
            "since_duration_seconds": 604800.0,
            "limit": 500,
        }
    }
    after = {
        "window": {
            "engine": "postgres",
            "stats_reset_at": "2026-08-04T00:00:00",
            "since": None,
            "since_duration_seconds": None,
            "limit": 500,
        }
    }
    assert classify_windows(before, after) is WindowRelation.INCOMPARABLE


# --- Ruling 5: since_duration_seconds set on both sides but unequal ---


def test_ruling_5_unequal_since_duration_is_incomparable():
    """Two different explicit durations is a different claim from the reset-driven
    containment that earns NESTED its MEDIUM — reusing that reasoning here would be a
    different claim wearing the same label, so this is graded exactly as no relation at
    all, not as a directional containment."""
    before = {
        "window": {
            "engine": "redshift",
            "stats_reset_at": None,
            "since": "2026-08-01T00:00:00",
            "since_duration_seconds": 604800.0,
            "limit": 500,
        }
    }
    after = {
        "window": {
            "engine": "redshift",
            "stats_reset_at": None,
            "since": "2026-08-01T00:00:00",
            "since_duration_seconds": 259200.0,
            "limit": 500,
        }
    }
    assert classify_windows(before, after) is WindowRelation.INCOMPARABLE


def test_ruling_5_unequal_since_duration_beats_a_would_be_disjoint_reset():
    """Round-1 review, Important 2 — the other core-fix cell. `stats_reset_at` differs
    (would grade `DISJOINT`/`HIGH` alone) *and* both sides set `since_duration_seconds`,
    but to different values (7 days vs. 3 days). The duration mismatch must still win."""
    before = {
        "window": {
            "engine": "postgres",
            "stats_reset_at": "2026-08-01T00:00:00",
            "since": "2026-08-01T00:00:00",
            "since_duration_seconds": 604800.0,
            "limit": 500,
        }
    }
    after = {
        "window": {
            "engine": "postgres",
            "stats_reset_at": "2026-08-04T00:00:00",
            "since": "2026-08-01T00:00:00",
            "since_duration_seconds": 259200.0,
            "limit": 500,
        }
    }
    assert classify_windows(before, after) is WindowRelation.INCOMPARABLE


def test_comparable_is_checked_before_disjoint_when_both_conditions_hold():
    """No real payload can produce both a differing, non-null `stats_reset_at` pair and an
    equal, non-null `since_duration_seconds` pair at once (Postgres never sets
    `since_duration_seconds`; Redshift never sets `stats_reset_at`), but a malformed or
    hand-built artifact could. Unlike this task's first pass, `COMPARABLE`'s equal-
    duration fact now wins over `DISJOINT`'s reset-based fact, matching the precedence
    fixed for Important 2 above: *any* `since_duration_seconds` evidence — matching or
    mismatched — is decided before `stats_reset_at` is ever consulted, not only the
    mismatched cases."""
    before = {
        "window": {
            "engine": "postgres",
            "stats_reset_at": "2026-08-01T00:00:00",
            "since": "2026-08-01T00:00:00",
            "since_duration_seconds": 604800.0,
            "limit": 500,
        }
    }
    after = {
        "window": {
            "engine": "postgres",
            "stats_reset_at": "2026-08-04T00:00:00",
            "since": "2026-08-01T00:00:00",
            "since_duration_seconds": 604800.0,
            "limit": 500,
        }
    }
    assert classify_windows(before, after) is WindowRelation.COMPARABLE


def test_classify_windows_does_not_raise_on_a_malformed_or_partial_window():
    """Task 7 validates the payload shape before this runs; this function must not assume
    that already happened. A missing `"window"` key, a `"window"` that is not a dict, and
    a window missing/mistyping `engine` must all degrade to INCOMPARABLE rather than
    raise."""
    good = {"window": {"engine": "postgres", "stats_reset_at": "T", "since": None, "limit": 1}}
    assert classify_windows({}, good) is WindowRelation.INCOMPARABLE
    assert classify_windows({"window": "not-a-dict"}, good) is WindowRelation.INCOMPARABLE
    assert classify_windows({"window": {}}, good) is WindowRelation.INCOMPARABLE
    assert (
        classify_windows({"window": {"engine": None, "stats_reset_at": "T"}}, good)
        is WindowRelation.INCOMPARABLE
    )
    assert classify_windows({}, {}) is WindowRelation.INCOMPARABLE


# --- Minor 6 (round-1 review): a malformed *value* must not earn a grade ---


def test_a_non_string_stats_reset_at_does_not_earn_nested():
    """`stats_reset_at` compared with bare `==` would let a matching non-`str` value (a
    list, here) earn `NESTED` — `_window_string`'s `isinstance` guard must reject it as
    unknown instead."""
    window = {
        "engine": "postgres",
        "stats_reset_at": ["T"],
        "since": None,
        "since_duration_seconds": None,
        "limit": 500,
    }
    assert classify_windows({"window": dict(window)}, {"window": dict(window)}) is (
        WindowRelation.INCOMPARABLE
    )


def test_a_nan_stats_reset_at_does_not_earn_disjoint():
    """Python's `json` module accepts a bare `NaN` as a non-standard extension, and
    `float("nan") != float("nan")` is `True` — so an untyped `!=` would grade two `NaN`
    `stats_reset_at` values `DISJOINT`/`HIGH`, the most severe possible confidence
    inflation from a malformed value. `_window_string`'s `isinstance(..., str)` guard
    rejects a `float` outright, regardless of its value."""
    window = {
        "engine": "postgres",
        "stats_reset_at": float("nan"),
        "since": None,
        "since_duration_seconds": None,
        "limit": 500,
    }
    assert classify_windows({"window": dict(window)}, {"window": dict(window)}) is (
        WindowRelation.INCOMPARABLE
    )


def test_a_non_finite_since_duration_does_not_earn_comparable():
    """The mirror image of the `NaN` case above, on the other side of the equality:
    `float("inf") == float("inf")` is `True`, so two `since_duration_seconds: Infinity`
    values would otherwise earn `COMPARABLE`/`HIGH` — `_window_duration_seconds` rejects
    non-finite floats explicitly rather than relying on inequality to save it the way
    `_window_string` can for `NaN`."""
    window = {
        "engine": "redshift",
        "stats_reset_at": None,
        "since": None,
        "since_duration_seconds": float("inf"),
        "limit": 500,
    }
    assert classify_windows({"window": dict(window)}, {"window": dict(window)}) is (
        WindowRelation.INCOMPARABLE
    )


def test_confidence_for_grades_every_relation_per_the_spec():
    """A classifier where three of four rungs are pinned and one is only implied is
    precisely the defect this checks against: every member of `WindowRelation` is
    asserted individually, and the set comparison catches a fifth relation being added
    without a matching confidence entry."""
    expected = {
        WindowRelation.DISJOINT: Confidence.HIGH,
        WindowRelation.COMPARABLE: Confidence.HIGH,
        WindowRelation.NESTED: Confidence.MEDIUM,
        WindowRelation.INCOMPARABLE: Confidence.LOW,
    }
    assert set(WindowRelation) == set(expected)
    for relation, confidence in expected.items():
        assert confidence_for(relation) is confidence


# --- window_limits (Ruling 6; round-1 review Important 4 and Minor 7) ---


def test_window_limits_returns_the_raw_pair():
    before = {"window": {"engine": "postgres", "stats_reset_at": "T", "since": None, "limit": 500}}
    after = {"window": {"engine": "postgres", "stats_reset_at": "T", "since": None, "limit": 200}}
    result = window_limits(before, after)
    assert result == WindowLimits(before=500, after=200)
    assert result.may_be_sampling_artifact is True


def test_window_limits_matching_known_limits_are_not_a_sampling_artifact():
    before = {"window": {"engine": "postgres", "stats_reset_at": "T", "since": None, "limit": 500}}
    after = {"window": {"engine": "postgres", "stats_reset_at": "T", "since": None, "limit": 500}}
    result = window_limits(before, after)
    assert result == WindowLimits(before=500, after=500)
    assert result.may_be_sampling_artifact is False


def test_window_limits_treats_a_missing_or_non_int_limit_as_unknown_not_zero():
    """`None` here must carry the same weight as a genuine mismatch for Task 6 — it is not
    evidence the limits were equal."""
    before = {"window": {"engine": "postgres", "stats_reset_at": "T", "since": None, "limit": None}}
    after = {"window": {"engine": "postgres", "stats_reset_at": "T", "since": None, "limit": "500"}}
    assert window_limits(before, after) == WindowLimits(before=None, after=None)


def test_window_limits_none_none_may_be_a_sampling_artifact_not_a_match():
    """Round-1 review, Important 4 — the core fix. Task 6's naive derivation,
    `before == after`, reads `WindowLimits(None, None)` as "the limits matched"; they are
    both `None` here precisely *because* neither run recorded a limit, which is the
    opposite of a demonstrated match. `.may_be_sampling_artifact` must be `True`, making
    the correct reading the only one available from the property."""
    result = window_limits({}, {})
    assert result == WindowLimits(before=None, after=None)
    assert result.before == result.after, "the naive (and wrong) reading would call this a match"
    assert result.may_be_sampling_artifact is True


def test_window_limits_booleans_do_not_pass_as_limits():
    """Round-1 review, Minor 7. `isinstance(True, int)` is `True` in Python, so a stray
    boolean `limit` must be excluded explicitly rather than accepted as a sampled-group
    count."""
    before = {"window": {"engine": "postgres", "stats_reset_at": "T", "since": None, "limit": True}}
    after = {"window": {"engine": "postgres", "stats_reset_at": "T", "since": None, "limit": False}}
    assert window_limits(before, after) == WindowLimits(before=None, after=None)


def test_window_limits_does_not_raise_on_a_malformed_window():
    assert window_limits({}, {}) == WindowLimits(before=None, after=None)
    assert window_limits({"window": "not-a-dict"}, {"window": "not-a-dict"}) == WindowLimits(
        before=None, after=None
    )


def test_a_differing_limit_does_not_change_the_window_relation_or_confidence():
    """Ruling 6: `limit` is recorded and passed on via `window_limits`, but it must not be
    folded into `classify_windows`'s relation or `confidence_for`'s grade. The same window
    pair as the NESTED test above, but with different `limit` values on each side, must
    still classify as NESTED at MEDIUM — and `window_limits` must still report the
    mismatch for Task 6 to consume separately."""
    before = {
        "window": {
            "engine": "postgres",
            "stats_reset_at": "2026-08-01T00:00:00",
            "since": None,
            "since_duration_seconds": None,
            "limit": 500,
        }
    }
    after = {
        "window": {
            "engine": "postgres",
            "stats_reset_at": "2026-08-01T00:00:00",
            "since": None,
            "since_duration_seconds": None,
            "limit": 50,
        }
    }
    assert classify_windows(before, after) is WindowRelation.NESTED
    assert confidence_for(classify_windows(before, after)) is Confidence.MEDIUM
    result = window_limits(before, after)
    assert result == WindowLimits(before=500, after=50)
    assert result.may_be_sampling_artifact is True
