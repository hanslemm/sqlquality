"""The headline case, end to end against a real server: real `advise` -> real
`CREATE INDEX` -> real `advise` -> `verdicts()`.

This is the scenario `sqlquality verify` exists to celebrate — a proposal was applied and
the workload it cited actually got faster — and it is also the exact scenario Task 6's
fix round 3 found broken: `_query_groups_payload` (report.py) used to scope
`payload["query_groups"]` to the digests `proposals` currently cite, so the moment ADV001
stopped firing (because the index now exists), `after["query_groups"]` went to `[]` and
every cited digest became unresolvable — `verify` read that as `DISAPPEARED`, not
`IMPROVED`, in exactly the case this whole feature is for. Reproduced live during review:
`applied=True disappeared`, "Cited query group(s) no longer appear in the after run", for
a group that had in fact executed twelve times in `after`.

A synthetic fixture cannot, by itself, prove a bug like this is fixed: the bug was that a
fixture shape which *looked* plausible was never actually reachable through the real
`advise` path. This test builds both artifacts through that real path instead — its own
throwaway schema, isolated from every other integration test's fixtures and from the
shared, session-scoped `seeded` fixture, so nothing here can perturb another test's
`pg_stat_statements` state.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from sqlquality.cli import app
from sqlquality.models import Confidence
from sqlquality.verify import VerifyOutcome, verdicts

pytestmark = pytest.mark.integration
runner = CliRunner()

_SCHEMA = "verify_headline_it"
_ROWS = 300_000
#: Distinct `status` buckets. With `_ROWS` rows spread evenly, each bucket matches
#: `_ROWS / _STATUS_BUCKETS` rows — selective enough that a sequential scan over the whole
#: table is measurably slower than an index lookup restricted to one bucket.
_STATUS_BUCKETS = 3_000


@pytest.fixture
def headline_schema(live_dsn: str) -> str:
    """A fresh, disposable table with a hot, unindexed equality predicate — nothing
    shared with any other test's fixtures or connections."""
    import psycopg

    with psycopg.connect(live_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS pg_stat_statements")
            cur.execute(f"DROP SCHEMA IF EXISTS {_SCHEMA} CASCADE")
            cur.execute(f"CREATE SCHEMA {_SCHEMA}")
            cur.execute(
                f"CREATE TABLE {_SCHEMA}.orders (id bigserial PRIMARY KEY, status text NOT NULL)"
            )
            cur.execute(
                f"INSERT INTO {_SCHEMA}.orders (status) "
                f"SELECT 'status_' || (g % {_STATUS_BUCKETS}) FROM generate_series(1, {_ROWS}) g"
            )
            cur.execute(f"ANALYZE {_SCHEMA}.orders")
    return live_dsn


def _run_query(dsn: str, *, times: int) -> None:
    import psycopg

    with psycopg.connect(dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            for _ in range(times):
                cur.execute(f"SELECT id FROM {_SCHEMA}.orders WHERE status = %s", ("status_1",))
                cur.fetchall()


def _run_advise(dsn: str) -> dict:
    result = runner.invoke(
        app,
        [
            "advise",
            "--dsn",
            dsn,
            "--schema",
            _SCHEMA,
            "--json",
            "--min-cost-share",
            "0.0",
        ],
    )
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)


def _adv001_for_orders(payload: dict) -> dict | None:
    for p in payload["proposals"]:
        if p["code"] == "ADV001" and p["evidence"].get("table") == "orders":
            return p
    return None


def test_the_headline_case_end_to_end_against_a_real_server(headline_schema):
    """Real `advise` (no index, hot predicate) -> real `CREATE INDEX` -> real `advise`
    (rule resolved, zero proposals for this relation) -> `verdicts()`.

    No `pg_stat_statements_reset()` here: this test shares the live server with every
    other integration test in this package, and resetting mid-session would wipe out
    accumulated stats other tests still depend on. `pg_stat_statements` therefore stays
    cumulative across the before/after halves below, diluting any real improvement — so
    the pre-index calls are kept few and the post-index calls many, letting the fast calls
    dominate the cumulative mean by enough to clear `RELATIVE_CHANGE_THRESHOLD` regardless.

    This also means `window_facts`'s `stats_reset_at` (`pg_stat_database.stats_reset`, a
    *different* counter from `pg_stat_statements_reset()`) stays SQL `NULL` throughout —
    no test in this suite calls `pg_stat_reset()` — so the window classifies
    `INCOMPARABLE`, not `NESTED`: both sides being null is missing information about
    whether the counters were ever baselined, not evidence they were and never reset
    since. `Confidence.LOW` is the correct ceiling for that, asserted below.
    """
    dsn = headline_schema

    _run_query(dsn, times=5)
    before = _run_advise(dsn)
    before_adv001 = _adv001_for_orders(before)
    assert before_adv001 is not None, (
        f"no ADV001 for {_SCHEMA}.orders before the index exists; got "
        f"{[(p['code'], p['evidence'].get('table')) for p in before['proposals']]}"
    )
    assert f"{_SCHEMA}.orders" in before["analyzed"]["tables"], before["analyzed"]

    import psycopg

    with psycopg.connect(dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f"CREATE INDEX idx_headline_status ON {_SCHEMA}.orders (status)")

    _run_query(dsn, times=500)
    after = _run_advise(dsn)
    after_adv001 = _adv001_for_orders(after)
    assert after_adv001 is None, (
        f"ADV001 still fires for {_SCHEMA}.orders after the index was created — the "
        f"scenario never reached the 'resolved' shape this test needs: {after['proposals']}"
    )
    assert after["physical_state"].get(f"{_SCHEMA}.orders") is not None, (
        "physical_state omitted a relation the workload touched but no proposal remains "
        "for — the Task 6 fix-round-1 scoping fix is not wired in"
    )

    results = verdicts(before, after)
    ours = [v for v in results if v.code == "ADV001" and v.key[1:3] == (_SCHEMA, "orders")]
    assert len(ours) == 1, f"expected exactly one ADV001 verdict for {_SCHEMA}.orders: {ours}"
    [verdict] = ours

    assert verdict.applied is True
    # The fix round 3 pin: before the fix this was unconditionally None, because
    # `after["query_groups"]` was empty the moment ADV001 stopped citing it.
    assert verdict.mean_after is not None, (
        "mean_after is None -- the query group's own timing is gone from `after`'s "
        "query_groups even though it just ran 500 times; the round-3 scoping fix regressed"
    )
    assert verdict.mean_before is not None
    assert verdict.outcome is VerifyOutcome.IMPROVED, (
        f"expected IMPROVED, got {verdict.outcome} "
        f"(mean_before={verdict.mean_before}, mean_after={verdict.mean_after}, "
        f"note={verdict.note!r})"
    )
    # `pg_stat_database.stats_reset` (what `window_facts` actually reads — a different
    # counter from `pg_stat_statements_reset()`, which this test never calls either) stays
    # SQL NULL on a container that has never called `pg_stat_reset()`, on both the before
    # and after calls -- Ruling 2's "both null" case, INCOMPARABLE, never NESTED: nothing
    # here demonstrates the counters were baselined once and never reset, only that
    # neither run can tell. LOW is the correct, honest ceiling for that, not a weaker
    # stand-in for the MEDIUM a real reset-based NESTED pair would earn.
    assert verdict.confidence is Confidence.LOW
