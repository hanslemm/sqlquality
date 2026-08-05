"""Review finding B, end to end against a real server: two real `advise` runs differing only
in `--keep-literals`, and the false `DISAPPEARED` verdict that used to produce.

This is the **fourth** door onto this task's recurring defect — an absence produced by one
run's own limitations reported as a measurement about the user's database — and the first that
has nothing to do with `degraded`, so the round-4 gating mechanism structurally cannot see it.
The mechanism is that `payload["redacted"]` is `not keep_literals`, redaction rewrites the
canonical query text, and `fingerprint_id` hashes that text into the `digest` that
`query_groups` is keyed by. Two runs over an identical workload therefore agree on nothing in
`query_groups`: same queries, disjoint keys.

Review reproduced it as `applied=True outcome=DISAPPEARED note="Cited query group(s) no longer
appear in the after run."` — with no degradation anywhere, matching `--limit`, the
recommendation genuinely applied in between, and the query still running in the `after` run.

**Why two real `advise` runs and not a fixture.** A synthetic pair could only assert that
`verify` refuses when the two `redacted` flags differ, which is a tautology against the code.
What needs proving is the premise: that a real pair produced this way actually has disjoint
`query_groups` keys, so the refusal is load-bearing rather than decorative. Both artifacts here
come from `runner.invoke(app, ["advise", ...])`, and the disjointness is asserted from them
before any verdict is examined.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from sqlquality.cli import app
from sqlquality.verify import (
    ArtifactMismatch,
    VerifyOutcome,
    artifact_incomparabilities,
    group_index,
    verdicts,
    window_limits,
)

pytestmark = pytest.mark.integration
runner = CliRunner()

_SCHEMA = "verify_redaction_it"
_ROWS = 20_000
_STATUS_BUCKETS = 200
#: The literal the two runs disagree about keeping.
_STATUS = "status_7"


@pytest.fixture
def redaction_schema(live_dsn: str) -> str:
    """A fresh, disposable table with a hot, unindexed equality predicate on a literal.
    Nothing shared with any other test's fixtures.

    **`pg_stat_statements` is reset for this schema's statements only**, via PostgreSQL 13+'s
    per-`queryid` `pg_stat_statements_reset(userid, dbid, queryid)`. A blanket
    `pg_stat_statements_reset()` would wipe the cumulative counters the rest of this package
    shares (which is why `test_verify_headline_live.py` explicitly refuses to call it), but
    this test needs something stronger than schema isolation: any *other* statement against
    `{_SCHEMA}.orders` still sitting in the view — a plain `WHERE status = 'x'`, whose literal
    PostgreSQL already parameterized — would be cited by the same ADV001 proposal and would
    resolve identically in both runs, turning the headline case into review's milder
    "1 of 2 cited groups" variant. Resetting by `queryid` targets exactly this schema's rows
    and leaves every other test's counters untouched.
    """
    import psycopg

    with psycopg.connect(live_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS pg_stat_statements")
            cur.execute(
                "SELECT pg_stat_statements_reset(userid, dbid, queryid) "
                "FROM pg_stat_statements WHERE query LIKE %s",
                (f"%{_SCHEMA}.%",),
            )
            cur.fetchall()
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
    """The hot statement, and the one non-obvious thing this whole test turns on.

    **`ORDER BY 1` is load-bearing, not decoration.** `pg_stat_statements` normalizes every
    *parameterizable* constant to `$N` before sqlquality reads a single row — a bare
    `WHERE status = 'status_7'` is stored as `WHERE status = $1`, so sqlquality's own
    redaction has nothing left to remove and `--keep-literals` changes the fingerprint not at
    all. An `ORDER BY`/`GROUP BY` **ordinal** is the exception: parameterizing it would change
    the statement's meaning, so PostgreSQL leaves it verbatim, sqlquality's redaction rewrites
    it to a placeholder, and the digest moves. Verified on PostgreSQL 16:

        SELECT id FROM t WHERE status = 'x'            -> stored as `status = $1`, digest equal
        SELECT id FROM t WHERE status = 'x' ORDER BY 1  -> ordinal survives, digest differs

    So on Postgres this finding is narrower than "any two runs differing in the flag": it
    needs a surviving non-parameterizable constant. The assertions below check the disjointness
    rather than assuming it, so if a future PostgreSQL parameterizes ordinals too, this test
    says so instead of passing vacuously."""
    import psycopg

    with psycopg.connect(dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            for _ in range(times):
                cur.execute(
                    f"SELECT id FROM {_SCHEMA}.orders WHERE status = '{_STATUS}' ORDER BY 1"
                )
                cur.fetchall()


def _run_advise(dsn: str, *extra: str) -> dict:
    result = runner.invoke(
        app,
        ["advise", "--dsn", dsn, "--schema", _SCHEMA, "--json", "--min-cost-share", "0.0", *extra],
    )
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)


def _adv001_for_orders(payload: dict) -> dict | None:
    for proposal in payload["proposals"]:
        if proposal["code"] == "ADV001" and proposal["evidence"].get("table") == "orders":
            return proposal
    return None


def test_two_real_advise_runs_differing_only_in_keep_literals_claim_no_speed_comparison(
    redaction_schema,
):
    """Real `advise` (redacting) -> real `CREATE INDEX` -> real `advise --keep-literals`.

    Before this fix the verdict was `applied=True`, `DISAPPEARED`, "Cited query group(s) no
    longer appear in the after run" — for a query that had just run 300 more times. The group
    is not gone; it is recorded under a digest computed from unredacted text.
    """
    dsn = redaction_schema

    _run_query(dsn, times=20)
    before = _run_advise(dsn)
    before_adv001 = _adv001_for_orders(before)
    assert before_adv001 is not None, (
        f"no ADV001 for {_SCHEMA}.orders before the index exists; got "
        f"{[(p['code'], p['evidence'].get('table')) for p in before['proposals']]}"
    )
    cited = before_adv001["evidence"]["fingerprint_digests"]
    assert cited, "the before proposal cites no query group, so nothing here is exercised"

    import psycopg

    with psycopg.connect(dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f"CREATE INDEX idx_redaction_status ON {_SCHEMA}.orders (status)")

    _run_query(dsn, times=300)
    after = _run_advise(dsn, "--keep-literals")

    # The premise, asserted from the two real artifacts rather than assumed.
    assert before["redacted"] is True and after["redacted"] is False
    assert before["degraded"] == [] and after["degraded"] == []
    assert window_limits(before, after).may_be_sampling_artifact is False
    assert after["query_groups"], (
        "the after run recorded no query groups at all, so this pair cannot distinguish a "
        "digest mismatch from an empty workload"
    )
    after_groups = group_index(after)
    # The load-bearing fact, and the whole reason the refusal is not decoration: not one
    # digest the before proposal cites resolves in `after`, even though `after` observed a
    # perfectly healthy workload. `verify` therefore *sees* `present_after == 0` here — the
    # exact input it used to grade `DISAPPEARED`.
    assert not (set(cited) & set(after_groups)), (
        "a cited digest still resolves in the after run, so redaction did not move this "
        "proposal's own query group and the test would pass for the wrong reason: "
        f"cited={sorted(cited)} after={sorted(after_groups)}"
    )
    # …and the query did not stop running: some group in `after` carries the 300 executions
    # from above, under a digest computed from unredacted text.
    busiest = max(int(group["calls"] or 0) for group in after_groups.values())
    assert busiest >= 300, (
        "no after-run query group records the 300 executions this test just performed, so "
        f"the 'still running under a different digest' premise is unproven (busiest={busiest})"
    )
    # And the relation itself is still there, fully observed, with the index now present —
    # so `applied` is genuinely readable and the refusal below is not standing in for it.
    orders_state = after["physical_state"].get(f"{_SCHEMA}.orders")
    assert orders_state is not None and orders_state["indexes"] is not None

    # Callable on its own, which is how Task 7 should refuse the pair.
    found = artifact_incomparabilities(before, after)
    assert [item.mismatch for item in found] == [ArtifactMismatch.REDACTION]

    ours = [v for v in verdicts(before, after) if v.code == "ADV001" and v.key[2] == "orders"]
    assert len(ours) == 1, f"expected exactly one ADV001 verdict for {_SCHEMA}.orders: {ours}"
    [verdict] = ours

    assert verdict.applied is True
    assert verdict.outcome is VerifyOutcome.UNOBSERVABLE, (
        f"expected UNOBSERVABLE, got {verdict.outcome} "
        f"(mean_before={verdict.mean_before}, mean_after={verdict.mean_after}, "
        f"note={verdict.note!r})"
    )
    assert verdict.mean_before is None and verdict.mean_after is None
    assert "redaction" in verdict.note, verdict.note
    # Every phrasing in `verdicts` that states whether the cited groups are in the after run,
    # not only the `DISAPPEARED` branch's exact words: pinning one branch's wording left its
    # two siblings free to make the same claim in a different sentence, which is how sinking
    # the incomparability check below the `applied is None` branch survived the whole suite.
    # See `_GROUP_LEVEL_CLAIMS` in tests/test_verify.py, this assertion's unit-level twin.
    for claim in (
        "no longer appear in the after run",
        "absent from the after run",
        "present in the after run",
    ):
        assert claim not in verdict.note, verdict.note
