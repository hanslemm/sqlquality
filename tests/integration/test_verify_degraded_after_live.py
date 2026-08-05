"""Finding N1's reachable case, end to end against a real server: a real `advise` run whose
`CAP_WORKLOAD` read genuinely fails, compared against a real ordinary run.

This task's defect class has now recurred three times, always by the same mechanism: an
absence produced by one run's own limitations read as a measurement about the user's
database. The third instance was a denied capability read emptying the data, and fix round 3
closed it for one capability of the nine either adapter can record — leaving `CAP_WORKLOAD`,
the one capability the project's own `_HINTS` text says needs a real grant, unguarded.

**Why this test exists rather than only a fixture.** Fix round 3's own headline unit test
asserted against a payload shape `advise` could never emit, and so passed while the real path
stayed broken. So the `after` artifact here is not composed: it is produced by running the
real `advise` command against a real PostgreSQL 16 whose `pg_stat_statements` read fails, and
the test asserts the resulting shape before using it — including the detail that makes this
slip past Ruling 2's sampling gate, namely that `window.limit` is still recorded even though
the read it bounds never happened.

**Why a second database.** The read has to genuinely fail. Revoking the grant, or dropping
the extension, on the database every other integration test shares would either reset the
cumulative `pg_stat_statements` counters those tests depend on or leave a global grant
mutated if this test failed mid-way. A throwaway database on the same server has neither
hazard: `CREATE EXTENSION` is per-database, so a fresh one simply has no
`pg_stat_statements` view, which is also the single most ordinary way a real user meets this
degradation.
"""

from __future__ import annotations

import json
from urllib.parse import urlparse, urlunparse

import pytest
from typer.testing import CliRunner

from sqlquality.cli import app
from sqlquality.models import Confidence
from sqlquality.verify import VerifyOutcome, verdicts, window_limits

pytestmark = pytest.mark.integration
runner = CliRunner()

_SCHEMA = "verify_degraded_it"
#: A database that has never had `CREATE EXTENSION pg_stat_statements` run in it. Named for
#: this test so a leftover from a crashed run is identifiable.
_BARE_DATABASE = "sqlquality_verify_degraded_it"
_ROWS = 20_000
_STATUS_BUCKETS = 200


def _dsn_for_database(dsn: str, database: str) -> str:
    """`dsn` pointed at a different database on the same server, credentials untouched."""
    parsed = urlparse(dsn)
    return urlunparse(parsed._replace(path=f"/{database}"))


@pytest.fixture
def degraded_pair(live_dsn: str):
    """A hot, unindexed predicate in its own schema on the shared database, plus a bare
    database with no `pg_stat_statements` extension. The bare database is dropped again
    whether or not the test passes; nothing about the shared database's extension, grants or
    counters is touched."""
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
            cur.execute(f"DROP DATABASE IF EXISTS {_BARE_DATABASE} WITH (FORCE)")
            cur.execute(f"CREATE DATABASE {_BARE_DATABASE}")
    try:
        yield live_dsn, _dsn_for_database(live_dsn, _BARE_DATABASE)
    finally:
        with psycopg.connect(live_dsn, autocommit=True) as conn:
            with conn.cursor() as cur:
                cur.execute(f"DROP DATABASE IF EXISTS {_BARE_DATABASE} WITH (FORCE)")


def _run_query(dsn: str, *, times: int) -> None:
    import psycopg

    with psycopg.connect(dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            for _ in range(times):
                cur.execute(f"SELECT id FROM {_SCHEMA}.orders WHERE status = %s", ("status_1",))
                cur.fetchall()


def _run_advise(dsn: str, *args: str) -> dict:
    result = runner.invoke(
        app, ["advise", "--dsn", dsn, "--json", "--min-cost-share", "0.0", *args]
    )
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)


def test_a_denied_after_workload_read_is_not_reported_as_a_query_group_that_stopped(
    degraded_pair,
):
    """Real `advise` (hot predicate, ADV001 fires and cites a query group) versus real
    `advise` whose `pg_stat_statements` read failed.

    The second half of the assertions is the finding. Before this fix `verify` reported
    "its cited query group(s) are absent from the after run — possibly addressed (for
    example, by a query rewrite), but disappearance alone is not proof" — a claim about a
    query that, for all either artifact establishes, is still running unchanged. The run did
    not observe the query stopping; it did not observe anything.
    """
    seeded_dsn, bare_dsn = degraded_pair

    _run_query(seeded_dsn, times=20)
    before = _run_advise(seeded_dsn, "--schema", _SCHEMA)
    adv001 = [
        p
        for p in before["proposals"]
        if p["code"] == "ADV001" and p["evidence"].get("table") == "orders"
    ]
    assert len(adv001) == 1, (
        f"no single ADV001 for {_SCHEMA}.orders in the before run; got "
        f"{[(p['code'], p['evidence'].get('table')) for p in before['proposals']]}"
    )
    assert adv001[0]["evidence"]["fingerprint_digests"], (
        "the before proposal cites no query group, so this pair cannot exercise the "
        "query-group absence at all"
    )

    after = _run_advise(bare_dsn)

    # The `after` shape, asserted rather than assumed — this is what fix round 3's unit
    # fixture got wrong. Every emptiness below follows from the one denial.
    assert [entry["capability"] for entry in after["degraded"]] == ["workload"], after["degraded"]
    assert after["query_groups"] == []
    assert after["physical_state"] == {}
    assert after["proposals"] == []
    # And the reason Ruling 2's sampling gate cannot catch this: `--limit`'s default is
    # recorded on both sides even though the read it bounds never ran.
    assert before["window"]["limit"] == after["window"]["limit"] is not None
    assert window_limits(before, after).may_be_sampling_artifact is False

    ours = [v for v in verdicts(before, after) if v.code == "ADV001" and v.key[2] == "orders"]
    assert len(ours) == 1, f"expected exactly one ADV001 verdict for {_SCHEMA}.orders: {ours}"
    [verdict] = ours

    assert verdict.applied is None
    assert verdict.outcome is VerifyOutcome.UNOBSERVABLE
    assert verdict.confidence is Confidence.LOW
    assert verdict.mean_before is not None and verdict.mean_after is None
    assert "degraded" in verdict.note and "workload" in verdict.note, verdict.note
    assert "possibly addressed" not in verdict.note, verdict.note
    assert "absent from the after run" not in verdict.note, verdict.note
