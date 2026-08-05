"""The loop closed through the **command**, against a real server: real `advise --json >
before.json` -> real `CREATE INDEX` -> real `advise --json > after.json` -> real `sqlquality
verify before.json after.json`.

**What this adds over the three live tests beside it.** `test_verify_headline_live.py`,
`test_verify_degraded_after_live.py` and `test_verify_redaction_live.py` all call
`sqlquality.verify`'s functions directly (`verdicts()`, `artifact_incomparabilities()`) on
payloads they parsed themselves. Nothing in this package had ever run the `verify` *command*
over artifacts a real `advise` wrote to real files — so the layers between those functions and
a user (argument parsing, both file reads, the five refusals, `verify_context`, and all three
renderers) were exercised only against stubbed artifacts in `tests/test_verify_cli.py`. This
test closes that gap, and it is the only place where the command's contract meets bytes a
server actually produced.

It also proves the **premise one refusal rests on**: `verify` treats byte-identical artifacts
as the same run saved twice, on the argument that two genuinely distinct runs cannot be
byte-identical because `pg_stat_statements`' counters accumulate. That argument is about a real
server's behaviour, and until now it was only ever checked against artifacts a stub produced —
which is to say, against the assumption itself.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from sqlquality.cli import app

pytestmark = pytest.mark.integration
runner = CliRunner()

_SCHEMA = "verify_command_it"
_ROWS = 200_000
_STATUS_BUCKETS = 2_000


@pytest.fixture
def command_schema(live_dsn: str) -> str:
    """A fresh, disposable table with a hot, unindexed equality predicate.

    `pg_stat_statements` is reset for this schema's statements only, by `queryid` — the same
    targeted reset `test_verify_headline_live.py` and `test_verify_redaction_live.py` use, for
    the same two reasons: a blanket reset would wipe the cumulative counters the rest of this
    package shares, and without *some* reset the statement's identical normalized text lets a
    previous run's executions accumulate into this run's before window, which is history
    dependence disguised as a timing.
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
    import psycopg

    with psycopg.connect(dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            for _ in range(times):
                cur.execute(f"SELECT id FROM {_SCHEMA}.orders WHERE status = %s", ("status_3",))
                cur.fetchall()


def _write_artifact(dsn: str, path) -> dict:
    """One real `advise --json` run, written to `path` exactly as a shell redirect would."""
    result = runner.invoke(
        app, ["advise", "--dsn", dsn, "--schema", _SCHEMA, "--json", "--min-cost-share", "0.0"]
    )
    assert result.exit_code == 0, result.output
    path.write_text(result.stdout, encoding="utf-8")
    return json.loads(result.stdout)


def test_the_verify_command_closes_the_loop_over_two_real_artifacts(command_schema, tmp_path):
    """`advise --json` -> `CREATE INDEX` -> `advise --json` -> `verify`, all four for real.

    The non-vacuity guard is asserted first and from the artifact's own bytes: a before run
    that produced no ADV001 for this relation would leave the whole comparison below with
    nothing to be about, and `verify` would exit 0 having reported no verdicts at all — which
    is indistinguishable, in the exit code, from the success this test is named for.

    `applied` is the assertion the brief asks for. The outcome is asserted too, but only
    because this pair's margin is not a matter of timing luck: the before window holds exactly
    the pre-index executions (checked), and an index lookup into one of 2,000 buckets against a
    200,000-row sequential scan is two orders of magnitude apart. The confidence is `low`
    because `pg_stat_database.stats_reset` is SQL NULL on a container where nothing has called
    `pg_stat_reset()`, so neither window can be placed relative to the other — that is the
    honest ceiling, and asserting it keeps a window misclassification from hiding behind a
    correct outcome.
    """
    dsn = command_schema
    before_path = tmp_path / "before.json"
    after_path = tmp_path / "after.json"

    _run_query(dsn, times=5)
    before = _write_artifact(dsn, before_path)
    adv001 = [
        p
        for p in before["proposals"]
        if p["code"] == "ADV001" and p["evidence"].get("table") == "orders"
    ]
    assert len(adv001) == 1, (
        f"no single ADV001 for {_SCHEMA}.orders in the before artifact; got "
        f"{[(p['code'], p['evidence'].get('table')) for p in before['proposals']]}"
    )
    cited = adv001[0]["evidence"]["fingerprint_digests"]
    groups = {g["digest"]: g for g in before["query_groups"]}
    assert [groups[d]["calls"] for d in cited] == [5], (
        "the before window does not hold exactly the 5 executions this test performed, so an "
        f"earlier run's calls are being averaged in: {before['query_groups']}"
    )

    import psycopg

    with psycopg.connect(dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f"CREATE INDEX idx_command_status ON {_SCHEMA}.orders (status)")

    _run_query(dsn, times=500)
    _write_artifact(dsn, after_path)

    # The premise behind the byte-identity refusal, from two real runs rather than from the
    # assumption: the counters accumulated, so the bytes differ.
    assert before_path.read_bytes() != after_path.read_bytes()

    markdown = tmp_path / "verify.md"
    result = runner.invoke(
        app,
        [
            "verify",
            str(before_path),
            str(after_path),
            "--json",
            "--markdown",
            str(markdown),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    ours = [
        v
        for v in payload["verdicts"]
        if v["code"] == "ADV001" and v["key"][1:3] == [_SCHEMA, "orders"]
    ]
    assert len(ours) == 1, payload["verdicts"]
    [verdict] = ours
    assert verdict["applied"] is True, verdict
    assert verdict["mean_ms_before"] is not None and verdict["mean_ms_after"] is not None, verdict
    assert verdict["outcome"] == "improved", verdict
    assert verdict["confidence"] == "low", verdict
    assert payload["window_relation"] == "incomparable"
    # The markdown surface, over the same real artifacts.
    text = markdown.read_text(encoding="utf-8")
    assert "# sqlquality verify — postgres" in text
    assert f"ADV001 {_SCHEMA}.orders (status)" in text


def test_the_command_refuses_one_real_artifact_passed_twice(command_schema, tmp_path):
    """The same-file refusal against a real artifact, and the reason it is not redundant with
    the stubbed version: it confirms a genuine `advise` artifact reaches the refusal at all
    rather than tripping some earlier check on the way — a real payload carries keys, degraded
    entries and a window that a stub's does not."""
    dsn = command_schema
    _run_query(dsn, times=5)
    path = tmp_path / "one.json"
    _write_artifact(dsn, path)
    result = runner.invoke(app, ["verify", str(path), str(path)])
    assert result.exit_code == 2
    assert "same file" in result.stderr
