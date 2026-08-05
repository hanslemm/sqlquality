"""Regenerate the committed `verify` golden artifact pairs from a live PostgreSQL 16.

**Not a test, and not run by CI.** It is the provenance of every `*.before.json` /
`*.after.json` file beside it, kept in the tree so a reviewer can check that those artifacts
are shapes `advise` can actually emit rather than shapes someone believed it could. Task 6 of
this feature shipped a headline unit test asserting against a payload `advise` could never
produce, and it passed while the real path was broken; the whole point of generating these
through the real command is that no fixture here can repeat that.

Every artifact below is the verbatim stdout of a real `sqlquality advise --json` run against a
real server, byte for byte, with no post-processing whatsoever. Each scenario asserts the
properties the paired unit test depends on *before* writing anything, so a scenario that
stopped reaching its intended shape fails here instead of silently producing a fixture that
pins nothing.

Run it as::

    docker compose -f tests/integration/docker-compose.yml up -d
    .venv/bin/python tests/fixtures/verify/regenerate.py
    docker compose -f tests/integration/docker-compose.yml down

**Re-running rewrites the artifacts, and the new bytes will differ** — the timings and the
`stats_reset_at` instants are measurements, so they cannot be reproduced. That is fine and it
is the reason `tests/test_verify_goldens.py` asserts *relations* (outcome, confidence, window
relation, a ratio) rather than absolute numbers: a regenerated pair on a different machine
still grades identically, and the `check()` calls below fail loudly if it would not.

It works in a **throwaway database** (`sqlquality_verify_goldens`) rather than the one the
integration suite uses, for two independent reasons:

* `advise`'s workload read is scoped `WHERE d.datname = current_database()`, so a private
  database is the only way to guarantee the `query_groups` in a committed fixture are this
  script's own statements and not whatever else the container has run.
* Two of the scenarios need `pg_stat_reset()` (to move `pg_stat_database.stats_reset`, which
  is what `window.stats_reset_at` reports) and all of them need
  `pg_stat_statements_reset()`. Both are destructive to whatever else shares the server;
  scoping the statement reset by `dbid` and the database reset to a database nothing else
  uses keeps the integration suite's cumulative counters intact.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import psycopg
from typer.testing import CliRunner

from sqlquality.cli import app
from sqlquality.models import Confidence
from sqlquality.verify import VerifyOutcome, verdicts

HERE = Path(__file__).resolve().parent
ADMIN_DSN = "postgresql://postgres:sqlquality@127.0.0.1:27432/sqlquality_test"
GOLDEN_DB = "sqlquality_verify_goldens"
GOLDEN_DSN = f"postgresql://postgres:sqlquality@127.0.0.1:27432/{GOLDEN_DB}"
ROWS = 300_000
BUCKETS = 3_000

runner = CliRunner()


def sql(dsn: str, *statements: str) -> None:
    with psycopg.connect(dsn, autocommit=True) as conn, conn.cursor() as cur:
        for statement in statements:
            cur.execute(statement)


def reset_statements() -> None:
    """Clear `pg_stat_statements` for the golden database only, leaving every other
    database's counters (the integration suite's included) untouched."""
    with psycopg.connect(GOLDEN_DSN, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT pg_stat_statements_reset(0, (SELECT oid FROM pg_database "
            "WHERE datname = %s), 0)",
            (GOLDEN_DB,),
        )
        cur.fetchall()


def reset_database_stats() -> None:
    """Move `pg_stat_database.stats_reset` — what `window.stats_reset_at` reports — for the
    golden database. Two runs either side of this see two different instants (`DISJOINT`);
    two runs with no call in between see the same one (`NESTED`)."""
    with psycopg.connect(GOLDEN_DSN, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute("SELECT pg_stat_reset()")
        cur.fetchall()


def seed(schema: str, *, hot_rows: int = 0) -> None:
    """A table with a hot equality predicate on `status`, and a `note` column for the
    second, deliberately expensive statement the `--limit` scenario needs.

    `hot_rows` skews `status` so one value dominates: that is what lets the same
    parameterized statement (one query group) be cheap for a rare value and expensive for
    the hot one, which is how the `regressed` scenario gets a genuine per-call regression
    out of a workload shift rather than out of an index that somehow made things worse.
    """
    sql(
        GOLDEN_DSN,
        f"DROP SCHEMA IF EXISTS {schema} CASCADE",
        f"CREATE SCHEMA {schema}",
        f"CREATE TABLE {schema}.orders (id bigserial PRIMARY KEY, status text NOT NULL, "
        "note text NOT NULL)",
        f"INSERT INTO {schema}.orders (status, note) "
        f"SELECT 'status_' || (g % {BUCKETS}), 'note ' || g "
        f"FROM generate_series(1, {ROWS}) g",
        *(
            [
                f"INSERT INTO {schema}.orders (status, note) "
                f"SELECT 'hot', 'note ' || g FROM generate_series(1, {hot_rows}) g"
            ]
            if hot_rows
            else []
        ),
        f"ANALYZE {schema}.orders",
    )


def run_select(schema: str, *, times: int, status: str = "status_1") -> None:
    with psycopg.connect(GOLDEN_DSN, autocommit=True) as conn, conn.cursor() as cur:
        for _ in range(times):
            cur.execute(
                f"SELECT id FROM {schema}.orders WHERE status = %s",
                (status,),
            )
            cur.fetchall()


def run_count(schema: str, *, times: int, status: str = "status_1") -> None:
    """A *different* fingerprint over the same relation — the `disappeared` scenario's whole
    mechanism: the after run still analyses the relation (so `physical_state` records the new
    index and `applied` is genuinely readable) while the digest the before proposal cited is
    nowhere in the after window."""
    with psycopg.connect(GOLDEN_DSN, autocommit=True) as conn, conn.cursor() as cur:
        for _ in range(times):
            cur.execute(
                f"SELECT count(*) FROM {schema}.orders WHERE status = %s",
                (status,),
            )
            cur.fetchall()


def run_heavy(schema: str, *, times: int) -> None:
    """One statement far more expensive per call than the indexed `run_select`, so it, not the
    cited group, is what a `--limit 1` window keeps. It names a column in its predicate so the
    relation still reaches `aggregation.tables` and hence `physical_state`."""
    with psycopg.connect(GOLDEN_DSN, autocommit=True) as conn, conn.cursor() as cur:
        for _ in range(times):
            cur.execute(f"SELECT sum(length(note)) FROM {schema}.orders WHERE note IS NOT NULL")
            cur.fetchall()


def advise(schema: str, *extra: str) -> tuple[str, dict[str, Any]]:
    result = runner.invoke(
        app,
        [
            "advise",
            "--dsn",
            GOLDEN_DSN,
            "--schema",
            schema,
            "--json",
            "--min-cost-share",
            "0.0",
            *extra,
        ],
    )
    assert result.exit_code == 0, result.output
    return result.stdout, json.loads(result.stdout)


def adv001(payload: dict[str, Any]) -> dict[str, Any] | None:
    for proposal in payload["proposals"]:
        if proposal["code"] == "ADV001" and proposal["evidence"].get("table") == "orders":
            return proposal
    return None


def write(name: str, before_text: str, after_text: str) -> None:
    (HERE / f"{name}.before.json").write_text(before_text)
    (HERE / f"{name}.after.json").write_text(after_text)


def check(
    name: str,
    before: dict[str, Any],
    after: dict[str, Any],
    *,
    applied: bool | None,
    outcome: VerifyOutcome,
    confidence: Confidence,
) -> None:
    """Assert the pair actually reached the shape its golden test will pin.

    The non-vacuity guard is the first assertion, not the last: without it a scenario whose
    workload produced no proposal at all would write two artifacts that agree on nothing and
    a test asserting "no verdict" would pass for the wrong reason.
    """
    assert adv001(before) is not None, f"{name}: no ADV001 in before; {before['proposals']}"
    ours = [v for v in verdicts(before, after) if v.code == "ADV001" and v.key[2] == "orders"]
    assert len(ours) == 1, f"{name}: expected one ADV001 verdict, got {ours}"
    [verdict] = ours
    assert verdict.applied is applied, f"{name}: applied={verdict.applied!r} {verdict.note!r}"
    assert verdict.outcome is outcome, (
        f"{name}: outcome={verdict.outcome} mean_before={verdict.mean_before} "
        f"mean_after={verdict.mean_after} note={verdict.note!r}"
    )
    assert verdict.confidence is confidence, f"{name}: confidence={verdict.confidence}"
    print(
        f"  {name}: applied={verdict.applied} {verdict.outcome.value}/"
        f"{verdict.confidence.value} mean {verdict.mean_before} -> {verdict.mean_after}"
    )


def scenario_improved() -> None:
    """DISJOINT/HIGH, applied, `improved`. The counters are cleared between the two runs, so
    the after window holds only post-index executions and nothing dilutes the gain."""
    schema = "g_improved"
    seed(schema)
    reset_statements()
    reset_database_stats()
    run_select(schema, times=5)
    before_text, before = advise(schema)
    sql(GOLDEN_DSN, f"CREATE INDEX ON {schema}.orders (status)")
    reset_statements()
    reset_database_stats()
    run_select(schema, times=500)
    after_text, after = advise(schema)
    check(
        "improved",
        before,
        after,
        applied=True,
        outcome=VerifyOutcome.IMPROVED,
        confidence=Confidence.HIGH,
    )
    write("improved", before_text, after_text)


def scenario_unchanged_nested() -> None:
    """NESTED/MEDIUM, applied, `unchanged` — the single most valuable pair here.

    `pg_stat_reset()` is called once, before both runs, so both report the same
    `stats_reset_at` and neither restricts its window: the after window *contains* the
    before window. The index is genuinely applied and genuinely helps every call it serves,
    yet 100 pre-change executions still sit in the after mean beside 5 post-change ones, so
    the mean per call barely moves and the honest verdict is `unchanged`. This is the
    pathology the design spec documents and the README warns about, in an artifact rather
    than in prose.
    """
    schema = "g_unchanged"
    seed(schema)
    reset_statements()
    reset_database_stats()
    run_select(schema, times=100)
    before_text, before = advise(schema)
    sql(GOLDEN_DSN, f"CREATE INDEX ON {schema}.orders (status)")
    run_select(schema, times=5)
    after_text, after = advise(schema)
    check(
        "unchanged_nested",
        before,
        after,
        applied=True,
        outcome=VerifyOutcome.UNCHANGED,
        confidence=Confidence.MEDIUM,
    )
    write("unchanged_nested", before_text, after_text)


def scenario_regressed() -> None:
    """DISJOINT/HIGH, applied, `regressed` — and the index is not to blame.

    The same parameterized statement is one query group whatever value it binds. Before: a
    rare `status`, so few rows come back. After: the index exists *and* the workload shifted
    to the hot value, so each call returns 400,000 rows and costs far more than it did. The
    verdict `verify` must reach is `regressed`, because that is what the measurement says —
    crediting the index for an improvement it did deliver per row, or excusing the regression
    because an index was added, would both be the tool inventing a causal story.
    """
    schema = "g_regressed"
    seed(schema, hot_rows=400_000)
    reset_statements()
    reset_database_stats()
    run_select(schema, times=20)
    before_text, before = advise(schema)
    sql(GOLDEN_DSN, f"CREATE INDEX ON {schema}.orders (status)")
    reset_statements()
    reset_database_stats()
    run_select(schema, times=20, status="hot")
    after_text, after = advise(schema)
    check(
        "regressed",
        before,
        after,
        applied=True,
        outcome=VerifyOutcome.REGRESSED,
        confidence=Confidence.HIGH,
    )
    write("regressed", before_text, after_text)


def scenario_disappeared() -> None:
    """DISJOINT/HIGH, applied, `disappeared` — the cited group is really gone.

    The after run analyses the same relation through a *different* statement, so
    `physical_state` records the new index (`applied` is a genuine `True`, not an unknown)
    while the digest the before proposal cited appears nowhere in the after window. Both runs
    used the default `--limit`, so `WindowLimits.may_be_sampling_artifact` is `False` and the
    absence is allowed to mean what it says.
    """
    schema = "g_disappeared"
    seed(schema)
    reset_statements()
    reset_database_stats()
    run_select(schema, times=20)
    before_text, before = advise(schema)
    sql(GOLDEN_DSN, f"CREATE INDEX ON {schema}.orders (status)")
    reset_statements()
    reset_database_stats()
    run_count(schema, times=20)
    after_text, after = advise(schema)
    check(
        "disappeared",
        before,
        after,
        applied=True,
        outcome=VerifyOutcome.DISAPPEARED,
        confidence=Confidence.HIGH,
    )
    write("disappeared", before_text, after_text)


def scenario_not_applied() -> None:
    """DISJOINT/HIGH, not applied, `not_applied`. Nobody created the index; the query got
    faster or slower or neither, and none of it is attributable to advice never taken."""
    schema = "g_not_applied"
    seed(schema)
    reset_statements()
    reset_database_stats()
    run_select(schema, times=20)
    before_text, before = advise(schema)
    reset_statements()
    reset_database_stats()
    run_select(schema, times=20)
    after_text, after = advise(schema)
    check(
        "not_applied",
        before,
        after,
        applied=False,
        outcome=VerifyOutcome.NOT_APPLIED,
        confidence=Confidence.HIGH,
    )
    write("not_applied", before_text, after_text)


def scenario_limit_truncated() -> None:
    """A `--limit`-truncated after window: the cited group left the window **without getting
    cheaper enough to be reported as gone**.

    `--limit 5` then `--limit 1`. The expensive `sum(length(note))` statement outranks the
    now-indexed cited group by total time, so a one-row window keeps only that one and the
    cited digest resolves nowhere in the after artifact — the identical input the
    `disappeared` scenario above feeds `verify`. The difference is the two runs' `--limit`,
    and Task 6's rule is that this difference *forbids* `DISAPPEARED`: a smaller window can
    drop a group with nothing having changed about it. So the honest verdict is
    `unobservable` at `LOW` with the mismatch named — not a disappearance, even though the
    window relation itself would have supported `HIGH`.
    """
    schema = "g_limit"
    seed(schema)
    reset_statements()
    reset_database_stats()
    run_select(schema, times=20)
    run_heavy(schema, times=1)
    before_text, before = advise(schema, "--limit", "5")
    sql(GOLDEN_DSN, f"CREATE INDEX ON {schema}.orders (status)")
    reset_statements()
    reset_database_stats()
    run_select(schema, times=20)
    run_heavy(schema, times=1)
    after_text, after = advise(schema, "--limit", "1")
    check(
        "limit_truncated",
        before,
        after,
        applied=True,
        outcome=VerifyOutcome.UNOBSERVABLE,
        confidence=Confidence.LOW,
    )
    write("limit_truncated", before_text, after_text)


def main() -> None:
    sql(
        ADMIN_DSN,
        f"DROP DATABASE IF EXISTS {GOLDEN_DB} WITH (FORCE)",
        f"CREATE DATABASE {GOLDEN_DB}",
    )
    try:
        sql(GOLDEN_DSN, "CREATE EXTENSION IF NOT EXISTS pg_stat_statements")
        for scenario in (
            scenario_improved,
            scenario_unchanged_nested,
            scenario_regressed,
            scenario_disappeared,
            scenario_not_applied,
            scenario_limit_truncated,
        ):
            print(scenario.__name__)
            scenario()
    finally:
        sql(ADMIN_DSN, f"DROP DATABASE IF EXISTS {GOLDEN_DB} WITH (FORCE)")


if __name__ == "__main__":
    main()
