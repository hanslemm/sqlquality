"""One whole `advise` run against a real database.

Every other test stubs the querier. This is the only path that exercises resolve_connection
-> connect -> six statements -> ingest -> aggregate -> propose -> render as one piece.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from sqlquality.cli import app
from sqlquality.models import ConnectionParams
from sqlquality.workload.fingerprint import ingest
from sqlquality.workload.postgres import PostgresWorkloadAdapter

pytestmark = pytest.mark.integration
runner = CliRunner()


def _run_advise(seeded: tuple[str, str], *, schemas: tuple[str, ...]) -> list[dict]:
    """Invoke `advise --json` against the seeded database and return its proposals.

    `--min-cost-share 0.0` so a rule firing is never masked by an unrelated cost-share
    threshold — the tests using this helper are checking *whether a rule fires at all*,
    not how it ranks against `--min-cost-share`'s default.
    """
    dsn, _schema = seeded
    args = ["advise", "--dsn", dsn, "--json", "--min-cost-share", "0.0"]
    for name in schemas:
        args += ["--schema", name]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)["proposals"]


def test_advise_end_to_end(seeded, tmp_path):
    dsn, schema = seeded
    md = tmp_path / "report.md"
    ddl = tmp_path / "proposals.sql"
    result = runner.invoke(
        app,
        [
            "advise",
            "--dsn",
            dsn,
            "--schema",
            schema,
            "--json",
            "--markdown",
            str(md),
            "--ddl",
            str(ddl),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)

    assert payload["engine"] == "postgres"
    assert payload["redacted"] is True
    assert payload["analyzed"]["query_groups"] > 0
    assert payload["degraded"] == []
    assert md.read_text(encoding="utf-8").startswith("# sqlquality advise")
    assert "REVIEW BEFORE RUNNING" in ddl.read_text(encoding="utf-8")


def test_advise_output_carries_no_query_literal_from_a_real_server(seeded, tmp_path):
    """A regression guard on the pipeline, NOT a test of `redact_tree`. Read on.

    Postgres normalises constants to `$N` inside `pg_stat_statements` before sqlquality ever
    sees the query text, so `'paid'` is already gone on arrival. Measured: running this
    scenario with `--keep-literals`, which bypasses our redaction entirely, still shows no
    `'paid'` anywhere. **This test therefore cannot fail if `redact_tree` breaks**, and it
    would be dishonest to call it redaction coverage.

    What it does pin, which is worth pinning: that nothing downstream of ingest — evidence
    dicts, rationales, DDL, the renderers — reintroduces raw query text into an artifact.
    That is a real regression risk, since ADV005 and ADV006 both copy SQL into evidence.

    The actual guard on `redact_tree` is `tests/test_workload_redaction.py`, which feeds
    un-normalised literals through a fake querier and *does* fail when redaction is
    disabled — verified there by mutation.
    """
    dsn, schema = seeded
    md = tmp_path / "report.md"
    result = runner.invoke(
        app, ["advise", "--dsn", dsn, "--schema", schema, "--json", "--markdown", str(md)]
    )
    assert result.exit_code == 0, result.output
    assert "'paid'" not in result.stdout
    assert "'paid'" not in md.read_text(encoding="utf-8")
    # Pin the reason this test is weak, so nobody later mistakes it for redaction coverage:
    # the literal is already absent from what Postgres hands us. `sql` lives inside each
    # proposal's `evidence` dict (only ADV005's leading-wildcard-LIKE and ADV006 carry it),
    # not as a top-level proposal key — see `advise_payload` in report.py.
    fetch_sql = " ".join(
        p["evidence"]["sql"]
        for p in json.loads(result.stdout).get("proposals", [])
        if "sql" in p.get("evidence", {})
    )
    assert "'paid'" not in fetch_sql


def test_advise_dry_run_needs_no_server(tmp_path):
    """The audit path must not depend on anything being reachable."""
    result = runner.invoke(app, ["advise", "--engine", "postgres", "--dry-run"])
    assert result.exit_code == 0
    assert "pg_stat_statements" in result.stdout


def test_multi_schema_advise_run_produces_qualified_proposals(seeded):
    """A multi-schema run must attribute proposals to the schema they came from.

    `staging` holds an index (`idx_unused_staging_id`) the workload never touches, and
    `public` holds a hot, unindexed equality predicate — so a real run against both
    schemas must produce at least one DROP INDEX for `staging` and at least one CREATE
    INDEX for `public`, coexisting in the same proposal list.
    """
    proposals = _run_advise(seeded, schemas=("public", "staging"))
    schemas = {p["evidence"].get("schema") for p in proposals}
    assert "staging" in schemas, f"no proposal attributed to staging; got {sorted(schemas)}"
    assert "public" in schemas, f"no proposal attributed to public; got {sorted(schemas)}"

    # Every DDL statement names the schema of the relation it belongs to.
    ddl_actions = set()
    for proposal in proposals:
        ddl = proposal["ddl"]
        if not ddl:
            continue
        assert f'"{proposal["evidence"]["schema"]}".' in ddl, proposal
        ddl_actions.add((proposal["evidence"]["schema"], ddl.split()[0]))

    assert ("public", "CREATE") in ddl_actions, (
        f"no CREATE INDEX proposal for public; got {sorted(ddl_actions)}"
    )
    assert ("staging", "DROP") in ddl_actions, (
        f"no DROP INDEX proposal for staging; got {sorted(ddl_actions)}"
    )


def test_a_declared_cursor_reaches_the_analysis(seeded):
    """DECLARE is what psycopg2 server-side cursors emit; before Task 9 it was discarded."""
    dsn, _schema = seeded
    adapter = PostgresWorkloadAdapter()
    adapter.connect(ConnectionParams(engine="postgres", dsn=dsn, fields={}, source="--dsn"), 30)
    fetch = adapter.fetch_workload(None, 500)
    assert any(row.sql.upper().startswith("DECLARE") for row in fetch.rows), (
        "the seeded cursor never reached pg_stat_statements — fixture problem, not a bug"
    )
    workload = ingest(fetch, "postgres")
    assert not any(s.sql.upper().startswith("DECLARE") for s in workload.stats)
    assert any("orders" in s.sql and "pending" not in s.sql for s in workload.stats)


def test_the_new_rules_fire_on_a_real_workload(seeded):
    """ADV007 (join key) and/or ADV008 (GROUP BY) must fire on the seeded workload."""
    proposals = _run_advise(seeded, schemas=("public", "staging"))
    codes = {p["code"] for p in proposals}
    assert "ADV007" in codes or "ADV008" in codes, (
        f"neither join-key nor grouping rule fired; got {sorted(codes)}"
    )
