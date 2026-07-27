"""One whole `advise` run against a real database.

Every other test stubs the querier. This is the only path that exercises resolve_connection
-> connect -> six statements -> ingest -> aggregate -> propose -> render as one piece.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from sqlquality.cli import app

pytestmark = pytest.mark.integration
runner = CliRunner()


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
