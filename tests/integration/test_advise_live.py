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


def test_advise_does_not_leak_a_literal_from_a_real_server(seeded, tmp_path):
    """The redaction guarantee, against real pg_stat_statements rather than a fixture.

    The seeded workload filters on the literal 'paid'. pg_stat_statements normalises it to
    $1, but a run with --keep-literals proves the surfaces would carry it if we let them.
    """
    dsn, schema = seeded
    md = tmp_path / "report.md"
    result = runner.invoke(
        app, ["advise", "--dsn", dsn, "--schema", schema, "--json", "--markdown", str(md)]
    )
    assert result.exit_code == 0, result.output
    assert "'paid'" not in result.stdout
    assert "'paid'" not in md.read_text(encoding="utf-8")


def test_advise_dry_run_needs_no_server(tmp_path):
    """The audit path must not depend on anything being reachable."""
    result = runner.invoke(app, ["advise", "--engine", "postgres", "--dry-run"])
    assert result.exit_code == 0
    assert "pg_stat_statements" in result.stdout
