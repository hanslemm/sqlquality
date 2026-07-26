"""Guarantee: with default settings, no literal from the query log reaches any output.

Query history can contain personal data in predicates. This is the test that fails if a
future change lets a literal escape into a report, a DDL script, or the JSON payload.
"""

import json

import pytest
from typer.testing import CliRunner

from sqlquality.cli import app

runner = CliRunner()

#: Distinctive strings that must never appear downstream. If any of these leaks, the
#: redaction path is broken.
SECRETS = ("patient-4711", "hans@betterdoc.de", "DE89370400440532013000", "1990-04-17")

HOT_QUERY = (
    "select id, note from orders "
    "where status = 'shipped' "
    "and customer_email = 'hans@betterdoc.de' "
    "and reference = 'patient-4711' "
    "and iban = 'DE89370400440532013000' "
    "and created_at > '1990-04-17' "
    "order by created_at desc"
)

#: A second, distinct query group. It exists purely so the guarantee test can actually
#: fail: `propose_sargability`'s leading-wildcard-LIKE proposal is the only place in the
#: current pipeline that ever copies a query group's *SQL text* (not just column names)
#: into evidence, so it is the only surface a leaked literal could ever land on. A workload
#: built only from equality/range predicates (as in HOT_QUERY) yields index proposals whose
#: evidence carries column names and roles, never raw SQL — so `redact_tree` could be
#: deleted entirely and no secret would show up anywhere. This row closes that gap.
LIKE_QUERY = "select note from orders where note like '%hans@betterdoc.de%'"

COLUMNS = [
    ("orders", "id", "integer"),
    ("orders", "note", "text"),
    ("orders", "status", "text"),
    ("orders", "customer_email", "text"),
    ("orders", "reference", "text"),
    ("orders", "iban", "text"),
    ("orders", "created_at", "timestamp"),
]

ROWS = {
    "pg_stat_statements": [
        (HOT_QUERY, 500, 90_000.0, 10),
        (LIKE_QUERY, 300, 20_000.0, 5),
    ],
    "pg_stat_database": [("2026-07-01",)],
    "information_schema.columns": COLUMNS,
    "pg_total_relation_size": [("orders", 8_000_000, 10**9)],
    "pg_stats": [("orders", "status", 4.0), ("orders", "customer_email", 900_000.0)],
    "pg_index": [],
}


@pytest.fixture
def stubbed(monkeypatch):
    from sqlquality.workload.postgres import PostgresWorkloadAdapter

    def fake_connect(self, params, timeout_s):
        def query(sql, bind):
            for marker, result in ROWS.items():
                if marker in sql:
                    return result
            return []

        self._query = query

    monkeypatch.setattr(PostgresWorkloadAdapter, "connect", fake_connect)


def test_no_literal_reaches_json_markdown_ddl_or_stdout(stubbed, tmp_path):
    md = tmp_path / "report.md"
    ddl = tmp_path / "proposals.sql"
    result = runner.invoke(
        app,
        [
            "advise",
            "--dsn",
            "postgresql://u@h/db",
            "--markdown",
            str(md),
            "--ddl",
            str(ddl),
            "--json",
        ],
    )
    assert result.exit_code == 0

    surfaces = {
        "stdout": result.stdout,
        "markdown": md.read_text(),
        "ddl": ddl.read_text(),
        "json": json.dumps(json.loads(result.stdout)),
    }
    for name, content in surfaces.items():
        for secret in SECRETS:
            assert secret not in content, f"{secret!r} leaked into {name}"


def test_analysis_still_works_on_redacted_sql(stubbed):
    """Redaction must not cost us the advice — the point of flags-before-redaction."""
    result = runner.invoke(app, ["advise", "--dsn", "postgresql://u@h/db", "--json"])
    payload = json.loads(result.stdout)
    assert payload["proposals"], "redaction must not silence the analysis"
    assert payload["redacted"] is True


def test_keep_literals_is_the_only_way_to_retain_values(stubbed):
    result = runner.invoke(
        app, ["advise", "--dsn", "postgresql://u@h/db", "--keep-literals", "--json"]
    )
    payload = json.loads(result.stdout)
    assert payload["redacted"] is False
    # With the opt-in, values are retained — proving the default was doing real work.
    assert "hans@betterdoc.de" in json.dumps(payload)
