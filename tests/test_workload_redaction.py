"""Guarantee: with default settings, no literal from the query log reaches any output.

Query history can contain personal data in predicates. This is the test that fails if a
future change lets a literal escape into a report, a DDL script, or the JSON payload.
"""

import json

import pytest
from typer.testing import CliRunner

from sqlquality.cli import app

runner = CliRunner()

#: Distinctive strings that must never appear downstream.
#:
#: Be honest about which of these currently discriminate. Measured by disabling
#: `redact_tree` and sweeping all four surfaces: only `hans@betterdoc.de` leaks, because it
#: is the only one reaching ADV005's evidence — the single proposal type that copies query
#: text into output. The other three enter via HOT_QUERY, whose proposals carry column
#: names, roles and counts but never `stat.sql`, so they cannot leak today whatever
#: redaction does.
#:
#: They stay in the list deliberately, as regression insurance against a future proposal
#: type that widens what evidence carries. But they are future-proofing, not live
#: trip-wires, and a reader should not mistake four passing checks for four proofs.
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

#: Load-bearing, and the reason this whole module is not decoration.
#:
#: A workload built only from equality/range predicates (like HOT_QUERY) produces index
#: proposals whose evidence carries *column names and roles*, never raw SQL — so
#: `redact_tree` could be deleted entirely and no secret would appear in any output. The
#: guarantee test would pass while guaranteeing nothing.
#:
#: Exactly one path in the pipeline copies query text downstream: ADV005's
#: leading-wildcard-LIKE evidence, which reports at query-group level because the pattern
#: was redacted before we could attribute a column. This row is what reaches it.
#:
#: If a future proposal type starts embedding raw SQL, it needs its own row here — and the
#: `--keep-literals` test below is the canary that will tell you: it asserts a secret *is*
#: retained with the opt-in, which can only hold if some output surface actually carries
#: query text. If that test starts failing, this scenario has gone vacuous.
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

    # `ddl` currently has no live leak channel either: ADV005 (the only evidence block
    # carrying query text) always sets ddl=None, and generated DDL is built from quoted
    # identifiers, never predicate literals. Measured: with redaction disabled, stdout,
    # markdown and json all leak while ddl stays clean. It is asserted anyway, because a
    # literal-valued partial index would change that — but three surfaces, not four, are
    # doing real work today.
    surfaces = {
        "stdout": result.stdout,
        "markdown": md.read_text(),
        "ddl": ddl.read_text(),
        "json": json.dumps(json.loads(result.stdout)),
    }
    # Collect every leak before asserting. A bare assert inside the loop short-circuits on
    # the first hit, so a break affecting three surfaces would report only one.
    leaks = [
        f"{secret!r} leaked into {name}"
        for name, content in surfaces.items()
        for secret in SECRETS
        if secret in content
    ]
    assert not leaks, "; ".join(leaks)


def test_analysis_still_works_on_redacted_sql(stubbed):
    """Redaction must not cost us the advice — the point of flags-before-redaction."""
    result = runner.invoke(app, ["advise", "--dsn", "postgresql://u@h/db", "--json"])
    payload = json.loads(result.stdout)
    assert payload["proposals"], "redaction must not silence the analysis"
    assert payload["redacted"] is True
    # Non-vacuity guard: ADV005 is the only proposal type that carries query text into the
    # output, so its presence is what makes the leak assertions above meaningful. Without
    # it, "no secret in the output" holds trivially because no SQL is in the output at all.
    assert any(p["code"] == "ADV005" for p in payload["proposals"]), (
        "scenario no longer exercises a raw-SQL output path — the guarantee test above "
        "would pass vacuously"
    )


def test_keep_literals_is_the_only_way_to_retain_values(stubbed):
    result = runner.invoke(
        app, ["advise", "--dsn", "postgresql://u@h/db", "--keep-literals", "--json"]
    )
    payload = json.loads(result.stdout)
    assert payload["redacted"] is False
    # With the opt-in, values are retained — proving the default was doing real work.
    assert "hans@betterdoc.de" in json.dumps(payload)
