"""End-to-end pins for `physical_state`'s wiring into a real `advise --json` run.

Review finding 3 on Task 2: replacing `physical_state=adapter.physical_state(
proposal_relations(proposals))` with `physical_state=adapter.physical_state(frozenset())`
in `cli.py` makes the key permanently `{}` on every real run, and the whole suite (999
tests at the time of the finding) stayed green — the third time this project's wiring
could be silently deleted without a single failure. These tests assert against a real
`advise --json` invocation that the proposals' own relations actually appear in
`physical_state`, with real facts, not merely that the key exists.

Deliberately a separate file rather than an addition to `tests/test_advise_cli.py`: that
file's existing tests are a constraint of Task 2's brief ("must pass unmodified"), so
kept untouched. `_stub_adapter` below is a small, deliberate duplicate of that file's
private helper of the same name — cross-file imports of a leading-underscore name are
fragile, and this project's own tests never rely on one.
"""

import json
from pathlib import Path

from typer.testing import CliRunner

from sqlquality.cli import app
from sqlquality.workload.postgres import PostgresWorkloadAdapter

runner = CliRunner()

DBT_FIXTURE = Path(__file__).parent / "fixtures" / "manifest_v12.json"


def _stub_adapter(monkeypatch, rows):
    """Replace `connect()` with an injected fake querier that dispatches on a SQL
    substring — the same shape as `test_advise_cli._stub_adapter`."""
    recorded: list[tuple[str, object]] = []

    def fake_connect(self, params, timeout_s):
        def query(sql, bind):
            recorded.append((sql, bind))
            for marker, result in rows.items():
                if marker in sql:
                    return result
            return []

        self._query = query

    monkeypatch.setattr(PostgresWorkloadAdapter, "connect", fake_connect)
    return recorded


def test_physical_state_reaches_the_payload_from_a_real_cli_run(monkeypatch):
    """Positive wiring pin: a proposal's own relation must appear in `physical_state`,
    with its real, fetched facts — killed by deleting the `cli.py` wiring, and killed by
    neutering `proposal_relations` to always return an empty set.
    """
    _stub_adapter(
        monkeypatch,
        {
            "pg_stat_statements": [("select id from orders where status = $1", 50, 500.0, 50)],
            "pg_stat_database": [("2026-07-01",)],
            "information_schema.columns": [
                ("public", "orders", "id", "integer"),
                ("public", "orders", "status", "text"),
            ],
            "pg_total_relation_size": [("public", "orders", 50_000, 1024)],
            "pg_stats": [("public", "orders", "status", 5000.0)],
            "pg_index": [
                (
                    "public",
                    "orders",
                    "idx_status",
                    "status",
                    1,
                    False,
                    False,
                    0,
                    100,
                    False,
                    None,
                    False,
                    "...",
                )
            ],
        },
    )
    result = runner.invoke(app, ["advise", "--dsn", "postgresql://u@h/db", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    codes = {p["code"] for p in payload["proposals"]}
    # idx_status has zero scans and leads with the hot equality column, so ADV002 (unused
    # index) fires for `public.orders` regardless of whether ADV001 is also suppressed by
    # `_covered` seeing that same index — this scenario only needs *a* proposal targeting
    # `public.orders` to exist, not a specific one.
    assert "ADV002" in codes, (
        f"the scenario must produce a proposal targeting public.orders: {codes}"
    )
    proposal_relations_seen = {
        (p["evidence"]["schema"], p["evidence"]["table"])
        for p in payload["proposals"]
        if "schema" in p["evidence"] and "table" in p["evidence"]
    }
    assert ("public", "orders") in proposal_relations_seen
    assert "public.orders" in payload["physical_state"], (
        "physical_state was empty for a relation a real proposal targets — the CLI wiring "
        "from proposals to adapter.physical_state() is broken"
    )
    entry = payload["physical_state"]["public.orders"]
    assert entry["is_ordinary_table"] is True
    assert entry["indexes"] == [
        {"name": "idx_status", "columns": ["status"], "is_partial": False, "is_unique": False}
    ]


def test_physical_state_covers_a_relation_the_workload_touched_but_no_proposal_remains_for(
    monkeypatch,
):
    """The Critical fix from Task 6's fix round 1. Before this fix, `physical_state` was
    scoped to `proposal_relations(proposals)` alone — the relations *some proposal
    targets* — so a relation the workload actually touched, but for which no rule fired
    (because it is already well-covered, or too cold to clear `--min-cost-share`), carried
    no `physical_state` entry at all. That is precisely the shape of "the recommended
    index now exists, so the rule stopped firing": `sqlquality verify` would have been
    structurally unable to confirm its own headline case. `physical_state` is now scoped
    to `proposal_relations(proposals) | aggregation.tables`, so a workload-touched relation
    is reported whether or not it still has a finding.

    `public.customers` here is queried (so it reaches `aggregation.tables` via its `WHERE`
    predicate) but at a cost share far below `--min-cost-share`'s default (1%) and with no
    catalog index at all for ADV002/ADV003 to react to — so it produces zero proposals,
    the exact "resolved" shape this fix exists for. Before the fix, `"public.customers"`
    would be entirely absent from `payload["physical_state"]`.
    """
    _stub_adapter(
        monkeypatch,
        {
            "pg_stat_statements": [
                ("select id from orders where status = $1", 50, 500.0, 50),
                ("select id from customers where region = $1", 1, 0.01, 1),
            ],
            "pg_stat_database": [("2026-07-01",)],
            "information_schema.columns": [
                ("public", "orders", "id", "integer"),
                ("public", "orders", "status", "text"),
                ("public", "customers", "id", "integer"),
                ("public", "customers", "region", "text"),
            ],
            "pg_total_relation_size": [
                ("public", "orders", 50_000, 1024),
                ("public", "customers", 5_000, 2048),
            ],
            "pg_stats": [
                ("public", "orders", "status", 5000.0),
                ("public", "customers", "region", 500.0),
            ],
            "pg_index": [
                (
                    "public",
                    "orders",
                    "idx_status",
                    "status",
                    1,
                    False,
                    False,
                    0,
                    100,
                    False,
                    None,
                    False,
                    "...",
                )
            ],
        },
    )
    result = runner.invoke(app, ["advise", "--dsn", "postgresql://u@h/db", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)

    # The scenario must produce zero proposals for public.customers, or this test proves
    # nothing about the scoping fix specifically (as opposed to some other proposal
    # incidentally naming it).
    customers_proposals = {
        p["code"]
        for p in payload["proposals"]
        if p["evidence"].get("schema") == "public" and p["evidence"].get("table") == "customers"
    }
    assert customers_proposals == set(), (
        f"the scenario must produce no proposal for public.customers: {customers_proposals}"
    )
    assert "public.customers" in payload["analyzed"]["tables"], (
        "the scenario must actually reach aggregation.tables for this test to prove "
        "anything about the widened scoping"
    )
    assert "public.customers" in payload["physical_state"], (
        "physical_state omitted a relation the workload touched but no proposal remains "
        "for — the union-with-aggregation.tables scoping fix is not wired in"
    )
    entry = payload["physical_state"]["public.customers"]
    assert entry["is_ordinary_table"] is True
    assert entry["indexes"] == []


def test_physical_state_reports_a_never_fetched_adv303_relation_as_unknown(monkeypatch):
    """The exact ADV303 shape the review's Critical finding describes: `customer_orders`
    (in the fixture manifest's `main` schema) has no declared consumer, so `propose_unused_
    models` flags it — and it is entirely disjoint from the `public.orders` workload below,
    so `fetch_table_facts`/`fetch_indexes` were never called with it at all. Before the fix
    this reported `is_ordinary_table: False, indexes: []`; it must now report both as `None`.
    """
    rows = {
        "pg_stat_statements": [
            ("select id from orders where status = $1", 5, 100.0, 5),
        ],
        "pg_stat_database": [("2026-07-01",)],
        "information_schema.columns": [
            ("public", "orders", "id", "integer"),
            ("public", "orders", "status", "text"),
        ],
        "pg_total_relation_size": [("public", "orders", 5_000_000, 10**8)],
        "pg_stats": [("public", "orders", "status", 5000.0)],
        "pg_index": [],
    }
    _stub_adapter(monkeypatch, rows)
    result = runner.invoke(
        app,
        ["advise", "--dsn", "postgresql://u@h/db", "--manifest", str(DBT_FIXTURE), "--json"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    codes = {p["code"] for p in payload["proposals"]}
    assert "ADV303" in codes, (
        f"the scenario must produce ADV303 for this test to prove anything: {codes}"
    )
    assert "main.customer_orders" in payload["physical_state"]
    entry = payload["physical_state"]["main.customer_orders"]
    assert entry["is_ordinary_table"] is None, "never fetched must read as unknown, not False"
    assert entry["indexes"] is None, "never fetched must read as unknown, not []"
