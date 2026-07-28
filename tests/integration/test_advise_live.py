"""One whole `advise` run against a real database.

Every other test stubs the querier. This is the only path that exercises resolve_connection
-> connect -> six statements -> ingest -> aggregate -> propose -> render as one piece.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from sqlquality.cli import app
from sqlquality.models import ConnectionParams, Relation
from sqlquality.workload.fingerprint import ingest
from sqlquality.workload.postgres import PostgresWorkloadAdapter

pytestmark = pytest.mark.integration
runner = CliRunner()


def _run_advise(seeded: tuple[str, str], *, schemas: tuple[str, ...]) -> dict:
    """Invoke `advise --json` against the seeded database and return the whole payload.

    `--min-cost-share 0.0` so a rule firing is never masked by an unrelated cost-share
    threshold — the tests using this helper are checking *whether a rule fires at all*,
    not how it ranks against `--min-cost-share`'s default. The whole payload, not just
    `proposals`, so callers can check `analyzed.tables` as a non-vacuity guard: a rule that
    "fires" only because the relation it needed was never actually analyzed proves nothing.
    """
    dsn, _schema = seeded
    args = ["advise", "--dsn", dsn, "--json", "--min-cost-share", "0.0"]
    for name in schemas:
        args += ["--schema", name]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)


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
    # Both numbers, and the arithmetic between them. `query_groups` is what the run
    # *understood*; `query_groups_in_window` is what `pg_stat_statements` offered. A bare
    # `> 0` on the former passed equally well when it silently carried the raw window count.
    analyzed = payload["analyzed"]
    assert analyzed["query_groups"] > 0
    assert analyzed["query_groups"] == (
        analyzed["query_groups_in_window"]
        - payload["skipped"]["unqualifiable"]
        - payload["skipped"]["ambiguous"]
    )
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
    dsn, _schema = seeded
    raw = PostgresWorkloadAdapter()
    raw.connect(ConnectionParams(engine="postgres", dsn=dsn, fields={}, source="--dsn"), 30)
    fetch = raw.fetch_workload(None, 500)
    assert any("staging" in row.sql for row in fetch.rows), (
        "no staging-schema statement reached pg_stat_statements — fixture problem, not a bug"
    )

    payload = _run_advise(seeded, schemas=("public", "staging"))
    assert "public.orders" in payload["analyzed"]["tables"], payload["analyzed"]
    assert "staging.orders" in payload["analyzed"]["tables"], payload["analyzed"]

    proposals = payload["proposals"]
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
    """DECLARE is what psycopg2 server-side cursors emit; before Task 9 it was discarded.

    The cursor's inner query filters `tenant_id = 3`, a predicate no other seeded
    statement uses (see the comment in conftest.py) — its redacted query group can exist
    in `workload.stats` only if the DECLARE was actually unwrapped rather than dropped as
    noise, so asserting that group's presence (and its call count of exactly 1, since the
    cursor is opened once) pins unwrapping directly instead of merely pinning the absence
    of a literal `DECLARE` prefix, which would pass whether or not the row survived.
    """
    dsn, _schema = seeded
    adapter = PostgresWorkloadAdapter()
    adapter.connect(ConnectionParams(engine="postgres", dsn=dsn, fields={}, source="--dsn"), 30)
    fetch = adapter.fetch_workload(None, 500)
    assert any(row.sql.upper().startswith("DECLARE") for row in fetch.rows), (
        "the seeded cursor never reached pg_stat_statements — fixture problem, not a bug"
    )
    workload = ingest(fetch, "postgres")
    assert not any(s.sql.upper().startswith("DECLARE") for s in workload.stats)

    cursor_groups = [
        s
        for s in workload.stats
        if "tenant_id" in s.sql and "orders" in s.sql and "GROUP BY" not in s.sql.upper()
    ]
    assert cursor_groups, (
        "no query group carries the cursor's tenant_id = 3 predicate — the DECLARE was "
        "dropped as noise instead of unwrapped"
    )
    assert cursor_groups[0].calls == 1, (
        f"expected exactly one call (the cursor is opened once); got {cursor_groups[0].calls}"
    )


def test_the_new_rules_fire_on_a_real_workload(seeded):
    """ADV007 (join key) and ADV008 (GROUP BY) must each fire on the seeded workload.

    Asserted individually, not as a disjunction: `"ADV007" in codes or "ADV008" in codes`
    stays green if either rule's whole proposal block is deleted, so it pins neither rule.
    """
    dsn, _schema = seeded
    raw = PostgresWorkloadAdapter()
    raw.connect(ConnectionParams(engine="postgres", dsn=dsn, fields={}, source="--dsn"), 30)
    fetch = raw.fetch_workload(None, 500)
    assert any("order_items" in row.sql for row in fetch.rows), (
        "the seeded join query never reached pg_stat_statements — fixture problem, not a bug"
    )
    assert any("GROUP BY" in row.sql.upper() for row in fetch.rows), (
        "the seeded GROUP BY query never reached pg_stat_statements — fixture problem, not a bug"
    )

    payload = _run_advise(seeded, schemas=("public", "staging"))
    codes = {p["code"] for p in payload["proposals"]}
    assert "ADV007" in codes, f"join-key rule did not fire; got {sorted(codes)}"
    assert "ADV008" in codes, f"grouping rule did not fire; got {sorted(codes)}"


def test_never_analysed_join_key_still_proposes_at_low_confidence(seeded):
    """Batch 1's `reltuples = -1` bug, downstream of the catalog read: a join-key proposal
    on a never-analysed table must still fire, at LOW confidence with an unknown row
    estimate — not be silently suppressed by reading -1 as "this table is tiny".

    `public.order_items` (see conftest.py) has `autovacuum_enabled = false` and is never
    explicitly ANALYZEd, so its `reltuples` stays -1 for the whole run.
    """
    dsn, _schema = seeded
    order_items = Relation("public", "order_items")
    raw = PostgresWorkloadAdapter()
    raw.connect(ConnectionParams(engine="postgres", dsn=dsn, fields={}, source="--dsn"), 30)
    facts = raw.fetch_table_facts(("public",), frozenset({order_items}))
    assert facts[order_items].row_estimate is None, (
        "public.order_items reports a real row count — it was analysed before this "
        "assertion ran, so the None-path assertion below would be vacuous"
    )

    payload = _run_advise(seeded, schemas=("public", "staging"))
    order_items_proposals = [
        p
        for p in payload["proposals"]
        if p["code"] == "ADV007" and p["evidence"].get("table") == "order_items"
    ]
    assert order_items_proposals, "ADV007 did not fire for public.order_items"
    assert order_items_proposals[0]["evidence"]["row_estimate"] is None
    assert order_items_proposals[0]["confidence"] == "low", order_items_proposals[0]


def _adv001_for(payload: dict, *, schema: str, table: str) -> dict | None:
    for p in payload["proposals"]:
        if p["code"] == "ADV001" and p["evidence"].get("schema") == schema:
            if p["evidence"].get("table") == table:
                return p
    return None


def test_adv302_rewrites_a_real_index_proposal_into_dbt_config(seeded, tmp_path):
    """ADV302's whole reason to exist, proven on live data rather than a fixture.

    A raw `CREATE INDEX` on a dbt `table`-materialized relation does not survive the next
    `dbt run` (it drops and recreates the relation), so ADV302 rewrites that proposal into
    a config block instead of doomed DDL. `tests/fixtures/manifest_v12.json` cannot prove
    this live: its `relation_name`s are all schema `"main"`, which this seeded database
    never has (`public`/`staging`), and relation matching is on the qualified
    `(schema, table)` pair with no bare-name fallback -- so a manifest built from schemas
    that do not match what got seeded would match nothing, and the test would pass while
    proving nothing. This manifest declares `public.orders` instead, which conftest.py's
    `seeded` fixture actually creates.

    Non-vacuity guard first: the *un-enriched* run must really emit `CREATE INDEX` for
    `public.orders` (ADV001, on the hot `status` predicate -- see conftest.py's seeded
    workload). Without this half, the enriched assertion below would pass just as well if
    the workload simply produced no proposal for that relation at all.
    """
    dsn, _schema = seeded

    bare = _run_advise(seeded, schemas=("public", "staging"))
    bare_orders = _adv001_for(bare, schema="public", table="orders")
    assert bare_orders is not None, (
        f"no un-enriched ADV001 for public.orders; got "
        f"{[(p['code'], p['evidence'].get('schema'), p['evidence'].get('table')) for p in bare['proposals']]}"
    )
    assert bare_orders["ddl"] is not None
    assert bare_orders["ddl"].upper().lstrip().startswith("CREATE INDEX"), bare_orders
    assert "dbt_index_config" not in bare_orders["evidence"], bare_orders

    manifest = {
        "metadata": {
            "dbt_schema_version": "https://schemas.getdbt.com/dbt/manifest/v12.json",
            "adapter_type": "postgres",
        },
        "nodes": {
            "model.live_it.orders": {
                "unique_id": "model.live_it.orders",
                "name": "orders",
                "resource_type": "model",
                "config": {"materialized": "table"},
                "compiled_code": "select * from {{ source('raw', 'orders') }}",
                # The database part ("analytics") is deliberately NOT what conftest.py's
                # `seeded` fixture actually connects to -- parse_relation_name drops it,
                # since `advise` connects to one database at a time. Only the
                # (schema, table) pair below has to match what got seeded.
                "relation_name": '"analytics"."public"."orders"',
                "depends_on": {"macros": [], "nodes": []},
            }
        },
        "sources": {},
        "parent_map": {"model.live_it.orders": []},
        "child_map": {"model.live_it.orders": []},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "advise",
            "--dsn",
            dsn,
            "--schema",
            "public",
            "--schema",
            "staging",
            "--json",
            "--min-cost-share",
            "0.0",
            "--manifest",
            str(manifest_path),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["dbt"]["models"] == 1, payload.get("dbt")
    assert payload["dbt"]["dropped_collisions"] == 0, payload.get("dbt")

    enriched = _adv001_for(payload, schema="public", table="orders")
    assert enriched is not None, "the dbt-managed public.orders proposal disappeared entirely"
    assert not (enriched["ddl"] or "").upper().lstrip().startswith("CREATE INDEX"), enriched
    assert "indexes:" in (enriched["ddl"] or ""), enriched
    assert enriched["evidence"]["dbt_model"] == "model.live_it.orders"
    assert enriched["evidence"]["dbt_materialized"] == "table"
    assert "dbt_index_config" in enriched["evidence"], enriched
    assert enriched["ddl"] == enriched["evidence"]["dbt_index_config"]
