from sqlquality.delta import ModelDelta
from sqlquality.gate import GateReport
from sqlquality.models import (
    Aggregation,
    ColumnRole,
    ColumnUsage,
    Confidence,
    Proposal,
    QueryStat,
    Relation,
    TableFacts,
    Workload,
)
from sqlquality.report import advise_payload, render_advise_markdown, render_markdown
from sqlquality.workload.fingerprint import (
    FLAG_LEADING_WILDCARD_LIKE,
    FLAG_SELECT_STAR,
    fingerprint_id,
)
from sqlquality.workload.postgres import (
    propose_indexes,
    propose_join_keys,
    propose_sargability,
    propose_select_star,
)

PASS = GateReport(
    deltas=[ModelDelta("model.demo.orders", 10.7, 11.1, 0.4, False)],
    regressions=[],
    passed=True,
)
FAIL = GateReport(
    deltas=[ModelDelta("model.demo.orders", 5.0, 30.0, 25.0, False)],
    regressions=["model.demo.orders"],
    passed=False,
    mode="fail",
)
WARN = GateReport(
    deltas=[ModelDelta("model.demo.orders", 5.0, 30.0, 25.0, False)],
    regressions=["model.demo.orders"],
    passed=True,
    mode="warn",
)


def test_markdown_pass():
    md = render_markdown(PASS)
    assert "# sqlquality" in md
    assert "PASS" in md
    assert "model.demo.orders" in md
    assert "| model |" in md  # a markdown table header


def test_markdown_fail_marks_regression():
    md = render_markdown(FAIL)
    assert "FAIL" in md
    assert "model.demo.orders" in md


def test_markdown_lists_skipped():
    md = render_markdown(PASS, skipped=[("seed.demo.raw", "no compiled SQL")])
    assert "seed.demo.raw" in md
    assert "no compiled SQL" in md


def test_markdown_warn_mode_shows_warn():
    md = render_markdown(WARN)
    assert "WARN" in md
    assert "gate mode: warn" in md
    assert "PASS" not in md
    # single regression is singular, not "1 regressions"
    assert "1 regression," in md
    assert "1 regressions" not in md


def test_markdown_warn_mode_pluralizes():
    two = GateReport(
        deltas=[
            ModelDelta("model.demo.a", 5.0, 30.0, 25.0, False),
            ModelDelta("model.demo.b", 5.0, 30.0, 25.0, False),
        ],
        regressions=["model.demo.a", "model.demo.b"],
        passed=True,
        mode="warn",
    )
    assert "2 regressions," in render_markdown(two)


def test_markdown_injection_is_inert():
    injected = GateReport(
        deltas=[ModelDelta(" | 0 | ✅ PASS injected", 1.0, 2.0, 1.0, False)],
        regressions=[],
        passed=True,
    )
    md = render_markdown(injected, skipped=[("x", "<img src=x onerror=alert(1)>")])
    # pipe injection can't fabricate table columns, and raw HTML is neutralized.
    assert "\\|" in md
    assert " | 0 | ✅ PASS injected |" not in md
    assert "<img" not in md
    assert "&lt;img" in md


PROPOSALS = [
    Proposal(
        code="ADV001",
        title="Add index on orders(status)",
        rationale="hot predicate",
        evidence={"cost_share": 0.42, "calls": 100, "table": "orders"},
        confidence=Confidence.HIGH,
        ddl="CREATE INDEX ON orders (status);",
    ),
]
WORKLOAD = Workload(
    stats=(
        QueryStat(
            fingerprint="fp",
            sql="select id from orders where status = $1",
            calls=100,
            total_time_ms=500.0,
        ),
    ),
    window_description="since stats reset at 2026-07-01",
    skipped_unparseable=2,
    skipped_noise=7,
)
AGGREGATION = Aggregation(
    usage=(),
    total_cost_ms=500.0,
    skipped_unqualifiable=3,
    tables=frozenset({Relation("public", "orders")}),
    skipped_ambiguous=4,
)


def _payload():
    return advise_payload(
        PROPOSALS,
        WORKLOAD,
        AGGREGATION,
        engine="postgres",
        redacted=True,
        degraded=[("ndv", "permission denied")],
    )


def test_payload_reports_proposals_window_and_skips():
    payload = _payload()
    assert payload["engine"] == "postgres"
    assert payload["redacted"] is True
    assert payload["window"] == {
        "description": "since stats reset at 2026-07-01",
        "engine": "postgres",
        "stats_reset_at": None,
        "since": None,
        "since_duration_seconds": None,
        "limit": None,
    }
    assert payload["proposals"][0]["code"] == "ADV001"
    assert payload["skipped"] == {
        "unparseable": 2,
        "noise": 7,
        "unqualifiable": 3,
        "ambiguous": 4,
    }
    assert payload["degraded"] == [{"capability": "ndv", "reason": "permission denied"}]


def test_window_is_an_object_carrying_what_the_comparison_needs():
    """`verify` classifies two runs' windows as nested, disjoint or comparable, and cannot
    do that from a prose sentence. Each field answers one question: `stats_reset_at`
    whether Postgres's cumulative counters were cleared between runs, `since_duration_seconds`
    whether the same explicit duration was requested on both sides, `limit` whether the
    window was truncated."""
    payload = advise_payload(
        [],
        Workload(stats=(), window_description="since stats reset at 2026-08-01T00:00:00"),
        Aggregation(usage=(), total_cost_ms=0.0, skipped_unqualifiable=0, tables=frozenset()),
        engine="postgres",
        redacted=True,
        degraded=[],
        window_facts={
            "stats_reset_at": "2026-08-01T00:00:00",
            "since": None,
            "since_duration_seconds": None,
            "limit": 500,
        },
    )
    window = payload["window"]
    assert isinstance(window, dict), "a prose string cannot be compared across runs"
    assert window["description"] == "since stats reset at 2026-08-01T00:00:00"
    assert window["engine"] == "postgres"
    assert window["stats_reset_at"] == "2026-08-01T00:00:00"
    assert window["since"] is None
    assert window["since_duration_seconds"] is None
    assert window["limit"] == 500


def test_the_window_description_is_preserved_verbatim():
    """The prose sentence is what a human reads and it is already carefully worded — the
    structured fields are added beside it, not instead of it."""
    payload = advise_payload(
        [],
        Workload(stats=(), window_description="since stats reset at T (--since is not supported)"),
        Aggregation(usage=(), total_cost_ms=0.0, skipped_unqualifiable=0, tables=frozenset()),
        engine="postgres",
        redacted=True,
        degraded=[],
        window_facts={},
    )
    assert payload["window"]["description"] == ("since stats reset at T (--since is not supported)")


def test_missing_window_facts_are_null_not_absent():
    """A key that is absent and a key that is null are different to a consumer. `verify`
    distinguishes "this engine cannot tell you" from "this field was never written", and
    only the second is a reason to reject the artifact."""
    payload = advise_payload(
        [],
        Workload(stats=(), window_description="w"),
        Aggregation(usage=(), total_cost_ms=0.0, skipped_unqualifiable=0, tables=frozenset()),
        engine="postgres",
        redacted=True,
        degraded=[],
        window_facts={},
    )
    for key in ("stats_reset_at", "since", "since_duration_seconds", "limit"):
        assert key in payload["window"], f"{key} must be present even when unknown"
        assert payload["window"][key] is None


def test_physical_state_is_always_present_even_when_the_caller_gave_none():
    """Unlike `dbt`, this key must never be omitted: an *absent* `physical_state` is how
    `verify` tells a pre-this-feature artifact apart from one that genuinely found nothing
    physical to report — omitting it on an empty result would blur that distinction."""
    payload = advise_payload(
        [],
        Workload(stats=(), window_description="w"),
        Aggregation(usage=(), total_cost_ms=0.0, skipped_unqualifiable=0, tables=frozenset()),
        engine="postgres",
        redacted=True,
        degraded=[],
    )
    assert "physical_state" in payload
    assert payload["physical_state"] == {}


def test_physical_state_carries_what_the_adapter_reported():
    """The caller hands in `adapter.physical_state(relations)` verbatim; this function's
    only job is to place it under a stable key, not to reshape it."""
    given = {
        "public.orders": {
            "is_ordinary_table": True,
            "indexes": [
                {
                    "name": "idx_status",
                    "columns": ["status"],
                    "is_partial": False,
                    "is_unique": False,
                }
            ],
        }
    }
    payload = advise_payload(
        [],
        Workload(stats=(), window_description="w"),
        Aggregation(usage=(), total_cost_ms=0.0, skipped_unqualifiable=0, tables=frozenset()),
        engine="postgres",
        redacted=True,
        degraded=[],
        physical_state=given,
    )
    assert payload["physical_state"] == given


def test_query_groups_is_always_present_and_empty_when_the_workload_is_empty():
    """Like `physical_state`, this key must never be omitted: an *absent* `query_groups` is
    how `verify` (a later task) tells a pre-this-feature artifact apart from one that
    genuinely analysed no query groups at all."""
    payload = advise_payload(
        [],
        Workload(stats=(), window_description="w"),
        Aggregation(usage=(), total_cost_ms=0.0, skipped_unqualifiable=0, tables=frozenset()),
        engine="postgres",
        redacted=True,
        degraded=[],
    )
    assert "query_groups" in payload
    assert payload["query_groups"] == []


def test_query_groups_includes_every_group_regardless_of_citation():
    """Task 6's fix round 3 (Critical): `query_groups` used to be scoped to digests some
    proposal's `fingerprint_digests` cited — the identical blind spot the sanctioned
    `physical_state` scoping fix closed one payload key over. A group `unreferenced` here
    stands in for the exact real-world shape that bug missed: a group whose proposal
    *resolved* between two `advise` runs (the recommended index now exists, so the rule
    stops firing and stops citing the group) looks identical to a group nothing ever
    proposed on — both are simply absent from every proposal's citations. `verify` needs
    both, so `query_groups` now includes every group in `workload.stats`, cited or not."""
    referenced = QueryStat(fingerprint="select 1", sql="select 1", calls=10, total_time_ms=100.0)
    unreferenced = QueryStat(fingerprint="select 2", sql="select 2", calls=10, total_time_ms=100.0)
    proposal = Proposal(
        code="ADV005",
        title="x",
        rationale="x",
        evidence={"fingerprint_digests": (fingerprint_id("select 1"),)},
        confidence=Confidence.HIGH,
    )
    payload = advise_payload(
        [proposal],
        Workload(stats=(referenced, unreferenced), window_description="w"),
        Aggregation(usage=(), total_cost_ms=0.0, skipped_unqualifiable=0, tables=frozenset()),
        engine="postgres",
        redacted=True,
        degraded=[],
    )
    digests = {g["digest"] for g in payload["query_groups"]}
    assert digests == {fingerprint_id("select 1"), fingerprint_id("select 2")}


def test_query_groups_reports_digest_calls_total_time_ms_and_mean_ms_per_field():
    digest = fingerprint_id("select 1")
    proposal = Proposal(
        code="ADV005",
        title="x",
        rationale="x",
        evidence={"fingerprint_digests": (digest,)},
        confidence=Confidence.HIGH,
    )
    payload = advise_payload(
        [proposal],
        Workload(
            stats=(
                QueryStat(fingerprint="select 1", sql="select 1", calls=50, total_time_ms=500.0),
            ),
            window_description="w",
        ),
        Aggregation(usage=(), total_cost_ms=0.0, skipped_unqualifiable=0, tables=frozenset()),
        engine="postgres",
        redacted=True,
        degraded=[],
    )
    [group] = payload["query_groups"]
    # Asserted per field, not as one dict comparison a single wrong field could still pass
    # against by accident.
    assert group["digest"] == digest
    assert group["calls"] == 50
    assert group["total_time_ms"] == 500.0
    assert group["mean_ms"] == 10.0


def test_mean_ms_is_null_rather_than_zero_when_a_group_has_no_calls():
    """0.0 reads as "instantaneous", which is the opposite of "unknown"."""
    digest = fingerprint_id("select 1")
    proposal = Proposal(
        code="ADV005",
        title="x",
        rationale="x",
        evidence={"fingerprint_digests": (digest,)},
        confidence=Confidence.HIGH,
    )
    payload = advise_payload(
        [proposal],
        Workload(
            stats=(QueryStat(fingerprint="select 1", sql="select 1", calls=0, total_time_ms=0.0),),
            window_description="w",
        ),
        Aggregation(usage=(), total_cost_ms=0.0, skipped_unqualifiable=0, tables=frozenset()),
        engine="postgres",
        redacted=True,
        degraded=[],
    )
    [group] = payload["query_groups"]
    assert group["calls"] == 0
    assert group["total_time_ms"] == 0.0
    assert group["mean_ms"] is None


def test_query_groups_has_no_dangling_references_across_several_rules_firing_at_once():
    """Every digest a proposal cites must resolve to an entry in `query_groups` — a
    dangling digest would make a proposal unverifiable by a later `verify` run — and,
    since Task 6's fix round 3, `query_groups` also carries every *uncited* group in the
    workload, not only what these proposals happen to cite. Exercised against several
    rules firing together (ADV001, ADV007, ADV005, ADV006), not one rule in isolation: a
    single-rule test cannot catch a rule that forgets to route its usage's fingerprints
    through the same workload `report.py` reads back."""
    orders = Relation("public", "orders")
    order_items = Relation("public", "order_items")
    wide_table = Relation("public", "wide_table")

    fp_index = 'SELECT "id" FROM "orders" WHERE "status" = %s'
    fp_join = 'SELECT "id" FROM "order_items" WHERE "order_id" = %s'
    fp_wildcard = 'SELECT "id" FROM "orders" WHERE "note" LIKE %s'
    fp_star = 'SELECT * FROM "wide_table"'
    fp_unreferenced = "SELECT 1"

    aggregation = Aggregation(
        usage=(
            ColumnUsage(
                relation=orders,
                column="status",
                role=ColumnRole.EQUALITY,
                calls=50,
                cost_ms=500.0,
                cost_share=0.5,
                fingerprint_ids=frozenset({fp_index}),
            ),
            ColumnUsage(
                relation=order_items,
                column="order_id",
                role=ColumnRole.JOIN,
                calls=50,
                cost_ms=400.0,
                cost_share=0.4,
                fingerprint_ids=frozenset({fp_join}),
            ),
        ),
        total_cost_ms=1000.0,
        skipped_unqualifiable=0,
        tables=frozenset({orders, order_items}),
    )
    facts = {
        orders: TableFacts(
            relation=orders, row_estimate=100_000, size_bytes=10**8, columns=("id", "status")
        ),
        order_items: TableFacts(
            relation=order_items,
            row_estimate=100_000,
            size_bytes=10**8,
            columns=("id", "order_id"),
        ),
        wide_table: TableFacts(
            relation=wide_table,
            row_estimate=100_000,
            size_bytes=10**8,
            columns=tuple(f"c{i}" for i in range(20)),
        ),
    }
    workload = Workload(
        stats=(
            QueryStat(
                fingerprint=fp_index,
                sql="select id from orders where status = $1",
                calls=50,
                total_time_ms=500.0,
            ),
            QueryStat(
                fingerprint=fp_join,
                sql="select id from order_items where order_id = $1",
                calls=50,
                total_time_ms=400.0,
            ),
            QueryStat(
                fingerprint=fp_wildcard,
                sql="select id from orders where note like $1",
                calls=10,
                total_time_ms=300.0,
                flags=frozenset({FLAG_LEADING_WILDCARD_LIKE}),
            ),
            QueryStat(
                fingerprint=fp_star,
                sql="select * from wide_table",
                calls=10,
                total_time_ms=300.0,
                flags=frozenset({FLAG_SELECT_STAR}),
            ),
            QueryStat(fingerprint=fp_unreferenced, sql="select 1", calls=1000, total_time_ms=1.0),
        ),
        window_description="w",
    )

    proposals = (
        propose_indexes(aggregation.usage, facts, {}, min_cost_share=0.01)
        + propose_join_keys(aggregation.usage, facts, {}, min_cost_share=0.01)
        + propose_sargability(aggregation.usage, workload, min_cost_share=0.01)
        + propose_select_star(workload, facts, min_cost_share=0.01, dialect="postgres")
    )
    codes = {p.code for p in proposals}
    assert codes == {"ADV001", "ADV007", "ADV005", "ADV006"}

    payload = advise_payload(
        proposals, workload, aggregation, engine="postgres", redacted=True, degraded=[]
    )
    group_digests = {g["digest"] for g in payload["query_groups"]}

    referenced: set[str] = set()
    for p in proposals:
        for digest in p.evidence.get("fingerprint_digests", ()):
            referenced.add(digest)
            assert digest in group_digests, f"{p.code} cites {digest}, absent from query_groups"

    # query_groups now covers every group workload.stats carries, cited or not (Task 6's
    # fix round 3) -- the uncited group must still be present, not excluded.
    assert fingerprint_id(fp_unreferenced) in group_digests
    assert group_digests == {fingerprint_id(stat.fingerprint) for stat in workload.stats}


def test_payload_is_json_serializable():
    import json

    json.dumps(_payload())


def test_payload_omits_the_dbt_key_when_absent():
    """Every existing caller omits `dbt`; the key must not appear at all — not even set to
    `None` — so a no-manifest payload stays byte-identical to what callers got before this
    key existed, rather than merely equal apart from one known extra key. A consumer that
    wants the value unconditionally can still do `payload.get("dbt")`, which behaves the
    same either way."""
    assert "dbt" not in _payload()


def test_payload_carries_the_dbt_disclosure_when_given():
    dbt = {"manifest": "/proj/target/manifest.json", "models": 3, "dropped_collisions": 1}
    payload = advise_payload(
        PROPOSALS,
        WORKLOAD,
        AGGREGATION,
        engine="postgres",
        redacted=True,
        degraded=[],
        dbt=dbt,
    )
    assert payload["dbt"] == dbt


def test_markdown_omits_the_dbt_section_when_absent():
    """The no-manifest markdown must not mention dbt at all — this is the same additive-by-
    construction constraint `advise` proves byte-for-byte against `main`, pinned here at the
    renderer's own level."""
    md = render_advise_markdown(
        PROPOSALS, WORKLOAD, AGGREGATION, engine="postgres", redacted=True, degraded=[]
    )
    assert "dbt" not in md.lower()


def test_markdown_renders_the_dbt_disclosure_when_given():
    dbt = {"manifest": "/proj/target/manifest.json", "models": 3, "dropped_collisions": 2}
    md = render_advise_markdown(
        PROPOSALS, WORKLOAD, AGGREGATION, engine="postgres", redacted=True, degraded=[], dbt=dbt
    )
    assert "## dbt enrichment" in md
    assert "/proj/target/manifest.json" in md
    assert "models indexed: 3" in md
    assert "cross-database collisions dropped: 2" in md


def test_markdown_omits_the_collision_line_when_there_were_none():
    """A truthy-count check, not `"dropped_collisions" in dbt`: the key is always present
    (see cli.advise), so a bare membership check would print "dropped: 0" on every run."""
    dbt = {"manifest": "/proj/target/manifest.json", "models": 1, "dropped_collisions": 0}
    md = render_advise_markdown(
        PROPOSALS, WORKLOAD, AGGREGATION, engine="postgres", redacted=True, degraded=[], dbt=dbt
    )
    assert "## dbt enrichment" in md
    assert "collisions dropped" not in md


def test_markdown_shows_confidence_and_cost_share():
    md = render_advise_markdown(
        PROPOSALS, WORKLOAD, AGGREGATION, engine="postgres", redacted=True, degraded=[]
    )
    assert "ADV001" in md
    assert "high" in md
    assert "42.0%" in md


def test_markdown_discloses_the_window_and_the_skips():
    """Asserts the phrases, not bare digits.

    A `"2" in md` style check passes on the window description alone ("2026-07-01" contains
    both a 2 and a 7), so it would not notice the counts being transposed or dropped.
    """
    md = render_advise_markdown(
        PROPOSALS, WORKLOAD, AGGREGATION, engine="postgres", redacted=True, degraded=[]
    )
    assert "since stats reset at 2026-07-01" in md
    assert "2 unparseable" in md
    # Not "introspection/DDL": the same counter also swallows DECLARE ... CURSOR FOR SELECT
    # and COPY (SELECT ...), which are ordinary reads. "filtered" claims only what is true.
    assert "7 filtered" in md
    assert "introspection/DDL" not in md
    assert "3 unresolvable" in md
    assert "4 ambiguous" in md


def _eight_groups_two_ambiguous():
    """Eight query groups of which two were dropped as ambiguous — six were understood."""
    workload = Workload(
        stats=tuple(
            QueryStat(fingerprint=f"fp{i}", sql="select 1", calls=1, total_time_ms=1.0)
            for i in range(8)
        ),
        window_description="since stats reset at 2026-07-01",
        skipped_unparseable=1,
        skipped_noise=2,
    )
    aggregation = Aggregation(
        usage=(),
        total_cost_ms=8.0,
        skipped_unqualifiable=0,
        tables=frozenset(),
        skipped_ambiguous=2,
    )
    return workload, aggregation


def test_all_three_surfaces_report_the_same_analyzed_count():
    """The terminal said "analyzed 6 of 8" while markdown said "analyzed: 8" and the JSON
    payload carried 8 under a key named `analyzed` — directly above its own "2 ambiguous".

    The README promises the terminal, markdown *and* JSON paths all print how many query
    groups were actually understood, and nothing pinned the number on two of the three: it
    was the sole mutation to survive a 47-mutation whole-branch sweep. All three are asserted
    here together, so fixing one surface and leaving another cannot pass.
    """
    from sqlquality.cli import _coverage_line

    workload, aggregation = _eight_groups_two_ambiguous()
    terminal = _coverage_line(workload, aggregation)
    md = render_advise_markdown(
        PROPOSALS, workload, aggregation, engine="postgres", redacted=True, degraded=[]
    )
    payload = advise_payload(
        PROPOSALS, workload, aggregation, engine="postgres", redacted=True, degraded=[]
    )
    assert "analyzed 6 of 8 query group(s)" in terminal
    assert "**Query groups analyzed:** 6 of 8" in md
    assert payload["analyzed"]["query_groups"] == 6
    # The window total is still available, just no longer labelled "analyzed".
    assert payload["analyzed"]["query_groups_in_window"] == 8


def test_markdown_escapes_an_evidence_key_as_well_as_its_value():
    """Keys reach the output too. Static today, but the asymmetry is worth closing."""
    hostile = [
        Proposal(
            code="ADV001",
            title="t",
            rationale="r",
            evidence={"we|ird\nkey": "v", "cost_share": 0.1},
            confidence=Confidence.LOW,
            ddl=None,
        ),
    ]
    md = render_advise_markdown(
        hostile, WORKLOAD, AGGREGATION, engine="postgres", redacted=True, degraded=[]
    )
    assert "we\\|ird" in md
    assert "we|ird" not in md


def test_gate_markdown_also_survives_a_newline_in_a_skip_reason():
    """`_md_escape` is shared with the gate report, so pin the behavior on that side too.

    Today every gate skip reason is a hardcoded literal, so nothing exercises the newline
    path there — which means a future reason built from an exception message could regress
    it with no test failing.
    """
    # `warned` is a computed property on GateReport, not a constructor field.
    report = GateReport(deltas=[], regressions=[], passed=True, mode="warn")
    md = render_markdown(report, skipped=[("model.demo.x", "line one\nline two")])
    for line in md.splitlines():
        assert not line.startswith("line two"), "a newline broke out of the skipped bullet"


def test_markdown_escapes_pipes_from_query_text():
    hostile = [
        Proposal(
            code="ADV005",
            title="a | b",
            rationale="r",
            evidence={"cost_share": 0.1},
            confidence=Confidence.LOW,
            ddl=None,
        ),
    ]
    md = render_advise_markdown(
        hostile, WORKLOAD, AGGREGATION, engine="postgres", redacted=True, degraded=[]
    )
    assert "a \\| b" in md


def test_markdown_states_when_no_proposals_were_produced():
    md = render_advise_markdown(
        [], WORKLOAD, AGGREGATION, engine="postgres", redacted=True, degraded=[]
    )
    assert "no proposals" in md.lower()


def test_markdown_neutralizes_newlines_in_a_hostile_title():
    # A schema identifier can legally contain a newline (see workload/postgres.py's
    # _comment_lines precedent). Left raw, it would split the table row in two —
    # the tail becomes a bare line no longer prefixed with "|" — and fake a second
    # heading in the detail section.
    hostile = [
        Proposal(
            code="ADV999",
            title="line1\nline2 | fake row | more",
            rationale="r1\nr2",
            evidence={"cost_share": 0.1},
            confidence=Confidence.LOW,
            ddl=None,
        ),
    ]
    md = render_advise_markdown(
        hostile, WORKLOAD, AGGREGATION, engine="postgres", redacted=True, degraded=[]
    )
    table_lines = [line for line in md.splitlines() if line.startswith("|")]
    # exactly the header, the separator, and this proposal's single row — no bare
    # tail line split off by an unescaped newline in the title.
    assert len(table_lines) == 3
    assert "line1\\nline2 \\| fake row \\| more" in table_lines[-1]
    heading_lines = [line for line in md.splitlines() if line.startswith("### ADV999")]
    assert len(heading_lines) == 1
    assert "line1\\nline2 \\| fake row \\| more" in heading_lines[0]


def test_markdown_ddl_fence_survives_embedded_triple_backticks():
    # ddl is built from live identifiers (see _quote_ident in workload/postgres.py,
    # which does not forbid backticks). A fixed ```sql fence would let an identifier
    # containing "```" close the fence early and inject a fake heading / fake code
    # block into the report.
    hostile_ddl = (
        'CREATE INDEX ON "orders" ("status");\n```\n\n# Fake Header\n\n```sql\nDROP TABLE users;'
    )
    hostile = [
        Proposal(
            code="ADV001",
            title="t",
            rationale="r",
            evidence={},
            confidence=Confidence.HIGH,
            ddl=hostile_ddl,
        ),
    ]
    md = render_advise_markdown(
        hostile, WORKLOAD, AGGREGATION, engine="postgres", redacted=True, degraded=[]
    )
    lines = md.splitlines()
    open_idx = next(i for i, line in enumerate(lines) if line.startswith("`") and "sql" in line)
    close_idx = next(
        i
        for i in range(len(lines) - 1, -1, -1)
        if lines[i].startswith("`") and "sql" not in lines[i]
    )
    opening, closing = lines[open_idx], lines[close_idx]
    assert opening[:-3] == closing  # same backtick run, minus the "sql" language tag
    assert len(closing) > 3  # wider than the embedded ``` run, so it isn't closed early
    assert "# Fake Header" in lines[open_idx + 1 : close_idx]


def test_markdown_does_not_render_a_stray_true_as_a_full_cost_share():
    """bool is an int subclass, so `isinstance(share, (int, float))` accepts True.

    render_ddl guards this explicitly and documents why; markdown and the terminal table
    did not, and rendered `cost_share=True` as "100.0% of workload cost" — the single most
    prominent number in the report, fabricated.
    """
    stray = [
        Proposal(
            code="ADV001",
            title="t",
            rationale="r",
            evidence={"cost_share": True},
            confidence=Confidence.HIGH,
            ddl=None,
        ),
    ]
    md = render_advise_markdown(
        stray, WORKLOAD, AGGREGATION, engine="postgres", redacted=True, degraded=[]
    )
    assert "100.0%" not in md
    assert "—" in md


def test_markdown_renders_evidence_the_way_json_does():
    """`str(("status",))` is a Python repr — the markdown reader sees `('status',)` where
    the JSON reader sees `["status"]` for the same run."""
    proposals = [
        Proposal(
            code="ADV001",
            title="t",
            rationale="r",
            evidence={"columns": ("status", "created_at"), "cost_share": 0.5},
            confidence=Confidence.HIGH,
            ddl=None,
        ),
    ]
    md = render_advise_markdown(
        proposals, WORKLOAD, AGGREGATION, engine="postgres", redacted=True, degraded=[]
    )
    assert "columns=['status', 'created_at']" in md


def test_markdown_suppresses_fingerprint_digests_but_keeps_the_human_count():
    """`fingerprint_digests` is a machine correlation key for `sqlquality verify` — a list
    of opaque 12-character hashes, one per backing query group, meaningless to a human
    reading a report. For a proposal backed by many query groups this would print a line
    full of hashes where the human-relevant number (`co_occurring_fingerprints`) is already
    right beside it. `--json` still carries the key; only `--markdown` must not."""
    digests = ("00b9a0c6bf02", "11c0b1d7c013", "22d1c2e8d124")
    proposals = [
        Proposal(
            code="ADV001",
            title="Add index on orders(status, created_at)",
            rationale="r",
            evidence={
                "co_occurring_fingerprints": len(digests),
                "fingerprint_digests": digests,
                "cost_share": 0.5,
            },
            confidence=Confidence.HIGH,
            ddl=None,
        ),
    ]
    md = render_advise_markdown(
        proposals, WORKLOAD, AGGREGATION, engine="postgres", redacted=True, degraded=[]
    )
    assert "fingerprint_digests" not in md
    for digest in digests:
        assert digest not in md
    # The suppression must not swallow the whole evidence line, nor the human-meaningful
    # count that sits beside the suppressed key.
    assert "co_occurring_fingerprints=3" in md
    assert "('status', 'created_at')" not in md
