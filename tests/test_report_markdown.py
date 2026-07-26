from sqlquality.delta import ModelDelta
from sqlquality.gate import GateReport
from sqlquality.models import Aggregation, Confidence, Proposal, QueryStat, Workload
from sqlquality.report import advise_payload, render_advise_markdown, render_markdown

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
    usage=(), total_cost_ms=500.0, skipped_unqualifiable=3, tables=frozenset({"orders"})
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
    assert payload["window"] == "since stats reset at 2026-07-01"
    assert payload["proposals"][0]["code"] == "ADV001"
    assert payload["skipped"] == {"unparseable": 2, "noise": 7, "unqualifiable": 3}
    assert payload["degraded"] == [{"capability": "ndv", "reason": "permission denied"}]


def test_payload_is_json_serializable():
    import json

    json.dumps(_payload())


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
    assert "7 introspection/DDL" in md
    assert "3 unresolvable" in md


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
