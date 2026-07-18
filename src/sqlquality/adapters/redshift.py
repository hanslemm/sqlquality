"""Redshift performance adapter: anti-patterns + dist/sort inference + EXPLAIN markers."""

from __future__ import annotations

from sqlquality.adapters.base import PerfAdapter
from sqlquality.antipatterns import antipattern_findings
from sqlquality.keys import dist_sort_findings
from sqlquality.models import Finding, Severity

_MARKERS = [
    (
        "DS_BCAST_INNER",
        "RS010",
        "Broadcast of inner table (DS_BCAST_INNER) — tables not joined on their DISTKEYs.",
    ),
    (
        "DS_DIST_BOTH",
        "RS011",
        "Both sides redistributed (DS_DIST_BOTH) — the heaviest redistribution; align DISTKEYs on the join key.",
    ),
    (
        "DS_DIST_ALL_INNER",
        "RS012",
        "Serial execution (DS_DIST_ALL_INNER) — inner table sent to a single slice.",
    ),
    ("Nested Loop", "RS013", "Nested Loop join — usually a missing join condition / cross join."),
]


def parse_redshift_plan(explain_text: str) -> list[Finding]:
    """Findings from a Redshift EXPLAIN text plan (redistribution / join markers)."""
    findings: list[Finding] = []
    for marker, code, message in _MARKERS:
        if marker in explain_text:
            findings.append(Finding(code, message, 0, Severity.WARNING, False))
    return findings


class RedshiftAdapter(PerfAdapter):
    engine = "redshift"

    def static_findings(self, sql: str) -> list[Finding]:
        return antipattern_findings(sql, "redshift") + dist_sort_findings(sql, "redshift")

    def plan_findings(self, explain_text: str) -> list[Finding]:
        return parse_redshift_plan(explain_text)
