"""Postgres performance adapter: static anti-patterns + EXPLAIN-JSON parsing."""

from __future__ import annotations

from sqlquality.adapters.base import PerfAdapter
from sqlquality.antipatterns import antipattern_findings
from sqlquality.models import Finding, Severity


def parse_pg_plan(plan: object) -> list[Finding]:
    """Findings from Postgres EXPLAIN (FORMAT JSON) output."""
    findings: list[Finding] = []

    def walk(node: dict) -> None:
        if node.get("Node Type") == "Seq Scan":
            rel = node.get("Relation Name", "?")
            findings.append(
                Finding(
                    "PG001",
                    f"Seq Scan on {rel} — consider an index if the filter is selective.",
                    0,
                    Severity.WARNING,
                    False,
                )
            )
        if node.get("Sort Method") == "external merge Disk":
            findings.append(
                Finding(
                    "PG002",
                    "Sort spilled to disk (external merge) — raise work_mem or reduce the sorted set.",
                    0,
                    Severity.WARNING,
                    False,
                )
            )
        for child in node.get("Plans") or []:
            walk(child)

    items = plan if isinstance(plan, list) else [plan]
    for item in items:
        node = item.get("Plan", item) if isinstance(item, dict) else {}
        if node:
            walk(node)
    return findings


class PostgresAdapter(PerfAdapter):
    engine = "postgres"

    def static_findings(self, sql: str) -> list[Finding]:
        return antipattern_findings(sql, "postgres")

    def plan_findings(self, plan: object) -> list[Finding]:
        return parse_pg_plan(plan)
