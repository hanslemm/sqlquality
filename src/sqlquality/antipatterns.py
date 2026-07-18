"""Dialect-agnostic static SQL anti-pattern detectors (SQLGlot)."""

from __future__ import annotations

from sqlglot import exp

from sqlquality.models import Finding, Severity
from sqlquality.sqlast import SqlParseError, parse


def _has_select_star(tree: exp.Expression) -> bool:
    for select in tree.find_all(exp.Select):
        for projection in select.expressions:
            if isinstance(projection, exp.Star):
                return True
            if isinstance(projection, exp.Column) and isinstance(projection.this, exp.Star):
                return True
    return False


def _has_cartesian_join(tree: exp.Expression) -> bool:
    for join in tree.find_all(exp.Join):
        if join.args.get("on") is not None or join.args.get("using"):
            continue
        if (join.args.get("method") or "").upper() == "NATURAL":
            continue
        return True
    return False


def _has_leading_wildcard_like(tree: exp.Expression) -> bool:
    for node in tree.find_all((exp.Like, exp.ILike)):
        pattern = node.args.get("expression")
        if isinstance(pattern, exp.Literal) and pattern.is_string and pattern.this.startswith("%"):
            return True
    return False


def antipattern_findings(sql: str, dialect: str) -> list[Finding]:
    """Static anti-pattern findings for one SQL statement."""
    try:
        tree = parse(sql, dialect)
    except SqlParseError as exc:
        return [Finding("SQ000", f"Unparseable SQL: {exc}", 0, Severity.ERROR, False)]

    findings: list[Finding] = []
    if _has_select_star(tree):
        findings.append(
            Finding(
                "SQ001",
                "SELECT * projects an unknown/wide column set; list columns explicitly.",
                0,
                Severity.WARNING,
                False,
            )
        )
    if _has_cartesian_join(tree):
        findings.append(
            Finding(
                "SQ002",
                "Cartesian/cross join without an ON/USING condition.",
                0,
                Severity.WARNING,
                False,
            )
        )
    if _has_leading_wildcard_like(tree):
        findings.append(
            Finding(
                "SQ003",
                "Leading-wildcard LIKE ('%...') is non-sargable and cannot use an index.",
                0,
                Severity.WARNING,
                False,
            )
        )
    return findings
