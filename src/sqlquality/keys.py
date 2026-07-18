"""Redshift DISTKEY/SORTKEY inference from a model's JOIN/FILTER columns."""

from __future__ import annotations

from sqlglot import exp

from sqlquality.models import Finding, Severity
from sqlquality.sqlast import SqlParseError, parse


def _join_key_columns(tree: exp.Expression) -> list[str]:
    names: set[str] = set()
    for join in tree.find_all(exp.Join):
        on = join.args.get("on")
        if on is None:
            continue
        for eq in on.find_all(exp.EQ):
            left, right = eq.this, eq.expression
            if isinstance(left, exp.Column) and isinstance(right, exp.Column):
                names.add(left.name)
                names.add(right.name)
    return sorted(names)


def _filter_columns(tree: exp.Expression) -> list[str]:
    where = tree.find(exp.Where)
    if where is None:
        return []
    names: set[str] = set()
    for node in where.find_all((exp.EQ, exp.GT, exp.LT, exp.GTE, exp.LTE, exp.Between)):
        if isinstance(node, exp.Between):
            col = node.this
            low = node.args.get("low")
            high = node.args.get("high")
            if (
                isinstance(col, exp.Column)
                and isinstance(low, exp.Literal)
                and isinstance(high, exp.Literal)
            ):
                names.add(col.name)
        else:
            left, right = node.this, node.expression
            if isinstance(left, exp.Column) and isinstance(right, exp.Literal):
                names.add(left.name)
            elif isinstance(right, exp.Column) and isinstance(left, exp.Literal):
                names.add(right.name)
    return sorted(names)


def join_key_columns(sql: str, dialect: str) -> list[str]:
    return _join_key_columns(parse(sql, dialect))


def filter_columns(sql: str, dialect: str) -> list[str]:
    return _filter_columns(parse(sql, dialect))


def dist_sort_findings(sql: str, dialect: str) -> list[Finding]:
    """Suggest a DISTKEY (join keys) and SORTKEY (filter columns) for the model."""
    try:
        tree = parse(sql, dialect)
    except SqlParseError:
        return []
    findings: list[Finding] = []
    jk = _join_key_columns(tree)
    if jk:
        findings.append(
            Finding(
                "RS001",
                f"Consider a DISTKEY on the join key(s): {', '.join(jk)} — colocates joined rows and avoids redistribution.",
                0,
                Severity.INFO,
                False,
            )
        )
    fc = _filter_columns(tree)
    if fc:
        findings.append(
            Finding(
                "RS002",
                f"Consider a compound SORTKEY leading with the filter column(s): {', '.join(fc)} — enables zone-map block skipping.",
                0,
                Severity.INFO,
                False,
            )
        )
    return findings
