"""Redshift DISTKEY/SORTKEY inference from a model's JOIN/FILTER columns."""

from __future__ import annotations

from collections import Counter

from sqlglot import exp

from sqlquality.models import Finding, Severity
from sqlquality.sqlast import SqlParseError, parse


def _join_key_counts(tree: exp.Expression) -> Counter[str]:
    """Count each column's occurrences across all equi-join predicates."""
    counts: Counter[str] = Counter()
    for join in tree.find_all(exp.Join):
        on = join.args.get("on")
        if on is None:
            continue
        for eq in on.find_all(exp.EQ):
            left, right = eq.this, eq.expression
            if isinstance(left, exp.Column) and isinstance(right, exp.Column):
                counts[left.name] += 1
                counts[right.name] += 1
    return counts


def _join_key_columns(tree: exp.Expression) -> list[str]:
    return sorted(_join_key_counts(tree))


def _filter_columns(tree: exp.Expression) -> list[str]:
    where = tree.find(exp.Where)
    if where is None:
        return []
    names: set[str] = set()
    for node in where.find_all(exp.EQ, exp.GT, exp.LT, exp.GTE, exp.LTE, exp.Between, exp.In):
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
        elif isinstance(node, exp.In):
            col = node.this
            values = node.args.get("expressions") or []
            if (
                isinstance(col, exp.Column)
                and values
                and all(isinstance(value, exp.Literal) for value in values)
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
    counts = _join_key_counts(tree)
    if counts:
        # Redshift allows only ONE DISTKEY column: pick the most frequent equi-join
        # key (ties broken alphabetically); surface the rest as alternates.
        best = min(counts, key=lambda name: (-counts[name], name))
        alternates = sorted(name for name in counts if name != best)
        message = (
            f"Consider a single-column DISTKEY on the most frequent join key: {best} "
            "— colocates joined rows and avoids redistribution (Redshift permits only one DISTKEY)."
        )
        if alternates:
            message += f" Alternate candidate(s): {', '.join(alternates)}."
        findings.append(Finding("RS001", message, 0, Severity.INFO, False))
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
