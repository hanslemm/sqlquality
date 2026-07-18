"""SQLGlot-backed parsing and structural-metric extraction."""

from __future__ import annotations

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from sqlquality.models import ComplexityMetrics


class SqlParseError(ValueError):
    """Raised when a SQL string cannot be parsed for the given dialect."""


def parse(sql: str, dialect: str) -> exp.Expression:
    """Parse one SQL statement into a SQLGlot AST, or raise SqlParseError."""
    try:
        tree = sqlglot.parse_one(sql, dialect=dialect)
    except ParseError as exc:
        raise SqlParseError(f"Could not parse SQL ({dialect}): {exc}") from exc
    if tree is None:
        raise SqlParseError("Empty SQL produced no AST")
    return tree


def analyze_sql(sql: str, dialect: str) -> ComplexityMetrics:
    """Extract raw structural complexity counts from one SQL statement."""
    tree = parse(sql, dialect)

    def count(node_type: type[exp.Expression]) -> int:
        return sum(1 for _ in tree.find_all(node_type))

    max_depth = 0
    for select in tree.find_all(exp.Select):
        depth = 1
        parent = select.parent
        while parent is not None:
            if isinstance(parent, exp.Select):
                depth += 1
            parent = parent.parent
        max_depth = max(max_depth, depth)

    top_select = tree if isinstance(tree, exp.Select) else tree.find(exp.Select)
    projected = len(top_select.expressions) if top_select is not None else 0

    return ComplexityMetrics(
        join_count=count(exp.Join),
        cte_count=count(exp.CTE),
        subquery_count=count(exp.Subquery),
        window_count=count(exp.Window),
        case_count=count(exp.Case),
        union_count=count(exp.SetOperation),  # Union + Except + Intersect
        distinct_count=count(exp.Distinct),
        select_count=count(exp.Select),
        max_select_depth=max_depth,
        projected_columns=projected,
    )
