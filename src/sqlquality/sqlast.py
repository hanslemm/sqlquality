"""SQLGlot-backed parsing and structural-metric extraction."""

from __future__ import annotations

import re
from typing import cast

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError, TokenError

from sqlquality.models import ComplexityMetrics

#: Identifier substituted for each ``{{ ... }}`` Jinja expression by :func:`strip_jinja`.
JINJA_PLACEHOLDER = "__sqlquality_jinja__"

_JINJA_COMMENT = re.compile(r"\{#.*?#\}", re.DOTALL)
_JINJA_STATEMENT = re.compile(r"\{%.*?%\}", re.DOTALL)
_JINJA_EXPRESSION = re.compile(r"\{\{.*?\}\}", re.DOTALL)
_FIRST_STATEMENT_KEYWORD = re.compile(r"\b(?:with|select)\b", re.IGNORECASE)
_SQL_LINE_COMMENT = re.compile(r"--[^\n]*")
_SQL_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)


class SqlParseError(ValueError):
    """Raised when a SQL string cannot be parsed for the given dialect."""


def parse(sql: str, dialect: str) -> exp.Expression:
    """Parse one SQL statement into a SQLGlot AST, or raise SqlParseError."""
    try:
        tree = sqlglot.parse_one(sql, dialect=dialect)
    except (ParseError, TokenError) as exc:
        raise SqlParseError(f"Could not parse SQL ({dialect}): {exc}") from exc
    if tree is None:
        raise SqlParseError("Empty SQL produced no AST")
    return cast(exp.Expression, tree)


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

    # EXISTS (SELECT ...) is a correlated subquery too, but sqlglot models it as an
    # exp.Exists holding an exp.Select directly (no exp.Subquery), so `WHERE EXISTS`
    # would otherwise score lower than the equivalent `WHERE ... IN (SELECT ...)`.
    # Guard against double-counting EXISTS((SELECT ...)), which produces both nodes.
    exists_subqueries = sum(
        1 for node in tree.find_all(exp.Exists) if not isinstance(node.this, exp.Subquery)
    )

    return ComplexityMetrics(
        join_count=count(exp.Join),
        cte_count=count(exp.CTE),
        subquery_count=count(exp.Subquery) + exists_subqueries,
        window_count=count(exp.Window),
        case_count=count(exp.Case),
        union_count=count(exp.SetOperation),  # Union + Except + Intersect
        distinct_count=count(exp.Distinct),
        select_count=count(exp.Select),
        max_select_depth=max_depth,
        projected_columns=projected,
    )


def strip_jinja(sql: str) -> str:
    """Best-effort removal of dbt/Jinja templating so a raw model roughly parses.

    The result is *approximate* — it is meant to make an uncompiled dbt model
    parseable for structural analysis, not to reproduce dbt's compiled SQL:

    * ``{# ... #}`` comment blocks are removed entirely (multi-line aware).
    * ``{% ... %}`` statement blocks are removed entirely (multi-line aware).
    * ``{{ ... }}`` expressions are replaced with the placeholder identifier
      :data:`JINJA_PLACEHOLDER`, so ``from {{ ref('stg') }}`` becomes a valid table.
    * Any statement-leading Jinja is dropped: everything before the first
      ``WITH``/``SELECT`` keyword that is only placeholders, whitespace, or
      comments is removed, so a model opening with ``{{ config(...) }}`` parses.
    """
    text = _JINJA_COMMENT.sub(" ", sql)
    text = _JINJA_STATEMENT.sub(" ", text)
    text = _JINJA_EXPRESSION.sub(JINJA_PLACEHOLDER, text)

    keyword = _FIRST_STATEMENT_KEYWORD.search(text)
    if keyword is not None:
        prefix = text[: keyword.start()]
        residue = prefix.replace(JINJA_PLACEHOLDER, " ")
        residue = _SQL_BLOCK_COMMENT.sub(" ", residue)
        residue = _SQL_LINE_COMMENT.sub(" ", residue)
        if not residue.strip():
            text = text[keyword.start() :]

    return text
