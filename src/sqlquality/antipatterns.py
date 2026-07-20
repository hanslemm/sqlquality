"""Dialect-agnostic static SQL anti-pattern detectors (SQLGlot)."""

from __future__ import annotations

from sqlglot import exp

from sqlquality.models import Finding, Severity
from sqlquality.sqlast import SqlParseError, parse


def _is_star_projection(projection: exp.Expression) -> bool:
    return isinstance(projection, exp.Star) or (
        isinstance(projection, exp.Column) and isinstance(projection.this, exp.Star)
    )


def _within_exists(node: exp.Expression) -> bool:
    """True if ``node`` sits inside an ``EXISTS (...)`` subquery."""
    parent = node.parent
    while parent is not None:
        if isinstance(parent, exp.Exists):
            return True
        parent = parent.parent
    return False


def _is_cte_closer(select: exp.Select, cte_names: set[str]) -> bool:
    """True for the idiomatic dbt closer ``select * from <cte>`` (sole star, no joins)."""
    if len(select.expressions) != 1 or not isinstance(select.expressions[0], exp.Star):
        return False
    if select.args.get("joins"):
        return False
    from_clause = select.args.get("from_")
    if from_clause is None:
        return False
    table = from_clause.this
    return isinstance(table, exp.Table) and table.name.lower() in cte_names


def _has_select_star(tree: exp.Expression) -> bool:
    cte_names = {cte.alias.lower() for cte in tree.find_all(exp.CTE)}
    for select in tree.find_all(exp.Select):
        if not any(_is_star_projection(p) for p in select.expressions):
            continue
        # (a) `EXISTS (SELECT * ...)` only probes for row existence — semantically free.
        if _within_exists(select):
            continue
        # (b) idiomatic dbt closer `select * from final` where `final` is a local CTE.
        if _is_cte_closer(select, cte_names):
            continue
        return True
    return False


def _is_constant_true(on: exp.Expression) -> bool:
    """True for a constant-true join condition: ``ON TRUE`` or ``ON 1=1``."""
    if isinstance(on, exp.Boolean) and on.this is True:
        return True
    if isinstance(on, exp.EQ):
        left, right = on.this, on.expression
        if isinstance(left, exp.Literal) and isinstance(right, exp.Literal) and left == right:
            return True
    return False


def _nearest_select(node: exp.Expression) -> exp.Select | None:
    parent = node.parent
    while parent is not None:
        if isinstance(parent, exp.Select):
            return parent
        parent = parent.parent
    return None


def _where_joins_relation(select: exp.Select, relation: str) -> bool:
    """True if ``select``'s own WHERE equates ``relation`` to a *different* relation.

    Scoped deliberately: only the WHERE of the join's owning SELECT is consulted, and
    only equalities whose nearest SELECT ancestor is that same SELECT count — so a
    predicate nested in a subquery/EXISTS cannot exonerate an outer comma join, and an
    outer predicate cannot exonerate a comma join buried in a CTE.
    """
    where = select.args.get("where")
    if where is None:
        return False
    for eq in where.find_all(exp.EQ):
        if _nearest_select(eq) is not select:
            continue
        left, right = eq.this, eq.expression
        if not (isinstance(left, exp.Column) and isinstance(right, exp.Column)):
            continue
        left_tbl, right_tbl = left.table, right.table
        if left_tbl == relation and right_tbl and right_tbl != relation:
            return True
        if right_tbl == relation and left_tbl and left_tbl != relation:
            return True
    return False


def _has_cartesian_join(tree: exp.Expression) -> bool:
    for join in tree.find_all(exp.Join):
        if (join.args.get("method") or "").upper() == "NATURAL":
            continue
        # LATERAL joins correlate through the lateral body, not an ON clause; the
        # `... ON TRUE` idiom is expected and must not be flagged.
        if isinstance(join.this, exp.Lateral):
            continue
        if join.args.get("using"):
            continue
        on = join.args.get("on")
        if on is not None:
            # False negative: a constant-true ON is a disguised cross join.
            if _is_constant_true(on):
                return True
            continue
        # Explicit CROSS JOIN is always flagged (behavior kept as-is).
        if (join.args.get("kind") or "").upper() == "CROSS":
            return True
        # Old-style comma join: not cartesian if this SELECT's own WHERE joins it.
        owner = join.parent
        relation = join.this.alias_or_name if isinstance(join.this, exp.Expression) else ""
        if isinstance(owner, exp.Select) and relation and _where_joins_relation(owner, relation):
            continue
        return True
    return False


def _has_leading_wildcard_like(tree: exp.Expression) -> bool:
    for node in tree.find_all(exp.Like, exp.ILike):
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
