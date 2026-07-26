"""Resolve a query's columns to their tables and classify how each column is used."""

from __future__ import annotations

from sqlglot import exp
from sqlglot.errors import OptimizeError
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import Scope, build_scope

from sqlquality.models import ColumnRole

#: Comparison nodes that an index can satisfy with an equality probe.
_EQUALITY_NODES = (exp.EQ, exp.In)
#: Comparison nodes that an index can satisfy with a range scan.
_RANGE_NODES = (exp.GT, exp.GTE, exp.LT, exp.LTE, exp.Between)
#: Wrapping a column in one of these defeats a plain B-tree index on that column.
_SARGABILITY_BREAKERS = (exp.Cast, exp.Func)


class UnqualifiableQuery(ValueError):
    """Raised when a query's columns cannot be resolved against the supplied schema."""


def _within(node: exp.Expression, *types: type[exp.Expression]) -> bool:
    """True if any ancestor of ``node`` is one of ``types``. Mirrors antipatterns._within_exists."""
    parent = node.parent
    while parent is not None:
        if isinstance(parent, types):
            return True
        parent = parent.parent
    return False


def _role(column: exp.Column) -> ColumnRole | None:
    """Classify how a column is used, or None if it is merely projected or assigned.

    A window's PARTITION BY / ORDER BY is not a query-level sort key, so anything under
    an exp.Window is discarded before the ancestor walk begins.
    """
    if _within(column, exp.Window):
        return None

    # Comparison and null-check roles only count inside a filtering clause. Without this
    # gate, `UPDATE orders SET status = $1` records `status` as an EQUALITY predicate —
    # it parses to EQ(status, $1) with no Where ancestor — and we would propose indexing
    # the column being written. The same gate correctly ignores a projected comparison
    # such as `SELECT a = b AS flag`.
    predicate_scope = _within(column, exp.Where, exp.Join, exp.Having)

    # A column wrapped in a cast or function inside a predicate cannot use a plain index.
    if isinstance(column.parent, _SARGABILITY_BREAKERS) and predicate_scope:
        return ColumnRole.NON_SARGABLE

    comparison: ColumnRole | None = None
    node: exp.Expr | None = column.parent
    while node is not None:
        if isinstance(node, exp.Is) and predicate_scope:
            null_side = isinstance(node.expression, exp.Null)
            if null_side:
                # `is not null` parses as Not(Is(...)), so polarity comes from the parent.
                return (
                    ColumnRole.NOT_NULL_CHECK
                    if isinstance(node.parent, exp.Not)
                    else ColumnRole.NULL_CHECK
                )
        if comparison is None and predicate_scope:
            if isinstance(node, _EQUALITY_NODES):
                comparison = ColumnRole.EQUALITY
            elif isinstance(node, _RANGE_NODES):
                comparison = ColumnRole.RANGE
        # A join predicate is a join key, not a filter — Join must win over the EQ below it.
        if isinstance(node, exp.Join):
            return ColumnRole.JOIN
        if isinstance(node, exp.Order):
            return ColumnRole.SORT
        if isinstance(node, exp.Group):
            return ColumnRole.GROUP
        if isinstance(node, exp.Select):
            break
        node = node.parent
    return comparison


def _scope_tables(scope: Scope) -> dict[str, str]:
    """Alias (or bare name) -> real table name, for one scope only.

    A sub-scope source (CTE, derived table) maps to a ``Scope``, not an ``exp.Table``.
    Columns resolving to one of those reference a projection rather than a base-table
    column, so they are omitted here and skipped — the sub-scope contributes its own base
    tables when ``traverse()`` reaches it.
    """
    return {
        name: source.name for name, source in scope.sources.items() if isinstance(source, exp.Table)
    }


def _record(seen: set[tuple[str, str, ColumnRole]], table: str | None, column: exp.Column) -> None:
    """Add one (table, column, role) triple, skipping unattributable or unused columns."""
    if not table or not column.name:
        return
    role = _role(column)
    if role is None:
        return
    seen.add((table, column.name, role))


def _collect_dml(qualified: exp.Expression, seen: set[tuple[str, str, ColumnRole]]) -> None:
    """Attribute the columns of an UPDATE/DELETE to its sole target table.

    ``qualify()`` leaves DML columns bare (``column.table == ''``) rather than raising.
    With exactly one table in the statement the target is unambiguous; with more than one
    (``UPDATE ... FROM``) attribution would be a guess, so bare columns are dropped
    instead of misattributed.
    """
    tables = tuple(qualified.find_all(exp.Table))
    aliases = {t.alias_or_name: t.name for t in tables}
    sole_table = tables[0].name if len(tables) == 1 else None
    for column in qualified.find_all(exp.Column):
        _record(seen, aliases.get(column.table) if column.table else sole_table, column)


def extract_usage(
    tree: exp.Expression, dialect: str, schema: dict
) -> tuple[tuple[str, str, ColumnRole], ...]:
    """(table, column, role) triples for one query, deduplicated.

    Stars are not expanded: a projected star tells us nothing about which columns are
    filtered, and expanding it would drown the rollup in projection noise.
    """
    try:
        qualified = qualify(tree.copy(), dialect=dialect, schema=schema, expand_stars=False)
    except OptimizeError as exc:
        raise UnqualifiableQuery(str(exc)) from exc

    seen: set[tuple[str, str, ColumnRole]] = set()
    root = build_scope(qualified)
    if root is None:
        # build_scope() returns None for UPDATE/DELETE — they are not SELECT-rooted.
        _collect_dml(qualified, seen)
    else:
        # Resolve aliases per scope, never with one flat map over the whole tree. Two
        # different tables in different scopes can share an alias, and a flat map keeps
        # whichever `find_all` visited last — silently attributing an outer filter to an
        # inner table and losing the outer one entirely.
        for scope in root.traverse():
            aliases = _scope_tables(scope)
            for column in scope.columns:
                _record(seen, aliases.get(column.table), column)
    return tuple(sorted(seen, key=lambda triple: (triple[0], triple[1], triple[2].value)))
