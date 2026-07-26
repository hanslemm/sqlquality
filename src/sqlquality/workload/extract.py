"""Resolve a query's columns to their tables and classify how each column is used."""

from __future__ import annotations

from sqlglot import exp
from sqlglot.errors import OptimizeError
from sqlglot.optimizer.qualify import qualify

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
    node = column.parent
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

    tables = tuple(qualified.find_all(exp.Table))
    alias_to_table = {t.alias_or_name: t.name for t in tables}
    # qualify() only qualifies columns inside SELECT scopes. In single-table DML
    # (`UPDATE orders ... WHERE created_at < $1`) the columns come back bare — verified:
    # column.table == '' — but the target table is unambiguous, so attribute them to it.
    # With more than one table in scope (UPDATE ... FROM) the attribution would be a
    # guess, so those columns are dropped rather than misattributed.
    sole_table = tables[0].name if len(tables) == 1 else None

    seen: set[tuple[str, str, ColumnRole]] = set()
    for column in qualified.find_all(exp.Column):
        table = alias_to_table.get(column.table) if column.table else sole_table
        if not table or not column.name:
            continue
        role = _role(column)
        if role is None:
            continue
        seen.add((table, column.name, role))
    return tuple(sorted(seen, key=lambda triple: (triple[0], triple[1], triple[2].value)))
