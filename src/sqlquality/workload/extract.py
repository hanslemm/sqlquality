"""Resolve a query's columns to their tables and classify how each column is used."""

from __future__ import annotations

from sqlglot import exp
from sqlglot.errors import OptimizeError, SchemaError
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import Scope, build_scope

from sqlquality.models import ColumnRole, Relation

#: Comparison nodes that an index can satisfy with an equality probe.
_EQUALITY_NODES = (exp.EQ, exp.In)
#: Comparison nodes that an index can satisfy with a range scan.
_RANGE_NODES = (exp.GT, exp.GTE, exp.LT, exp.LTE, exp.Between)
#: Wrapping a column in one of these defeats a plain B-tree index on that column.
_SARGABILITY_BREAKERS = (exp.Cast, exp.Func)


class UnqualifiableQuery(ValueError):
    """Raised when a query's columns cannot be resolved against the supplied schema."""


class AmbiguousRelation(UnqualifiableQuery):
    """A table name that two introspected schemas both hold, in a query that did not qualify it.

    A subclass, not a sibling: every caller that wants to treat all resolution failures
    alike keeps working with one `except UnqualifiableQuery`, while `aggregate` can count
    this case separately because its remedy is different — qualify the query or run once per
    schema, rather than widen the schema.
    """


def _is_negated(node: exp.Is) -> bool:
    """True when an ``IS`` predicate is the negated form, under either sqlglot encoding.

    sqlglot changed how it represents `IS NOT NULL` *within the version range this package
    declares* (`sqlglot>=30.12,<31`):

    * up to 30.12 it wraps the node — ``Not(Is(col, Null()))`` — so polarity lives on the parent
    * from 30.13 it sets a flag on the node itself — ``Is(col, Null(), negate=True)`` — and no
      ``exp.Not`` appears in the tree at all

    Reading only the parent silently inverted every `IS NOT NULL` into a `NULL_CHECK` on the
    newer parse, which is not a cosmetic misclassification: ADV004 turns these roles straight
    into a partial index's `WHERE` clause, so it emitted `WHERE col IS NULL` for a workload that
    filters `IS NOT NULL` — an index over exactly the wrong subset of rows, at MEDIUM confidence.
    The lockfile hid it, since `uv sync` pins 30.12 while a fresh `pip install sqlquality`
    resolves the newest 30.x.

    Both encodings are accepted rather than picking one and tightening the version floor: the
    flag is additive, so a tree built either way answers correctly, and users are not forced to
    a particular sqlglot to get correct DDL.
    """
    return bool(node.args.get("negate")) or isinstance(node.parent, exp.Not)


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
                return ColumnRole.NOT_NULL_CHECK if _is_negated(node) else ColumnRole.NULL_CHECK
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


def resolve_relation(table: exp.Table, schema: dict) -> Relation | None:
    """The schema-qualified relation for one `exp.Table`, or None if it is not attributable.

    `table.db` is authoritative when present, but only once it is checked against the
    introspected schema map. `qualify()` leaves it EMPTY for a bare table reference even when
    the nested schema resolves the name unambiguously — and bare references are the normal
    case, because production SQL relies on `search_path`. So the fallback is a lookup in the
    schema map we actually introspected:

    * exactly one introspected schema holds the name -> that is the schema, no guess involved
    * more than one -> ambiguous, and attributing it would be a coin flip. `qualify()` will
      normally have raised `SchemaError` before we get here, but a table whose columns are
      never referenced by name reaches this line, so the guard is real.
    * none -> the table lives outside the introspected schemas; the caller drops the column.

    The `table.db` branch is guarded the same way, and not just defensively: probing sqlglot
    30.12 confirms `qualify()` validates SELECT-scope column references against the schema
    (an explicitly-qualified table `qualify()` never introspected raises `OptimizeError`
    before this function runs), but it does **not** validate UPDATE/DELETE targets or their
    bare columns — `UPDATE other.orders SET status = 'x' WHERE id = 1` against a schema
    without an `other` key passes `qualify()` untouched. Trusting `table.db` unconditionally
    there would manufacture `Relation("other", "orders")`, a phantom that matches no catalog
    fact, from a schema-qualified DML statement we never introspected.
    """
    if table.db:
        if table.name in schema.get(table.db, {}):
            return Relation(schema=table.db, table=table.name)
        return None
    owners = [name for name, tables in schema.items() if table.name in tables]
    if len(owners) == 1:
        return Relation(schema=owners[0], table=table.name)
    return None


def _scope_relations(scope: Scope, schema: dict) -> dict[str, Relation]:
    """Alias (or bare name) -> schema-qualified relation, for one scope only.

    A sub-scope source (CTE, derived table) maps to a ``Scope``, not an ``exp.Table``.
    Columns resolving to one of those reference a projection rather than a base-table
    column, so they are omitted here and skipped — the sub-scope contributes its own base
    tables when ``traverse()`` reaches it. A source we cannot attribute to a schema is
    omitted for the same reason: no key, no usage.
    """
    resolved: dict[str, Relation] = {}
    for name, source in scope.sources.items():
        if isinstance(source, exp.Table):
            relation = resolve_relation(source, schema)
            if relation is not None:
                resolved[name] = relation
    return resolved


def _record(
    seen: set[tuple[Relation, str, ColumnRole]],
    relation: Relation | None,
    column: exp.Column,
) -> None:
    """Add one (relation, column, role) triple, skipping unattributable or unused columns."""
    if relation is None or not column.name:
        return
    role = _role(column)
    if role is None:
        return
    seen.add((relation, column.name, role))


def _collect_dml(
    qualified: exp.Expression, seen: set[tuple[Relation, str, ColumnRole]], schema: dict
) -> None:
    """Attribute the columns of an UPDATE/DELETE to its sole target table.

    ``qualify()`` leaves DML columns bare (``column.table == ''``) rather than raising.
    With exactly one table in the statement the target is unambiguous; with more than one
    (``UPDATE ... FROM``) attribution would be a guess, so bare columns are dropped
    instead of misattributed.

    A `len(tables) == 1` target is the one case that needs its own ambiguity check rather
    than deferring to `resolve_relation`'s plain `None`: `qualify()` does not validate
    UPDATE/DELETE targets (see `resolve_relation`'s docstring), so a bare name held by two
    introspected schemas reaches here with no `SchemaError` ever raised. Left as a silent
    `None`, the statement would vanish with no usage recorded and *neither* counter
    incremented — reported as analysed when it was not. Raising here routes it into the
    same `skipped_ambiguous` counter as the SELECT-path ambiguity sqlglot itself detects.
    """
    tables = tuple(qualified.find_all(exp.Table))
    aliases: dict[str, Relation] = {}
    for table in tables:
        relation = resolve_relation(table, schema)
        if relation is not None:
            aliases[table.alias_or_name] = relation
    sole: Relation | None = None
    if len(tables) == 1:
        target = tables[0]
        sole = resolve_relation(target, schema)
        if sole is None and not target.db:
            owners = [name for name, tbls in schema.items() if target.name in tbls]
            if len(owners) > 1:
                raise AmbiguousRelation(
                    f"Ambiguous mapping for DML target '{target.name}': "
                    f"held by {', '.join(sorted(owners))}."
                )
    for column in qualified.find_all(exp.Column):
        _record(seen, aliases.get(column.table) if column.table else sole, column)


def extract_usage(
    tree: exp.Expression, dialect: str, schema: dict
) -> tuple[tuple[Relation, str, ColumnRole], ...]:
    """(relation, column, role) triples for one query, deduplicated.

    ``schema`` is nested — ``{schema_name: {table: {column: type}}}`` — because relations
    are keyed by schema. Stars are not expanded: a projected star tells us nothing about
    which columns are filtered, and expanding it would drown the rollup in projection noise.

    ``SchemaError`` is *not* an ``OptimizeError`` subclass — its bases are ``SqlglotError``,
    ``Exception`` — so catching only ``OptimizeError`` let an ambiguous bare table name
    (`Ambiguous mapping for orders: sales, staging.`) escape `aggregate()` and abort the whole
    run with a traceback. It is now caught on its own and, when the message signals ambiguity
    specifically, re-raised as `AmbiguousRelation` rather than the plain `UnqualifiableQuery`
    every other resolution failure gets.
    """
    try:
        qualified = qualify(tree.copy(), dialect=dialect, schema=schema, expand_stars=False)
    except SchemaError as exc:
        # sqlglot has exactly one ambiguity message and no error code to match on, so the
        # text is the only signal available. Matching it loosely (lowercased substring)
        # rather than exactly, because a wording change upstream should degrade this to
        # "counted as unqualifiable" — the pre-existing behaviour — not crash.
        if "ambiguous mapping" in str(exc).lower():
            raise AmbiguousRelation(str(exc)) from exc
        raise UnqualifiableQuery(str(exc)) from exc
    except OptimizeError as exc:
        raise UnqualifiableQuery(str(exc)) from exc

    seen: set[tuple[Relation, str, ColumnRole]] = set()
    root = build_scope(qualified)
    if root is None:
        # build_scope() returns None for UPDATE/DELETE — they are not SELECT-rooted.
        _collect_dml(qualified, seen, schema)
    else:
        # Resolve aliases per scope, never with one flat map over the whole tree. Two
        # different tables in different scopes can share an alias, and a flat map keeps
        # whichever `find_all` visited last — silently attributing an outer filter to an
        # inner table and losing the outer one entirely.
        for scope in root.traverse():
            aliases = _scope_relations(scope, schema)
            for column in scope.columns:
                _record(seen, aliases.get(column.table), column)
    return tuple(sorted(seen, key=lambda triple: (triple[0], triple[1], triple[2].value)))
