# Advise Batch 2 Implementation Plan — schema-qualified keying, join/group rules, wrapper unwrapping

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `sqlquality advise` correct across multiple schemas, consume the two column
roles it already collects and throws away (JOIN, GROUP), and stop discarding the ordinary
reads that arrive wrapped in `DECLARE ... CURSOR FOR` or `COPY (...) TO`.

**Architecture:** Three phases in dependency order. Phase C replaces the bare-table-name
key with a `Relation(schema, table)` value type threaded through extract → aggregate →
facts → rules → report, which is the enabling model change. Phase A adds ADV007 (join-key
index) and ADV008 (group-by index) on top of that model, and restores the `_dedupe_by_ddl`
tie-break those rules make reachable again. Phase B is independent: a pre-parse unwrap step
in ingest.

**Tech Stack:** Python 3.11+, sqlglot 30.12 (`qualify`, `build_scope`), psycopg 3
(integration only), typer, rich, pytest.

## Global Constraints

Every task's requirements implicitly include this section.

- **All four CI gates must pass before every commit:** `uv run ruff check .`,
  `uv run ruff format --check .`, `uv run mypy src/sqlquality`, `uv run pytest -q`.
- **The default test suite must need no extras and no Docker.** `uv run pytest` after a
  plain `uv sync` must report `N passed, M deselected` — never `skipped`. Integration tests
  are marked `integration` and deselected by default.
- **sqlquality never executes user SQL.** `advise` opens a read-only session
  (`SET default_transaction_read_only = on`) with a statement timeout, runs only the
  statements in `PostgresWorkloadAdapter.SQL`, and writes DDL to a file for human review.
- **No credential, and no user literal, may reach stdout, stderr, a report, or an exception
  message.** Redaction happens at ingest; secrets are scrubbed via
  `workload/secrets.py`'s `secrets_for`/`scrub`.
- **Confidence must never overstate evidence.** If a check could not run — denied grant,
  unknown row count, unreadable index list — the proposal is emitted at LOW (or is
  suppressed) and the rationale names the check that was skipped. "Probably wrong" is not a
  confidence level.
- **Baseline is `main` at 442 default tests + 8 integration, all gates green.** Every task
  must leave the suite green; the count only goes up.
- **A test that passes with the production change reverted is not a test.** Every new test
  must be run against the un-fixed code, or against a deliberate mutation of the line it
  claims to pin, and observed to FAIL. Report the mutation you used.
- Public identifiers get a docstring saying *why*, matching the density of the surrounding
  module.

---

## Phase C — schema-qualified keying

### Task 1: `Relation`, and resolving it from the schema map rather than the AST

**Files:**
- Modify: `src/sqlquality/models.py` (add `Relation`; change `ColumnUsage.table` →
  `ColumnUsage.relation`)
- Modify: `src/sqlquality/workload/extract.py:84-149`
- Test: `tests/test_workload_extract.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `sqlquality.models.Relation` — `@dataclass(frozen=True, order=True)` with fields
    `schema: str`, `table: str`, and `__str__` returning `f"{schema}.{table}"`.
  - `sqlquality.models.ColumnUsage.relation: Relation` replacing `table: str`.
  - `sqlquality.workload.extract.resolve_relation(table: exp.Table, schema: dict) -> Relation | None`
  - `extract_usage(tree, dialect, schema) -> tuple[tuple[Relation, str, ColumnRole], ...]`
    — first tuple element is now a `Relation`, not a `str`.
  - `extract_usage` now also raises `UnqualifiableQuery` for `sqlglot.errors.SchemaError`.
  - The `schema` argument to `extract_usage` is now **nested**:
    `{schema_name: {table: {column: type}}}`.

**The constraint that decides this task.** `qualify()` does *not* populate `Table.db` for a
bare table reference, even when the nested schema resolves it unambiguously. Verified
against sqlglot 30.12:

```
schema={'public': {'orders': {'id': 'int', 'status': 'text'}}}
sql=select id from orders where status='x'
  -> SELECT "orders"."id" AS "id" FROM "orders" AS "orders" WHERE "orders"."status" = 'x'
     source alias='orders' name='orders' db=''        <-- db is EMPTY
```

Reading `table.db` and trusting it therefore keys almost every real workload under
`schema=""`, because production queries rely on `search_path` and say `from orders`, not
`from public.orders`. A phantom `Relation("", "orders")` matches no catalog fact, so every
table falls through the `facts.get(...)` lookup and **every proposal is silently
suppressed** — the same failure shape as the `reltuples = -1` bug. The schema must be
resolved from the schema map we introspected, with `table.db` used only when it is
non-empty.

Also verified: `SchemaError` is **not** a subclass of `OptimizeError`
(`SchemaError.__mro__` is `SchemaError → SqlglotError → Exception`). `extract_usage`
catches only `OptimizeError` today, so an ambiguous bare name
(`SchemaError: Ambiguous mapping for orders: sales, staging.`) propagates out of
`aggregate()` and crashes the whole run with a traceback. Multi-schema is what makes that
reachable, so it must be caught in the same task that makes it reachable.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_workload_extract.py`:

```python
import pytest
from sqlglot import exp

from sqlquality.models import ColumnRole, Relation
from sqlquality.sqlast import parse
from sqlquality.workload.extract import (
    UnqualifiableQuery,
    extract_usage,
    resolve_relation,
)

ONE_SCHEMA = {"public": {"orders": {"id": "int", "status": "text", "shipped_at": "timestamp"}}}
TWO_SCHEMAS = {
    "sales": {"orders": {"id": "int", "status": "text"}},
    "staging": {"items": {"sku": "text", "qty": "int"}},
}
COLLIDING = {
    "sales": {"orders": {"id": "int", "status": "text"}},
    "staging": {"orders": {"id": "int", "status": "text"}},
}


def test_bare_table_resolves_to_its_only_owning_schema():
    """The common case: production SQL relies on search_path and says `from orders`.

    qualify() leaves Table.db empty here, so a `table.db`-only implementation keys this
    under Relation("", "orders") and every catalog lookup misses.
    """
    tree = parse("select id from orders where status = 'x'", "postgres")
    usage = extract_usage(tree, "postgres", ONE_SCHEMA)
    assert {relation for relation, _c, _r in usage} == {Relation("public", "orders")}


def test_explicitly_qualified_table_uses_the_schema_it_names():
    tree = parse("select id from staging.items where qty > 1", "postgres")
    usage = extract_usage(tree, "postgres", TWO_SCHEMAS)
    assert {relation for relation, _c, _r in usage} == {Relation("staging", "items")}


def test_two_schemas_distinct_names_attribute_to_the_right_one():
    """A join across schemas must not collapse both sides onto one relation."""
    tree = parse(
        "select o.id, i.sku from orders o join items i on i.sku = o.status", "postgres"
    )
    usage = extract_usage(tree, "postgres", TWO_SCHEMAS)
    assert {relation for relation, _c, _r in usage} == {
        Relation("sales", "orders"),
        Relation("staging", "items"),
    }


def test_ambiguous_bare_name_is_unqualifiable_not_a_crash():
    """sqlglot raises SchemaError, which is NOT an OptimizeError subclass."""
    tree = parse("select id from orders where status = 'x'", "postgres")
    with pytest.raises(UnqualifiableQuery):
        extract_usage(tree, "postgres", COLLIDING)


def test_resolve_relation_prefers_an_explicit_db_over_the_map():
    table = exp.Table(this=exp.to_identifier("orders"), db=exp.to_identifier("sales"))
    assert resolve_relation(table, COLLIDING) == Relation("sales", "orders")


def test_resolve_relation_returns_none_when_ambiguous():
    """Two owners is not a guess we are entitled to make."""
    table = exp.Table(this=exp.to_identifier("orders"))
    assert resolve_relation(table, COLLIDING) is None


def test_resolve_relation_returns_none_for_a_table_outside_the_map():
    table = exp.Table(this=exp.to_identifier("nowhere"))
    assert resolve_relation(table, ONE_SCHEMA) is None


def test_dml_columns_attribute_to_the_qualified_target():
    tree = parse("update orders set status = 'y' where id = 1", "postgres")
    usage = extract_usage(tree, "postgres", ONE_SCHEMA)
    assert (Relation("public", "orders"), "id", ColumnRole.EQUALITY) in usage
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_workload_extract.py -x -q`

Expected: FAIL — `ImportError: cannot import name 'Relation'`.

- [ ] **Step 3: Add `Relation` to models.py**

Insert immediately above `ColumnRole` in `src/sqlquality/models.py`:

```python
@dataclass(frozen=True, order=True)
class Relation:
    """A schema-qualified relation — the key every catalog fact is stored under.

    Bare table names were the key until multi-schema support landed, and they aliased: two
    schemas each holding an `orders` merged into one entry, so the last catalog row won the
    row estimate while `qualify()` resolved columns against the union of both column sets.

    ``order=True`` because the rules sort their output for canonical, run-to-run stable
    report ordering, and a bare `sorted()` over relation keys has to work. Field order is
    (schema, table) so that ordering groups a schema's tables together.
    """

    schema: str
    table: str

    def __str__(self) -> str:
        """`schema.table` — how the relation appears in a proposal title or JSON key."""
        return f"{self.schema}.{self.table}"
```

- [ ] **Step 4: Change `ColumnUsage.table` to `ColumnUsage.relation`**

In `src/sqlquality/models.py`, replace the `table: str` field of `ColumnUsage` with:

```python
    relation: Relation
```

Leave every other field, and the whole `cost_share` docstring, exactly as it is. Do **not**
add a `table` compatibility property: this codebase has already paid for two fields
carrying one fact (see the `fingerprints` / `fingerprint_ids` note in the same class), and
a second spelling of the identity is how the two drift.

- [ ] **Step 5: Resolve relations in extract.py**

In `src/sqlquality/workload/extract.py`, add the `SchemaError` import and replace
`_scope_tables`, `_record` and `_collect_dml`:

```python
from sqlglot.errors import OptimizeError, SchemaError
```

```python
def resolve_relation(table: exp.Table, schema: dict) -> Relation | None:
    """The schema-qualified relation for one `exp.Table`, or None if it is not attributable.

    `table.db` is authoritative when present, but `qualify()` leaves it EMPTY for a bare
    table reference even when the nested schema resolves the name unambiguously — and bare
    references are the normal case, because production SQL relies on `search_path`. So the
    fallback is a lookup in the schema map we actually introspected:

    * exactly one introspected schema holds the name -> that is the schema, no guess involved
    * more than one -> ambiguous, and attributing it would be a coin flip. `qualify()` will
      normally have raised `SchemaError` before we get here, but a table whose columns are
      never referenced by name reaches this line, so the guard is real.
    * none -> the table lives outside the introspected schemas; the caller drops the column.
    """
    if table.db:
        return Relation(schema=table.db, table=table.name)
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
    """
    tables = tuple(qualified.find_all(exp.Table))
    aliases: dict[str, Relation] = {}
    for table in tables:
        relation = resolve_relation(table, schema)
        if relation is not None:
            aliases[table.alias_or_name] = relation
    sole = resolve_relation(tables[0], schema) if len(tables) == 1 else None
    for column in qualified.find_all(exp.Column):
        _record(seen, aliases.get(column.table) if column.table else sole, column)
```

Then update `extract_usage`'s body and signature docstring:

```python
def extract_usage(
    tree: exp.Expression, dialect: str, schema: dict
) -> tuple[tuple[Relation, str, ColumnRole], ...]:
    """(relation, column, role) triples for one query, deduplicated.

    ``schema`` is nested — ``{schema_name: {table: {column: type}}}`` — because relations
    are keyed by schema. Stars are not expanded: a projected star tells us nothing about
    which columns are filtered, and expanding it would drown the rollup in projection noise.

    ``SchemaError`` is caught alongside ``OptimizeError`` and re-raised as
    ``UnqualifiableQuery``. It is *not* an ``OptimizeError`` subclass — its bases are
    ``SqlglotError``, ``Exception`` — so catching only ``OptimizeError`` let an ambiguous
    bare table name (`Ambiguous mapping for orders: sales, staging.`) escape `aggregate()`
    and abort the whole run with a traceback.
    """
    try:
        qualified = qualify(tree.copy(), dialect=dialect, schema=schema, expand_stars=False)
    except (OptimizeError, SchemaError) as exc:
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
    return tuple(
        sorted(seen, key=lambda triple: (triple[0], triple[1], triple[2].value))
    )
```

Add `Relation` to the `sqlquality.models` import at the top of the module.

- [ ] **Step 6: Run the new tests**

Run: `uv run pytest tests/test_workload_extract.py -q`

Expected: PASS.

- [ ] **Step 7: Prove `test_bare_table_resolves_to_its_only_owning_schema` discriminates**

Temporarily change `resolve_relation` to `return Relation(schema=table.db, table=table.name)`
— i.e. the naive `table.db`-only implementation this task exists to prevent. Run the test
file. Expected: that test FAILS with
`{Relation(schema='', table='orders')} != {Relation(schema='public', table='orders')}`.
Restore the real implementation. Report the mutation and the failure in your report.

- [ ] **Step 8: Fix the rest of the suite's call sites mechanically**

`uv run pytest -q` will now fail in `tests/test_workload_aggregate.py`,
`tests/test_workload_rules.py`, `tests/test_workload_postgres.py` and
`tests/test_models.py`. Do **not** fix them yet — Tasks 2-4 own those layers. For this
task's commit it is enough that `tests/test_workload_extract.py` and
`tests/test_workload_fingerprint.py` pass. Note the failing count in your report so the
next task can confirm it shrinks.

If `mypy` reports errors in `aggregate.py` / `postgres.py` from the changed tuple type,
that is expected and Tasks 2-4 resolve it — say so in the report rather than papering over
it with `type: ignore`.

- [ ] **Step 9: Commit**

```bash
git add src/sqlquality/models.py src/sqlquality/workload/extract.py tests/test_workload_extract.py
git commit -m "feat(advise): key column usage by schema-qualified relation"
```

---

### Task 2: aggregate on relations, and count ambiguity separately

**Files:**
- Modify: `src/sqlquality/workload/aggregate.py`
- Modify: `src/sqlquality/models.py` (`Aggregation.tables`, `Aggregation.skipped_ambiguous`)
- Test: `tests/test_workload_aggregate.py`

**Interfaces:**
- Consumes: `Relation`, `extract_usage` returning `(Relation, str, ColumnRole)` triples,
  nested `schema` (Task 1).
- Produces:
  - `Aggregation.tables: frozenset[Relation]` (was `frozenset[str]`)
  - `Aggregation.skipped_ambiguous: int = 0` — statements dropped specifically because a
    table name was ambiguous across the introspected schemas.
  - `star_tables(workload, schema) -> frozenset[Relation]`
  - `aggregate(workload, schema, dialect)` unchanged in signature; `schema` is now nested.

**Why ambiguity gets its own counter.** `skipped_unqualifiable` already exists, and folding
ambiguity into it would be defensible — except the remedy is different. An unresolvable
statement means the schema is incomplete (fetch more, or grant more); an ambiguous one means
*this* run introspected two schemas holding the same table name and the query did not say
which. The fix is "qualify the query, or run `advise` once per schema", and the report has
to be able to say that. A single bucket cannot.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_workload_aggregate.py`:

```python
from sqlquality.models import ColumnRole, QueryStat, Relation, Workload
from sqlquality.workload.aggregate import aggregate, star_tables

ONE_SCHEMA = {"public": {"orders": {"id": "int", "status": "text"}}}
TWO_SCHEMAS = {
    "sales": {"orders": {"id": "int", "status": "text"}},
    "staging": {"items": {"sku": "text", "qty": "int"}},
}
COLLIDING = {
    "sales": {"orders": {"id": "int", "status": "text"}},
    "staging": {"orders": {"id": "int", "status": "text"}},
}


def _workload(*sql: str) -> Workload:
    return Workload(
        stats=tuple(
            QueryStat(fingerprint=f"fp{i}", sql=s, calls=1, total_time_ms=100.0)
            for i, s in enumerate(sql)
        ),
        window_description="test",
    )


def test_usage_is_keyed_by_relation():
    result = aggregate(_workload("select id from orders where status = 'x'"), ONE_SCHEMA, "postgres")
    assert {u.relation for u in result.usage} == {Relation("public", "orders")}
    assert result.tables == frozenset({Relation("public", "orders")})


def test_same_table_name_in_two_schemas_does_not_alias():
    """The bug multi-schema keying exists to fix: two relations, not one merged entry."""
    result = aggregate(
        _workload(
            "select id from sales.orders where status = 'x'",
            "select id from staging.orders where status = 'y'",
        ),
        COLLIDING,
        "postgres",
    )
    assert result.tables == frozenset(
        {Relation("sales", "orders"), Relation("staging", "orders")}
    )


def test_ambiguous_bare_name_is_counted_not_crashed():
    result = aggregate(_workload("select id from orders where status = 'x'"), COLLIDING, "postgres")
    assert result.skipped_ambiguous == 1
    assert result.usage == ()


def test_a_plain_parse_failure_is_not_counted_as_ambiguous():
    """The two counters must not both fire for the same statement."""
    result = aggregate(_workload("this is not sql at all"), ONE_SCHEMA, "postgres")
    assert result.skipped_ambiguous == 0
    assert result.skipped_unqualifiable == 1


def test_star_tables_returns_qualified_relations():
    workload = Workload(
        stats=(
            QueryStat(
                fingerprint="fp",
                sql="select * from items",
                calls=1,
                total_time_ms=1.0,
                flags=frozenset({"select_star"}),
            ),
        ),
        window_description="test",
    )
    assert star_tables(workload, TWO_SCHEMAS) == frozenset({Relation("staging", "items")})


def test_star_tables_skips_an_ambiguous_name():
    """Attributing a bare `select *` to one of two same-named tables would be a guess."""
    workload = Workload(
        stats=(
            QueryStat(
                fingerprint="fp",
                sql="select * from orders",
                calls=1,
                total_time_ms=1.0,
                flags=frozenset({"select_star"}),
            ),
        ),
        window_description="test",
    )
    assert star_tables(workload, COLLIDING) == frozenset()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_workload_aggregate.py -x -q`

Expected: FAIL — `AttributeError: 'ColumnUsage' object has no attribute 'relation'` or a
`TypeError` from the nested schema, depending on which test runs first.

- [ ] **Step 3: Add the ambiguity signal to extract.py**

`aggregate` must distinguish an ambiguity from any other resolution failure, and the only
place that knows is the raise site. Add a subclass to
`src/sqlquality/workload/extract.py`, right below `UnqualifiableQuery`:

```python
class AmbiguousRelation(UnqualifiableQuery):
    """A table name that two introspected schemas both hold, in a query that did not qualify it.

    A subclass, not a sibling: every caller that wants to treat all resolution failures
    alike keeps working with one `except UnqualifiableQuery`, while `aggregate` can count
    this case separately because its remedy is different — qualify the query or run once per
    schema, rather than widen the schema.
    """
```

and raise it from `extract_usage` when sqlglot reports ambiguity:

```python
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
```

- [ ] **Step 4: Rewrite the aggregate internals**

In `src/sqlquality/workload/aggregate.py`:

- `_Key` becomes `tuple[Relation, str, ColumnRole]`.
- Import `AmbiguousRelation` and `Relation`.
- `star_tables` iterates the nested schema and resolves each mentioned name to a single
  owning schema:

```python
def star_tables(workload: Workload, schema: dict) -> frozenset[Relation]:
    """Relations a `SELECT *` query group merely *mentions*, matched against ``schema``.

    A bare `select * from wide_t` filters nothing, so it contributes no column usage and
    the relation never appears in ``Aggregation.tables``. Introspecting only the relations
    that produced usage therefore left the star rule with no column counts to test — inert
    for precisely the workload it exists to catch. These names are unioned in before catalog
    facts are fetched.

    A name held by two introspected schemas is skipped rather than attributed to either or
    to both: over-reporting would put a wide-table warning on a table the query never
    touched. Consistent with `resolve_relation`, which declines the same guess.

    Deliberately *not* added to ``Aggregation.tables``: that set means "relations with
    recorded column usage" and feeds the unused-index rule's notion of a hot table.
    """
    found: set[Relation] = set()
    for stat in workload.stats:
        if FLAG_SELECT_STAR not in stat.flags:
            continue
        for table in _table_names(schema):
            if not mentions_table(table, stat.sql):
                continue
            owners = [name for name, tables in schema.items() if table in tables]
            if len(owners) == 1:
                found.add(Relation(schema=owners[0], table=table))
    return frozenset(found)


def _table_names(schema: dict) -> frozenset[str]:
    """Every bare table name in a nested schema map, deduplicated across schemas."""
    return frozenset(table for tables in schema.values() for table in tables)
```

- `aggregate` counts ambiguity separately and builds `ColumnUsage(relation=...)`:

```python
    skipped_unqualifiable = 0
    skipped_ambiguous = 0

    for stat in workload.stats:
        try:
            tree = parse(stat.sql, dialect)
            triples = extract_usage(tree, dialect, schema)
        except AmbiguousRelation:
            # Counted before the broader handler below, because AmbiguousRelation *is* an
            # UnqualifiableQuery — ordering these the other way round makes the specific
            # counter unreachable and the specific remedy unreportable.
            skipped_ambiguous += 1
            continue
        except (SqlParseError, UnqualifiableQuery):
            skipped_unqualifiable += 1
            continue
        for key in triples:
            calls[key] += stat.calls
            cost[key] += stat.total_time_ms
            contributors[key].add(stat.fingerprint)
            tables.add(key[0])
```

with the `ColumnUsage` construction using `relation=relation` and the sort key becoming
`key=lambda u: (-u.cost_ms, u.relation, u.column, u.role.value)` (`Relation` is
`order=True`, so it sorts directly), and the `Aggregation` gaining
`skipped_ambiguous=skipped_ambiguous`.

- [ ] **Step 5: Add the field to `Aggregation`**

In `src/sqlquality/models.py`:

```python
@dataclass(frozen=True)
class Aggregation:
    usage: tuple[ColumnUsage, ...]
    total_cost_ms: float
    skipped_unqualifiable: int
    tables: frozenset[Relation]
    #: Statements dropped because a bare table name is held by two introspected schemas.
    #: Separate from `skipped_unqualifiable` because the remedy differs: qualify the query
    #: or run once per schema, rather than widen the schema.
    skipped_ambiguous: int = 0
```

- [ ] **Step 6: Run the tests**

Run: `uv run pytest tests/test_workload_aggregate.py -q`

Expected: PASS.

- [ ] **Step 7: Prove the ordering of the two handlers matters**

Swap the `except AmbiguousRelation` clause below the `except (SqlParseError,
UnqualifiableQuery)` clause. Run `tests/test_workload_aggregate.py`. Expected:
`test_ambiguous_bare_name_is_counted_not_crashed` FAILS with
`assert 0 == 1`, because the subclass is swallowed by the base handler. Restore the order.
Report this.

- [ ] **Step 8: Commit**

```bash
git add src/sqlquality/models.py src/sqlquality/workload/aggregate.py \
        src/sqlquality/workload/extract.py tests/test_workload_aggregate.py
git commit -m "feat(advise): aggregate per relation, count schema ambiguity separately"
```

---

### Task 3: catalog facts and indexes keyed by relation

**Files:**
- Modify: `src/sqlquality/workload/base.py` (`fetch_table_facts` signature)
- Modify: `src/sqlquality/workload/postgres.py` (all six SQL statements, `fetch_schema`,
  `fetch_table_facts`, `fetch_indexes`)
- Modify: `src/sqlquality/models.py` (`TableFacts.name` → `TableFacts.relation`)
- Test: `tests/test_workload_postgres.py`

**Interfaces:**
- Consumes: `Relation` (Task 1), `Aggregation.tables: frozenset[Relation]` (Task 2).
- Produces:
  - `TableFacts.relation: Relation` replacing `name: str`.
  - `fetch_schema(schemas) -> dict` now nested: `{schema: {table: {column: type}}}`.
  - `fetch_table_facts(schemas, relations: frozenset[Relation]) -> dict[Relation, TableFacts]`
  - `fetch_indexes(schemas, relations: frozenset[Relation]) -> dict[Relation, tuple[PgIndex, ...]]`
  - Every `SQL` statement that returns a relation now returns its schema as the **first**
    column.

**The SQL change.** Each of `CAP_SCHEMA`, `CAP_TABLE_FACTS`, `CAP_NDV` and `CAP_INDEXES`
already filters on `= ANY(%s)` over the schema list but does not *return* the schema, so a
row from `sales.orders` and one from `staging.orders` are indistinguishable in the result
set. Add the schema to the select list and to the grouping key.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_workload_postgres.py`, following the existing `_FakeCursor`/querier
fixture style in that file (read it first and match it — do not invent a second harness):

```python
def test_fetch_schema_is_nested_by_schema():
    rows = {
        CAP_SCHEMA: [
            ("sales", "orders", "id", "integer"),
            ("sales", "orders", "status", "text"),
            ("staging", "orders", "id", "integer"),
        ]
    }
    adapter = PostgresWorkloadAdapter(querier=_canned(rows))
    assert adapter.fetch_schema(("sales", "staging")) == {
        "sales": {"orders": {"id": "integer", "status": "text"}},
        "staging": {"orders": {"id": "integer"}},
    }


def test_table_facts_do_not_alias_across_schemas():
    """Two same-named tables must keep their own row estimates."""
    rows = {
        CAP_SCHEMA: [("sales", "orders", "id", "integer"), ("staging", "orders", "id", "integer")],
        CAP_TABLE_FACTS: [("sales", "orders", 50_000, 1024), ("staging", "orders", 7, 64)],
        CAP_NDV: [],
    }
    adapter = PostgresWorkloadAdapter(querier=_canned(rows))
    facts = adapter.fetch_table_facts(
        ("sales", "staging"),
        frozenset({Relation("sales", "orders"), Relation("staging", "orders")}),
    )
    assert facts[Relation("sales", "orders")].row_estimate == 50_000
    assert facts[Relation("staging", "orders")].row_estimate == 7


def test_ndv_does_not_leak_between_same_named_tables():
    rows = {
        CAP_SCHEMA: [("sales", "orders", "id", "integer"), ("staging", "orders", "id", "integer")],
        CAP_TABLE_FACTS: [("sales", "orders", 50_000, 1024), ("staging", "orders", 50_000, 1024)],
        CAP_NDV: [("sales", "orders", "id", 5000.0), ("staging", "orders", "id", 3.0)],
    }
    adapter = PostgresWorkloadAdapter(querier=_canned(rows))
    facts = adapter.fetch_table_facts(
        ("sales", "staging"),
        frozenset({Relation("sales", "orders"), Relation("staging", "orders")}),
    )
    assert facts[Relation("sales", "orders")].ndv["id"] == 5000.0
    assert facts[Relation("staging", "orders")].ndv["id"] == 3.0


def test_indexes_do_not_alias_across_schemas():
    rows = {
        CAP_INDEXES: [
            ("sales", "orders", "idx_a", "id", 1, False, False, 0, 100, False, None, False, "..."),
            ("staging", "orders", "idx_b", "id", 1, False, False, 9, 200, False, None, False, "..."),
        ]
    }
    adapter = PostgresWorkloadAdapter(querier=_canned(rows))
    indexes = adapter.fetch_indexes(
        ("sales", "staging"),
        frozenset({Relation("sales", "orders"), Relation("staging", "orders")}),
    )
    assert [i.name for i in indexes[Relation("sales", "orders")]] == ["idx_a"]
    assert [i.name for i in indexes[Relation("staging", "orders")]] == ["idx_b"]
    assert indexes[Relation("staging", "orders")][0].scans == 9


def test_every_relation_returning_statement_selects_its_schema():
    """A statement that filters on schema but does not return it cannot be keyed by it.

    This is the whole defect class of this task: the rows come back indistinguishable and
    the last one silently wins.
    """
    for capability in (CAP_SCHEMA, CAP_TABLE_FACTS, CAP_NDV, CAP_INDEXES):
        sql = PostgresWorkloadAdapter.SQL[capability].lower()
        assert "nspname" in sql or "schemaname" in sql or "table_schema" in sql, capability
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_workload_postgres.py -x -q`

Expected: FAIL — the canned rows have one more element than the current unpacking expects
(`ValueError: too many values to unpack`).

- [ ] **Step 3: Change the SQL to return the schema**

In `PostgresWorkloadAdapter.SQL`, prepend the schema to each select list. `CAP_WORKLOAD`
and `CAP_STATS_RESET` are unchanged — neither returns a relation.

```python
        CAP_SCHEMA: """
            SELECT c.table_schema, c.table_name, c.column_name, c.data_type
            FROM information_schema.columns c
            WHERE c.table_schema = ANY(%s)
        """,
        CAP_TABLE_FACTS: """
            SELECT n.nspname, c.relname, c.reltuples::bigint, pg_total_relation_size(c.oid)
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE c.relkind = 'r' AND n.nspname = ANY(%s) AND c.relname = ANY(%s)
        """,
        CAP_NDV: """
            SELECT s.schemaname, s.tablename, s.attname, s.n_distinct
            FROM pg_stats s
            WHERE s.schemaname = ANY(%s) AND s.tablename = ANY(%s)
        """,
```

and for `CAP_INDEXES` add `n.nspname` first and extend the `ORDER BY`, keeping every
existing comment in the statement verbatim:

```python
        CAP_INDEXES: """
            SELECT n.nspname, t.relname, i.relname, a.attname, k.ordinality,
                   ix.indisunique, ix.indisprimary,
                   COALESCE(psui.idx_scan, 0), pg_relation_size(i.oid),
                   ix.indpred IS NOT NULL,
                   pg_get_expr(ix.indpred, ix.indrelid),
                   ix.indexprs IS NOT NULL,
                   pg_get_indexdef(ix.indexrelid)
            FROM pg_index ix
            JOIN pg_class i ON i.oid = ix.indexrelid
            JOIN pg_class t ON t.oid = ix.indrelid
            JOIN pg_namespace n ON n.oid = t.relnamespace
            JOIN LATERAL unnest(ix.indkey) WITH ORDINALITY AS k(attnum, ordinality) ON TRUE
            LEFT JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = k.attnum
            LEFT JOIN pg_stat_user_indexes psui ON psui.indexrelid = i.oid
            WHERE n.nspname = ANY(%s) AND t.relname = ANY(%s)
            ORDER BY n.nspname, t.relname, i.relname, k.ordinality
        """,
```

Note the `= ANY(%s)` **table** parameter stays a list of bare names: Postgres filters on
`relname`, and narrowing per-schema would need one statement per schema. Passing the union
of bare names over-fetches slightly (a same-named table in a schema we do not care about
comes back) and the relation key then simply has no consumer. Say this in a comment.

- [ ] **Step 4: Rework the fetch methods**

`TableFacts.name` → `TableFacts.relation: Relation` in models.py, with a docstring note
that the field is the schema-qualified key rather than a display name.

`fetch_schema` nests by schema:

```python
    def fetch_schema(self, schemas: tuple[str, ...]) -> dict:
        """Nested schema mapping for sqlglot qualify(): {schema: {table: {column: type}}}.

        Nested rather than flat because `qualify()` needs to be able to *tell* two
        same-named tables apart — a flat map resolves a column against the union of both
        column sets, which is how a filter on a column that exists in only one of them was
        silently accepted.
        """
        schema: dict[str, dict[str, dict[str, str]]] = {}
        for schema_name, table, column, data_type in self._schema_rows(schemas):
            schema.setdefault(str(schema_name), {}).setdefault(str(table), {})[str(column)] = str(
                data_type
            )
        return schema
```

`fetch_table_facts` takes and returns relations. Keep the negative-`n_distinct` logic and
its whole comment intact, changing only the key from `str(table)` to
`Relation(schema=str(schema_name), table=str(table))`. Same for `fetch_indexes`: the
`grouped` dict key becomes `tuple[Relation, str]` and the result
`dict[Relation, tuple[PgIndex, ...]]`.

Update the `fetch_table_facts` abstract signature in `base.py` to
`relations: frozenset[Relation]` and its docstring to say relations.

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_workload_postgres.py -q`

Expected: PASS. `tests/test_workload_rules.py` still fails — Task 4 owns it.

- [ ] **Step 6: Prove the aliasing tests discriminate**

Revert `fetch_table_facts`'s key to the bare `str(table)` while leaving everything else in
place (a `dict[str, ...]` keyed on bare name, looked up by `relation.table`). Run
`tests/test_workload_postgres.py`. Expected: `test_table_facts_do_not_alias_across_schemas`
FAILS — both relations report whichever row arrived last. Restore. Report the mutation.

- [ ] **Step 7: Commit**

```bash
git add src/sqlquality/models.py src/sqlquality/workload/base.py \
        src/sqlquality/workload/postgres.py tests/test_workload_postgres.py
git commit -m "feat(advise): key catalog facts and indexes by relation"
```

---

### Task 4: the six existing rules on relations, with schema-qualified DDL

**Files:**
- Modify: `src/sqlquality/workload/postgres.py` (`_by_table`, `_covered` call sites, all six
  `propose_*`, `propose`)
- Test: `tests/test_workload_rules.py`

**Interfaces:**
- Consumes: everything from Tasks 1-3.
- Produces:
  - `_by_relation(usage) -> dict[Relation, list[ColumnUsage]]` replacing `_by_table`.
  - Every rule takes `facts: Mapping[Relation, TableFacts]` and
    `existing: Mapping[Relation, Sequence[PgIndex]]`.
  - Every rule's `evidence` dict gains `"schema": relation.schema` and keeps
    `"table": relation.table` (the bare name, so existing JSON consumers still read the
    same value from the same key).
  - Rule titles render the relation as `schema.table`.
  - The module-level `schema: str = DEFAULT_SCHEMA` keyword argument is **removed** from
    every rule — each proposal now knows its own schema from its relation, so a
    single run-wide schema is no longer meaningful.

**Why `schema=` goes away.** `propose(...)` currently passes one `schema` to every rule and
`_qualified(schema, table)` stamps it onto the DDL. With more than one schema in play that
is wrong for every relation but one. Each rule now calls `_qualified(relation.schema,
relation.table)`. The `DEFAULT_SCHEMA` constant stays — `WorkloadAdapter.schemas` still
defaults to `("public",)`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_workload_rules.py` (match the existing fixture helpers in that file):

```python
def test_adv001_ddl_is_qualified_with_the_relations_own_schema():
    usage = (
        _usage(Relation("sales", "orders"), "status", ColumnRole.EQUALITY, cost_share=0.5),
    )
    facts = {Relation("sales", "orders"): _facts(Relation("sales", "orders"), rows=50_000)}
    proposals = propose_indexes(usage, facts, {}, min_cost_share=0.01)
    assert proposals[0].ddl == 'CREATE INDEX ON "sales"."orders" ("status");'
    assert proposals[0].evidence["schema"] == "sales"
    assert proposals[0].evidence["table"] == "orders"
    assert "sales.orders" in proposals[0].title


def test_two_same_named_relations_get_two_independent_proposals():
    """One proposal per relation, each stamped with its own schema."""
    usage = (
        _usage(Relation("sales", "orders"), "status", ColumnRole.EQUALITY, cost_share=0.5),
        _usage(Relation("staging", "orders"), "status", ColumnRole.EQUALITY, cost_share=0.5),
    )
    facts = {
        Relation("sales", "orders"): _facts(Relation("sales", "orders"), rows=50_000),
        Relation("staging", "orders"): _facts(Relation("staging", "orders"), rows=50_000),
    }
    ddls = {p.ddl for p in propose_indexes(usage, facts, {}, min_cost_share=0.01)}
    assert ddls == {
        'CREATE INDEX ON "sales"."orders" ("status");',
        'CREATE INDEX ON "staging"."orders" ("status");',
    }


def test_an_index_in_one_schema_does_not_cover_the_other_schemas_candidate():
    """The coverage check must not reach across schemas."""
    usage = (
        _usage(Relation("sales", "orders"), "status", ColumnRole.EQUALITY, cost_share=0.5),
        _usage(Relation("staging", "orders"), "status", ColumnRole.EQUALITY, cost_share=0.5),
    )
    facts = {
        Relation("sales", "orders"): _facts(Relation("sales", "orders"), rows=50_000),
        Relation("staging", "orders"): _facts(Relation("staging", "orders"), rows=50_000),
    }
    existing = {
        Relation("sales", "orders"): (
            PgIndex(name="idx_status", columns=("status",), is_unique=False,
                    is_primary=False, scans=1, size_bytes=1),
        )
    }
    proposals = propose_indexes(usage, facts, existing, min_cost_share=0.01)
    assert [p.evidence["schema"] for p in proposals] == ["staging"]


def test_adv002_drop_ddl_qualifies_the_index_with_its_relations_schema():
    existing = {
        Relation("staging", "orders"): (
            PgIndex(name="idx_cold", columns=("note",), is_unique=False,
                    is_primary=False, scans=0, size_bytes=1),
        )
    }
    proposals = propose_unused_indexes(
        existing, hot_tables=frozenset({Relation("staging", "orders")})
    )
    assert proposals[0].ddl == 'DROP INDEX "staging"."idx_cold";'
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_workload_rules.py -x -q`

Expected: FAIL — `TypeError` on `_usage(...)` taking a `Relation`, or a `KeyError`/`ddl`
mismatch showing `"public"` where `"sales"` is expected.

- [ ] **Step 3: Rework the rules**

Rename `_by_table` and re-key it:

```python
def _by_relation(usage: Sequence[ColumnUsage]) -> dict[Relation, list[ColumnUsage]]:
    grouped: dict[Relation, list[ColumnUsage]] = {}
    for item in usage:
        grouped.setdefault(item.relation, []).append(item)
    return grouped
```

Then, in each of `propose_indexes`, `propose_partial_indexes`, `propose_unused_indexes`,
`propose_redundant_indexes`, `propose_sargability`, `propose_select_star`:

- iterate `for relation, items in sorted(_by_relation(usage).items())` (`Relation` is
  `order=True`, so this sorts canonically without a key function);
- drop the `schema: str = DEFAULT_SCHEMA` keyword and build DDL with
  `_qualified(relation.schema, relation.table)`;
- change `facts.get(table)` to `facts.get(relation)` and `existing.get(table, ())` to
  `existing.get(relation, ())`;
- put `"schema": relation.schema` and `"table": relation.table` in `evidence`;
- render titles with `f"...on {relation}(...)"` — `Relation.__str__` gives `schema.table`.

`propose_select_star` needs care: `wide` is built from `facts.items()` and matched against
statement text with `mentions_table`. Keep matching on the **bare** name (that is what the
SQL says) but carry the relation through, so the evidence reports qualified names:

```python
    wide = {
        relation: fact for relation, fact in facts.items() if len(fact.columns) >= min_columns
    }
    ...
        touched = sorted(
            (relation for relation in wide if mentions_table(relation.table, stat.sql)),
        )
        ...
                    "tables": tuple(str(relation) for relation in touched),
                    "column_counts": {str(relation): len(facts[relation].columns) for relation in touched},
```

`propose_sargability`'s per-usage branch reports `item.relation`; its
leading-wildcard branch is statement-level and unchanged.

Finally, in `PostgresWorkloadAdapter.propose`, delete the `schema = self.schemas[0] ...`
line and the `schema=schema` arguments, and update the surrounding comment: it currently
explains that only one schema is introspected because the CLI rejects more, which stops
being true in Task 5.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_workload_rules.py -q`

Expected: PASS.

- [ ] **Step 5: Prove the cross-schema coverage test discriminates**

Change `_covered(columns, existing.get(relation, ()))` to
`_covered(columns, existing.get(Relation("sales", relation.table), ()))` — a deliberate
cross-schema lookup. Run `tests/test_workload_rules.py`. Expected:
`test_an_index_in_one_schema_does_not_cover_the_other_schemas_candidate` FAILS with
`[] != ["staging"]`. Restore. Report the mutation.

- [ ] **Step 6: Run the whole suite**

Run: `uv run pytest -q`

Expected: PASS, except `tests/test_advise_cli.py` and `tests/test_report*.py` which Task 5
owns. Report the remaining failure count.

- [ ] **Step 7: Commit**

```bash
git add src/sqlquality/workload/postgres.py tests/test_workload_rules.py
git commit -m "feat(advise): propose per relation, with schema-qualified DDL"
```

---

### Task 5: accept multiple `--schema`, and disclose ambiguity

**Files:**
- Modify: `src/sqlquality/cli.py` (`_validate_schemas`, `_coverage_line`,
  `_coverage_warning`, `_analyzed_count`, the `--schema` help text, the `advise` body)
- Modify: `src/sqlquality/report.py:149` (`"tables"` must be JSON-serializable)
- Modify: `README.md`
- Test: `tests/test_advise_cli.py`, `tests/test_report_markdown.py`

**Interfaces:**
- Consumes: everything from Tasks 1-4.
- Produces: `advise --schema a --schema b` runs; `_validate_schemas` deduplicates and
  returns `tuple[str, ...]` without rejecting; `Aggregation.skipped_ambiguous` surfaces in
  the coverage line, the coverage warning, the JSON payload and the markdown report.

- [ ] **Step 1: Write the failing tests**

```python
def test_two_schemas_are_accepted():
    result = runner.invoke(app, ["advise", "--schema", "sales", "--schema", "staging", "--dry-run"])
    assert result.exit_code == 0


def test_duplicate_schemas_are_deduplicated():
    assert _validate_schemas(["public", "public"]) == ("public",)


def test_schema_order_is_preserved():
    assert _validate_schemas(["b", "a"]) == ("b", "a")


def test_coverage_line_reports_ambiguous_separately():
    workload = _workload_with(stats=3, unparseable=1, noise=0)
    aggregation = _aggregation_with(skipped_unqualifiable=1, skipped_ambiguous=2)
    line = _coverage_line(workload, aggregation)
    assert "2 ambiguous" in line


def test_ambiguity_warning_names_the_remedy():
    workload = _workload_with(stats=1, unparseable=0, noise=0)
    aggregation = _aggregation_with(skipped_unqualifiable=0, skipped_ambiguous=4)
    warning = _ambiguity_warning(aggregation)
    assert warning is not None
    assert "--schema" in warning


def test_no_ambiguity_means_no_warning():
    """The warning must not fire on the single-schema path, which is every existing run."""
    assert _ambiguity_warning(_aggregation_with(skipped_unqualifiable=3, skipped_ambiguous=0)) is None


def test_payload_tables_are_qualified_strings():
    payload = advise_payload(
        [], _workload_with(stats=0, unparseable=0, noise=0),
        _aggregation_with(tables=frozenset({Relation("sales", "orders")})),
        engine="postgres", redacted=True, degraded=[],
    )
    assert payload["tables"] == ["sales.orders"]
    json.dumps(payload)  # must not raise
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_advise_cli.py -x -q`

Expected: FAIL — exit code 2 from `_validate_schemas` on the two-schema invocation, and
`ImportError`/`NameError` for `_ambiguity_warning`.

- [ ] **Step 3: Replace `_validate_schemas`**

```python
def _validate_schemas(values: list[str]) -> tuple[str, ...]:
    """Deduplicate `--schema` values, preserving the order they were given in.

    Multiple schemas used to be rejected because every catalog fact was keyed on the bare
    relation name, so two schemas each holding an `orders` aliased into one another. Facts,
    NDV maps, index lists and the `qualify()` schema are all keyed by `Relation` now, so the
    rejection is gone. What survives is a narrower caveat, surfaced by
    `_ambiguity_warning`: a query that says `from orders` when two introspected schemas both
    hold `orders` is genuinely ambiguous, and is counted and reported rather than guessed at.
    """
    return tuple(dict.fromkeys(values))
```

- [ ] **Step 4: Surface the new counter**

Add `f"{aggregation.skipped_ambiguous} ambiguous"` to `_coverage_line`, add
`aggregation.skipped_ambiguous` to `_coverage_warning`'s `unexplained` sum (an ambiguous
statement is one we tried and failed to use, exactly like an unresolvable one), and add:

```python
def _ambiguity_warning(aggregation: Aggregation) -> str | None:
    """A warning naming the remedy for schema-ambiguous statements, or None.

    Separate from `_coverage_warning`, which fires on a *fraction* and says "coverage is
    low". This fires on any occurrence at all, because the remedy is specific and
    actionable — and because a handful of ambiguous statements can be the hottest ones in
    the workload without moving the coverage fraction enough to trip a threshold.
    """
    if not aggregation.skipped_ambiguous:
        return None
    return (
        f"{aggregation.skipped_ambiguous} statement(s) named a table held by more than one "
        "of the introspected schemas without qualifying it, so they could not be attributed "
        "and were dropped. Qualify the table in the query, or run advise once per --schema."
    )
```

Call it in the `advise` body immediately after the `_coverage_warning` block, echoing to
`err`. Also add `skipped_ambiguous` to the `advise_payload` counts dict and to the markdown
report's coverage section, alongside the existing skip counts — the JSON and markdown
reports carry every other counter, and this is the one a multi-schema user most needs.

In `report.py:149`, change `"tables": sorted(aggregation.tables)` to
`"tables": sorted(str(relation) for relation in aggregation.tables)`. A `Relation` is not
JSON-serializable, so leaving it would make `--json` raise `TypeError` after the whole
analysis had already run.

- [ ] **Step 5: Update the `--schema` help and the README**

Help text: `"Schema to introspect. Repeat for several: --schema public --schema sales."`

In `README.md`, find the `advise` section's statement that only one schema is supported at
a time and replace it with the multi-schema behaviour plus the ambiguity caveat. Also check
the Limitations section for the same claim and narrow it.

- [ ] **Step 6: Run the whole suite**

Run: `uv run pytest -q` and all four gates.

Expected: PASS, 442 + the new tests.

- [ ] **Step 7: Prove the JSON test discriminates**

Revert `report.py:149` to `sorted(aggregation.tables)`. Run
`tests/test_report_markdown.py -k payload_tables`. Expected: FAIL with
`TypeError: Object of type Relation is not JSON serializable` from the `json.dumps` line.
Restore. Report the mutation.

- [ ] **Step 8: Commit**

```bash
git add src/sqlquality/cli.py src/sqlquality/report.py README.md \
        tests/test_advise_cli.py tests/test_report_markdown.py
git commit -m "feat(advise): accept multiple --schema, disclose ambiguous statements"
```

---

## Phase A — the two roles nothing consumed

Context for the implementer: `ColumnRole.JOIN` and `ColumnRole.GROUP` are classified in
`extract.py:73` and `extract.py:77`, cost-weighted in `aggregate`, included in every
`cost_share` denominator — and then read by **no rule at all**. Verified:
`grep -rn "ColumnRole.JOIN\|ColumnRole.GROUP" src/` matches only the two lines that produce
them. This phase spends them.

### Task 6: ADV007 — index the hot join key

**Files:**
- Modify: `src/sqlquality/workload/postgres.py`
- Test: `tests/test_workload_rules.py`

**Interfaces:**
- Consumes: `_by_relation`, `_covered`, `_is_prefix`, `MIN_ROWS_FOR_INDEX`, `SELECTIVE_NDV`,
  `_UNKNOWN_ROWS_NOTE`, `_qualified` (Tasks 1-4).
- Produces:
  `propose_join_keys(usage, facts, existing, *, min_cost_share, min_rows=MIN_ROWS_FOR_INDEX, have_index_data=True) -> list[Proposal]`
  emitting code `"ADV007"`, wired into `PostgresWorkloadAdapter.propose`.

**Why this is a separate rule and not a fourth role in ADV001.** ADV001's rationale is
"equality columns first so the range column can be scanned last" — the B-tree ordering
argument. A join key is not a filter predicate: it is probed once per outer row, and its
selectivity story is about the join's inner side, not about narrowing a scan. Folding JOIN
into ADV001's candidate list would make that rationale false for the resulting index while
leaving the text in place. Postgres also does not create an index on the *referencing* side
of a foreign key, so an unindexed hot join key is a common and genuinely costly gap.

Confidence, following the house rule that a check which could not run caps the claim:
- LOW when `rows is None` or `not have_index_data`;
- HIGH when the join column's NDV is `>= SELECTIVE_NDV` (a selective key really does make a
  nested-loop probe cheap);
- MEDIUM when NDV is unknown;
- LOW when NDV is below `SELECTIVE_NDV` — a low-cardinality join column is usually the
  wrong thing to index, and saying so at LOW is more useful than suppressing it.

- [ ] **Step 1: Write the failing tests**

```python
def test_adv007_proposes_an_index_on_an_unindexed_hot_join_key():
    relation = Relation("public", "order_items")
    usage = (_usage(relation, "order_id", ColumnRole.JOIN, cost_share=0.4, cost_ms=400.0),)
    facts = {relation: _facts(relation, rows=100_000, ndv={"order_id": 5000.0})}
    proposals = propose_join_keys(usage, facts, {}, min_cost_share=0.01)
    assert [p.code for p in proposals] == ["ADV007"]
    assert proposals[0].ddl == 'CREATE INDEX ON "public"."order_items" ("order_id");'
    assert proposals[0].confidence is Confidence.HIGH


def test_adv007_is_silent_when_an_index_already_leads_with_the_join_key():
    relation = Relation("public", "order_items")
    usage = (_usage(relation, "order_id", ColumnRole.JOIN, cost_share=0.4),)
    facts = {relation: _facts(relation, rows=100_000, ndv={"order_id": 5000.0})}
    existing = {
        relation: (
            PgIndex(name="idx_oi_order", columns=("order_id", "sku"), is_unique=False,
                    is_primary=False, scans=5, size_bytes=1),
        )
    }
    assert propose_join_keys(usage, facts, existing, min_cost_share=0.01) == []


def test_adv007_respects_the_small_table_floor():
    relation = Relation("public", "tiny")
    usage = (_usage(relation, "order_id", ColumnRole.JOIN, cost_share=0.9),)
    facts = {relation: _facts(relation, rows=10, ndv={"order_id": 5.0})}
    assert propose_join_keys(usage, facts, {}, min_cost_share=0.01) == []


def test_adv007_caps_at_low_when_the_index_list_could_not_be_read():
    relation = Relation("public", "order_items")
    usage = (_usage(relation, "order_id", ColumnRole.JOIN, cost_share=0.4),)
    facts = {relation: _facts(relation, rows=100_000, ndv={"order_id": 5000.0})}
    proposals = propose_join_keys(usage, facts, {}, min_cost_share=0.01, have_index_data=False)
    assert proposals[0].confidence is Confidence.LOW
    assert "could not be read" in proposals[0].rationale


def test_adv007_caps_at_low_and_discloses_an_unknown_row_count():
    relation = Relation("public", "order_items")
    usage = (_usage(relation, "order_id", ColumnRole.JOIN, cost_share=0.4),)
    facts = {relation: _facts(relation, rows=None, ndv={"order_id": 5000.0})}
    proposals = propose_join_keys(usage, facts, {}, min_cost_share=0.01)
    assert proposals[0].confidence is Confidence.LOW
    assert "small-table floor" in proposals[0].rationale


def test_adv007_is_low_for_a_low_cardinality_join_key():
    relation = Relation("public", "order_items")
    usage = (_usage(relation, "kind", ColumnRole.JOIN, cost_share=0.4),)
    facts = {relation: _facts(relation, rows=100_000, ndv={"kind": 3.0})}
    proposals = propose_join_keys(usage, facts, {}, min_cost_share=0.01)
    assert proposals[0].confidence is Confidence.LOW


def test_adv007_ignores_non_join_roles():
    """The rule must not re-propose what ADV001 already covers."""
    relation = Relation("public", "orders")
    usage = (_usage(relation, "status", ColumnRole.EQUALITY, cost_share=0.9),)
    facts = {relation: _facts(relation, rows=100_000)}
    assert propose_join_keys(usage, facts, {}, min_cost_share=0.01) == []


def test_adv007_reports_the_hottest_join_key_per_relation():
    relation = Relation("public", "order_items")
    usage = (
        _usage(relation, "order_id", ColumnRole.JOIN, cost_share=0.4, cost_ms=400.0),
        _usage(relation, "sku", ColumnRole.JOIN, cost_share=0.1, cost_ms=100.0),
    )
    facts = {relation: _facts(relation, rows=100_000)}
    proposals = propose_join_keys(usage, facts, {}, min_cost_share=0.01)
    assert [p.evidence["columns"] for p in proposals] == [("order_id",), ("sku",)]
```

Note the last test: one proposal per join column, ordered by cost descending. A composite
of two join keys is not proposed — two separate joins on the same table want two separate
indexes, and a composite serves only the leading one.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_workload_rules.py -k adv007 -q`

Expected: FAIL — `NameError: name 'propose_join_keys' is not defined`.

- [ ] **Step 3: Implement `propose_join_keys`**

Place it directly after `propose_indexes` in `postgres.py`:

```python
def propose_join_keys(
    usage: Sequence[ColumnUsage],
    facts: Mapping[Relation, TableFacts],
    existing: Mapping[Relation, Sequence[PgIndex]],
    *,
    min_cost_share: float,
    min_rows: int = MIN_ROWS_FOR_INDEX,
    have_index_data: bool = True,
) -> list[Proposal]:
    """ADV007 — a hot join key with no index leading with it.

    Deliberately one proposal per join column rather than a composite: two joins against the
    same table want two indexes, and a composite `(a, b)` serves only probes on `a`.

    Not folded into ADV001. That rule's rationale is the B-tree ordering argument —
    "equality columns first so the range column can be scanned last" — and a join key is not
    a filter predicate: it is probed once per outer row. Adding JOIN to ADV001's candidate
    list would have left that sentence in the report while making it false of the index it
    describes. Postgres does not index the referencing side of a foreign key either, so this
    gap is both common and expensive.
    """
    proposals: list[Proposal] = []
    for relation, items in sorted(_by_relation(usage).items()):
        table_facts = facts.get(relation)
        rows = table_facts.row_estimate if table_facts else None
        if rows is not None and rows < min_rows:
            continue
        ndv = table_facts.ndv if table_facts else {}
        joins = sorted(
            (i for i in items if i.role is ColumnRole.JOIN),
            key=lambda i: (-i.cost_ms, i.column),
        )
        for item in joins:
            if item.cost_share < min_cost_share:
                continue
            if _covered((item.column,), existing.get(relation, ())) is not None:
                continue
            column_ndv = ndv.get(item.column)
            if rows is None or not have_index_data:
                confidence = Confidence.LOW
            elif column_ndv is None:
                confidence = Confidence.MEDIUM
            elif column_ndv >= SELECTIVE_NDV:
                confidence = Confidence.HIGH
            else:
                confidence = Confidence.LOW

            rationale = (
                "This column carries the table's hottest join predicate. A join key is "
                "probed once per outer row, so without an index leading with it every probe "
                "is a scan."
            )
            if have_index_data:
                rationale += " No existing index leads with it."
            else:
                rationale += (
                    " The existing-index list could not be read, so whether an index "
                    "already leads with it is unknown — check before applying."
                )
            if rows is None:
                rationale += _UNKNOWN_ROWS_NOTE
            if column_ndv is not None and column_ndv < SELECTIVE_NDV:
                rationale += (
                    f" Only about {column_ndv:.0f} distinct values, so the index may not be "
                    "selective enough to be worth its write cost."
                )

            proposals.append(
                Proposal(
                    code="ADV007",
                    title=f"Add index on join key {relation}({item.column})",
                    rationale=rationale,
                    evidence={
                        "schema": relation.schema,
                        "table": relation.table,
                        "columns": (item.column,),
                        "roles": (item.role.value,),
                        "cost_share": item.cost_share,
                        "calls": item.calls,
                        "fingerprints": item.fingerprints,
                        "row_estimate": rows,
                        "leading_ndv": column_ndv,
                    },
                    confidence=confidence,
                    ddl=(
                        f"CREATE INDEX ON {_qualified(relation.schema, relation.table)} "
                        f"({_quote_ident(item.column)});"
                    ),
                )
            )
    return proposals
```

- [ ] **Step 4: Wire it into `propose`**

Add `*propose_join_keys(aggregation.usage, facts, existing, min_cost_share=min_cost_share,
have_index_data=have_index_data),` to the `proposals` list in
`PostgresWorkloadAdapter.propose`, directly after the `propose_indexes(...)` entry.

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_workload_rules.py -q` then `uv run pytest -q`

Expected: PASS.

- [ ] **Step 6: Prove the coverage test discriminates**

Delete the `if _covered(...) is not None: continue` guard. Run
`tests/test_workload_rules.py -k adv007`. Expected:
`test_adv007_is_silent_when_an_index_already_leads_with_the_join_key` FAILS with a
one-proposal list where `[]` was expected. Restore. Report the mutation.

- [ ] **Step 7: Update `--min-cost-share` help and README**

The `--min-cost-share` help enumerates the cost-weighted rules by code
(`ADV001, ADV004, ADV005, ADV006`). ADV007 is cost-weighted, so add it. Add ADV007 to the
README's rule table.

- [ ] **Step 8: Commit**

```bash
git add src/sqlquality/workload/postgres.py src/sqlquality/cli.py README.md \
        tests/test_workload_rules.py
git commit -m "feat(advise): ADV007 -- index the hot join key"
```

---

### Task 7: ADV008 — an index to serve a hot GROUP BY

**Files:**
- Modify: `src/sqlquality/workload/postgres.py`
- Test: `tests/test_workload_rules.py`

**Interfaces:**
- Consumes: `_by_relation`, `_covered`, `_first_co_occurring`-style fingerprint overlap,
  `MIN_ROWS_FOR_INDEX`, `_UNKNOWN_ROWS_NOTE`.
- Produces:
  `propose_grouping_indexes(usage, facts, existing, *, min_cost_share, min_rows=MIN_ROWS_FOR_INDEX, max_arity=MAX_INDEX_ARITY, have_index_data=True) -> list[Proposal]`
  emitting code `"ADV008"`, wired into `propose`.

**Confidence is capped at MEDIUM, always.** Whether Postgres uses an index for grouping
depends on the choice between `GroupAggregate` (needs sorted input, which the index
provides) and `HashAggregate` (does not), and that choice depends on `work_mem`, the number
of groups and the aggregate functions used — none of which `advise` can see. HIGH would be
a claim about the planner's decision, not about the catalog. So: MEDIUM when the row count
is known, LOW when it is not or the index list could not be read. Never HIGH. State this in
the docstring so a later reader does not "fix" the missing HIGH branch.

**Multiple grouping columns are proposed as one composite,** in the grouping order the
queries use — unlike ADV007. A `GROUP BY a, b` needs input sorted by `(a, b)`; two
single-column indexes serve it no better than one. Because a redacted fingerprint does not
preserve which position each column held, order the composite by cost descending with the
column name as tiebreak, and say in the rationale that the order is inferred from cost
rather than read from the query.

- [ ] **Step 1: Write the failing tests**

```python
def test_adv008_proposes_a_composite_index_for_a_hot_group_by():
    relation = Relation("public", "events")
    usage = (
        _usage(relation, "tenant_id", ColumnRole.GROUP, cost_share=0.5, cost_ms=500.0,
               fingerprint_ids=frozenset({"fp1"})),
        _usage(relation, "day", ColumnRole.GROUP, cost_share=0.5, cost_ms=400.0,
               fingerprint_ids=frozenset({"fp1"})),
    )
    facts = {relation: _facts(relation, rows=5_000_000)}
    proposals = propose_grouping_indexes(usage, facts, {}, min_cost_share=0.01)
    assert [p.code for p in proposals] == ["ADV008"]
    assert proposals[0].evidence["columns"] == ("tenant_id", "day")
    assert proposals[0].ddl == 'CREATE INDEX ON "public"."events" ("tenant_id", "day");'


def test_adv008_never_reaches_high_confidence():
    """Whether the planner picks GroupAggregate over HashAggregate is not visible to us."""
    relation = Relation("public", "events")
    usage = (
        _usage(relation, "tenant_id", ColumnRole.GROUP, cost_share=0.9, cost_ms=900.0,
               fingerprint_ids=frozenset({"fp1"})),
    )
    facts = {relation: _facts(relation, rows=5_000_000, ndv={"tenant_id": 100_000.0})}
    proposals = propose_grouping_indexes(usage, facts, {}, min_cost_share=0.01)
    assert proposals[0].confidence is Confidence.MEDIUM


def test_adv008_groups_only_columns_that_co_occur_in_one_query():
    """Two GROUP BYs in two different queries are not one composite index."""
    relation = Relation("public", "events")
    usage = (
        _usage(relation, "tenant_id", ColumnRole.GROUP, cost_share=0.5, cost_ms=500.0,
               fingerprint_ids=frozenset({"fp1"})),
        _usage(relation, "day", ColumnRole.GROUP, cost_share=0.5, cost_ms=400.0,
               fingerprint_ids=frozenset({"fp2"})),
    )
    facts = {relation: _facts(relation, rows=5_000_000)}
    proposals = propose_grouping_indexes(usage, facts, {}, min_cost_share=0.01)
    assert proposals[0].evidence["columns"] == ("tenant_id",)


def test_adv008_is_silent_when_an_index_already_leads_with_the_grouping_columns():
    relation = Relation("public", "events")
    usage = (
        _usage(relation, "tenant_id", ColumnRole.GROUP, cost_share=0.5, cost_ms=500.0,
               fingerprint_ids=frozenset({"fp1"})),
    )
    facts = {relation: _facts(relation, rows=5_000_000)}
    existing = {
        relation: (
            PgIndex(name="idx_events_tenant", columns=("tenant_id", "day"), is_unique=False,
                    is_primary=False, scans=3, size_bytes=1),
        )
    }
    assert propose_grouping_indexes(usage, facts, existing, min_cost_share=0.01) == []


def test_adv008_respects_max_arity():
    relation = Relation("public", "events")
    usage = tuple(
        _usage(relation, name, ColumnRole.GROUP, cost_share=0.5, cost_ms=500.0 - i,
               fingerprint_ids=frozenset({"fp1"}))
        for i, name in enumerate(["a", "b", "c", "d"])
    )
    facts = {relation: _facts(relation, rows=5_000_000)}
    proposals = propose_grouping_indexes(usage, facts, {}, min_cost_share=0.01)
    assert proposals[0].evidence["columns"] == ("a", "b", "c")


def test_adv008_discloses_that_the_column_order_is_inferred():
    relation = Relation("public", "events")
    usage = (
        _usage(relation, "tenant_id", ColumnRole.GROUP, cost_share=0.5, cost_ms=500.0,
               fingerprint_ids=frozenset({"fp1"})),
        _usage(relation, "day", ColumnRole.GROUP, cost_share=0.5, cost_ms=400.0,
               fingerprint_ids=frozenset({"fp1"})),
    )
    facts = {relation: _facts(relation, rows=5_000_000)}
    proposals = propose_grouping_indexes(usage, facts, {}, min_cost_share=0.01)
    assert "inferred" in proposals[0].rationale.lower()


def test_adv008_ignores_non_group_roles():
    relation = Relation("public", "orders")
    usage = (_usage(relation, "status", ColumnRole.EQUALITY, cost_share=0.9),)
    facts = {relation: _facts(relation, rows=5_000_000)}
    assert propose_grouping_indexes(usage, facts, {}, min_cost_share=0.01) == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_workload_rules.py -k adv008 -q`

Expected: FAIL — `NameError: name 'propose_grouping_indexes' is not defined`.

- [ ] **Step 3: Implement `propose_grouping_indexes`**

Place it after `propose_join_keys`. The co-occurrence rule: take the hottest GROUP column as
the seed, then extend the composite only with GROUP columns sharing at least one fingerprint
with the seed, up to `max_arity`.

```python
def propose_grouping_indexes(
    usage: Sequence[ColumnUsage],
    facts: Mapping[Relation, TableFacts],
    existing: Mapping[Relation, Sequence[PgIndex]],
    *,
    min_cost_share: float,
    min_rows: int = MIN_ROWS_FOR_INDEX,
    max_arity: int = MAX_INDEX_ARITY,
    have_index_data: bool = True,
) -> list[Proposal]:
    """ADV008 — an index that can feed a hot GROUP BY already sorted.

    Confidence is capped at MEDIUM and there is deliberately no HIGH branch. Whether
    Postgres uses such an index depends on its choice between `GroupAggregate` (which wants
    sorted input, and is what the index provides) and `HashAggregate` (which does not) — a
    decision driven by `work_mem`, the number of groups and the aggregates involved, none of
    which this tool can see. Claiming HIGH would be asserting something about the planner
    rather than about the catalog. Do not add a HIGH branch here for symmetry with ADV001.

    One composite rather than several single-column indexes, unlike ADV007: `GROUP BY a, b`
    wants input ordered by `(a, b)`, which two separate indexes cannot provide. The column
    *order* is inferred from cost, not read from the query — redaction and fingerprinting do
    not preserve each column's position in the GROUP BY clause — and the rationale says so,
    because getting the order wrong makes the index serve only its leading column.
    """
    proposals: list[Proposal] = []
    for relation, items in sorted(_by_relation(usage).items()):
        table_facts = facts.get(relation)
        rows = table_facts.row_estimate if table_facts else None
        if rows is not None and rows < min_rows:
            continue
        grouping = sorted(
            (i for i in items if i.role is ColumnRole.GROUP),
            key=lambda i: (-i.cost_ms, i.column),
        )
        if not grouping:
            continue
        seed = grouping[0]
        # Extend the composite only with columns some single query groups by *alongside* the
        # seed. Without this, two unrelated GROUP BYs on the same table are welded into one
        # composite index that serves neither beyond its leading column.
        chosen = [seed]
        for candidate in grouping[1:]:
            if len(chosen) >= max_arity:
                break
            if candidate.fingerprint_ids & seed.fingerprint_ids:
                chosen.append(candidate)

        cost_share = max(i.cost_share for i in chosen)
        if cost_share < min_cost_share:
            continue
        columns = tuple(i.column for i in chosen)
        if _covered(columns, existing.get(relation, ())) is not None:
            continue

        rationale = (
            "This grouping carries a hot share of workload cost. An index on these columns "
            "lets the planner read the rows already ordered and group them without a sort. "
            "The column order here is inferred from cost, not read from the query — "
            "redaction does not preserve each column's position in the GROUP BY — so check "
            "it against the actual grouping before applying, since a composite index only "
            "serves the grouping it leads with."
        )
        if not have_index_data:
            rationale += (
                " The existing-index list could not be read, so whether an index already "
                "leads with these columns is unknown."
            )
        if rows is None:
            rationale += _UNKNOWN_ROWS_NOTE

        proposals.append(
            Proposal(
                code="ADV008",
                title=f"Add index for GROUP BY on {relation}({', '.join(columns)})",
                rationale=rationale,
                evidence={
                    "schema": relation.schema,
                    "table": relation.table,
                    "columns": columns,
                    "roles": tuple(i.role.value for i in chosen),
                    "cost_share": cost_share,
                    "calls": max(i.calls for i in chosen),
                    "fingerprints": max(i.fingerprints for i in chosen),
                    "row_estimate": rows,
                },
                confidence=(
                    Confidence.LOW if rows is None or not have_index_data else Confidence.MEDIUM
                ),
                ddl=(
                    f"CREATE INDEX ON {_qualified(relation.schema, relation.table)} "
                    f"({', '.join(_quote_ident(c) for c in columns)});"
                ),
            )
        )
    return proposals
```

- [ ] **Step 4: Wire it into `propose`**

Add `*propose_grouping_indexes(aggregation.usage, facts, existing,
min_cost_share=min_cost_share, have_index_data=have_index_data),` after the
`propose_join_keys(...)` entry.

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_workload_rules.py -q` then `uv run pytest -q`

Expected: PASS.

- [ ] **Step 6: Prove the co-occurrence test discriminates**

Change the extension condition to unconditional (`chosen.append(candidate)` with no
fingerprint check). Run `tests/test_workload_rules.py -k adv008`. Expected:
`test_adv008_groups_only_columns_that_co_occur_in_one_query` FAILS with
`("tenant_id", "day") != ("tenant_id",)`. Restore. Report the mutation.

- [ ] **Step 7: Update help text, README**

Add ADV008 to `--min-cost-share`'s enumerated cost-weighted rules and to the README rule
table.

- [ ] **Step 8: Commit**

```bash
git add src/sqlquality/workload/postgres.py src/sqlquality/cli.py README.md \
        tests/test_workload_rules.py
git commit -m "feat(advise): ADV008 -- index to serve a hot GROUP BY"
```

---

### Task 8: restore the `_dedupe_by_ddl` tie-break the new rules make reachable

**Files:**
- Modify: `src/sqlquality/workload/postgres.py` (`_dedupe_by_ddl` and its docstring)
- Test: `tests/test_workload_postgres.py`

**Interfaces:**
- Consumes: ADV007 and ADV008 (Tasks 6-7).
- Produces: `_dedupe_by_ddl` with a deterministic tie-break; `_CODE_PREFERENCE` mapping.

**Why this task exists.** `_dedupe_by_ddl`'s docstring currently argues at length that a
tie-break is unnecessary and was deliberately deleted:

> That preference needs no tie-break rule to state it: the two codes cannot tie. ADV002 is
> hardcoded MEDIUM ... and ADV003 is hardcoded HIGH ... A tie-break that cannot be reached
> is worse than none.

That reasoning was sound when the only colliding pair was ADV002/ADV003. Tasks 6-7 break
it: ADV001, ADV007 and ADV008 all emit `CREATE INDEX ON <relation> (<columns>);`, and their
confidences overlap — ADV001 MEDIUM (NDV unknown) and ADV008 MEDIUM (rows known) can
produce byte-identical DDL at the same confidence. `best[proposal.ddl] is p` then keeps
whichever the list order happened to put first, which is stable today only because
`propose` hardcodes the call order. That is a coincidence, not a rule, and it decides which
rationale the operator reads.

- [ ] **Step 1: Write the failing test**

```python
def test_identical_ddl_at_equal_confidence_resolves_by_code_preference():
    """ADV001 and ADV008 can emit byte-identical DDL at the same confidence."""
    ddl = 'CREATE INDEX ON "public"."events" ("tenant_id");'
    adv008 = Proposal(code="ADV008", title="group", rationale="g",
                      evidence={"cost_share": 0.5}, confidence=Confidence.MEDIUM, ddl=ddl)
    adv001 = Proposal(code="ADV001", title="filter", rationale="f",
                      evidence={"cost_share": 0.5}, confidence=Confidence.MEDIUM, ddl=ddl)
    # Both orderings must pick the same winner, or list order is deciding.
    assert [p.code for p in PostgresWorkloadAdapter._dedupe_by_ddl([adv008, adv001])] == ["ADV001"]
    assert [p.code for p in PostgresWorkloadAdapter._dedupe_by_ddl([adv001, adv008])] == ["ADV001"]


def test_confidence_still_beats_code_preference():
    ddl = 'CREATE INDEX ON "public"."events" ("tenant_id");'
    adv001_low = Proposal(code="ADV001", title="filter", rationale="f",
                          evidence={"cost_share": 0.5}, confidence=Confidence.LOW, ddl=ddl)
    adv008_med = Proposal(code="ADV008", title="group", rationale="g",
                          evidence={"cost_share": 0.5}, confidence=Confidence.MEDIUM, ddl=ddl)
    assert [p.code for p in PostgresWorkloadAdapter._dedupe_by_ddl([adv001_low, adv008_med])] == [
        "ADV008"
    ]


def test_every_ddl_emitting_code_has_a_preference_rank():
    """A code missing from the map would raise KeyError mid-run, after all the analysis."""
    ddl_codes = {"ADV001", "ADV002", "ADV003", "ADV004", "ADV007", "ADV008"}
    assert ddl_codes <= set(PostgresWorkloadAdapter._CODE_PREFERENCE)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `uv run pytest tests/test_workload_postgres.py -k dedupe -q` plus the new names.

Expected: FAIL — `AttributeError: _CODE_PREFERENCE`, and the both-orderings assertion fails
because list order decides today.

- [ ] **Step 3: Add the tie-break**

```python
    #: Which rule's rationale to keep when two rules propose byte-identical DDL at equal
    #: confidence. Lower wins. The order is by how directly the evidence supports *this*
    #: index: a filter predicate (ADV001) is the most direct reason to build a B-tree, a
    #: join key (ADV007) next, and a grouping (ADV008) last, since whether the planner uses
    #: an index for grouping depends on choices this tool cannot see. The DROP rules are
    #: ranked below them so a CREATE never loses to a DROP that happens to render the same
    #: text — which it cannot today, but this map is the place that would have to change.
    _CODE_PREFERENCE = {
        "ADV001": 0,
        "ADV007": 1,
        "ADV004": 2,
        "ADV008": 3,
        "ADV003": 4,
        "ADV002": 5,
    }
```

and use it as the second element of the comparison, replacing the `is`-identity dance with
an explicit key so both the winner and the filter agree:

```python
    @classmethod
    def _dedupe_by_ddl(cls, proposals: list[Proposal]) -> list[Proposal]:
        """Collapse proposals that would run identical DDL, keeping the strongest evidence.

        Two rules can genuinely reach the same index from different evidence — a filter
        predicate, a join key and a grouping on the same column all render the same
        `CREATE INDEX` — and an unused index that is also a prefix of a wider one is flagged
        by both ADV002 and ADV003 as the same `DROP INDEX`. They do not contradict each
        other, but a reader should not have to notice they are the same object twice.

        Confidence decides first. When it ties, `_CODE_PREFERENCE` decides, because
        something has to and list order must not: the losing proposal's rationale is
        discarded, so "whichever `propose()` happened to append first" is not an acceptable
        answer to which explanation the operator reads.

        There was a window where no tie was reachable — ADV002 is hardcoded MEDIUM and
        ADV003 HIGH, the only colliding pair at the time — and the tie-break was removed as
        unreachable code. ADV007 and ADV008 made it reachable again: ADV001 at MEDIUM (NDV
        unknown) and ADV008 at MEDIUM (row count known) produce byte-identical DDL at equal
        confidence.
        """
        def rank(proposal: Proposal) -> tuple[int, int]:
            return (
                cls._CONFIDENCE_ORDER[proposal.confidence],
                cls._CODE_PREFERENCE.get(proposal.code, len(cls._CODE_PREFERENCE)),
            )

        best: dict[str, Proposal] = {}
        for proposal in proposals:
            if not proposal.ddl:
                continue
            incumbent = best.get(proposal.ddl)
            if incumbent is None or rank(proposal) < rank(incumbent):
                best[proposal.ddl] = proposal
        return [p for p in proposals if not p.ddl or best[p.ddl] is p]
```

`.get(..., len(...))` rather than `[...]`: an unranked future code sorts last instead of
raising `KeyError` after the whole analysis has run. The test above is what keeps the map
complete for the codes that exist.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_workload_postgres.py -q` then `uv run pytest -q`

Expected: PASS.

- [ ] **Step 5: Prove the both-orderings assertion discriminates**

Remove the `_CODE_PREFERENCE` element from `rank`, leaving only confidence. Run the test.
Expected: FAIL on the second assertion (`["ADV008"] != ["ADV001"]`) while the first still
passes — which is precisely the list-order dependence this task removes. Restore. Report
the mutation and note that a test asserting only one ordering would have passed against the
broken code.

- [ ] **Step 6: Commit**

```bash
git add src/sqlquality/workload/postgres.py tests/test_workload_postgres.py
git commit -m "fix(advise): break identical-DDL ties by rule, not by list order"
```

---

## Phase B — stop discarding wrapped reads

### Task 9: unwrap `DECLARE ... CURSOR FOR` and `COPY (...) TO`

**Files:**
- Modify: `src/sqlquality/workload/fingerprint.py`
- Modify: `src/sqlquality/cli.py` (the `_coverage_line` docstring, which documents this as a
  known wart)
- Test: `tests/test_workload_fingerprint.py`

**Interfaces:**
- Consumes: nothing from Phases C/A — this task is independent.
- Produces: `unwrap(sql: str) -> str` in `fingerprint.py`, applied in `ingest` **before**
  `is_noise`.

**What the engine actually hands us.** Verified against sqlglot 30.12:

| statement | parses as | notes |
|---|---|---|
| `DECLARE c CURSOR FOR SELECT ...` | `exp.Command` (`this='DECLARE'`) | "unsupported syntax", falls back to `Command`; the tail is a **string literal**, so the inner query is not in the AST |
| `COPY (SELECT ...) TO STDOUT` | `exp.Copy` (`this=Subquery`, `kind=False`) | inner SELECT **is** in the AST |
| `COPY orders TO STDOUT` | `exp.Copy` (`this=Table`, `kind=False`) | whole-table dump, no predicates |
| `COPY orders (id) FROM STDIN` | `exp.Copy` (`this=Schema`, `kind=True`) | a write |
| `FETCH 100 FROM c` | `exp.Command` | no query text at all |

So `DECLARE` needs text extraction and `COPY` can be done on the AST — but doing both on
the raw string, before parsing, keeps one code path and lets `is_noise` run on the unwrapped
text. `FETCH` and `CLOSE` stay noise: they carry no query.

Every psycopg2 server-side cursor (`cursor(name=...)`) emits `DECLARE`, so on a Django or
SQLAlchemy workload this is not an edge case — it can be the majority of the read traffic,
and today all of it is counted as "filtered" and thrown away.

- [ ] **Step 1: Write the failing tests**

```python
import pytest

from sqlquality.models import RawQueryRow, WorkloadFetch
from sqlquality.workload.fingerprint import ingest, is_noise, unwrap


@pytest.mark.parametrize(
    "sql,expected",
    [
        (
            "DECLARE c CURSOR FOR SELECT id FROM orders WHERE status = 'x'",
            "SELECT id FROM orders WHERE status = 'x'",
        ),
        (
            "DECLARE c CURSOR WITH HOLD FOR SELECT id FROM orders",
            "SELECT id FROM orders",
        ),
        (
            "DECLARE c NO SCROLL CURSOR FOR SELECT id FROM orders",
            "SELECT id FROM orders",
        ),
        (
            "DECLARE c BINARY INSENSITIVE SCROLL CURSOR WITH HOLD FOR SELECT a FROM t",
            "SELECT a FROM t",
        ),
        (
            'DECLARE "my cursor" CURSOR FOR SELECT a FROM t',
            "SELECT a FROM t",
        ),
        (
            "COPY (SELECT id FROM orders WHERE status = 'x') TO STDOUT",
            "SELECT id FROM orders WHERE status = 'x'",
        ),
        ("copy (select 1) to stdout", "select 1"),
    ],
)
def test_unwrap_recovers_the_inner_query(sql, expected):
    assert unwrap(sql) == expected


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT id FROM orders",
        "COPY orders TO STDOUT",
        "COPY orders (id, status) FROM STDIN",
        "FETCH 100 FROM c",
        "CLOSE c",
        "DECLARE c CURSOR FOR",
        "DECLARE",
    ],
)
def test_unwrap_leaves_everything_else_alone(sql):
    """Anything without a recoverable inner query is returned unchanged, not mangled."""
    assert unwrap(sql) == sql


def test_a_declared_cursor_is_analyzed_not_filtered():
    fetch = WorkloadFetch(
        rows=(
            RawQueryRow(
                sql="DECLARE c CURSOR FOR SELECT id FROM orders WHERE status = 'x'",
                calls=3,
                total_time_ms=300.0,
            ),
        ),
        window_description="w",
    )
    workload = ingest(fetch, "postgres")
    assert workload.skipped_noise == 0
    assert len(workload.stats) == 1
    assert "DECLARE" not in workload.stats[0].sql.upper()
    assert workload.stats[0].calls == 3


def test_a_copy_subquery_is_analyzed_not_filtered():
    fetch = WorkloadFetch(
        rows=(
            RawQueryRow(
                sql="COPY (SELECT id FROM orders WHERE status = 'x') TO STDOUT",
                calls=1,
                total_time_ms=10.0,
            ),
        ),
        window_description="w",
    )
    workload = ingest(fetch, "postgres")
    assert workload.skipped_noise == 0
    assert len(workload.stats) == 1


def test_a_declared_cursor_over_introspection_is_still_filtered():
    """Unwrapping must not become a way to smuggle our own catalog reads into the workload."""
    fetch = WorkloadFetch(
        rows=(
            RawQueryRow(
                sql="DECLARE c CURSOR FOR SELECT * FROM pg_stat_statements",
                calls=1,
                total_time_ms=1.0,
            ),
        ),
        window_description="w",
    )
    workload = ingest(fetch, "postgres")
    assert workload.skipped_noise == 1
    assert workload.stats == ()


def test_a_whole_table_copy_is_still_filtered():
    fetch = WorkloadFetch(
        rows=(RawQueryRow(sql="COPY orders TO STDOUT", calls=1, total_time_ms=1.0),),
        window_description="w",
    )
    assert ingest(fetch, "postgres").skipped_noise == 1


def test_the_unwrapped_query_is_still_redacted():
    """Redaction runs after unwrapping, so the inner literal must not survive."""
    fetch = WorkloadFetch(
        rows=(
            RawQueryRow(
                sql="DECLARE c CURSOR FOR SELECT id FROM orders WHERE email = 'a@b.test'",
                calls=1,
                total_time_ms=1.0,
            ),
        ),
        window_description="w",
    )
    workload = ingest(fetch, "postgres")
    assert "a@b.test" not in workload.stats[0].sql
    assert "a@b.test" not in workload.stats[0].fingerprint
```

That last test is the one that matters most: unwrapping introduces a **new path by which
raw user SQL reaches the pipeline**, and the redaction guarantee has to hold on it.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_workload_fingerprint.py -x -q`

Expected: FAIL — `ImportError: cannot import name 'unwrap'`.

- [ ] **Step 3: Implement `unwrap`**

Add to `fingerprint.py`, above `is_noise`:

```python
#: `DECLARE <name> [BINARY] [ASENSITIVE|INSENSITIVE] [[NO] SCROLL] CURSOR
#:  [WITH|WITHOUT HOLD] FOR <query>` — the full PostgreSQL grammar for the statement every
#: psycopg2 server-side cursor emits.
#:
#: Anchored on `CURSOR ... FOR` rather than on the first `FOR`, because a cursor name is an
#: identifier and a quoted one may contain the word: `DECLARE "for sale" CURSOR FOR ...`
#: would otherwise be cut at the wrong place and yield unparseable text. The name alternative
#: matches a quoted identifier (with doubled quotes escaped) before an unquoted one for the
#: same reason.
_DECLARE_CURSOR = re.compile(
    r"^\s*DECLARE\s+"
    r'(?:"(?:[^"]|"")*"|[A-Za-z_]\w*)\s+'
    r"(?:BINARY\s+)?"
    r"(?:ASENSITIVE\s+|INSENSITIVE\s+)?"
    r"(?:NO\s+SCROLL\s+|SCROLL\s+)?"
    r"CURSOR\s+"
    r"(?:WITH\s+HOLD\s+|WITHOUT\s+HOLD\s+)?"
    r"FOR\s+(?P<query>\S.*)$",
    re.IGNORECASE | re.DOTALL,
)

#: `COPY ( <query> ) TO ...` — the only COPY form carrying predicates worth analysing.
#: `COPY <table> TO` is a whole-relation dump with no predicates, and `COPY ... FROM` is a
#: write; both stay noise. The capture is greedy to the last `)` so a query containing
#: parentheses survives; the result is validated by the caller's parse, so a mis-cut yields
#: an unparseable count rather than a wrong analysis.
_COPY_QUERY = re.compile(
    r"^\s*COPY\s*\(\s*(?P<query>.*)\s*\)\s*TO\b",
    re.IGNORECASE | re.DOTALL,
)


def unwrap(sql: str) -> str:
    """The inner query of a cursor declaration or `COPY (...) TO`, else ``sql`` unchanged.

    `DECLARE ... CURSOR FOR SELECT ...` and `COPY (SELECT ...) TO ...` are ordinary reads
    with real predicates, but both begin with a keyword the noise filter drops — so on any
    workload using server-side cursors (every psycopg2 `cursor(name=...)`, which is what
    Django and SQLAlchemy emit for large result sets) the hottest reads were counted as
    "filtered" and thrown away.

    Text surgery rather than AST surgery, for a reason that is not a preference: sqlglot
    cannot parse `DECLARE` at all — it falls back to `exp.Command` and leaves the entire
    tail as a single string literal, so there is no inner tree to lift. `COPY` *does* parse
    (to `exp.Copy` with a `Subquery`), but doing both here keeps one code path and, more
    importantly, lets `is_noise` run on the *unwrapped* text — which is what stops a
    `DECLARE c CURSOR FOR SELECT * FROM pg_stat_statements` from smuggling our own
    introspection into the analysed workload.

    Returns the input unchanged when nothing matches. The caller parses the result, so a
    partial or malformed wrapper degrades to the pre-existing behaviour — counted
    unparseable or filtered — rather than producing a wrong analysis.
    """
    for pattern in (_DECLARE_CURSOR, _COPY_QUERY):
        match = pattern.match(sql)
        if match is not None:
            return match.group("query").strip()
    return sql
```

- [ ] **Step 4: Apply it in `ingest`**

Change the loop head so unwrapping happens before the noise test:

```python
    for row in fetch.rows:
        # Unwrap *before* the noise test, so a cursor declaration is judged on the query it
        # declares. Judging the wrapper drops the read; judging the inner query keeps a real
        # read and still filters an inner introspection query.
        sql = unwrap(row.sql)
        if is_noise(sql):
            skipped_noise += 1
            continue
        try:
            tree = parse(sql, dialect)
        except SqlParseError:
            skipped_unparseable += 1
            continue
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_workload_fingerprint.py -q` then
`uv run pytest tests/test_workload_redaction.py -q`

Expected: PASS. The redaction suite is called out separately because this task widens what
reaches the redactor.

- [ ] **Step 6: Prove the ordering of unwrap and is_noise matters**

Move the `sql = unwrap(row.sql)` line to *after* the `is_noise(row.sql)` check (testing the
raw string, unwrapping only what survives). Run `tests/test_workload_fingerprint.py`.
Expected: `test_a_declared_cursor_is_analyzed_not_filtered` FAILS with
`assert 1 == 0` on `skipped_noise`. Restore.

Then run the opposite mutation: delete the `_INTROSPECTION` half of `is_noise`'s return
(`return bool(_LEADING_NOISE.match(sql))`). Expected:
`test_a_declared_cursor_over_introspection_is_still_filtered` FAILS. Restore. Report both.

- [ ] **Step 7: Correct the `_coverage_line` docstring**

`src/sqlquality/cli.py`'s `_coverage_line` docstring currently documents this exact wart as
a live limitation:

> The filter is a statement-prefix match, so it also swallows `DECLARE cur CURSOR FOR
> SELECT ...` and `COPY (SELECT ...) TO STDOUT` — ordinary reads with real predicates ...

Rewrite that paragraph: those two forms are now unwrapped and analysed, and "filtered"
counts session control, DDL, maintenance, introspection, whole-table `COPY`, and cursor
statements that carry no query (`FETCH`, `CLOSE`). Leave the "N of M" reasoning intact.

- [ ] **Step 8: Commit**

```bash
git add src/sqlquality/workload/fingerprint.py src/sqlquality/cli.py \
        tests/test_workload_fingerprint.py
git commit -m "fix(advise): analyse cursor and COPY-subquery reads instead of filtering them"
```

---

### Task 10: prove all three phases against a live Postgres

**Files:**
- Modify: `tests/integration/conftest.py` (seed a second schema)
- Modify: `tests/integration/test_advise_live.py`
- Modify: `tests/integration/test_introspection_live.py`
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-07-26-advise-workload-analysis-design.md`

**Interfaces:**
- Consumes: everything above.
- Produces: integration coverage for multi-schema keying, ADV007/ADV008 and wrapper
  unwrapping; documentation matching shipped behaviour.

Batch 1's live suite found two production bugs the 400-test unit suite structurally could
not see (`reltuples = -1` suppressing every proposal; `redact_tree` dismembering `$N` and
silently dropping a whole query group). Both were shape-of-real-data problems. This task
exists because Batch 2 changes the shape of every catalog row it reads.

- [ ] **Step 1: Seed a second schema and a wrapped read**

In `tests/integration/conftest.py`'s `seeded` fixture, add alongside the existing setup —
read the fixture first and match its style:

```sql
CREATE SCHEMA IF NOT EXISTS staging;
CREATE TABLE IF NOT EXISTS staging.orders (
    id bigint, status text, tenant_id bigint, day date
);
INSERT INTO staging.orders
SELECT g, 'draft', g % 7, current_date FROM generate_series(1, 50000) g;
ANALYZE staging.orders;
```

so that `orders` exists in **both** `public` and `staging` with different statistics — the
exact collision the old `_validate_schemas` refused to allow.

Then drive workload that exercises the new paths, and `ANALYZE` both tables so
`reltuples` is not the `-1` sentinel:

```sql
-- a schema-qualified filter on each side, so both relations get their own usage
SELECT id FROM public.orders WHERE status = 'shipped';
SELECT id FROM staging.orders WHERE status = 'draft';
-- a join key with no index (ADV007)
SELECT o.id FROM public.orders o JOIN public.order_items i ON i.order_id = o.id;
-- a hot GROUP BY with no index (ADV008)
SELECT tenant_id, day, count(*) FROM staging.orders GROUP BY tenant_id, day;
-- a server-side cursor: filtered before Task 9, analysed after
DECLARE live_cur CURSOR FOR SELECT id FROM public.orders WHERE status = 'pending';
FETCH 10 FROM live_cur;
CLOSE live_cur;
```

- [ ] **Step 2: Write the failing live tests**

```python
@pytest.mark.integration
def test_two_same_named_tables_keep_their_own_row_estimates(seeded):
    """The aliasing bug, against real catalog rows rather than canned ones."""
    adapter = PostgresWorkloadAdapter()
    adapter.connect(seeded, timeout_s=30)
    adapter.schemas = ("public", "staging")
    facts = adapter.fetch_table_facts(
        ("public", "staging"),
        frozenset({Relation("public", "orders"), Relation("staging", "orders")}),
    )
    public_rows = facts[Relation("public", "orders")].row_estimate
    staging_rows = facts[Relation("staging", "orders")].row_estimate
    assert public_rows is not None and public_rows > 0
    assert staging_rows is not None and staging_rows > 0
    assert public_rows != staging_rows, "both relations reported the same estimate"


@pytest.mark.integration
def test_multi_schema_advise_run_produces_qualified_proposals(seeded, tmp_path):
    result = runner.invoke(
        app,
        ["advise", "--dsn", seeded_dsn(seeded), "--schema", "public", "--schema", "staging",
         "--json", "--min-cost-share", "0.0"],
    )
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    schemas = {p["evidence"].get("schema") for p in payload["proposals"]}
    assert "staging" in schemas
    # Every DDL statement names the schema it belongs to.
    for proposal in payload["proposals"]:
        if proposal["ddl"]:
            assert f'"{proposal["evidence"]["schema"]}".' in proposal["ddl"]


@pytest.mark.integration
def test_a_declared_cursor_reaches_the_analysis(seeded):
    """DECLARE is what psycopg2 server-side cursors emit; before Task 9 it was discarded."""
    adapter = PostgresWorkloadAdapter()
    adapter.connect(seeded, timeout_s=30)
    fetch = adapter.fetch_workload(None, 500)
    assert any(row.sql.upper().startswith("DECLARE") for row in fetch.rows), (
        "the seeded cursor never reached pg_stat_statements — fixture problem, not a bug"
    )
    workload = ingest(fetch, "postgres")
    assert not any(s.sql.upper().startswith("DECLARE") for s in workload.stats)
    assert any("pending" not in s.sql and "orders" in s.sql for s in workload.stats)


@pytest.mark.integration
def test_the_new_rules_fire_on_a_real_workload(seeded):
    codes = {p.code for p in _run_advise(seeded, schemas=("public", "staging"))}
    assert "ADV007" in codes or "ADV008" in codes, (
        f"neither join-key nor grouping rule fired; got {sorted(codes)}"
    )
```

The non-vacuity guard in the third test is deliberate: if the fixture's `DECLARE` never
lands in `pg_stat_statements` the test would otherwise pass while proving nothing, which is
how a hollow test survives. Follow the same pattern for any test you add here.

- [ ] **Step 3: Run the integration suite**

```bash
docker compose -f tests/integration/docker-compose.yml up -d
sleep 8
uv run pytest -m integration -q
docker compose -f tests/integration/docker-compose.yml down
```

Expected: PASS. If any test fails, the live behaviour is the source of truth — fix the
production code, not the assertion, and report what real Postgres did differently.

- [ ] **Step 4: Confirm the default suite still needs neither Docker nor extras**

```bash
docker compose -f tests/integration/docker-compose.yml down
uv run pytest -q
```

Expected: `N passed, M deselected` — **no skips**. If anything skips, the marker or the
guard is wrong.

- [ ] **Step 5: Update the docs**

- `README.md`: multi-schema `--schema` usage with the ambiguity caveat; ADV007 and ADV008 in
  the rule table; remove the "wrapped reads are filtered" limitation if it is stated there.
- The design spec's deviations section: record that bare-name-plus-schema-map resolution was
  chosen over reading `Table.db`, and why (`qualify()` leaves `db` empty for bare
  references, so `Table.db` alone keys everything under `schema=""`).
- Check the spec and README for any surviving claim that only one schema is supported.

- [ ] **Step 6: Run all four gates**

```bash
uv run ruff check . && uv run ruff format --check . && \
  uv run mypy src/sqlquality && uv run pytest -q
```

- [ ] **Step 7: Commit**

```bash
git add tests/integration README.md docs/superpowers/specs/2026-07-26-advise-workload-analysis-design.md
git commit -m "test(integration): multi-schema, join/group rules and cursor reads against real postgres"
```

---

## Self-Review

**Spec coverage.** The three items the user approved for Batch 2:

| Item | Tasks |
|---|---|
| join-key and grouping-column proposals | 6 (ADV007), 7 (ADV008), 8 (the tie-break they make reachable) |
| `DECLARE`/`COPY` unwrapping | 9 |
| multi-schema `(schema, table)` keying | 1, 2, 3, 4, 5 |
| proof against real data + docs | 10 |

**Known ripple this plan accepts deliberately.** Task 1 leaves the suite red for three test
modules that Tasks 2-4 then fix. The alternative — one giant task threading `Relation`
through every layer at once — is not reviewable. Each task's own tests pass at its own
commit, and Task 1's step 8 says so explicitly so an implementer does not try to fix
`test_workload_rules.py` out of turn.

**Ordering rationale.** Phase C is first because Phases A and B would otherwise be written
against the bare-name model and rewritten immediately. Phase B is genuinely independent and
could run at any point; it is last because it is the smallest.

**Type consistency.** `Relation` is introduced in Task 1 and used with the same field names
(`schema`, `table`) and the same `__str__` contract in Tasks 2-10. `TableFacts.name` →
`TableFacts.relation` happens once, in Task 3, which is also where every `fetch_*` return
type changes. `propose_join_keys` and `propose_grouping_indexes` take the parameter names
Task 4 establishes for the existing rules (`usage`, `facts`, `existing`, `min_cost_share`,
`min_rows`, `have_index_data`), minus the `schema=` keyword Task 4 removes.

**Carried forward from Batch 1** (surfaced to the user, not implemented here): `ci.yml`
runs `uv sync --all-extras`, so no CI job can catch a regression of the psycopg guard that
keeps the default suite Docker-free and extra-free. Global Constraints restate the
invariant; a permanent guard remains a separate follow-up.
