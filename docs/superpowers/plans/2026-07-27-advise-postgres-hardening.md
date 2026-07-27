# Postgres `advise` Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the shipped Postgres `advise` adapter trustworthy — teach it about expression and partial indexes, validate its introspection SQL against a real server for the first time, and move the credential-scrubbing helpers somewhere a second adapter cannot bypass them.

**Architecture:** Three independent strands. The index-metadata strand widens `CAP_INDEXES` to carry `indpred`/`indexprs` and stops the `pg_attribute` join silently discarding expression columns, which lets `_covered` and ADV003 stop guessing. The integration strand adds a real Postgres behind an opt-in pytest marker, so the six introspection statements are finally executed rather than only diffed. The extraction strand moves engine-neutral secret handling into its own module before Redshift can hand-roll it.

**Tech Stack:** Python 3.11+, psycopg 3, Postgres 16 via Docker, pytest markers, sqlglot 30.12.

Follow-up to `docs/superpowers/plans/2026-07-26-advise-postgres.md` (shipped in PRs #9/#10). Every item here was recorded as deferred during that plan's fifteen task reviews or its final whole-branch review; the reasoning for each is in `.superpowers/sdd/2026-07-26-advise-postgres/progress.md`.

## Global Constraints

- Python `>=3.11`. Every new module starts with `from __future__ import annotations`.
- Ruff line length 100.
- **CI gates all four of these over the whole repo; every task must pass all four before committing:**
  ```
  uv run ruff check .
  uv run ruff format --check .
  uv run mypy src/sqlquality
  uv run pytest
  ```
  The code blocks below are written for readability and are **not** guaranteed `ruff format` clean. Run `uv run ruff format .` after transcribing and commit the formatted result.
- Baseline at the start of this plan: **411 tests**, `main` at `d6729f9`.
- Invariants from the shipped feature that must survive every task:
  1. sqlquality never executes user SQL; only the statements in `PostgresWorkloadAdapter.SQL` ever run. `advise` never issues DDL or DML.
  2. `connect()` sets `default_transaction_read_only` and a statement timeout before the session is usable.
  3. No secret reaches stdout, stderr, an exception message, or the exception chain.
  4. A missing grant costs exactly one capability, never the whole run.
  5. `advise` exits 0 on any successful analysis and 2 on error — **never 1**.
  6. Absent evidence lowers confidence and says so in the artifact the operator reads.
- **New introspection SQL must keep `tests/test_workload_postgres.py`'s write-verb guard green.** `_write_verbs_in` matches whole words via `\b`, so a statement containing the *word* `create` — including inside a `pg_get_indexdef()` result at runtime — is fine, but the statement text itself must not contain one.
- The integration tests must be **skipped by default**. A contributor without Docker runs `uv run pytest` and sees no failures and no errors.

## File Structure

**Create:**

| File | Responsibility |
|---|---|
| `src/sqlquality/workload/secrets.py` | Engine-neutral credential handling: which fields are secret, how to collect them from a `ConnectionParams`, how to scrub them out of driver text, and the statement-timeout clamp. |
| `tests/integration/__init__.py` | Marks the integration package. |
| `tests/integration/conftest.py` | The opt-in gate and the live-connection fixture. |
| `tests/integration/docker-compose.yml` | Postgres 16 with `pg_stat_statements` preloaded. |
| `tests/integration/test_introspection_live.py` | Executes all six introspection statements against a real server and asserts their shapes. |
| `tests/integration/test_advise_live.py` | One end-to-end `advise` run against a seeded database. |
| `tests/test_workload_secrets.py` | Moves the secret-handling unit tests to sit beside their new module. |

**Modify:** `src/sqlquality/workload/postgres.py` (widened `CAP_INDEXES`, richer `PgIndex`, coverage and ADV003 changes, secrets re-exported from the new module), `src/sqlquality/workload/aggregate.py` (`star_tables` regex cache), `src/sqlquality/models.py` (`ColumnUsage.fingerprints` becomes a property), `pyproject.toml` (pytest marker), `CONTRIBUTING.md` (how to run the integration suite), `README.md` (two limitations become narrower).

`secrets.py` is deliberately its own module rather than a section of `postgres.py`: the final whole-branch review identified it as the seam that matters, because a Redshift or Snowflake `connect()` that cannot see these helpers will re-implement scrubbing by hand, and scrubbing is the one thing on this branch that took three fix rounds to get right.

---

### Task 1: Extract engine-neutral credential handling

**Files:**
- Create: `src/sqlquality/workload/secrets.py`
- Modify: `src/sqlquality/workload/postgres.py:84-170` (remove the moved definitions, import them instead)
- Create: `tests/test_workload_secrets.py`
- Modify: `tests/test_workload_postgres.py` (move the secret-handling tests out)

**Interfaces:**
- Consumes: `ConnectionParams` from `sqlquality.models`.
- Produces, all importable from `sqlquality.workload.secrets`:
  - `SECRET_FIELDS: frozenset[str]`
  - `MIN_SCRUBBABLE_SECRET: int`
  - `WITHHELD: str`
  - `secrets_for(params: ConnectionParams) -> tuple[str, ...]`
  - `scrub(text: str, secrets: Iterable[str]) -> str`
  - `clamp_timeout_ms(timeout_s: int, *, minimum: int, maximum: int) -> int`

Note the rename from the private `_`-prefixed names: these are now a module boundary other adapters import across, so they are public. `clamp_timeout_ms` gains explicit bounds parameters rather than reading module constants, because the CLI already owns the user-facing bounds and duplicating them was a recorded finding (`_TIMEOUT_MIN_S`/`_MAX_S` in `cli.py` versus `_MIN_TIMEOUT_S`/`_MAX_TIMEOUT_S` in `postgres.py`).

- [ ] **Step 1: Write the failing test**

Create `tests/test_workload_secrets.py`:

```python
from __future__ import annotations

import pytest

from sqlquality.models import ConnectionParams
from sqlquality.workload.secrets import (
    MIN_SCRUBBABLE_SECRET,
    SECRET_FIELDS,
    WITHHELD,
    clamp_timeout_ms,
    scrub,
    secrets_for,
)


def _params(**kwargs) -> ConnectionParams:
    base = {"engine": "postgres", "dsn": None, "fields": {}, "source": "--dsn"}
    base.update(kwargs)
    return ConnectionParams(**base)  # type: ignore[arg-type]


def test_secrets_for_collects_password_fields():
    assert secrets_for(_params(fields={"host": "db", "password": "hunter2"})) == ("hunter2",)


def test_secrets_for_covers_both_forms_of_a_dsn_password():
    """urlparse leaves the password encoded; libpq decodes it before authenticating."""
    got = secrets_for(_params(dsn="postgresql://u:p%40ss@h/db"))
    assert "p%40ss" in got
    assert "p@ss" in got


def test_secrets_for_tolerates_a_dsn_with_no_password_or_a_malformed_one():
    for dsn in ("postgresql://u@h/db", "not a valid dsn :: at all ///"):
        assert secrets_for(_params(dsn=dsn)) == (dsn,)


def test_scrub_redacts_a_present_secret():
    assert scrub('failed for user "u" (hunter2)', ("hunter2",)) == 'failed for user "u" (***)'


def test_scrub_withholds_rather_than_mangles_an_unredactable_secret():
    assert scrub("a database has an admin", ("a",)) == WITHHELD
    assert scrub("connection refused", ("a",)) == "connection refused"


def test_min_scrubbable_secret_is_the_documented_floor():
    assert MIN_SCRUBBABLE_SECRET == 4
    assert "password" in SECRET_FIELDS


@pytest.mark.parametrize(
    ("given", "expected_ms"),
    [(0, 1_000), (-5, 1_000), (30, 30_000), (99_999, 3_600_000)],
)
def test_clamp_timeout_ms_bounds_and_converts(given, expected_ms):
    assert clamp_timeout_ms(given, minimum=1, maximum=3600) == expected_ms
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_workload_secrets.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sqlquality.workload.secrets'`

- [ ] **Step 3: Write minimal implementation**

Create `src/sqlquality/workload/secrets.py` by **moving** — not copying — the bodies currently in `postgres.py`, renaming them public and dropping the module-constant coupling in the clamp:

```python
"""Credential handling shared by every workload adapter.

This lives outside any one adapter deliberately. Scrubbing took three fix rounds to get
right on the Postgres adapter — the driver's exception text quoted the offending value, then
`from None` turned out to suppress only the traceback while leaving `__context__` reachable,
then a percent-encoded DSN password slipped past because ``urlparse`` returns it still
encoded. An adapter that cannot see these helpers will re-derive that sequence badly.
"""

from __future__ import annotations

from collections.abc import Iterable
from urllib.parse import unquote, urlparse

from sqlquality.models import ConnectionParams

#: profiles.yml keys whose values must never appear in any message we emit.
SECRET_FIELDS = frozenset({"password", "pass"})

#: A secret shorter than this cannot be redacted by substring replacement without destroying
#: the message — a one-character password would blank every occurrence of that letter. When
#: one actually appears, the driver's text is withheld rather than mangled.
MIN_SCRUBBABLE_SECRET = 4
WITHHELD = "(driver message withheld: it contained a value too short to redact safely)"


def secrets_for(params: ConnectionParams) -> tuple[str, ...]:
    """Every value we know to be secret for this connection.

    A DSN is added *and* its password extracted separately, in both its percent-encoded and
    decoded forms. The whole-DSN token only helps if the driver echoes the connection string
    back verbatim, which real libpq errors do not do — they report the offending value on its
    own. And ``urlparse().password`` returns it still encoded while libpq decodes a URI DSN
    before authenticating, so for ``postgresql://u:p%40ss@h/db`` the driver reports ``p@ss``
    while urlparse yields ``p%40ss``: a token of only the encoded form never matches.
    """
    secrets = tuple(
        value for key, value in params.fields.items() if key in SECRET_FIELDS and value
    )
    if params.dsn:
        secrets += (params.dsn,)
        encoded = urlparse(params.dsn).password
        if encoded:
            secrets += (encoded,)
            decoded = unquote(encoded)
            if decoded != encoded:
                secrets += (decoded,)
    return secrets


def scrub(text: str, secrets: Iterable[str]) -> str:
    """Replace any known secret occurring in ``text`` with a redaction marker.

    Defence in depth for driver exceptions. libpq is not believed to echo a password, but the
    auth-failure path — the most common real connect failure — cannot be exercised without a
    live server, and we hold the secret anyway, so its absence can be guaranteed rather than
    trusted.
    """
    present = [secret for secret in secrets if secret and secret in text]
    if any(len(secret) < MIN_SCRUBBABLE_SECRET for secret in present):
        return WITHHELD
    scrubbed = text
    for secret in present:
        scrubbed = scrubbed.replace(secret, "***")
    return scrubbed


def clamp_timeout_ms(timeout_s: int, *, minimum: int, maximum: int) -> int:
    """Statement timeout in milliseconds, clamped into ``[minimum, maximum]`` seconds.

    Bounds are parameters rather than module constants: the CLI owns the user-facing range
    and rejects out-of-range input, so a second copy of the numbers here could drift out of
    step with the message the user was shown.
    """
    return max(minimum, min(int(timeout_s), maximum)) * 1000
```

In `postgres.py`, delete `_SECRET_FIELDS`, `_MIN_SCRUBBABLE_SECRET`, `_WITHHELD`, `_secrets_for`, `_scrub` and `_clamp_timeout_ms`, and import instead:

```python
from sqlquality.workload.secrets import clamp_timeout_ms, scrub, secrets_for
```

Update the two call sites in `connect()`: `secrets = secrets_for(params)` and `scrub(str(exc), secrets)`, and the timeout call becomes:

```python
                cursor.execute(
                    "SELECT set_config('statement_timeout', %s, false)",
                    (
                        f"{clamp_timeout_ms(timeout_s, minimum=MIN_TIMEOUT_S, "
                        f"maximum=MAX_TIMEOUT_S)}ms",
                    ),
                )
```

Keep `_pg_fields` and `_PG_FIELD_MAP` in `postgres.py` — they map to libpq keywords and are not engine-neutral.

- [ ] **Step 4: Move the existing tests rather than duplicating them**

`tests/test_workload_postgres.py` currently holds tests for these helpers. Delete the ones now covered by `tests/test_workload_secrets.py` — specifically any asserting on `_secrets_for`, `_scrub`, percent-encoded DSN passwords, or the withheld message. **Keep** `test_connect_scrubs_a_password_from_a_driver_failure`, which exercises the adapter's use of them rather than the helpers themselves, and update it to patch or reference the new names if it does so directly.

Run: `uv run pytest tests/test_workload_secrets.py tests/test_workload_postgres.py -v`
Expected: PASS, with no test name appearing in both files.

- [ ] **Step 5: Run the full gates**

Run: `uv run ruff format . && uv run ruff check . && uv run ruff format --check . && uv run mypy src/sqlquality && uv run pytest -q`
Expected: all pass. Total count should be unchanged or slightly higher — if it *dropped*, a test was deleted without a replacement.

- [ ] **Step 6: Commit**

```bash
git add src/sqlquality/workload/secrets.py src/sqlquality/workload/postgres.py tests/test_workload_secrets.py tests/test_workload_postgres.py
git commit -m "refactor(workload): move credential handling into its own module"
```

---

### Task 2: Stop discarding expression-index columns

**Files:**
- Modify: `src/sqlquality/workload/postgres.py` — `CAP_INDEXES` SQL, `PgIndex`, `_IndexRows`, `fetch_indexes`
- Test: `tests/test_workload_postgres.py`

**Interfaces:**
- Consumes: `_as_int`, `_run`, `CAP_INDEXES` (existing).
- Produces: `PgIndex` gains three fields, in this exact order after `size_bytes`:
  ```python
  is_partial: bool = False
  predicate: str | None = None
  has_expressions: bool = False
  ```
  All defaulted, so existing test constructors that pass six positional or keyword arguments keep working. Task 3 and Task 4 read all three.

**The defect being fixed.** `CAP_INDEXES` inner-joins `pg_attribute` on `a.attnum = k.attnum`. Postgres stores **0** in `indkey` for an expression column, and no `pg_attribute` row has `attnum = 0`, so every expression column is silently dropped — an index on `lower(status)` currently arrives with an *empty* column tuple, and `_covered` then compares against `()`. Recorded during Task 7's review and deferred.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_workload_postgres.py`:

```python
def test_fetch_indexes_records_an_expression_index_rather_than_dropping_it():
    """`indkey` holds 0 for an expression column and no pg_attribute row has attnum 0.

    The old inner join therefore discarded those rows, so an index on `lower(status)`
    arrived with an empty column tuple and could not be reasoned about at all.
    """
    querier = FakeQuerier({"pg_index": [
        # attname is NULL for the expression column, as a LEFT JOIN yields.
        ("orders", "idx_lower_status", None, 1, False, False, 3, 8192,
         False, None, True, "CREATE INDEX idx_lower_status ON orders (lower(status))"),
    ]})
    indexes = PostgresWorkloadAdapter(querier=querier).fetch_indexes(
        ("public",), frozenset({"orders"})
    )
    index = indexes["orders"][0]
    assert index.has_expressions is True
    assert index.columns == ()
    assert "lower(status)" in (index.definition or "")


def test_fetch_indexes_records_a_partial_index_predicate():
    querier = FakeQuerier({"pg_index": [
        ("orders", "idx_open", "status", 1, False, False, 7, 4096,
         True, "(shipped_at IS NULL)", False,
         "CREATE INDEX idx_open ON orders (status) WHERE shipped_at IS NULL"),
    ]})
    index = PostgresWorkloadAdapter(querier=querier).fetch_indexes(
        ("public",), frozenset({"orders"})
    )["orders"][0]
    assert index.is_partial is True
    assert index.predicate == "(shipped_at IS NULL)"
    assert index.columns == ("status",)


def test_fetch_indexes_leaves_a_plain_index_unmarked():
    querier = FakeQuerier({"pg_index": [
        ("orders", "idx_status", "status", 1, False, False, 12, 4096,
         False, None, False, "CREATE INDEX idx_status ON orders (status)"),
    ]})
    index = PostgresWorkloadAdapter(querier=querier).fetch_indexes(
        ("public",), frozenset({"orders"})
    )["orders"][0]
    assert (index.is_partial, index.predicate, index.has_expressions) == (False, None, False)


def test_the_indexes_statement_reads_predicate_and_expression_metadata():
    sql = PostgresWorkloadAdapter().SQL[CAP_INDEXES].lower()
    assert "indpred" in sql, "the partial-index predicate must be selected"
    assert "indexprs" in sql, "expression presence must be selected"
    assert "left join pg_attribute" in sql, (
        "an inner join drops expression columns, whose indkey entry is 0"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_workload_postgres.py -k "expression_index or partial_index or unmarked or predicate_and_expression" -v`
Expected: FAIL — `ValueError: too many values to unpack` from `fetch_indexes`, and the SQL assertions fail on the missing clauses.

- [ ] **Step 3: Widen the statement**

Replace `CAP_INDEXES` in `PostgresWorkloadAdapter.SQL`:

```python
        # LEFT JOIN, not JOIN: Postgres stores 0 in indkey for an expression column and no
        # pg_attribute row has attnum 0, so an inner join silently discarded every expression
        # index's columns — they arrived with an empty tuple. The NULL attname a LEFT JOIN
        # yields is what tells us the position was an expression.
        #
        # indpred / indexprs are selected as booleans plus the rendered predicate, because a
        # partial index does not serve an unfiltered lookup and an expression index does not
        # serve its bare column — both of which the coverage and redundancy rules previously
        # had to guess at.
        CAP_INDEXES: """
            SELECT t.relname, i.relname, a.attname, k.ordinality,
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
            ORDER BY t.relname, i.relname, k.ordinality
        """,
```

- [ ] **Step 4: Carry the new columns through**

Extend `PgIndex`:

```python
    size_bytes: int
    #: True when the index has a WHERE predicate. A partial index does not serve an
    #: unfiltered lookup, so it can never be assumed to cover a proposed index.
    is_partial: bool = False
    #: The rendered predicate, for showing an operator why a drop was not recommended.
    predicate: str | None = None
    #: True when any indexed position is an expression rather than a plain column. Such a
    #: position contributes no name to `columns`, so the tuple understates the index.
    has_expressions: bool = False
    #: The full CREATE INDEX text, the only place an expression is legible.
    definition: str | None = None
```

Extend `_IndexRows` with the same four fields (mutable dataclass, so plain defaults), and rewrite the unpack in `fetch_indexes`:

```python
        for row in self._run(CAP_INDEXES, (list(schemas), sorted(tables))):
            (table, index, column, ordinality, unique, primary, scans, size,
             is_partial, predicate, has_expressions, definition) = row
            entry = grouped.setdefault(
                (str(table), str(index)),
                _IndexRows(
                    is_unique=bool(unique),
                    is_primary=bool(primary),
                    scans=_as_int(scans),
                    size_bytes=_as_int(size) if size is not None else 0,
                    is_partial=bool(is_partial),
                    predicate=str(predicate) if predicate is not None else None,
                    has_expressions=bool(has_expressions),
                    definition=str(definition) if definition is not None else None,
                ),
            )
            # A NULL attname is an expression position: it has no column name to record, and
            # `has_expressions` already marks the index, so skip it rather than storing "None".
            if column is not None:
                entry.columns.append((_as_int(ordinality), str(column)))
```

and pass the four through when building each `PgIndex`.

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_workload_postgres.py tests/test_workload_rules.py -q`
Expected: PASS. The rules tests construct `PgIndex` with six arguments; the new fields are defaulted, so they must still pass unchanged. If any fails, the defaults are wrong — report it rather than editing the rules tests.

- [ ] **Step 6: Gates and commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ruff format --check . && uv run mypy src/sqlquality && uv run pytest -q
git add src/sqlquality/workload/postgres.py tests/test_workload_postgres.py
git commit -m "fix(advise): stop discarding expression-index columns from the catalog"
```

---

### Task 3: A partial or expression index no longer counts as covering

**Files:**
- Modify: `src/sqlquality/workload/postgres.py` — `_covered`, `propose_indexes`
- Test: `tests/test_workload_rules.py`

**Interfaces:**
- Consumes: `PgIndex.is_partial` / `.has_expressions` / `.definition` (Task 2); `_is_prefix`, `_covered`, `propose_indexes` (existing).
- Produces: `_covered` keeps its signature `(candidate, existing) -> str | None`. `propose_indexes` gains two evidence keys, `"expression_indexes"` (a tuple of index names) and `"partial_indexes_skipped"` (a tuple of names), and appends a rationale sentence when either is non-empty.

**Why this is two changes, not one.** Suppression and disclosure pull opposite ways here. A partial index must *stop* suppressing a candidate — `idx ON orders(status) WHERE shipped_at IS NULL` does not serve `WHERE status = $1`, so treating it as coverage silently withholds a good proposal. But an expression index must not be silently ignored either: if `lower(status)` is indexed and we propose `status`, the proposal may be redundant in a way we cannot prove. So: neither suppresses, and both are disclosed.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_workload_rules.py`:

```python
def test_a_partial_index_does_not_suppress_a_candidate():
    """`idx ON orders(status) WHERE shipped_at IS NULL` does not serve `WHERE status = $1`.

    Treating it as coverage silently withheld a good proposal — the inverse of the
    confidently-wrong failures, and just as invisible.
    """
    existing = {"orders": (
        PgIndex("idx_open", ("status",), False, False, 5, 4096,
                is_partial=True, predicate="(shipped_at IS NULL)"),
    )}
    proposals = propose_indexes(
        [usage("status", ColumnRole.EQUALITY)], facts(), existing, min_cost_share=0.01,
    )
    assert codes(proposals) == ["ADV001"]
    assert proposals[0].evidence["partial_indexes_skipped"] == ("idx_open",)
    assert "partial" in proposals[0].rationale.lower()


def test_a_plain_index_still_suppresses_a_candidate():
    """The control. Task 2's new fields default to False, so this must not have changed."""
    existing = {"orders": (PgIndex("idx_status", ("status",), False, False, 5, 4096),)}
    proposals = propose_indexes(
        [usage("status", ColumnRole.EQUALITY)], facts(), existing, min_cost_share=0.01,
    )
    assert proposals == []


def test_an_expression_index_is_disclosed_not_silently_ignored():
    """We cannot prove `lower(status)` makes an index on `status` redundant — or that it
    doesn't. Saying so beats both suppressing and pretending it isn't there."""
    existing = {"orders": (
        PgIndex("idx_lower_status", (), False, False, 5, 4096,
                has_expressions=True,
                definition="CREATE INDEX idx_lower_status ON orders (lower(status))"),
    )}
    proposals = propose_indexes(
        [usage("status", ColumnRole.EQUALITY)], facts(), existing, min_cost_share=0.01,
    )
    assert codes(proposals) == ["ADV001"]
    assert proposals[0].evidence["expression_indexes"] == ("idx_lower_status",)
    assert "expression" in proposals[0].rationale.lower()


def test_an_expression_index_not_mentioning_the_column_is_not_disclosed():
    """Only expression indexes that plausibly relate to the candidate are worth naming."""
    existing = {"orders": (
        PgIndex("idx_lower_note", (), False, False, 5, 4096,
                has_expressions=True,
                definition="CREATE INDEX idx_lower_note ON orders (lower(note))"),
    )}
    proposals = propose_indexes(
        [usage("status", ColumnRole.EQUALITY)], facts(), existing, min_cost_share=0.01,
    )
    assert proposals[0].evidence["expression_indexes"] == ()
    assert "expression index" not in proposals[0].rationale.lower()


def test_a_column_name_inside_a_longer_identifier_is_not_disclosed():
    """`id` is a substring of `guid`, and a substring test said so out loud.

    The rationale would have told the operator an index "mentions id" when it indexes
    `lower(guid)` — a false claim in the text someone reads before running DDL.
    """
    existing = {"orders": (
        PgIndex("idx_lower_guid", (), False, False, 5, 4096,
                has_expressions=True,
                definition="CREATE INDEX idx_lower_guid ON orders (lower(guid))"),
    )}
    proposals = propose_indexes(
        [usage("id", ColumnRole.EQUALITY)],
        facts(columns=("id", "guid")), existing, min_cost_share=0.01,
    )
    assert proposals[0].evidence["expression_indexes"] == ()


def test_an_expression_index_on_a_cast_of_the_column_is_still_disclosed():
    """The control for the fix: word boundaries must not cost a true positive."""
    existing = {"orders": (
        PgIndex("idx_status_cast", (), False, False, 5, 4096,
                has_expressions=True,
                definition="CREATE INDEX idx_status_cast ON orders ((status::text))"),
    )}
    proposals = propose_indexes(
        [usage("status", ColumnRole.EQUALITY)], facts(), existing, min_cost_share=0.01,
    )
    assert proposals[0].evidence["expression_indexes"] == ("idx_status_cast",)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_workload_rules.py -k "partial_index_does_not_suppress or expression_index" -v`
Expected: FAIL — the partial index currently suppresses (so `codes(proposals) == []`), and `evidence["expression_indexes"]` raises `KeyError`.

- [ ] **Step 2b: Generalise the existing word-boundary helper instead of writing a third**

`src/sqlquality/workload/aggregate.py` already has `mentions_table(name, sql)`, whose whole
purpose is whole-identifier matching, and the test suite has `_write_verbs_in` doing the same
for a different vocabulary. Adding a third copy in `postgres.py` would be the duplication the
final review of the previous plan specifically called out. Rename the mechanism and keep the
caller:

```python
def mentions_identifier(name: str, text: str) -> bool:
    """True if ``name`` appears in ``text`` as a whole identifier, not merely a substring.

    A plain `name in text` test false-positives on any name that is a substring of a longer
    identifier — `order` inside `orders`, `cart` inside `shopping_cart`, `id` inside `guid` —
    and on a name appearing only in an alias like `orders_total`. `\b` already treats `_` as
    a word character in Python's `re`, so it rejects all of those without a custom class,
    while still matching across the punctuation Postgres puts around identifiers: parens,
    commas, dots and `::` are all non-word characters.
    """
    return re.search(rf"\b{re.escape(name)}\b", text) is not None


def mentions_table(name: str, sql: str) -> bool:
    """True if a query mentions this table. See :func:`mentions_identifier`."""
    return mentions_identifier(name, sql)
```

`postgres.py` then imports `mentions_identifier` from `sqlquality.workload.aggregate`
alongside whatever it already imports from there. This widens the task's file scope to
`aggregate.py` by two functions, deliberately: a third hand-rolled `\b` regex is worse.

- [ ] **Step 3: Make coverage refuse to guess**

Replace `_covered`:

```python
def _covered(candidate: tuple[str, ...], existing: Sequence[PgIndex]) -> str | None:
    """Name of a *plain* existing index whose leading columns already cover ``candidate``.

    Partial and expression indexes are excluded, for opposite reasons that land in the same
    place. A partial index does not serve an unfiltered lookup, so calling it coverage
    silently withholds a real proposal. An expression index's `columns` tuple understates it
    — the expression positions contribute no name — so a prefix match against it is not a
    match at all. Neither can be *proven* irrelevant either, which is why `propose_indexes`
    discloses them instead of dropping them on the floor.
    """
    for index in existing:
        if index.is_partial or index.has_expressions:
            continue
        if _is_prefix(candidate, index.columns):
            return index.name
    return None
```

- [ ] **Step 4: Disclose what was skipped**

In `propose_indexes`, after `covered_by = _covered(columns, existing.get(table, ()))` and its `continue`, gather the two lists and fold them into the evidence and rationale:

```python
        table_indexes = existing.get(table, ())
        partial_skipped = tuple(
            index.name
            for index in table_indexes
            if index.is_partial and _is_prefix(columns, index.columns)
        )
        # Only expression indexes whose definition mentions the leading column are worth
        # naming. Proving `lower(status)` equivalent to `status` would need the expression
        # parsed and matched; naming it lets the operator make that call in one glance.
        #
        # Whole-identifier matching, not a substring test. `columns[0] in definition` reports
        # a candidate on `id` against an index on `lower(guid)`, and the rationale then tells
        # the operator an index "mentions id" when it does not — a claim the tool cannot
        # support, in the string someone reads while deciding whether to run DDL. Verified
        # that `\b` keeps the true positives: `lower(status)`, `lower(customer_id::text)`
        # and `(id::text)` all still match, because Postgres separates identifiers with
        # parens, commas, dots and `::`, none of which are word characters.
        expression_indexes = tuple(
            index.name
            for index in table_indexes
            if index.has_expressions and mentions_identifier(columns[0], index.definition or "")
        )
```

Add both to the `evidence` dict, and after the existing `rationale` assignment:

```python
        if partial_skipped:
            rationale += (
                f" A partial index ({', '.join(partial_skipped)}) leads with these columns "
                "but carries a WHERE predicate, so it does not serve an unfiltered lookup — "
                "it is not treated as covering this proposal."
            )
        if expression_indexes:
            rationale += (
                f" An expression index ({', '.join(expression_indexes)}) mentions "
                f"{columns[0]}; sqlquality cannot tell whether it already serves this "
                "lookup, so confirm before applying."
            )
```

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/test_workload_rules.py -q`
Expected: PASS, including the two controls (`test_a_plain_index_still_suppresses_a_candidate`, `test_an_expression_index_not_mentioning_the_column_is_not_disclosed`).

- [ ] **Step 6: Gates and commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ruff format --check . && uv run mypy src/sqlquality && uv run pytest -q
git add src/sqlquality/workload/postgres.py tests/test_workload_rules.py
git commit -m "fix(advise): a partial or expression index no longer counts as coverage"
```

---

### Task 4: ADV003 earns HIGH back, precisely

**Files:**
- Modify: `src/sqlquality/workload/postgres.py` — `propose_redundant_indexes`
- Test: `tests/test_workload_rules.py`

**Interfaces:**
- Consumes: `PgIndex.is_partial` / `.has_expressions` / `.predicate` (Task 2).
- Produces: no signature change. `propose_redundant_indexes` now returns HIGH when both indexes in a pair are plain, and skips the pair entirely when either is partial or expression-bearing.

**Why skip rather than downgrade.** The shipped rule caps every ADV003 at MEDIUM with a blanket caveat, because it could not see predicates. Now it can. For a genuinely plain pair, prefix redundancy is provable from the column lists and HIGH is honest. For a pair involving a partial index, the recommendation is not merely less certain — it is very likely *wrong*, since the partial index exists precisely to serve a subset the wider index serves differently. Emitting it at MEDIUM would still be advising a `DROP INDEX` we have no basis for.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_workload_rules.py`:

```python
def test_a_plain_redundant_pair_is_high_confidence():
    existing = {"orders": (
        PgIndex("idx_narrow", ("status",), False, False, 5, 1),
        PgIndex("idx_wide", ("status", "created_at"), False, False, 5, 1),
    )}
    proposals = propose_redundant_indexes(existing)
    assert codes(proposals) == ["ADV003"]
    assert proposals[0].confidence is Confidence.HIGH
    assert proposals[0].evidence["index"] == "idx_narrow"
    # Pin the claim, not just the confidence. Deleting the old MEDIUM test removed the only
    # assertion on this rationale's wording, so a future edit could reintroduce a hedge, or
    # drop the "both are plain" claim while leaving HIGH, with nothing failing.
    assert "plain" in proposals[0].rationale
    assert "partial" not in proposals[0].rationale


def test_a_partial_narrow_index_is_never_called_redundant():
    """The partial index exists to serve a subset; the wider full index serves it
    differently. Dropping it is not less certain, it is probably wrong."""
    existing = {"orders": (
        PgIndex("idx_open", ("status",), False, False, 5, 1,
                is_partial=True, predicate="(shipped_at IS NULL)"),
        PgIndex("idx_wide", ("status", "created_at"), False, False, 5, 1),
    )}
    assert propose_redundant_indexes(existing) == []


def test_a_partial_wider_index_does_not_supersede_a_plain_one():
    existing = {"orders": (
        PgIndex("idx_narrow", ("status",), False, False, 5, 1),
        PgIndex("idx_wide_open", ("status", "created_at"), False, False, 5, 1,
                is_partial=True, predicate="(shipped_at IS NULL)"),
    )}
    assert propose_redundant_indexes(existing) == []


def test_a_wider_expression_index_does_not_supersede_a_plain_one():
    """The wider index must be strictly wider, or the length guard skips the pair anyway.

    An earlier version of this test gave both indexes one column, so
    `len(other.columns) > len(narrow.columns)` was already False and it passed whether or
    not the has_expressions filter existed at all.
    """
    existing = {"orders": (
        PgIndex("idx_narrow", ("status",), False, False, 5, 1),
        PgIndex("idx_expr", ("status", "note"), False, False, 5, 1, has_expressions=True,
                definition="CREATE INDEX idx_expr ON orders (status, note, lower(note))"),
    )}
    assert propose_redundant_indexes(existing) == []


def test_a_narrow_expression_index_is_never_called_redundant():
    """The other direction, and the reason it matters.

    `columns` understates an expression index — the expression positions contribute no
    name — so a narrow one may index something the wider one does not. Dropping it on a
    column-list comparison would discard an index nothing else provides.
    """
    existing = {"orders": (
        PgIndex("idx_narrow_expr", ("status",), False, False, 5, 1, has_expressions=True,
                definition="CREATE INDEX idx_narrow_expr ON orders (status, lower(note))"),
        PgIndex("idx_wide", ("status", "created_at"), False, False, 5, 1),
    )}
    assert propose_redundant_indexes(existing) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_workload_rules.py -k redundant -v`
Expected: FAIL — the plain pair currently returns MEDIUM, and the partial and expression pairs are currently proposed rather than skipped.

One existing test must be **replaced, not kept**: `test_redundant_prefix_index_proposed_for_drop` at `tests/test_workload_rules.py:381` asserts `Confidence.MEDIUM` with a comment explaining the cap ("PgIndex carries no predicate"). That comment stops being true in Task 2, so delete that test — `test_a_plain_redundant_pair_is_high_confidence` above is its replacement and covers the same case. Keeping both would leave a test asserting the old behaviour.

- [ ] **Step 3: Implement**

In `propose_redundant_indexes`, skip any index that is not plain, and restore HIGH:

```python
    for table, indexes in sorted(existing.items()):
        for narrow in indexes:
            # A partial or expression index is not comparable on column lists alone: the
            # predicate or the expression is the whole point of it. Skipping the pair is the
            # honest answer, because "probably wrong" is not a confidence level.
            if narrow.is_unique or narrow.is_primary or narrow.is_partial:
                continue
            if narrow.has_expressions:
                continue
            wider = next(
                (
                    other
                    for other in indexes
                    if other.name != narrow.name
                    and not other.is_partial
                    and not other.has_expressions
                    and len(other.columns) > len(narrow.columns)
                    and _is_prefix(narrow.columns, other.columns)
                ),
                None,
            )
```

and change the emitted proposal's `confidence` to `Confidence.HIGH`, replacing the blanket caveat with the now-true statement:

```python
                    rationale=(
                        f"Its columns are a leading prefix of {wider.name}, which can serve "
                        "the same lookups. Both indexes are plain — neither carries a WHERE "
                        "predicate nor an indexed expression — so the column lists are the "
                        "whole comparison."
                    ),
```

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/test_workload_rules.py -q`
Expected: PASS.

- [ ] **Step 5: Confirm the dedup precedence still holds**

`_dedupe_by_ddl` keeps the strongest confidence when ADV002 and ADV003 target the same `DROP INDEX`, and a `_RULE_PRECEDENCE` tiebreak was added when ADV003 was capped at MEDIUM and the two tied. ADV003 is HIGH again, so it now wins on confidence alone.

Run: `uv run pytest tests/test_workload_rules.py -k dedupe -v`
Expected: PASS with ADV003 still surviving. If it fails, report it — do not adjust `_RULE_PRECEDENCE` without saying so, since it was added deliberately.

- [ ] **Step 6: Gates and commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ruff format --check . && uv run mypy src/sqlquality && uv run pytest -q
git add src/sqlquality/workload/postgres.py tests/test_workload_rules.py
git commit -m "fix(advise): ADV003 is HIGH for plain pairs and silent for the rest"
```

---

### Task 5: Two recorded trivia

**Files:**
- Modify: `src/sqlquality/models.py` (`ColumnUsage`), `src/sqlquality/workload/aggregate.py` (`star_tables`)
- Test: `tests/test_models.py`, `tests/test_workload_aggregate.py`

**Interfaces:**
- Produces: `ColumnUsage.fingerprints` becomes a read-only property returning `len(fingerprint_ids)`; it is no longer a constructor argument. `star_tables` keeps its signature.

Both were recorded by the final whole-branch review. `fingerprints` and `fingerprint_ids` are redundant and kept in sync only by convention, with no invariant enforcing it — and there are only three `fingerprints=` call sites in the whole repo. `star_tables` compiles a fresh regex per (star-stat × table) pair, which thrashes `re`'s pattern cache on a schema with many tables.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_models.py`:

```python
def test_fingerprints_is_derived_from_the_id_set():
    """One source of truth. The two used to be separate fields kept in step by convention,
    with nothing stopping a caller setting one and not the other."""
    usage = ColumnUsage(
        table="orders", column="status", role=ColumnRole.EQUALITY, calls=5,
        cost_ms=50.0, cost_share=0.5, fingerprint_ids=frozenset({"a", "b"}),
    )
    assert usage.fingerprints == 2

    with pytest.raises(TypeError):
        ColumnUsage(  # type: ignore[call-arg]
            table="orders", column="status", role=ColumnRole.EQUALITY, calls=5,
            cost_ms=50.0, cost_share=0.5, fingerprints=2,
        )
```

Append to `tests/test_workload_aggregate.py`:

```python
def test_star_tables_compiles_each_table_pattern_once(monkeypatch):
    """A fresh regex per (stat x table) pair thrashes re's pattern cache on a wide schema."""
    import re as _re

    from sqlquality.workload import aggregate as agg

    compiles: list[str] = []
    real_compile = _re.compile

    def counting_compile(pattern, *args, **kwargs):
        compiles.append(pattern)
        return real_compile(pattern, *args, **kwargs)

    monkeypatch.setattr(agg._re if hasattr(agg, "_re") else _re, "compile", counting_compile)
    workload = Workload(
        stats=tuple(
            QueryStat(fingerprint=f"fp{i}", sql="select * from orders", calls=1,
                      total_time_ms=1.0, flags=frozenset({FLAG_SELECT_STAR}))
            for i in range(5)
        ),
        window_description="w",
    )
    schema = {f"t{i}": {"c": "int"} for i in range(20)} | {"orders": {"c": "int"}}
    assert agg.star_tables(workload, schema) == frozenset({"orders"})
    assert len(compiles) <= len(schema), (
        f"compiled {len(compiles)} patterns for {len(schema)} tables across 5 stats"
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_models.py -k fingerprints_is_derived tests/test_workload_aggregate.py -k star_tables_compiles -v`
Expected: FAIL — `fingerprints=` is currently accepted so no `TypeError` is raised, and the compile count is 5 × 21.

- [ ] **Step 3: Implement**

In `models.py`, delete the `fingerprints: int` field and add a property after the dataclass fields:

```python
    @property
    def fingerprints(self) -> int:
        """How many query groups contributed this usage.

        Derived rather than stored: it and `fingerprint_ids` were two fields carrying one
        fact, kept in step only by convention.
        """
        return len(self.fingerprint_ids)
```

In `aggregate.py`, drop `fingerprints=...` from the `ColumnUsage(...)` construction and delete the now-unused `fingerprints` counter dict. In `star_tables`, hoist the per-table patterns out of the per-stat loop with a module-level cache:

```python
@lru_cache(maxsize=4096)
def _identifier_pattern(name: str) -> re.Pattern[str]:
    """Compiled whole-identifier matcher for one table name, compiled once per name."""
    return re.compile(rf"\b{re.escape(name)}\b")
```

and have both `mentions_table` and `star_tables` use `_identifier_pattern(name).search(sql)`.

- [ ] **Step 4: Fix the remaining call sites**

`grep -rn "fingerprints=" src/ tests/` and remove every one — the property is computed. Any test asserting `usage.fingerprints == N` should now build a `fingerprint_ids` set of size N.

Run: `uv run pytest -q`
Expected: PASS.

- [ ] **Step 5: Gates and commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ruff format --check . && uv run mypy src/sqlquality && uv run pytest -q
git add src/sqlquality/models.py src/sqlquality/workload/aggregate.py tests/
git commit -m "refactor: derive fingerprints from its id set, cache identifier patterns"
```

---

### Task 6: A real Postgres, behind an opt-in marker

**Files:**
- Create: `tests/integration/__init__.py`, `tests/integration/conftest.py`, `tests/integration/docker-compose.yml`, `tests/integration/test_introspection_live.py`
- Modify: `pyproject.toml` (register the marker and default deselection), `CONTRIBUTING.md`

**Interfaces:**
- Produces: a `live_dsn` fixture yielding a DSN string, and a `seeded` fixture yielding `(dsn, schema_name)` against a database with a table, an index, a partial index, an expression index and non-empty `pg_stat_statements`.

**Why this matters more than its size suggests.** Not one introspection statement in this feature has ever executed against a real server. They are tested for drift — that the text has not changed — which cannot catch a wrong column name, a wrong join, or a view that does not exist on a supported version. The `unnest(...) WITH ORDINALITY` join and Task 2's new `pg_get_expr` call are the two I would least like to be wrong about.

- [ ] **Step 1: Register the marker so the suite stays green without Docker**

In `pyproject.toml`, under `[tool.pytest.ini_options]`:

```toml
markers = [
    "integration: requires a live Postgres (opt in with SQLQUALITY_TEST_DSN or `-m integration`)",
]
addopts = "-m 'not integration'"
```

`addopts` is what keeps `uv run pytest` green for a contributor without Docker: the integration tests are deselected, not skipped-with-noise. Running them is `uv run pytest -m integration`.

- [ ] **Step 2: Write the compose file and the gate**

`tests/integration/docker-compose.yml`:

```yaml
# pg_stat_statements must be preloaded at server start; CREATE EXTENSION alone is not
# enough, which is why this is a compose file rather than a plain `services:` block.
services:
  postgres:
    image: postgres:16
    environment:
      POSTGRES_PASSWORD: sqlquality
      POSTGRES_DB: sqlquality_test
    command:
      - postgres
      - -c
      - shared_preload_libraries=pg_stat_statements
      - -c
      - pg_stat_statements.track=all
    ports:
      - "55432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres -d sqlquality_test"]
      interval: 2s
      timeout: 3s
      retries: 30
```

`tests/integration/__init__.py`: empty.

`tests/integration/conftest.py`:

```python
"""Opt-in live-Postgres fixtures.

Every test in this package is marked `integration` and deselected by default (see
pyproject.toml's addopts), so a contributor without Docker sees a clean `uv run pytest`.

Bring the server up with:
    docker compose -f tests/integration/docker-compose.yml up -d
    uv run pytest -m integration
"""

from __future__ import annotations

import os

import pytest

pytest.importorskip("psycopg", reason="integration tests need the postgres extra")

DEFAULT_DSN = "postgresql://postgres:sqlquality@127.0.0.1:55432/sqlquality_test"

pytestmark = pytest.mark.integration


@pytest.fixture(scope="session")
def live_dsn() -> str:
    """A reachable Postgres, or skip with an actionable message."""
    import psycopg

    dsn = os.environ.get("SQLQUALITY_TEST_DSN", DEFAULT_DSN)
    try:
        with psycopg.connect(dsn, connect_timeout=3) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
    except Exception as exc:  # driver-specific; the message is what matters
        pytest.skip(
            f"no Postgres at {dsn}: {exc}\n"
            "start one with: docker compose -f tests/integration/docker-compose.yml up -d"
        )
    return dsn


@pytest.fixture(scope="session")
def seeded(live_dsn: str) -> tuple[str, str]:
    """A schema with the index shapes the catalog query has to survive, plus real workload.

    Deliberately includes a partial and an expression index: those are exactly the rows the
    shipped statement discarded, and the only way to know the fix works is to read them back
    out of a real catalog.
    """
    import psycopg

    schema = "advise_it"
    with psycopg.connect(live_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS pg_stat_statements")
            cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
            cur.execute(f"CREATE SCHEMA {schema}")
            cur.execute(
                f"""CREATE TABLE {schema}.orders (
                       id bigserial PRIMARY KEY,
                       status text NOT NULL,
                       note text,
                       shipped_at timestamptz,
                       created_at timestamptz NOT NULL DEFAULT now())"""
            )
            cur.execute(f"CREATE INDEX idx_plain ON {schema}.orders (status, created_at)")
            cur.execute(
                f"CREATE INDEX idx_open ON {schema}.orders (status) "
                "WHERE shipped_at IS NULL"
            )
            cur.execute(f"CREATE INDEX idx_lower_note ON {schema}.orders (lower(note))")
            cur.execute(
                f"INSERT INTO {schema}.orders (status, note) "
                "SELECT 'paid', 'n' || g FROM generate_series(1, 500) g"
            )
            cur.execute("SELECT pg_stat_statements_reset()")
            # Real workload for the history statement to find.
            for _ in range(3):
                cur.execute(
                    f"SELECT id FROM {schema}.orders WHERE status = %s "
                    "AND created_at > now() - interval '1 day'",
                    ("paid",),
                )
                cur.fetchall()
    return live_dsn, schema
```

- [ ] **Step 3: Write the failing test**

`tests/integration/test_introspection_live.py`:

```python
"""Execute every introspection statement against a real server.

The unit suite only checks these statements for drift, which cannot catch a wrong column
name, a wrong join, or a view that does not exist. This is the only place they run.
"""

from __future__ import annotations

import pytest

from sqlquality.models import ConnectionParams
from sqlquality.workload.postgres import (
    CAP_INDEXES,
    CAP_NDV,
    CAP_SCHEMA,
    CAP_STATS_RESET,
    CAP_TABLE_FACTS,
    CAP_WORKLOAD,
    PostgresWorkloadAdapter,
)


@pytest.fixture
def adapter(seeded: tuple[str, str]) -> PostgresWorkloadAdapter:
    dsn, schema = seeded
    a = PostgresWorkloadAdapter()
    a.schemas = (schema,)
    a.connect(ConnectionParams(engine="postgres", dsn=dsn, fields={}, source="--dsn"), 30)
    return a


def test_every_introspection_statement_executes(adapter, seeded):
    """No statement may raise, and none may report a degraded capability."""
    _dsn, schema = seeded
    adapter.fetch_workload(None, 500)
    adapter.fetch_schema((schema,))
    adapter.fetch_table_facts((schema,), frozenset({"orders"}))
    adapter.fetch_indexes((schema,), frozenset({"orders"}))
    assert adapter.degraded == [], f"a statement failed against a real server: {adapter.degraded}"


def test_workload_statement_returns_our_own_queries(adapter):
    fetch = adapter.fetch_workload(None, 500)
    assert fetch.rows, "pg_stat_statements returned nothing"
    assert "since stats reset at" in fetch.window_description


def test_table_facts_reports_a_real_row_estimate_and_ndv(adapter, seeded):
    _dsn, schema = seeded
    facts = adapter.fetch_table_facts((schema,), frozenset({"orders"}))["orders"]
    assert facts.row_estimate is not None and facts.row_estimate > 0
    assert "status" in facts.columns
    assert facts.ndv, "pg_stats returned no distinct-value estimates"


def test_indexes_statement_reads_partial_and_expression_metadata(adapter, seeded):
    """The reason Task 2 exists, verified against a real catalog rather than a fixture."""
    _dsn, schema = seeded
    by_name = {i.name: i for i in adapter.fetch_indexes((schema,), frozenset({"orders"}))["orders"]}

    assert by_name["idx_plain"].columns == ("status", "created_at")
    assert by_name["idx_plain"].is_partial is False
    assert by_name["idx_plain"].has_expressions is False

    assert by_name["idx_open"].is_partial is True
    assert "shipped_at IS NULL" in (by_name["idx_open"].predicate or "")

    # The row the shipped statement silently dropped.
    assert by_name["idx_lower_note"].has_expressions is True
    assert "lower(note)" in (by_name["idx_lower_note"].definition or "")

    assert by_name["orders_pkey"].is_primary is True


def test_the_session_really_is_read_only(adapter, seeded):
    """Invariant 2, against a real server: the session must refuse a write."""
    import psycopg

    _dsn, schema = seeded
    with pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
        adapter._query(f"CREATE TABLE {schema}.should_not_exist (x int)", ())
```

- [ ] **Step 4: Run it**

```bash
docker compose -f tests/integration/docker-compose.yml up -d
uv run pytest -m integration -v
```

Expected: all pass. If `test_indexes_statement_reads_partial_and_expression_metadata` fails, Task 2's SQL is wrong in a way no fixture could reveal — that is precisely what this task exists to find, so report the actual failure rather than adjusting the assertion.

Then confirm the default suite is unaffected:

```bash
docker compose -f tests/integration/docker-compose.yml down
uv run pytest -q
```

Expected: the same count as before this task, no skips, no errors.

- [ ] **Step 5: Document it**

Add to `CONTRIBUTING.md` after the four-checks section:

```markdown
## Integration tests (optional)

`advise`'s introspection SQL is only checked for drift by the default suite. To run it
against a real Postgres:

```bash
docker compose -f tests/integration/docker-compose.yml up -d
uv run pytest -m integration
docker compose -f tests/integration/docker-compose.yml down
```

These are deselected by default, so `uv run pytest` stays green without Docker. Point them
at your own server with `SQLQUALITY_TEST_DSN`. They need the `postgres` extra
(`uv sync --extra postgres`).
```

- [ ] **Step 6: Gates and commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ruff format --check . && uv run mypy src/sqlquality && uv run pytest -q
git add tests/integration pyproject.toml CONTRIBUTING.md
git commit -m "test(advise): execute the introspection SQL against a real postgres"
```

---

### Task 7: An end-to-end run, and narrower README limitations

**Files:**
- Create: `tests/integration/test_advise_live.py`
- Modify: `README.md` (the expression-index and ADV003 limitations)

**Interfaces:**
- Consumes: the `seeded` fixture (Task 6); the `advise` CLI.

- [ ] **Step 1: Write the failing test**

`tests/integration/test_advise_live.py`:

```python
"""One whole `advise` run against a real database.

Every other test stubs the querier. This is the only path that exercises resolve_connection
-> connect -> six statements -> ingest -> aggregate -> propose -> render as one piece.
"""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from sqlquality.cli import app

pytestmark = pytest.mark.integration
runner = CliRunner()


def test_advise_end_to_end(seeded, tmp_path):
    dsn, schema = seeded
    md = tmp_path / "report.md"
    ddl = tmp_path / "proposals.sql"
    result = runner.invoke(
        app,
        ["advise", "--dsn", dsn, "--schema", schema, "--json",
         "--markdown", str(md), "--ddl", str(ddl)],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)

    assert payload["engine"] == "postgres"
    assert payload["redacted"] is True
    assert payload["analyzed"]["query_groups"] > 0
    assert payload["degraded"] == []
    assert md.read_text(encoding="utf-8").startswith("# sqlquality advise")
    assert "REVIEW BEFORE RUNNING" in ddl.read_text(encoding="utf-8")


def test_advise_does_not_leak_a_literal_from_a_real_server(seeded, tmp_path):
    """The redaction guarantee, against real pg_stat_statements rather than a fixture.

    The seeded workload filters on the literal 'paid'. pg_stat_statements normalises it to
    $1, but a run with --keep-literals proves the surfaces would carry it if we let them.
    """
    dsn, schema = seeded
    md = tmp_path / "report.md"
    result = runner.invoke(
        app, ["advise", "--dsn", dsn, "--schema", schema, "--json", "--markdown", str(md)]
    )
    assert result.exit_code == 0, result.output
    assert "'paid'" not in result.stdout
    assert "'paid'" not in md.read_text(encoding="utf-8")


def test_advise_dry_run_needs_no_server(tmp_path):
    """The audit path must not depend on anything being reachable."""
    result = runner.invoke(app, ["advise", "--engine", "postgres", "--dry-run"])
    assert result.exit_code == 0
    assert "pg_stat_statements" in result.stdout
```

- [ ] **Step 2: Run it**

```bash
docker compose -f tests/integration/docker-compose.yml up -d
uv run pytest -m integration -v
```

Expected: all pass. `payload["degraded"] == []` is the sharp one — it asserts every capability succeeded against a real server with a superuser role.

- [ ] **Step 3: Narrow the README's limitations**

Two limitations were written when the catalog query could not see this metadata. Replace the expression-index bullet with:

```markdown
- **Expression indexes are read but not matched.** `advise` now sees that an index on
  `lower(status)` exists and names it in the proposal's evidence, but it cannot tell whether
  that index already serves a lookup on `status` — so it proposes and says so, rather than
  suppressing or ignoring. Confirm before applying.
```

and the ADV003 bullet with:

```markdown
- **ADV003 only compares plain indexes.** A pair where either index carries a `WHERE`
  predicate or an indexed expression is skipped entirely rather than proposed at lower
  confidence: a partial index exists to serve a subset, so recommending its removal is
  likely wrong rather than merely uncertain. Plain pairs are reported at HIGH.
```

Also add:

```markdown
- **A partial index does not suppress a proposal.** `idx ON orders(status) WHERE
  shipped_at IS NULL` does not serve `WHERE status = $1`, so it is not treated as covering
  a candidate index — it is named in the evidence instead.
```

- [ ] **Step 4: Gates and commit**

```bash
uv run ruff format . && uv run ruff check . && uv run ruff format --check . && uv run mypy src/sqlquality && uv run pytest -q
git add tests/integration/test_advise_live.py README.md
git commit -m "test(advise): end-to-end run against a real postgres; narrow two limitations"
```

---

## Self-Review

**Coverage of the recorded items.** Every Batch-1 item from the ledger maps to a task: `secrets.py` extraction → Task 1; expression-index blindness → Tasks 2 and 3; ADV003 partial-predicate blindness → Task 4; `fingerprints` redundancy and `star_tables` regex churn → Task 5; integration test → Tasks 6 and 7; the two README limitations that those fixes narrow → Task 7.

Deliberately **not** in this plan, and staying in the ledger for Batch 2: join-key and grouping-column proposals; `DECLARE`/`COPY` unwrapping; multi-schema `(schema, table)` keying. Each changes what `advise` *says*, not whether what it says is trustworthy, so they belong after this.

**Type consistency.** `PgIndex` gains `is_partial`, `predicate`, `has_expressions`, `definition` in Task 2 and every later task reads exactly those names. `_covered` keeps `(candidate, existing) -> str | None` throughout. `clamp_timeout_ms` takes keyword-only `minimum`/`maximum` in Task 1 and is called with `MIN_TIMEOUT_S`/`MAX_TIMEOUT_S`, which already live in `workload/base.py` and are imported by both the CLI and the adapter. Passing literals there would break `test_the_timeout_bounds_have_a_single_definition`, which asserts `3600` never appears in `postgres.py`'s source — the guard added when the duplicated bounds were first found. `ColumnUsage.fingerprints` stops being a constructor argument in Task 5, and Task 5's Step 4 sweeps the call sites.

**Known risks for the implementer.**
1. Task 2's twelve-column unpack is order-sensitive and a transposed pair would be invisible to a fixture test that uses the same wrong order. Task 6's live test is the real check — if the two disagree, the live one is right.
2. `pg_get_expr(ix.indpred, ix.indrelid)` returns the predicate with Postgres's own parenthesisation, so assert with `in` rather than equality on anything but a fixture.
3. Task 5's `TypeError` assertion depends on `@dataclass(frozen=True)` rejecting unknown keywords, which it does — but if `ColumnUsage` ever gains `**kwargs` handling the test silently stops discriminating.
