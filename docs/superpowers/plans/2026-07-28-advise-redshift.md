# Advise Redshift Adapter Implementation Plan (Batch 3b)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `sqlquality advise --engine redshift` reads Redshift query history and catalog metadata over a read-only connection and proposes **distribution and sort key** changes — not indexes, which Redshift does not have.

**Architecture:** A new `RedshiftWorkloadAdapter` implementing the existing `WorkloadAdapter` ABC. Everything above the adapter is reused unchanged: ingest, redaction, the `Relation`-keyed rollup, the proposal collapse, the report renderers, and the dbt enrichment layer from Batch 3a. Only the four things an adapter owns are new — its driver session, its introspection statements, its proposal rules, and its DDL syntax.

**Tech Stack:** Python 3.11+, psycopg 3 (Redshift speaks the PostgreSQL wire protocol), sqlglot 30.x with its `redshift` dialect, typer, rich, pytest.

## Global Constraints

Every task's requirements implicitly include this section.

- **All four CI gates before every commit:** `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy src/sqlquality`, `uv run pytest -q`. The `no-extras` and `highest-deps` jobs must also stay green.
- **`uv run pytest` must report `N passed, M deselected`, never `skipped`.**
- **sqlquality never executes user SQL.** `advise` opens a read-only session with a statement timeout and runs only the statements the adapter declares. Generated DDL goes to a file for human review.
- **No credential and no user literal reaches stdout, stderr, a report, or an exception message.** Reuse `workload/secrets.py`'s `secrets_for`/`scrub` — do not write a second credential path.
- **One missing grant costs one capability, never the run.** Record it in `self.degraded` and continue.
- **Confidence never overstates evidence.** A check that could not run is disclosed, not assumed.
- **No existing Postgres behaviour may change.** The `postgres` engine's output must be byte-identical; a shared helper may be extracted, but not altered in effect.
- **A test that passes with the production change reverted is not a test.** Mutate the line each test claims to pin and report the failure. Where a test asserts over a set, it must discriminate for **every** member — this feature has produced **eight** findings of that shape, most recently a wiring call that could be deleted with all 665 tests green.

## The honesty constraint that shapes this whole plan

**None of the catalog SQL in this plan can be executed against a real Redshift cluster during development.** There is no Redshift container, and its `svv_*` / `sys_*` / `stl_*` catalogs do not exist in Postgres. That matters more here than usual: on this feature's three predecessor branches, *every* silent-suppression bug was found by running against a live database, and none by a fixture — `reltuples = -1` suppressing all proposals, redaction dismembering `$N`, a `toplevel` filter producing confidently-wrong advice, an `IS NOT NULL` polarity inversion that shipped to PyPI.

So this plan buys back what verification it can, and is explicit about the rest:

1. **The connection path IS testable**, because Redshift speaks the PostgreSQL wire protocol through psycopg. The read-only session, the statement timeout, secret scrubbing and per-capability degradation are all exercised against the existing Postgres container.
2. **Every statement is syntax-validated** by parsing it with sqlglot's `redshift` dialect in a unit test. That cannot catch a wrong column name, but it catches the malformed-statement class, which is otherwise invisible until a user runs it.
3. **Every statement's select-list arity is pinned against its Python unpacking.** Batch 2 shipped a column-count mismatch that no fixture caught; a test comparing the two is cheap and closes it.
4. **The limitation is documented prominently, not buried** — README and `--dry-run` both say the introspection SQL has not been executed against a live cluster, and invite the user to run `--dry-run` output by hand.

**Column names and catalog shapes in this plan come from AWS documentation, not from observation.** Encode them as module-level constants with that provenance in a comment, and make every row unpacking defensive in the way `postgres.py` already is (`_as_int`, `_as_float`, and the `reltuples`-style sentinel translation). Where this plan states a column name, treat it as *to be confirmed by the first user with a cluster*, and make the failure mode a recorded `degraded` entry rather than a traceback.

## Why the rules are not the Postgres rules renamed

**Redshift has no indexes.** ADV001–ADV008 are inapplicable and must not be inherited. Redshift's physical-design levers are different, and so is the blast radius:

| lever | what it does | how bad is the DDL |
|---|---|---|
| SORTKEY | zone maps let a scan skip whole blocks for range/equality predicates | `ALTER TABLE … ALTER SORTKEY` rewrites the table |
| DISTKEY | co-locates join keys so a join needs no redistribution | `ALTER TABLE … ALTER DISTKEY` rewrites the table |
| DISTSTYLE ALL | replicates a small dimension to every node | rewrite, and costs storage per node |
| VACUUM / ANALYZE | reclaims sort order and refreshes statistics | no rewrite, but VACUUM is heavy |

**Every one of the first three rewrites the entire table.** That is categorically more dangerous than `CREATE INDEX CONCURRENTLY`, and the generated script must say so far more loudly than the Postgres one does. This is the single most important difference between the two adapters.

The existing offline `RedshiftAdapter` (a `PerfAdapter`, unrelated to `WorkloadAdapter`) already encodes the domain vocabulary this plan needs — `DS_BCAST_INNER`, `DS_DIST_BOTH`, `DS_DIST_ALL_INNER` as redistribution markers, and `keys.py`'s `join_key_columns` / `filter_columns`. Read both before starting; reuse the concepts and the finding codes' reasoning, but note that adapter works from **one SQL file**, whereas this one works from an **aggregated workload**.

## Capabilities

Mirror `postgres.py`'s structure exactly — a `SQL: dict[str, str]`, a `_HINTS` dict, and `_run()` recording degradation rather than raising.

| capability | source | purpose |
|---|---|---|
| `CAP_WORKLOAD` | `sys_query_history` | query text, execution time, call counts |
| `CAP_SCHEMA` | `svv_redshift_columns` | `{schema: {table: {column: type}}}` for `qualify()` |
| `CAP_TABLE_FACTS` | `svv_table_info` | rows, size, `unsorted`, `stats_off`, `diststyle`, `sortkey1`, `skew_rows` |
| `CAP_ADVISOR` | `svv_alter_table_recommendations` | Redshift's *own* recommendations |

There is deliberately **no** `CAP_NDV` and **no** `CAP_INDEXES`: Redshift exposes no per-column distinct-value statistic comparable to `pg_stats.n_distinct`, and it has no indexes. Do not fabricate either — a rule that needs NDV must disclose that it could not check selectivity, exactly as ADV001 now does on Postgres.

**`sys_query_history` is the current, documented view and is what this plan targets.** The older `stl_query` + `svl_statementtext` pair has a shape hazard worth knowing about even though we are not using it: `svl_statementtext` **chunks query text across multiple rows** ordered by a `sequence` column, so reading it without reassembly silently truncates every query at ~200 characters. If a later task needs a fallback for older clusters, that reassembly is the thing to get right, and a truncated-text test is the thing that would catch it.

---

### Task 1: the adapter skeleton, its statements, and `--dry-run`

**Files:**
- Create: `src/sqlquality/workload/redshift.py`
- Modify: `src/sqlquality/workload/__init__.py` (register the engine)
- Create: `tests/test_workload_redshift.py`

**Interfaces:**
- Produces: `RedshiftWorkloadAdapter` with `engine = "redshift"`, a `SQL` dict over the four
  capabilities above, `_HINTS` for each, `introspection_sql()`, and `_run()`. Registered in
  `_ADAPTERS` so `get_workload_adapter("redshift")` returns it.
- Consumes: `WorkloadAdapter`, `IntrospectionStatement`, `Querier`, `MIN_TIMEOUT_S`/`MAX_TIMEOUT_S`.

`connect()`, `fetch_workload`, `fetch_schema`, `fetch_table_facts`, `propose` and `render_ddl` may
raise `NotImplementedError` in this task **only if** a test pins that they do — an adapter that
half-exists and returns empty results silently is worse than one that says it is unfinished. Later
tasks fill them in.

- [ ] **Step 1: Write the failing tests**

```python
import sqlglot
import pytest

from sqlquality.workload import get_workload_adapter
from sqlquality.workload.redshift import (
    CAP_ADVISOR,
    CAP_SCHEMA,
    CAP_TABLE_FACTS,
    CAP_WORKLOAD,
    RedshiftWorkloadAdapter,
)

EXPECTED_CAPABILITIES = {CAP_WORKLOAD, CAP_SCHEMA, CAP_TABLE_FACTS, CAP_ADVISOR}


def test_registry_returns_the_redshift_adapter():
    adapter = get_workload_adapter("redshift")
    assert adapter.engine == "redshift"


def test_every_capability_has_a_statement_and_a_hint():
    statements = RedshiftWorkloadAdapter().introspection_sql()
    assert {s.capability for s in statements} == EXPECTED_CAPABILITIES
    for statement in statements:
        assert statement.sql.strip()
        assert statement.privilege_hint.strip()


@pytest.mark.parametrize("capability", sorted(EXPECTED_CAPABILITIES))
def test_every_statement_parses_as_redshift_sql(capability):
    """Syntax validation is the one correctness check available without a cluster.

    The catalog SQL in this adapter cannot be executed during development — there is no
    Redshift container and `svv_*`/`sys_*` do not exist in Postgres. Parsing each statement
    with sqlglot's redshift dialect cannot catch a wrong column name, but it catches a
    malformed statement, which would otherwise be invisible until a user ran it.
    """
    sql = RedshiftWorkloadAdapter.SQL[capability]
    # `%s` placeholders are libpq's, not SQL — sqlglot cannot parse them, so they become
    # bind markers for the purposes of this check.
    parsed = sqlglot.parse_one(sql.replace("%s", "?"), dialect="redshift")
    assert parsed is not None


@pytest.mark.parametrize("capability", sorted(EXPECTED_CAPABILITIES))
def test_no_statement_writes(capability):
    """Same guard the Postgres adapter carries, for the same reason."""
    forbidden = ("insert", "update", "delete", "create", "drop", "alter", "truncate",
                 "grant", "revoke", "vacuum", "analyze")
    lowered = RedshiftWorkloadAdapter.SQL[capability].lower()
    import re
    found = {verb for verb in forbidden if re.search(rf"\b{verb}\b", lowered)}
    assert not found, f"{capability} contains write verb(s): {sorted(found)}"


def test_there_is_no_ndv_or_index_capability():
    """Redshift exposes no `pg_stats.n_distinct` equivalent and has no indexes.

    Declaring either capability would invite a rule to assume evidence that cannot exist.
    """
    capabilities = {s.capability for s in RedshiftWorkloadAdapter().introspection_sql()}
    assert not any("ndv" in c or "index" in c for c in capabilities)


def test_unimplemented_methods_say_so_rather_than_returning_empty():
    """A half-built adapter that returns nothing looks exactly like a healthy cluster with
    no workload, which is the worst possible failure mode for this command."""
    adapter = RedshiftWorkloadAdapter()
    with pytest.raises(NotImplementedError):
        adapter.fetch_schema(("public",))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_workload_redshift.py -x -q`

Expected: FAIL — `ModuleNotFoundError: No module named 'sqlquality.workload.redshift'`.

- [ ] **Step 3: Write the module**

Mirror `postgres.py`'s shape. Each statement gets a comment recording that its column names come
from AWS documentation and are **unconfirmed against a live cluster**. Start from:

```python
CAP_WORKLOAD = "workload"
CAP_SCHEMA = "schema"
CAP_TABLE_FACTS = "table_facts"
CAP_ADVISOR = "advisor"
```

and statements of this shape — adjust only if you find a documented reason to, and say what:

```python
        # Column names below come from AWS's Redshift system-view documentation and have NOT
        # been executed against a live cluster (see the module docstring). Every consumer
        # unpacks defensively and a denied or malformed statement is recorded in `degraded`
        # rather than raised, so a wrong name costs one capability instead of the run.
        CAP_WORKLOAD: """
            SELECT query_text, elapsed_time
            FROM sys_query_history
            WHERE database_name = current_database()
              AND status = 'success'
            ORDER BY elapsed_time DESC
            LIMIT %s
        """,
```

`_HINTS` must name the real grant each view needs — `sys_query_history` shows only the current
user's queries without `SYSLOG ACCESS UNRESTRICTED`, which is exactly the kind of partial-result
trap the Postgres hints already warn about for `pg_stats`.

- [ ] **Step 4: Register the engine**

Add `"redshift": RedshiftWorkloadAdapter` to `_ADAPTERS` in `workload/__init__.py`.

- [ ] **Step 5: Run the tests, then the suite and all four gates.**

- [ ] **Step 6: Prove the syntax check discriminates**

Introduce a deliberate syntax error into one statement (`SELECT FROM WHERE`). Expected: that
capability's `test_every_statement_parses_as_redshift_sql` case FAILS **and the other three still
pass**, so the parametrisation discriminates per member rather than as a block. Restore, and report
which case failed.

- [ ] **Step 7: Commit**

```bash
git add src/sqlquality/workload/redshift.py src/sqlquality/workload/__init__.py \
        tests/test_workload_redshift.py
git commit -m "feat(advise): a Redshift workload adapter skeleton with syntax-checked statements"
```

---

### Task 2: connect over the Postgres wire protocol, proven against a real server

**Files:**
- Modify: `src/sqlquality/workload/redshift.py`
- Modify: `tests/test_workload_redshift.py`
- Modify: `tests/integration/` (a live connection test)

**Interfaces:**
- Produces: `RedshiftWorkloadAdapter.connect(params, timeout_s)`.

**This is the one part of the adapter that can be verified for real.** Redshift speaks the
PostgreSQL wire protocol, so `connect()` can and must be exercised against the existing
`postgres:16` container: the read-only session, the clamped statement timeout, secret scrubbing on
a bad password, and a clear install hint when psycopg is missing.

Extract the shared logic rather than copying it — `postgres.py`'s `connect` already does exactly
this, and two copies of a credential-handling path is how one of them drifts. **But the Postgres
adapter's behaviour must not change**: prove that by running the existing Postgres tests unchanged.

Redshift-specific difference to preserve: Redshift does not support `SET default_transaction_read_only`
in all configurations. Establish read-only intent the way that works there, and if the statement is
refused, record it as a degradation and say plainly in the hint that the session could not be
proven read-only — do **not** silently continue as if it had succeeded, because "we never write" is
this tool's central promise.

- [ ] **Step 1: Write the failing tests** — including a live one:

```python
@pytest.mark.integration
def test_redshift_adapter_connects_over_the_postgres_wire_protocol(live_dsn):
    """Redshift speaks the PostgreSQL protocol, so the session setup is genuinely testable.

    The catalog statements are not — `svv_*` does not exist here — so this test deliberately
    covers connect() only, and asserts nothing about introspection.
    """
    adapter = RedshiftWorkloadAdapter()
    params = ConnectionParams(engine="redshift", dsn=live_dsn, fields={}, source="test")
    adapter.connect(params, timeout_s=30)
    assert adapter._query is not None
    rows = adapter._query("SELECT 1", ())
    assert rows == [(1,)]


@pytest.mark.integration
def test_a_wrong_password_leaks_nothing(live_dsn):
    adapter = RedshiftWorkloadAdapter()
    bad = live_dsn.replace(":sqlquality@", ":wr0ng-p4ss@")
    params = ConnectionParams(engine="redshift", dsn=bad, fields={}, source="test")
    with pytest.raises(ConnectionError) as exc:
        adapter.connect(params, timeout_s=5)
    assert "wr0ng-p4ss" not in str(exc.value)
    assert "wr0ng-p4ss" not in repr(exc.value)
```

- [ ] **Step 2–5:** run red, extract the shared session helper, implement, run green, and confirm
      `tests/test_workload_postgres.py` and `tests/test_workload_secrets.py` pass **unchanged**.
- [ ] **Step 6: Prove the scrubbing test discriminates** by removing the `scrub(...)` call and
      confirming the wrong-password test fails with the password visible. Restore, report.
- [ ] **Step 7: Commit.**

---

### Task 3: workload and schema fetch

**Files:** modify `redshift.py`, `tests/test_workload_redshift.py`.

**Interfaces:** `fetch_workload(since, limit) -> WorkloadFetch`, `fetch_schema(schemas) -> dict`.

Two things to get right, both of which the Postgres adapter learned the hard way:

- **The window description must be honest.** `sys_query_history` *does* carry timestamps, unlike
  `pg_stat_statements` — so unlike Postgres, `--since` can be honoured. If you honour it, say so in
  `window_description`; if you do not, say that instead. Do not describe a window you did not apply.
- **`sys_query_history` returns one row per execution, not per normalised statement.** Postgres's
  `pg_stat_statements` pre-aggregates by fingerprint; Redshift does not. So `calls` is 1 per row and
  the aggregation happens in `ingest()` via fingerprinting — which already sums `calls` and
  `total_time_ms` per fingerprint. Confirm that is what happens rather than assuming, and add a test
  with two executions of the same statement asserting they collapse to one `QueryStat` with
  `calls == 2`.

- [ ] Tests, red, implement, green, mutation-prove the aggregation claim, commit.

---

### Task 4: table facts, and the sentinels Redshift uses

**Files:** modify `redshift.py`, `tests/test_workload_redshift.py`.

**Interfaces:** `fetch_table_facts(schemas, relations) -> dict[Relation, TableFacts]`, plus a
Redshift-specific `RedshiftTableFacts` (or extra fields carried in the adapter) for `unsorted`,
`stats_off`, `diststyle`, `sortkey1` and `skew_rows`, which `TableFacts` does not model.

`TableFacts` is engine-neutral and must stay so — do not add Redshift columns to it. Hold the
Redshift-specific physical facts in the adapter, keyed by `Relation`, the way `postgres.py` holds
`PgIndex`.

**The sentinel lesson applies here.** Batch 1 lost a day to `pg_class.reltuples = -1` meaning "never
analysed" being read as "tiny table", which silently suppressed every proposal. Redshift's
`svv_table_info` has its own version of this: `tbl_rows` and `size` are only meaningful once the
table has been analysed, and `stats_off` is precisely the column that says how stale they are.
**Translate unknown to `None` at the boundary** and let the rules disclose it, rather than letting a
zero or a `-1` read as a fact. Add a test per sentinel you handle.

- [ ] Tests, red, implement, green, mutation-prove each sentinel translation, commit.

---

### Task 5: ADV101 (SORTKEY) and ADV102 (DISTKEY)

**Files:** modify `redshift.py`, create `tests/test_workload_redshift_rules.py`.

**Interfaces:**
- `propose_sortkey(usage, facts, physical, *, min_cost_share) -> list[Proposal]` — `"ADV101"`
- `propose_distkey(usage, facts, physical, *, min_cost_share) -> list[Proposal]` — `"ADV102"`

**ADV101 — SORTKEY from hot range and equality predicates.** Redshift's zone maps store min/max per
block, so a scan can skip blocks entirely when the predicate column is the sort key. The candidate
is the hottest `RANGE`/`EQUALITY` column, and time-series columns are the canonical win.

Suppress when the table's existing `sortkey1` already **is** that column — the equivalent of
`_covered`, and the same silent-suppression trap: if the sort key could not be read, the claim
"the table is not sorted on this column" is unknowable and confidence caps at LOW with the gap
stated.

**ADV102 — DISTKEY from hot join keys.** A join whose two sides are not distributed on the join key
forces redistribution — `DS_BCAST_INNER` or the heavier `DS_DIST_BOTH`, which the offline
`RedshiftAdapter` already names. The candidate is the hottest `JOIN`-role column.

**Both cap at MEDIUM, and there is deliberately no HIGH branch.** Follow ADV008's precedent and say
so in the docstring so a later reader does not add the missing rung. The reasons are specific:
- Redshift exposes no per-column NDV, so distribution **skew** — the thing that makes a DISTKEY
  choice good or catastrophic — cannot be predicted. `svv_table_info.skew_rows` describes the
  *current* distribution, not the proposed one.
- A SORTKEY change is only worth its rewrite if the predicate is selective, and selectivity is
  exactly what cannot be measured without NDV.

Claiming HIGH would assert something about data distribution this tool cannot see, while
recommending DDL that **rewrites the entire table**.

- [ ] Tests per rule, red, implement, green, mutate each confidence rung and each suppression gate
      independently, commit.

---

### Task 6: ADV103 (DISTSTYLE ALL), ADV104 (VACUUM/ANALYZE), ADV105 (Redshift's own advice)

**Files:** modify `redshift.py`, modify `tests/test_workload_redshift_rules.py`.

- **ADV103** — a small, frequently-joined dimension is a candidate for `DISTSTYLE ALL`, which
  replicates it to every node and removes redistribution entirely. Gate on a row-count **ceiling**
  (the inverse of Postgres's floor) and on the table actually being joined in the workload. Disclose
  the cost: storage is multiplied by the node count, and every write is amplified.
- **ADV104** — `svv_table_info.unsorted` above a threshold, or `stats_off` above one, means the
  table needs `VACUUM` or `ANALYZE`. This is the one Redshift rule whose remediation does **not**
  rewrite the table, so it is the only one that can reasonably reach HIGH — and it is also the
  cheapest thing a user can act on, so it belongs near the top of a report.
- **ADV105** — surface `svv_alter_table_recommendations`, which is **Redshift Advisor's own
  output**. Present it as the engine's opinion, clearly attributed, alongside ours. Where Advisor
  and one of our rules agree, say so — that agreement is the strongest evidence this adapter can
  produce, since it is the only signal in the whole plan that comes from the cluster itself rather
  than from our inference. Never present an Advisor row as our own conclusion.

- [ ] Tests per rule, red, implement, green, mutation-prove, commit.

---

### Task 7: `render_ddl` — the dangerous-DDL header, and wiring

**Files:** modify `redshift.py`, modify `tests/test_workload_redshift_rules.py`, possibly `cli.py`.

**Interfaces:** `RedshiftWorkloadAdapter.render_ddl(proposals) -> str`, `propose(...)` composing all
five rules and returning them collapsed and ranked.

**The header must be much louder than the Postgres one.** Every ADV101/102/103 statement rewrites
the whole table: it holds a lock, consumes disk for a full copy, and on a large table takes hours.
The Postgres header recommends `CONCURRENTLY`; there is no such escape here. Say plainly that these
are table rewrites, that they should be scheduled, and that `ADV104`'s `VACUUM`/`ANALYZE` are the
only statements in the file that are not.

Reuse, do not reimplement: the proposal collapse, `_ranking_key` (now public on the ABC after Batch
3a's F7), `cost_share_of`, and `_is_fully_commented`'s protection that no emitted line is ever
executable-looking. Confirm by test that a hostile identifier cannot produce a bare executable line
in a Redshift script either.

**Check the dbt enrichment interaction.** Batch 3a's `enrich_proposals` keys on a `CREATE INDEX`
DDL prefix, so it will not touch Redshift's `ALTER TABLE` statements — meaning a dbt-managed
Redshift model currently gets a table-rewrite proposal with **no** warning that `dbt run` will undo
it. Decide deliberately: either extend the enrichment to recognise Redshift's statements, or
document that dbt enrichment covers Postgres index proposals only. State the reasoning either way,
and note that Batch 3a's ADV302 already warns about `adapter_type` mismatches, which is adjacent
but not the same thing.

- [ ] Tests, red, implement, green, mutation-prove the header and the collapse reuse, commit.

---

### Task 8: prove what can be proven, and document what cannot

**Files:** `tests/integration/`, `README.md`, `CHANGELOG.md`, the design spec.

- [ ] **Step 1: Live tests** for everything reachable — `connect()` against the Postgres container,
      the read-only session, the timeout clamp, secret scrubbing, and `--dry-run` printing all four
      statements with no connection at all.
- [ ] **Step 2: Confirm the whole default suite stays green** with zero skips, and that the Postgres
      adapter's output is unchanged (run its tests unmodified).
- [ ] **Step 3: Document the verification gap prominently.** README must say, in the `advise`
      section rather than a footnote, that Redshift's introspection SQL is syntax-checked and
      shape-tested but **has not been executed against a live cluster**, that column names come from
      AWS documentation, and that `advise --engine redshift --dry-run` prints every statement for a
      user to run by hand and report back. A user pointing this at a production cluster deserves to
      know which parts are proven.
- [ ] **Step 4: Document the rules** — ADV101–105 in the rule table, with the table-rewrite warning
      attached to the three that rewrite, and the Advisor attribution for ADV105.
- [ ] **Step 5: CHANGELOG and a spec deviation** recording that Redshift declares no NDV or index
      capability, why HIGH is unreachable for ADV101/102/103, and that the connection path is
      verified while the catalog path is not.
- [ ] **Step 6: All four gates plus the integration suite, then commit.**

---

## Self-Review

**Spec coverage.** ADV101–105 map to Tasks 5 and 6; the adapter contract to Tasks 1–4; DDL and
wiring to Task 7; proof and docs to Task 8. Snowflake remains out of scope pending an account.

**Placeholder scan.** Tasks 3–6 give test *intent* and the specific mutation to run rather than
full test bodies, deliberately: the column names those tests assert on are unverified, so pinning
exact fixtures in the plan would harden guesses into requirements. Each of those tasks names what
must be pinned and what must be mutated; the implementer writes fixtures matching whatever the
statements end up selecting. Tasks 1 and 2 — where the assertions are about *our* code rather than
Redshift's schema — carry real test code.

**Type consistency.** `Relation`, `TableFacts`, `Proposal`, `Aggregation`, `Workload`,
`ConnectionParams`, `IntrospectionStatement` and `Querier` are all pre-existing and used with
current field names. `TableFacts` is deliberately *not* extended; Redshift's physical facts live in
an adapter-local structure, mirroring `PgIndex`.

**The biggest risk this plan accepts.** The catalog SQL may simply be wrong — a misremembered column
name produces an adapter that degrades on every capability and proposes nothing. The mitigations are
that each failure is a recorded `degraded` entry naming the statement rather than a traceback, that
`--dry-run` lets a user check every statement before connecting, and that the README says plainly
which parts are unverified. That is honest, but it is weaker than every predecessor batch, and the
first user with a cluster should be treated as part of the verification loop rather than as a
consumer of a finished feature.
