# `sqlquality verify` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A new offline command, `sqlquality verify before.json after.json`, that diffs two `advise --json` artifacts and reports, per proposal, whether it was applied and whether the queries that justified it actually got faster.

**Architecture:** All diff logic lives in one new pure module, `src/sqlquality/verify.py`, operating on two payload dicts — no I/O, no database, no engine knowledge. `advise`'s payload gains the four things the diff needs, with each adapter exposing its own physical state through one new ABC method so the payload stays engine-neutral. `verify` joins `check` on the offline side of the line; `advise` remains the only command that opens a connection.

**Tech Stack:** Python 3.11+, typer, rich, pytest. No new dependencies — deliberately, since `verify` must run in the `no-extras` CI job.

**Spec:** `docs/superpowers/specs/2026-08-03-advise-verify-design.md`. Read it before Task 1; it carries the reasoning behind every decision below, and the reasoning is what will get re-litigated.

## Global Constraints

Every task's requirements implicitly include this section.

- **All four CI gates before every commit:** `uv run ruff check .`, `uv run ruff format --check .`, `uv run mypy src/sqlquality`, `uv run pytest -q`. The `no-extras`, `highest-deps` and `integration` jobs must also stay green.
- **`uv run pytest` must report `N passed, M deselected`, never `skipped`.**
- **`verify` never connects to a database and never writes to one.** It reads two JSON files. If a task needs a connection to do its job, that task is in the wrong place.
- **`verify` imports nothing outside the base dependency set.** psycopg is an optional extra; `verify.py` must not import it, directly or transitively.
- **`verify` refuses rather than guesses.** Every incomparable input exits 2 with the reason. A verdict must never be derived from absent data.
- **Confidence never overstates evidence.** `verify` grades every verdict, and a comparison it cannot make cleanly is graded down with the gap named — the same discipline `advise`'s own rules follow.
- **`cost_share` is context, never the verdict.** Mean time per call is the improvement metric. See the spec's decision 4 for why; a task that compares `cost_share` to decide `improved` has misread it.
- **No existing `advise` behaviour changes** beyond the payload additions this plan specifies. `advise`'s proposals, confidences, DDL and stderr must be unchanged; its tests pass unmodified.
- **A test that passes with the production change reverted is not a test.** Mutate the line each test claims to pin and report the failure. Where a test asserts over a *set*, it must discriminate for **every** member — this codebase has produced **eight** findings of that shape, and a `propose()` whose entire wiring could be deleted with the suite green appeared **three** times.
- Public identifiers get a docstring saying *why*, matching the surrounding module's density.

## A correction to the spec, made here

The spec says `physical_state` records `relkind` so ADV301's application ("the relation is no longer a view") is observable. Grounding the plan showed that is not available without new SQL: `postgres.py`'s `CAP_TABLE_FACTS` filters `WHERE c.relkind = 'r'`, so a view never appears in table facts at all.

That absence is itself the signal, and a better one — it needs no new query. `physical_state` therefore records **`is_ordinary_table: bool`**, true when the relation appeared in the table-facts result for the relations we asked about. On Postgres this is reliable: we requested that exact relation, and only ordinary tables come back.

On Redshift the same field carries a known ambiguity already recorded in `RedshiftTableFacts`' docstring — `svv_table_info` also omits genuinely *empty* tables, so `is_ordinary_table: false` there means "not a populated local table", which is weaker. Task 2 must state that in the field's docstring rather than let a reader assume parity between engines.

## File structure

| file | responsibility |
|---|---|
| `src/sqlquality/verify.py` | **new.** The whole diff: proposal keys, group matching, window classification, verdicts. Pure functions over two dicts. |
| `src/sqlquality/models.py` | **modify.** `VerifyOutcome` enum, `ProposalVerdict` dataclass, `WindowRelation` enum. |
| `src/sqlquality/workload/base.py` | **modify.** One new ABC method, `physical_state`. |
| `src/sqlquality/workload/postgres.py` | **modify.** Implement `physical_state`; expose structured window fields. |
| `src/sqlquality/workload/redshift.py` | **modify.** Same, for its own levers. |
| `src/sqlquality/report.py` | **modify.** `advise_payload` gains four keys; new `verify_payload` and `render_verify_markdown`. |
| `src/sqlquality/cli.py` | **modify.** The `verify` command. |
| `tests/test_verify.py` | **new.** The core, unit-tested with dicts. |
| `tests/test_verify_cli.py` | **new.** Command wiring, exit codes, output surfaces. |
| `tests/fixtures/verify/` | **new.** Golden before/after payload pairs, one per outcome. |
| `tests/integration/test_verify_live.py` | **new.** The loop closed for real against live Postgres. |

`verify.py` stays one file because its parts are meaningless apart — a proposal key is only useful to the matcher, which is only useful to the verdict builder. If it passes roughly 400 lines, split the window classification out first; it is the piece with the fewest dependencies.

---

### Task 1: structured `window` in the payload

**Files:**
- Modify: `src/sqlquality/report.py` (`advise_payload`)
- Modify: `src/sqlquality/workload/base.py`, `postgres.py`, `redshift.py`
- Modify: `CHANGELOG.md`
- Test: `tests/test_report.py`, `tests/test_workload_postgres.py`, `tests/test_workload_redshift.py`

**Interfaces:**
- Produces: `WorkloadAdapter.window_facts() -> dict[str, object]` — the structured window for the payload. Default returns `{}`; each adapter overrides.
- Produces: payload key `window` as an **object**: `{"description": str, "engine": str, "stats_reset_at": str | None, "since": str | None, "limit": int}`.
- Consumes: nothing from earlier tasks.

**This is the plan's only breaking payload change.** `window` is a string today. It is acceptable because `advise` shipped in 0.3.0 hours ago and its own CHANGELOG states it has no previously-released behaviour — but it must be announced, not slipped in. A 0.3.0 artifact will be rejected by `verify` in Task 7 precisely because this key's shape identifies the payload version.

- [ ] **Step 1: Write the failing tests**

```python
def test_window_is_an_object_carrying_what_the_comparison_needs():
    """`verify` classifies two runs' windows as nested, disjoint or comparable, and cannot
    do that from a prose sentence. Each field answers one question: `stats_reset_at`
    whether Postgres's cumulative counters were cleared between runs, `since` whether a
    duration was applied, `limit` whether the window was truncated."""
    payload = advise_payload(
        [],
        Workload(stats=(), window_description="since stats reset at 2026-08-01T00:00:00"),
        Aggregation(usage=(), total_cost_ms=0.0, skipped_unqualifiable=0, tables=frozenset()),
        engine="postgres",
        redacted=True,
        degraded=[],
        window_facts={"stats_reset_at": "2026-08-01T00:00:00", "since": None, "limit": 500},
    )
    window = payload["window"]
    assert isinstance(window, dict), "a prose string cannot be compared across runs"
    assert window["description"] == "since stats reset at 2026-08-01T00:00:00"
    assert window["engine"] == "postgres"
    assert window["stats_reset_at"] == "2026-08-01T00:00:00"
    assert window["since"] is None
    assert window["limit"] == 500


def test_the_window_description_is_preserved_verbatim():
    """The prose sentence is what a human reads and it is already carefully worded — the
    structured fields are added beside it, not instead of it."""
    payload = advise_payload(
        [],
        Workload(stats=(), window_description="since stats reset at T (--since is not supported)"),
        Aggregation(usage=(), total_cost_ms=0.0, skipped_unqualifiable=0, tables=frozenset()),
        engine="postgres", redacted=True, degraded=[], window_facts={},
    )
    assert payload["window"]["description"] == (
        "since stats reset at T (--since is not supported)"
    )


def test_missing_window_facts_are_null_not_absent():
    """A key that is absent and a key that is null are different to a consumer. `verify`
    distinguishes "this engine cannot tell you" from "this field was never written", and
    only the second is a reason to reject the artifact."""
    payload = advise_payload(
        [], Workload(stats=(), window_description="w"),
        Aggregation(usage=(), total_cost_ms=0.0, skipped_unqualifiable=0, tables=frozenset()),
        engine="postgres", redacted=True, degraded=[], window_facts={},
    )
    for key in ("stats_reset_at", "since", "limit"):
        assert key in payload["window"], f"{key} must be present even when unknown"
        assert payload["window"][key] is None
```

And one per adapter, asserting the adapter reports what it actually knows:

```python
def test_postgres_reports_its_stats_reset_time_and_cannot_report_since():
    """`pg_stat_statements` is cumulative with no per-statement timestamps, so `--since`
    is never applied and must be reported as such rather than echoed back."""
    adapter = PostgresWorkloadAdapter(querier=_canned({CAP_STATS_RESET: [("2026-08-01T00:00:00",)]}))
    adapter.fetch_workload(timedelta(days=7), 500)
    facts = adapter.window_facts()
    assert facts["stats_reset_at"] == "2026-08-01T00:00:00"
    assert facts["since"] is None, "Postgres cannot honour --since; claiming it did would lie"
    assert facts["limit"] == 500


def test_redshift_reports_the_since_cutoff_it_actually_bound():
    """Unlike Postgres, `sys_query_history` carries timestamps, so `--since` is real and the
    window is comparable by construction — which is what lets `verify` grade it HIGH."""
    adapter = RedshiftWorkloadAdapter(querier=_canned({CAP_WORKLOAD: []}))
    adapter.fetch_workload(timedelta(days=7), 500)
    facts = adapter.window_facts()
    assert facts["since"] is not None
    assert facts["limit"] == 500
    assert facts["stats_reset_at"] is None, "Redshift has no cumulative-counter reset"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_report.py -x -q -k window`

Expected: FAIL — `advise_payload() got an unexpected keyword argument 'window_facts'`.

- [ ] **Step 3: Add `window_facts` to the ABC**

In `src/sqlquality/workload/base.py`, on `WorkloadAdapter`:

```python
    def window_facts(self) -> dict[str, object]:
        """Structured facts about the window `fetch_workload` just read, for the payload.

        Not abstract, and returns `{}` by default, because an adapter that knows none of
        these is a legitimate state rather than an unfinished one — the payload fills the
        gaps with `None`. What each field is for: `stats_reset_at` tells a later
        comparison whether cumulative counters were cleared between two runs (the
        difference between two independent samples and one containing the other), `since`
        whether a duration filter was genuinely applied, and `limit` whether the window
        was truncated. Reporting a `since` an engine did not apply would make an
        incomparable pair look comparable, which is worse than reporting nothing.
        """
        return {}
```

- [ ] **Step 4: Implement it on both adapters**

`postgres.py` records the reset timestamp it already reads in `fetch_workload`, and reports `since=None` always — it cannot honour one. `redshift.py` records the cutoff it bound and `stats_reset_at=None`. Both record `limit`.

Store what is needed on the instance during `fetch_workload` rather than re-querying; `window_facts()` must not issue SQL.

- [ ] **Step 5: Build the object in `advise_payload`**

```python
    window_facts = dict(window_facts or {})
    payload["window"] = {
        "description": workload.window_description,
        "engine": engine,
        # Present-but-null rather than absent: `verify` treats an absent key as a payload
        # from a version that predates this feature and refuses the artifact, while null
        # means "this engine cannot tell you", which is comparable information.
        "stats_reset_at": window_facts.get("stats_reset_at"),
        "since": window_facts.get("since"),
        "limit": window_facts.get("limit"),
    }
```

- [ ] **Step 6: Run the tests, then the full suite and all four gates**

Expected: PASS. `advise`'s markdown and terminal output are unchanged — only the JSON payload's `window` differs. Confirm by running the existing `advise` CLI tests unmodified.

- [ ] **Step 7: Prove the present-but-null distinction discriminates**

Change the payload builder to omit keys whose value is `None` (`{k: v for k, v in … if v is not None}`). Expected: `test_missing_window_facts_are_null_not_absent` FAILS. Restore, and report the mutation.

- [ ] **Step 8: Record the breaking change in the CHANGELOG**

Under a new `## [Unreleased]` heading, a `### Changed` entry stating that `advise --json`'s `window` is now an object rather than a string, why (a prose sentence cannot be compared across runs), and that artifacts from 0.3.0 are not accepted by `verify`.

- [ ] **Step 9: Commit**

```bash
git add src/sqlquality/report.py src/sqlquality/workload/ CHANGELOG.md tests/
git commit -m "feat(advise): report the workload window as structured facts, not prose"
```

---

### Task 2: `physical_state` in the payload

**Files:**
- Modify: `src/sqlquality/workload/base.py`, `postgres.py`, `redshift.py`
- Modify: `src/sqlquality/report.py`
- Test: `tests/test_workload_postgres.py`, `tests/test_workload_redshift.py`, `tests/test_report.py`

**Interfaces:**
- Produces: `WorkloadAdapter.physical_state(relations: frozenset[Relation]) -> dict[str, dict]`, keyed by the **string** `"schema.table"`.
- Produces: payload key `physical_state`.
- Consumes: `window_facts` pattern from Task 1 (same shape of ABC addition).

**Two things to get right.**

`Relation` is not JSON-serializable. Key the dict by `str(relation)`, which is already `"schema.table"`. A `TypeError` here fires *after* the entire analysis has run, and this project has shipped exactly that bug once before.

Record only relations that appear in a proposal, so payload size scales with findings rather than schema size.

Postgres records per relation: `is_ordinary_table` (see "A correction to the spec" above) and a list of indexes with `name`, `columns`, `is_partial`, `is_unique`. Redshift records `is_ordinary_table`, `sortkey1`, `diststyle`, `unsorted`, `stats_off`.

- [ ] **Step 1: Write the failing tests**

```python
def test_physical_state_is_keyed_by_a_json_serializable_string():
    """A `Relation` is not JSON-serializable, and a TypeError here fires after the whole
    analysis has already run — this project has shipped that bug once."""
    adapter = PostgresWorkloadAdapter(querier=_canned({...}))
    state = adapter.physical_state(frozenset({Relation("public", "orders")}))
    assert list(state) == ["public.orders"]
    import json
    json.dumps(state)  # must not raise


def test_physical_state_records_the_indexes_verify_needs_to_detect_application():
    """"Was the proposed index created?" is answered by comparing two artifacts' index
    lists. Without `columns` the question is unanswerable, and without `is_partial` an
    ADV004 partial index is indistinguishable from a plain one leading with the same
    column."""
    adapter = PostgresWorkloadAdapter(querier=_canned({
        CAP_INDEXES: [("public", "orders", "idx_status", "status", 1,
                       False, False, 3, 100, False, None, False, "...")],
        ...
    }))
    state = adapter.physical_state(frozenset({Relation("public", "orders")}))
    [index] = state["public.orders"]["indexes"]
    assert index["name"] == "idx_status"
    assert index["columns"] == ["status"]
    assert index["is_partial"] is False
    assert index["is_unique"] is False


def test_a_relation_absent_from_table_facts_is_recorded_as_not_an_ordinary_table():
    """Postgres's CAP_TABLE_FACTS filters `relkind = 'r'`, so a view never appears. That
    absence is how `verify` detects ADV301's application — the relation becoming a table —
    without any new catalog query."""
    adapter = PostgresWorkloadAdapter(querier=_canned({CAP_TABLE_FACTS: [], ...}))
    state = adapter.physical_state(frozenset({Relation("public", "stg_orders")}))
    assert state["public.stg_orders"]["is_ordinary_table"] is False


def test_only_the_relations_asked_about_are_recorded():
    """Payload size must scale with findings, not with schema size."""
    adapter = PostgresWorkloadAdapter(querier=_canned({...many relations...}))
    state = adapter.physical_state(frozenset({Relation("public", "orders")}))
    assert list(state) == ["public.orders"]


def test_redshift_records_its_own_levers_and_says_absence_means_less_here():
    """`svv_table_info` omits empty tables as well as external ones, so
    `is_ordinary_table: False` on Redshift is weaker evidence than on Postgres."""
    adapter = RedshiftWorkloadAdapter(querier=_canned({...}))
    state = adapter.physical_state(frozenset({Relation("public", "events")}))
    entry = state["public.events"]
    assert entry["sortkey1"] == "created_at"
    assert entry["diststyle"] == "KEY(tenant_id)"
    assert set(entry) >= {"is_ordinary_table", "sortkey1", "diststyle", "unsorted", "stats_off"}
```

- [ ] **Step 2: Run to verify they fail.** Expected: `AttributeError: 'PostgresWorkloadAdapter' object has no attribute 'physical_state'`.

- [ ] **Step 3: Add the ABC method**, defaulting to `{}` with a docstring explaining that an adapter with no physical levers is a legitimate state, and that the key is a string for JSON's sake.

- [ ] **Step 4: Implement both adapters**, reusing the index and facts data each already fetched. `physical_state` must not issue new SQL — it reads what `fetch_indexes` / `fetch_table_facts` already returned.

- [ ] **Step 5: Add the payload key**, populated from the relations appearing in `proposals`.

- [ ] **Step 6: Run the tests, full suite, all four gates.**

- [ ] **Step 7: Prove three things discriminate, three separate mutations**

1. Key the dict by `relation` instead of `str(relation)`. Expected: the serializability test FAILS with a `TypeError`.
2. Drop `columns` from the recorded index. Expected: the index test FAILS.
3. Record every relation rather than only those asked about. Expected: the scoping test FAILS.

Report all three.

- [ ] **Step 8: Commit**

```bash
git commit -am "feat(advise): record the physical state behind each proposal in the payload"
```

---

### Task 3: `fingerprint_digests` and `query_groups`

**Files:**
- Modify: `src/sqlquality/workload/postgres.py`, `redshift.py` (evidence), `src/sqlquality/report.py`
- Test: `tests/test_workload_rules.py`, `tests/test_workload_redshift_rules.py`, `tests/test_report.py`

**Interfaces:**
- Produces: `proposals[].evidence.fingerprint_digests: list[str]` — 12-char digests, sorted.
- Produces: payload key `query_groups: list[dict]` with `digest`, `calls`, `total_time_ms`, `mean_ms`.
- Consumes: `_fingerprint_id` (existing, `postgres.py`).

**`ColumnUsage.fingerprint_ids` holds whole canonical SQL, not digests** — the field name is misleading and that is a recorded known-wart. Digest them with the existing `_fingerprint_id` before they reach the payload, or every proposal will carry every backing query's full text.

`mean_ms` is `total_time_ms / calls`, and **`None` when `calls` is 0** — a group with no recorded calls has no meaningful mean, and `0.0` would read as "instant".

Add `fingerprint_digests` **beside** the existing `fingerprints` / `co_occurring_fingerprints` counts; do not replace them.

- [ ] **Step 1: Write the failing tests** — including the digest-not-SQL guard, which is the one that matters:

```python
def test_evidence_carries_digests_not_query_text():
    """`ColumnUsage.fingerprint_ids` holds the whole canonical SQL despite its name. Putting
    it in the payload verbatim would print every backing query's full text inside every
    proposal — and the payload already carries redacted SQL where a rule needs it."""
    relation = Relation("public", "orders")
    usage = (_usage(relation, "status", ColumnRole.EQUALITY, cost_share=0.5,
                    fps=("SELECT \"id\" FROM \"orders\" WHERE \"status\" = %s",)),)
    [proposal] = propose_indexes(usage, _facts_map(relation, rows=100_000), {}, min_cost_share=0.01)
    digests = proposal.evidence["fingerprint_digests"]
    assert digests == [_fingerprint_id('SELECT "id" FROM "orders" WHERE "status" = %s')]
    assert all(len(d) == 12 for d in digests)
    assert not any("SELECT" in d for d in digests), "query text must not reach the payload"


def test_mean_ms_is_null_rather_than_zero_when_a_group_has_no_calls():
    """0.0 reads as "instantaneous", which is the opposite of "unknown"."""
    payload = advise_payload(..., workload=Workload(stats=(
        QueryStat(fingerprint="select 1", sql="select 1", calls=0, total_time_ms=0.0),
    ), window_description="w"), ...)
    [group] = payload["query_groups"]
    assert group["mean_ms"] is None
```

Plus: every digest in every proposal's `fingerprint_digests` resolves to a group (no dangling references — a dangling digest would make a proposal unverifiable).

**Corrected during Task 6's fix round 3, recorded here rather than silently amended in the code:** this criterion originally also required that `query_groups` contain *only* digests referenced by a proposal. Review found that to be the identical blind spot as the `physical_state` scoping corrected in the design doc, one payload key over — a group whose proposal is resolved between two runs (the index now exists, so the rule stops citing it) vanished from the later artifact even though the query was still running, and `verify` graded the success case `DISAPPEARED`. `query_groups` therefore carries one entry per `workload.stats` group, unconditionally; it stays bounded by `--limit`, so size still scales with the workload rather than the schema.

- [ ] **Step 2–6:** run red, implement, run green, then run the full suite and all four gates.

- [ ] **Step 7: Prove two mutations**

1. Put `fingerprint_ids` in the evidence unmodified. Expected: the digest test FAILS, showing query text.
2. Compute `mean_ms` as `total_time_ms / max(calls, 1)`. Expected: the null test FAILS.

- [ ] **Step 8: Commit.**

---

### Task 4: matching — proposal keys and group lookup

**Files:**
- Create: `src/sqlquality/verify.py`
- Modify: `src/sqlquality/models.py`
- Test: `tests/test_verify.py`

**Interfaces:**
- Produces: `proposal_key(proposal: dict) -> tuple[str, ...] | None`
- Produces: `index_proposals(payload: dict) -> dict[tuple[str, ...], dict]`
- Produces: `group_index(payload: dict) -> dict[str, dict]`
- Consumes: payloads shaped by Tasks 1–3.

**The rules have two evidence shapes and a single key silently fails on one of them.** Relation-scoped rules (ADV001–004, ADV007, ADV008, ADV101–105, ADV301) carry `schema`, `table` and usually `columns`. Statement-scoped findings do not: ADV006 carries a `tables` tuple plus one `fingerprint`, and ADV005's leading-wildcard branch carries a `fingerprint` and no relation at all.

A single `(code, schema, table, columns)` key would match nothing for every ADV005 and ADV006 finding and report them all as `disappeared` — the tool confidently announcing a finding had gone away while it sat in both artifacts. Derive the key from the evidence shape, and **test both shapes**.

- [ ] **Step 1: Write the failing tests**

```python
def test_a_relation_scoped_proposal_keys_on_code_relation_and_columns():
    key = proposal_key({
        "code": "ADV001",
        "evidence": {"schema": "public", "table": "orders", "columns": ["status", "created_at"]},
    })
    assert key == ("ADV001", "public", "orders", "status", "created_at")


def test_a_statement_scoped_finding_keys_on_code_and_fingerprint():
    """ADV005's wildcard branch and ADV006 carry no schema/table/columns triple. Keying them
    the same way would match nothing and report every one as `disappeared`."""
    assert proposal_key({"code": "ADV005", "evidence": {"fingerprint": "abc123abc123"}}) == (
        "ADV005", "abc123abc123",
    )
    assert proposal_key({
        "code": "ADV006",
        "evidence": {"tables": ["public.orders"], "fingerprint": "def456def456"},
    }) == ("ADV006", "def456def456")


def test_a_proposal_with_neither_shape_is_unkeyable_rather_than_mis_keyed():
    """Returning a partial key would silently group unrelated proposals together."""
    assert proposal_key({"code": "ADV999", "evidence": {}}) is None


def test_the_same_recommendation_in_two_runs_produces_the_same_key():
    """Column order is part of the recommendation — `(a, b)` and `(b, a)` are different
    indexes, and `advise` already keeps both when it proposes them."""
    a = {"code": "ADV001", "evidence": {"schema": "s", "table": "t", "columns": ["a", "b"]}}
    b = {"code": "ADV001", "evidence": {"schema": "s", "table": "t", "columns": ["b", "a"]}}
    assert proposal_key(a) != proposal_key(b)


def test_group_index_maps_digest_to_its_measurements():
    payload = {"query_groups": [
        {"digest": "aaa", "calls": 10, "total_time_ms": 100.0, "mean_ms": 10.0},
    ]}
    assert group_index(payload)["aaa"]["mean_ms"] == 10.0
```

- [ ] **Step 2: Run to verify they fail.** Expected: `ModuleNotFoundError: sqlquality.verify`.
- [ ] **Step 3: Implement**, with the key derivation branching on evidence shape and a docstring recording why one key is not enough.
- [ ] **Step 4: Run the tests.**
- [ ] **Step 5: Prove the two-shape branch discriminates.** Collapse `proposal_key` to the relation-scoped form only. Expected: **both** statement-scoped cases FAIL while the relation-scoped ones pass. Report both, not one — a fix that handles one shape is the defect in a new place.
- [ ] **Step 6: Commit.**

---

### Task 5: window classification

**Files:**
- Modify: `src/sqlquality/verify.py`, `src/sqlquality/models.py`
- Test: `tests/test_verify.py`

**Interfaces:**
- Produces: `WindowRelation` enum — `DISJOINT`, `COMPARABLE`, `NESTED`, `INCOMPARABLE`.
- Produces: `classify_windows(before: dict, after: dict) -> WindowRelation`
- Produces: `confidence_for(relation: WindowRelation) -> Confidence`

The mapping, from the spec:

| relation | detected by | confidence | why |
|---|---|---|---|
| `DISJOINT` | `stats_reset_at` differs between runs | HIGH | counters were cleared; two independent samples |
| `COMPARABLE` | both `since` set and equal | HIGH | equal durations by construction |
| `NESTED` | same `stats_reset_at`, no `since` | MEDIUM | the later window **contains** the earlier one, so a real gain is understated |
| `INCOMPARABLE` | engines differ, or the fields needed are absent | LOW | refuses to grade |

- [ ] **Step 1: Write one test per relation, plus the two that matter most:**

```python
def test_the_same_stats_reset_means_the_later_window_contains_the_earlier():
    """This is the common Postgres case — baselined last week, never reset. The comparison
    is still worth making, but a real improvement is diluted by pre-change executions, so
    it is graded MEDIUM rather than HIGH and the report says so."""
    w = {"engine": "postgres", "stats_reset_at": "2026-08-01T00:00:00", "since": None, "limit": 500}
    assert classify_windows({"window": w}, {"window": dict(w)}) is WindowRelation.NESTED
    assert confidence_for(WindowRelation.NESTED) is Confidence.MEDIUM


def test_two_engines_are_never_comparable():
    """A Postgres mean and a Redshift mean measure different servers. Grading this at all
    would be inventing a comparison."""
    before = {"window": {"engine": "postgres", "stats_reset_at": "T", "since": None, "limit": 500}}
    after = {"window": {"engine": "redshift", "stats_reset_at": None, "since": "T2", "limit": 500}}
    assert classify_windows(before, after) is WindowRelation.INCOMPARABLE
```

- [ ] **Steps 2–4:** red, implement, green.
- [ ] **Step 5: Prove every rung discriminates independently** — four mutations, one per relation, each collapsing that branch into its neighbour. Four results. A classifier where three of four cases are pinned is the "asserts over a set, checks one member" defect.
- [ ] **Step 6: Commit.**

---

### Task 6: applied detection and the verdict

**Files:**
- Modify: `src/sqlquality/verify.py`, `src/sqlquality/models.py`
- Test: `tests/test_verify.py`

**Interfaces:**
- Produces: `VerifyOutcome` enum — `IMPROVED`, `UNCHANGED`, `REGRESSED`, `DISAPPEARED`, `NOT_APPLIED`, `UNOBSERVABLE`.
- Produces: `ProposalVerdict` frozen dataclass — `key`, `code`, `applied: bool | None`, `outcome`, `confidence`, `mean_before: float | None`, `mean_after: float | None`, `cost_share_before`, `cost_share_after`, `note: str`.
- Produces: `verdicts(before: dict, after: dict) -> list[ProposalVerdict]`
- Consumes: Tasks 4 and 5.

**Applied detection** compares the two artifacts' `physical_state` for the proposal's relation, per the spec's table. `applied` is `None` — not `False` — when the proposal has nothing observable (ADV005, ADV006, ADV303): those are different states and conflating them is how "we did not look" becomes "you did not do it".

**The verdict** uses mean per call, never `cost_share`:

- both means known and after < before by more than a threshold → `IMPROVED`
- both known, within threshold → `UNCHANGED`
- both known, after > before beyond threshold → `REGRESSED`
- the group is absent from `after` → `DISAPPEARED`
- `applied is False` → `NOT_APPLIED` (whatever the means say — an unapplied change did not cause anything)
- `applied is None` → `UNOBSERVABLE`, with the group's presence and cost reported as data

Pick the threshold as a *relative* change (10% is a reasonable default) and put the number in one named constant with a docstring, so a reviewer can argue with it in one place.

- [ ] **Step 1: Write the failing tests.** The one this feature exists for:

```python
def test_a_diluted_cost_share_with_an_unchanged_mean_is_not_an_improvement():
    """The confound this whole design exists to survive. The workload grew, so every
    `cost_share` fell — but this query takes exactly as long as it did. Reporting
    `IMPROVED` here would credit the tool for someone else's traffic."""
    before = _payload(groups=[("aaa", 10, 1000.0)], cost_share=0.50)
    after = _payload(groups=[("aaa", 10, 1000.0)], cost_share=0.05)  # mean identical
    [verdict] = verdicts(before, after)
    assert verdict.outcome is VerifyOutcome.UNCHANGED
    assert verdict.mean_before == verdict.mean_after == 100.0
    assert verdict.cost_share_before == 0.50 and verdict.cost_share_after == 0.05
```

Then one per outcome, plus:

```python
def test_an_unapplied_change_is_never_credited_with_an_improvement():
    """If the index was not created, something else made the query faster, and saying
    otherwise is the confidently-wrong failure this rule set exists to avoid."""
    before = _payload(groups=[("aaa", 10, 1000.0)], indexes=[])
    after = _payload(groups=[("aaa", 10, 100.0)], indexes=[])  # 10x faster, no index
    [verdict] = verdicts(before, after)
    assert verdict.applied is False
    assert verdict.outcome is VerifyOutcome.NOT_APPLIED


def test_an_unobservable_proposal_reports_none_not_false_for_applied():
    """`False` says "you did not do it". `None` says "we cannot tell". An ADV005 rewrite
    has no catalog state, so only the second is true."""
    [verdict] = verdicts(_payload(adv005=True), _payload(adv005=True))
    assert verdict.applied is None
    assert verdict.outcome is VerifyOutcome.UNOBSERVABLE
```

- [ ] **Steps 2–4:** red, implement, green.
- [ ] **Step 5: Prove each outcome branch and the applied gate independently.** Six outcome mutations plus one collapsing `applied is None` into `False`. Seven results. Confirm specifically that the `NOT_APPLIED` gate takes precedence over a mean improvement — inverting that order must redden the unapplied test.
- [ ] **Step 6: Commit.**

---

### Task 7: the `verify` command, its output, and its refusals

**Files:**
- Modify: `src/sqlquality/cli.py`, `src/sqlquality/report.py`
- Test: `tests/test_verify_cli.py`

**Interfaces:**
- Produces: `sqlquality verify BEFORE AFTER [--json] [--markdown PATH]`
- Produces: `verify_payload(verdicts, before, after) -> dict`, `render_verify_markdown(...) -> str`
- Consumes: Tasks 4–6.

**Exit codes** follow the house contract: 0 = reported, 2 = usage/input error. `verify` never gates. A `--gate` flag is explicitly out of scope.

**Refusals, all exit 2 with the reason:** different engines; the same artifact passed twice; `after` older than `before`; malformed or unreadable JSON; and — the important one — **an artifact lacking the keys Tasks 1–3 add**, i.e. produced by 0.3.0. Detect that by `window` being a string rather than an object, and say "regenerate with 0.4.0 or later" rather than deriving a verdict from absent data.

Output: a rich table (proposal, applied, outcome, mean before → after, confidence), the workload-context line (total window cost and group count, both runs), and the window relation with its caveat spelled out when `NESTED`.

- [ ] **Step 1: Write the failing tests** — one per refusal, each asserting exit code **and** that the message names the cause:

```python
def test_a_0_3_0_artifact_is_refused_rather_than_half_understood():
    """0.3.0's `window` is a string. Proceeding would mean classifying the window relation
    from absent data, and every verdict downstream would inherit that guess."""
    before = _write(tmp_path / "b.json", {"window": "since stats reset at T", "proposals": []})
    result = runner.invoke(app, ["verify", str(before), str(before)])
    assert result.exit_code == 2
    assert "0.4.0" in result.stderr
    assert "regenerate" in result.stderr.lower()


def test_the_same_artifact_twice_is_refused():
    """It would report every proposal unchanged, which looks like a finding rather than a
    mistake."""
    result = runner.invoke(app, ["verify", str(path), str(path)])
    assert result.exit_code == 2


def test_two_engines_are_refused_by_name():
    ...
    assert "postgres" in result.stderr and "redshift" in result.stderr


def test_the_nested_window_caveat_reaches_the_user():
    """On the common Postgres path a real improvement is understated. A user who does not
    know that will read `UNCHANGED` as "it did not work"."""
    result = runner.invoke(app, ["verify", str(before), str(after)])
    assert "understate" in result.stderr.lower()
    assert "pg_stat_statements_reset" in result.stderr
```

- [ ] **Steps 2–4:** red, implement, green.
- [ ] **Step 5: Prove the wiring cannot be deleted.** Replace the `verdicts(...)` call in the command body with `[]`. Expected: at least one test FAILS. This exact defect — a feature disconnected from the CLI with the suite green — has appeared **three** times in this codebase; do not make it four.
- [ ] **Step 6: Prove each refusal independently** — five mutations, one per refusal, each removing only that check. Five results.
- [ ] **Step 7: Commit.**

---

### Task 8: golden pairs, the loop closed live, and docs

**Files:**
- Create: `tests/fixtures/verify/*.json`, `tests/integration/test_verify_live.py`
- Modify: `README.md`, `CHANGELOG.md`, the design spec's deviations

**Interfaces:**
- Consumes: everything from Tasks 1–7.
- Produces: **no new production interfaces.** This task adds fixtures, tests and prose only. If you find yourself changing `src/`, something in Tasks 1–7 was incomplete — report that rather than fixing it here, so the gap is attributed to the task that owns it.

- [ ] **Step 1: Commit golden before/after pairs**, one per outcome, including the three cases that have bitten this project: a nested window, a `--limit`-truncated window where a group left the window without getting cheaper, and a group that vanished entirely. Each pair is a test that `verify` produces the expected verdict *and* confidence.

- [ ] **Step 2: Write the integration test that closes the loop for real**

Against the live Postgres container: seed a workload with a hot unindexed predicate, run `advise --json` to get `before`, **create the index it proposed**, drive the workload again, run `advise --json` to get `after`, then `verify` and assert `applied is True`.

Include a non-vacuity guard: assert the `before` artifact actually contained an ADV001 proposal for that relation first. Without it the test passes when the workload produced no proposal at all — the shape of hollow test this project has found eight times.

Note what this test can and cannot show: it proves application detection works end to end. Whether the outcome is `IMPROVED` depends on real timings on a small seeded table and is **not** a safe assertion — assert on `applied`, and on the mean being *present* for both runs, not on its direction.

- [ ] **Step 3: Run both suites and all four gates.**

```bash
docker ps --filter publish=27432          # confirm the port is free first
uv run pytest -q                          # zero skips
docker compose -f tests/integration/docker-compose.yml up -d && sleep 8
uv run pytest -m integration -q
docker compose -f tests/integration/docker-compose.yml down
```

- [ ] **Step 4: Document it.** A README section for `verify`: the two-artifact workflow, that it never connects, mean-per-call as the metric with `cost_share` as context, and the nested-window limitation with the `pg_stat_statements_reset()` suggestion. Add `verify` to the commands table and the contents list. CHANGELOG under `[Unreleased]`. A spec deviation recording the `relkind` → `is_ordinary_table` correction and why.

- [ ] **Step 5: Commit.**

---

## Self-Review

**Spec coverage.** Every spec section maps to a task: the four payload additions to Tasks 1–3, matching to Task 4, window classification to Task 5, the verdict model to Task 6, error handling and output to Task 7, testing and docs to Task 8. The spec's non-goals — new engines, any write, `--gate`, verifying ADV303 — appear in no task.

**One spec correction, made deliberately and recorded above:** `relkind` is not available without new SQL because `CAP_TABLE_FACTS` filters `relkind = 'r'`, so `physical_state` records `is_ordinary_table` instead. Task 8 writes that into the spec's deviations rather than leaving the two documents disagreeing.

**Placeholder scan.** Task 2's and Task 3's test bodies use `_canned({...})` and `_payload(...)` with elided arguments, because both helpers already exist in the test files being modified and copying their full fixture setup here would drift from the originals. Every step names the helper to reuse. Tasks 4–6 — the new module, where nothing pre-exists — carry complete test code. No step says "add error handling" or "write tests for the above".

**Type consistency.** `WindowRelation`, `VerifyOutcome`, `ProposalVerdict` and `Confidence` are used with the same names and members from their defining task onward. `proposal_key` returns `tuple[str, ...] | None` in both Task 4 and Task 6. `physical_state` is keyed by `str` in Tasks 2, 6 and 7. `mean_ms` is `float | None` everywhere.

**Known risk this plan accepts.** Task 1 changes `window` from a string to an object, which breaks any consumer of a 0.3.0 payload. That is why Task 7 detects it explicitly and refuses. The alternative — a parallel `window_facts` key leaving `window` a string — would leave two representations of one fact in the payload forever, which this codebase has an explicit rule against.
