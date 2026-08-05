# `sqlquality verify` — closing the advise feedback loop

**Status:** design approved 2026-08-03. Implementation plan not yet written.

## The problem

Every `advise` proposal is a hypothesis that is never scored. The tool says an index will help, someone applies it, and nothing ever checks whether it did. Across five development batches and sixteen rules we cannot answer, for any single rule, whether following its advice made a database faster.

That gap is not only a missing feature. It means the confidence ladder — the discipline this whole rule set is built on — has never been validated against an outcome. A rule that emits HIGH could be wrong every time and nothing in the project would notice.

`verify` closes the loop: given an `advise` run before a change and one after, it reports for each proposal whether it was applied, whether the queries that justified it actually got faster, and how much to trust that answer.

## Four decisions, and why

### 1. Both proposal-level and query-group-level

The headline is per proposal ("6 of 9 applied, 4 improved"); the evidence beneath it is per query group. Proposal-level alone is too coarse — it cannot distinguish "the index worked" from "that query stopped running," and refusing to distinguish those is exactly the conflation this project has repeatedly had to fix. Group-level alone buries the answer.

### 2. Application is observed, not declared

`verify` reads whether a change was applied from the artifact's recorded physical state, rather than accepting a flag or asking. A flag records intent; the catalog records reality, and they diverge — an index created by hand and then destroyed by the next `dbt run` is precisely the case ADV302 exists for.

Detection per rule family:

| proposal | detected by |
|---|---|
| ADV001, ADV004, ADV007, ADV008 (create index) | an index now leads with the proposed columns |
| ADV002, ADV003 (drop index) | the named index is absent |
| ADV101, ADV102, ADV103 (Redshift keys) | `sortkey1` / `diststyle` changed |
| ADV104 (vacuum/analyze) | `unsorted` / `stats_off` fell |
| ADV105 (Advisor) | Advisor's row is gone — read at proposal level, not from physical state |
| ADV301 (materialize a view) | `relkind` is no longer a view |
| ADV005, ADV006 (advisory) | **nothing** — see below |
| ADV303 (unused model) | **nothing** — needs a manifest, which an offline `verify` lacks |

### 3. Graded verdicts, not silence and not inference

ADV005 (non-sargable predicate, leading-wildcard `LIKE`) and ADV006 (hot `SELECT *`) advise rewriting a query. There is no catalog state to inspect; the only trace is the group's fingerprint changing because the SQL changed.

Two tempting answers are both wrong. Inferring "you rewrote it" from "a similar group appeared" is correlation presented as causation — the shape of every confidently-wrong finding this project has had to repair. Reporting nothing is the opposite failure: a tool that will not say whether its advice worked.

The resolution is the vocabulary the codebase already has. `advise` does not refuse to propose when evidence is thin; it proposes at LOW and names the gap. `verify` grades the same way: an unobservable proposal whose group has vanished is *possibly addressed* at LOW, with the statement that disappearance is not proof; one whose group is still there at the same cost is *not addressed*, and that verdict is solid.

### 4. Mean per call is the metric — `cost_share` is not

This is the load-bearing decision.

`cost_share` divides by the whole window's cost, and on Postgres `pg_stat_statements` is cumulative since the last stats reset, carrying no per-statement timestamps. So a baseline taken today and a verification next week with no reset in between are not two samples — **the baseline is a subset of the follow-up**. A group's share falls simply because a week of other traffic accumulated. `--limit` truncation compounds it: a group can leave the window entirely without ever getting cheaper.

`cost_share` is the right metric for *prioritising* work, which is why `advise` uses it, and close to useless for *measuring* an improvement.

`total_time_ms / calls` — mean time per execution — answers the actual question and is robust to workload shift, which is the confound that motivated this feature. Both figures are already on `QueryStat`.

So: mean per call is the verdict, `cost_share` rides along as context (whether the finding still *matters*, not whether it got *better*), and a workload-context line prints total window cost and group count for both runs so a global shift is visible rather than deduced.

## Architecture

`sqlquality verify before.json after.json` — a new **offline** command that diffs two `advise --json` artifacts. It never opens a connection, joining `check` on the offline side and leaving `advise` as the only command that connects.

Consequences worth stating plainly:

- **No new file format.** The baseline is an ordinary `advise --json` run.
- **No credentials.** Anyone reviewing a PR can run it, including people who will never have production access.
- **The payload must carry more than it does today**, because "was it applied?" has to be readable from the artifact rather than inferred from a proposal's absence. Without recorded physical state, a proposal missing from the second run could mean "index created" *or* "cost fell below threshold" — and choosing between those is the guessing this design exists to avoid.

`advise` shipped 0.3.0 on 2026-08-03, so nothing yet depends on the payload's current shape. This is the cheapest moment the additions will ever be.

### Payload additions

Today's top-level keys are `analyzed`, `degraded`, `engine`, `proposals`, `redacted`, `skipped`, `window`. Three additions are purely additive; the fourth changes an existing key's type.

- **`proposals[].evidence.fingerprint_digests`** — digests of the groups backing this proposal. **Added alongside** today's `fingerprints` / `co_occurring_fingerprints` counts, which stay as they are. Reuses the existing `_fingerprint_id` (sha256, 12 hex chars).
- **`query_groups`** — new top level: `digest`, `calls`, `total_time_ms`, `mean_ms`. Query text is not duplicated here; it already appears in the evidence of the rules that carry it. `mean_ms` is `total_time_ms / calls`, and is `null` when `calls` is 0 — a group with no recorded calls has no meaningful mean, and emitting `0.0` would read as "instant".

  **Every query group the run's workload analysis carries (`workload.stats`)** — corrected
  during Task 6's fix round 3 after review found the original requirement here, "only
  groups referenced by a proposal", to be the *identical* blind spot as the `physical_state`
  scoping corrected below, one payload key over: a group whose proposal is resolved between
  two runs (the index now exists, so the rule stops citing it) vanished from `after` even
  though it was still running, and `verify` graded the success case `DISAPPEARED`. Size
  still scales with the workload rather than the schema, because `workload.stats` is bounded
  by the same `--limit` — `analyzed.query_groups_in_window` reports the count. Recorded here
  rather than silently amended in the implementation so this requirement cannot instruct a
  future implementer to reintroduce the bug review already found once.
- **`physical_state`** — new top level, keyed by the **string** `"schema.table"` (a `Relation` is not JSON-serializable — a `TypeError` here would fire after the whole analysis had run, which this project has already shipped once). Records, per relation:
  - both engines: `relkind` — table, view, or materialized view — which is what makes ADV301's application observable;
  - Postgres: each index's `name`, `columns`, `is_partial`, `is_unique`;
  - Redshift: `sortkey1`, `diststyle`, `unsorted`, `stats_off`.

  **Every relation appearing in a proposal, plus every relation the run's workload
  analysis touched (`aggregation.tables`)** — corrected during Task 6's fix round after
  review found the narrower "only proposal relations" scoping left `verify` structurally
  unable to confirm its own headline case: a relation whose proposal is resolved between
  two runs (the recommended index now exists, so the rule stops firing) carried no
  `physical_state` entry at all in the very run that mattered. Sizing against
  `aggregation.tables` rather than the whole introspected schema keeps the original
  intent (size scales with the workload, not schema size) while closing that gap.
- **`window`** — promoted from a bare string to an object: `description` (the existing string, unchanged), `engine`, `stats_reset_at`, `since`, `limit`.

That last one is the single **breaking** payload change. Acceptable because `advise` is hours old and, per its own CHANGELOG, has no previously-released behaviour — but it must be called out in the 0.4.0 notes rather than slipped in.

### Matching

Query groups match on digest — a hash of the canonical SQL, stable across runs **that used the same redaction setting**.

**Corrected during Task 6's fix round 5, after review found the original claim ("stable across runs", unqualified) false as written.** The digest is computed over the canonicalized query text, and `--keep-literals` changes that text, so two runs differing in that flag record the same query group under different digests. Reproduced from two real `advise` runs on PostgreSQL 16 — no degraded read, matching `--limit`, the recommendation genuinely applied in between, the query still running — as `applied=True outcome=DISAPPEARED note="Cited query group(s) no longer appear in the after run."` The group had not gone anywhere; only its name had. Same statement-scoped consequence for proposal matching, since `(code, fingerprint)` keys on the same text.

`verify` therefore compares `payload["redacted"]` as a **pair-level precondition** (`sqlquality.verify.artifact_incomparabilities`) and claims no query-group-level verdict at all when the two runs disagree, or when either flag is unreadable. This is deliberately *not* modelled as a missing reading (`degraded`) nor as a weak window (`INCOMPARABLE`, which still yields a verdict at `LOW`): the two artifacts do not share a coordinate system, so there is no confidence at which the comparison could be stated. `sqlquality verify` (Task 7) must refuse such a pair up front and say why.

On PostgreSQL the reachable surface is narrower than the flag suggests: `pg_stat_statements` already normalizes every *parameterizable* constant to `$N` before sqlquality reads a row, so redaction only moves the digest for a statement retaining a non-parameterizable constant — an `ORDER BY`/`GROUP BY` ordinal, verified on 16. On Redshift, `sys_query_history` stores statement text verbatim, so every literal-bearing statement moves.

Proposals need two keys, because the rules have two evidence shapes:

- **Relation-scoped rules** (ADV001–004, ADV007, ADV008, ADV101–105, ADV301) carry `schema`, `table` and usually `columns`, and match on `(code, schema, table, columns)`. That tuple is what makes two proposals *the same recommendation*.
- **Statement-scoped findings** carry no such triple. ADV006 records a `tables` tuple of qualified strings plus a single `fingerprint`; ADV005's leading-wildcard branch records a `fingerprint` and no relation at all. These match on `(code, fingerprint)`.

A single key would have silently failed to match every ADV005 and ADV006 finding, reporting them all as `disappeared`. The implementation must derive the key from the evidence shape rather than assume one, and a test must cover both shapes.

ADV105 is a special case within the relation-scoped group: its application is not a physical-state change but the **absence of Advisor's own row** in the later run, which surfaces as the proposal no longer being emitted. It is therefore verified at proposal level only, and `physical_state` records nothing for it.

### The verdict model

Each proposal in `before.json` yields:

- **`applied`** — from the `physical_state` delta.
- **`outcome`** — one of `improved`, `unchanged`, `regressed`, `disappeared`, `not_applied`, `unobservable`. The valuable one is **applied but unchanged**: the work was done and did not help, which no other tool reports.
- **`confidence`** — from the window relationship:

| windows | grade | reasoning |
|---|---|---|
| disjoint (stats reset between runs) | HIGH | clean comparison |
| comparable duration (same `--since`) | HIGH | clean by construction |
| nested (Postgres cumulative, no reset) | MEDIUM | pre-change executions dilute the mean, so a real gain is **understated** |
| incomparable (different engines, missing window data) | LOW | refuses to grade and says why |

The nested case is the common one — someone who baselined last Tuesday and never reset stats. Its improvement is understated, sometimes badly. That is documented loudly rather than engineered around, because the alternative is writing to the user's database. `verify` *suggests* `pg_stat_statements_reset()` for an undiluted comparison; sqlquality never performs it.

## Error handling

`verify` refuses rather than guesses. Exit 2, with the reason, for:

- artifacts from different engines
- the same artifact passed twice (which would report everything unchanged)
- `after` older than `before`
- malformed or unreadable JSON
- **an artifact lacking the new keys** (i.e. produced by 0.3.0) — regenerate it rather than let a verdict be derived from absent data

Exit 0 otherwise. Like `advise`, `verify` reports and never gates. A `--gate` flag for "fail CI if something you applied regressed" is an obvious future addition and deliberately out of scope.

## Output

A terminal table — proposal, applied, outcome, mean before → after, confidence — plus the workload-context line. `--json` and `--markdown` mirror `advise`'s existing options.

## Testing

Matching, grading and window classification are pure functions over two dicts, so most of this is unit-testable with no database and no extras — it stays inside the `no-extras` CI job.

Two kinds carry the real weight.

**Golden artifact pairs**, committed, one per outcome — including the cases that have bitten this project: a nested window, a `--limit`-truncated window, a group that vanished.

**An integration test that closes the loop for real**: baseline against live Postgres, create the proposed index, re-run `advise`, verify the verdict. This is the first test in the entire feature that can show a proposal was *correct* rather than merely well-formed, and it belongs in the `integration` CI job that now exists.

The single most important test: **workload grows, `cost_share` falls, mean per call is unchanged → must report `unchanged`, not `improved`.** That is the confound this design exists to survive, and without that test the feature is worthless.

## Scope

**In:** Postgres and Redshift, engine-agnostic where the recorded state allows.

**Out:** new engines; any write to the user's database, including the stats reset; a `--gate` mode; verifying ADV303 (needs a manifest an offline command does not have).

## Known limitations to document

- On the nested-window Postgres path a real improvement is understated, because cumulative statistics include pre-change executions.
- ADV005, ADV006 and ADV303 cannot have their application observed and are graded accordingly.
- Redshift's verification inherits the adapter's standing caveat: its catalog SQL has still never been executed against a live cluster.
