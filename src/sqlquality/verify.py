"""Matching layer for `sqlquality verify`: decide which proposal in one `advise --json`
artifact is "the same recommendation" as which in another.

Deliberately dependency-free and fully offline: this module reads two already-parsed JSON
payloads (plain `dict`/`list`/`str`/`float`/`None` values, exactly what `json.load` returns)
and touches nothing else — no `psycopg` import, no network, no filesystem beyond what the
caller already handed it. Everything downstream (the diff itself, in a later task) depends
on the keys produced here being right: a mis-key makes `verify` confidently report a live
finding as `disappeared` while it sits in both artifacts, or collapse two genuinely
different recommendations into one.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite

from sqlquality.models import Confidence


def proposal_key(proposal: dict[str, object]) -> tuple[str, ...] | None:
    """A key identifying "the same recommendation" across two `advise --json` artifacts.

    The rules' evidence comes in three shapes, and a single `(code, schema, table,
    *columns)` key silently fails two of them:

    * **Relation-scoped, plural `columns`** (ADV001, ADV002, ADV003, ADV004, ADV007,
      ADV008): keyed on `(code, schema, table, *columns)`. Column *order* is part of the
      recommendation — `(a, b)` and `(b, a)` are different indexes, and `advise` already
      keeps both when it proposes them — so the columns are appended in evidence order,
      never sorted.
    * **Relation-scoped, singular `column`** (ADV005's sargability branch, ADV101, ADV102):
      keyed on `(code, schema, table, column)`. Folding this into the plural case (checking
      only `columns`) would silently drop the column for every one of these three rules: the
      key would degrade to `(code, schema, table)`, so if run A proposed `SORTKEY(a)` and run
      B proposed `SORTKEY(b)` on the same table, both would key identically and `verify`
      would report "still present, unchanged" for advice that in fact *changed* — a
      different and more interesting answer than the one it would give.
    * **Statement-scoped, no relation at all** (ADV005's leading-wildcard branch, ADV006):
      keyed on `(code, fingerprint)`. These carry `evidence["fingerprint"]` and no
      `schema`/`table` whatsoever, so the relation-scoped key would match nothing for every
      one of them and report them all as `disappeared` — the tool confidently announcing a
      finding had gone away while it sat in both artifacts. Checked only when `schema`/
      `table` are absent, never the other way around: a proposal is relation-scoped or
      statement-scoped, never both today, but should a future rule's evidence ever carry
      *both* a relation and a `fingerprint`, it must still key by the relation — the
      identity of a rule that names a table is the table, not incidentally which single
      query happened to trigger it this run.

    Several further hazards, beyond the three shapes above, need their own discriminator
    folded in — each is a real proposal shape, not a hypothetical:

    * **Two proposals for one relation with no distinguishing column at all.**
      `propose_maintenance` (ADV104) emits an independent `VACUUM` proposal and an
      independent `ANALYZE` proposal from two separate `if` statements (not `if`/`elif`)
      whenever a relation's `unsorted` *and* `stats_off` both cross their thresholds, and
      neither proposal's evidence carries a `column`, `columns` or `index`. For this bucket
      (ADV103, ADV104, ADV301, ADV303 — see the next two bullets for ADV105's own,
      separate discriminator) this falls back to the proposal's own `ddl`, which is
      f-string-deterministic from the relation for every rule in this bucket: `CREATE
      INDEX` proposals are emitted unnamed, so `ddl` would otherwise be redundant with
      `columns` there, but for exactly the shapes with nothing else to discriminate on,
      `ddl` distinguishes `VACUUM` from `ANALYZE`. `ddl` is `None` on some proposals in
      this bucket (ADV301, ADV303) — harmless there, since those rules only ever emit one
      proposal per relation and have nothing to disambiguate in the first place.
    * **Two indexes on the same columns.** `index` (when present) is always appended after
      the column discriminator: ADV002/ADV003's whole reason for existing is that two
      catalog indexes can share the same column list — the identical `columns` is not a
      bug there, it is the finding — so `index` is what tells the two `DROP INDEX`
      proposals apart.
    * **ADV105 must never key on its own `ddl` at all — not even as the same fallback the
      rest of this bucket uses.** `propose_advisor`'s evidence is Amazon Redshift Advisor's
      own, relayed rather than generated: `ddl` is `row.recommended_ddl`, `str | None`.
      Two Advisor rows for one relation with different `rec_type` and a `NULL`
      `recommended_ddl` would otherwise collide on the exact same key the `ddl` fallback
      gives every other member of this bucket — a real payload can reach this, not merely
      a hypothetical one. And even when `recommended_ddl` is present, it is foreign text
      sqlquality neither generates nor validates: any drift between two runs — whitespace,
      column ordering inside the statement, anything Advisor's own analysis phrases
      differently — would change the key and make `verify` report one stable
      recommendation as both `disappeared` and `new`. `evidence["recommendation_type"]`
      (Advisor's own `rec_type`, always a plain `str`, never the relayed DDL) is the
      stable, structured discriminator instead, checked ahead of the `ddl` fallback so it
      always wins for ADV105.
    * **ADV004's partial-index proposal carries a `guard_column`/`guard_predicate` beside
      its leading `columns`.** `WHERE shipped_at IS NULL` and `WHERE cancelled_at IS NOT
      NULL` restricting the *same* leading column are different proposals, so both guard
      fields are folded into the key when present. Both are structural, not literal text —
      `guard_column` is a column identifier and `guard_predicate` is one of exactly two
      constants (`"IS NULL"`/`"IS NOT NULL"`) — so, unlike ADV105's `ddl`, including them
      does not risk the key drifting between two runs of the same workload. Safe without
      this today too (`propose_partial_indexes` emits at most one proposal per relation,
      picking the single highest-cost co-occurring pair), but that safety lives in the
      rule's loop structure, not in anything this key otherwise guarantees — folding the
      guard in removes the dependency on that structure staying true.

    Returns `None` for a proposal with neither a relation nor a `fingerprint` in its
    evidence: returning a partial key (say, `(code,)` alone) would silently group every
    such proposal together as "the same recommendation" regardless of what it actually
    says, which is worse than refusing to key it at all.

    A proposal_key collision that still gets past every discriminator above is not raised
    on here — see `index_proposals` and `ProposalIndex` for how that is disclosed instead.
    """
    code = proposal.get("code")
    evidence = proposal.get("evidence")
    if not isinstance(code, str) or not isinstance(evidence, dict):
        return None

    schema = evidence.get("schema")
    table = evidence.get("table")
    if not (isinstance(schema, str) and isinstance(table, str)):
        fingerprint = evidence.get("fingerprint")
        if isinstance(fingerprint, str):
            return (code, fingerprint)
        return None

    parts: list[str] = [code, schema, table]
    discriminated = False

    columns = evidence.get("columns")
    if isinstance(columns, (list, tuple)) and all(isinstance(c, str) for c in columns):
        parts.extend(columns)
        discriminated = True
    else:
        column = evidence.get("column")
        if isinstance(column, str):
            parts.append(column)
            discriminated = True

    # ADV004's partial-index guard: structural (a column identifier and one of two fixed
    # predicate strings), so folding it in cannot destabilize the key across runs.
    guard_column = evidence.get("guard_column")
    if isinstance(guard_column, str):
        parts.append(guard_column)
        discriminated = True

    guard_predicate = evidence.get("guard_predicate")
    if isinstance(guard_predicate, str):
        parts.append(guard_predicate)
        discriminated = True

    # ADV002/ADV003: two indexes on the same columns, told apart only by catalog name.
    index = evidence.get("index")
    if isinstance(index, str):
        parts.append(index)
        discriminated = True

    # ADV105's own discriminator — Advisor's structured `rec_type`, never its relayed
    # `ddl`. Checked ahead of the `ddl` fallback below so that fallback never applies to
    # ADV105, whether or not `recommended_ddl` happens to be present.
    recommendation_type = evidence.get("recommendation_type")
    if isinstance(recommendation_type, str):
        parts.append(recommendation_type)
        discriminated = True

    if not discriminated:
        ddl = proposal.get("ddl")
        if isinstance(ddl, str):
            parts.append(ddl)

    return tuple(parts)


@dataclass(frozen=True)
class ProposalIndex:
    """The result of indexing one payload's proposals by `proposal_key`.

    **Deviates from the brief's stated `index_proposals(payload) -> dict[tuple[str, ...],
    dict]` return type.** The brief's contract could not distinguish "this key identified
    one proposal" from "this key identified several, and only one was kept" — the first is
    exactly what `verify` needs, the second is silent data loss. This dataclass makes both
    outcomes explicit instead of choosing one and hiding the other.

    `matched` holds every key that identified **exactly one** proposal: safe for `verify`
    to compare directly against the same key in the other artifact.

    `collisions` holds every key that identified **more than one** proposal, in the order
    they appeared in `payload["proposals"]`. Every proposal sharing an ambiguous key is
    preserved here in full — none is dropped, and none is picked over the others.

    **Read this before consuming `index_proposals` (Task 5 and Task 6):** a key present in
    `collisions` for a run is not the same fact as that key being `disappeared` or `new`
    relative to the other run — it means *this run's own artifact* could not tell two of
    its proposals apart under this key, which is a fact about this payload, prior to any
    comparison across payloads. Report it as "N recommendations could not be matched
    unambiguously," not as a disappearance or an addition, and do not let it block
    comparing every other, unambiguous key.

    This replaces an earlier design that raised `ProposalKeyCollisionError` on any
    collision. That was reachable in a real payload — two Amazon Redshift Advisor rows for
    one relation with a `NULL` `recommended_ddl` both keyed to `("ADV105", schema, table)`
    before `proposal_key` gained its `recommendation_type` discriminator — and a `raise`
    there aborted comparing every other proposal in the payload over that one ambiguous
    pair. That is backwards for this project, which discloses a check that could not run
    rather than assuming its answer everywhere else (`window`'s and `physical_state`'s
    present-but-null fields, Redshift's ADV104/ADV105 omitting `fingerprint_digests`
    entirely rather than emitting `[]`) — a matching ambiguity deserves exactly the same
    treatment, not a harder failure than any of those.
    """

    matched: dict[tuple[str, ...], dict[str, object]]
    collisions: dict[tuple[str, ...], tuple[dict[str, object], ...]]


def index_proposals(payload: dict[str, object]) -> ProposalIndex:
    """Every keyable proposal in an `advise --json` payload, grouped by `proposal_key`.

    A proposal `proposal_key` cannot key at all (see its docstring) is silently excluded
    from both `matched` and `collisions`: it cannot participate in the run-to-run
    comparison `verify` exists to do, and that is a fact about the proposal's evidence
    shape, not an ambiguity between two proposals.

    See `ProposalIndex` for the contract Task 5 and Task 6 must follow when consuming this
    function's result, and for why its return type is not the brief's stated bare `dict`.
    """
    proposals = payload.get("proposals")
    grouped: dict[tuple[str, ...], list[dict[str, object]]] = {}
    if isinstance(proposals, list):
        for proposal in proposals:
            if not isinstance(proposal, dict):
                continue
            key = proposal_key(proposal)
            if key is None:
                continue
            grouped.setdefault(key, []).append(proposal)

    matched: dict[tuple[str, ...], dict[str, object]] = {}
    collisions: dict[tuple[str, ...], tuple[dict[str, object], ...]] = {}
    for key, group in grouped.items():
        if len(group) == 1:
            matched[key] = group[0]
        else:
            collisions[key] = tuple(group)
    return ProposalIndex(matched=matched, collisions=collisions)


def group_index(payload: dict[str, object]) -> dict[str, dict[str, object]]:
    """The payload's top-level `query_groups` list, by `digest`.

    Passes each group dict through unchanged — in particular, `mean_ms` stays `None` when
    a group's `calls` is `0` rather than being coerced to `0.0`. `report.py`'s
    `_query_groups_payload` already makes that distinction deliberately (`None` means "no
    calls to average", `0.0` would read as "instantaneous"), and re-deriving or
    normalizing the value here would risk quietly losing that distinction on this side of
    the comparison `verify` is built to do.

    A group missing a string `digest` is skipped rather than indexed under a fabricated
    key: `digest` is how a later run's `fingerprint_digests` looks a group up here, so a
    group with no usable digest cannot be joined to anything regardless.
    """
    groups = payload.get("query_groups")
    index: dict[str, dict[str, object]] = {}
    if not isinstance(groups, list):
        return index
    for group in groups:
        if not isinstance(group, dict):
            continue
        digest = group.get("digest")
        if isinstance(digest, str):
            index[digest] = group
    return index


class WindowRelation(str, Enum):
    """How comparable two `advise --json` payloads' `"window"` objects are — and
    therefore what confidence, if any, a verdict built from them may claim.

    Every field the `"window"` object carries (`engine`, `stats_reset_at`, `since`,
    `since_duration_seconds`, `limit`) is *present-but-null* rather than absent when the
    engine cannot supply it (see `report.py`'s `advise_report_payload`): `null` means
    "this engine cannot tell you," not a value comparable to anything — including another
    `null`. Every rule below exists to keep that distinction intact; see
    `classify_windows` for the exact decision order and the rulings each guards against.

    * `DISJOINT` — both sides' `stats_reset_at` are non-null and differ, *and neither side
      reports a `since_duration_seconds`*: the counters were cleared between the two runs
      with no explicit window filter in play, so the two measurements are independent
      samples.
    * `COMPARABLE` — both sides' `since_duration_seconds` are non-null and equal: both
      runs measured the same explicit *duration*, regardless of the absolute cutoff each
      run happened to bind that duration to (two runs a week apart with the same
      `--since 7d` write different absolute cutoffs but the same duration — see
      `classify_windows`'s docstring for why the duration, not the cutoff, is what this
      checks).
    * `NESTED` — both sides' `stats_reset_at` are non-null and equal, and *neither side
      reports a `since_duration_seconds`*: the common Postgres case — baselined once,
      never reset since. `pg_stat_statements` is cumulative, so the later window's
      counters necessarily *contain* the earlier window's; a real improvement is still
      visible, just diluted by pre-change executions.
    * `INCOMPARABLE` — every other case: the engines differ or either window is
      missing/malformed (checked first, ahead of every other rule); `since_duration_seconds`
      set on only one side, meaning one run filtered its window and the other did not;
      `since_duration_seconds` set on both sides but unequal, meaning the two runs
      requested different durations; a `stats_reset_at` pair with a null on either side,
      which is missing information rather than evidence of a reset; and both
      `stats_reset_at` and both `since_duration_seconds` null on both sides, which is no
      evidence about either window's extent at all.

    Any `since_duration_seconds` evidence — matching, mismatched, or one-sided — is
    decided *before* `stats_reset_at` is even consulted: a duration mismatch overrides
    what would otherwise be a `DISJOINT`-grading reset difference, because a user who
    restricted one window with `--since` and not the other (or restricted them
    differently) gets no claim to `HIGH` just because the counters also happen to look
    cleared. See `classify_windows`'s docstring for the full ordering and the decision
    table.

    `since` (the absolute cutoff each run actually bound) and `limit` (how many query
    groups a run sampled) play no part in this classification — see `window_limits` for
    why `limit` is exposed separately instead of being folded into the relation or the
    confidence it earns. `since` remains in the payload purely for human-readable report
    text; `since_duration_seconds` is the field this module actually compares.
    """

    DISJOINT = "disjoint"
    COMPARABLE = "comparable"
    NESTED = "nested"
    INCOMPARABLE = "incomparable"


#: The spec's fixed mapping from relation to confidence. A plain lookup rather than a
#: re-derivation: the grading judgement already happened in `classify_windows`, which
#: earned each relation by the specific fact it detected (or the absence of one, for
#: `INCOMPARABLE`). This table only attaches the number the spec assigns to that fact.
_CONFIDENCE_BY_RELATION: dict[WindowRelation, Confidence] = {
    WindowRelation.DISJOINT: Confidence.HIGH,
    WindowRelation.COMPARABLE: Confidence.HIGH,
    WindowRelation.NESTED: Confidence.MEDIUM,
    WindowRelation.INCOMPARABLE: Confidence.LOW,
}


def confidence_for(relation: WindowRelation) -> Confidence:
    """The confidence grade every downstream verdict may claim for this window relation.

    See `WindowRelation`'s docstring for why each grade is what it is; this function only
    looks the fixed mapping up, so a fifth relation added to the enum without an entry
    here raises `KeyError` rather than silently defaulting to some existing grade.
    """
    return _CONFIDENCE_BY_RELATION[relation]


def _window_string(window: dict[str, object], key: str) -> str | None:
    """`window[key]` if it is a `str`, else `None`.

    A malformed value — a list, a number, a bare `NaN` (Python's `json` module accepts
    `NaN`/`Infinity` as a non-standard extension, and `float("nan") != float("nan")` is
    `True`, which would otherwise let two `NaN` `stats_reset_at` values earn `DISJOINT` at
    `HIGH`) — must read as "unknown" here, never as a value that can win an equality or
    inequality comparison and earn a grade it did not actually demonstrate.
    """
    value = window.get(key)
    return value if isinstance(value, str) else None


def _window_duration_seconds(window: dict[str, object]) -> float | None:
    """`window["since_duration_seconds"]` as a finite `float`, else `None`.

    Two guards beyond a bare `isinstance` check, both closing a route to an unearned
    grade rather than a crash:

    * `bool` is excluded even though `isinstance(True, int)` is `True` in Python — a
      stray boolean must not be read as a duration.
    * Non-finite floats are excluded: `float("inf") == float("inf")` is `True`, so two
      `since_duration_seconds: Infinity` values (again, valid JSON via Python's `json`
      module) would otherwise earn `COMPARABLE` at `HIGH` — the same shape of bug as the
      `NaN`-`stats_reset_at` one `_window_string` guards against, arrived at through
      floats' self-equality rather than strings' self-inequality.
    """
    value = window.get("since_duration_seconds")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    seconds = float(value)
    return seconds if isfinite(seconds) else None


def classify_windows(before: dict[str, object], after: dict[str, object]) -> WindowRelation:
    """How comparable `before` and `after`'s `"window"` objects are, per `WindowRelation`.

    `before` and `after` are the two payloads' top-level dicts (matching the brief's own
    `classify_windows({"window": w}, {"window": dict(w)})`), not the `"window"` objects
    themselves — this function reads `["window"]` out of each.

    Checked in this order — **engine, then `since_duration_seconds`, then
    `stats_reset_at`** — each stage deciding the answer outright rather than leaving it to
    an emergent property of `if`/`elif` fall-through:

    1. **Engines first.** If either `"window"` is missing, not a dict, or its `engine` is
       missing/not a string, or the two `engine` values differ, the answer is
       `INCOMPARABLE` before any other rule runs. A Postgres mean and a Redshift mean
       measure different servers, and nothing below this line can override that — not
       even a `stats_reset_at` pair that would otherwise grade `DISJOINT` at `HIGH`.
    2. **`since_duration_seconds`, decided completely before `stats_reset_at` is ever
       read.** If either side reports a duration (`_window_duration_seconds` returns
       non-`None`):
       * both non-`None` and equal → `COMPARABLE`.
       * anything else — one side `None` and the other not (Ruling 4), or both non-`None`
         but unequal (Ruling 5) → `INCOMPARABLE`, **immediately**. `stats_reset_at` is not
         consulted at all in this branch. A user who restricted one window with `--since`
         and not the other (or restricted them to different durations) gets no claim to
         `HIGH` merely because the counters also happen to look cleared — a differing,
         non-null `stats_reset_at` would otherwise grade `DISJOINT`, and letting that
         override an acknowledged `since` mismatch was Important Finding 2 of this task's
         round-1 review: three cells of the original decision table graded `DISJOINT`/
         `HIGH` where the `since` evidence alone already demanded `INCOMPARABLE`/`LOW`.
    3. **Only once neither side reports a duration** does `stats_reset_at` decide it:
       both non-null and equal → `NESTED` (Ruling 2's double-null case is a `None` on at
       least one side here, which is *not* "both non-null and equal", so it falls through
       correctly); both non-null and different → `DISJOINT` (Ruling 1's null-vs-timestamp
       case is likewise excluded by the same "both non-null" requirement); anything else
       (a null on either side, or both null) → `INCOMPARABLE`.

    **Why the comparability question moved from the absolute `since` cutoff to
    `since_duration_seconds` (Important Finding 3 of the round-1 review).** `--since 7d`
    is a *duration*; Redshift's adapter records the *absolute* cutoff it actually bound
    (`datetime.now() - since`, at microsecond precision), not the duration itself. Two
    runs a week apart with the identical `--since 7d` therefore bind two different
    absolute cutoffs and would never compare equal — so gating `COMPARABLE` on the
    absolute `since` field made it **unreachable from any real Redshift pair**, the exact
    opposite of the design's intent ("comparable duration … clean by construction"). Task
    1's report.py/adapter payload now also records the requested duration itself
    (`since_duration_seconds`, always `None` on Postgres — see `window_facts()` in both
    adapters), and that is what this function actually compares. `since` remains in the
    payload, unused by this function, purely as report text.

    Never raises. Every field is read through `.get`, `_window_string` or
    `_window_duration_seconds`, none of which can raise on a value of unexpected JSON
    type — malformed input degrades to `INCOMPARABLE` rather than earning a grade it did
    not demonstrate (see those two helpers' docstrings for the specific inflation each one
    closes) or raising. Task 7 validates the payload shape before handing artifacts here,
    but this function does not assume that already happened, exactly like `proposal_key`
    above does not assume `evidence` is well-formed.
    """
    before_window = before.get("window")
    after_window = after.get("window")
    if not isinstance(before_window, dict) or not isinstance(after_window, dict):
        return WindowRelation.INCOMPARABLE

    before_engine = before_window.get("engine")
    after_engine = after_window.get("engine")
    if (
        not isinstance(before_engine, str)
        or not isinstance(after_engine, str)
        or before_engine != after_engine
    ):
        return WindowRelation.INCOMPARABLE

    before_duration = _window_duration_seconds(before_window)
    after_duration = _window_duration_seconds(after_window)
    if before_duration is not None or after_duration is not None:
        if (
            before_duration is not None
            and after_duration is not None
            and before_duration == after_duration
        ):
            return WindowRelation.COMPARABLE
        return WindowRelation.INCOMPARABLE

    before_reset = _window_string(before_window, "stats_reset_at")
    after_reset = _window_string(after_window, "stats_reset_at")
    if before_reset is not None and after_reset is not None:
        return WindowRelation.NESTED if before_reset == after_reset else WindowRelation.DISJOINT

    return WindowRelation.INCOMPARABLE


@dataclass(frozen=True)
class WindowLimits:
    """The raw `--limit` each side's window reports, plus the one derived fact Task 6
    actually needs — see `window_limits`.

    `before`/`after` are kept for report text ("run A sampled 500, run B sampled 200").
    **`may_be_sampling_artifact` is the field Task 6 must consult before grading any
    `disappeared` verdict**, not `before == after`: `WindowLimits(None, None) == (None,
    None)` reads as "the limits matched" to that naive comparison, which is the exact
    inversion this dataclass exists to make impossible to reach by accident.
    `may_be_sampling_artifact` is `True` whenever `before` or `after` is `None` (an
    unknown limit carries the same "cannot rule out sampling" weight as a demonstrated
    mismatch — see `window_limits`) or whenever the two known limits differ, and `False`
    only when both sides recorded the *same known* limit.
    """

    before: int | None
    after: int | None

    @property
    def may_be_sampling_artifact(self) -> bool:
        return self.before is None or self.after is None or self.before != self.after


def window_limits(before: dict[str, object], after: dict[str, object]) -> WindowLimits:
    """The `--limit` each side's window reports, wrapped in `WindowLimits`.

    **For Task 6: read `WindowLimits`'s docstring before grading any `disappeared`
    verdict** — consult `.may_be_sampling_artifact`, not `.before == .after`. `limit` is
    how many query groups a run sampled, and it plays no part in `classify_windows`'s
    relation or `confidence_for`'s grade — a differing `limit` does not make two windows
    more or less comparable as *measurements*. But it changes what a missing query group
    *means*: when `.may_be_sampling_artifact` is `True`, a query group present in one
    artifact's `query_groups` and absent from the other's **may be a sampling artifact
    rather than a real disappearance** — a smaller `limit` can drop a group from the
    sampled set with no change to the group's underlying workload at all. Task 6 must not
    report a group as gone on the strength of a smaller `limit` alone, and must surface
    the mismatch (or the unknown) alongside any `disappeared` verdict it grades, rather
    than silently trusting the absence.

    Deliberately kept out of `classify_windows`'s relation and out of `confidence_for`'s
    grade (Ruling 6): folding it in there would let a sampling difference silently change
    a verdict's stated confidence instead of being reported as the distinct fact it is.

    `.before`/`.after` are `None` for a side whose `"window"` is missing/not a dict, or
    whose `limit` is not a genuine `int` — excluding `bool` (`isinstance(True, int)` is
    `True` in Python, and a stray boolean must not pass as a sampled-group count) as well
    as the null case both engines report whenever `--limit` was not passed. Never raises.
    """
    before_window = before.get("window")
    after_window = after.get("window")
    before_limit = before_window.get("limit") if isinstance(before_window, dict) else None
    after_limit = after_window.get("limit") if isinstance(after_window, dict) else None
    return WindowLimits(
        before=before_limit
        if isinstance(before_limit, int) and not isinstance(before_limit, bool)
        else None,
        after=after_limit
        if isinstance(after_limit, int) and not isinstance(after_limit, bool)
        else None,
    )
