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

import re
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Final

from sqlquality.models import Confidence, cost_share_of


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


def _payload_engine(payload: dict[str, object]) -> str | None:
    """The engine this artifact's window reports, or `None` when it is missing/not a string.

    Reads `payload["window"]["engine"]`, the same field `classify_windows` compares, rather
    than the top-level `payload["engine"]`. `advise_payload` writes both from one argument,
    so a real artifact cannot disagree with itself here; reading the field this module
    already reads keeps one source of truth for "which server did this run measure", and
    keeps `artifact_incomparabilities` consistent with the relation `classify_windows`
    grades from the same value.
    """
    window = payload.get("window")
    if not isinstance(window, dict):
        return None
    engine = window.get("engine")
    return engine if isinstance(engine, str) else None


class ArtifactMismatch(str, Enum):
    """A pair-level fact that makes two artifacts' **query-group identities** incommensurable
    — deliberately a separate concept from both `Absence` and `WindowRelation.INCOMPARABLE`.

    This module now distinguishes three different ways a comparison can fail, and collapsing
    any two of them is how this task has produced false claims four times:

    * **`Absence` / `degraded`** — "one run could not *look*." The coordinate system is
      shared; a reading is missing from it. Handled per absence, per proposal, by
      `_absence_disclosure`.
    * **`WindowRelation.INCOMPARABLE`** — "both runs looked, and their measurements are
      commensurable, but the *windows* they measured make the comparison weak." Handled as a
      confidence ceiling: a verdict is still claimed, just at `LOW`.
    * **`ArtifactMismatch` (this enum)** — "the two runs do not share a coordinate system at
      all." No reading is missing and no window is at fault: the identifiers the two
      artifacts use for the same query group are simply different strings, so *every*
      group-level comparison between them is meaningless rather than weak. There is no
      confidence at which such a comparison could be reported, so `verdicts` claims no
      group-level outcome and no mean at all — see `artifact_incomparabilities`.

    `REDACTION` (fix round 5, review finding B). `--keep-literals` changes the canonical
    query text `fingerprint_id` hashes, so two runs differing only in that flag produce
    disjoint `query_groups` key sets from an identical workload. Before this fix `verify`
    read that as `DISAPPEARED` — "Cited query group(s) no longer appear in the after run" —
    for a query still running, with no degradation anywhere and the recommendation genuinely
    applied. That is the governing rule's exact violation (an absence produced by one run's
    own settings reported as a measurement about the user's database) through a door the
    `degraded`-based gate structurally cannot see.

    `ENGINE` (Task 7, closing Task 6's final open finding). Two artifacts from two different
    engines describe two different servers, so no query group in one corresponds to anything
    in the other — and a Redshift `digest` and a Postgres one are computed from differently
    normalized text besides. `classify_windows` has always graded such a pair `INCOMPARABLE`,
    but that is only a confidence ceiling: a cross-engine pair still reached a group-level
    outcome, and a `before`-only group was reported as `DISAPPEARED, mean_before=100.0,
    mean_after=None, "no longer appear"` — the same false claim `REDACTION` fixes, arrived at
    through a wider door. `LOW` is not a grade at which "the query stopped running" becomes
    honest, so this is an incomparability rather than a confidence cap. `verify`'s CLI refuses
    such a pair outright; this member is what protects a direct API caller, which was
    unreachable-by-CLI but never unreachable-by-import.
    """

    REDACTION = "redaction"
    ENGINE = "engine"


#: What each `ArtifactMismatch` makes incommensurable, as a noun phrase for the disclosure
#: sentence. A plain lookup, like `_CONFIDENCE_BY_RELATION` and `_ABSENCE_DEPENDENCIES`, so a
#: second member added without an entry raises `KeyError` rather than silently disclosing
#: nothing; `test_every_artifact_mismatch_is_classified` pins it directly.
_MISMATCH_EFFECT: Final[dict[ArtifactMismatch, str]] = {
    ArtifactMismatch.REDACTION: (
        "every query-group digest, and the key of every statement-scoped proposal "
        "(ADV005's leading-wildcard branch, ADV006), which is a fingerprint"
    ),
    ArtifactMismatch.ENGINE: (
        "every query group, every relation and every timing the two artifacts record"
    ),
}


@dataclass(frozen=True)
class Incomparability:
    """One reason two artifacts cannot be compared at query-group level at all.

    `mismatch` is the machine-readable kind; `detail` is a complete sentence naming the two
    values that disagree and what they make incommensurable. **Task 7 (the CLI) should call
    `artifact_incomparabilities` directly and refuse the comparison up front**, rendering
    every `detail`, rather than relying on each individual verdict's `note` — a user handed
    two artifacts with different redaction settings wants one clear refusal, not the same
    sentence repeated once per proposal.
    """

    mismatch: ArtifactMismatch
    detail: str


def artifact_incomparabilities(
    before: dict[str, object], after: dict[str, object]
) -> tuple[Incomparability, ...]:
    """Every reason `before` and `after` do not share a query-group coordinate system.

    Empty when they do. **Callable independently of `verdicts`, and meant to be** — the
    `verify` command consults it before rendering anything and refuses the pair, rendering
    every `detail` once instead of once per proposal (see `Incomparability`).

    `ENGINE` is checked first, exactly as `classify_windows` checks it first: a Postgres mean
    and a Redshift mean measure different servers, and no other fact can override that. Read
    through `_payload_engine` (`window["engine"]`), reported for **either** a readable
    disagreement or an unreadable value on one/both sides — the same two conditions, for the
    same reason, as `REDACTION` below. Both-unreadable matters on its own: two `None`s compare
    *equal*, so without the readability arm a pair of truncated artifacts would silently earn a
    full group-level verdict on the inference `WindowRelation.INCOMPARABLE`'s docstring already
    forbids for `stats_reset_at`.

    `REDACTION`: `payload["redacted"]` is `not keep_literals` (cli.py), and redaction
    rewrites the canonical text that `fingerprint_id` hashes into a `digest`. So two runs
    over an identical workload that differ in `--keep-literals` agree on nothing in
    `query_groups`: same queries, disjoint keys. Reported for **either** of two conditions,
    and the second is not merely defensive tidiness:

    * the two flags are readable and differ — the reachable case, reproduced from two real
      `advise` runs;
    * either flag is not a readable `bool` — then whether the two runs canonicalized their
      query text the same way *cannot be established*, and this module does not assume an
      unverifiable premise. This follows `classify_windows`'s existing precedent exactly:
      a missing or non-string `engine` yields `INCOMPARABLE` rather than being read as "the
      engines presumably match". Every artifact `advise` has ever written carries `redacted`
      as a genuine `bool` (it is a required argument of `advise_payload`, present since
      before this feature), so this arm is unreachable from a well-formed artifact — it
      exists for the truncated or hand-edited one, and Task 7's shape validation is the
      other place it should be caught.

    Never raises: every field is read through `.get` and type-checked, like every other
    entry point in this module.
    """
    found: list[Incomparability] = []
    engine_effect = _MISMATCH_EFFECT[ArtifactMismatch.ENGINE]
    before_engine = _payload_engine(before)
    after_engine = _payload_engine(after)
    if before_engine is None or after_engine is None:
        found.append(
            Incomparability(
                mismatch=ArtifactMismatch.ENGINE,
                detail=(
                    "At least one run does not record a readable engine in its window "
                    f"(before={before_engine!r}, after={after_engine!r}), so whether the two "
                    "runs measured the same database server cannot be established — and that "
                    f"is what decides whether {engine_effect} can be compared at all. No "
                    "query-group comparison is claimed."
                ),
            )
        )
    elif before_engine != after_engine:
        found.append(
            Incomparability(
                mismatch=ArtifactMismatch.ENGINE,
                detail=(
                    f"The two runs measured different engines (before={before_engine}, "
                    f"after={after_engine}). Those are two different database servers, so "
                    f"{engine_effect} belongs to a different system on each side: nothing in "
                    "one artifact corresponds to anything in the other. No query-group "
                    "comparison is claimed."
                ),
            )
        )
    before_redacted = before.get("redacted")
    after_redacted = after.get("redacted")
    effect = _MISMATCH_EFFECT[ArtifactMismatch.REDACTION]
    if not isinstance(before_redacted, bool) or not isinstance(after_redacted, bool):
        found.append(
            Incomparability(
                mismatch=ArtifactMismatch.REDACTION,
                detail=(
                    "At least one run does not record a readable `redacted` flag "
                    f"(before={before_redacted!r}, after={after_redacted!r}), so whether "
                    "the two runs canonicalized their query text the same way cannot be "
                    f"established — and redaction determines {effect}. No query-group "
                    "comparison is claimed."
                ),
            )
        )
    elif before_redacted != after_redacted:
        found.append(
            Incomparability(
                mismatch=ArtifactMismatch.REDACTION,
                detail=(
                    "The two runs disagree about literal redaction "
                    f"(before redacted={before_redacted}, after redacted={after_redacted}): "
                    "`--keep-literals` changes the canonical query text every digest is "
                    f"computed from, so {effect} differs between these two artifacts. The "
                    "query groups are not missing from the after run — they are recorded "
                    "under different identifiers, so no query-group comparison is claimed."
                ),
            )
        )
    return tuple(found)


class VerifyOutcome(str, Enum):
    """How a `before`-proposal's cited workload evidence compares in `after`.

    `NOT_APPLIED` and `UNOBSERVABLE` are decided from `applied` alone, in `verdicts`, and
    take precedence over every mean-based outcome below regardless of what the means say:
    an unapplied change did not cause anything, so it cannot be credited with a mean
    improvement someone else's fix (or workload shift) produced, and an unobservable one
    cannot be credited either way. See `verdicts`'s docstring for the full decision order.
    """

    IMPROVED = "improved"
    UNCHANGED = "unchanged"
    REGRESSED = "regressed"
    DISAPPEARED = "disappeared"
    NOT_APPLIED = "not_applied"
    UNOBSERVABLE = "unobservable"


@dataclass(frozen=True)
class ProposalVerdict:
    """One `before.json` proposal's verdict against a later `after.json` run.

    `applied` is `bool | None` — `None` means "this run's `physical_state` (or, for
    ADV105, its proposal list) could not tell us," never a guessed `False`. See
    `_detect_applied` for the per-code-family detection this comes from, and Ruling 1 in
    the Task 6 report for why the distinction is load-bearing.

    `mean_before`/`mean_after` are the call-weighted mean per call across this proposal's
    cited query groups (Ruling 5) — never `cost_share`, which is carried alongside purely
    as *context* (`cost_share_before`/`cost_share_after`, both from `cost_share_of`) about
    whether the finding still matters, not whether it got faster. Either mean can be
    `None` when there is nothing usable to average (Ruling 7): no citations, no group
    resolving on that side, or a resolving group whose calls summed to zero.

    `note` is free text explaining the verdict — in particular, disclosing a collision,
    a `--limit` mismatch that forbids `DISAPPEARED`, a partial citation match, or the
    "applied but unchanged" case the whole feature exists to surface.
    """

    key: tuple[str, ...]
    code: str
    applied: bool | None
    outcome: VerifyOutcome
    confidence: Confidence
    mean_before: float | None
    mean_after: float | None
    cost_share_before: float | None
    cost_share_after: float | None
    note: str


#: Relative change in call-weighted mean-per-call time, beyond which a proposal is graded
#: `IMPROVED` or `REGRESSED` rather than `UNCHANGED`. A single named constant so a reviewer
#: arguing with the number only has to change it in one place. 10% is chosen as comfortably
#: above the run-to-run noise an unrelated cache-warmth or plan-choice difference would
#: produce in a repeated mean, while still catching a genuinely meaningful change — not
#: derived from any specific workload, and deliberately conservative (a real change of, say,
#: 8% is reported `UNCHANGED` rather than risk crediting noise as an improvement).
RELATIVE_CHANGE_THRESHOLD: Final[float] = 0.10

#: `Confidence`'s three grades, ranked low-to-high, so a verdict's confidence can be
#: *capped* (never raised) against a second ceiling beyond the window relation's — see
#: `_cap_confidence` and its ADV105 use in `verdicts`.
_CONFIDENCE_RANK: dict[Confidence, int] = {
    Confidence.LOW: 0,
    Confidence.MEDIUM: 1,
    Confidence.HIGH: 2,
}


def _cap_confidence(value: Confidence, cap: Confidence) -> Confidence:
    """`value`, or `cap` if `value` outranks it — never the reverse. A second, narrower
    ceiling than the window relation's own (`confidence_for`), for evidence that is weaker
    for a reason the window comparison itself does not capture — see ADV105's use in
    `verdicts`.
    """
    return value if _CONFIDENCE_RANK[value] <= _CONFIDENCE_RANK[cap] else cap


def _relation_key(evidence: dict[str, object]) -> str | None:
    """`"schema.table"`, matching `physical_state`'s own key (`str(Relation)`), or `None`
    when `evidence` carries no relation at all — the statement-scoped shapes (ADV005's
    wildcard branch, ADV006), which have no `physical_state` entry to read regardless of
    what this function returns.
    """
    schema = evidence.get("schema")
    table = evidence.get("table")
    if isinstance(schema, str) and isinstance(table, str):
        return f"{schema}.{table}"
    return None


def _physical_entry(payload: dict[str, object], relation_key: str) -> dict[str, object]:
    """`payload["physical_state"][relation_key]`, or `{}` when the top-level key, the
    relation's own key, or the entry itself is missing or malformed.

    Collapsing every one of those routes into the same empty dict is deliberate, not
    laziness: a subsequent `.get(field)` against `{}` returns `None` exactly like a
    present-but-null field would, which is the one null-handling path every `_applied_*`
    helper below needs. In particular, a relation that does not appear in `after`'s
    `physical_state` at all — which happens whenever the proposal that used to name it is
    the *only* reason that relation was ever included there, and applying the change made
    that proposal (and so the relation) vanish from `after`'s own run — must read exactly
    like a present-but-null entry: "this run cannot tell you," never a guessed `False`. See
    the Task 6 report for why this is a real gap worth flagging, not a hypothetical one.
    """
    state = payload.get("physical_state")
    if not isinstance(state, dict):
        return {}
    entry = state.get(relation_key)
    return entry if isinstance(entry, dict) else {}


def _index_leading_columns_match(indexes: object, columns: tuple[str, ...]) -> bool | None:
    """Whether some index in `indexes` leads with `columns`, in the same order.

    `None` when `indexes` itself is not a list (covers both `null` — already excluded by
    every caller before this runs — and any other malformed shape): a malformed value must
    read as "unknown," never earn `True` or `False` it did not demonstrate, matching this
    module's established discipline elsewhere (`_window_string`, `_window_duration_seconds`).
    """
    if not isinstance(indexes, list):
        return None
    for entry in indexes:
        if not isinstance(entry, dict):
            continue
        entry_columns = entry.get("columns")
        if not isinstance(entry_columns, list) or len(entry_columns) < len(columns):
            continue
        leading = entry_columns[: len(columns)]
        if all(isinstance(c, str) for c in leading) and tuple(leading) == columns:
            return True
    return False


#: ADV001/ADV004/ADV007/ADV008 — create-index rules. Applied when some index in `after`'s
#: `indexes` leads with the proposed `columns`, in order.
_CREATE_INDEX_CODES = frozenset({"ADV001", "ADV004", "ADV007", "ADV008"})

#: ADV002/ADV003 — drop-index rules. Applied when the named index is absent from `after`.
_DROP_INDEX_CODES = frozenset({"ADV002", "ADV003"})

#: ADV101/ADV102/ADV103 — Redshift sortkey/diststyle rules. See `_applied_redshift_key`.
_REDSHIFT_KEY_CODES = frozenset({"ADV101", "ADV102", "ADV103"})

#: ADV104 — vacuum/analyze. Applied when `unsorted`/`stats_off` (whichever this specific
#: proposal's evidence carries) fell below its own recorded baseline.
_MAINTENANCE_CODE = "ADV104"

#: ADV105 — Amazon Redshift Advisor. Verified at proposal level, not from `physical_state`
#: — see `_applied_advisor`.
_ADVISOR_CODE = "ADV105"

#: ADV301 — materialize a view. Applied when `after`'s `is_ordinary_table` is `True`.
_MATERIALIZE_CODE = "ADV301"

#: ADV005, ADV006 (a query rewrite) and ADV303 (needs a manifest an offline `verify`
#: lacks) — nothing about their application is observable from this artifact pair at all,
#: per the design spec. `applied` is always `None` for these, never derived from
#: `physical_state`.
_UNOBSERVABLE_CODES = frozenset({"ADV005", "ADV006", "ADV303"})


def _applied_create_index(
    evidence: dict[str, object],
    before_phys: dict[str, object],
    after_phys: dict[str, object],
) -> bool | None:
    """ADV001/ADV004/ADV007/ADV008 — see `_CREATE_INDEX_CODES`.

    **Ruling 1.** `None`, never a guessed `False`, whenever either side's `indexes` is
    `null`: a `before` run that never fetched this relation's indexes cannot establish
    there was nothing to begin with, and an `after` run that did not fetch them cannot say
    whether one now exists. Checking *both* sides (not only `after`) is deliberate — see
    the Task 6 report.
    """
    columns = evidence.get("columns")
    if not (isinstance(columns, (list, tuple)) and all(isinstance(c, str) for c in columns)):
        return None
    before_indexes = before_phys.get("indexes")
    after_indexes = after_phys.get("indexes")
    if before_indexes is None or after_indexes is None:
        return None
    return _index_leading_columns_match(after_indexes, tuple(columns))


def _applied_drop_index(
    evidence: dict[str, object],
    before_phys: dict[str, object],
    after_phys: dict[str, object],
) -> bool | None:
    """ADV002/ADV003 — see `_DROP_INDEX_CODES`. `None` under the identical Ruling 1
    null-gate as `_applied_create_index`; otherwise `True` iff the named index is absent
    from `after`'s `indexes`.
    """
    index_name = evidence.get("index")
    if not isinstance(index_name, str):
        return None
    before_indexes = before_phys.get("indexes")
    after_indexes = after_phys.get("indexes")
    if before_indexes is None or after_indexes is None:
        return None
    if not isinstance(after_indexes, list):
        return None
    still_present = any(
        isinstance(entry, dict) and entry.get("name") == index_name for entry in after_indexes
    )
    return not still_present


#: Unwraps an outer `AUTO(...)` from a Redshift `diststyle` string, if present — AWS
#: documents `AUTO(ALL)`, `AUTO(EVEN)` and `AUTO(KEY(col))` alongside the un-wrapped forms,
#: so a shape check must look through the wrapper rather than treat it as unrecognized.
_DISTSTYLE_AUTO_RE = re.compile(r"^AUTO\(\s*(.*)\s*\)$", re.IGNORECASE | re.DOTALL)

#: The column name inside `KEY(column)`, once any `AUTO(...)` wrapper is already stripped.
_DISTSTYLE_KEY_RE = re.compile(r"^KEY\(\s*([^)]+?)\s*\)$", re.IGNORECASE)


def _diststyle_shape(diststyle: object) -> tuple[str, str | None] | None:
    """Classify a Redshift `diststyle` value into one of AWS's documented shapes —
    `("even", None)`, `("all", None)`, `("key", column)` — or `None` when `diststyle` is
    not a string, or is a string matching none of them.

    **Why an unrecognized shape reads as `None`, never `False` (Concern 1, fix round 2,
    overruling this function's first version).** The closed-vocabulary argument that
    version made would be sound if the vocabulary were *verified* — it is not. This
    project carries a standing, documented limitation (CHANGELOG.md, README.md):
    Redshift's introspection SQL, `svv_table_info` included, has never been executed
    against a live cluster. Every shape this function recognizes comes from AWS's
    documentation alone, not observation, and `AUTO(...)` already nests three ways
    (`AUTO(ALL)`, `AUTO(EVEN)`, `AUTO(KEY(col))`) — good evidence the space is larger than
    a flat enum read off a doc page could confidently claim to have enumerated. Reporting
    "this did not parse" as though it were "measured, and it is not a key/ALL/EVEN
    distribution" is the identical shape of error Task 2's Critical fixed one layer down
    (a never-fetched relation reported as hard fact) and this task's own Concern 1 fix:
    a check that could not run must be disclosed, never assumed. `False` stays reserved
    for a value this function *did* recognize, just not as the shape being checked for —
    a recognized `EVEN` when the caller asked "is this `KEY(customer_id)`" is a genuine
    negative, not an unknown.
    """
    if not isinstance(diststyle, str):
        return None
    text = diststyle.strip()
    auto_match = _DISTSTYLE_AUTO_RE.match(text)
    if auto_match:
        text = auto_match.group(1).strip()
    if text.upper() == "EVEN":
        return ("even", None)
    if text.upper() == "ALL":
        return ("all", None)
    key_match = _DISTSTYLE_KEY_RE.match(text)
    if key_match:
        return ("key", key_match.group(1).strip())
    return None


def _applied_redshift_key(
    code: str,
    evidence: dict[str, object],
    before_phys: dict[str, object],
    after_phys: dict[str, object],
) -> tuple[bool | None, str | None]:
    """ADV101/ADV102/ADV103 — see `_REDSHIFT_KEY_CODES`.

    **Applied means the observed state now matches the *proposed target*, never merely
    "the field changed."** Crediting any change to `sortkey1`/`diststyle` — this
    function's first version, reverted after review — would tell a user who set a
    *different* sortkey or distkey than the one recommended that their own, unrelated
    change was the one this tool proposed: a false statement about their database, and a
    checkable one, since ADV101/ADV102 both carry the proposed `column` in evidence and
    ADV103 proposes `DISTSTYLE ALL` specifically.

    - ADV101: `True` iff `after`'s `sortkey1` equals the proposed `column`, exactly.
      `sortkey1` is a bare column name with nothing to parse, so no shape question
      arises here — but a `null` `sortkey1` is still ambiguous even once
      `is_ordinary_table` is known `True` (Finding 3, fix round 3): `propose_sortkey`'s
      own docstring treats a null `sortkey1` as "could not be read," not "no sort key
      set," and drops to `LOW` for exactly that reason. Reading `None == column` as a
      plain equality would silently grade this `False` — this function's first version
      did, inconsistently with `_diststyle_shape` three lines away already reading a
      null `diststyle` as unrecognized-hence-`None` — so `after`'s `sortkey1` is checked
      for `None` explicitly before the comparison, matching ADV102/ADV103's discipline
      rather than contradicting it.
    - ADV102: `after`'s `diststyle` classified (`_diststyle_shape`) as `("key",
      column)` with `column` equal to the proposed one -> `True`; classified as
      anything else recognized (`EVEN`, `ALL`, or `KEY(`some other column`)`) -> `False`,
      a genuine mismatch; unrecognized -> `None` (Concern 1, fix round 2).
    - ADV103: classified as `("all", None)` -> `True`; classified as `EVEN` or any `KEY`
      shape -> `False`; unrecognized -> `None`.

    **Ruling 1.** `None` when either side's `is_ordinary_table` is `null` (`sortkey1`/
    `diststyle` are themselves ambiguous `None`s until `is_ordinary_table` is known — see
    `RedshiftWorkloadAdapter.physical_state`'s docstring), or when ADV101/ADV102's
    evidence does not carry the `column` this comparison needs (defensive — a real
    proposal from either rule always does). Both routes carry no note of their own here;
    `verdicts`'s generic "cannot be observed" wording already covers them.
    """
    if before_phys.get("is_ordinary_table") is None or after_phys.get("is_ordinary_table") is None:
        return None, None
    after_diststyle = after_phys.get("diststyle")
    if code == "ADV103":
        shape = _diststyle_shape(after_diststyle)
        if shape is None:
            return None, (
                "after's diststyle did not match any recognized shape (Redshift's "
                "introspection has never been verified against a live cluster); cannot "
                "tell whether DISTSTYLE ALL was applied."
            )
        return shape[0] == "all", None
    column = evidence.get("column")
    if not isinstance(column, str):
        return None, None
    if code == "ADV101":
        after_sortkey1 = after_phys.get("sortkey1")
        if after_sortkey1 is None:
            return None, (
                "after's sortkey1 could not be read (SQL NULL), even though the relation "
                "is an ordinary table; whether the proposed SORTKEY was applied is "
                "unknowable, not a measured absence."
            )
        return after_sortkey1 == column, None
    # ADV102
    shape = _diststyle_shape(after_diststyle)
    if shape is None:
        return None, (
            "after's diststyle did not match any recognized shape (Redshift's "
            "introspection has never been verified against a live cluster); cannot tell "
            "whether the proposed DISTKEY was applied."
        )
    return shape == ("key", column), None


def _applied_maintenance(
    evidence: dict[str, object],
    before_phys: dict[str, object],
    after_phys: dict[str, object],
) -> bool | None:
    """ADV104 — see `_MAINTENANCE_CODE`. Applied when `unsorted` (the VACUUM branch) or
    `stats_off` (the ANALYZE branch) — whichever this specific proposal's evidence
    carries — is lower in `after` than the value recorded in evidence at proposal time.

    **Ruling 1** (the `is_ordinary_table` gate, identical to `_applied_redshift_key`) and
    **Ruling 7** (a non-numeric or missing baseline/`after` value reads as "cannot tell,"
    never coerced into a comparison) both apply here.
    """
    if before_phys.get("is_ordinary_table") is None or after_phys.get("is_ordinary_table") is None:
        return None
    if "unsorted" in evidence:
        field = "unsorted"
    elif "stats_off" in evidence:
        field = "stats_off"
    else:
        return None
    baseline = evidence.get(field)
    after_value = after_phys.get(field)
    if isinstance(baseline, bool) or not isinstance(baseline, (int, float)):
        return None
    if isinstance(after_value, bool) or not isinstance(after_value, (int, float)):
        return None
    return after_value < baseline


def _applied_materialize(
    before_phys: dict[str, object], after_phys: dict[str, object]
) -> bool | None:
    """ADV301 — see `_MATERIALIZE_CODE`. Applied when `after`'s `is_ordinary_table` is
    `True` — the relation is no longer a view. `None` under the same Ruling 1 gate as
    `_applied_redshift_key`.
    """
    before_ordinary = before_phys.get("is_ordinary_table")
    after_ordinary = after_phys.get("is_ordinary_table")
    if before_ordinary is None or after_ordinary is None:
        return None
    return after_ordinary is True


#: Every capability and degradation name either workload adapter can record in a payload's
#: `degraded` list. Duplicated as plain strings rather than imported for the same reason
#: `_diststyle_shape`'s regexes are duplicated: `workload/postgres.py` and
#: `workload/redshift.py` transitively pull in this project's live-connection machinery,
#: which `verify.py` must never import (see this module's own docstring). Only the string
#: values are needed here, and they are the adapters' own capability names, not project
#: logic that could drift between the two copies.
_CAP_WORKLOAD: Final = "workload"
_CAP_STATS_RESET: Final = "stats_reset"
_CAP_SCHEMA: Final = "schema"
_CAP_TABLE_FACTS: Final = "table_facts"
_CAP_NDV: Final = "ndv"
_CAP_INDEXES: Final = "indexes"
_CAP_ADVISOR: Final = "advisor"
_DEGRADATION_READ_ONLY: Final = "read_only"
_DEGRADATION_PHYSICAL_FACTS_GAP: Final = "physical_facts_gap"

#: The complete vocabulary of `payload["degraded"][*]["capability"]` values this version of
#: `verify` has *classified* — i.e. for which someone has decided, in
#: `_ABSENCE_DEPENDENCIES` below, which of `verify`'s absence-derived conclusions the
#: degradation can fabricate. Pinned against the adapters' own module constants by
#: `test_every_degradation_either_adapter_can_record_is_classified`, so a seventh capability
#: added to either adapter reddens the suite until it has been classified here: the omission
#: this module's recurring defect has arrived through three times cannot be made silently.
#:
#: A name *outside* this set, encountered at runtime, is the same situation seen from the
#: other side — an artifact written by a newer `sqlquality` than the `verify` reading it —
#: and `_absence_disclosure` treats it as blocking **every** absence, because a degradation
#: this version cannot name is a degradation whose blind spot it cannot bound.
KNOWN_DEGRADATIONS: Final[frozenset[str]] = frozenset(
    {
        _CAP_WORKLOAD,
        _CAP_STATS_RESET,
        _CAP_SCHEMA,
        _CAP_TABLE_FACTS,
        _CAP_NDV,
        _CAP_INDEXES,
        _CAP_ADVISOR,
        _DEGRADATION_READ_ONLY,
        _DEGRADATION_PHYSICAL_FACTS_GAP,
    }
)


class Absence(str, Enum):
    """Each absence in an `advise --json` artifact that `verify` reads as *evidence about
    the user's database*, rather than as a fact about the artifact.

    **This enum exists because one defect class has now recurred three times through three
    different doors, and each time the mechanism was identical: an absence produced by one
    run's own limitations was read as a measurement.** `physical_state` scoped to that
    run's proposal relations (Task 6 fix round 1); `query_groups` scoped to that run's own
    citations (fix round 3); a denied capability read emptying the data (fix round 3's
    ADV105 fix, applied to exactly one capability of nine — finding N1). Naming the
    absences instead of the capabilities inverts the question that kept being answered
    incompletely: not "which capability did we remember to check?" but "which conclusions
    rest on something *not* being there, and what could have produced that emptiness
    without the database having changed at all?"

    A member here is a claim `verify` makes **from nothing being present**. An absence is
    only evidence when every read that would have put something there actually ran — which
    is what `_ABSENCE_DEPENDENCIES` records and `_absence_disclosure` enforces. Adding a
    member without an entry in that table raises `KeyError` at first use rather than
    defaulting to "nothing can fabricate this", and
    `test_every_absence_verify_reads_as_evidence_is_classified` pins it directly.

    * `CITED_QUERY_GROUP` — "a query group this proposal cited is not in `after`'s
      `query_groups`". Read three ways in `verdicts`: as `DISAPPEARED`, as the
      `applied is None` "possibly addressed by a query rewrite" reading, and (negatively)
      as the absence of a comparable `mean_after`.
    * `ADVISOR_RECOMMENDATION` — "this ADV105 recommendation is not in `after`'s
      `proposals`". ADV105 is the one code whose applied signal is a *proposal's* absence
      rather than a `physical_state` delta (see `_applied_advisor`), so it is the one code
      for which a missing proposal is a claim about the database rather than about the
      artifact.

    **Absences deliberately not members, each checked rather than assumed:**

    * *A relation missing from `physical_state`, or a null field within an entry.* Both
      adapters fold every capability denial into `null` themselves — `have_facts`/
      `have_indexes` in `PostgresWorkloadAdapter.physical_state` and
      `RedshiftWorkloadAdapter.physical_state` gate on `... not in self.degraded` — and
      `_physical_entry` maps a wholly missing relation onto the same `null`s. Every
      `_applied_*` helper then returns `None` on a `null`, never a guessed `False`
      (Ruling 1). So the denial is already disclosed one layer down, in the artifact
      itself, and re-checking `degraded` here would be a second, weaker copy of a gate
      that is already load-bearing and already pinned.
    * *A null `window` field.* `classify_windows` reads a null `stats_reset_at` as
      `INCOMPARABLE` (never as "equal to the other null"), and `WindowLimits.
      may_be_sampling_artifact` reads a null `limit` as "cannot rule out sampling". Both
      already treat the absence as unknown rather than as a measurement, which is exactly
      what a `CAP_STATS_RESET` denial needs.
    * *Only N of M cited groups resolving in `after`* (the partial-citation note). That
      note states a property of the artifact ("N of M were present … the figures above
      reflect only those") and draws no conclusion about the database from the M−N that
      were not, so there is nothing for a degradation to fabricate.
    * *`cost_share_after` being `None`.* Ruling 4: `None` there already means "could not be
      read off unambiguously", identical for an absent and an ambiguous key.
    * *A proposal absent from `after` for any code other than ADV105.* `verdicts` derives
      `applied` for every other code from `physical_state` alone, and Ruling 10 keeps
      "new in `after`" out of this module entirely.
    """

    CITED_QUERY_GROUP = "cited_query_group"
    ADVISOR_RECOMMENDATION = "advisor_recommendation"


#: Which degradations can produce each `Absence` **without the database having changed** —
#: the general form of finding N1, which fix round 3 solved for one cell of this table only.
#: A plain lookup, like `_CONFIDENCE_BY_RELATION`, so a third `Absence` member added without
#: an entry raises `KeyError` rather than silently earning "nothing can fabricate this".
#:
#: Capability by capability, for both engines:
#:
#: * `workload` — the whole of `query_groups`. `_query_groups_payload` emits one entry per
#:   `workload.stats` group, and a denied read leaves `stats` empty: `_run` swallows the
#:   failure into `degraded` and returns `[]` (postgres.py, redshift.py), while
#:   `self._window_limit` is recorded regardless, so both sides report the same `limit` and
#:   Ruling 2's `may_be_sampling_artifact` gate does *not* fire. Blocks
#:   `CITED_QUERY_GROUP`. It also empties `aggregation.tables`, and ADV105's Advisor rows
#:   are fetched for `aggregation.tables | facts` (redshift.py's `propose`), so it blocks
#:   `ADVISOR_RECOMMENDATION` too — a route the round-3 fix missed entirely.
#: * `schema` — no column metadata, so the aggregator cannot qualify relations and
#:   `aggregation.tables` shrinks or empties; same consequence for ADV105's row fetch.
#:   Blocks `ADVISOR_RECOMMENDATION`. Does *not* touch `workload.stats`, so
#:   `query_groups` is unaffected.
#: * `table_facts` — empties `facts`, the other half of ADV105's fetch set: a relation
#:   reached only through `star_tables` has facts but never appears in
#:   `aggregation.tables`, so its Advisor row is fetched solely because `facts` covers it.
#:   Blocks `ADVISOR_RECOMMENDATION`. Its effect on `physical_state` is already `null`, per
#:   `Absence`'s docstring.
#: * `advisor` — Advisor's own rows; `_advisor_rows` records the failure and returns `[]`,
#:   producing the identical empty list a resolved recommendation does. Blocks
#:   `ADVISOR_RECOMMENDATION` (fix round 3's original fix, preserved).
#: * `indexes` — Postgres index lists. Postgres emits no ADV105, and this read feeds no
#:   query group, so it fabricates neither absence; it is already `null`-disclosed in
#:   `physical_state`.
#: * `ndv` — `pg_stats` selectivity, Postgres-only. Can suppress a Postgres *proposal*, but
#:   no Postgres code reads a proposal's absence as evidence (Ruling 10, and ADV105 is
#:   Redshift-only), and it feeds no query group. Fabricates neither absence.
#: * `stats_reset` — `pg_stat_database.stats_reset`, Postgres-only; only ever reaches the
#:   `window` object, where null already means `INCOMPARABLE`. Fabricates neither absence.
#: * `read_only` — Redshift's session could not be pinned read-only. Reads no data at all,
#:   so it removes nothing from any payload key. Fabricates neither absence — and it is the
#:   reason `_degraded_names` must compare capability *names* rather than merely asking
#:   whether `degraded` is non-empty (finding N3): this entry is common on real clusters and
#:   would otherwise collapse every ADV105 verdict to "unknown".
#: * `physical_facts_gap` — Redshift relations with a hot predicate that `svv_table_info`
#:   omits, which `propose` declines to grade rather than guessing. It suppresses
#:   ADV101/ADV102/ADV103 proposals, whose applied signal is `physical_state`, never a
#:   proposal's absence; ADV105's own row fetch is unaffected by it. Fabricates neither
#:   absence.
_ABSENCE_DEPENDENCIES: Final[dict[Absence, frozenset[str]]] = {
    Absence.CITED_QUERY_GROUP: frozenset({_CAP_WORKLOAD}),
    Absence.ADVISOR_RECOMMENDATION: frozenset(
        {_CAP_ADVISOR, _CAP_WORKLOAD, _CAP_SCHEMA, _CAP_TABLE_FACTS}
    ),
}

#: How to name each absence inside `_absence_disclosure`'s sentence. Separate from the enum
#: values so the wording can be a readable clause without the enum member becoming prose.
_ABSENCE_SUBJECTS: Final[dict[Absence, str]] = {
    Absence.CITED_QUERY_GROUP: "a cited query group's absence from that run's query_groups",
    Absence.ADVISOR_RECOMMENDATION: "this recommendation's absence from that run's proposals",
}


def _degraded_names(payload: dict[str, object]) -> frozenset[str]:
    """The `capability` names `payload["degraded"]` records, as a set.

    `payload["degraded"]` entries are `{"capability": ..., "reason": ...}` dicts (see
    `report.py`'s `advise_payload`); a non-list `degraded`, a non-dict entry, or a
    non-string `capability` is ignored rather than raised on, matching this module's
    discipline everywhere else. An **absent** `degraded` key likewise yields the empty set:
    an artifact predating the key is not an artifact that declared a denial, and Task 7
    validates payload shape before handing artifacts here.
    """
    degraded = payload.get("degraded")
    if not isinstance(degraded, list):
        return frozenset()
    names: set[str] = set()
    for entry in degraded:
        if not isinstance(entry, dict):
            continue
        capability = entry.get("capability")
        if isinstance(capability, str):
            names.add(capability)
    return frozenset(names)


def _absence_disclosure(payload: dict[str, object], absence: Absence, side: str) -> str | None:
    """Why `absence`, observed in `payload`, is **not** evidence — or `None` when it is.

    The single gate every absence-derived conclusion in this module passes through. `side`
    is `"before"`/`"after"`, for the sentence only.

    Two independent reasons to withhold, both reported when both apply:

    * `payload` declares a degradation `_ABSENCE_DEPENDENCIES` says can produce this exact
      emptiness. The absence is then indistinguishable from the read simply not having
      happened.
    * `payload` declares a degradation this version of `verify` cannot name at all — an
      artifact from a newer `sqlquality`. `KNOWN_DEGRADATIONS` is pinned to the adapters'
      own constants, so this cannot mean "someone forgot to add it here" within one
      version; across versions it means the blind spot is real and unbounded, which is
      precisely the case for disclosing rather than assuming.

    Deliberately *not* consulted when the absence is not actually being read as evidence:
    a fully-resolved `after` still grades `IMPROVED` at full confidence even under an
    unrecognized degradation, because nothing about that verdict rests on something being
    missing. Over-caution is cheaper than over-claiming, but only where the claim exists.
    """
    declared = _degraded_names(payload)
    blocking = sorted(declared & _ABSENCE_DEPENDENCIES[absence])
    unrecognized = sorted(declared - KNOWN_DEGRADATIONS)
    if not blocking and not unrecognized:
        return None
    causes: list[str] = []
    if blocking:
        causes.append("degraded read(s) " + ", ".join(blocking))
    if unrecognized:
        causes.append(
            "degradation(s) this version of sqlquality cannot interpret ("
            + ", ".join(unrecognized)
            + ")"
        )
    return (
        f"The {side} run reported {' and '.join(causes)} in its `degraded` list; a read "
        f"that could not run produces the same emptiness a real change would, so "
        f"{_ABSENCE_SUBJECTS[absence]} is not evidence here."
    )


def _applied_advisor(
    key: tuple[str, ...], after: dict[str, object], after_index: ProposalIndex
) -> tuple[bool | None, str | None]:
    """ADV105 — verified at proposal level, per the design spec, not from `physical_state`
    at all: no catalog field corresponds to whatever Advisor's own internal heuristic
    flagged, so `physical_state` has nothing to say about this code specifically, even
    though the relation itself may well have an entry there for other reasons. Advisor's
    row disappearing *is* the applied signal — a deliberate, disclosed exception to every
    other code's "from `physical_state` and nothing else" rule (see the Task 6 report).

    **Checked first, before proposal presence at all (Finding 2, fix round 3; widened for
    finding N1, fix round 4):** if anything in `after`'s `degraded` list could have removed
    this recommendation from `after`'s own proposal list, its absence proves nothing — the
    read never happened, so `None` with the reason in the note, never a guessed `True`.
    Round 3 checked `CAP_ADVISOR` alone; `Absence.ADVISOR_RECOMMENDATION` covers the three
    further capabilities that suppress an ADV105 proposal just as completely, because
    `propose` fetches Advisor's rows for `aggregation.tables | facts` rather than for the
    whole cluster — see `_ABSENCE_DEPENDENCIES` for the per-capability reasoning.

    Otherwise: `True` when this exact key is absent from `after`'s *matched* proposals
    (Advisor no longer recommends it); `False` when still present, unambiguously, under
    the same key; `None`, with an explanatory note, when the key collides in `after`
    (Ruling 4 — an ambiguous match must never be read as either "gone" or "still there").
    """
    withheld = _absence_disclosure(after, Absence.ADVISOR_RECOMMENDATION, "after")
    if withheld is not None:
        return None, withheld
    if key in after_index.collisions:
        return None, (
            "This recommendation's key matched more than one proposal in the after run; "
            "whether it was addressed could not be determined unambiguously."
        )
    if key in after_index.matched:
        return False, None
    return True, None


def _detect_applied(
    proposal: dict[str, object],
    key: tuple[str, ...],
    before: dict[str, object],
    after: dict[str, object],
    after_index: ProposalIndex,
) -> tuple[bool | None, str | None]:
    """`applied` for one `before`-side proposal, dispatched by code per the design spec's
    table. `None` with an explanatory note for a code this function does not recognize
    (a future rule), rather than guessing.
    """
    code = proposal.get("code")
    evidence = proposal.get("evidence")
    if not isinstance(code, str) or not isinstance(evidence, dict):
        return None, "Malformed proposal: cannot determine whether it was applied."
    if code in _UNOBSERVABLE_CODES:
        return None, None
    if code == _ADVISOR_CODE:
        return _applied_advisor(key, after, after_index)
    relation_key = _relation_key(evidence)
    if relation_key is None:
        return None, "No relation identified in this proposal's evidence."
    before_phys = _physical_entry(before, relation_key)
    after_phys = _physical_entry(after, relation_key)
    if code in _CREATE_INDEX_CODES:
        return _applied_create_index(evidence, before_phys, after_phys), None
    if code in _DROP_INDEX_CODES:
        return _applied_drop_index(evidence, before_phys, after_phys), None
    if code in _REDSHIFT_KEY_CODES:
        return _applied_redshift_key(code, evidence, before_phys, after_phys)
    if code == _MAINTENANCE_CODE:
        return _applied_maintenance(evidence, before_phys, after_phys), None
    if code == _MATERIALIZE_CODE:
        return _applied_materialize(before_phys, after_phys), None
    return None, f"Unrecognized proposal code {code!r}: applied cannot be determined."


def _cited_digests(evidence: dict[str, object]) -> tuple[str, ...]:
    """The query-group digests this proposal cites, or `()`.

    Read with `.get(..., [])`: `fingerprint_digests` is *absent*, not `[]`, on rules with
    no query backing (ADV002, ADV003, ADV104, ADV105) — see `report.py`'s
    `_query_groups_payload`. A malformed non-list value, or a non-string element, is
    dropped rather than raising — Ruling 8 already established that a surviving
    proposal's citations can legitimately be incomplete, so this function must not treat a
    thin or malformed list as anything other than what it literally, safely contains.
    """
    digests = evidence.get("fingerprint_digests", [])
    if not isinstance(digests, (list, tuple)):
        return ()
    return tuple(d for d in digests if isinstance(d, str))


def _aggregate_mean(
    digests: tuple[str, ...], groups: dict[str, dict[str, object]]
) -> tuple[float | None, int]:
    """The call-weighted mean per call across `digests`' groups found in `groups`, and how
    many of `digests` resolved.

    **Ruling 5 — call-weighted, not a mean of means.** Sums `total_time_ms` and `calls`
    across every cited group present, then divides once, so a 10,000-call group and a
    10-call group contribute in proportion to their own call counts rather than being
    averaged as equals. A naive `mean(group.mean_ms for group in ...)` would let a rare,
    lightly-called slow query dominate the combined figure exactly as much as the group
    that actually carries the workload's cost.

    **Ruling 6 — partial presence.** The returned count is how many of `digests` actually
    resolved in `groups`; a caller comparing it against `len(digests)` can tell a complete
    aggregate from a partial one and disclose the gap, rather than this function silently
    treating an unresolved digest as a zero-cost, zero-call group — which the call-weighted
    sum would otherwise do with no visible trace, since adding `(0, 0)` to a running total
    changes nothing about the arithmetic while quietly discarding the fact that a citation
    could not be checked at all.

    **Ruling 7 — `None`, never a fabricated number.** Returns `(None, 0)` when no cited
    digest resolves at all, and `(None, present)` when every resolved group's `calls` sums
    to `0` — a group's own `mean_ms` may legitimately be `None` for exactly this reason
    (Task 3), and this aggregate must not divide by a zero total and call the result a mean.
    """
    total_calls = 0
    total_time_ms = 0.0
    present = 0
    for digest in digests:
        group = groups.get(digest)
        if not isinstance(group, dict):
            continue
        calls = group.get("calls")
        time_ms = group.get("total_time_ms")
        if isinstance(calls, bool) or not isinstance(calls, int):
            continue
        if isinstance(time_ms, bool) or not isinstance(time_ms, (int, float)):
            continue
        present += 1
        total_calls += calls
        total_time_ms += float(time_ms)
    if present == 0:
        return None, 0
    if total_calls == 0:
        return None, present
    return total_time_ms / total_calls, present


def _grade(mean_before: float, mean_after: float) -> VerifyOutcome:
    """`IMPROVED`/`UNCHANGED`/`REGRESSED` from the relative change against
    `RELATIVE_CHANGE_THRESHOLD`.

    Guards `mean_before <= 0.0` explicitly rather than dividing by it: a group that
    genuinely cost nothing per call before cannot get a *relative* improvement (there is
    nothing left to improve), while any nonzero `mean_after` is a real regression in
    absolute terms even though the relative change is undefined.
    """
    if mean_before <= 0.0:
        return VerifyOutcome.UNCHANGED if mean_after <= 0.0 else VerifyOutcome.REGRESSED
    change = (mean_after - mean_before) / mean_before
    if change <= -RELATIVE_CHANGE_THRESHOLD:
        return VerifyOutcome.IMPROVED
    if change >= RELATIVE_CHANGE_THRESHOLD:
        return VerifyOutcome.REGRESSED
    return VerifyOutcome.UNCHANGED


def verdicts(before: dict[str, object], after: dict[str, object]) -> list[ProposalVerdict]:
    """One `ProposalVerdict` per proposal in `before` that keys unambiguously (see
    `index_proposals`): applied, whether its cited workload evidence actually got faster,
    and how much to trust that answer.

    **Ruling 10 — scope.** Only `before`'s own proposals are represented: a proposal
    present only in `after` (a genuinely new finding, not a match for anything in
    `before`) has no verdict here, by design. The design spec frames this whole model as
    "each proposal in before.json yields...", and `VerifyOutcome` has no member for "new".
    Surfacing "N new findings in `after`" is a presentation concern for the CLI (a later
    task), derivable directly from `index_proposals(after)` against
    `index_proposals(before)`'s key set — not this function's job to invent an outcome the
    brief's enum has no room for.

    **Ruling 4 — collisions.** Iterates `index_proposals(before).matched` only, never
    `.collisions`: a `before`-side key collision has no single proposal to report a
    verdict *for*, so it is silently excluded from this list — never graded `DISAPPEARED`
    or folded into some other proposal's verdict. A caller wanting to disclose "N
    recommendations could not be matched unambiguously" calls `index_proposals(before)`
    directly, exactly as `ProposalIndex`'s own docstring prescribes. An `after`-side
    collision at an otherwise-unambiguous `before` key does not corrupt that proposal's
    verdict either: `applied` for every code except ADV105 comes from `physical_state`,
    keyed by relation rather than by proposal, so an ambiguity among `after`'s proposals
    cannot reach it; ADV105 handles its own collision case in `_applied_advisor`. The one
    place an `after`-side collision is visible here is `cost_share_after`, which is `None`
    whenever `key` is not a *uniquely* matched proposal in `after` — the same value it
    would have if the key were simply absent, which is honest: either way, `after`'s cost
    share for this exact recommendation cannot be read off unambiguously.

    **Decision order — the one thing Step 5's mutation testing exists to pin:**

    1. `applied is False` -> `NOT_APPLIED`, regardless of what the means say. An unapplied
       change did not cause anything, so crediting it with a mean improvement (however
       real that improvement is) would attribute someone else's fix, or ordinary workload
       shift, to this proposal.
    1b. `artifact_incomparabilities(before, after)` non-empty -> `UNOBSERVABLE` at `LOW`,
       with `mean_before` and `mean_after` both `None` and the reason in `note`, for every
       proposal whose `applied` is not `False`. Finding B, fix round 5: two runs differing
       in `--keep-literals` record the same query group under different digests, so no
       group-level outcome exists to claim — `DISAPPEARED` least of all. Ordered *after*
       `applied is False` deliberately: that verdict rests only on `physical_state`, which
       carries no digest, so it keeps both its outcome and the full ceiling. See
       `ArtifactMismatch` for why this is a third concept rather than a wider `Absence`.
    2. `applied is None` -> `UNOBSERVABLE`. Ruling 1 makes this reachable via two
       independent routes — a code with nothing to observe by design (ADV005, ADV006,
       ADV303), or a code that could observe something but `physical_state` reports
       `null` for the field it needs — and both must land here, never at `False`. The
       "cited group vanished, so possibly addressed by a rewrite" reading inside this
       branch is itself gated by `_absence_disclosure` (finding N1): the same denied
       `CAP_WORKLOAD` read that empties `query_groups` also empties `physical_state`, so
       this is the *reachable* route by which a degraded `after` run used to be reported
       as a query that stopped running.
    3. Only once `applied is True` does this function look at the cited query groups at
       all:
       - no citation resolves to a usable mean in `before` (no citations at all, by
         design for ADV002/ADV003/ADV104/ADV105 — Ruling 8 — or a malformed artifact's
         citations that do not resolve, or resolve to a zero-call total — Ruling 7) ->
         `UNOBSERVABLE`: an applied signal exists, but no workload evidence to grade a
         speed change against.
       - at least one citation resolves in `before` but none resolve in `after` -> a
         candidate `DISAPPEARED`, gated **twice**, in this order: first by
         `_absence_disclosure(after, Absence.CITED_QUERY_GROUP, ...)` (finding N1 — a
         degraded workload read in `after` empties `query_groups` while still recording
         `limit`, so it slips past the second gate entirely), then by Ruling 2's
         `window_limits(before, after).may_be_sampling_artifact`. Either forbids grading
         `DISAPPEARED`, downgrading to `UNOBSERVABLE`/`LOW` with the reason stated in
         `note` instead.
       - citations resolve in `after` but sum to zero calls -> `UNOBSERVABLE` (Ruling 7
         again, this time on the `after` side).
       - otherwise -> `_grade(mean_before, mean_after)`, with `note` disclosing when only
         some of the cited groups resolved in `after` (Ruling 6), and flagging the
         "applied but unchanged" case in unmistakable words when it is reached — the
         single most valuable outcome this whole feature exists to surface.

    **Confidence (Ruling 3)** starts at `confidence_for(classify_windows(before, after))`
    — the ceiling every verdict in this run shares — and is lowered no further except in
    the cases the design spec calls "possibly addressed... at LOW": an
    unobservable-application proposal whose cited group vanished, the Ruling-2-forced
    downgrade above, finding N1's degraded-read downgrades, and step 1b's
    `ArtifactMismatch`. Every other verdict, including a genuine `DISAPPEARED` (Ruling 2
    already filtered the untrustworthy ones into the case above), keeps the full ceiling.
    """
    before_index = index_proposals(before)
    after_index = index_proposals(after)
    limits = window_limits(before, after)
    ceiling = confidence_for(classify_windows(before, after))
    group_before = group_index(before)
    group_after = group_index(after)
    # Finding B (fix round 5). Pair-level and computed once: an `ArtifactMismatch` is a
    # property of the two artifacts, not of any one proposal. Checked ahead of every
    # group-level branch below, because when it fires there is no coordinate system in which
    # a group-level comparison could be expressed at all — see `ArtifactMismatch` for why
    # this is a third concept rather than a wider `Absence` or a lower confidence.
    incomparabilities = artifact_incomparabilities(before, after)
    incomparable_note = " ".join(item.detail for item in incomparabilities) or None

    results: list[ProposalVerdict] = []
    for key, proposal in before_index.matched.items():
        code = proposal.get("code")
        evidence = proposal.get("evidence")
        if not isinstance(code, str) or not isinstance(evidence, dict):
            continue  # unreachable via index_proposals; never trusted blindly regardless
        applied, applied_note = _detect_applied(proposal, key, before, after, after_index)
        digests = _cited_digests(evidence)
        mean_before, present_before = _aggregate_mean(digests, group_before)
        mean_after, present_after = _aggregate_mean(digests, group_after)
        if incomparable_note is not None:
            # Both means, not only `mean_after`. `mean_before` is a real measurement of the
            # before run, but reporting it beside a `None` `mean_after` is *exactly* the
            # shape a reader interprets as "the query stopped running" — the false claim
            # this finding is about. A caller wanting the raw per-side figures can call
            # `group_index` on either payload; what this pair cannot express is a comparison.
            mean_before = None
            mean_after = None
        cost_share_before = cost_share_of(evidence)
        after_proposal = after_index.matched.get(key)
        after_evidence = after_proposal.get("evidence") if after_proposal is not None else None
        cost_share_after = (
            cost_share_of(after_evidence) if isinstance(after_evidence, dict) else None
        )

        notes: list[str] = []
        if applied_note:
            notes.append(applied_note)
        if incomparable_note is not None:
            notes.append(incomparable_note)
        confidence = ceiling

        # Finding N1 (fix round 4): the one gate every conclusion drawn from a cited
        # group's *absence* from `after` passes through. Evaluated once, here, and
        # consulted as the first condition of each branch below that reads
        # `present_after == 0` — so a branch cannot reach a conclusion about the user's
        # workload from an emptiness `after`'s own degraded read produced. See
        # `Absence`/`_ABSENCE_DEPENDENCIES` for why this is a table rather than one more
        # hand-placed capability check, and `_absence_disclosure` for why it is not
        # consulted where no absence is being read.
        after_groups_withheld = _absence_disclosure(after, Absence.CITED_QUERY_GROUP, "after")

        if applied is False:
            # Survives an `ArtifactMismatch` untouched, and keeps the full ceiling: this
            # verdict rests entirely on `physical_state`, which carries no digest and is
            # unaffected by redaction. Withholding it would be its own false claim — "we
            # cannot tell" about something the artifacts do establish.
            outcome = VerifyOutcome.NOT_APPLIED
            notes.append("physical_state shows the proposed change was not made.")
        elif incomparable_note is not None:
            # No group-level outcome exists to claim — not `DISAPPEARED`, not a graded
            # speed change, not even the `applied is None` branch's "possibly addressed",
            # every one of which reads a digest-keyed lookup as a fact about the workload.
            # `applied` itself is reported unchanged: it comes from `physical_state`.
            outcome = VerifyOutcome.UNOBSERVABLE
            confidence = Confidence.LOW
        elif applied is None:
            vanished = bool(digests) and present_before > 0 and present_after == 0
            outcome = VerifyOutcome.UNOBSERVABLE
            if vanished and after_groups_withheld is not None:
                # The reachable half of finding N1: a denied `CAP_WORKLOAD` read in `after`
                # empties `query_groups` *and* `physical_state` at once (the same denial
                # empties `aggregation.tables`, so no relation is fetched), so `applied` is
                # `None` and every cited digest is unresolvable — which read as "the query
                # stopped running, possibly because you rewrote it" for a query that may
                # well still be running unchanged.
                confidence = Confidence.LOW
                notes.append(
                    "This proposal's application cannot be observed directly. "
                    + after_groups_withheld
                )
            elif vanished:
                confidence = Confidence.LOW
                notes.append(
                    "This proposal's application cannot be observed directly, and its "
                    "cited query group(s) are absent from the after run — possibly "
                    "addressed (for example, by a query rewrite), but disappearance "
                    "alone is not proof."
                )
            elif present_after > 0:
                # Finding 4 (fix round 3): the design spec's "still there at the same
                # cost" claim is conditional on the cost actually being the same — never
                # asserted without checking `_grade(mean_before, mean_after)` first.
                if mean_before is None or mean_after is None:
                    notes.append(
                        "This proposal's application cannot be observed directly, and "
                        "its cited query group(s) are present in the after run, but "
                        "their mean time per call could not be compared; whether the "
                        "recommendation was addressed is unknown."
                    )
                elif _grade(mean_before, mean_after) is VerifyOutcome.UNCHANGED:
                    notes.append(
                        "This proposal's application cannot be observed directly, and "
                        "its cited query group(s) are still present in the after run at "
                        "a similar cost — not addressed."
                    )
                else:
                    notes.append(
                        "This proposal's application cannot be observed directly, and "
                        "its cited query group(s) are present in the after run, but "
                        "their mean time per call changed; this alone does not "
                        "establish whether the recommendation was addressed."
                    )
            else:
                notes.append("This proposal's application cannot be observed from physical_state.")
        elif not digests or mean_before is None:
            outcome = VerifyOutcome.UNOBSERVABLE
            if not digests:
                notes.append(
                    "This proposal cites no query-group evidence to grade a speed "
                    "change against; only the applied signal above is available."
                )
            else:
                notes.append(
                    "None of this proposal's cited query groups resolved to a usable "
                    "mean in the before run; only the applied signal above is available."
                )
        elif present_after == 0:
            if after_groups_withheld is not None:
                # Ruling 2's sibling, and the door finding N1 came through: `limit` is
                # recorded even when the workload read itself was refused, so the two sides'
                # limits match and `may_be_sampling_artifact` stays `False`. A degraded read
                # must be checked *before* that gate, not after it.
                outcome = VerifyOutcome.UNOBSERVABLE
                confidence = Confidence.LOW
                notes.append(after_groups_withheld)
            elif limits.may_be_sampling_artifact:
                outcome = VerifyOutcome.UNOBSERVABLE
                confidence = Confidence.LOW
                notes.append(
                    "Cited query group(s) are absent from the after run, but the two "
                    f"runs' --limit differ or are unknown (before={limits.before}, "
                    f"after={limits.after}), so this may be a sampling artifact rather "
                    "than a real disappearance."
                )
            else:
                outcome = VerifyOutcome.DISAPPEARED
                notes.append("Cited query group(s) no longer appear in the after run.")
        elif mean_after is None:
            outcome = VerifyOutcome.UNOBSERVABLE
            notes.append(
                "Cited query group(s) are present in the after run but recorded zero "
                "calls; mean time per call cannot be computed."
            )
        else:
            if present_after < len(digests):
                notes.append(
                    f"Only {present_after} of {len(digests)} cited query groups were "
                    "present in the after run; the figures above reflect only those."
                )
            outcome = _grade(mean_before, mean_after)
            if applied is True and outcome is VerifyOutcome.UNCHANGED:
                notes.append("Applied but unchanged: the work was done and did not help.")

        if code == _ADVISOR_CODE:
            # Concern 3 (fix round 1): ADV105's applied signal is Advisor's own row
            # disappearing, not this project's own catalog read — a disclosed, weaker
            # kind of evidence than every other code's physical_state delta, regardless
            # of how comparable the two windows are. No ADV105 verdict may claim more
            # than MEDIUM, even under a DISJOINT or COMPARABLE window.
            confidence = _cap_confidence(confidence, Confidence.MEDIUM)

        results.append(
            ProposalVerdict(
                key=key,
                code=code,
                applied=applied,
                outcome=outcome,
                confidence=confidence,
                mean_before=mean_before,
                mean_after=mean_after,
                cost_share_before=cost_share_before,
                cost_share_after=cost_share_after,
                note=" ".join(notes),
            )
        )
    return results
