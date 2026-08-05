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
