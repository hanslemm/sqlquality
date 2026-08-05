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


class ProposalKeyCollisionError(ValueError):
    """Two proposals in one payload reduced to the same `proposal_key`.

    `index_proposals` builds a `dict` keyed by `proposal_key`, so two proposals sharing a
    key can only ever occupy one slot. A silent `{proposal_key(p): p for p in proposals}`
    would let the second proposal quietly overwrite the first — and `verify` (a later task)
    would then report the discarded one as `disappeared` between two runs, even though it
    was sitting in this very artifact the whole time: exactly the failure mode this whole
    matching layer exists to prevent, reintroduced one step downstream of `proposal_key`
    itself.

    Raised rather than silently kept-one/dropped-one/merged: a collision that reaches this
    point means `proposal_key` did not fully discriminate two real proposals (see its own
    docstring for the discriminators it already applies — `columns`/`column`, `index`, and
    a `ddl` fallback for the shapes that carry neither). That is a bug in the matching
    layer, not a fact about the workload, and this module has no principled way to decide
    which of the two colliding proposals to keep — surfacing it loudly, at the exact call
    that would otherwise have hidden it, is the only defensible response.
    """


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
      finding had gone away while it sat in both artifacts.

    A fourth, easy-to-miss hazard: several relation-scoped rules can legitimately emit *two*
    proposals for the same relation with no distinguishing column at all —
    `propose_maintenance` (ADV104) emits an independent `VACUUM` proposal and an independent
    `ANALYZE` proposal from two separate `if` statements (not `if`/`elif`) whenever a
    relation's `unsorted` *and* `stats_off` both cross their thresholds, and neither
    proposal's evidence carries a `column`, `columns` or `index` to tell them apart. And two
    proposals that *do* carry `columns`/`column` can still collide on it: `index` also has
    to be folded in for ADV002/ADV003, whose whole reason for existing is that two catalog
    indexes can share the same column list — the identical `columns` is not a bug there, it
    is the finding. So: `index` (when present) is always appended after the column
    discriminator, and when a proposal carries *none* of `columns`/`column`/`index` at all
    (ADV103, ADV104, ADV105, ADV301, ADV303), this falls back to the proposal's own `ddl`.
    `CREATE INDEX` proposals are emitted unnamed, so `ddl` is otherwise redundant with
    `columns` there — but for exactly the shapes that carry none of the other
    discriminators, `ddl` is deterministic from the statement itself (`VACUUM` vs.
    `ANALYZE`, one Advisor `rec_type` vs. another) and so distinguishes them. `ddl` is
    `None` on some proposals in this same bucket (ADV303) — harmless there, since those
    rules only ever emit one proposal per relation and have nothing to disambiguate in the
    first place. Whatever this fallback cannot discriminate, `index_proposals` refuses to
    silently drop — see `ProposalKeyCollisionError`.

    Returns `None` for a proposal with neither a relation nor a `fingerprint` in its
    evidence: returning a partial key (say, `(code,)` alone) would silently group every
    such proposal together as "the same recommendation" regardless of what it actually
    says, which is worse than refusing to key it at all.
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

    index = evidence.get("index")
    if isinstance(index, str):
        parts.append(index)
        discriminated = True

    if not discriminated:
        ddl = proposal.get("ddl")
        if isinstance(ddl, str):
            parts.append(ddl)

    return tuple(parts)


def index_proposals(payload: dict[str, object]) -> dict[tuple[str, ...], dict[str, object]]:
    """Every keyable proposal in an `advise --json` payload, by `proposal_key`.

    A proposal `proposal_key` cannot key (see its docstring) is silently excluded, not
    raised on: it simply cannot participate in the run-to-run comparison `verify` exists to
    do, and that is a fact about the proposal's evidence shape, not a defect in this
    payload.

    A key collision between two *keyable* proposals is different: it is never silently
    resolved by last-write-wins, because that is precisely the failure mode this whole
    matching layer exists to prevent one step further downstream. See
    `ProposalKeyCollisionError`.
    """
    proposals = payload.get("proposals")
    index: dict[tuple[str, ...], dict[str, object]] = {}
    if not isinstance(proposals, list):
        return index
    for proposal in proposals:
        if not isinstance(proposal, dict):
            continue
        key = proposal_key(proposal)
        if key is None:
            continue
        existing = index.get(key)
        if existing is not None:
            raise ProposalKeyCollisionError(
                f"key {key!r} matches two proposals: {existing.get('title')!r} and "
                f"{proposal.get('title')!r}. Refusing to silently keep one and drop the "
                "other — see ProposalKeyCollisionError's docstring."
            )
        index[key] = proposal
    return index


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
