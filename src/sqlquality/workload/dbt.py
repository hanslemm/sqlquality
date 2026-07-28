"""Optional dbt enrichment for `advise`.

dbt is *layered on top of* the engine-agnostic core, never underneath it: no workload adapter
imports this module, and every `advise` run behaves identically without a manifest. The
project's positioning is that the dbt-free path is first-class, so enrichment has to be
additive by construction rather than by discipline.
"""

from __future__ import annotations

import dataclasses
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from sqlquality.dbtproject import DbtProject, DbtProjectError, ModelNode
from sqlquality.models import Aggregation, ColumnUsage, Confidence, Proposal, Relation, Workload


def _split_relation_parts(text: str) -> list[str] | None:
    """Tokenize a dot-separated, optionally double-quoted identifier list.

    A hand-written scanner rather than a regex `findall`: `findall` silently *skips* any
    character that matches neither alternative — which is exactly how a stray unescaped
    quote, an empty quoted segment (`""`), or a trailing dot with nothing after it used to
    disappear instead of being rejected, shifting every later part one slot to the left (a
    `catalog.schema` name silently misread as `schema.table`). This scanner instead returns
    `None` the moment the input can't be tiled into parts with nothing left over, so a
    malformed name always declines rather than mis-parsing.
    """
    parts: list[str] = []
    i, n = 0, len(text)
    while i < n:
        if text[i] == '"':
            j = i + 1
            buf: list[str] = []
            closed = False
            while j < n:
                if text[j] == '"':
                    if j + 1 < n and text[j + 1] == '"':
                        # A doubled quote is SQL's escape for a literal `"` inside a
                        # quoted identifier — not the end of the segment.
                        buf.append('"')
                        j += 2
                        continue
                    closed = True
                    j += 1
                    break
                buf.append(text[j])
                j += 1
            if not closed:
                return None  # unterminated quote
            part = "".join(buf)
            i = j
        else:
            start = i
            while i < n and text[i] not in '."':
                i += 1
            part = text[start:i]
            if i < n and text[i] == '"':
                return None  # a quote appearing mid-bare-segment is not a valid name
        if not part:
            return None  # an empty segment (`""` or two dots in a row) can't be qualified
        parts.append(part)
        if i < n:
            if text[i] != ".":
                return None
            i += 1
            if i >= n:
                return None  # a trailing dot with nothing after it is malformed
    return parts


def parse_relation_name(relation_name: str) -> Relation | None:
    """`(schema, table)` from a dbt `relation_name`, or None if it cannot be qualified.

    dbt writes a quoted three-part name — `'"dev"."main"."stg_orders"'` — and the raw node's
    own `schema` field is `None` in practice, so this string is the only reliable source. The
    database part is dropped because `advise` connects to one database at a time.

    A name with fewer than two parts, or one that cannot be cleanly tokenized (an empty
    segment, an unterminated quote, a trailing dot), returns None rather than a guess: a
    `Relation` needs an exact schema, and inventing one — or shifting onto the wrong part
    because a malformed segment silently vanished — is how a production table gets
    attributed to an unrelated model.
    """
    text = relation_name.strip()
    if not text:
        return None
    parts = _split_relation_parts(text)
    if parts is None or len(parts) < 2:
        return None
    return Relation(schema=parts[-2], table=parts[-1])


def _is_test_id(unique_id: str) -> bool:
    """Whether `unique_id` names a dbt test, by dbt's own unique_id convention.

    A test's unique_id is always `test.<package>.<name>.<hash>` — this is dbt's own
    naming scheme, the same one that makes `unique_id.startswith("model.")` reliable
    elsewhere in this module. Checking the id rather than resolving a node is deliberate:
    an exposure or a source referenced in `child_map` may not even have an entry in
    `manifest["nodes"]` (both live in their own top-level manifest sections), so a lookup
    through `DbtProject.node` would raise for exactly the consumers this check must not
    reject.
    """
    return unique_id.split(".", 1)[0] == "test"


@dataclass(frozen=True)
class DbtContext:
    """dbt models indexed by the relation they build, for joining against workload facts."""

    models: dict[Relation, ModelNode]
    #: Relations dropped because two *different* models claimed the same (schema, table) —
    #: see `from_project`. Surfaced so the CLI disclosure can tell a user "we found nothing"
    #: apart from "we found two candidates and refused to guess."
    dropped_collisions: int = 0
    #: How many other *declared consumers* — anything in the manifest's child_map except a
    #: test — the model building this relation has, keyed the same way as `models`. ADV303
    #: needs this to exclude a model that only looks unused because nothing but another
    #: dbt-declared consumer reads it.
    #:
    #: Deliberately **not** `DbtProject.model_children`'s count: that method filters to
    #: `resource_type == "model"`, which is correct for its own callers (the model DAG) but
    #: wrong here — a snapshot or an exposure is a real, dbt-declared consumer (an exposure
    #: exists specifically to say "a BI dashboard reads this"), and a model whose only child
    #: is one of those is used, just not by another model. `from_project` instead counts
    #: `DbtProject.child_ids`, which reads the manifest's child_map with no resource-type
    #: filter, and excludes only `test.*` ids: a `not_null` test is an assertion *about* a
    #: model, not a consumer *of* it, so it must not count toward "something reads this."
    #:
    #: Holding the whole `DbtProject` just to ask this on demand would let dbt-shaped
    #: knowledge (unique_ids, the child map) leak past this module's boundary into whatever
    #: calls `DbtContext.model_for` today. Carrying only the count keeps `DbtContext` a plain
    #: fact about relations, the same shape `model_for` already promises.
    consumer_count: dict[Relation, int] = field(default_factory=dict)

    @classmethod
    def from_project(cls, project: DbtProject) -> DbtContext:
        """Index every model by the relation it builds.

        `project.model_ids()` already filters to `resource_type == "model"`, so seeds,
        tests and snapshots never reach this loop — there is nothing left here to
        re-check that against. A model with no `relation_name` *is* skipped, and that
        guard is reachable: an ephemeral materialization is inlined as a CTE and never
        occupies a physical relation, so dbt leaves its `relation_name` unset.

        Two different models can each build a relation with the same `(schema, table)`
        in two different databases — `'"prod"."main"."orders"'` and
        `'"stage"."main"."orders"'` both key `Relation("main", "orders")` once the
        database is dropped. `advise` connects to one database at a time, so there is no
        way to tell which model is the right one. Rather than let dict insertion order
        silently pick a winner, a colliding relation is dropped from the index entirely
        (and counted in `dropped_collisions`): an unmatched relation reads as "we
        couldn't tell," not as a guess that a later rule then rewrites DDL on the
        strength of.
        """
        candidates: dict[Relation, ModelNode] = {}
        collided: set[Relation] = set()
        for uid in project.model_ids():
            node = project.node(uid)
            if not node.relation_name:
                continue
            relation = parse_relation_name(node.relation_name)
            if relation is None:
                continue
            existing = candidates.get(relation)
            if existing is not None and existing.unique_id != node.unique_id:
                collided.add(relation)
                continue
            candidates[relation] = node
        models = {r: n for r, n in candidates.items() if r not in collided}
        consumer_count = {
            r: sum(1 for cid in project.child_ids(n.unique_id) if not _is_test_id(cid))
            for r, n in models.items()
        }
        return cls(models=models, dropped_collisions=len(collided), consumer_count=consumer_count)

    def model_for(self, relation: Relation) -> ModelNode | None:
        """The model building this exact relation, matching schema *and* table.

        Deliberately no bare-table-name fallback: dbt's `main`/`dev` target schemas routinely
        differ from the schema `advise` introspects, so a name-only match would attribute a
        production table to an unrelated development model — and then, via ADV302, rewrite
        that table's DDL on the strength of it.
        """
        return self.models.get(relation)


def load_dbt_context(
    project_dir: Path | None, manifest: Path | None
) -> tuple[DbtContext | None, str | None]:
    """Load a manifest if one was requested, returning `(context, disclosure)`.

    Never raises. A manifest that is missing, unreadable or malformed degrades to "no
    enrichment" plus a line for the user, because by the time this runs the whole catalog
    analysis has already happened — aborting would throw away real work over an optional
    input. Same reasoning as the report-write failure path in `cli.py`.
    """
    if manifest is not None:
        path = manifest
    elif project_dir is not None:
        path = project_dir / "target" / "manifest.json"
    else:
        return None, None
    try:
        project = DbtProject.from_path(path)
        context = DbtContext.from_project(project)
    except DbtProjectError as exc:
        # The expected failure mode: `DbtProject.from_path` already wraps a missing file
        # or unparseable JSON into a `DbtProjectError` whose own message names `path`, so
        # nothing is added here — doing so would print the path twice.
        return None, f"dbt enrichment unavailable: {exc}"
    except Exception as exc:
        # The wide net, deliberately: manifest.json is a file some *other* tool wrote, and
        # a structurally-valid-JSON-but-wrong-shaped manifest (`{"nodes": null}`, a node
        # that isn't an object, a non-string `relation_name`, ...) can raise almost
        # anything — AttributeError from a misplaced `.get()`, TypeError from iterating
        # `None` — well past the narrow set `DbtProject` raises on purpose. This runs
        # after the whole catalog analysis, so the cost of a miss here is an aborted run
        # that already did the real work; the cost of this wide a net is at most
        # swallowing a bug in our own parsing, which the disclosure below still surfaces.
        return None, f"dbt enrichment unavailable: could not read {path}: {exc}"
    disclosure = f"dbt enrichment from {path} ({len(context.models)} model(s)"
    if context.dropped_collisions:
        disclosure += f", {context.dropped_collisions} cross-database collision(s) dropped"
    disclosure += ")"
    return context, disclosure


#: dbt materializations whose relation is rebuilt out from under a raw `CREATE INDEX`, and
#: what that rebuild does to it. `view` and anything unrecognised are handled separately —
#: a view has no relation to index at all, and an unrecognised materialization is unknown
#: rather than known-safe, so neither belongs in a table keyed by "known to be rebuilt."
#: `ephemeral` never reaches this table either, but for a different reason: an ephemeral
#: model is inlined as a CTE and so has no `relation_name`, which means it never gets far
#: enough through `DbtContext.from_project`/`model_for` to reach a proposal at all.
_REBUILD = {
    "table": "every `dbt run` drops and recreates this relation, so a raw CREATE INDEX is lost",
    "incremental": (
        "a normal `dbt run` keeps this relation, but `dbt run --full-refresh` rebuilds it and a "
        "raw CREATE INDEX is lost"
    ),
    #: A dbt `materialized_view` model (dbt-core 1.6+) also accepts an `indexes` config, and
    #: like `incremental` it is not rebuilt on *every* run: a normal `dbt run` refreshes it
    #: in place. It is rebuilt when `dbt run --full-refresh` runs, or when a configuration
    #: change forces dbt to drop and recreate it rather than refresh in place — either way a
    #: raw CREATE INDEX outside dbt's config does not survive that path.
    "materialized_view": (
        "a normal `dbt run` refreshes this materialized view in place, but `dbt run "
        "--full-refresh` (or a config change dbt can't apply in place) drops and recreates "
        "it, and a raw CREATE INDEX is lost"
    ),
}

#: `CREATE INDEX` or `CREATE UNIQUE INDEX`, optionally followed by `CONCURRENTLY` — matched
#: as a prefix, so whatever comes after (`CONCURRENTLY`, `ON`, ...) is irrelevant here.
_INDEX_CREATE_RE = re.compile(r"(?i)^CREATE\s+(?:UNIQUE\s+)?INDEX\b")
_UNIQUE_INDEX_RE = re.compile(r"(?i)^CREATE\s+UNIQUE\s+INDEX\b")


def _is_index_creating(ddl: str | None) -> bool:
    """An index-creating proposal, detected by its DDL prefix rather than its rule code.

    ADV001, ADV007 and ADV008 all emit `CREATE INDEX` today and Batch 3b adds more; a
    hardcoded set of codes would silently stop matching the day a new rule ships. Also
    matches `CREATE UNIQUE INDEX` (dbt's `indexes` config has a `unique` field for exactly
    this) and tolerates an operator-facing `CONCURRENTLY` in between — the DDL script's own
    header recommends `CONCURRENTLY` for a live table, so a proposal that used it must not
    silently stop being recognised as index-creating.
    """
    return ddl is not None and _INDEX_CREATE_RE.match(ddl.lstrip()) is not None


def _is_unique_index(ddl: str) -> bool:
    """Whether `ddl` is a `CREATE UNIQUE INDEX`, which dbt's config expresses as `unique: true`."""
    return _UNIQUE_INDEX_RE.match(ddl.lstrip()) is not None


def _is_partial_index(proposal: Proposal) -> bool:
    """A WHERE-restricted proposal (ADV004's partial index), detected structurally.

    A substring search for `"WHERE"` in the DDL is foolable two ways: a column genuinely
    named `WHERE` (quoted, so syntactically a plain identifier) makes an ordinary index
    proposal look partial, and a lowercase `where` — plausible from a future engine's rule,
    even though every rule here emits uppercase today — would not match at all, silently
    dropping a real predicate into a config block that has nowhere to put it. ADV004
    already carries `guard_column`/`guard_predicate` in its own evidence for exactly this
    proposal shape, so keying on their presence is structural rather than textual: it
    cannot be spoofed by an identifier and cannot miss on casing.
    """
    return "guard_column" in proposal.evidence or "guard_predicate" in proposal.evidence


def _relation_of(proposal: Proposal) -> Relation | None:
    """The relation a proposal is about, from its own evidence — every rule stores one."""
    schema = proposal.evidence.get("schema")
    table = proposal.evidence.get("table")
    if isinstance(schema, str) and isinstance(table, str):
        return Relation(schema, table)
    return None


def _comment_block(lines: list[str]) -> str:
    """Render `lines` as a `--`-commented block, safe even if a line's *content* smuggles
    a raw newline.

    `parse_relation_name` accepts a newline inside a quoted identifier — dbt's own
    `relation_name` field can carry one — and the column/table names this module
    interpolates ultimately come from a live catalog, which permits the same thing.
    Wrapping an interpolated column list in `list(...)` already neutralizes that: a
    built-in `list` has no `__str__` of its own, so formatting it falls back to `__repr__`,
    which escapes an embedded `\\n` in each contained string into the two literal
    characters `\\` and `n` before this function ever sees it — `repr()`'s own escaping is
    what does that work, not the value's *outer* formatting conversion, so an explicit
    `!r` on top of an already-listed value adds nothing. Splitting each logical line again
    here is a second, independent defense: it protects a future caller that interpolates a
    raw value without going through a list's own escaping, so nothing reaching here can
    produce an output line lacking a leading `--` even then. `render_ddl` defends the same
    hazard the same way for raw DDL; this is that defense's equivalent for a generated
    config block.
    """
    out: list[str] = []
    for line in lines:
        physical = line.splitlines() or [""]
        out.extend(f"-- {p}" for p in physical)
    return "\n".join(out)


def _dbt_attribution(model: ModelNode) -> str:
    return f"`{model.unique_id}` (materialized as `{model.materialized}`)"


def enrich_proposals(proposals: list[Proposal], context: DbtContext) -> list[Proposal]:
    """Rewrite index-creating proposals whose relation dbt manages; attribute the rest.

    A `CREATE INDEX` (or `CREATE UNIQUE INDEX`, optionally `CONCURRENTLY`) proposal on a
    `table`-, `incremental`- or `materialized_view`-materialized dbt model is expressed
    instead as a config block a human can paste into that model's `.yml`, since the raw
    DDL is destroyed the next time dbt rebuilds the relation. A `view` cannot carry an
    index at all, so the proposal is dropped and explained rather than rewritten. An
    unrecognised (or absent) materialization is left alone — unknown is not the same as
    known-safe, so the DDL is not touched on a guess.

    Everything else passes through with only its evidence enriched: a `DROP INDEX`
    proposal is ordinary regardless of dbt (dbt never created the index, so there is
    nothing for its `indexes` config to un-express), and an advisory proposal with no DDL
    has nothing to rewrite either. Both are still attributed to the model they concern, so
    a reader knows where to make the fix.

    A proposal whose relation dbt does not manage — or that carries no `(schema, table)`
    evidence at all — is returned completely unchanged.
    """
    out: list[Proposal] = []
    for proposal in proposals:
        relation = _relation_of(proposal)
        model = context.model_for(relation) if relation is not None else None
        if model is None:
            out.append(proposal)
            continue
        out.append(_enrich_one(proposal, model))
    return out


def _enrich_one(proposal: Proposal, model: ModelNode) -> Proposal:
    evidence = dict(proposal.evidence)
    evidence["dbt_model"] = model.unique_id
    evidence["dbt_materialized"] = model.materialized

    if not _is_index_creating(proposal.ddl):
        # DROP INDEX, and any advisory proposal with no DDL at all: attributed, not
        # rewritten. Dropping an index dbt never created is ordinary, and there is no
        # `indexes` config entry that expresses a removal.
        return dataclasses.replace(proposal, evidence=evidence)

    ddl = proposal.ddl
    assert ddl is not None  # _is_index_creating(None) is False, so this branch guarantees it
    materialized = model.materialized

    if materialized == "view":
        rationale = (
            f"{proposal.rationale} This relation is a dbt view ({_dbt_attribution(model)}): "
            "a view has no storage of its own to index, so this proposal does not apply."
        )
        return dataclasses.replace(
            proposal,
            ddl=None,
            rationale=rationale,
            confidence=Confidence.LOW,
            evidence=evidence,
        )

    if materialized not in _REBUILD:
        label = materialized if materialized else "(absent)"
        rationale = (
            f"{proposal.rationale} This relation is a dbt model ({_dbt_attribution(model)}), "
            f"but materialization '{label}' is unrecognised, so the DDL below is left as-is "
            "rather than rewritten on a guess."
        )
        return dataclasses.replace(proposal, rationale=rationale, evidence=evidence)

    # `table`, `incremental` or `materialized_view`: the relation genuinely gets rebuilt,
    # so a raw CREATE INDEX is lost sooner or later. A partial index (ADV004's
    # WHERE-restricted proposal) has no dbt `indexes`-config equivalent — that config has
    # no predicate field — so it must be disclosed as not expressible rather than silently
    # rewritten into a config block that quietly drops the WHERE clause and turns a
    # correct proposal into a wrong one.
    if _is_partial_index(proposal):
        rationale = (
            f"{proposal.rationale} This relation is a dbt model ({_dbt_attribution(model)}): "
            f"{_REBUILD[materialized]}. dbt's `indexes` config has no predicate field, so this "
            "partial index cannot be expressed as config — it will be dropped on the next "
            "rebuild unless you reapply the DDL above by hand afterward."
        )
        return dataclasses.replace(proposal, rationale=rationale, evidence=evidence)

    columns = proposal.evidence.get("columns")
    if (
        not isinstance(columns, (tuple, list))
        or not columns
        or not all(isinstance(c, str) for c in columns)
    ):
        # No plain column list to express as config — leave the DDL untouched rather than
        # invent one.
        return dataclasses.replace(proposal, evidence=evidence)

    # `!r` is deliberately absent: `list(columns)` has no `__str__` of its own, so plain
    # `{list(columns)}` formatting already falls back to `__repr__` and gets the same
    # per-element escaping `!r` would have asked for explicitly — see `_comment_block`.
    config_lines = [
        "ADV302: express this as dbt config, not DDL. Add to the model's config block:",
        "  indexes:",
        f"    - columns: {list(columns)}",
        "      type: btree",
    ]
    if _is_unique_index(ddl):
        config_lines.append("      unique: true")
    config_ddl = _comment_block(config_lines)
    evidence["dbt_index_config"] = config_ddl
    rationale = (
        f"{proposal.rationale} This relation is a dbt model ({_dbt_attribution(model)}): "
        f"{_REBUILD[materialized]}. Add the config block above to the model instead of running "
        "this DDL directly; `dbt run` applies it."
    )
    return dataclasses.replace(proposal, ddl=config_ddl, rationale=rationale, evidence=evidence)


def propose_materialization(
    aggregation: Aggregation, context: DbtContext, *, min_cost_share: float
) -> list[Proposal]:
    """ADV301 — a dbt model materialized as a `view` that carries a hot share of workload cost.

    A view re-executes its defining query on every read, so any cost saved by a `table` or
    `incremental` build is instead paid, in full, every time something reads it. When the
    workload shows a view carrying a hot share of cost, that repeated cost is exactly what a
    materialization change would trade for a heavier (and scheduled) `dbt run`. This is only
    computable by joining workload cost — which this tool has — to the model graph's
    materialization — which only the manifest has; neither alone is enough.

    Confidence is capped at MEDIUM and there is deliberately no HIGH branch, mirroring
    ADV008's precedent for the same shape of gap: whether the trade actually pays off depends
    on how often the model is *rebuilt* versus how often it is *read*, and on how fresh the
    data needs to be — neither is visible from query history. Claiming HIGH would be a claim
    about a build schedule this tool cannot see. Do not add a HIGH branch here for symmetry
    with ADV001; the missing rung is deliberate, not an oversight.

    One proposal per relation, not per column: two hot columns on the same view are one
    materialization decision, not two. `cost_share` is the *max* over that relation's usage,
    not the sum — `ColumnUsage.cost_share` is deliberately not a partition (see its own
    docstring), so a query hot on two columns of the same view would otherwise be counted
    twice, exactly the double-count ADV001 and ADV008 already avoid the same way.

    Only `materialized == "view"` qualifies, and excluding `materialized_view` needs no
    separate branch: the equality check already excludes it by construction, and correctly
    so — unlike a plain view, a materialized view refreshes its *stored* result rather than
    re-executing its query on every read, so it has already made the build-time-for-read-time
    trade this rule proposes; recommending it again would be pointless. Contrast ADV302,
    which *does* need an explicit `materialized_view` branch, because it answers a different
    question (does a rebuild destroy an index) that a materialized view answers the same way
    a table does.
    """
    by_relation: dict[Relation, list[ColumnUsage]] = defaultdict(list)
    for item in aggregation.usage:
        by_relation[item.relation].append(item)

    proposals: list[Proposal] = []
    for relation in sorted(by_relation):
        model = context.model_for(relation)
        if model is None or model.materialized != "view":
            continue
        cost_share = max(item.cost_share for item in by_relation[relation])
        if cost_share < min_cost_share:
            continue
        proposals.append(
            Proposal(
                code="ADV301",
                title=f"Materialize {relation} instead of a view",
                rationale=(
                    f"This dbt model ({_dbt_attribution(model)}) carries a hot share of "
                    f"workload cost ({cost_share:.1%}) but, as a view, re-executes its "
                    "defining query on every read. Materializing it as `table` or "
                    "`incremental` trades that repeated read cost for a build cost paid on "
                    "each `dbt run` instead — worth it only if the model is read far more "
                    "often than it is rebuilt, and if it does not need to reflect every "
                    "write immediately. Neither is visible from query history, which is why "
                    "this is capped at MEDIUM: it names the trade, not a verdict on it."
                ),
                evidence={
                    "schema": relation.schema,
                    "table": relation.table,
                    "dbt_model": model.unique_id,
                    "cost_share": cost_share,
                },
                confidence=Confidence.MEDIUM,
                ddl=None,
            )
        )
    return proposals


def propose_unused_models(
    aggregation: Aggregation, context: DbtContext, workload: Workload
) -> list[Proposal]:
    """ADV303 — a dbt model within reach of the manifest that the analyzed workload never
    touched.

    A model that costs a build every night and that nothing queries is worth knowing about,
    but the evidence here is *absence*, which is far weaker than presence, so this carries
    the loudest caveat in this module and a hard confidence cap of LOW rather than a
    downgrade for each individual caveat:

    * the analyzed window may simply not cover this model's reader — a monthly report, a
      BI tool with its own cache, an ad-hoc job that only runs quarterly;
    * `--limit` truncates the query history handed to `advise`, so a cold-but-genuinely-used
      model can look exactly like an unused one within the slice this tool actually saw.

    A model with a declared consumer is excluded outright — not merely downgraded — because
    that is a correctness gate, not a caveat: a staging model consumed only by another model,
    a snapshot, or a dbt exposure (which exists specifically to declare "a BI dashboard reads
    this") *is* used, just not by an ad-hoc query, and without this exclusion the rule would
    propose deleting every staging model in a well-formed project. See
    `DbtContext.consumer_count` for what counts as a consumer and why a test does not.

    This only looks at a model's *immediate* consumers, not the whole downstream chain: if
    dead model A feeds dead model B, B being unused does not get attributed back to A — A is
    judged solely on its own `consumer_count`, which B's presence still satisfies. This is
    conservative by construction (it never flags something it shouldn't) but it also means a
    fully dead sub-DAG is only ever reported from its leaf, not from its root — cascading
    would require knowing that every consumer along the chain is itself unused, which this
    rule does not attempt.

    An `Aggregation` with no usage at all is refused rather than treated as evidence: every
    relation in `context.models` would trivially be "untouched" (none can be in
    `aggregation.tables`, which is built only from usage), so an empty or fully-unparseable
    workload must not read as proof that every childless model is unused.
    """
    if not aggregation.usage:
        return []
    proposals: list[Proposal] = []
    for relation in sorted(context.models):
        if relation in aggregation.tables:
            continue
        if context.consumer_count.get(relation, 0) > 0:
            continue
        model = context.models[relation]
        rationale = (
            f"No query in the analyzed workload ({workload.window_description}) referenced "
            f"this dbt model ({_dbt_attribution(model)}). This is evidence of absence, not "
            "proof of it: the window may simply not cover this model's reader — a monthly "
            "report, a BI tool with its own cache, a quarterly job — and `--limit` truncates "
            "the query history this tool actually saw, so a cold-but-used model can look "
            "unused within that slice. A model with a declared consumer — another model, a "
            "snapshot, or a dbt exposure — is excluded from this rule outright rather than "
            "merely downgraded, because it is used, just not by an ad-hoc query."
        )
        proposals.append(
            Proposal(
                code="ADV303",
                title=f"{relation} is a dbt model the analyzed workload never touched",
                rationale=rationale,
                evidence={
                    "schema": relation.schema,
                    "table": relation.table,
                    "dbt_model": model.unique_id,
                },
                confidence=Confidence.LOW,
                ddl=None,
            )
        )
    return proposals
