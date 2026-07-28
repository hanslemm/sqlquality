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


def resolve_manifest_path(project_dir: Path | None, manifest: Path | None) -> Path | None:
    """The manifest path `load_dbt_context` will read, or None if neither option was given.

    An explicit `--manifest` wins; otherwise `--project-dir/target/manifest.json`; otherwise
    neither was given. One function with two callers, deliberately: `load_dbt_context`
    resolves the path it loads, and `cli.advise` reports which path *was* loaded in its JSON
    payload. Those were two independent copies of this same precedence, and swapping the
    order in either copy alone left the whole suite green — so the payload could name a
    manifest that was never read.
    """
    if manifest is not None:
        return manifest
    if project_dir is not None:
        return project_dir / "target" / "manifest.json"
    return None


#: dbt adapters that could plausibly be building the relations `advise` introspects. `advise`
#: connects to Postgres only, and redshift is its derivative (same `CREATE INDEX`, same
#: `indexes` model config), so a manifest naming either is consistent with the connection.
#: Anything else names a different warehouse entirely — see `_manifest_warnings`.
_CONSISTENT_ADAPTERS = frozenset({"postgres", "redshift"})


def _manifest_warnings(project: DbtProject) -> list[str]:
    """The two manifest checks `check` makes and the dbt `advise` path did not.

    `check` warns on a non-v12 `dbt_schema_version` and resolves its dialect from
    `adapter_type`; `advise` read neither, so it silently accepted a v10/v11 manifest whose
    node shapes it reads as if they were v12, and said nothing at all when the manifest
    described a different warehouse from the one it had just connected to.

    **What a foreign `adapter_type` actually means.** Not merely "ADV302 might emit a config
    key that adapter lacks" — the deeper problem is that dbt is then not building the Postgres
    relations `advise` just introspected *at all*. A Snowflake manifest paired with a Postgres
    connection means every `(schema, table)` match is a coincidence of naming: ADV302's
    premise (a `dbt run` rebuilds this relation, so raw DDL does not survive) is false,
    ADV301 attributes Postgres cost to a model that builds a Snowflake table, and ADV303 calls
    a Postgres relation an unused dbt model. So the warning is about the pairing, not about one
    config key.

    Warn rather than suppress. Two commands reading the same file and disagreeing about
    whether it is even the right shape is what a user of both would not expect, and the
    mismatch is something the user has to fix in their invocation — silently dropping all dbt
    output would hide the very thing they need told. For ADV302 specifically, its alternative
    to a config block is raw DDL that a rebuild destroys, so declining the rewrite would leave
    the operator *less* informed, not more.

    **An absent `adapter_type` warns too, deliberately.** `dbt compile` always writes one, so
    absence means a hand-written or truncated manifest, and the honest statement is that the
    pairing cannot be checked rather than that it is fine. Warning on "different" while
    staying silent on "unknown" would make silence mean two different things. `check` makes
    the same distinction, disclosing "manifest adapter_type absent or unrecognized" rather
    than assuming.

    Both values are `isinstance`-guarded because `metadata` is a section some other tool
    wrote: a non-string version would otherwise raise from the `in` test, and this function
    runs where a raise degrades the whole enrichment.
    """
    warnings: list[str] = []
    schema_version = project.schema_version()
    if not isinstance(schema_version, str) or "/v12" not in schema_version:
        found = schema_version if schema_version else "(absent)"
        warnings.append(
            f"warning: manifest dbt_schema_version is {found}, expected a v12 schema — "
            "dbt enrichment may be unreliable"
        )
    adapter_type = project.adapter_type()
    if not isinstance(adapter_type, str) or not adapter_type:
        warnings.append(
            "warning: manifest records no adapter_type, so it cannot be confirmed that these "
            "models build the relations being introspected — dbt enrichment assumes they do"
        )
    elif adapter_type not in _CONSISTENT_ADAPTERS:
        warnings.append(
            f"warning: manifest adapter_type is {adapter_type} but advise connects to "
            "postgres, so these models do not build the relations being introspected — every "
            "dbt match is a name coincidence and ADV301/ADV302/ADV303 will be wrong"
        )
    return warnings


def load_dbt_context(
    project_dir: Path | None, manifest: Path | None
) -> tuple[DbtContext | None, str | None]:
    """Load a manifest if one was requested, returning `(context, disclosure)`.

    Never raises. A manifest that is missing, unreadable or malformed degrades to "no
    enrichment" plus a line for the user, because by the time this runs the whole catalog
    analysis has already happened — aborting would throw away real work over an optional
    input. Same reasoning as the report-write failure path in `cli.py`.
    """
    path = resolve_manifest_path(project_dir, manifest)
    if path is None:
        return None, None
    try:
        project = DbtProject.from_path(path)
        context = DbtContext.from_project(project)
        warnings = _manifest_warnings(project)
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
    for warning in warnings:
        disclosure += f"\n{warning}"
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
#: `DROP INDEX`, optionally `CONCURRENTLY` / `IF EXISTS` — matched as a prefix, so whatever
#: follows is irrelevant. The DDL script's own header recommends `CONCURRENTLY` for a live
#: table, so a proposal that used it must not stop being recognised as index-dropping.
_INDEX_DROP_RE = re.compile(r"(?i)^DROP\s+INDEX\b")

#: A `USING <method>` clause naming anything but btree. `\S` after the lookahead is load
#: bearing: without it, `\s+` backtracks so that `USING  btree` (two spaces) satisfies a
#: bare `(?!btree\b)` one space in, and a plain btree index would be refused.
_NON_BTREE_RE = re.compile(r"(?i)\bUSING\s+(?!btree\b)\S")


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


def _is_index_dropping(ddl: str) -> bool:
    """A `DROP INDEX` proposal (ADV002, ADV003), detected by prefix like its create-side twin.

    Kept separate from `_is_index_creating` rather than folded into one "touches an index"
    check: the two need opposite advice. A create is *replaced* by dbt config; a drop cannot
    be — dbt's `indexes` config has no way to express a removal — so a drop keeps its DDL and
    gains a warning that the config entry has to go too.
    """
    return _INDEX_DROP_RE.match(ddl.lstrip()) is not None


def _is_unique_index(ddl: str) -> bool:
    """Whether `ddl` is a `CREATE UNIQUE INDEX`, which dbt's config expresses as `unique: true`."""
    return _UNIQUE_INDEX_RE.match(ddl.lstrip()) is not None


def _names_a_non_btree_method(ddl: str) -> bool:
    """Whether `ddl` asks for an access method the config reconstruction cannot express.

    The config block is rebuilt from `evidence["columns"]` and hardcodes `type: btree`; the
    DDL text itself is discarded. That is faithful for every rule shipping today — all of
    them emit a plain btree over a column list, with no `USING`, no expression, no
    `DESC`/`NULLS`/opclass — but the day a rule proposes `USING gin`, a silent rewrite to
    `type: btree` would hand back a *different index* than the one the evidence justified.
    So a non-btree access method declines the rewrite and discloses instead.

    Textual, unlike `_is_partial_index`, which keys on `guard_column`/`guard_predicate` in
    evidence. That asymmetry is forced rather than chosen: no rule records an access method in
    its evidence, so there is nothing structural to key on here today. The cost is a possible
    over-trigger — **not** on a column merely named `USING`, since quoting puts a `"` where
    `\\s+` needs whitespace and `"USING"` therefore does not match, but on one whose name
    *contains* the whole clause, like `"USING gin"`. That declines a rewrite which would have
    been fine — the conservative direction, since a missed detection instead quietly changes
    the recommended index. Ordering, opclasses and expression indexes are *not* detectable
    this way and remain a documented limitation of the reconstruction rather than a guard.
    """
    return _NON_BTREE_RE.search(ddl) is not None


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
    """Which model this is and how it is built, as one operator-facing phrase.

    An absent materialization says so once, in words. Interpolating it directly rendered
    "materialized as `None`" — a Python literal leaking into a sentence an operator reads —
    and `materialized=""` rendered "materialized as ``", an empty code span, in both cases
    *beside* a second spelling of the same fact ("materialization '(absent)'") supplied by
    the caller. One fact, one phrase, and callers no longer restate it.
    """
    if not model.materialized:
        return f"`{model.unique_id}`, with no materialization recorded in the manifest"
    return f"`{model.unique_id}`, materialized as `{model.materialized}`"


def _dbt_ddl_note(model: ModelNode, reason: str) -> str:
    """A `Proposal.note` for a statement that stays executable on a dbt-managed relation.

    ADV302 declines to rewrite on several paths (a partial index, an unrecognised
    materialization, no plain column list, a non-btree access method) and each leaves real,
    runnable DDL in place. The explanation for that used to live only in `rationale`, which
    never reaches the `--ddl` file — so that file could hold a config block explaining that
    raw DDL is destroyed by `dbt run` and, a few lines below, a bare `CREATE INDEX` on that
    same dbt-managed table. This is the disclosure that travels with the statement instead.

    No backticks and no markdown: this is rendered into a SQL script as `--` comment lines,
    where markdown emphasis is noise. Pre-wrapped rather than one long line for the same
    reason — `_comment_lines` prefixes each physical line and wraps nothing.
    """
    return (
        f"dbt WARNING: this relation is built by dbt model {model.unique_id}\n"
        f"({_built_as(model)}), so the statement below is not durable. {reason}\n"
        "Reapply it by hand after any rebuild, or it silently disappears."
    )


def _built_as(model: ModelNode) -> str:
    return model.materialized if model.materialized else "materialization not recorded"


def _dbt_drop_note(model: ModelNode) -> str:
    """A `Proposal.note` for a `DROP INDEX` on a relation dbt manages.

    The mirror image of the bug ADV302 exists to fix. A dropped index that the model's
    `indexes:` config still declares is put straight back by the next `dbt run`, so the
    statement below silently reverts and this tool proposes the same drop again next time —
    unless the config entry goes too. Conditional on the config declaring it, which is why it
    is worded as a condition rather than a verdict: nothing here can read the model's `.yml`,
    only the manifest's materialization.
    """
    return (
        f"dbt WARNING: this relation is built by dbt model {model.unique_id}\n"
        f"({_built_as(model)}). If this index is declared in that model's indexes\n"
        "config, the next dbt run recreates it: remove the config entry as well,\n"
        "or the drop below does not stick and will be proposed again next run."
    )


@dataclass(frozen=True)
class _IndexEntry:
    """One `- columns: [...]` item in a dbt model's `indexes` config list.

    Frozen and comparable so two proposals that reduce to the same index (same columns, same
    uniqueness) contribute one entry to the merged block rather than two identical ones.
    """

    columns: tuple[str, ...]
    unique: bool

    def render(self) -> list[str]:
        # `!r` is deliberately absent: `list(columns)` has no `__str__` of its own, so plain
        # `{list(...)}` formatting already falls back to `__repr__` and gets the same
        # per-element escaping `!r` would have asked for explicitly — see `_comment_block`.
        lines = [f"    - columns: {list(self.columns)}", "      type: btree"]
        if self.unique:
            lines.append("      unique: true")
        return lines


def enrich_proposals(proposals: list[Proposal], context: DbtContext) -> list[Proposal]:
    """Rewrite index-creating proposals whose relation dbt manages; attribute the rest.

    A `CREATE INDEX` (or `CREATE UNIQUE INDEX`, optionally `CONCURRENTLY`) proposal on a
    `table`-, `incremental`- or `materialized_view`-materialized dbt model is expressed
    instead as a config block a human can paste into that model's `.yml`, since the raw
    DDL is destroyed the next time dbt rebuilds the relation. A `view` cannot carry an
    index at all, so the *DDL* is dropped and explained — the proposal itself survives at
    LOW, since "this index cannot apply here" is the finding. An unrecognised (or absent)
    materialization is left alone — unknown is not the same as known-safe, so the DDL is
    not touched on a guess.

    **One model gets exactly one `indexes` block, however many proposals it collects.**
    This is the reason for the two passes below and not an optimization. `indexes` is a
    single YAML mapping key, so two standalone blocks pasted into one model's config are a
    duplicate key and PyYAML — dbt's own parser — silently keeps the last: the other
    recommended index is discarded with no error at all. Two survivors per relation is the
    *normal* case, not an edge case, because the adapter's collapse layer never folds
    non-prefix column lists and deliberately preserves same-set-different-order pairs. So
    the first (highest-ranked) proposal for a model carries the complete merged block, and
    every later one for that same model is rewritten to point at it rather than emit a
    second block. Per-model rather than per-relation only in spelling: `DbtContext` indexes
    one model per relation.

    Everything else passes through with only its evidence enriched: a `DROP INDEX`
    proposal is ordinary regardless of dbt (dbt never created the index, so there is
    nothing for its `indexes` config to un-express), and an advisory proposal with no DDL
    has nothing to rewrite either. Both are still attributed to the model they concern, so
    a reader knows where to make the fix.

    A proposal whose relation dbt does not manage — or that carries no `(schema, table)`
    evidence at all — is returned completely unchanged.

    Output order is the input order. The caller re-sorts by the adapter's own ranking key
    after appending ADV301/ADV303, and this function must not pre-empt that.
    """
    # Pass 1: decide every proposal that can be decided alone, and collect the index entries
    # of the ones that cannot — a config block cannot be rendered until every proposal for
    # that model has been seen.
    models: list[ModelNode | None] = []
    decided: list[Proposal | None] = []
    entries: list[_IndexEntry | None] = []
    for proposal in proposals:
        relation = _relation_of(proposal)
        model = context.model_for(relation) if relation is not None else None
        models.append(model)
        if model is None:
            decided.append(proposal)
            entries.append(None)
            continue
        finished, entry = _classify(proposal, model)
        decided.append(finished)
        entries.append(entry)

    merged: dict[str, list[_IndexEntry]] = {}
    owner: dict[str, int] = {}
    for position, (model, entry) in enumerate(zip(models, entries)):
        if model is None or entry is None:
            continue
        block = merged.setdefault(model.unique_id, [])
        if entry not in block:
            block.append(entry)
        owner.setdefault(model.unique_id, position)

    # Pass 2: render the config block once per model.
    out: list[Proposal] = []
    for position, proposal in enumerate(proposals):
        finished = decided[position]
        if finished is not None:
            out.append(finished)
            continue
        model = models[position]
        entry = entries[position]
        assert model is not None and entry is not None  # the only shape pass 1 leaves undecided
        if owner[model.unique_id] == position:
            out.append(_as_config_block(proposal, model, merged[model.unique_id]))
        else:
            out.append(
                _deferred_to_block(proposal, model, proposals[owner[model.unique_id]], entry)
            )
    return out


def _dbt_evidence(proposal: Proposal, model: ModelNode) -> dict[str, object]:
    """`proposal.evidence` plus the model attribution — as a *copy*.

    Copied, not mutated in place: enrichment is a transformation over proposals the adapter
    already produced, and quietly editing the caller's dict would make the same proposal
    object read differently depending on whether enrichment ran.
    """
    evidence = dict(proposal.evidence)
    evidence["dbt_model"] = model.unique_id
    evidence["dbt_materialized"] = model.materialized
    return evidence


def _classify(proposal: Proposal, model: ModelNode) -> tuple[Proposal | None, _IndexEntry | None]:
    """Either the finished proposal, or the index entry it contributes to a merged block.

    Exactly one of the two is non-None. Returning `(None, entry)` means "this one becomes
    dbt config, but which text it gets depends on the other proposals for this model", which
    only `enrich_proposals`'s second pass can know.
    """
    evidence = _dbt_evidence(proposal, model)

    if not _is_index_creating(proposal.ddl):
        # Not rewritten: there is no `indexes` config entry that expresses a *removal*, and
        # an advisory proposal has no DDL to rewrite. But "dbt never created this index, so a
        # drop is ordinary" — the original reasoning here — is false in exactly the case this
        # module exists for. If the index *is* declared in the model's `indexes:` config, the
        # next `dbt run` recreates it: the operator drops it, dbt puts it back, and the next
        # run of this tool proposes dropping it again. That is the same silently-reverting
        # advice ADV302 was built to eliminate, pointing the other way, so it is disclosed —
        # in the rationale *and*, since `render_ddl` never emits a rationale, in a note beside
        # the statement itself.
        #
        # Not suppressed: dropping a genuinely unused index is still the right call, and the
        # operator needs both halves of the instruction, not neither.
        if proposal.ddl is None:
            # Nothing is emitted into the DDL script, so there is no statement to warn
            # beside, and `rationale` already reaches every surface that shows this proposal.
            return dataclasses.replace(proposal, evidence=evidence), None
        if _is_index_dropping(proposal.ddl):
            rationale = (
                f"{proposal.rationale} This relation is a dbt model "
                f"({_dbt_attribution(model)}): if this index is declared in that model's "
                "`indexes:` config, `dbt run` recreates it, so the config entry has to be "
                "removed as well or the drop will not stick — and this proposal will come "
                "back on the next run."
            )
            return (
                dataclasses.replace(
                    proposal, rationale=rationale, evidence=evidence, note=_dbt_drop_note(model)
                ),
                None,
            )
        # Some other statement for a relation dbt owns — no rule emits one today, and
        # `_is_index_creating` matches by DDL prefix specifically so this stays covered when
        # one does. Unknown shape, so the note claims only what is certainly true.
        rationale = (
            f"{proposal.rationale} This relation is a dbt model ({_dbt_attribution(model)}), "
            "which dbt rebuilds on its own schedule, so this statement is not expressed as "
            "dbt config and may not outlive the next rebuild."
        )
        note = _dbt_ddl_note(
            model,
            "dbt rebuilds this relation on its own schedule, and this statement is not\n"
            "expressed as dbt config.",
        )
        return dataclasses.replace(
            proposal, rationale=rationale, evidence=evidence, note=note
        ), None

    ddl = proposal.ddl
    assert ddl is not None  # _is_index_creating(None) is False, so this branch guarantees it
    materialized = model.materialized

    if materialized == "view":
        rationale = (
            f"{proposal.rationale} This relation is a dbt view ({_dbt_attribution(model)}): "
            "a view has no storage of its own to index, so this proposal does not apply."
        )
        return (
            dataclasses.replace(
                proposal,
                ddl=None,
                rationale=rationale,
                confidence=Confidence.LOW,
                evidence=evidence,
            ),
            None,
        )

    if materialized not in _REBUILD:
        why = (
            f"but materialization '{materialized}' is unrecognised"
            if materialized
            else "but an unrecorded materialization is unknown rather than known-safe"
        )
        rationale = (
            f"{proposal.rationale} This relation is a dbt model ({_dbt_attribution(model)}), "
            f"{why}, so the DDL below is left as-is rather than rewritten on a guess."
        )
        note = _dbt_ddl_note(
            model,
            "sqlquality does not recognise that materialization, so it cannot tell whether\n"
            "a dbt run destroys this index.",
        )
        return dataclasses.replace(
            proposal, rationale=rationale, evidence=evidence, note=note
        ), None

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
        note = _dbt_ddl_note(
            model,
            "dbt's indexes config has no predicate field, so this partial index cannot be\n"
            "expressed as config.",
        )
        return dataclasses.replace(
            proposal, rationale=rationale, evidence=evidence, note=note
        ), None

    columns = proposal.evidence.get("columns")
    if (
        not isinstance(columns, (tuple, list))
        or not columns
        or not all(isinstance(c, str) for c in columns)
    ):
        # No plain column list to express as config — leave the DDL untouched rather than
        # invent one, and *say so*. Unreachable from today's rules, all of which populate
        # `columns`; reachable by design, because `_is_index_creating` matches on the DDL
        # prefix precisely so a future index-creating rule is covered without being
        # enumerated here. This path used to decline in complete silence — no rationale
        # amendment, executable DDL kept — which is the one outcome this module exists to
        # prevent, so the disclosure matters more here than on the paths that are exercised.
        rationale = (
            f"{proposal.rationale} This relation is a dbt model ({_dbt_attribution(model)}): "
            f"{_REBUILD[materialized]}. This proposal carries no plain column list, so it "
            "cannot be expressed as dbt `indexes` config and the DDL below is left as-is — "
            "reapply it by hand after each rebuild."
        )
        note = _dbt_ddl_note(
            model,
            "This proposal carries no plain column list, so it cannot be expressed as\n"
            "dbt indexes config.",
        )
        return dataclasses.replace(
            proposal, rationale=rationale, evidence=evidence, note=note
        ), None

    if _names_a_non_btree_method(ddl):
        rationale = (
            f"{proposal.rationale} This relation is a dbt model ({_dbt_attribution(model)}): "
            f"{_REBUILD[materialized]}. This index names a non-btree access method, which the "
            "config block below cannot express without changing the index it recommends, so "
            "the DDL is left as-is — reapply it by hand after each rebuild."
        )
        note = _dbt_ddl_note(
            model,
            "This index names a non-btree access method, which dbt indexes config as\n"
            "reconstructed here cannot express.",
        )
        return dataclasses.replace(
            proposal, rationale=rationale, evidence=evidence, note=note
        ), None

    return None, _IndexEntry(columns=tuple(columns), unique=_is_unique_index(ddl))


def _as_config_block(proposal: Proposal, model: ModelNode, entries: list[_IndexEntry]) -> Proposal:
    """Rewrite `proposal`'s DDL as the one `indexes` block covering this whole model.

    The block names the model. It used to say only "the model's config block", so two
    different relations recommending the same column list rendered byte-identical `ddl` —
    distinguishable in the DDL file only by the title comment above it, and in the JSON
    payload by nothing at all. It also no longer says "above": in markdown the fenced DDL is
    *below* the rationale, in the DDL file the rationale is absent entirely, and in JSON
    there is no spatial relation to be right or wrong about.
    """
    lines = [
        "ADV302: express this as dbt config, not DDL — raw DDL does not survive a rebuild.",
        f"Add to the config of dbt model {model.unique_id}:",
    ]
    if len(entries) > 1:
        lines += [
            f"(all {len(entries)} indexes this run recommends for that model, in ONE block on",
            "purpose: an `indexes` mapping key can appear once, so two blocks pasted into the",
            "same config silently keep only the last)",
        ]
    lines.append("  indexes:")
    for entry in entries:
        lines.extend(entry.render())
    config_ddl = _comment_block(lines)
    evidence = _dbt_evidence(proposal, model)
    # A flag, not the block itself. This key used to hold a byte-identical copy of `ddl`,
    # which earned nothing and smeared the markdown Evidence line — evidence renders as flat
    # `k=v` pairs, so a multi-line value lands inline with literal `\n` escapes mid-sentence.
    # As a flag it is the one thing a consumer cannot get elsewhere: ADV302 is never a
    # proposal `code`, so `code == "ADV302"` matches nothing and this is how a `--json`
    # consumer filters for "the DDL here is dbt config, not runnable SQL".
    evidence["dbt_index_config"] = True
    # `or ""` is for the type checker only: `_classify` returns an entry — the sole way to
    # reach this function — exactly when `materialized` is a key of `_REBUILD`.
    rationale = (
        f"{proposal.rationale} This relation is a dbt model ({_dbt_attribution(model)}): "
        f"{_REBUILD[model.materialized or '']}. Add the config block this proposal carries in "
        f"place of its DDL to `{model.unique_id}` instead of running that DDL directly; "
        "`dbt run` applies it."
    )
    if len(entries) > 1:
        rationale += (
            f" That block covers all {len(entries)} indexes this run recommends for the model, "
            "because dbt reads only one `indexes` key per config."
        )
    return dataclasses.replace(proposal, ddl=config_ddl, rationale=rationale, evidence=evidence)


def _deferred_to_block(
    proposal: Proposal, model: ModelNode, owner: Proposal, entry: _IndexEntry
) -> Proposal:
    """Point a second index proposal for one model at that model's single config block.

    Emitting its own standalone block instead is the silent-data-loss bug this exists to
    prevent: two `indexes` keys in one config, and dbt keeps the last.
    """
    lines = [
        f"ADV302: the index on {list(entry.columns)} for dbt model {model.unique_id}",
        f"is already included in the single dbt config block reported under {owner.code}:",
        # `_comment_block` splits each logical line again, so an interpolated title carrying
        # a raw newline still cannot produce an uncommented output line.
        owner.title,
        "Paste that one block. A second `indexes` key in the same config would be a",
        "duplicate YAML mapping key, and dbt would silently keep only one of them.",
    ]
    evidence = _dbt_evidence(proposal, model)
    evidence["dbt_index_config_reported_with"] = owner.code
    rationale = (
        f"{proposal.rationale} This relation is a dbt model ({_dbt_attribution(model)}): "
        f"{_REBUILD[model.materialized or '']}. This index is already part of the single dbt "
        f"`indexes` config block reported under {owner.code} for this model — dbt reads one "
        "`indexes` key per config, so both indexes have to be expressed in that one block "
        "rather than in a block of their own."
    )
    return dataclasses.replace(
        proposal, ddl=_comment_block(lines), rationale=rationale, evidence=evidence
    )


def describe_rewrites(proposals: list[Proposal]) -> str | None:
    """One line saying ADV302 fired, or None when it did not.

    ADV302 is never a proposal `code` — it is a rewrite applied to another rule's proposal —
    so `code == "ADV302"` matches nothing and an enriched terminal row is byte-identical to
    the same proposal from a dbt-free run: same code, same confidence, same cost share, same
    title. The terminal never prints `rationale`, where the whole disclosure lives, so
    without this line a user who reads only the terminal cannot tell enrichment happened.
    Counted off the two evidence flags rather than by searching the DDL text for "ADV302",
    which would depend on the wording of a string meant for humans.
    """
    rewritten = sum(1 for p in proposals if p.evidence.get("dbt_index_config") is True)
    merged = sum(1 for p in proposals if "dbt_index_config_reported_with" in p.evidence)
    if not rewritten and not merged:
        return None
    line = (
        f"ADV302 expressed {rewritten + merged} index proposal(s) as dbt `indexes` config: "
        "their DDL is a config block to add to the model, not runnable SQL"
    )
    if merged:
        line += (
            f" ({merged} folded into another proposal's block, since dbt reads one `indexes` "
            "key per model config)"
        )
    return line


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
            "merely downgraded, because it is used, just not by an ad-hoc query. That "
            "exclusion is not transitive: only a model's immediate consumers are considered, "
            "so a dead chain surfaces one model per run, from its leaf — if this model feeds "
            "another that turns out to be dead too, that one is only reported after this one "
            "is gone."
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
