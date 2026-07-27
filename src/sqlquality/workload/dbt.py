"""Optional dbt enrichment for `advise`.

dbt is *layered on top of* the engine-agnostic core, never underneath it: no workload adapter
imports this module, and every `advise` run behaves identically without a manifest. The
project's positioning is that the dbt-free path is first-class, so enrichment has to be
additive by construction rather than by discipline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sqlquality.dbtproject import DbtProject, DbtProjectError, ModelNode
from sqlquality.models import Relation


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


@dataclass(frozen=True)
class DbtContext:
    """dbt models indexed by the relation they build, for joining against workload facts."""

    models: dict[Relation, ModelNode]
    #: Relations dropped because two *different* models claimed the same (schema, table) —
    #: see `from_project`. Surfaced so the CLI disclosure can tell a user "we found nothing"
    #: apart from "we found two candidates and refused to guess."
    dropped_collisions: int = 0

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
        return cls(models=models, dropped_collisions=len(collided))

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
