"""Optional dbt enrichment for `advise`.

dbt is *layered on top of* the engine-agnostic core, never underneath it: no workload adapter
imports this module, and every `advise` run behaves identically without a manifest. The
project's positioning is that the dbt-free path is first-class, so enrichment has to be
additive by construction rather than by discipline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from sqlquality.dbtproject import DbtProject, DbtProjectError, ModelNode
from sqlquality.models import Relation

#: Splits a dbt `relation_name` on unquoted dots. dbt quotes each part, so a dot *inside* a
#: quoted identifier (`"Weird.Name"`) must not split — hence matching quoted segments first.
_PART = re.compile(r'"((?:[^"]|"")*)"|([^.]+)')


def parse_relation_name(relation_name: str) -> Relation | None:
    """`(schema, table)` from a dbt `relation_name`, or None if it cannot be qualified.

    dbt writes a quoted three-part name — `'"dev"."main"."stg_orders"'` — and the raw node's
    own `schema` field is `None` in practice, so this string is the only reliable source. The
    database part is dropped because `advise` connects to one database at a time.

    A name with fewer than two parts returns None rather than a guess: a `Relation` needs a
    schema, and inventing one is how a production table gets attributed to an unrelated model.
    """
    # `re.findall` represents a non-participating alternative as "", not None — so a bare
    # segment's `quoted` slot and a quoted segment's `bare` slot are indistinguishable from
    # an actually-empty quoted identifier ('""'). `quoted or bare` still resolves correctly:
    # when the quoted group truly matched, it wins over the (also empty) bare slot; when it
    # didn't, `bare` carries the real text.
    parts = [
        (quoted or bare).replace('""', '"') for quoted, bare in _PART.findall(relation_name.strip())
    ]
    parts = [p for p in parts if p]
    if len(parts) < 2:
        return None
    return Relation(schema=parts[-2], table=parts[-1])


@dataclass(frozen=True)
class DbtContext:
    """dbt models indexed by the relation they build, for joining against workload facts."""

    models: dict[Relation, ModelNode]

    @classmethod
    def from_project(cls, project: DbtProject) -> DbtContext:
        """Index every *model* by its relation.

        Only `resource_type == "model"` is indexed. Seeds, tests and snapshots also occupy
        relations, but "materialize this dbt test as a table" and "express this seed's index
        as dbt config" are both nonsense, and a seed sharing a name with a model's relation
        would otherwise silently win the mapping.
        """
        models: dict[Relation, ModelNode] = {}
        for uid in project.model_ids():
            node = project.node(uid)
            if node.resource_type != "model" or not node.relation_name:
                continue
            relation = parse_relation_name(node.relation_name)
            if relation is not None:
                models[relation] = node
        return cls(models=models)

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
    if manifest is None and project_dir is None:
        return None, None
    path = (
        manifest if manifest is not None else (project_dir or Path()) / "target" / "manifest.json"
    )
    try:
        project = DbtProject.from_path(path)
    except (OSError, ValueError, DbtProjectError) as exc:
        # `ValueError` covers `json.JSONDecodeError`, and `DbtProjectError` is a ValueError
        # subclass — both listed so the intent survives a refactor of either.
        return None, f"dbt enrichment unavailable: could not read {path}: {exc}"
    context = DbtContext.from_project(project)
    return context, f"dbt enrichment from {path} ({len(context.models)} model(s))"
