"""Read a dbt project's manifest.json (schema v12) as a model graph."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from sqlquality.models import DagFacts


class DbtProjectError(ValueError):
    """Raised when the manifest is malformed or a node is missing/uncompiled."""


@dataclass(frozen=True)
class ModelNode:
    unique_id: str
    name: str
    resource_type: str
    materialized: str | None
    compiled_code: str | None
    relation_name: str | None
    depends_on: list[str]
    config: dict


class DbtProject:
    """A loaded dbt manifest, indexed for model-graph queries."""

    def __init__(self, manifest: dict) -> None:
        self._manifest = manifest
        self._nodes: dict = manifest.get("nodes", {})
        self._parent_map: dict = manifest.get("parent_map", {})
        self._child_map: dict = manifest.get("child_map", {})
        self._depth_cache: dict[str, int] = {}

    @classmethod
    def from_manifest(cls, manifest: dict) -> "DbtProject":
        return cls(manifest)

    @classmethod
    def from_path(cls, manifest_path: str | Path) -> "DbtProject":
        path = Path(manifest_path)
        try:
            manifest = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise DbtProjectError(f"Could not read manifest {path}: {exc}") from exc
        return cls(manifest)

    def adapter_type(self) -> str:
        return self._manifest.get("metadata", {}).get("adapter_type", "")

    def schema_version(self) -> str:
        return self._manifest.get("metadata", {}).get("dbt_schema_version", "")

    def _is_model(self, uid: str) -> bool:
        node = self._nodes.get(uid)
        return node is not None and node.get("resource_type") == "model"

    def model_ids(self) -> list[str]:
        return sorted(uid for uid in self._nodes if self._is_model(uid))

    def node(self, uid: str) -> ModelNode:
        raw = self._nodes.get(uid)
        if raw is None:
            raise DbtProjectError(f"No such node: {uid}")
        config = raw.get("config") or {}
        return ModelNode(
            unique_id=uid,
            name=raw.get("name", ""),
            resource_type=raw.get("resource_type", ""),
            materialized=config.get("materialized"),
            compiled_code=raw.get("compiled_code"),
            relation_name=raw.get("relation_name"),
            depends_on=list(raw.get("depends_on", {}).get("nodes", [])),
            config=config,
        )

    def model_parents(self, uid: str) -> list[str]:
        return sorted(p for p in self._parent_map.get(uid, []) if self._is_model(p))

    def model_children(self, uid: str) -> list[str]:
        return sorted(c for c in self._child_map.get(uid, []) if self._is_model(c))

    def compiled_sql(self, uid: str) -> str:
        node = self.node(uid)
        if not node.compiled_code:
            raise DbtProjectError(
                f"{uid} has no compiled_code — run `dbt compile` first"
            )
        return node.compiled_code

    def dag_facts(self, uid: str) -> DagFacts:
        return DagFacts(
            fan_in=len(self.model_parents(uid)),
            fan_out=len(self.model_children(uid)),
            lineage_depth=self._lineage_depth(uid),
        )

    def _lineage_depth(self, uid: str) -> int:
        if uid in self._depth_cache:
            return self._depth_cache[uid]
        parents = self.model_parents(uid)
        depth = 1 + max((self._lineage_depth(p) for p in parents), default=0)
        self._depth_cache[uid] = depth
        return depth
