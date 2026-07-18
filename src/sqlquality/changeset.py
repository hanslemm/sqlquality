"""Turn dbt `state:modified` selection into a changed-model ChangeSet."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from sqlquality.dbtproject import DbtProject


class ChangeSetError(RuntimeError):
    """Raised when the `dbt ls` invocation fails."""


@dataclass(frozen=True)
class ChangeSet:
    changed: list[str]
    neighbors: list[str]

    @property
    def analysis_set(self) -> list[str]:
        return sorted(set(self.changed) | set(self.neighbors))


def parse_state_modified(stdout: str) -> list[str]:
    """Extract model unique_ids from `dbt ls --output json` JSONL output."""
    ids: set[str] = set()
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue  # dbt can interleave non-JSON log lines
        if isinstance(obj, dict) and obj.get("resource_type") == "model":
            uid = obj.get("unique_id")
            if uid:
                ids.add(uid)
    return sorted(ids)


def compute_changeset(project: DbtProject, ls_stdout: str) -> ChangeSet:
    """Changed models (from `dbt ls`) plus their 1-hop model neighbors."""
    models = set(project.model_ids())
    changed = [uid for uid in parse_state_modified(ls_stdout) if uid in models]
    changed_set = set(changed)
    neighbors: set[str] = set()
    for uid in changed:
        neighbors.update(project.model_parents(uid))
        neighbors.update(project.model_children(uid))
    neighbors -= changed_set
    return ChangeSet(changed=changed, neighbors=sorted(neighbors))


def run_state_modified(project_dir: str | Path, state_dir: str | Path, dbt: str = "dbt") -> str:
    """Run `dbt ls --select state:modified ... --output json` and return stdout."""
    cmd = [
        dbt,
        "ls",
        "--select",
        "state:modified",
        "--state",
        str(state_dir),
        "--resource-type",
        "model",
        "--output",
        "json",
    ]
    result = subprocess.run(cmd, cwd=Path(project_dir), capture_output=True, text=True)
    if result.returncode != 0:
        raise ChangeSetError(f"`dbt ls` failed (exit {result.returncode}): {result.stderr.strip()}")
    return result.stdout
