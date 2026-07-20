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
    # --no-write-json: `dbt ls` would otherwise rewrite the candidate's
    # target/manifest.json without compiled_code, self-neutralizing the gate on
    # the next run. --state is resolved absolute because the subprocess runs
    # with cwd=project_dir, but the CLI resolves --state against the real cwd.
    cmd = [
        dbt,
        "ls",
        "--no-write-json",
        "--select",
        "state:modified",
        "--state",
        str(Path(state_dir).resolve()),
        "--resource-type",
        "model",
        "--output",
        "json",
    ]
    try:
        result = subprocess.run(
            cmd, cwd=Path(project_dir), capture_output=True, text=True, timeout=600
        )
    except FileNotFoundError as exc:
        raise ChangeSetError(
            f"dbt executable '{dbt}' not found on PATH — install dbt or pass --dbt"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise ChangeSetError(f"`dbt ls` timed out after {exc.timeout:.0f}s") from exc
    if result.returncode != 0:
        # dbt logs errors to stdout, not stderr; fall back to the stdout tail.
        detail = result.stderr.strip()
        if not detail:
            detail = "\n".join(result.stdout.strip().splitlines()[-20:])
        raise ChangeSetError(f"`dbt ls` failed (exit {result.returncode}): {detail}")
    return result.stdout
