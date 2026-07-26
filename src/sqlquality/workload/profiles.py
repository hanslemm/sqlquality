"""Read a dbt profiles.yml — a convenience for dbt users, never a requirement.

sqlquality is not a dbt tool: `advise` works against any database via --dsn or
SQLQUALITY_DSN. This module exists only so dbt users need not restate connection
details they already have.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path

import yaml

#: dbt's env_var() Jinja call, the one templating form that appears in real profiles.
_ENV_VAR = re.compile(r"\{\{\s*env_var\(\s*['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]\s*\)\s*\}\}")

#: dbt adapter type -> sqlquality engine name.
ENGINE_BY_DBT_TYPE = {"postgres": "postgres", "redshift": "redshift", "snowflake": "snowflake"}


class ProfileError(ValueError):
    """Raised when profiles.yml is missing, malformed, or references an unset env var."""


def _interpolate(value: object, env: Mapping[str, str]) -> str:
    """Substitute env_var() references, or raise naming the missing variable."""
    text = str(value)

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in env:
            raise ProfileError(f"profiles.yml references env_var('{name}') but {name} is not set")
        return env[name]

    return _ENV_VAR.sub(replace, text)


def read_profiles_file(profiles_dir: Path) -> dict:
    """Load profiles.yml from a directory, or raise ProfileError."""
    path = Path(profiles_dir) / "profiles.yml"
    try:
        raw = yaml.safe_load(path.read_text())
    except FileNotFoundError:
        raise ProfileError(f"No profiles.yml in {profiles_dir}")
    except OSError as exc:
        raise ProfileError(f"Could not read {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ProfileError(f"Malformed YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ProfileError(f"top-level of {path} must be a mapping")
    return raw


def read_output(
    profiles_dir: Path, profile: str, target: str | None, env: Mapping[str, str]
) -> tuple[str, dict[str, str]]:
    """Return (engine, connection fields) for one profile/target."""
    profiles = read_profiles_file(profiles_dir)
    block = profiles.get(profile)
    if not isinstance(block, dict):
        available = ", ".join(k for k in profiles if k != "config") or "none"
        raise ProfileError(f"No profile '{profile}' in profiles.yml (found: {available})")

    chosen = target or block.get("target")
    outputs = block.get("outputs")
    if not isinstance(outputs, dict) or chosen not in outputs:
        available = ", ".join(outputs) if isinstance(outputs, dict) else "none"
        raise ProfileError(f"No target '{chosen}' in profile '{profile}' (found: {available})")
    output = outputs[chosen]
    if not isinstance(output, dict):
        raise ProfileError(f"Target '{chosen}' in profile '{profile}' must be a mapping")

    dbt_type = str(output.get("type", "")).lower()
    engine = ENGINE_BY_DBT_TYPE.get(dbt_type)
    if engine is None:
        raise ProfileError(
            f"dbt adapter type '{dbt_type}' has no workload adapter. "
            f"Supported: {', '.join(sorted(ENGINE_BY_DBT_TYPE))}."
        )
    fields = {k: _interpolate(v, env) for k, v in output.items() if k != "type"}
    return engine, fields
