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

#: dbt's env_var() Jinja call, in both its forms: env_var('NAME') and the two-argument
#: env_var('NAME', 'default'). The second form is common in real profiles, and matching
#: only the first leaves literal `{{ ... }}` text in a host or port value.
_ENV_VAR = re.compile(
    r"\{\{\s*env_var\(\s*['\"](?P<name>[A-Za-z_][A-Za-z0-9_]*)['\"]"
    r"(?:\s*,\s*['\"](?P<default>[^'\"]*)['\"])?\s*\)\s*\}\}"
)
#: Any Jinja left after interpolation. sqlquality resolves env_var() only — it is not a
#: Jinja engine — and passing an unresolved `{{ ... }}` through as a hostname produces a
#: baffling connection error, so this fails loudly instead.
_UNRESOLVED_JINJA = re.compile(r"\{\{|\{%")

#: dbt adapter type -> sqlquality engine name.
ENGINE_BY_DBT_TYPE = {"postgres": "postgres", "redshift": "redshift", "snowflake": "snowflake"}


class ProfileError(ValueError):
    """Raised when profiles.yml is missing, malformed, or references an unset env var."""


def _interpolate(key: str, value: object, env: Mapping[str, str]) -> str:
    """Substitute env_var() references, or raise naming the missing variable.

    Never includes the resolved value in an error message: these fields can hold a
    password, and an exception message ends up in CI logs and stack traces.
    """
    text = str(value)

    def replace(match: re.Match[str]) -> str:
        name = match.group("name")
        if name in env:
            return env[name]
        default = match.group("default")
        if default is not None:
            return default
        raise ProfileError(f"profiles.yml references env_var('{name}') but {name} is not set")

    resolved = _ENV_VAR.sub(replace, text)
    if _UNRESOLVED_JINJA.search(resolved):
        # The field name only — never the value, which may be or contain a secret.
        raise ProfileError(
            f"profiles.yml field '{key}' contains Jinja that sqlquality cannot resolve. "
            "Only env_var('NAME') and env_var('NAME', 'default') are supported — render "
            "the profile with dbt first, or pass --dsn instead."
        )
    return resolved


def _yaml_location(exc: yaml.YAMLError) -> str:
    """Position and reason for a YAML error, *without* PyYAML's quoted source snippet.

    ``str(exc)`` embeds the offending line verbatim, so a syntax error on a ``password:``
    line would put the secret into the message. Only the mark and the parser's reason are
    value-free enough to repeat.
    """
    mark = getattr(exc, "problem_mark", None)
    problem = getattr(exc, "problem", None)
    where = f" at line {mark.line + 1}, column {mark.column + 1}" if mark else ""
    why = f": {problem}" if problem else ""
    return f"{where}{why}"


def read_profiles_file(profiles_dir: Path) -> dict:
    """Load profiles.yml from a directory, or raise ProfileError."""
    path = Path(profiles_dir) / "profiles.yml"
    # The YAML failure is recorded here and raised *after* the handler exits. This is not
    # style. Raising inside an `except` block populates the new exception's __context__
    # with the original, whatever the `from` clause says — `from None` only sets
    # __suppress_context__, which hides it from a printed traceback but leaves it
    # reachable to anything that walks the chain. Since PyYAML's message quotes the
    # offending source line, a syntax error on a `password:` line would otherwise leave
    # the secret retrievable. Leaving the handler first is what actually removes it.
    yaml_problem: str | None = None
    try:
        raw = yaml.safe_load(path.read_text())
    except FileNotFoundError:
        raise ProfileError(f"No profiles.yml in {profiles_dir}") from None
    except OSError as exc:
        # OSError carries the path and errno, never file content — safe to chain.
        raise ProfileError(f"Could not read {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        yaml_problem = _yaml_location(exc)
    if yaml_problem is not None:
        raise ProfileError(f"Malformed YAML in {path}{yaml_problem}")
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
        if chosen is None:
            # A profile with no `target:` key and no --target: say that, rather than
            # rendering the literal "No target 'None'".
            raise ProfileError(
                f"Profile '{profile}' sets no default target — pass --target "
                f"(available: {available})"
            )
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
    fields = {k: _interpolate(k, v, env) for k, v in output.items() if k != "type"}
    return engine, fields
