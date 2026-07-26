"""Resolve connection details: explicit flag > environment variable > profiles.yml."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from urllib.parse import urlparse

from sqlquality.models import ConnectionParams
from sqlquality.workload.profiles import ProfileError, read_output

#: Environment variable holding a full DSN.
DSN_ENV_VAR = "SQLQUALITY_DSN"

#: DSN URL scheme -> sqlquality engine name.
ENGINE_BY_SCHEME = {
    "postgresql": "postgres",
    "postgres": "postgres",
    "redshift": "redshift",
    "redshift+psycopg2": "redshift",
    "snowflake": "snowflake",
}


class ConnectionResolutionError(ValueError):
    """Raised when connection details cannot be resolved from any source."""


def _engine_from_dsn(dsn: str) -> str:
    scheme = urlparse(dsn).scheme.lower()
    engine = ENGINE_BY_SCHEME.get(scheme)
    if engine is None:
        raise ConnectionResolutionError(
            f"Unsupported DSN scheme {scheme or '(none)'!r}. "
            f"Supported: {', '.join(sorted(ENGINE_BY_SCHEME))}."
        )
    return engine


def read_profile(
    profiles_dir: Path, profile: str, target: str | None, env: Mapping[str, str]
) -> tuple[str, dict[str, str]]:
    """Read (engine, fields) from profiles.yml, re-raising as ConnectionResolutionError."""
    try:
        return read_output(profiles_dir, profile, target, env)
    except ProfileError as exc:
        # `from None`, not `from exc`: str(exc) is already folded into this message, so
        # chaining adds nothing — and profiles.py's own suppression (for the YAML-error
        # case) would otherwise be defeated one boundary up, by re-attaching the
        # ProfileError as this exception's __cause__.
        raise ConnectionResolutionError(str(exc)) from None


def resolve_connection(
    *,
    dsn: str | None,
    engine: str | None,
    profile: str | None,
    target: str | None,
    profiles_dir: Path | None,
    env: Mapping[str, str],
) -> ConnectionParams:
    """Resolve connection details, honoring flag > env > profiles.yml precedence."""
    if dsn:
        return ConnectionParams(
            engine=engine or _engine_from_dsn(dsn), dsn=dsn, fields={}, source="--dsn"
        )

    env_dsn = env.get(DSN_ENV_VAR)
    if env_dsn:
        return ConnectionParams(
            engine=engine or _engine_from_dsn(env_dsn), dsn=env_dsn, fields={}, source="env"
        )

    if profile:
        directory = profiles_dir or Path.home() / ".dbt"
        profile_engine, fields = read_profile(directory, profile, target, env)
        return ConnectionParams(
            engine=engine or profile_engine, dsn=None, fields=fields, source="profiles.yml"
        )

    raise ConnectionResolutionError(
        "No connection details. Pass --dsn, set SQLQUALITY_DSN, "
        "or pass --profile (with an optional --target) to read from profiles.yml."
    )
