"""Validation of SQL dialect names against SQLGlot's registry."""

from __future__ import annotations

import sqlglot

#: A short, friendly sample of dialects to suggest when an unknown name is given.
KNOWN_DIALECTS: tuple[str, ...] = ("postgres", "redshift", "snowflake", "bigquery", "duckdb")


def validate_dialect(name: str) -> str:
    """Return the normalized dialect name, or raise ``ValueError`` if unknown.

    The name is lowercased and stripped, then checked against SQLGlot's dialect
    registry via :meth:`sqlglot.Dialect.get`. An empty/blank name is rejected too,
    since it is not a meaningful dialect to validate.
    """
    normalized = name.strip().lower()
    if not normalized or sqlglot.Dialect.get(normalized) is None:
        raise ValueError(
            f"Unknown SQL dialect {name!r}. Known dialects include: {', '.join(KNOWN_DIALECTS)}."
        )
    return normalized
