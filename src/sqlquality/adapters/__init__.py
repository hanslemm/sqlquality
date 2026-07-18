"""Perf adapter registry."""

from __future__ import annotations

from sqlquality.adapters.base import PerfAdapter
from sqlquality.adapters.postgres import PostgresAdapter
from sqlquality.adapters.redshift import RedshiftAdapter

_ADAPTERS: dict[str, type[PerfAdapter]] = {
    "postgres": PostgresAdapter,
    "redshift": RedshiftAdapter,
}


def get_adapter(engine: str) -> PerfAdapter:
    """Return the perf adapter for an engine, or raise ValueError."""
    try:
        return _ADAPTERS[engine]()
    except KeyError:
        raise ValueError(f"No perf adapter for dialect '{engine}'")
