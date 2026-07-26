"""Workload analysis: query-history ingestion, column-usage rollup, per-engine adapters."""

from __future__ import annotations

from sqlquality.workload.base import WorkloadAdapter
from sqlquality.workload.postgres import PostgresWorkloadAdapter

_ADAPTERS: dict[str, type[WorkloadAdapter]] = {
    "postgres": PostgresWorkloadAdapter,
}


def get_workload_adapter(engine: str) -> WorkloadAdapter:
    """Return the workload adapter for an engine, or raise ValueError."""
    try:
        return _ADAPTERS[engine]()
    except KeyError:
        raise ValueError(
            f"No workload adapter for engine '{engine}'. Supported: {', '.join(sorted(_ADAPTERS))}."
        )
