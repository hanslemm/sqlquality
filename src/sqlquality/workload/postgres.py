# src/sqlquality/workload/postgres.py — expanded in Tasks 7-11
"""Postgres workload adapter."""

from __future__ import annotations

from datetime import timedelta

from sqlquality.models import (
    Aggregation,
    ConnectionParams,
    Proposal,
    TableFacts,
    Workload,
    WorkloadFetch,
)
from sqlquality.workload.base import IntrospectionStatement, WorkloadAdapter


class PostgresWorkloadAdapter(WorkloadAdapter):
    engine = "postgres"

    def introspection_sql(self) -> list[IntrospectionStatement]:
        return [
            IntrospectionStatement(
                capability="workload",
                sql="SELECT 1",
                privilege_hint="requires pg_read_all_stats or superuser",
            )
        ]

    def connect(self, params: ConnectionParams, timeout_s: int) -> None:
        raise NotImplementedError

    def fetch_workload(self, since: timedelta | None, limit: int) -> WorkloadFetch:
        raise NotImplementedError

    def fetch_schema(self, schemas: tuple[str, ...]) -> dict:
        raise NotImplementedError

    def fetch_table_facts(
        self, schemas: tuple[str, ...], tables: frozenset[str]
    ) -> dict[str, TableFacts]:
        raise NotImplementedError

    def propose(
        self,
        aggregation: Aggregation,
        facts: dict[str, TableFacts],
        workload: Workload,
        *,
        min_cost_share: float,
    ) -> list[Proposal]:
        raise NotImplementedError

    def render_ddl(self, proposals: list[Proposal]) -> str:
        raise NotImplementedError
