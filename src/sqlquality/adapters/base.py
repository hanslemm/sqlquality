"""PerfAdapter interface — one per SQL dialect/engine."""

from __future__ import annotations

from abc import ABC, abstractmethod

from sqlquality.models import Finding


class PerfAdapter(ABC):
    """Per-engine performance analyzer (static + EXPLAIN-plan)."""

    engine: str

    @abstractmethod
    def static_findings(self, sql: str) -> list[Finding]:
        """Static anti-pattern findings from the SQL text."""

    @abstractmethod
    def plan_findings(self, explain_text: str) -> list[Finding]:
        """Findings from a captured EXPLAIN output (raw file text; format per engine)."""
