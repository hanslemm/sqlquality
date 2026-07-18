"""Optional, provider-agnostic LLM layer for enriching findings (advisory only)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from sqlquality.models import Finding


@runtime_checkable
class LLMProvider(Protocol):
    """Anything that can turn a prompt into a suggestion string."""

    def suggest(self, prompt: str) -> str: ...


@dataclass(frozen=True)
class Suggestion:
    code: str
    text: str


def build_prompt(finding: Finding, sql: str) -> str:
    """Build the prompt asking for a concrete fix for one finding."""
    return (
        "You are a SQL performance and maintainability expert. "
        "A static analyzer flagged this finding on a dbt model's SQL:\n"
        f"- code: {finding.code}\n"
        f"- message: {finding.message}\n\n"
        f"SQL:\n{sql}\n\n"
        "Suggest a concrete, minimal rewrite or configuration change that "
        "addresses the finding. Be brief (2-4 sentences). If no change is "
        "warranted, say so and explain why."
    )


class CallableProvider:
    """Adapt any Callable[[str], str] into an LLMProvider (bring your own model)."""

    def __init__(self, fn: Callable[[str], str]) -> None:
        self._fn = fn

    def suggest(self, prompt: str) -> str:
        return self._fn(prompt)


def enrich_findings(findings: list[Finding], sql: str, provider: LLMProvider) -> list[Suggestion]:
    """One suggestion per finding. Advisory — never affects severity or the gate."""
    return [Suggestion(f.code, provider.suggest(build_prompt(f, sql))) for f in findings]
