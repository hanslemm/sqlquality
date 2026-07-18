"""Optional, provider-agnostic LLM layer for enriching findings (advisory only)."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

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


class AnthropicProvider:
    """LLM provider backed by the Anthropic Messages API (optional extra)."""

    def __init__(self, model: str | None = None, client: object | None = None) -> None:
        if client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - exercised only without the extra
                raise RuntimeError(
                    "The 'anthropic' package is required for AnthropicProvider. "
                    "Install it with: pip install 'sqlquality[llm]'"
                ) from exc
            client = anthropic.Anthropic()
        self._client: Any = client
        self._model = model or os.environ.get("SQLQUALITY_LLM_MODEL", "claude-opus-4-8")

    def suggest(self, prompt: str) -> str:
        message = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(
            block.text for block in message.content if getattr(block, "type", None) == "text"
        )


def resolve_provider() -> LLMProvider | None:
    """Return a provider if configured via SQLQUALITY_LLM, else None (off by default)."""
    if os.environ.get("SQLQUALITY_LLM", "").strip().lower() not in {"anthropic", "1", "true"}:
        return None
    return AnthropicProvider()
