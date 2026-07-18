from sqlquality.llm import (
    CallableProvider,
    LLMProvider,
    Suggestion,
    build_prompt,
    enrich_findings,
)
from sqlquality.models import Finding, Severity

FINDING = Finding(
    "SQ001", "SELECT * projects an unknown/wide column set.", 0, Severity.WARNING, False
)


def test_build_prompt_includes_finding_and_sql():
    prompt = build_prompt(FINDING, "select * from users")
    assert "SQ001" in prompt
    assert "select * from users" in prompt


def test_callable_provider_is_an_llmprovider():
    provider = CallableProvider(lambda p: "ok")
    assert isinstance(provider, LLMProvider)
    assert provider.suggest("anything") == "ok"


def test_enrich_findings_one_suggestion_per_finding():
    calls = []

    def fake(prompt: str) -> str:
        calls.append(prompt)
        return "list the columns explicitly"

    findings = [
        FINDING,
        Finding("SQ002", "Cartesian join.", 0, Severity.WARNING, False),
    ]
    suggestions = enrich_findings(findings, "select * from a, b", CallableProvider(fake))
    assert [s.code for s in suggestions] == ["SQ001", "SQ002"]
    assert all(isinstance(s, Suggestion) for s in suggestions)
    assert suggestions[0].text == "list the columns explicitly"
    assert len(calls) == 2  # one prompt per finding


def test_enrich_findings_empty():
    assert enrich_findings([], "select 1", CallableProvider(lambda p: "x")) == []
