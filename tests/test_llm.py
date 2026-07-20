from sqlquality.llm import (
    MAX_PROMPT_SQL_CHARS,
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


def test_build_prompt_truncates_long_sql():
    # 'Z' does not appear in the static prompt template, so counting it isolates
    # exactly how much of the embedded SQL survived.
    long_sql = "Z" * (MAX_PROMPT_SQL_CHARS + 5_000)
    prompt = build_prompt(FINDING, long_sql)
    assert "... [truncated]" in prompt
    assert prompt.count("Z") == MAX_PROMPT_SQL_CHARS  # SQL bounded to the cap


def test_build_prompt_does_not_truncate_short_sql():
    prompt = build_prompt(FINDING, "select 1")
    assert "... [truncated]" not in prompt


def test_build_prompt_exact_boundary_not_truncated():
    # SQL exactly at the cap must pass through untouched (strictly-greater cut).
    sql = "Z" * MAX_PROMPT_SQL_CHARS
    prompt = build_prompt(FINDING, sql)
    assert "... [truncated]" not in prompt
    assert prompt.count("Z") == MAX_PROMPT_SQL_CHARS


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


def test_enrich_findings_skips_failed_calls():
    findings = [
        Finding("SQ001", "first.", 0, Severity.WARNING, False),
        Finding("SQ002", "second.", 0, Severity.WARNING, False),
        Finding("SQ003", "third.", 0, Severity.WARNING, False),
    ]

    def fake(prompt: str) -> str:
        if "SQ002" in prompt:  # the finding code is embedded in the prompt
            raise RuntimeError("boom on the second finding")
        return "ok"

    suggestions = enrich_findings(findings, "select 1", CallableProvider(fake))
    # a single failed call is skipped; the others survive
    assert [s.code for s in suggestions] == ["SQ001", "SQ003"]
