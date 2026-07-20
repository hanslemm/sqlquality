import json

from typer.testing import CliRunner

from sqlquality import cli
from sqlquality.cli import app
from sqlquality.llm import CallableProvider

runner = CliRunner()


def test_perf_suggest_adds_suggestions(tmp_path, monkeypatch):
    monkeypatch.setattr(
        cli, "resolve_provider", lambda: CallableProvider(lambda p: "list columns explicitly")
    )
    f = tmp_path / "m.sql"
    f.write_text("select * from a, b")
    result = runner.invoke(app, ["perf", str(f), "--suggest", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    codes = {s["code"] for s in payload["suggestions"]}
    assert "SQ001" in codes and "SQ002" in codes
    assert payload["suggestions"][0]["text"] == "list columns explicitly"


def test_perf_suggest_no_provider_is_advisory(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "resolve_provider", lambda: None)
    f = tmp_path / "m.sql"
    f.write_text("select * from t")
    result = runner.invoke(app, ["perf", str(f), "--suggest", "--json"])
    assert result.exit_code == 0  # advisory: still succeeds
    payload = json.loads(result.stdout)
    assert payload["suggestions"] == []


def test_perf_without_suggest_has_empty_suggestions(tmp_path):
    f = tmp_path / "m.sql"
    f.write_text("select * from t")
    result = runner.invoke(app, ["perf", str(f), "--json"])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["suggestions"] == []


def test_perf_suggest_provider_failure_is_advisory(tmp_path, monkeypatch):
    class _Boom:
        def suggest(self, prompt: str) -> str:
            raise RuntimeError("rate limited")

    monkeypatch.setattr(cli, "resolve_provider", lambda: _Boom())
    f = tmp_path / "m.sql"
    f.write_text("select * from t")  # SQ001 WARNING only -> exit 0
    result = runner.invoke(app, ["perf", str(f), "--suggest", "--json"])
    assert result.exit_code == 0  # provider failure must NOT change the exit code
    payload = json.loads(result.stdout)
    assert payload["suggestions"] == []
    assert any(x["code"] == "SQ001" for x in payload["findings"])  # report preserved


def test_perf_suggest_provider_construction_failure_is_advisory(tmp_path, monkeypatch):
    # SQLQUALITY_LLM is set but the provider can't even be constructed
    # (missing anthropic package or credentials) -> resolve_provider() raises.
    def _explode() -> None:
        raise RuntimeError("The 'anthropic' package is required for AnthropicProvider.")

    monkeypatch.setattr(cli, "resolve_provider", _explode)
    f = tmp_path / "m.sql"
    f.write_text("select * from t")  # SQ001 WARNING only -> exit 0
    result = runner.invoke(app, ["perf", str(f), "--suggest", "--json"])
    assert result.exit_code == 0  # construction failure must NOT change the exit code
    assert result.exception is None  # no traceback leaked
    payload = json.loads(result.stdout)
    assert payload["suggestions"] == []
    assert any(x["code"] == "SQ001" for x in payload["findings"])  # report preserved
    assert "LLM suggestions unavailable" in result.stderr  # one friendly note
