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
