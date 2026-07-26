import json

from typer.testing import CliRunner

from sqlquality.cli import app

runner = CliRunner()


def test_dry_run_prints_statements_and_never_connects(monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("--dry-run must not connect")

    monkeypatch.setattr("sqlquality.workload.postgres.PostgresWorkloadAdapter.connect", explode)
    result = runner.invoke(app, ["advise", "--engine", "postgres", "--dry-run"])
    assert result.exit_code == 0
    assert "pg_stat_statements" in result.stdout


def test_dry_run_needs_no_credentials():
    result = runner.invoke(app, ["advise", "--engine", "postgres", "--dry-run"])
    assert result.exit_code == 0


def test_missing_credentials_exit_2():
    result = runner.invoke(app, ["advise"])
    assert result.exit_code == 2
    assert "--dsn" in result.output


def test_unsupported_engine_exit_2():
    result = runner.invoke(app, ["advise", "--dsn", "postgresql://u@h/db", "--engine", "duckdb"])
    assert result.exit_code == 2


def test_bad_dsn_scheme_exit_2():
    result = runner.invoke(app, ["advise", "--dsn", "mysql://u@h/db"])
    assert result.exit_code == 2


def test_malformed_since_exit_2():
    result = runner.invoke(app, ["advise", "--dsn", "postgresql://u@h/db", "--since", "banana"])
    assert result.exit_code == 2


def test_out_of_range_timeout_exit_2_before_connecting(monkeypatch):
    """Reject rather than silently clamp. 0 means "no limit" to Postgres — the opposite."""

    def explode(*args, **kwargs):
        raise AssertionError("must not connect with an invalid --timeout")

    monkeypatch.setattr("sqlquality.workload.postgres.PostgresWorkloadAdapter.connect", explode)
    for bad in ("0", "-5", "99999"):
        result = runner.invoke(app, ["advise", "--dsn", "postgresql://u@h/db", "--timeout", bad])
        assert result.exit_code == 2, f"--timeout {bad} should be rejected"
        assert "between 1 and 3600" in result.output


def test_low_coverage_warns_on_stderr(monkeypatch):
    """ "No proposals" must be distinguishable from "I understood almost none of this"."""
    _stub_adapter(
        monkeypatch,
        {
            "pg_stat_statements": [
                ("select id from mystery_table where a = $1", 100, 5000.0, 10),
                ("select id from other_mystery where b = $1", 100, 5000.0, 10),
                ("select id from orders where status = $1", 1, 10.0, 1),
            ],
            "pg_stat_database": [("2026-07-01",)],
            "information_schema.columns": WIDE_COLUMNS,
            "pg_total_relation_size": [("orders", 5_000_000, 10**8)],
            "pg_stats": [("orders", "status", 5000.0)],
            "pg_index": [],
        },
    )
    result = runner.invoke(app, ["advise", "--dsn", "postgresql://u@h/db"])
    assert result.exit_code == 0
    assert "low coverage" in result.output
    assert "min-cost-share" in result.output


def test_good_coverage_does_not_warn(monkeypatch):
    _stub_adapter(
        monkeypatch,
        {
            "pg_stat_statements": [
                ("select id from orders where status = $1 and created_at > $2", 100, 5000.0, 10),
            ],
            "pg_stat_database": [("2026-07-01",)],
            "information_schema.columns": WIDE_COLUMNS,
            "pg_total_relation_size": [("orders", 5_000_000, 10**8)],
            "pg_stats": [("orders", "status", 5000.0)],
            "pg_index": [],
        },
    )
    result = runner.invoke(app, ["advise", "--dsn", "postgresql://u@h/db"])
    assert result.exit_code == 0
    assert "low coverage" not in result.output


def _stub_adapter(monkeypatch, rows):
    """Replace connect() with an injected fake querier."""
    from sqlquality.workload.postgres import PostgresWorkloadAdapter

    def fake_connect(self, params, timeout_s):
        self.schemas = ("public",)

        def query(sql, bind):
            for marker, result in rows.items():
                if marker in sql:
                    return result
            return []

        self._query = query

    monkeypatch.setattr(PostgresWorkloadAdapter, "connect", fake_connect)


WIDE_COLUMNS = [
    ("orders", "id", "integer"),
    ("orders", "status", "text"),
    ("orders", "created_at", "timestamp"),
]


def test_successful_run_exits_0_and_emits_json(monkeypatch):
    _stub_adapter(
        monkeypatch,
        {
            "pg_stat_statements": [
                ("select id from orders where status = $1 and created_at > $2", 100, 5000.0, 10),
            ],
            "pg_stat_database": [("2026-07-01",)],
            "information_schema.columns": WIDE_COLUMNS,
            "pg_total_relation_size": [("orders", 5_000_000, 10**8)],
            "pg_stats": [("orders", "status", 5000.0)],
            "pg_index": [],
        },
    )
    result = runner.invoke(app, ["advise", "--dsn", "postgresql://u@h/db", "--json"])
    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["engine"] == "postgres"
    assert payload["redacted"] is True
    assert any(p["code"] == "ADV001" for p in payload["proposals"])


def test_empty_workload_exits_0(monkeypatch):
    _stub_adapter(monkeypatch, {"pg_stat_statements": [], "pg_stat_database": [("2026-07-01",)]})
    result = runner.invoke(app, ["advise", "--dsn", "postgresql://u@h/db", "--json"])
    assert result.exit_code == 0
    assert json.loads(result.stdout)["proposals"] == []


def test_ddl_and_markdown_files_are_written(monkeypatch, tmp_path):
    _stub_adapter(
        monkeypatch,
        {
            "pg_stat_statements": [
                ("select id from orders where status = $1 and created_at > $2", 100, 5000.0, 10),
            ],
            "pg_stat_database": [("2026-07-01",)],
            "information_schema.columns": WIDE_COLUMNS,
            "pg_total_relation_size": [("orders", 5_000_000, 10**8)],
            "pg_stats": [("orders", "status", 5000.0)],
            "pg_index": [],
        },
    )
    ddl = tmp_path / "out.sql"
    md = tmp_path / "out.md"
    result = runner.invoke(
        app, ["advise", "--dsn", "postgresql://u@h/db", "--ddl", str(ddl), "--markdown", str(md)]
    )
    assert result.exit_code == 0
    assert "CREATE INDEX" in ddl.read_text()
    assert "sqlquality advise" in md.read_text()
    assert "REVIEW BEFORE RUNNING" in ddl.read_text()


def test_connection_failure_exits_2(monkeypatch):
    from sqlquality.workload.postgres import PostgresWorkloadAdapter

    def boom(self, params, timeout_s):
        raise RuntimeError("could not connect to server")

    monkeypatch.setattr(PostgresWorkloadAdapter, "connect", boom)
    result = runner.invoke(app, ["advise", "--dsn", "postgresql://u@h/db"])
    assert result.exit_code == 2
    assert "could not connect" in result.output


def test_missing_driver_exits_2_with_install_hint(monkeypatch):
    from sqlquality.workload.postgres import PostgresWorkloadAdapter

    def no_driver(self, params, timeout_s):
        raise ImportError("Install it with: pip install 'sqlquality[postgres]'")

    monkeypatch.setattr(PostgresWorkloadAdapter, "connect", no_driver)
    result = runner.invoke(app, ["advise", "--dsn", "postgresql://u@h/db"])
    assert result.exit_code == 2
    assert "sqlquality[postgres]" in result.output
