"""Opt-in live-Postgres fixtures.

Every test in this package is marked `integration` and deselected by default (see
pyproject.toml's addopts), so a contributor without Docker sees a clean `uv run pytest`.

Nothing here may fail, skip, or import psycopg at *module* scope. Conftest import happens
during collection, before `addopts` deselects anything, so a module-scope
`pytest.importorskip("psycopg")` turned the whole package into one collection-level skip for
anyone who ran the plain `uv sync` CONTRIBUTING.md documents — psycopg is an optional extra,
not part of the `dev` group. The result was `442 passed, 1 skipped` where the promise above
says `deselected`. The guard belongs in `live_dsn`, which is reached only once a test has
already been selected.

Bring the server up with:
    docker compose -f tests/integration/docker-compose.yml up -d
    uv run pytest -m integration
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

DEFAULT_DSN = "postgresql://postgres:sqlquality@127.0.0.1:55432/sqlquality_test"

_PACKAGE_DIR = Path(__file__).parent


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark every test collected under this package as `integration`.

    A module-level `pytestmark` only applies to the module that defines it — pytest does
    not propagate a conftest.py's `pytestmark` to sibling test files in the same
    directory. This hook is what actually makes "every test in this package is marked
    integration" true, including for test files added here later.
    """
    for item in items:
        if _PACKAGE_DIR in item.path.parents:
            item.add_marker(pytest.mark.integration)


@pytest.fixture(scope="session")
def live_dsn() -> str:
    """A reachable Postgres, or skip with an actionable message."""
    psycopg = pytest.importorskip(
        "psycopg", reason="integration tests need the postgres extra: uv sync --extra postgres"
    )

    dsn = os.environ.get("SQLQUALITY_TEST_DSN", DEFAULT_DSN)
    try:
        with psycopg.connect(dsn, connect_timeout=3) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
    except Exception as exc:  # driver-specific; the message is what matters
        pytest.skip(
            f"no Postgres at {dsn}: {exc}\n"
            "start one with: docker compose -f tests/integration/docker-compose.yml up -d"
        )
    return dsn


@pytest.fixture(scope="session")
def seeded(live_dsn: str) -> tuple[str, str]:
    """A schema with the index shapes the catalog query has to survive, plus real workload.

    Deliberately includes a partial and an expression index: those are exactly the rows the
    shipped statement discarded, and the only way to know the fix works is to read them back
    out of a real catalog.
    """
    import psycopg

    schema = "advise_it"
    with psycopg.connect(live_dsn, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS pg_stat_statements")
            cur.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
            cur.execute(f"CREATE SCHEMA {schema}")
            cur.execute(
                f"""CREATE TABLE {schema}.orders (
                       id bigserial PRIMARY KEY,
                       status text NOT NULL,
                       note text,
                       shipped_at timestamptz,
                       created_at timestamptz NOT NULL DEFAULT now())"""
            )
            cur.execute(f"CREATE INDEX idx_plain ON {schema}.orders (status, created_at)")
            cur.execute(
                f"CREATE INDEX idx_open ON {schema}.orders (status) WHERE shipped_at IS NULL"
            )
            cur.execute(f"CREATE INDEX idx_lower_note ON {schema}.orders (lower(note))")
            cur.execute(
                f"INSERT INTO {schema}.orders (status, note) "
                "SELECT 'paid', 'n' || g FROM generate_series(1, 500) g"
            )
            # A freshly loaded table reports reltuples = -1 and has no pg_stats rows until
            # analyzed — autovacuum gets to it eventually, but not necessarily before this
            # fixture's caller queries it. ANALYZE makes the row estimate and NDV
            # deterministic instead of racing autovacuum.
            cur.execute(f"ANALYZE {schema}.orders")
            cur.execute("SELECT pg_stat_statements_reset()")
            # Real workload for the history statement to find.
            for _ in range(3):
                cur.execute(
                    f"SELECT id FROM {schema}.orders WHERE status = %s "
                    "AND created_at > now() - interval '1 day'",
                    ("paid",),
                )
                cur.fetchall()
    return live_dsn, schema
