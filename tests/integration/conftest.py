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

            # Multi-schema keying: `orders` in both `public` and `staging`, with
            # deliberately different row counts. This is the exact collision the old
            # bare-name-keyed `_validate_schemas` refused to allow, and the only way an
            # aliasing regression in `Relation`-keyed catalog facts can be caught.
            cur.execute("DROP TABLE IF EXISTS public.orders CASCADE")
            cur.execute(
                "CREATE TABLE public.orders (id bigint, status text, tenant_id bigint, day date)"
            )
            cur.execute(
                "INSERT INTO public.orders "
                "SELECT g, 'shipped', g % 7, current_date FROM generate_series(1, 20000) g"
            )
            cur.execute("DROP SCHEMA IF EXISTS staging CASCADE")
            cur.execute("CREATE SCHEMA staging")
            cur.execute(
                "CREATE TABLE staging.orders (id bigint, status text, tenant_id bigint, day date)"
            )
            cur.execute(
                "INSERT INTO staging.orders "
                "SELECT g, 'draft', g % 7, current_date FROM generate_series(1, 50000) g"
            )
            cur.execute("ANALYZE public.orders")
            cur.execute("ANALYZE staging.orders")

            # An unindexed join key (ADV007) and, on the other side of it, a table left
            # deliberately un-analyzed: `public.order_items.order_id` therefore carries
            # `reltuples = -1` (Postgres's never-analyzed sentinel) at fetch time, proving
            # the `row_estimate is None` path still proposes (at LOW confidence) rather
            # than the pre-fix behaviour of reading -1 as "tiny table" and suppressing the
            # proposal outright. Deliberately NOT analyzed — do not add an ANALYZE here.
            cur.execute("DROP TABLE IF EXISTS public.order_items CASCADE")
            cur.execute("CREATE TABLE public.order_items (id bigint, order_id bigint, sku text)")
            cur.execute(
                "INSERT INTO public.order_items "
                "SELECT g, (g % 20000) + 1, 'sku' || g FROM generate_series(1, 20000) g"
            )

            # An index nothing in the workload below ever touches: not on `status` (the
            # only equality predicate), not part of the GROUP BY -- so its scan count stays
            # genuinely zero, giving ADV002 a real DROP INDEX candidate in `staging` to pair
            # against the CREATE INDEX candidate `advise` proposes in `public`.
            cur.execute("CREATE INDEX idx_unused_staging_id ON staging.orders (id)")

            cur.execute("SELECT pg_stat_statements_reset()")
            # Real workload for the history statement to find.
            for _ in range(3):
                cur.execute(
                    f"SELECT id FROM {schema}.orders WHERE status = %s "
                    "AND created_at > now() - interval '1 day'",
                    ("paid",),
                )
                cur.fetchall()

            # A schema-qualified filter on each side of the public/staging collision, so
            # both relations get their own usage and their own cost share.
            for _ in range(5):
                cur.execute("SELECT id FROM public.orders WHERE status = 'shipped'")
                cur.fetchall()
                cur.execute("SELECT id FROM staging.orders WHERE status = 'draft'")
                cur.fetchall()
            # A join key with no index leading with it (ADV007).
            for _ in range(5):
                cur.execute(
                    "SELECT o.id FROM public.orders o "
                    "JOIN public.order_items i ON i.order_id = o.id"
                )
                cur.fetchall()
            # A hot GROUP BY with no covering index (ADV008).
            for _ in range(5):
                cur.execute(
                    "SELECT tenant_id, day, count(*) FROM staging.orders GROUP BY tenant_id, day"
                )
                cur.fetchall()
            # A server-side cursor read. `WITH HOLD` is what keeps the cursor alive past
            # this connection's per-statement autocommit boundary -- without it the cursor
            # is dropped the instant the DECLARE's own implicit transaction commits, and
            # FETCH fails with "cursor does not exist", not merely a filtered read.
            cur.execute(
                "DECLARE live_cur CURSOR WITH HOLD FOR "
                "SELECT id FROM public.orders WHERE status = 'pending'"
            )
            cur.execute("FETCH 10 FROM live_cur")
            cur.fetchall()
            cur.execute("CLOSE live_cur")
    return live_dsn, schema
