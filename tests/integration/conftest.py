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

`live_dsn` verifies it reached the server compose started rather than trusting the
connection — a published port that is already taken makes `docker compose up` a silent no-op.
"""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import unquote, urlparse

import pytest

from sqlquality.workload.secrets import scrub

#: Must match the host port `docker-compose.yml` publishes — see the comment there for why it
#: is 27432 and not 55432.
DEFAULT_PORT = 27432
#: The database `docker-compose.yml` creates. Checked, not assumed: see `live_dsn`.
EXPECTED_DATABASE = "sqlquality_test"
DEFAULT_DSN = f"postgresql://postgres:sqlquality@127.0.0.1:{DEFAULT_PORT}/{EXPECTED_DATABASE}"

_PACKAGE_DIR = Path(__file__).parent


def describe_dsn(dsn: str) -> str:
    """`host:port/database` — where we connected, with the credentials left out.

    These messages are printed by CI now, and `SQLQUALITY_TEST_DSN` can point anywhere, so the
    DSN itself must not be echoed: the project's rule that no credential appears in any output
    applies to a fixture's failure text as much as to the tool's.

    A keyword-form DSN (`host=... password=...`) is not a URL, and `urlparse` puts the whole
    string — password included — in `path`, so anything but a recognised URI scheme with a
    hostname is described generically rather than picked apart. Host and port are safe by
    construction; `path` is only ever the database name once a scheme parsed.
    """
    parsed = urlparse(dsn)
    if parsed.scheme not in {"postgres", "postgresql"} or not parsed.hostname:
        return "the server SQLQUALITY_TEST_DSN points at"
    database = (parsed.path or "").lstrip("/") or "(no database in the DSN)"
    return f"{parsed.hostname}:{parsed.port or 5432}/{database}"


def dsn_secrets(dsn: str) -> tuple[str, ...]:
    """The DSN's password, in both the encoded and decoded forms a driver may echo.

    Same reasoning as `sqlquality.workload.secrets.secrets_for`, which cannot be reused
    directly: it takes a `ConnectionParams`, and reconstructing one here to reach one field
    would couple this fixture to a model it has no other use for. The *scrubbing* is reused —
    only the token extraction is local.
    """
    encoded = urlparse(dsn).password
    if not encoded:
        return ()
    decoded = unquote(encoded)
    return (encoded, decoded) if decoded != encoded else (encoded,)


def server_mismatches(database: str, preloaded: str) -> list[str]:
    """Every reason the server we reached is not the one this suite needs, or `[]`.

    A module-level function rather than inline fixture code so it can be exercised without
    Docker: the whole point is what happens when the server is *wrong*, and a check that only
    runs when the server is right is a check nobody ever sees run.

    Both facts are cheap single-round-trip reads and both discriminate a stranger's Postgres
    from this suite's. `shared_preload_libraries` is the one that matters most: without
    `pg_stat_statements` preloaded, `CREATE EXTENSION` still succeeds and every later read of
    its view fails, deep inside a test, with an error that says nothing about the real cause.
    """
    problems = []
    if database != EXPECTED_DATABASE:
        problems.append(f"connected to database {database!r}, expected {EXPECTED_DATABASE!r}")
    if "pg_stat_statements" not in preloaded:
        problems.append(
            "the server has no pg_stat_statements in shared_preload_libraries "
            f"(it reports {preloaded!r}), so the workload tests cannot read query history"
        )
    return problems


def collision_hint(dsn: str, problems: list[str]) -> str:
    """The failure message for a server that answered but is the wrong one.

    It names a port collision explicitly. That is not a guess dressed up as a diagnosis: it is
    the *only* way this state is normally reached, and the reason it cost four people time was
    that the symptom (a password-authentication failure, or an unexpected schema) points
    anywhere but at the port. It also says *how to look*, since the collision is invisible from
    compose's own output.

    The DSN is described, not printed — see `describe_dsn`.
    """
    return (
        f"reached a Postgres at {describe_dsn(dsn)}, but it is not this suite's server:\n"
        + "\n".join(f"  - {p}" for p in problems)
        + f"\nThe likely cause is a port collision: something else already holds {DEFAULT_PORT}, "
        "and `docker compose up` neither binds nor fails in that case, so the suite connects to "
        "whatever is listening.\n"
        f"Check `docker ps --filter publish={DEFAULT_PORT}`, then either free the port or point "
        "SQLQUALITY_TEST_DSN at the right server.\n"
        "Bring the intended one up with: "
        "docker compose -f tests/integration/docker-compose.yml up -d"
    )


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
    """A reachable Postgres that is verifiably *this* suite's server, or skip.

    **Connecting is not the same as connecting to the right server, and the difference is not
    cosmetic.** A published host port that is already taken does not stop `docker compose up`:
    compose reports success, the port keeps belonging to whatever bound it first, and every
    test in this package silently talks to a stranger's database. That happened for the whole
    of this feature's development — an unrelated `postgres:16` container held the old 55432 —
    and it surfaced as a password-authentication failure that three reviewers and the author
    all read as a code bug.

    So two cheap facts are checked before any test runs, and a mismatch is a hard **failure**,
    not a skip: a skip is how the original problem stayed invisible, and by the time this
    fixture runs the caller has explicitly asked for `-m integration`.

    * the database name, which pins that this is the server compose created rather than one
      that merely answers on the port;
    * `shared_preload_libraries`, because `pg_stat_statements` cannot be loaded by `CREATE
      EXTENSION` alone. Without it the extension installs and then every read of its view
      fails deep inside a test, which says nothing about the real cause.

    An unreachable port stays a *skip*: "no Docker" is the documented, supported state for a
    contributor running the default suite. Only a server that answers and is the wrong one
    fails.
    """
    psycopg = pytest.importorskip(
        "psycopg", reason="integration tests need the postgres extra: uv sync --extra postgres"
    )

    dsn = os.environ.get("SQLQUALITY_TEST_DSN", DEFAULT_DSN)
    try:
        with psycopg.connect(dsn, connect_timeout=3) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT current_database(), current_setting('shared_preload_libraries')"
                )
                database, preloaded = cur.fetchone()
    except Exception as exc:  # driver-specific; the message is what matters
        # The driver's own text is scrubbed with the project's own helper before being shown:
        # the most common real connect failure *is* an authentication failure, and this message
        # now reaches CI logs.
        pytest.skip(
            f"no Postgres at {describe_dsn(dsn)}: {scrub(str(exc), dsn_secrets(dsn))}\n"
            "start one with: docker compose -f tests/integration/docker-compose.yml up -d\n"
            f"if that was already running, something else may hold port {DEFAULT_PORT}: compose "
            "neither binds an already-taken port nor fails, so this can equally be a stranger's "
            f"server rejecting our credentials. Check `docker ps --filter publish={DEFAULT_PORT}` "
            f"and `lsof -nP -iTCP:{DEFAULT_PORT} -sTCP:LISTEN`."
        )

    problems = server_mismatches(database, preloaded)
    if problems:
        pytest.fail(collision_hint(dsn, problems))
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
            # deliberately un-analyzed: `public.order_items` therefore carries
            # `reltuples = -1` (Postgres's never-analyzed sentinel) for the whole run,
            # proving the `row_estimate is None` path still proposes (at LOW confidence)
            # rather than the pre-fix behaviour of reading -1 as "tiny table" and
            # suppressing the proposal outright.
            #
            # `autovacuum_enabled = false` is load-bearing, not decoration: a bare "don't
            # ANALYZE it" was measured to be non-deterministic — autovacuum picked up this
            # table and analyzed it mid-run, about 2.5 seconds after seeding, well before
            # any test's assertions ran, which silently turned this into a *never* case
            # rather than a "not yet" case. Disabling autovacuum on this table, set before
            # its INSERT, is what actually keeps `reltuples = -1` for the fixture's whole
            # lifetime. Do not add an ANALYZE (or remove this setting) here.
            cur.execute("DROP TABLE IF EXISTS public.order_items CASCADE")
            cur.execute(
                "CREATE TABLE public.order_items (id bigint, order_id bigint, sku text) "
                "WITH (autovacuum_enabled = false)"
            )
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
            #
            # The predicate is `tenant_id = 3` deliberately, not `status = 'shipped'`:
            # every other seeded statement filters on `status`, and once literals are
            # redacted `status = 'shipped'` and `status = 'pending'` fingerprint
            # identically. A test asserting unwrapping worked would pass just as well if
            # unwrapping were reverted and this row were dropped as noise, because the
            # *other* `status` query already produces the exact same query group. Filtering
            # on a column no other statement filters on makes this query group exist if
            # and only if the DECLARE was actually unwrapped.
            cur.execute(
                "DECLARE live_cur CURSOR WITH HOLD FOR "
                "SELECT id FROM public.orders WHERE tenant_id = 3"
            )
            cur.execute("FETCH 10 FROM live_cur")
            cur.fetchall()
            cur.execute("CLOSE live_cur")
    return live_dsn, schema
