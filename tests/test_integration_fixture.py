"""The integration suite's own preflight checks, tested without Docker.

These run in the *default* suite deliberately. The checks in `tests/integration/conftest.py`
exist for the case where the server is wrong, so testing them only from a run against a
correct server would never exercise them at all — and they would be free to rot until the next
time somebody lost a day to a port collision.

Nothing here imports psycopg or connects to anything: `server_mismatches` and `collision_hint`
are pure functions over two strings, which is why they were lifted out of the fixture body.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

_TESTS_DIR = Path(__file__).parent
if str(_TESTS_DIR) not in sys.path:  # pragma: no cover - pytest normally does this itself
    sys.path.insert(0, str(_TESTS_DIR))

from integration.conftest import (  # noqa: E402
    DEFAULT_DSN,
    DEFAULT_PORT,
    EXPECTED_DATABASE,
    collision_hint,
    describe_dsn,
    dsn_secrets,
    server_mismatches,
)

_COMPOSE = _TESTS_DIR / "integration" / "docker-compose.yml"
_PRELOADED = "pg_stat_statements"


def _published_ports() -> list[str]:
    compose = yaml.safe_load(_COMPOSE.read_text(encoding="utf-8"))
    return list(compose["services"]["postgres"]["ports"])


def test_the_fixtures_port_matches_the_port_compose_actually_publishes():
    """Two files naming the same port, and nothing tying them together.

    If they drift, `docker compose up` starts a server the fixture never looks at, and the
    fixture skips with "no Postgres" — a green run that ran none of the 23 live tests. That is
    the same class of invisible failure the port change was made to end, so it is pinned rather
    than left to review.
    """
    assert _published_ports() == [f"{DEFAULT_PORT}:5432"]
    assert f":{DEFAULT_PORT}/" in DEFAULT_DSN


def test_compose_still_publishes_a_port_no_common_postgres_tooling_claims():
    """The port change is the first half of the fix and is otherwise unpinned: reverting it to
    55432 (or any of the other ports Postgres tooling gravitates to) would leave every test
    green while restoring the collision. Also asserts it is outside the ephemeral range on
    Linux and macOS, so no outbound socket can take it before compose binds."""
    assert DEFAULT_PORT not in {5432, 5433, 15432, 54320, 54321, 55432}
    assert 1024 < DEFAULT_PORT < 32768


def test_compose_preloads_pg_stat_statements_with_full_tracking():
    """What the fixture's `shared_preload_libraries` check is checking *for*. Dropping either
    `-c` flag from the compose command leaves a server that accepts connections and then fails
    every workload read."""
    compose = yaml.safe_load(_COMPOSE.read_text(encoding="utf-8"))
    command = compose["services"]["postgres"]["command"]
    assert f"shared_preload_libraries={_PRELOADED}" in command
    assert "pg_stat_statements.track=all" in command
    assert compose["services"]["postgres"]["environment"]["POSTGRES_DB"] == EXPECTED_DATABASE


def test_the_expected_server_reports_no_mismatches():
    """The control. A check that fails on everything is not a check — it would turn every
    correct run into a hard failure."""
    assert server_mismatches(EXPECTED_DATABASE, f"{_PRELOADED},auto_explain") == []
    assert server_mismatches(EXPECTED_DATABASE, _PRELOADED) == []


def test_a_stranger_on_the_port_is_caught_by_its_database_name():
    """The measured failure: an unrelated `postgres:16` held the port, so the suite connected
    to a different database entirely (or was refused by it). Connecting proves nothing about
    *what* answered."""
    [problem] = server_mismatches("some_other_app", _PRELOADED)
    assert "some_other_app" in problem
    assert EXPECTED_DATABASE in problem


def test_a_server_without_pg_stat_statements_preloaded_is_caught():
    """A same-named database on a plain `postgres:16` is the hardest collision to spot: it
    connects, it seeds, and then it fails inside a test with an error about the extension that
    says nothing about the port. `CREATE EXTENSION` cannot fix this — the library has to be
    loaded at server start."""
    [problem] = server_mismatches(EXPECTED_DATABASE, "auto_explain")
    assert "shared_preload_libraries" in problem
    assert "auto_explain" in problem, "name what the server actually reports"


def test_both_mismatches_are_reported_together_rather_than_one_at_a_time():
    """Reporting only the first would send someone to fix the database name and then hit the
    preload failure on the next run."""
    assert len(server_mismatches("postgres", "")) == 2


def test_the_failure_message_names_a_port_collision_as_the_likely_cause():
    """The whole point of the message. The symptom — a password failure, or an unexpected
    schema — points anywhere but at the port, which is why this cost the author and three
    reviewers time. It must also say how to look, not just what happened.
    """
    message = collision_hint(DEFAULT_DSN, server_mismatches("other", ""))
    assert "port collision" in message
    assert str(DEFAULT_PORT) in message
    assert f"docker ps --filter publish={DEFAULT_PORT}" in message
    assert "SQLQUALITY_TEST_DSN" in message
    assert "docker compose -f tests/integration/docker-compose.yml up -d" in message
    assert "neither binds nor fails" in message, (
        "the non-obvious fact that makes this diagnosable: compose does not report the collision"
    )


def test_the_failure_message_carries_no_credential():
    """These messages reach CI logs, and `SQLQUALITY_TEST_DSN` can point anywhere, so the DSN
    must be described rather than echoed — the project's no-credential-in-any-output rule
    applies to a fixture's failure text too."""
    message = collision_hint(DEFAULT_DSN, server_mismatches("other", ""))
    assert "postgres:sqlquality@" not in message
    assert "sqlquality@" not in message
    # Still says where it connected: a message that redacts the location too is unactionable.
    assert f"127.0.0.1:{DEFAULT_PORT}/{EXPECTED_DATABASE}" in message


def test_describe_dsn_keeps_the_location_and_drops_the_credentials():
    assert describe_dsn(DEFAULT_DSN) == f"127.0.0.1:{DEFAULT_PORT}/{EXPECTED_DATABASE}"
    assert describe_dsn("postgresql://user:pw@db.example:6543/analytics") == (
        "db.example:6543/analytics"
    )
    # No port in the DSN means libpq's default, which is what the reader needs told.
    assert describe_dsn("postgresql://u:pw@db.example/analytics") == "db.example:5432/analytics"


def test_describe_dsn_refuses_to_pick_apart_a_keyword_form_dsn():
    """`urlparse` puts a whole keyword-form DSN — password included — in `path`, so anything
    that is not a recognised URI is described generically instead of dissected. Getting this
    wrong prints the password verbatim, which is precisely the failure being guarded."""
    keyword = "host=db.example port=6543 dbname=analytics user=u password=hunter2"
    described = describe_dsn(keyword)
    assert "hunter2" not in described
    assert "password" not in described
    assert described == "the server SQLQUALITY_TEST_DSN points at"


def test_dsn_secrets_yields_the_password_in_both_the_forms_a_driver_may_echo():
    """Same trap `secrets_for` documents: `urlparse` returns the password still
    percent-encoded, while libpq decodes it before authenticating, so an auth-failure message
    carries the *decoded* form and a token of only the encoded one never matches."""
    assert dsn_secrets("postgresql://u:p%40ss@h/db") == ("p%40ss", "p@ss")
    assert dsn_secrets("postgresql://u:plain@h/db") == ("plain",)
    assert dsn_secrets("postgresql://h/db") == ()


def test_a_driver_message_echoing_the_password_is_scrubbed_before_it_is_shown():
    """The measured failure was an authentication failure, whose text is the one place a
    password can surface. Asserted through the same `scrub` the tool uses, over a message shaped
    like libpq's own."""
    from sqlquality.workload.secrets import scrub

    dsn = "postgresql://postgres:s3cretpw@127.0.0.1:27432/sqlquality_test"
    libpq = 'connection failed: password authentication failed for user "postgres" (s3cretpw)'
    scrubbed = scrub(libpq, dsn_secrets(dsn))
    assert "s3cretpw" not in scrubbed
    assert "password authentication failed" in scrubbed
