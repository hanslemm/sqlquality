"""Shared psycopg session setup for wire-protocol adapters (Postgres, Redshift).

Both engines speak libpq through psycopg, so opening a session — importing the driver
with an install hint, building conninfo inside the scrubbing envelope, arming a
statement timeout, and handing back a `Querier` — is exactly the same work for both.
Extracted here so there is one place to audit rather than two that can silently drift,
the same reasoning that consolidated credential scrubbing itself into `secrets.py` —
see that module's own docstring.

The one thing that is *not* shared is what "read-only" means to each engine. Postgres's
`SET default_transaction_read_only = on` is expected to succeed unconditionally, so its
refusal aborts the whole connection like any other setup failure. Redshift refuses that
same statement in some configurations, and there a refusal degrades the session instead
of aborting it — see `open_session`'s `read_only_required` parameter and `redshift.py`'s
`connect()` for why silently continuing as though it had succeeded is not an option.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from sqlquality.workload.base import Querier
from sqlquality.workload.secrets import clamp_timeout_ms, scrub

#: The statement establishing read-only intent for a libpq-speaking session. Available
#: unconditionally on Postgres; Redshift refuses it in some configurations — see
#: `open_session`'s `read_only_required` parameter for how each engine treats a refusal.
READ_ONLY_SQL = "SET default_transaction_read_only = on"

#: dbt profiles.yml field names -> libpq connection keywords, shared by every
#: libpq-speaking adapter. One definition, not one per adapter: Postgres's and
#: Redshift's dbt profiles use the identical field names for the keywords that matter
#: here, and two copies of this table is exactly the kind of credential-handling drift
#: this module's own docstring exists to prevent. If an engine ever needs a genuinely
#: different mapping, that is a reason to pass a different table into
#: `translate_libpq_fields`/`dropped_libpq_fields`, not to fork this one.
LIBPQ_FIELD_MAP = {
    "dbname": "dbname",
    "database": "dbname",
    "host": "host",
    "port": "port",
    "user": "user",
    "username": "user",
    "password": "password",
}
#: profiles.yml keys forwarded to libpq unchanged, because the name already *is* the
#: libpq keyword. The TLS group is here for a security reason, not a completeness one: a
#: profile saying `sslmode: verify-full` that silently connects under libpq's default
#: `prefer` performs no certificate verification at all, and the user is never told. For
#: a tool pitched as safe to point at production that is the wrong way to fail.
LIBPQ_PASSTHROUGH_FIELDS = frozenset(
    {"sslmode", "sslcert", "sslkey", "sslrootcert", "connect_timeout"}
)


def import_psycopg(engine_label: str, extra: str) -> Any:
    """Import psycopg, or raise an ImportError naming the extra to install.

    Every caller imports inside its own `connect()`, never at module scope: psycopg is
    an optional extra, and a module-scope import here would break the `no-extras` CI
    job, which never installs it.
    """
    try:
        import psycopg
    except ImportError as exc:
        raise ImportError(
            f"{engine_label} support requires psycopg. "
            f"Install it with: pip install 'sqlquality[{extra}]'"
        ) from exc
    return psycopg


def translate_libpq_fields(
    fields: Mapping[str, str], field_map: Mapping[str, str], passthrough: frozenset[str]
) -> dict[str, str]:
    """Translate profiles.yml keys to libpq keywords, dropping anything unrecognized."""
    translated = {field_map[k]: v for k, v in fields.items() if k in field_map}
    translated.update({k: v for k, v in fields.items() if k in passthrough})
    return translated


def dropped_libpq_fields(
    fields: Mapping[str, str], field_map: Mapping[str, str], passthrough: frozenset[str]
) -> tuple[str, ...]:
    """profiles.yml keys this adapter cannot forward, by name.

    Names only, never values: one of them could be a secret (`sslpassword`), and this
    text goes to stderr and from there into CI logs.
    """
    return tuple(sorted(k for k in fields if k not in field_map and k not in passthrough))


def open_session(
    *,
    psycopg: Any,
    conninfo_factory: Callable[[], str],
    secrets: Sequence[str],
    timeout_s: int,
    min_timeout_s: int,
    max_timeout_s: int,
    read_only_sql: str,
    read_only_required: bool,
) -> tuple[Querier, str | None]:
    """Connect, arm read-only intent and a statement timeout, and hand back a Querier.

    ``min_timeout_s``/``max_timeout_s`` are parameters rather than a module-level import
    of `workload.base`'s constants, so each caller passes (and therefore keeps using,
    rather than merely importing) the one shared pair `base.py` defines — the same
    "defined once so two layers cannot drift apart" reasoning `base.py` gives for their
    existence at all.

    ``conninfo_factory`` is called *inside* the scrubbing envelope deliberately:
    building a conninfo from caller-supplied fields (``psycopg.conninfo.make_conninfo``)
    can itself raise, and that message can quote the offending value — for a `password`
    keyword, the password. Calling it here rather than before this function is what
    keeps that failure covered.

    ``read_only_required`` decides what a failed ``read_only_sql`` means:

    - Required (Postgres): the statement always succeeds, so a failure is
      indistinguishable from any other setup failure and aborts the connection like one.
    - Not required (Redshift): the statement is refused in some configurations, so its
      failure is caught right here, scrubbed, and returned as a degradation message
      instead of raised. The connection still proceeds — the adapter calling this only
      ever issues SELECT statements regardless of whether this belt-and-braces guard
      could be armed.

    On any other failure, ``ConnectionError`` is raised *after* this function's own
    `except` clause has already exited — not from inside it — so the driver's original
    exception can never survive as `__context__`. That severance is deliberate: an
    unscrubbed driver exception reachable via `__context__` would defeat the whole
    scrubbing exercise the moment anything printed a traceback.
    """
    failure: str | None = None
    degradation: str | None = None
    connection: Any = None
    try:
        conninfo = conninfo_factory()
        connection = psycopg.connect(conninfo, autocommit=True)
        with connection.cursor() as cursor:
            if read_only_required:
                # Belt and braces: the session cannot write even if a statement tried
                # to. A failure here is a setup failure like any other and is handled
                # by the broad `except` below.
                cursor.execute(read_only_sql)
            else:
                try:
                    cursor.execute(read_only_sql)
                except Exception as exc:  # driver-specific; only the message matters
                    degradation = scrub(
                        "the session could not be proven read-only: the server refused "
                        f"{read_only_sql!r} ({exc}). Belt-and-braces read-only could not "
                        "be applied. This does not mean the session might write — this "
                        "adapter only ever issues the SELECT statements in its own "
                        "introspection SQL — but this extra safeguard could not be armed.",
                        secrets,
                    )
            # set_config() rather than `SET`, because bind parameters are not accepted
            # in a SET statement and string-building one with a caller-controlled value
            # is the wrong habit to establish in the one place we talk to a database.
            cursor.execute(
                "SELECT set_config('statement_timeout', %s, false)",
                (f"{clamp_timeout_ms(timeout_s, minimum=min_timeout_s, maximum=max_timeout_s)}ms",),
            )
    except Exception as exc:
        failure = scrub(str(exc), secrets)
    if failure is not None:
        # Raised after the handler, and scrubbed: see this function's own docstring —
        # leaving the handler is the only way to keep the original exception out of
        # __context__. No "Could not connect" prefix here; the CLI adds it.
        raise ConnectionError(failure)

    def query(sql: str, bind: tuple[object, ...]) -> list[tuple[object, ...]]:
        with connection.cursor() as cur:
            cur.execute(sql, bind)
            return list(cur.fetchall())

    return query, degradation
