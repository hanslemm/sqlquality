"""Credential handling shared by every workload adapter.

This lives outside any one adapter deliberately. Scrubbing took three fix rounds to get
right on the Postgres adapter — the driver's exception text quoted the offending value, then
`from None` turned out to suppress only the traceback while leaving `__context__` reachable,
then a percent-encoded DSN password slipped past because ``urlparse`` returns it still
encoded. An adapter that cannot see these helpers will re-derive that sequence badly.
"""

from __future__ import annotations

from collections.abc import Iterable
from urllib.parse import unquote, urlparse

from sqlquality.models import ConnectionParams

#: profiles.yml keys whose values must never appear in any message we emit.
SECRET_FIELDS = frozenset({"password", "pass"})

#: A secret shorter than this cannot be redacted by substring replacement without
#: destroying the message — a one-character password would blank every occurrence of that
#: letter. When one actually appears, the driver's text is withheld rather than mangled.
MIN_SCRUBBABLE_SECRET = 4
WITHHELD = "(driver message withheld: it contained a value too short to redact safely)"


def secrets_for(params: ConnectionParams) -> tuple[str, ...]:
    """Every value we know to be secret for this connection.

    A DSN is added *and* its password extracted separately. The whole-DSN token only helps
    if the driver echoes the connection string back verbatim, which real libpq errors do
    not do — they report the offending value on its own. Without the extracted password,
    DSN-based connections would have no effective protection at all.

    The password is added in **both** its percent-encoded and decoded forms.
    ``urlparse().password`` returns it still encoded, but libpq decodes a URI DSN before
    authenticating, so the value a real auth-failure message carries is the decoded one:
    for ``postgresql://u:p%40ss@h/db`` the driver reports ``p@ss`` while urlparse yields
    ``p%40ss``, and a token of only the encoded form never matches. Any password containing
    ``@``, ``:``, ``/``, ``%`` or a space hits this. The encoded form is kept too, since a
    URI-parse error can echo the raw string back instead.
    """
    secrets = tuple(value for key, value in params.fields.items() if key in SECRET_FIELDS and value)
    if params.dsn:
        secrets += (params.dsn,)
        encoded = urlparse(params.dsn).password
        if encoded:
            secrets += (encoded,)
            decoded = unquote(encoded)
            if decoded != encoded:
                secrets += (decoded,)
    return secrets


def scrub(text: str, secrets: Iterable[str]) -> str:
    """Replace any known secret occurring in ``text`` with a redaction marker.

    Defence in depth for driver exceptions. libpq is not believed to echo a password, but
    the auth-failure path — the most common real connect failure — cannot be exercised
    without a live server, and we hold the secret anyway, so its absence can be guaranteed
    instead of trusted.
    """
    present = [secret for secret in secrets if secret and secret in text]
    if any(len(secret) < MIN_SCRUBBABLE_SECRET for secret in present):
        return WITHHELD
    scrubbed = text
    for secret in present:
        scrubbed = scrubbed.replace(secret, "***")
    return scrubbed


def clamp_timeout_ms(timeout_s: int, *, minimum: int, maximum: int) -> int:
    """Statement timeout in milliseconds, clamped into ``[minimum, maximum]`` seconds.

    Bounds are parameters rather than module constants: the CLI owns the user-facing range
    and rejects out-of-range input, so a second copy of the numbers here could drift out of
    step with the message the user was shown.
    """
    return max(minimum, min(int(timeout_s), maximum)) * 1000
