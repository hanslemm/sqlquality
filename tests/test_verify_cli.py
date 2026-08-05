"""Tests for `sqlquality verify` — the command, its output, and its refusals.

**Every artifact here is produced by the real `advise` path**, through
`runner.invoke(app, ["advise", "--json", ...])` over a stubbed querier, and written to disk
exactly as `advise --json > before.json` would write it. That is deliberate and it is the
single most important property of this file: Task 6's headline test asserted against a
hand-built payload pairing `"proposals": []` with a populated `"query_groups"`, which the
then-current `_query_groups_payload` — scoped to the digests proposals cited — could not
produce, and it passed while the real path was broken. A fixture that `advise` cannot emit
proves nothing about `verify`, which only ever reads what `advise` wrote.

**That particular shape is emittable now** (Task 6's fix made `query_groups` cover the whole
workload, so a resolved proposal leaves an empty `proposals` beside a populated
`query_groups` — see `tests/fixtures/verify/improved.after.json`, generated from a real
server). It is named here as the mechanism, not as a shape to avoid: the only way to know a
fixture is emittable is to have emitted it, which is why every artifact below comes from the
real command.

The two places a payload *is* edited after the fact are the pre-0.4.0 refusals, and they
edit by *removing* keys from a real artifact rather than by inventing a shape: an artifact
from an older sqlquality is exactly a current one minus the keys that version did not have,
and no other way to obtain one exists (0.3.0 cannot be run from this test suite).

**Each refusal test names the cause in the message it asserts, not merely the exit code.**
Several refusals can fire on one malformed pair, so exit 2 alone would let a test pass while
proving nothing about the refusal it is named after — the failure mode a mandated gate test
in Task 5 had. Where a pair could plausibly trip a second refusal, the assertions pin the
wording only the intended refusal produces, and the docstring says which.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from sqlquality.cli import app
from sqlquality.report import _WINDOW_RELATION_CAVEAT
from sqlquality.verify import WindowRelation

runner = CliRunner()

#: Any ANSI escape sequence. rich splits a styled token like `--json` across several escapes,
#: so a raw-substring assertion on `--help` output fails while the flag is perfectly present.
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    """`text` with rich's styling escapes removed."""
    return _ANSI.sub("", text)


# --- artifacts, produced through the real `advise` path -----------------------------------

#: `public.orders` as `information_schema.columns` returns it.
_COLUMNS = [
    ("public", "orders", "id", "integer"),
    ("public", "orders", "status", "text"),
    ("public", "orders", "created_at", "timestamp"),
]

#: One `pg_index` row-set: a btree index on `status`, in the 13-column shape the real
#: `CAP_INDEXES` statement selects (see `PostgresWorkloadAdapter.SQL[CAP_INDEXES]`).
_INDEX_ON_STATUS = [
    (
        "public",
        "orders",
        "idx_orders_status",
        "status",
        1,
        False,
        False,
        42,
        8192,
        False,
        None,
        False,
        "CREATE INDEX idx_orders_status ON public.orders USING btree (status)",
    )
]

#: The hot statement whose ADV001 proposal every headline test below follows.
_HOT_STATEMENT = "select id from orders where status = $1"


def _pg_rows(
    *,
    statement: str = _HOT_STATEMENT,
    calls: int = 100,
    total_ms: float = 5000.0,
    indexed: bool = False,
    stats_reset: str | None = "2026-07-01",
) -> dict[str, object]:
    """Canned rows for a full Postgres `advise` run over one relation.

    `indexed` is the whole before/after axis: `False` is the run that proposes ADV001,
    `True` the run where the index exists, so the rule stops firing and `physical_state`
    records the index — the transition `verify` reads as `applied`.
    """
    return {
        "pg_stat_statements": [(statement, calls, total_ms, 10)],
        "pg_stat_database": [(stats_reset,)] if stats_reset is not None else [],
        "information_schema.columns": _COLUMNS,
        "pg_total_relation_size": [("public", "orders", 5_000_000, 10**8)],
        "pg_stats": [("public", "orders", "status", 5000.0)],
        "pg_index": _INDEX_ON_STATUS if indexed else [],
    }


def _advise_stdout(
    monkeypatch, rows: dict[str, object], *extra: str, engine: str = "postgres"
) -> str:
    """The exact stdout of a real `advise --json` run over `rows`.

    Marker matching on the statement text, exactly like `tests/test_advise_cli.py`'s
    `_stub_adapter`: only `connect()` is replaced, so every layer from `fetch_workload`
    through `advise_payload` is the production one.

    A row value that is an `Exception` is *raised* instead of returned, which is how a denied
    grant is simulated: `PostgresWorkloadAdapter._run` swallows it into `degraded` and
    returns `[]`, the same as a real permission failure.
    """
    from sqlquality.workload.postgres import PostgresWorkloadAdapter

    def fake_connect(self, params, timeout_s):
        def query(sql, bind):
            for marker, result in rows.items():
                if marker in sql:
                    if isinstance(result, Exception):
                        raise result
                    return result
            return []

        self._query = query

    monkeypatch.setattr(PostgresWorkloadAdapter, "connect", fake_connect)
    result = runner.invoke(
        app, ["advise", "--engine", engine, "--dsn", "postgresql://u@h/db", "--json", *extra]
    )
    assert result.exit_code == 0, result.output
    return result.stdout


def _artifact(tmp_path: Path, name: str, text: str) -> Path:
    """An artifact on disk, byte-for-byte as `advise --json > name` would write it."""
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _pg_artifact(tmp_path: Path, monkeypatch, name: str, *extra: str, **kwargs) -> Path:
    return _artifact(tmp_path, name, _advise_stdout(monkeypatch, _pg_rows(**kwargs), *extra))


def _headline_pair(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    """The pair every non-refusal test starts from: ADV001 proposed, the index created, the
    same query group four-and-a-half times cheaper per call.

    Both runs report the same `stats_reset_at` and neither passes `--since`, so the window
    classifies `NESTED` — the common Postgres path, and the one whose caveat matters most.
    """
    before = _pg_artifact(tmp_path, monkeypatch, "before.json", calls=100, total_ms=5000.0)
    after = _pg_artifact(
        tmp_path, monkeypatch, "after.json", calls=1000, total_ms=2000.0, indexed=True
    )
    return before, after


def _verify(*args: str):
    return runner.invoke(app, ["verify", *args])


def _payload_of(before: Path, after: Path, *extra: str) -> dict:
    result = _verify(str(before), str(after), "--json", *extra)
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)


def _strip(text: str, *keys: str) -> str:
    """`text` (an artifact's bytes) with top-level `keys` removed, re-serialized the same way
    `advise --json` serializes.

    How an artifact from an older sqlquality is obtained: exactly a current artifact minus
    the keys that version did not carry. Nothing is invented or reshaped.
    """
    payload = json.loads(text)
    for key in keys:
        del payload[key]
    return json.dumps(payload, indent=2, sort_keys=True)


# --- the pre-0.4.0 refusal ----------------------------------------------------------------


def test_an_artifact_missing_the_keys_this_feature_added_is_refused_rather_than_half_understood(
    tmp_path, monkeypatch
):
    """**The refusal Ruling 2 is about, and the one this pair proves independently.**

    `window`-being-a-string is the 0.3.0 marker specifically; it is not the contract. An
    artifact from an intermediate build can carry the whole `window` object and still lack
    `physical_state` and `query_groups` — which are emitted as `{}`/`[]` rather than omitted
    when empty *precisely* so an absent key can only mean "this artifact predates the
    feature" (see `advise_payload`'s docstring). Without those two keys `verify` can read
    neither whether a change was applied nor whether anything got faster, and would report
    every proposal `unobservable` as though the database had told it so.

    This pair passes every other refusal — two distinct files, distinct bytes, one engine,
    one redaction setting, a readable window — so it fails only if *this* check is present.
    """
    before_text, after_text = (
        _advise_stdout(monkeypatch, _pg_rows(calls=100, total_ms=5000.0)),
        _advise_stdout(monkeypatch, _pg_rows(calls=1000, total_ms=2000.0, indexed=True)),
    )
    before = _artifact(
        tmp_path, "before.json", _strip(before_text, "physical_state", "query_groups")
    )
    after = _artifact(tmp_path, "after.json", _strip(after_text, "physical_state", "query_groups"))

    result = _verify(str(before), str(after))
    assert result.exit_code == 2
    assert "0.4.0" in result.stderr
    assert "regenerate" in result.stderr.lower()
    # Names what is actually missing, so a user knows what to regenerate rather than being
    # told only that something is wrong.
    assert "physical_state" in result.stderr and "query_groups" in result.stderr


def test_a_0_3_0_artifact_is_refused_rather_than_half_understood(tmp_path, monkeypatch):
    """0.3.0's `window` is a prose string. Proceeding would mean classifying the window
    relation from absent data, and every verdict downstream would inherit that guess.

    The two artifacts are a real 0.4.0 pair reduced to 0.3.0's own shape: the three keys this
    feature added removed, and `window` collapsed back to the bare `description` sentence
    0.3.0 wrote there.
    """

    def to_0_3_0(text: str) -> str:
        payload = json.loads(_strip(text, "physical_state", "query_groups"))
        payload["window"] = payload["window"]["description"]
        for proposal in payload["proposals"]:
            proposal["evidence"].pop("fingerprint_digests", None)
        return json.dumps(payload, indent=2, sort_keys=True)

    before = _artifact(
        tmp_path, "before.json", to_0_3_0(_advise_stdout(monkeypatch, _pg_rows(calls=100)))
    )
    after = _artifact(
        tmp_path,
        "after.json",
        to_0_3_0(_advise_stdout(monkeypatch, _pg_rows(calls=1000, indexed=True))),
    )
    result = _verify(str(before), str(after))
    assert result.exit_code == 2
    assert "0.4.0" in result.stderr
    assert "regenerate" in result.stderr.lower()
    # The prose window is named as the thing that cannot be compared, rather than the
    # artifact being rejected without saying why.
    assert "window" in result.stderr and "sentence" in result.stderr


def test_a_missing_window_field_is_named_rather_than_waved_through(tmp_path, monkeypatch):
    """The narrowest case of the same refusal, and the reason the window's *fields* are
    checked and not only its type. `since_duration_seconds` is what `classify_windows`
    grades `COMPARABLE` on; an artifact predating it would silently classify as though
    neither run had ever passed `--since`, which is a verdict derived from an absent key.
    """
    before_text = _advise_stdout(monkeypatch, _pg_rows(calls=100))
    after_text = _advise_stdout(monkeypatch, _pg_rows(calls=1000, indexed=True))

    def drop_field(text: str) -> str:
        payload = json.loads(text)
        del payload["window"]["since_duration_seconds"]
        return json.dumps(payload, indent=2, sort_keys=True)

    before = _artifact(tmp_path, "before.json", drop_field(before_text))
    after = _artifact(tmp_path, "after.json", drop_field(after_text))
    result = _verify(str(before), str(after))
    assert result.exit_code == 2
    assert "since_duration_seconds" in result.stderr
    assert "0.4.0" in result.stderr


def test_a_complete_artifact_pair_is_not_refused_by_the_version_check(tmp_path, monkeypatch):
    """The control the three tests above need to be worth anything: a pair straight out of
    `advise --json`, untouched, must satisfy the whole key contract. Without this, tightening
    `_VERIFY_REQUIRED_KEYS` to something `advise` never emits would redden nothing."""
    before, after = _headline_pair(tmp_path, monkeypatch)
    result = _verify(str(before), str(after))
    assert result.exit_code == 0, result.output
    assert "0.4.0" not in result.stderr


def test_the_required_key_set_is_exactly_what_advise_emits(monkeypatch):
    """The refusal is pinned in **both** directions, and the second one is the gap this test
    closes.

    The control above catches *over-tightening*: a required key `advise` never writes would
    refuse every artifact. Nothing caught the opposite — a key added to `advise_payload` later
    that `_VERIFY_REQUIRED_KEYS` does not know about. An artifact from an intermediate build,
    carrying every key listed here but not the new one, would then pass the version check and
    earn a verdict derived from data it does not contain, which is the exact failure this
    refusal exists to prevent. Set equality against a freshly generated artifact means such a
    key cannot be added without this test naming it.

    `dbt` cannot be in the required set and is absent here for the reason it can never be
    required: `advise_payload` omits it entirely unless a manifest was loaded, so demanding it
    would refuse every dbt-free run — and this artifact, produced without `--manifest`, does
    not carry it.
    """
    from sqlquality.cli import _VERIFY_REQUIRED_KEYS, _VERIFY_REQUIRED_WINDOW_KEYS

    payload = json.loads(_advise_stdout(monkeypatch, _pg_rows()))
    assert "dbt" not in payload
    assert set(_VERIFY_REQUIRED_KEYS) == set(payload)
    assert set(_VERIFY_REQUIRED_WINDOW_KEYS) == set(payload["window"])


# --- the same artifact twice ---------------------------------------------------------------


def test_the_same_artifact_twice_is_refused(tmp_path, monkeypatch):
    """It would report every proposal unchanged, which looks like a finding rather than a
    mistake.

    Asserts the *path* arm's own wording ("same file"): the byte arm below would also fire on
    one file passed twice, so exit 2 alone would pass with this arm deleted.
    """
    before, _ = _headline_pair(tmp_path, monkeypatch)
    result = _verify(str(before), str(before))
    assert result.exit_code == 2
    assert "same file" in result.stderr
    assert "unchanged" in result.stderr


def test_two_byte_identical_copies_are_refused(tmp_path, monkeypatch):
    """Ruling 6's second arm, and the one a user actually reaches: two different paths
    holding the same run, because someone copied the baseline instead of taking a second
    one. Byte-identity is sound evidence of that rather than of a real no-change result —
    `pg_stat_statements`' counters accumulate, so even an unchanged workload moves the
    numbers between two genuine runs.

    Two distinct paths, so the path arm cannot fire: with only the byte arm removed this pair
    is reported at exit 0.
    """
    text = _advise_stdout(monkeypatch, _pg_rows())
    before = _artifact(tmp_path, "before.json", text)
    after = _artifact(tmp_path, "copy-of-before.json", text)
    assert before.read_bytes() == after.read_bytes()
    result = _verify(str(before), str(after))
    assert result.exit_code == 2
    assert "byte-identical" in result.stderr


def test_two_genuinely_different_runs_are_not_refused_as_the_same_artifact(tmp_path, monkeypatch):
    """The control: the headline pair differs in its timings, so neither arm may fire. This
    is what keeps the byte check from being a blanket refusal of every second run."""
    before, after = _headline_pair(tmp_path, monkeypatch)
    assert before.read_bytes() != after.read_bytes()
    result = _verify(str(before), str(after))
    assert result.exit_code == 0, result.output
    assert "byte-identical" not in result.stderr and "same file" not in result.stderr


# --- two engines, and two redaction settings ----------------------------------------------


def _redshift_stdout(monkeypatch, *, advisor_rows=(), extra=()) -> str:
    """A real `advise --engine redshift --json` run over a fake querier.

    Rows shaped for the real statements, following `tests/test_advise_cli.py`'s
    `_redshift_dbt_run`: one hot range predicate on `main.orders.created_at` so ADV101 fires,
    and `svv_table_info` facts saying the table is sorted on a different column so the
    proposal is not suppressed.
    """
    from sqlquality.workload.redshift import (
        CAP_ADVISOR,
        CAP_SCHEMA,
        CAP_TABLE_FACTS,
        CAP_WORKLOAD,
        RedshiftWorkloadAdapter,
    )

    rows = {
        CAP_WORKLOAD: [("select id from main.orders where created_at > '2026-01-01'", 5_000_000)],
        CAP_SCHEMA: [
            ("main", "orders", "id", "integer"),
            ("main", "orders", "created_at", "timestamp"),
        ],
        CAP_TABLE_FACTS: [("main", "orders", 10_000, 50, 0.0, 0.0, "EVEN", "id", 0.0)],
        CAP_ADVISOR: list(advisor_rows),
    }

    def fake_connect(self, params, timeout_s):
        def query(sql, bind):
            for capability, canned in rows.items():
                if RedshiftWorkloadAdapter.SQL[capability] in sql:
                    return canned
            return []

        self._query = query

    monkeypatch.setattr(RedshiftWorkloadAdapter, "connect", fake_connect)
    result = runner.invoke(
        app,
        [
            "advise",
            "--engine",
            "redshift",
            "--dsn",
            "postgresql://u@h/db",
            "--schema",
            "main",
            "--json",
            *extra,
        ],
    )
    assert result.exit_code == 0, result.output
    return result.stdout


def test_two_engines_are_refused_by_name(tmp_path, monkeypatch):
    """A Postgres mean and a Redshift mean measure two different servers, so no query group
    in one artifact corresponds to anything in the other. `classify_windows` has always
    graded such a pair `INCOMPARABLE`, but that is only a confidence *ceiling*: before this
    the pair still reached a group-level outcome and reported a `before`-only group as
    `DISAPPEARED` — "no longer appear in the after run" — for a Postgres query whose absence
    from a *Redshift* artifact says nothing whatsoever about it. `LOW` is not a grade at which
    that becomes honest, so it is refused.

    Both artifacts are genuine `advise --json` runs of their own engine. Everything else
    about the pair is fine — distinct files, distinct bytes, one redaction setting, complete
    keys, and Redshift reports `stats_reset_at: null` so the swapped-pair check cannot fire —
    so with only the engine check removed this pair is reported at exit 0.
    """
    before = _artifact(tmp_path, "before.json", _advise_stdout(monkeypatch, _pg_rows()))
    after = _artifact(tmp_path, "after.json", _redshift_stdout(monkeypatch))
    assert json.loads(before.read_text())["engine"] == "postgres"
    assert json.loads(after.read_text())["engine"] == "redshift"

    result = _verify(str(before), str(after))
    assert result.exit_code == 2
    assert "postgres" in result.stderr and "redshift" in result.stderr
    assert "different engines" in result.stderr


def test_two_runs_disagreeing_about_literal_redaction_are_refused(tmp_path, monkeypatch):
    """Ruling 3. `--keep-literals` changes the canonical query text `fingerprint_id` hashes,
    so two runs over an identical workload record the same query group under different
    digests: same queries, disjoint keys. Read as a measurement, that is "the query stopped
    running" for a query that never stopped — the governing rule's exact violation, and one
    reproduced from two real `advise` runs (see
    `tests/integration/test_verify_redaction_live.py`, which establishes that the digests
    really do diverge; this test's job is that the CLI refuses on the *declared settings*,
    before it ever looks at whether the keys happen to line up).

    Distinct files, distinct bytes, one engine, complete keys, same `stats_reset_at`: with
    only the redaction check removed this pair is reported at exit 0.
    """
    before = _artifact(tmp_path, "before.json", _advise_stdout(monkeypatch, _pg_rows(calls=100)))
    after = _artifact(
        tmp_path,
        "after.json",
        _advise_stdout(monkeypatch, _pg_rows(calls=1000, indexed=True), "--keep-literals"),
    )
    assert json.loads(before.read_text())["redacted"] is True
    assert json.loads(after.read_text())["redacted"] is False

    result = _verify(str(before), str(after))
    assert result.exit_code == 2
    assert "redaction" in result.stderr
    assert "--keep-literals" in result.stderr


# --- a swapped pair, as far as it is detectable at all ------------------------------------


def test_a_swapped_pair_is_refused_when_stats_reset_at_proves_it(tmp_path, monkeypatch):
    """Ruling 1. An `advise` artifact carries no run timestamp — deliberately, since Task 3
    established byte-determinism — so run order is not detectable in general. The one
    exception is sound: a server's statistics-reset instant cannot move backwards, so when
    both runs report a `stats_reset_at` and the AFTER run's is *earlier*, the two artifacts
    were passed in the wrong order.

    Both windows report a reset instant and the two differ, so this pair classifies
    `DISJOINT` and would otherwise be the cleanest comparison `verify` can make: with only
    this check removed it is reported at exit 0, with confidence `high`.
    """
    before = _pg_artifact(tmp_path, monkeypatch, "before.json", stats_reset="2026-08-01")
    after = _pg_artifact(
        tmp_path, monkeypatch, "after.json", stats_reset="2026-07-01", indexed=True, calls=1000
    )
    result = _verify(str(before), str(after))
    assert result.exit_code == 2
    assert "stats_reset_at" in result.stderr
    assert "wrong order" in result.stderr


def test_the_same_pair_in_the_right_order_is_reported(tmp_path, monkeypatch):
    """The control, and the proof the check tests the *direction* rather than merely noticing
    that two reset instants differ."""
    before = _pg_artifact(tmp_path, monkeypatch, "before.json", stats_reset="2026-07-01")
    after = _pg_artifact(
        tmp_path, monkeypatch, "after.json", stats_reset="2026-08-01", indexed=True, calls=1000
    )
    result = _verify(str(before), str(after))
    assert result.exit_code == 0, result.output
    assert "wrong order" not in result.stderr


def test_an_undetectable_run_order_is_disclosed_rather_than_guessed_at(tmp_path, monkeypatch):
    """The limit itself, surfaced where a user reads it rather than only in the code. A
    Postgres server that has never had `pg_stat_reset()` called reports `stats_reset_at:
    null` on both sides — the ordinary case — and then nothing in either artifact can place
    the two runs in time. `verify` says so and trusts the argument order, rather than
    inferring an order it cannot establish.
    """
    before = _pg_artifact(tmp_path, monkeypatch, "before.json", stats_reset=None)
    after = _pg_artifact(
        tmp_path, monkeypatch, "after.json", stats_reset=None, indexed=True, calls=1000
    )
    result = _verify(str(before), str(after))
    assert result.exit_code == 0, result.output
    assert "Run order is taken from the argument order" in result.stderr
    assert "no run timestamp" in result.stderr
    # …and the same limit is in the command's own --help, which is where someone deciding
    # which argument goes first will look.
    help_result = runner.invoke(app, ["verify", "--help"])
    assert help_result.exit_code == 0
    assert "argument order" in help_result.stdout


def test_the_reset_instant_is_parsed_not_string_compared(tmp_path, monkeypatch):
    """The swapped-pair check compares two *instants*, and it has to: a real Postgres artifact
    records `stats_reset_at` as `str(datetime)` of `pg_stat_database.stats_reset` — space
    separated, microseconds, and offset-aware — and two such strings from different UTC offsets
    sort the wrong way lexicographically.

    `2026-08-05 09:00:00-04:00` (13:00Z) is *later* than `2026-08-05 10:11:12.345678+00:00`,
    but sorts *earlier* as text. A string comparison would therefore refuse a correctly-ordered
    pair, telling a user their artifacts were swapped when they were not.
    """
    from sqlquality.cli import _verify_reset_instant

    earlier = _verify_reset_instant(
        {"window": {"stats_reset_at": "2026-08-05 10:11:12.345678+00:00"}}
    )
    later = _verify_reset_instant({"window": {"stats_reset_at": "2026-08-05 09:00:00-04:00"}})
    assert earlier is not None and later is not None
    assert earlier < later, "the aware-datetime comparison is not ordering by instant"
    assert "2026-08-05 09:00:00-04:00" < "2026-08-05 10:11:12.345678+00:00", (
        "the text comparison no longer disagrees with the instant comparison, so this test no "
        "longer demonstrates why the parse is needed"
    )
    # Anything unreadable is `None` — "cannot establish", never a substituted instant, since
    # refusing a pair on a value that could not be read would be its own false claim.
    for bad in (None, 42, "not a timestamp", [1]):
        assert _verify_reset_instant({"window": {"stats_reset_at": bad}}) is None


def test_a_reset_pair_that_cannot_be_ordered_is_not_refused(tmp_path, monkeypatch):
    """One instant timezone-aware and the other naive cannot be ordered at all (Python raises
    on the comparison). That is "cannot establish", not "in order" and not "swapped": the pair
    is reported, with the run-order caveat every report carries, rather than refused on a
    comparison that never happened."""
    before = _pg_artifact(tmp_path, monkeypatch, "before.json", stats_reset="2026-08-01 00:00:00")
    after = _pg_artifact(
        tmp_path,
        monkeypatch,
        "after.json",
        stats_reset="2026-07-01 00:00:00+00:00",
        indexed=True,
        calls=1000,
    )
    result = _verify(str(before), str(after))
    assert result.exit_code == 0, result.output
    assert "wrong order" not in result.stderr
    assert "Run order is taken from the argument order" in result.stderr


# --- unreadable or malformed input --------------------------------------------------------


def test_a_missing_file_is_refused_naming_which_argument_it_was(tmp_path, monkeypatch):
    before, _ = _headline_pair(tmp_path, monkeypatch)
    result = _verify(str(before), str(tmp_path / "nope.json"))
    assert result.exit_code == 2
    assert "AFTER" in result.stderr and "no such file" in result.stderr


def test_malformed_json_is_refused_with_the_parse_error(tmp_path, monkeypatch):
    """Exit 2, never a traceback and never exit 1: exit 1 is what `check` and `lint` use for
    real findings, so CI could not tell an unreadable file from a failed gate."""
    before, _ = _headline_pair(tmp_path, monkeypatch)
    broken = _artifact(tmp_path, "broken.json", '{"engine": "postgres", ')
    result = _verify(str(before), str(broken))
    assert result.exit_code == 2
    assert "not valid JSON" in result.stderr
    assert "broken.json" in result.stderr


def test_json_that_is_not_an_object_is_refused(tmp_path, monkeypatch):
    before, _ = _headline_pair(tmp_path, monkeypatch)
    listy = _artifact(tmp_path, "listy.json", "[1, 2, 3]")
    result = _verify(str(before), str(listy))
    assert result.exit_code == 2
    assert "not a JSON object" in result.stderr


def test_a_non_utf8_artifact_is_refused(tmp_path, monkeypatch):
    before, _ = _headline_pair(tmp_path, monkeypatch)
    latin = tmp_path / "latin.json"
    latin.write_bytes(b'{"engine": "postgr\xe8s"}')
    result = _verify(str(before), str(latin))
    assert result.exit_code == 2
    assert "not valid UTF-8" in result.stderr


# --- the report itself --------------------------------------------------------------------


def test_the_headline_case_is_reported_and_verify_never_gates(tmp_path, monkeypatch):
    """The whole point of the feature, end to end through the command: ADV001 was proposed,
    the index was created, and the query group it cited got faster per call.

    This is also the test that fails if the `verdicts(...)` call in the command body is
    replaced by `[]` — the defect (a feature disconnected from the CLI, suite green) that has
    appeared three times in this codebase.
    """
    before, after = _headline_pair(tmp_path, monkeypatch)
    result = _verify(str(before), str(after))
    assert result.exit_code == 0, result.output
    assert "ADV001 public.orders (status)" in result.stdout
    assert "improved" in result.stdout
    # applied, and the two means, both on screen — the table's whole content.
    assert "yes" in result.stdout
    assert "50.0" in result.stdout and "2.0" in result.stdout
    # NESTED caps at medium; nothing here may claim high.
    assert "medium" in result.stdout
    assert "high" not in result.stdout


def test_the_nested_window_caveat_reaches_the_user(tmp_path, monkeypatch):
    """On the common Postgres path a real improvement is understated. A user who does not
    know that will read `unchanged` as "it did not work" — so the caveat is printed, and it
    names the remedy (`pg_stat_statements_reset()`, which the *user* runs: sqlquality never
    writes to their database)."""
    before, after = _headline_pair(tmp_path, monkeypatch)
    result = _verify(str(before), str(after))
    assert result.exit_code == 0, result.output
    assert "nested" in result.stderr
    assert "understate" in result.stderr.lower()
    assert "pg_stat_statements_reset" in result.stderr


def test_a_disjoint_window_does_not_print_the_nested_caveat(tmp_path, monkeypatch):
    """The control the test above needs: the caveat must be attributable to the window
    relation, not printed unconditionally. Two different reset instants (in the right order)
    are `DISJOINT` — independent samples, the cleanest comparison — and must say so."""
    before = _pg_artifact(tmp_path, monkeypatch, "before.json", stats_reset="2026-07-01")
    after = _pg_artifact(
        tmp_path,
        monkeypatch,
        "after.json",
        stats_reset="2026-08-01",
        indexed=True,
        calls=1000,
        total_ms=2000.0,
    )
    result = _verify(str(before), str(after))
    assert result.exit_code == 0, result.output
    assert "disjoint" in result.stderr
    assert "understate" not in result.stderr.lower()
    assert "high" in result.stdout


def test_the_workload_context_line_reports_both_runs(tmp_path, monkeypatch):
    """Design decision 4: a global workload shift is the confound this feature exists to
    survive, and a reader can only see it if both runs' totals are on screen rather than
    deduced from the per-proposal figures."""
    before, after = _headline_pair(tmp_path, monkeypatch)
    result = _verify(str(before), str(after))
    assert result.exit_code == 0, result.output
    assert "workload: before 5000.0 ms across 1 query group(s)" in result.stderr
    assert "after 2000.0 ms across 1 query group(s)" in result.stderr


def test_applied_but_unchanged_reaches_the_user(tmp_path, monkeypatch):
    """The single most valuable outcome, per the design spec: the work was done and it did
    not help. It must not be buried — the note is printed under the table, not only carried
    in `--json`."""
    before = _pg_artifact(tmp_path, monkeypatch, "before.json", calls=100, total_ms=5000.0)
    after = _pg_artifact(
        tmp_path, monkeypatch, "after.json", calls=100, total_ms=5000.0, indexed=True
    )
    result = _verify(str(before), str(after))
    assert result.exit_code == 0, result.output
    assert "unchanged" in result.stdout
    assert "Applied but unchanged" in result.stdout


def test_a_regression_is_reported_and_still_exits_0(tmp_path, monkeypatch):
    """`verify` reports and never gates: there is no `--gate` flag, and a regression must not
    exit non-zero, or every CI job that runs it becomes a gate nobody asked for."""
    before = _pg_artifact(tmp_path, monkeypatch, "before.json", calls=100, total_ms=5000.0)
    after = _pg_artifact(
        tmp_path, monkeypatch, "after.json", calls=100, total_ms=20000.0, indexed=True
    )
    result = _verify(str(before), str(after))
    assert result.exit_code == 0, result.output
    assert "regressed" in result.stdout


def test_the_json_payload_carries_the_verdicts_the_caveats_and_the_window_relation(
    tmp_path, monkeypatch
):
    """`--json` is the machine surface, and it must carry the caveats as data: a consumer
    reporting "1 improved" without the window relation and its consequences would reproduce,
    one layer up, exactly the over-claim this feature exists to prevent."""
    before, after = _headline_pair(tmp_path, monkeypatch)
    payload = _payload_of(before, after)
    assert payload["window_relation"] == "nested"
    assert payload["confidence_ceiling"] == "medium"
    [verdict] = payload["verdicts"]
    assert verdict["code"] == "ADV001"
    assert verdict["applied"] is True
    assert verdict["outcome"] == "improved"
    assert verdict["confidence"] == "medium"
    assert verdict["mean_ms_before"] == 50.0 and verdict["mean_ms_after"] == 2.0
    # cost_share rides along as context — whether the finding still matters — never as the
    # measure of whether it got better.
    assert verdict["cost_share_before"] == 1.0
    assert payload["summary"] == {
        "proposals": 1,
        "applied": 1,
        "not_applied": 0,
        "applied_unknown": 0,
        "outcomes": {
            "improved": 1,
            "unchanged": 0,
            "regressed": 0,
            "disappeared": 0,
            "not_applied": 0,
            "unobservable": 0,
        },
    }
    assert any("understate" in caveat for caveat in payload["caveats"])
    assert payload["before"]["total_cost_ms"] == 5000.0
    assert payload["after"]["query_groups"] == 1
    # No stderr chatter on the json path: the payload carries all of it.
    result = _verify(str(before), str(after), "--json")
    assert result.stderr == ""


def test_the_markdown_report_carries_the_table_the_caveats_and_the_notes(tmp_path, monkeypatch):
    before, after = _headline_pair(tmp_path, monkeypatch)
    out = tmp_path / "verify.md"
    result = _verify(str(before), str(after), "--markdown", str(out))
    assert result.exit_code == 0, result.output
    text = out.read_text(encoding="utf-8")
    assert "# sqlquality verify — postgres" in text
    assert "| proposal | applied | outcome | mean per call | confidence |" in text
    assert "ADV001 public.orders (status)" in text
    assert "improved" in text
    assert "understate" in text
    assert "50.0 → 2.0 ms" in text


def test_an_unreadable_engine_renders_as_unknown_not_as_none(tmp_path, monkeypatch):
    """One rendering of "this artifact does not say", not two.

    `{"engine": null}` passes every check `verify` makes: the version check tests only whether
    the key is *present*, and `artifact_incomparabilities` reads `window["engine"]`, which is
    intact here. The heading then interpolated `ArtifactFacts.engine`'s `None` directly and
    printed `Verify — None`, while `verify_workload_line` two lines above rendered its own
    unreadable values as `unknown`. A Python repr in user-facing text is also indistinguishable
    from an engine literally named `None`.

    Cosmetic — `advise_payload` writes `engine` from a required argument, so only a hand-edited
    artifact reaches it — and asserted on both rendered surfaces. `--json` deliberately keeps
    the real `null`, which the last assertion pins: a machine consumer needs the distinction the
    label flattens.
    """

    def blank_engine(text: str) -> str:
        payload = json.loads(text)
        payload["engine"] = None
        return json.dumps(payload, indent=2, sort_keys=True)

    before = _artifact(
        tmp_path, "before.json", blank_engine(_advise_stdout(monkeypatch, _pg_rows(calls=100)))
    )
    after = _artifact(
        tmp_path,
        "after.json",
        blank_engine(_advise_stdout(monkeypatch, _pg_rows(calls=1000, indexed=True))),
    )
    out = tmp_path / "verify.md"
    result = _verify(str(before), str(after), "--markdown", str(out))
    assert result.exit_code == 0, result.output
    assert "# sqlquality verify — unknown" in out.read_text(encoding="utf-8")
    stdout = _plain(result.stdout)
    assert "unknown" in stdout
    assert "None" not in stdout
    assert _payload_of(before, after)["before"]["engine"] is None


def test_an_unwritable_markdown_path_exits_2_not_1(tmp_path, monkeypatch):
    """Same house rule as `advise` and `check`: exit 1 is reserved for findings, so an
    unwritable path must never look like one."""
    before, after = _headline_pair(tmp_path, monkeypatch)
    result = _verify(str(before), str(after), "--markdown", str(tmp_path / "nodir" / "out.md"))
    assert result.exit_code == 2
    assert "Could not write --markdown" in result.stderr


# --- what one run's own conditions produced, disclosed rather than measured ---------------


def test_a_limit_mismatch_is_disclosed_and_forbids_a_disappeared_verdict(tmp_path, monkeypatch):
    """Task 5's derived predicate, surfaced. The `after` run sampled fewer query groups, so a
    group present in `before` and absent from `after` may be a sampling artifact rather than
    a real disappearance — and `verify` must not grade `disappeared` on the strength of that
    absence alone.

    The `after` run's hot statement is a different one against the same relation, which is
    how the cited digest goes missing while the relation stays fully observed (index present,
    `applied` genuinely `True`) — a shape `advise` emits whenever the hot query changes
    between two runs.
    """
    before = _pg_artifact(tmp_path, monkeypatch, "before.json")
    after = _pg_artifact(
        tmp_path,
        monkeypatch,
        "after.json",
        "--limit",
        "5",
        statement="select id from orders where created_at > $1",
        indexed=True,
    )
    payload = _payload_of(before, after)
    assert payload["limit"] == {"before": 500, "after": 5, "may_be_sampling_artifact": True}
    assert any("sampling artifact" in caveat for caveat in payload["caveats"])
    [verdict] = [v for v in payload["verdicts"] if v["key"][1:] == ["public", "orders", "status"]]
    assert verdict["applied"] is True
    assert verdict["outcome"] == "unobservable", verdict
    assert verdict["confidence"] == "low"
    assert "sampling artifact" in verdict["note"]


def test_a_degraded_read_in_either_run_is_disclosed(tmp_path, monkeypatch):
    """A capability the run could not read produces the same emptiness a real change would.
    Both runs' `degraded` lists reach the user, and the verdict that rests on the missing
    read says `unknown` rather than `no`.
    """
    before = _pg_artifact(tmp_path, monkeypatch, "before.json")
    denied = _pg_rows(calls=1000, total_ms=2000.0)
    denied["pg_index"] = RuntimeError("permission denied for relation pg_index")
    after = _artifact(tmp_path, "after.json", _advise_stdout(monkeypatch, denied))
    assert json.loads(after.read_text())["degraded"], "the denial never reached the artifact"

    result = _verify(str(before), str(after))
    assert result.exit_code == 0, result.output
    assert "The after run ran with reduced coverage — indexes:" in result.stderr
    assert "permission denied for relation pg_index" in result.stderr
    payload = _payload_of(before, after)
    [verdict] = payload["verdicts"]
    # Never a guessed `False`: "we could not tell whether you did it" is a different
    # statement from "you did not do it".
    assert verdict["applied"] is None
    assert verdict["outcome"] == "unobservable"


def test_a_key_collision_is_disclosed_as_neither_a_disappearance_nor_a_new_finding(
    tmp_path, monkeypatch
):
    """Task 4's ambiguity, surfaced rather than swallowed. Two Amazon Redshift Advisor rows
    for one relation with the same `type` produce two ADV105 proposals under one key (see
    `propose_advisor` — one proposal per row, no dedupe), so the `before` run cannot tell its
    own two recommendations apart.

    That is a fact about the `before` artifact, prior to any comparison: it must not abort the
    run, must not be graded `disappeared`, and — the sharp case here — its `after`-side
    counterpart, which *is* unambiguous, must not be reported as a *new* finding either.
    """
    advisor_two = [
        ("db", "main", "orders", "sortkey", "SORTKEY(id)", "SORTKEY(created_at)"),
        ("db", "main", "orders", "sortkey", "SORTKEY(id)", None),
    ]
    advisor_one = [("db", "main", "orders", "sortkey", "SORTKEY(id)", "SORTKEY(created_at)")]
    before = _artifact(
        tmp_path, "before.json", _redshift_stdout(monkeypatch, advisor_rows=advisor_two)
    )
    after = _artifact(
        tmp_path, "after.json", _redshift_stdout(monkeypatch, advisor_rows=advisor_one)
    )
    before_codes = [p["code"] for p in json.loads(before.read_text())["proposals"]]
    assert before_codes.count("ADV105") == 2, before_codes

    result = _verify(str(before), str(after))
    assert result.exit_code == 0, result.output
    assert "could not be matched unambiguously" in result.stderr
    assert "ADV105 main.orders" in result.stderr

    payload = _payload_of(before, after)
    assert [list(key) for key in payload["before"]["unmatched_keys"]] == [
        ["ADV105", "main", "orders", "sortkey"]
    ]
    assert all(v["code"] != "ADV105" for v in payload["verdicts"]), payload["verdicts"]
    # The half a naive implementation gets wrong: `after` matches that key unambiguously, so
    # subtracting only `before`'s *matched* keys would announce it as new.
    assert payload["new_in_after"] == []
    assert "disappeared" not in [v["outcome"] for v in payload["verdicts"]]


def test_a_proposal_only_the_after_run_makes_is_reported_as_new_not_as_a_verdict(
    tmp_path, monkeypatch
):
    """The other side of Ruling 10: `verdicts` grades each *before*-run proposal, so a finding
    with no counterpart to compare against is new rather than changed — and saying nothing
    about it at all would hide a fresh regression the second run just found."""
    before = _pg_artifact(tmp_path, monkeypatch, "before.json")
    after = _pg_artifact(
        tmp_path,
        monkeypatch,
        "after.json",
        statement="select id from orders where created_at > $1",
        indexed=True,
        calls=1000,
        total_ms=2000.0,
    )
    payload = _payload_of(before, after)
    assert payload["new_in_after"] == [["ADV001", "public", "orders", "created_at"]]
    result = _verify(str(before), str(after))
    assert "appear only in the after run" in result.stderr
    assert "ADV001 public.orders (created_at)" in result.stderr
    # The control for the two gated routes below: this pair's `before` run read everything and
    # both runs sampled the same known number of query groups, so "new" is a claim the
    # artifacts do support and it must still be made in as many words. Without this assertion,
    # gating the claim *unconditionally* would leave the suite green.
    assert "new rather than changed" in result.stderr
    assert payload["new_in_after_withheld"] is None
    assert "cannot be established" not in result.stderr


def test_a_degraded_before_run_does_not_make_an_after_only_proposal_a_new_finding(
    tmp_path, monkeypatch
):
    """**F1 of the whole-branch review — the fifth door of this feature's dominant defect
    class, and the one that shipped.**

    `new_in_after` announced every after-only proposal as "new rather than changed", inferred
    purely from its absence in `before`. Here `before`'s `pg_stat_statements` read is denied, so
    `before` emits no proposals at all — the recommendation is not new, the earlier run simply
    could not look. Reproduced during review from two real `advise` runs on live PostgreSQL 16
    (one database before `CREATE EXTENSION pg_stat_statements`, one after), and again here
    through the same real `advise` path this whole file uses.

    **The two assertions that make this the damning route rather than a variant of the next
    test:** both runs record the same known `--limit`, so `may_be_sampling_artifact` is `False`
    and the sampling gate structurally cannot fire — the *only* thing standing between the user
    and the false claim is the degraded-read arm. And the caveat printed one bullet above
    promises "any verdict resting on something being absent from that run says so", so before
    the fix a single output contradicted itself.
    """
    before = _artifact(
        tmp_path,
        "before.json",
        _advise_stdout(
            monkeypatch,
            {**_pg_rows(), "pg_stat_statements": RuntimeError("permission denied")},
        ),
    )
    after = _pg_artifact(tmp_path, monkeypatch, "after.json")

    before_payload = json.loads(before.read_text())
    assert [d["capability"] for d in before_payload["degraded"]] == ["workload"]
    assert before_payload["proposals"] == [], "the before run must make no proposal at all"

    payload = _payload_of(before, after)
    assert payload["new_in_after"] == [["ADV001", "public", "orders", "status"]]
    assert payload["limit"]["may_be_sampling_artifact"] is False, (
        "the --limit gate can fire here, so this test would not prove the degraded-read arm"
    )
    assert payload["new_in_after_withheld"] is not None
    assert "workload" in payload["new_in_after_withheld"]

    result = _verify(str(before), str(after))
    assert result.exit_code == 0, result.output
    # The false claim is gone…
    assert "new rather than changed" not in result.stderr
    # …the reason is named…
    assert "cannot rule out that they were already true and unseen" in result.stderr
    assert "degraded read(s) workload" in result.stderr
    # …and the keys are still listed, because withholding them too would hide a genuinely new
    # finding. The gate is on the claim, never on the disclosure.
    assert "ADV001 public.orders (status)" in result.stderr


def test_a_truncated_before_window_does_not_make_an_after_only_proposal_a_new_finding(
    tmp_path, monkeypatch
):
    """F1's second route: a `before` run taken at a smaller `--limit`.

    A window that sampled fewer query groups may never have held the evidence the later run's
    proposal rests on. Review reproduced this live with `--limit 1` then `--limit 500`; here the
    two runs record limits 1 and 500 in their windows, which is what `window_limits` reads.
    Before the fix a sampling caveat *was* printed for this pair — but as a sibling bullet about
    *disappearance*, never linked to the "new" claim, which stayed unqualified.

    The degraded arm cannot be what fires: both runs' `degraded` lists are asserted empty.
    """
    before = _pg_artifact(tmp_path, monkeypatch, "before.json", "--limit", "1")
    after = _pg_artifact(
        tmp_path,
        monkeypatch,
        "after.json",
        statement="select id from orders where created_at > $1",
        indexed=True,
        calls=1000,
        total_ms=2000.0,
    )
    for path in (before, after):
        assert json.loads(path.read_text())["degraded"] == [], path

    payload = _payload_of(before, after)
    assert payload["new_in_after"] == [["ADV001", "public", "orders", "created_at"]]
    assert payload["limit"] == {"before": 1, "after": 500, "may_be_sampling_artifact": True}
    assert payload["new_in_after_withheld"] is not None

    result = _verify(str(before), str(after))
    assert result.exit_code == 0, result.output
    assert "new rather than changed" not in result.stderr
    assert "cannot rule out that they were already true and unseen" in result.stderr
    assert "--limit before=1, after=500" in result.stderr
    assert "ADV001 public.orders (created_at)" in result.stderr


#: The two measured routes by which a **degraded `after` run** emits a proposal a fully-observed
#: run withholds — the inverse of F1's mechanism and the same false sentence to the user. Each
#: entry is `(marker to deny, capability name, the base rows that make the fully-observed run
#: propose nothing)`.
#:
#: * `table_facts` — a 50-row table is below the size threshold ADV001 applies when it can read
#:   the size, and the threshold cannot be applied at all when `pg_total_relation_size` is denied.
#: * `indexes` — an existing index already covers the hot predicate, so ADV001 has nothing to
#:   recommend; with `pg_index` denied the coverage check cannot see it.
_RELAXATION_ROUTES = [
    (
        "pg_total_relation_size",
        "table_facts",
        {"pg_total_relation_size": [("public", "orders", 50, 8192)]},
    ),
    ("pg_index", "indexes", {"pg_index": _INDEX_ON_STATUS}),
]


@pytest.mark.parametrize(
    ("marker", "capability", "base_rows"),
    _RELAXATION_ROUTES,
    ids=[route[1] for route in _RELAXATION_ROUTES],
)
def test_a_degraded_after_run_does_not_make_its_own_relaxed_proposal_a_new_finding(
    tmp_path, monkeypatch, marker, capability, base_rows
):
    """**The sixth door: a *presence* fabricated by missing coverage, rather than an absence read
    as a measurement.** Found while measuring F1's dependency table, and fixed here because the
    user-visible outcome is identical — `verify` asserting "this is new" on the strength of one
    run's own limitations.

    A rule that cannot evaluate a threshold proposes anyway at reduced confidence, which is the
    right posture for `advise` (disclose rather than withhold) and a false claim once `verify`
    reads the result as new. Both routes are measured, not theorised — each `base_rows` below
    makes the fully-observed run propose **nothing**, and the assertions check that before using
    it, so a route that stopped relaxing would fail here rather than pass vacuously.

    **The isolation that makes this test about the new arm and nothing else:** `before`'s
    `degraded` list is empty, so F1's before-side arm cannot fire, and both runs record the same
    known `--limit`, so the sampling arm cannot either.
    """
    base = {**_pg_rows(), **base_rows}
    before = _artifact(tmp_path, "before.json", _advise_stdout(monkeypatch, base))
    after = _artifact(
        tmp_path,
        "after.json",
        _advise_stdout(monkeypatch, {**base, marker: RuntimeError("denied")}),
    )

    before_payload = json.loads(before.read_text())
    after_payload = json.loads(after.read_text())
    # The premise of the whole test, asserted from the two real artifacts.
    assert before_payload["degraded"] == [], "the before run must be fully observed"
    assert before_payload["proposals"] == [], (
        f"the fully-observed run already proposes something, so {capability} is not relaxing "
        f"anything here: {before_payload['proposals']}"
    )
    assert [d["capability"] for d in after_payload["degraded"]] == [capability]
    assert [p["code"] for p in after_payload["proposals"]] == ["ADV001"], (
        f"denying {marker} did not relax a rule into proposing, so this route no longer "
        f"reproduces: {after_payload['proposals']}"
    )

    payload = _payload_of(before, after)
    assert payload["new_in_after"] == [["ADV001", "public", "orders", "status"]]
    assert payload["limit"]["may_be_sampling_artifact"] is False, (
        "the --limit arm can fire here, so this test would not prove the relaxation arm"
    )
    assert payload["new_in_after_withheld"] is not None
    assert capability in payload["new_in_after_withheld"]

    result = _verify(str(before), str(after))
    assert result.exit_code == 0, result.output
    assert "new rather than changed" not in result.stderr
    assert "cannot rule out that they were already true and unseen" in result.stderr
    # The relaxation's own wording, not the absence arm's: the after run may have made a
    # recommendation nobody fully observing the database would have made.
    assert "a fully-observed run would not have made at all" in result.stderr
    assert "ADV001 public.orders (status)" in result.stderr


def test_an_ndv_denied_after_run_is_gated_and_a_workload_denied_one_has_nothing_to_claim(
    tmp_path, monkeypatch
):
    """Two facts about the relaxation set that the two routes above do not establish.

    First, that the set is consulted for `ndv` — the member included on the
    disclose-an-unproven-premise posture rather than on a demonstrated route (see
    `_PROPOSAL_RELAXING_DEGRADATIONS`), so a narrowing of the set would show up here.

    Second, why the *inverse* control cannot be written from the Postgres side at all: the only
    removal-only capabilities Postgres can degrade are `workload` and `schema`, and both empty
    `proposals` entirely, so a run degraded that way has no after-only proposal to mis-call new
    in the first place. The genuine "a non-relaxing degradation must not gate" control is
    therefore a unit test over a synthetic `degraded` list — see
    `tests/test_verify.py::test_new_proposal_disclosure_withholds_on_a_degraded_before_and_on_a_limit_mismatch`
    — and this assertion records why it lives there rather than here.
    """
    before = _pg_artifact(tmp_path, monkeypatch, "before.json")
    after_rows = {
        **_pg_rows(
            statement="select id from orders where created_at > $1",
            indexed=True,
            calls=1000,
            total_ms=2000.0,
        ),
        "pg_stats": RuntimeError("denied"),
    }
    after = _artifact(tmp_path, "after.json", _advise_stdout(monkeypatch, after_rows))
    assert [d["capability"] for d in json.loads(after.read_text())["degraded"]] == ["ndv"]

    payload = _payload_of(before, after)
    assert payload["new_in_after"] == [["ADV001", "public", "orders", "created_at"]]
    # `ndv` *is* in the relaxation set (see `_PROPOSAL_RELAXING_DEGRADATIONS`: included on the
    # disclose-an-unproven-premise posture rather than a demonstrated route), so this pair is
    # correctly gated — which is exactly what pins that the set is consulted rather than ignored.
    assert payload["new_in_after_withheld"] is not None
    assert "ndv" in payload["new_in_after_withheld"]

    # And the genuinely-inert direction: a `workload` denial in `after` empties `proposals`
    # entirely, so there is no after-only proposal to mis-call new in the first place.
    empty_after = _artifact(
        tmp_path,
        "empty_after.json",
        _advise_stdout(monkeypatch, {**_pg_rows(), "pg_stat_statements": RuntimeError("denied")}),
    )
    assert json.loads(empty_after.read_text())["proposals"] == []
    assert _payload_of(before, empty_after)["new_in_after"] == []


def test_the_new_claim_and_the_disappeared_claim_are_gated_symmetrically(tmp_path, monkeypatch):
    """The asymmetry was the tell, so it is what this test pins.

    `DISAPPEARED` (a claim from absence in `after`) has always been forbidden under a `--limit`
    mismatch; its mirror image (a claim from absence in `before`) had no gate at all until F1.
    One `--limit`-mismatched pair, both directions asserted in one place, so a future edit
    cannot restore the asymmetry by touching only one side.
    """
    before = _pg_artifact(tmp_path, monkeypatch, "before.json", "--limit", "1")
    after = _pg_artifact(
        tmp_path,
        monkeypatch,
        "after.json",
        statement="select id from orders where created_at > $1",
        indexed=True,
        calls=1000,
        total_ms=2000.0,
    )
    payload = _payload_of(before, after)
    assert payload["limit"]["may_be_sampling_artifact"] is True
    # The `after`-side direction: the before proposal's cited group is gone from `after`, and
    # `disappeared` is refused for it.
    outcomes = {v["outcome"] for v in payload["verdicts"]}
    assert "disappeared" not in outcomes, payload["verdicts"]
    # The `before`-side direction: the after-only proposal is not called new.
    assert payload["new_in_after"], "no after-only proposal, so the mirror claim is not exercised"
    assert payload["new_in_after_withheld"] is not None


# --- structural pins ----------------------------------------------------------------------


def test_every_window_relation_has_a_user_facing_caveat():
    """The same structural pin `_CONFIDENCE_BY_RELATION` and `_MISMATCH_EFFECT` already
    carry, for the table a user actually reads: a fifth `WindowRelation` added without an
    entry must redden the suite rather than silently print nothing about what the relation
    means for the verdicts under it."""
    assert set(WindowRelation) == set(_WINDOW_RELATION_CAVEAT), (
        "unclassified: "
        f"{sorted(r.value for r in set(WindowRelation) - set(_WINDOW_RELATION_CAVEAT))}"
    )
    for relation, caveat in _WINDOW_RELATION_CAVEAT.items():
        assert caveat, f"{relation.value} discloses nothing"
        assert relation.value in caveat, f"{relation.value}'s caveat does not name it"


def test_verify_is_listed_in_the_root_help_and_documents_its_contract():
    """`--help` is `verify`'s primary documentation surface, so it is asserted rather than
    assumed. Read through `_plain`: rich styles an option name in pieces, so `--json` is not
    a contiguous substring of the raw output — a raw assertion here fails while the flag is
    perfectly present, which is the ANSI trap this repository's mutation harnesses have hit
    before."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "verify" in _plain(result.stdout)
    detail = runner.invoke(app, ["verify", "--help"])
    assert detail.exit_code == 0
    text = _plain(detail.stdout)
    assert "--json" in text and "--markdown" in text
    # No --gate flag: explicitly out of scope, and saying so is part of the contract.
    assert "There is no --gate flag" in text
    assert "never gates" in text


@pytest.mark.parametrize("flag", ["--json", "--markdown"])
def test_the_refusals_do_not_depend_on_the_output_flags(tmp_path, monkeypatch, flag):
    """A refusal must fire before anything is rendered or written, whichever surface was
    asked for — otherwise `--json` would be a way to get a verdict out of a pair the
    terminal path refuses."""
    before, _ = _headline_pair(tmp_path, monkeypatch)
    args = [str(before), str(before), flag]
    if flag == "--markdown":
        args.append(str(tmp_path / "out.md"))
    result = _verify(*args)
    assert result.exit_code == 2
    assert "same file" in result.stderr
    assert not (tmp_path / "out.md").exists()
