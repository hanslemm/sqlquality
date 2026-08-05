"""Golden `advise --json` pairs, one per `VerifyOutcome`, and what `verify` must say about
each.

**Every artifact in `tests/fixtures/verify/` is the verbatim stdout of a real `sqlquality
advise --json` run against a real PostgreSQL 16** — see `tests/fixtures/verify/regenerate.py`
for the exact workload, index and counter-reset sequence behind each pair, and for the
assertions each scenario had to satisfy before its bytes were written. Nothing here is
hand-composed, and nothing was edited after generation.

That matters more than it might appear. Task 6 of this feature shipped a headline test whose
`after` fixture paired an empty `"proposals"` with a populated `"query_groups"`, which the
then-current `_query_groups_payload` (scoped to what proposals cited) could not produce, so
the test passed while the real path was broken. A fixture `advise` cannot emit proves nothing
about `verify`, which only ever reads what `advise` wrote. **That particular shape is
reachable today** — `improved.after.json` below *is* an empty `proposals` beside three
`query_groups`, straight out of the real command, because a resolved proposal stops firing
while the query it was about keeps running. The lesson is not about that one shape: it is that
the only way to know a fixture is emittable is to have emitted it.

`test_every_golden_has_the_shape_advise_emits_today` is the guard that keeps that true as the
payload evolves: it compares each fixture's key structure against a *freshly generated*
`advise --json` artifact, so a payload key added later without regenerating these files
reddens here rather than leaving the goldens quietly describing an older contract.

Each pair asserts the **outcome and the confidence**, because they are decided by two
independent mechanisms — the outcome by `verdicts`' branch order and `_grade`'s threshold, the
confidence by `classify_windows` and the ceiling it earns — and a pair pinning only the
outcome would let a window misclassification through unnoticed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from sqlquality.cli import app
from sqlquality.models import Confidence
from sqlquality.verify import (
    VerifyOutcome,
    WindowRelation,
    classify_windows,
    group_index,
    verdicts,
)

runner = CliRunner()

FIXTURES = Path(__file__).parent / "fixtures" / "verify"

#: Any ANSI escape sequence — rich splits a styled cell across several of them.
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _load(name: str) -> tuple[dict, dict]:
    before = json.loads((FIXTURES / f"{name}.before.json").read_text())
    after = json.loads((FIXTURES / f"{name}.after.json").read_text())
    return before, after


def _adv001(payload: dict) -> dict | None:
    for proposal in payload["proposals"]:
        if proposal["code"] == "ADV001" and proposal["evidence"].get("table") == "orders":
            return proposal
    return None


def _verdict(before: dict, after: dict):
    """The one ADV001 verdict for `orders`, with the non-vacuity guard in front of it.

    The guard is the point: a golden whose `before` artifact carried no ADV001 proposal at all
    would produce no verdict, and every assertion below would then be asserting about an empty
    list. That is the shape of hollow test this feature has produced repeatedly, so the
    proposal, its citation, and the citation actually resolving in `before`'s own
    `query_groups` are each checked before any verdict is examined.
    """
    proposal = _adv001(before)
    assert proposal is not None, f"no ADV001 for orders in before: {before['proposals']}"
    cited = proposal["evidence"]["fingerprint_digests"]
    assert cited, "the before proposal cites no query group, so no speed change is gradable"
    groups = group_index(before)
    assert set(cited) & set(groups), (
        f"no cited digest resolves in before's own query_groups: cited={sorted(cited)} "
        f"present={sorted(groups)}"
    )
    ours = [v for v in verdicts(before, after) if v.code == "ADV001" and v.key[2] == "orders"]
    assert len(ours) == 1, f"expected exactly one ADV001 verdict for orders: {ours}"
    return ours[0]


#: Every golden pair, with the two facts each pins and the window relation it was built to
#: produce. `DISJOINT` pairs called `pg_stat_reset()` between the two runs; the `NESTED` pair
#: called it once, before both.
_GOLDENS = [
    ("improved", True, VerifyOutcome.IMPROVED, Confidence.HIGH, WindowRelation.DISJOINT),
    ("unchanged_nested", True, VerifyOutcome.UNCHANGED, Confidence.MEDIUM, WindowRelation.NESTED),
    ("regressed", True, VerifyOutcome.REGRESSED, Confidence.HIGH, WindowRelation.DISJOINT),
    ("disappeared", True, VerifyOutcome.DISAPPEARED, Confidence.HIGH, WindowRelation.DISJOINT),
    ("not_applied", False, VerifyOutcome.NOT_APPLIED, Confidence.HIGH, WindowRelation.DISJOINT),
    (
        "limit_truncated",
        True,
        VerifyOutcome.UNOBSERVABLE,
        Confidence.LOW,
        WindowRelation.DISJOINT,
    ),
]


@pytest.mark.parametrize(
    ("name", "applied", "outcome", "confidence", "relation"),
    _GOLDENS,
    ids=[row[0] for row in _GOLDENS],
)
def test_each_golden_pair_grades_to_its_recorded_outcome_and_confidence(
    name, applied, outcome, confidence, relation
):
    """One real artifact pair per outcome, graded on applied, outcome **and** confidence.

    `limit_truncated`'s row is the interesting one: its window relation is `DISJOINT`, which
    would earn `HIGH`, and its cited group is absent from the after artifact exactly as
    `disappeared`'s is. It still grades `unobservable`/`LOW`, because the two runs used
    different `--limit` values and a smaller window can drop a group with nothing having
    changed about it — so `verdicts` refuses `DISAPPEARED` there and says why.
    """
    before, after = _load(name)
    assert classify_windows(before, after) is relation
    verdict = _verdict(before, after)
    assert verdict.applied is applied, f"{name}: {verdict.note!r}"
    assert verdict.outcome is outcome, (
        f"{name}: mean_before={verdict.mean_before} mean_after={verdict.mean_after} "
        f"note={verdict.note!r}"
    )
    assert verdict.confidence is confidence, f"{name}: note={verdict.note!r}"


def test_the_improved_golden_reports_a_real_mean_on_both_sides():
    """The gain is not a rounding artifact: the measured pair is roughly 150x faster per
    call, against a 10% threshold. Recorded as a ratio rather than as the raw numbers, which
    are timings and would make this an assertion about one machine."""
    before, after = _load("improved")
    verdict = _verdict(before, after)
    assert verdict.mean_before is not None and verdict.mean_after is not None
    assert verdict.mean_after < verdict.mean_before / 10


def test_the_nested_golden_says_applied_but_unchanged_in_as_many_words():
    """The single most valuable thing this feature reports, and the reason the `unchanged`
    golden is the nested pair rather than a contrived one.

    The index is genuinely there and genuinely helps every call it serves. But
    `pg_stat_statements` is cumulative and the counters were never reset, so 100 pre-index
    executions still sit in the after mean beside 5 post-index ones and the mean per call
    moves under 5%. `unchanged` is the honest verdict, `medium` the honest ceiling, and the
    note has to say so — a bare `unchanged` reads as "the work did nothing", which is a
    different claim from "the measurement cannot see it yet".
    """
    before, after = _load("unchanged_nested")
    verdict = _verdict(before, after)
    assert verdict.applied is True
    assert "Applied but unchanged" in verdict.note, verdict.note
    # Both means are real measurements here: this is not an absence being reported as a
    # non-result.
    assert verdict.mean_before is not None and verdict.mean_after is not None


def test_the_limit_truncated_golden_forbids_disappeared_and_names_the_mismatch():
    """Ruling 2, pinned against real artifacts: a group's absence from a *smaller* window is
    not evidence the group stopped running.

    This asserts the absence is genuinely what `verdicts` sees — no cited digest resolves in
    the after artifact — so the test cannot pass because the group happened to still be there.
    """
    before, after = _load("limit_truncated")
    proposal = _adv001(before)
    assert proposal is not None
    cited = proposal["evidence"]["fingerprint_digests"]
    assert not (set(cited) & set(group_index(after))), (
        "a cited digest still resolves in the after artifact, so this pair no longer "
        "exercises the absence at all"
    )
    assert before["window"]["limit"] != after["window"]["limit"]
    verdict = _verdict(before, after)
    assert verdict.outcome is VerifyOutcome.UNOBSERVABLE
    assert "--limit" in verdict.note and "sampling artifact" in verdict.note, verdict.note
    assert "no longer appear in the after run" not in verdict.note, verdict.note


def test_the_disappeared_golden_is_the_same_absence_with_the_limits_agreeing():
    """The control that makes the test above mean something: identical inputs but for the
    `--limit`, and here `DISAPPEARED` *is* graded. Without this pair, deleting the
    `may_be_sampling_artifact` gate would still leave the suite green, since every absence
    would simply grade `DISAPPEARED`."""
    before, after = _load("disappeared")
    proposal = _adv001(before)
    assert proposal is not None
    cited = proposal["evidence"]["fingerprint_digests"]
    assert not (set(cited) & set(group_index(after)))
    assert before["window"]["limit"] == after["window"]["limit"] is not None
    verdict = _verdict(before, after)
    assert verdict.outcome is VerifyOutcome.DISAPPEARED
    assert "no longer appear in the after run" in verdict.note, verdict.note


def test_the_regressed_golden_does_not_credit_the_index_for_the_workload_shift():
    """The index was applied and the mean per call still tripled, because the after workload
    binds a value matching 400,000 rows where the before workload bound a rare one. `verify`
    reports what it measured — `applied` yes, `regressed` — rather than excusing the
    regression because work was done, or crediting the index with a per-row gain nobody
    asked it about."""
    before, after = _load("regressed")
    verdict = _verdict(before, after)
    assert verdict.applied is True
    assert verdict.outcome is VerifyOutcome.REGRESSED
    assert verdict.mean_after is not None and verdict.mean_before is not None
    assert verdict.mean_after > verdict.mean_before


def test_the_not_applied_golden_reports_the_advice_untaken_whatever_the_timings_did():
    """`NOT_APPLIED` takes precedence over every mean-based outcome: nobody created the index,
    so nothing about the timings can be attributed to this proposal. Asserted together with
    the means being present, which is what makes the precedence visible rather than
    incidental — the branch is reached even though a graded comparison was available."""
    before, after = _load("not_applied")
    verdict = _verdict(before, after)
    assert verdict.applied is False
    assert verdict.outcome is VerifyOutcome.NOT_APPLIED
    assert verdict.mean_before is not None and verdict.mean_after is not None
    assert "not made" in verdict.note, verdict.note


# --- the goldens are, and stay, shapes `advise` emits --------------------------------------


#: `public.orders` as `information_schema.columns` returns it.
_COLUMNS = [
    ("public", "orders", "id", "integer"),
    ("public", "orders", "status", "text"),
    ("public", "orders", "created_at", "timestamp"),
]

#: One `pg_index` row in the 13-column shape the real `CAP_INDEXES` statement selects, so the
#: reference artifact's `physical_state` carries a populated `indexes` list rather than `[]`.
#: Deliberately on a column the hot statement does not filter, so the same single run yields
#: *both* a populated `indexes` list and a live ADV001 proposal — an index on `status` would
#: cover the predicate, ADV001 would not fire, and `proposals` would be empty, leaving the
#: proposal-key comparison below with nothing to compare against.
_INDEX_ROW = [
    (
        "public",
        "orders",
        "idx_orders_created_at",
        "created_at",
        1,
        False,
        False,
        42,
        8192,
        False,
        None,
        False,
        "CREATE INDEX idx_orders_created_at ON public.orders USING btree (created_at)",
    )
]


def _reference_artifact(monkeypatch) -> dict:
    """A fresh `advise --json` payload, from the production path over a stubbed querier.

    Only `connect()` is replaced — every layer from `fetch_workload` through `advise_payload`
    is the real one, exactly as `tests/test_advise_cli.py` and `tests/test_verify_cli.py` do
    it. This is the authority the goldens' shape is checked against: it is regenerated on
    every run, so it cannot itself go stale.
    """
    from sqlquality.workload.postgres import PostgresWorkloadAdapter

    rows: dict[str, object] = {
        "pg_stat_statements": [("select id from orders where status = $1", 100, 5000.0, 10)],
        "pg_stat_database": [("2026-07-01",)],
        "information_schema.columns": _COLUMNS,
        "pg_total_relation_size": [("public", "orders", 5_000_000, 10**8)],
        "pg_stats": [("public", "orders", "status", 5000.0)],
        "pg_index": _INDEX_ROW,
    }

    def fake_connect(self, params, timeout_s):
        def query(sql, bind):
            for marker, result in rows.items():
                if marker in sql:
                    return result
            return []

        self._query = query

    monkeypatch.setattr(PostgresWorkloadAdapter, "connect", fake_connect)
    result = runner.invoke(app, ["advise", "--dsn", "postgresql://u@h/db", "--json"])
    assert result.exit_code == 0, result.output
    return json.loads(result.stdout)


@pytest.mark.parametrize("name", [row[0] for row in _GOLDENS])
@pytest.mark.parametrize("side", ["before", "after"])
def test_every_golden_has_the_shape_advise_emits_today(name, side, monkeypatch):
    """Each committed artifact's key structure, against a freshly generated one.

    Checked key set by key set rather than by a hand-written list, so a key added to
    `advise_payload` (or to `window`, `analyzed`, `skipped`, a `query_groups` entry, a
    `physical_state` entry or an index entry) reddens here until the goldens are regenerated.
    That is the point: a golden that silently describes an older payload contract is a golden
    that stops pinning the thing it was committed for — and `verify`'s pre-0.4.0 refusal is
    built on exactly which keys an artifact carries.

    `dbt` is excluded from the comparison because it is the one key `advise_payload` omits
    entirely rather than emitting empty (no manifest was loaded for either artifact), and
    `proposals`' evidence keys are not compared: they are per-rule, and the reference run's
    single ADV001 cannot stand in for every rule a golden might contain.
    """
    payload = json.loads((FIXTURES / f"{name}.{side}.json").read_text())
    reference = _reference_artifact(monkeypatch)

    assert set(payload) == set(reference) - {"dbt"}
    for section in ("window", "analyzed", "skipped"):
        assert set(payload[section]) == set(reference[section]), section
    for group in payload["query_groups"]:
        assert set(group) == set(reference["query_groups"][0])
    reference_entry = next(iter(reference["physical_state"].values()))
    for entry in payload["physical_state"].values():
        assert set(entry) == set(reference_entry)
        for index in entry["indexes"] or ():
            assert set(index) == set(reference_entry["indexes"][0])
    for proposal in payload["proposals"]:
        assert set(proposal) == set(reference["proposals"][0])


# --- through the command itself -------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "outcome", "confidence"),
    [(row[0], row[2].value, row[3].value) for row in _GOLDENS],
    ids=[row[0] for row in _GOLDENS],
)
def test_the_command_reports_each_golden_pair_and_exits_0(name, outcome, confidence):
    """`sqlquality verify BEFORE AFTER` over the committed files, end to end and offline.

    The unit assertions above call `verdicts()`; this one goes through argument parsing, both
    file reads, every refusal, `verify_context` and the renderers. Content is asserted from
    `--json` rather than from the terminal table because rich wraps cells to the terminal
    width, and a width-dependent substring assertion is how a passing test stops meaning
    anything. The table path is exercised separately below.
    """
    result = runner.invoke(
        app,
        [
            "verify",
            str(FIXTURES / f"{name}.before.json"),
            str(FIXTURES / f"{name}.after.json"),
            "--json",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    ours = [v for v in payload["verdicts"] if v["code"] == "ADV001" and v["key"][2] == "orders"]
    assert len(ours) == 1, payload["verdicts"]
    assert ours[0]["outcome"] == outcome
    assert ours[0]["confidence"] == confidence


def test_the_terminal_table_and_the_notes_reach_the_user_for_a_real_pair():
    """The default surface, on real artifacts: the table on stdout, the caveats on stderr, and
    the note that carries the headline. Asserted on the nested pair, where the note is the
    whole answer — `unchanged` alone would read as "it did not work"."""
    result = runner.invoke(
        app,
        [
            "verify",
            str(FIXTURES / "unchanged_nested.before.json"),
            str(FIXTURES / "unchanged_nested.after.json"),
        ],
    )
    assert result.exit_code == 0, result.output
    stdout = _ANSI.sub("", result.stdout)
    assert "unchanged" in stdout
    assert "Applied but unchanged" in stdout
    assert "pg_stat_statements_reset()" in result.stderr
