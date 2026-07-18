import copy
import json
from pathlib import Path

import pytest

from sqlquality.dbtproject import DbtProject
from sqlquality.delta import ModelDelta, compute_deltas

FIXTURE = Path(__file__).parent / "fixtures" / "manifest_v12.json"


def _candidate() -> DbtProject:
    return DbtProject.from_path(FIXTURE)


def _baseline_with_simpler_orders() -> DbtProject:
    manifest = json.loads(FIXTURE.read_text())
    manifest = copy.deepcopy(manifest)
    manifest["nodes"]["model.demo.orders"]["compiled_code"] = (
        'select customer_id from "dev"."main"."stg_orders" group by customer_id'
    )
    return DbtProject.from_manifest(manifest)


def test_positive_delta_against_baseline():
    deltas, skipped = compute_deltas(
        _baseline_with_simpler_orders(), _candidate(), ["model.demo.orders"], "postgres"
    )
    assert skipped == []
    d = deltas[0]
    assert d.unique_id == "model.demo.orders"
    assert d.is_new is False
    assert d.baseline == pytest.approx(10.7)
    assert d.candidate == pytest.approx(11.1)
    assert d.delta == pytest.approx(0.4)


def test_no_baseline_marks_new():
    deltas, skipped = compute_deltas(None, _candidate(), ["model.demo.orders"], "postgres")
    d = deltas[0]
    assert d.is_new is True
    assert d.baseline == 0.0
    assert d.delta == pytest.approx(d.candidate)


def test_skips_uncompiled_candidate():
    deltas, skipped = compute_deltas(
        None, _candidate(), ["seed.demo.raw_orders"], "postgres"
    )
    assert deltas == []
    assert skipped and skipped[0][0] == "seed.demo.raw_orders"
