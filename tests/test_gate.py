from sqlquality.config import Config, GateConfig
from sqlquality.delta import ModelDelta
from sqlquality.gate import evaluate_gate

BIG = ModelDelta("m.big", baseline=5.0, candidate=20.0, delta=15.0, is_new=False)
SMALL = ModelDelta("m.small", baseline=5.0, candidate=8.0, delta=3.0, is_new=False)


def test_regression_flagged_but_warn_passes():
    r = evaluate_gate([BIG, SMALL], Config(gate=GateConfig(mode="warn")))
    assert r.regressions == ["m.big"]
    assert r.passed is True


def test_fail_mode_blocks_on_regression():
    r = evaluate_gate([BIG], Config(gate=GateConfig(mode="fail")))
    assert r.passed is False


def test_waiver_clears_regression():
    r = evaluate_gate(
        [BIG], Config(gate=GateConfig(mode="fail"), waivers=("m.big",))
    )
    assert r.regressions == []
    assert r.passed is True


def test_under_threshold_passes_in_fail_mode():
    r = evaluate_gate([SMALL], Config(gate=GateConfig(mode="fail")))
    assert r.regressions == []
    assert r.passed is True


def test_delta_equal_to_threshold_is_not_regression():
    edge = ModelDelta("m.edge", baseline=5.0, candidate=15.0, delta=10.0, is_new=False)
    r = evaluate_gate([edge], Config(gate=GateConfig(mode="fail")))  # default threshold 10.0
    assert r.regressions == []
    assert r.passed is True
