"""Apply gate config to model deltas to produce a pass/fail verdict."""

from __future__ import annotations

from dataclasses import dataclass

from sqlquality.config import Config
from sqlquality.delta import ModelDelta


@dataclass(frozen=True)
class GateReport:
    deltas: list[ModelDelta]
    regressions: list[str]
    passed: bool


def evaluate_gate(deltas: list[ModelDelta], config: Config) -> GateReport:
    """Flag regressions (over threshold, not waived); fail only in 'fail' mode."""
    threshold = config.gate.max_complexity_increase
    waivers = set(config.waivers)
    regressions = [
        d.unique_id
        for d in deltas
        if d.unique_id not in waivers and d.delta > threshold
    ]
    passed = config.gate.mode != "fail" or not regressions
    return GateReport(deltas=deltas, regressions=regressions, passed=passed)
