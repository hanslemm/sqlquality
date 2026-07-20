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
    mode: str = "warn"

    @property
    def warned(self) -> bool:
        """True when warn mode let regressions through (passed, but not clean)."""
        return self.mode == "warn" and bool(self.regressions)


def evaluate_gate(deltas: list[ModelDelta], config: Config) -> GateReport:
    """Flag regressions (over threshold, not waived); fail only in 'fail' mode."""
    threshold = config.gate.max_complexity_increase
    waivers = set(config.waivers)
    # A brand-new model (no baseline) has delta == its full composite; a *delta*
    # gate does not treat net-new surface as a regression. Absolute-complexity
    # gating of new models is a separate (future) feature.
    regressions = [
        d.unique_id
        for d in deltas
        if not d.is_new and d.unique_id not in waivers and d.delta > threshold
    ]
    passed = config.gate.mode != "fail" or not regressions
    return GateReport(deltas=deltas, regressions=regressions, passed=passed, mode=config.gate.mode)
