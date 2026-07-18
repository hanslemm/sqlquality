"""Load sqlquality.yml into typed config (with defaults)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class GateConfig:
    mode: str = "warn"  # "warn" | "fail"
    max_complexity_increase: float = 10.0


@dataclass(frozen=True)
class Config:
    gate: GateConfig = field(default_factory=GateConfig)
    waivers: tuple[str, ...] = ()


def load_config(path: Path | None) -> Config:
    """Load config from YAML, or return defaults if absent."""
    if path is None or not Path(path).exists():
        return Config()
    data = yaml.safe_load(Path(path).read_text()) or {}
    gate_data = data.get("gate")
    if not isinstance(gate_data, dict):
        gate_data = {}
    defaults = GateConfig()
    gate = GateConfig(
        mode=gate_data.get("mode", defaults.mode),
        max_complexity_increase=float(
            gate_data.get("max_complexity_increase", defaults.max_complexity_increase)
        ),
    )
    raw_waivers = data.get("waivers") or ()
    if isinstance(raw_waivers, str):
        raw_waivers = [raw_waivers]
    waivers = tuple(raw_waivers)
    return Config(gate=gate, waivers=waivers)
