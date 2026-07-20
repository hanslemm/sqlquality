"""Load sqlquality.yml into typed config (with defaults)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

_VALID_MODES = ("warn", "fail")


class ConfigError(ValueError):
    """Raised when sqlquality.yml is malformed or holds an invalid value."""


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
    try:
        data = yaml.safe_load(Path(path).read_text()) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Malformed YAML in {path}: {exc}") from exc

    defaults = GateConfig()
    gate_data = data.get("gate")
    if gate_data is None:  # null or absent -> defaults (e.g. a commented-out block)
        gate_data = {}
    if not isinstance(gate_data, dict):
        raise ConfigError(
            f"`gate` must be a mapping (got {type(gate_data).__name__}: {gate_data!r})"
        )

    mode = gate_data.get("mode", defaults.mode)
    if mode not in _VALID_MODES:
        raise ConfigError(f"`gate.mode` must be 'warn' or 'fail' (got {mode!r})")

    raw_threshold = gate_data.get("max_complexity_increase", defaults.max_complexity_increase)
    try:
        max_complexity_increase = float(raw_threshold)
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            f"`gate.max_complexity_increase` must be a number (got {raw_threshold!r})"
        ) from exc

    gate = GateConfig(mode=mode, max_complexity_increase=max_complexity_increase)

    raw_waivers = data.get("waivers") or ()
    if isinstance(raw_waivers, str):
        raw_waivers = [raw_waivers]
    waivers = tuple(raw_waivers)
    return Config(gate=gate, waivers=waivers)
