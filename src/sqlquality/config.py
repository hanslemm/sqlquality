"""Load sqlquality.yml into typed config (with defaults)."""

from __future__ import annotations

import math
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
        raw_text = Path(path).read_text()
    except OSError as exc:  # e.g. --config points at a directory
        raise ConfigError(f"Could not read config {path}: {exc}") from exc
    try:
        data = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Malformed YAML in {path}: {exc}") from exc
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(f"top-level of {path} must be a mapping (got {type(data).__name__})")

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
    # Reject bools up front: bool is an int subclass, so float(True) == 1.0 would
    # silently accept `max_complexity_increase: true` as a threshold of 1.0.
    if isinstance(raw_threshold, bool):
        raise ConfigError(
            f"`gate.max_complexity_increase` must be a number (got bool {raw_threshold!r})"
        )
    try:
        max_complexity_increase = float(raw_threshold)
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            f"`gate.max_complexity_increase` must be a number (got {raw_threshold!r})"
        ) from exc
    # NaN/inf would make every `delta > threshold` comparison False, silently
    # turning a fail-mode gate into a no-op.
    if not math.isfinite(max_complexity_increase):
        raise ConfigError(
            f"`gate.max_complexity_increase` must be a finite number (got {raw_threshold!r})"
        )

    gate = GateConfig(mode=mode, max_complexity_increase=max_complexity_increase)

    raw_waivers = data.get("waivers") or ()
    if isinstance(raw_waivers, str):
        raw_waivers = [raw_waivers]
    waivers = tuple(raw_waivers)
    return Config(gate=gate, waivers=waivers)
