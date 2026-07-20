"""Compute per-model complexity deltas between a baseline and candidate project."""

from __future__ import annotations

from dataclasses import dataclass

from sqlquality.complexity import ComplexityEngine
from sqlquality.dbtproject import DbtProject, DbtProjectError
from sqlquality.sqlast import SqlParseError, analyze_sql


@dataclass(frozen=True)
class ModelDelta:
    unique_id: str
    baseline: float
    candidate: float
    delta: float
    is_new: bool


_ENGINE = ComplexityEngine()


def _composite(project: DbtProject, uid: str, dialect: str) -> float | None:
    """Complexity composite for one model, or None if SQL missing/unparseable."""
    try:
        sql = project.compiled_sql(uid)
        metrics = analyze_sql(sql, dialect)
    except (DbtProjectError, SqlParseError):
        return None
    return _ENGINE.score(metrics, project.dag_facts(uid)).composite


def compute_deltas(
    baseline: DbtProject | None,
    candidate: DbtProject,
    changed_ids: list[str],
    dialect: str,
) -> tuple[list[ModelDelta], list[tuple[str, str]]]:
    """Score each changed model on candidate and baseline; return deltas + skips."""
    deltas: list[ModelDelta] = []
    skipped: list[tuple[str, str]] = []
    baseline_model_ids = set() if baseline is None else set(baseline.model_ids())
    for uid in changed_ids:
        cand = _composite(candidate, uid, dialect)
        if cand is None:
            skipped.append((uid, "no compiled SQL or unparseable in candidate"))
            continue
        if baseline is None:
            is_new = True
            base_value = 0.0
        else:
            base = _composite(baseline, uid, dialect)
            if base is None:
                # A node absent from the baseline is genuinely net-new. A node
                # *present* in the baseline but uncompiled/unparseable must NOT be
                # exempted as "new" — that would let a real regression through the
                # gate. Skip it instead of emitting a misleading delta.
                if uid in baseline_model_ids:
                    skipped.append((uid, "baseline present but unscoreable in baseline"))
                    continue
                is_new = True
                base_value = 0.0
            else:
                is_new = False
                base_value = base
        deltas.append(
            ModelDelta(
                unique_id=uid,
                baseline=base_value,
                candidate=cand,
                delta=round(cand - base_value, 1),
                is_new=is_new,
            )
        )
    return deltas, skipped
