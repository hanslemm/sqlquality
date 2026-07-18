"""Core scoring logic for `sqlquality check` (no CLI / no subprocess)."""

from __future__ import annotations

from dataclasses import dataclass

from sqlquality.complexity import ComplexityEngine
from sqlquality.dbtproject import DbtProject, DbtProjectError
from sqlquality.models import DagFacts
from sqlquality.sqlast import SqlParseError, analyze_sql


@dataclass(frozen=True)
class CheckResult:
    unique_id: str
    composite: float
    dag: DagFacts


def run_check(
    project: DbtProject, changed_ids: list[str], dialect: str
) -> tuple[list[CheckResult], list[tuple[str, str]]]:
    """Score each changed model's complexity; skip missing/unparseable SQL."""
    engine = ComplexityEngine()
    results: list[CheckResult] = []
    skipped: list[tuple[str, str]] = []
    for uid in changed_ids:
        try:
            sql = project.compiled_sql(uid)
            metrics = analyze_sql(sql, dialect)
        except (DbtProjectError, SqlParseError) as exc:
            skipped.append((uid, str(exc)))
            continue
        dag = project.dag_facts(uid)
        score = engine.score(metrics, dag)
        results.append(CheckResult(unique_id=uid, composite=score.composite, dag=dag))
    return results, skipped
