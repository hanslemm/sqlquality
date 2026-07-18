"""Best-practice linting + auto-fix via SQLFluff's programmatic API."""

from __future__ import annotations

import sqlfluff

from sqlquality.models import Finding, Severity


def _to_finding(violation: dict) -> Finding:
    code = violation.get("code", "")
    severity = Severity.ERROR if code == "PRS" else Severity.WARNING
    # PRS (parse-error) dicts have no "fixes" key — always use .get().
    fixes = violation.get("fixes") or []
    return Finding(
        code=code,
        message=violation.get("description", ""),
        line=violation.get("start_line_no", 0),
        severity=severity,
        fixable=bool(fixes),
    )


def lint_sql(
    sql: str, dialect: str, exclude_rules: list[str] | None = None
) -> list[Finding]:
    """Lint one SQL string; return findings (parse errors included as PRS)."""
    violations = sqlfluff.lint(sql, dialect=dialect, exclude_rules=exclude_rules)
    return [_to_finding(v) for v in violations]


def fix_sql(sql: str, dialect: str, exclude_rules: list[str] | None = None) -> str:
    """Return SQL with SQLFluff auto-fixes applied (unchanged if unparseable)."""
    return sqlfluff.fix(sql, dialect=dialect, exclude_rules=exclude_rules)
