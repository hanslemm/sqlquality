"""Best-practice linting + auto-fix via SQLFluff's programmatic API."""

from __future__ import annotations

import sqlfluff

from sqlquality.models import Finding, Severity

# Templating-caused codes: raised when SQLFluff's Jinja templater can't resolve a
# macro (dbt_utils.*, custom macros). TMP is the undefined-variable error; PRS is
# the unparsable section that follows. On a raw dbt model these are advisory only.
_TEMPLATING_CODES = frozenset({"TMP", "PRS"})
_JINJA_HINT = (
    " (unresolved Jinja — lint the compiled SQL under target/compiled/ "
    "or pass a dbt-templater config via --sqlfluff-config)"
)


def _to_finding(violation: dict, *, templating: bool) -> Finding:
    code = violation.get("code", "")
    message = violation.get("description", "")
    if templating and code in _TEMPLATING_CODES:
        # Unresolvable-Jinja noise the user can't fix on the raw model: advise, never gate.
        severity = Severity.INFO
        message = message + _JINJA_HINT
    elif code == "PRS":
        # Genuine parse error on plain SQL.
        severity = Severity.ERROR
    else:
        severity = Severity.WARNING
    # PRS (parse-error) dicts have no "fixes" key — always use .get().
    fixes = violation.get("fixes") or []
    return Finding(
        code=code,
        message=message,
        line=violation.get("start_line_no", 0),
        severity=severity,
        fixable=bool(fixes),
    )


def lint_sql(
    sql: str,
    dialect: str,
    exclude_rules: list[str] | None = None,
    config_path: str | None = None,
) -> list[Finding]:
    """Lint one SQL string; return findings (parse errors included as PRS).

    A file whose violations include a TMP code is treated as unresolved-templating:
    its TMP/PRS findings are demoted to INFO. TMP is the only reliable evidence that
    templating failed — a genuine PRS on rendered SQL (Jinja only in comments, valid
    Jinja that renders to broken SQL, templater=raw configs) carries no TMP and stays
    ERROR.
    """
    violations = sqlfluff.lint(
        sql, dialect=dialect, exclude_rules=exclude_rules, config_path=config_path
    )
    templating = any(v.get("code") == "TMP" for v in violations)
    return [_to_finding(v, templating=templating) for v in violations]


def fix_sql(
    sql: str,
    dialect: str,
    exclude_rules: list[str] | None = None,
    config_path: str | None = None,
) -> str:
    """Return SQL with SQLFluff auto-fixes applied (unchanged if unparseable)."""
    return sqlfluff.fix(sql, dialect=dialect, exclude_rules=exclude_rules, config_path=config_path)
