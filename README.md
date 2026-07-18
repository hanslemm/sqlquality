# sqlquality

Measure the **performance** and **complexity** of dbt models, and gate a change
on whether it improved or worsened them — considering the model and its direct
upstream/downstream neighbors, with per-database best-practice suggestions.

> Status: early. Plan 1 ships an offline SQL structural-complexity scorer:
> `sqlquality complexity path/to/model.sql`.

## Install (dev)

```bash
uv sync
uv run sqlquality --version
```

## Surfaces

### Pre-commit hook

In your dbt repo's `.pre-commit-config.yaml`:

````yaml
repos:
  - repo: https://github.com/hanslemm/sqlquality
    rev: v0.1.0
    hooks:
      - id: sqlquality-lint
```
````

### CI change-impact gate

Run the complexity delta gate on a PR and post the result as a comment:

````yaml
- run: uv run sqlquality check --project-dir . --state prod-artifacts/ --markdown report.md
- uses: actions/github-script@v7   # or any PR-comment action
  with:
    script: |
      const body = require('fs').readFileSync('report.md', 'utf8')
      github.rest.issues.createComment({ ...context.repo, issue_number: context.issue.number, body })
```
````
