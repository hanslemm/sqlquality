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
