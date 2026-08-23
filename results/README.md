# Stored result artifacts

The checked-in files are an archival snapshot of the benchmark described in the working paper.

## `benchmark.csv`

One row represents a model, country, rolling origin, horizon and metric combination.

| Column | Meaning |
|---|---|
| `model_name` | Implementation identifier |
| `country` | Human Mortality Database population code |
| `sex` | Population sex; the stored benchmark uses `Total` |
| `origin` | Last training year |
| `horizon` | Number of forecast years evaluated |
| `metric` | Statistical, actuarial or scenario-specific error |
| `value` | Metric value for the country-origin pair |
| `train_time` | Model fit time in seconds for that run |

The file contains 27,312 rows. Aggregated results in the paper are unweighted means over valid country-origin pairs. CBD only covers ages 60–100 and is not directly comparable with full-grid log-rate results.

## `case_study.csv`

Illustrative 2033 valuation of 1,000 French annuitants aged 65, with annual payments of €12,000 and a 2% discount rate. `longevity_stress_impact_eur` is the increase in the modelled liability after multiplying mortality rates by 0.80. It is not a complete insurer-level SCR.

## Regeneration

```bash
uv run python scripts/run_benchmark.py
uv run python scripts/run_case_study.py
uv run python scripts/plot_summary.py
```

Raw HMD files are not included. The exact retrieval date of the snapshot behind these stored results was not preserved, so later HMD revisions can produce small differences. New downloads create an untracked `data/raw/download_manifest.json` with timestamps and SHA-256 hashes.
