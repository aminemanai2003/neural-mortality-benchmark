# Neural Mortality Benchmark

[![CI](https://github.com/aminemanai2003/neural-mortality-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/aminemanai2003/neural-mortality-benchmark/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**When should actuaries trust neural networks for mortality forecasting?**

A reproducible comparison of 14 mortality-forecasting implementations, including a proposed hybrid (LC-ResNet) and an actuarial case study that translates model risk into EUR.

---

## Author and institutions

**Author:** Amine Manai

**Associated institutions:**
- ESPRIT School of Engineering, Tunisia
- Institut du Risque et de l'Assurance (IRA), Le Mans Universite, France — incoming M1 Actuariat student, 2026-2027

## Highlights

- **14 implementations**: 5 classical structural models + 2 simple baselines + 6 neural models + 1 hybrid
- **8 countries** from the Human Mortality Database (1950–2023, including COVID)
- **Rolling-origin validation** with log-rate and age-100-truncated actuarial metrics (e₀, annuity ä₆₅)
- **Objective-specific guidance** based on data length, horizon, age group, and transparency constraints
- **LC-ResNet**: Poisson Lee-Carter skeleton + neural residual correction with horizon shrinkage
- **Reproducible case study**: 1,000 French pensioners, 2033 rates, Solvency II longevity shock
- **Interactive dashboard** (Streamlit)

## Models

| Structured / baseline | Neural (PyTorch) | Hybrid |
|---|---|---|
| Lee-Carter (SVD) | LSTM on κ_t | **LC-ResNet** |
| Lee-Miller | GRU on κ_t | |
| Poisson Lee-Carter | Bi-LSTM on κ_t | |
| CBD (ages 60–100) | Transformer on κ_t | |
| H-U-style functional model | FFNN with embeddings | |
| Random walk with drift | CNN on mortality surface | |
| Frozen rates | | |

The H-U implementation uses spline smoothing, six functional principal components, and random-walk-with-drift score forecasts; it is a simplified H-U-style model rather than the full robust automatic-ARIMA procedure.

## Main findings

- The age-specific random walk with drift has the lowest full-grid log-rate RMSE at every tested horizon.
- LC-ResNet has the lowest one-year error among structured models, but Lee-Miller is better at 20 years.
- H-U has the lowest 20-year life-expectancy RMSE; the FFNN has the lowest 20-year annuity-factor RMSE.
- Frozen rates lead the cumulative 2020–2022 COVID stress test and the 20-year-history scenario.
- The five-model French case study produces a **€6.26M provision spread**; among the four full-grid models, the spread is €1.22M.

These are results for this data design and loss functions—not universal model rankings.

## Key Questions Answered

- Which model works best with **short historical data** (20 vs 50 years)?
- Which is most **robust to mortality shocks** (COVID)?
- Which performs best at **5, 10, and 20-year horizons**?
- Which works best for **elderly ages (65+)** — the longevity risk segment?
- When is the **extra complexity of neural networks justified**?
- What is the **model risk in EUR** on an annuity portfolio?

## Quickstart

```bash
# 1. Install
pip install -e ".[dev]"

# 2. Download HMD data (requires free account at mortality.org)
#    Set HMD_USERNAME and HMD_PASSWORD in .env
python scripts/download_hmd.py

# 3. Run benchmark (quick mode: 1 country, reduced origins)
python scripts/run_benchmark.py --countries FRATNP --quick

# 4. Reproduce the French 2033 case study
python scripts/run_case_study.py

# 5. Launch dashboard
streamlit run dashboard/app.py
```

## Data

Uses the [Human Mortality Database](https://www.mortality.org/) (8 countries: France, England & Wales, USA, Japan, Italy, Spain, Sweden, Netherlands).

HMD data is **not redistributable** — register for free at [mortality.org](https://www.mortality.org/), then set your credentials in `.env`:

```
HMD_USERNAME=your@email.com
HMD_PASSWORD=yourpassword
```

## Project Structure

```
mortality-benchmark/
├── src/mortality/
│   ├── data/           # HMD loader and preprocessing
│   ├── models/
│   │   ├── classical/  # Lee-Carter, Lee-Miller, Poisson LC, CBD, H-U, baselines
│   │   ├── neural/     # LSTM, GRU, BiLSTM, Transformer, FFNN, CNN
│   │   └── hybrid/     # LC-ResNet (our contribution)
│   ├── evaluation/     # Rolling-origin, metrics, scenarios, decision framework
│   ├── actuarial/      # Life tables, e₀, annuities ä₆₅, Solvency II shock
│   └── viz/            # Heatmaps, fan charts, result tables
├── scripts/            # benchmark, case-study, and HMD download entry points
├── dashboard/          # Streamlit interactive app
├── report/             # French report + CV kit
├── config/             # YAML configs (data, models, evaluation)
├── tests/              # pytest test suite
└── results/            # Benchmark output CSV + figures
```

## The LC-ResNet Hybrid Model

The proposed hybrid combines:

1. A **Poisson Lee-Carter skeleton** (aₓ, bₓ, κ_t) — transparent and stable at long horizons
2. A **small residual neural network** that learns nonlinear age-time residual structure
3. **Horizon-dependent shrinkage**: the neural correction is multiplied by exp(−λh), so at long horizons the model reverts to stable LC extrapolation

The LC skeleton remains reportable, but the neural correction is opaque; LC-ResNet is therefore only partially interpretable.

## Manuscript and reproducibility

- Manuscript: [`paper/main.pdf`](paper/main.pdf)
- Stored benchmark rows: [`results/benchmark.csv`](results/benchmark.csv)
- Case-study output: [`results/case_study.csv`](results/case_study.csv)

Each horizon-`h` result is an RMSE over the complete forecast path from year 1 through year `h`, averaged without weighting over valid country–origin pairs. Age-band scenarios refit each model within the band. HMD data are not included because their licence does not permit redistribution.

## References

1. Lee & Carter (1992). *Modeling and forecasting U.S. mortality.* JASA.
2. Brouhns, Denuit & Vermunt (2002). *A Poisson log-bilinear regression approach.* IME.
3. Cairns, Blake & Dowd (2006). *A two-factor model for stochastic mortality.* Journal of Risk and Insurance.
4. Hyndman & Ullah (2007). *Robust forecasting of mortality and fertility rates.* CSDA.
5. Richman & Wüthrich (2021). *A neural network extension of the Lee-Carter model.* Annals of Actuarial Science.
6. Barigou et al. (2023). *Bayesian model averaging for mortality forecasting using leave-future-out validation.* IJF.
7. Li, Li & Panagiotelis (2025). *Boosting domain-specific models with shrinkage.* IJF.

## License

MIT — Amine Manai, 2026
