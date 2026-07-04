# Neural Mortality Benchmark

[![CI](https://github.com/aminemanai2003/neural-mortality-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/aminemanai2003/neural-mortality-benchmark/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**When should actuaries trust neural networks for mortality forecasting?**

A comprehensive benchmark of 14 classical and neural mortality models with a practical decision framework, an original hybrid model (LC-ResNet), and an actuarial case study quantifying model risk in EUR.

---

## Highlights

- **14 models** compared: 8 classical (implemented from scratch) + 6 neural (PyTorch)
- **8 countries** from the Human Mortality Database (1950–2023, including COVID)
- **Rolling-origin validation** with actuarial metrics (e₀, annuity ä₆₅)
- **Decision framework**: "which model to use when" based on data length, horizon, age group, interpretability
- **Original hybrid model (LC-ResNet)**: Lee-Carter Poisson skeleton + neural residual correction with horizon shrinkage
- **Actuarial case study**: annuity pricing for 1,000 pensioners, Solvency II longevity shock, model risk in EUR
- **Interactive dashboard** (Streamlit)

## Models

| Classical (from scratch) | Neural (PyTorch) | Hybrid (ours) |
|---|---|---|
| Lee-Carter (SVD) | LSTM on κ_t | **LC-ResNet** |
| Lee-Miller | GRU on κ_t | |
| Booth-Maindonald-Smith | Bi-LSTM on κ_t | |
| Poisson Lee-Carter | Transformer on κ_t | |
| CBD (Cairns-Blake-Dowd) | FFNN with embeddings | |
| Hyndman-Ullah (FDA) | CNN on mortality surface | |
| Random walk with drift | | |
| Frozen rates | | |

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

# 4. Launch dashboard
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
│   │   ├── classical/  # Lee-Carter, Lee-Miller, BMS, Poisson LC, CBD, H-U, baselines
│   │   ├── neural/     # LSTM, GRU, BiLSTM, Transformer, FFNN, CNN
│   │   └── hybrid/     # LC-ResNet (our contribution)
│   ├── evaluation/     # Rolling-origin, metrics, Diebold-Mariano, scenarios, decision framework
│   ├── actuarial/      # Life tables, e₀, annuities ä₆₅, Solvency II shock
│   └── viz/            # Heatmaps, fan charts, result tables
├── scripts/            # run_benchmark.py, download_hmd.py
├── dashboard/          # Streamlit interactive app
├── report/             # French report + CV kit
├── config/             # YAML configs (data, models, evaluation)
├── tests/              # pytest test suite
└── results/            # Benchmark output CSV + figures
```

## The LC-ResNet Hybrid Model

Our original contribution combines:

1. A **Poisson Lee-Carter skeleton** (aₓ, bₓ, κ_t) — interpretable, stable at long horizons
2. A **small residual neural network** that learns the structured residuals (non-linearities, cohort effects)
3. **Horizon-dependent shrinkage**: the neural correction is multiplied by exp(−λh), so at long horizons the model reverts to stable LC extrapolation

This preserves interpretability while capturing patterns that LC misses at short/medium horizons.

## References

1. Lee & Carter (1992). *Modeling and forecasting U.S. mortality.* JASA.
2. Brouhns, Denuit & Vermunt (2002). *A Poisson log-bilinear regression approach.* IME.
3. Cairns, Blake & Dowd (2006). *A two-factor model for stochastic mortality.* NAAJ.
4. Hyndman & Ullah (2007). *Robust forecasting of mortality and fertility rates.* CSDA.
5. Richman & Wüthrich (2019). *A neural network extension of the Lee-Carter model.* AAS.
6. Perla et al. (2021). *A brief review of deep learning methods in mortality forecasting.* AAS.

## License

MIT — Amine Manai, 2026
