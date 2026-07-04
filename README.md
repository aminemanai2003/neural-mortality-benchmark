# Neural Mortality Benchmark

**When should actuaries trust neural networks for mortality forecasting?**

A comprehensive benchmark of classical and neural Lee-Carter models with a practical decision framework and an original hybrid model.

## Models

| Classical | Neural |
|---|---|
| Lee-Carter (SVD) | LSTM Lee-Carter |
| Lee-Miller | GRU Lee-Carter |
| Booth-Maindonald-Smith | Bi-LSTM Lee-Carter |
| Poisson Lee-Carter | Transformer Lee-Carter |
| CBD (Cairns-Blake-Dowd) | FFNN with embeddings |
| Hyndman-Ullah (FDA) | CNN surface model |
| Random walk with drift | **LC-ResNet** (hybrid, ours) |

## Key Questions Answered

- Which model works best with short historical data (20 vs 50 years)?
- Which is most robust to mortality shocks (COVID)?
- Which performs best at 5, 10, and 20-year horizons?
- Which works best for elderly ages (65+) — the longevity risk segment?
- When is neural network complexity justified?

## Data

Uses the [Human Mortality Database](https://www.mortality.org/) (8 countries, 1950–2023).
HMD data is not redistributable — register (free) and download with:

```bash
cp .env.example .env  # fill in HMD_USERNAME and HMD_PASSWORD
python scripts/download_hmd.py
```

## Quickstart

```bash
pip install -e ".[dev]"
python scripts/download_hmd.py
python scripts/run_benchmark.py --countries FRATNP --quick
streamlit run dashboard/app.py
```

## Project Structure

```
src/mortality/
├── data/          # HMD loader and preprocessing
├── models/        # Classical, neural, and hybrid models
├── evaluation/    # Rolling-origin, metrics, Diebold-Mariano
├── actuarial/     # Life tables, e0, annuities, Solvency II shocks
└── viz/           # Heatmaps, fan charts, result tables
```

## License

MIT
