# Defense Notes

This file summarizes the methodological choices, validation logic, limitations, and interview-style explanations for the project. It is intended to make the project easier to review, discuss, and defend professionally.

## Project In One Sentence

This project benchmarks classical, neural, and hybrid mortality forecasting models to understand when neural networks are useful for actuarial mortality prediction and when simpler actuarial baselines remain preferable.

## Why This Project Matters

Mortality forecasting affects life insurance pricing, annuity valuation, pension liabilities, longevity risk, and Solvency II capital calculations. In practice, actuaries need models that are not only accurate, but also stable, interpretable, reproducible, and appropriate for the business decision.

Neural networks can capture nonlinear age-time patterns, but they can also overfit and become difficult to explain. This project studies that trade-off through a reproducible benchmark instead of assuming that model complexity automatically improves actuarial forecasting.

## Data Source

The project uses mortality data from the Human Mortality Database (HMD), a standard source for demographic and actuarial mortality research.

Key points:

- The benchmark uses multiple national populations rather than a single country.
- The data include yearly mortality rates and exposures by age and year.
- The implementation works with ages 0 to 100.
- HMD raw data are not redistributed in the repository because the HMD licence requires users to download the data directly.
- The repository provides scripts and documentation so users with HMD access can reproduce the workflow.

## Models Used

### Classical And Baseline Models

- Lee-Carter
- Lee-Miller
- Poisson Lee-Carter
- Cairns-Blake-Dowd for older ages
- Hyndman-Ullah-style functional model
- Age-specific random walk with drift
- Frozen rates baseline

### Neural Models

- LSTM on mortality-index dynamics
- GRU on mortality-index dynamics
- Bi-LSTM on mortality-index dynamics
- Transformer-style sequence model
- Feedforward neural network with embeddings
- CNN on mortality surfaces

### Hybrid Model

- LC-ResNet

LC-ResNet combines a Poisson Lee-Carter backbone with a small neural residual correction. The neural correction is shrunk as the forecast horizon increases, so the model remains closer to the stable actuarial structure at long horizons.

## Evaluation Protocol

The benchmark uses rolling-origin validation.

Simple explanation:

1. Train each model using only the historical mortality data available up to a given year.
2. Forecast future mortality rates.
3. Compare the forecasts against observed future values.
4. Repeat the process across countries, origins, horizons, and scenarios.

This is more appropriate than a single random train-test split because mortality forecasting is a time-series problem. At each valuation date, a real actuary only has access to past data.

## Metrics

The benchmark uses both statistical and actuarial metrics.

Statistical metrics:

- RMSE on log mortality rates
- Horizon-specific forecast error

Actuarial metrics:

- Life-expectancy error
- Annuity-factor error
- EUR impact in a French annuity portfolio case study
- Sensitivity under a Solvency II longevity shock

This matters because the statistically best model is not always the best actuarial model. A model can perform well on global log-rate accuracy but still be less appropriate for annuity valuation or longevity-risk analysis.

## Main Results To Defend

- The age-specific random walk with drift is a very strong benchmark for full-grid log-rate RMSE.
- LC-ResNet performs well at short horizons among structured models.
- Classical models remain competitive, especially when interpretability and long-horizon stability matter.
- Neural models do not dominate universally.
- The best model depends on the objective: log-rate accuracy, life expectancy, annuity valuation, shock robustness, or interpretability.
- The French annuity case study shows that model choice can create material financial differences.

## Practical Interpretation

The main actuarial message is:

Model complexity should not replace objective-specific validation.

An actuary should choose the model based on:

- forecast horizon
- age range
- length and quality of historical data
- shock environment
- need for interpretability
- final business use, such as pricing, reserving, capital, or risk analysis

## Limitations

Important limitations:

- The benchmark is based on a specific data design and a finite set of countries.
- The results should not be interpreted as universal model rankings.
- Neural models may require more tuning and more data than classical models.
- Predictive intervals and full probabilistic validation are limited.
- HMD national-population mortality can differ from insured-portfolio mortality.
- The project focuses on reproducible comparison, not on proving a new universal mortality model.

## What I Would Improve Next

Possible extensions:

- Add predictive intervals for all models.
- Add more countries and population subgroups.
- Compare national mortality with insured-portfolio mortality if data are available.
- Add Bayesian model averaging.
- Improve the neural hyperparameter search.
- Add formal pairwise significance tests.
- Build a fuller Streamlit dashboard for actuarial users.
- Extend the case study to pricing, reserving, and capital under multiple portfolios.

## Questions I Should Be Able To Answer

### Why compare classical and neural models?

Because actuarial model choice is not only about accuracy. It also involves stability, interpretability, regulatory acceptability, and sensitivity to shocks. Neural networks may help in some settings, but they must be compared against strong classical baselines.

### Why use Lee-Carter?

Lee-Carter is a standard mortality forecasting model. It provides a transparent structure based on age effects, a time mortality index, and age-specific sensitivity to that index.

### Why use rolling-origin validation?

Because mortality forecasting is temporal. Rolling-origin validation better simulates real forecasting, where only past data are available at each valuation date.

### Why include simple baselines?

Simple baselines are essential. If a complex neural model cannot beat a simple random walk or frozen-rate benchmark, then the additional complexity is not justified.

### Why can a model perform well on log-rate RMSE but not on actuarial metrics?

Because actuarial quantities such as annuity values or life expectancy weight ages and horizons differently. A small error at a financially important older age can matter more than a larger error at a less relevant age.

### What is the idea behind LC-ResNet?

LC-ResNet keeps the actuarial structure of Poisson Lee-Carter while allowing a neural network to learn residual corrections. The correction is reduced at longer horizons to avoid unstable neural extrapolation.

### What is the biggest lesson?

The best mortality model depends on the actuarial objective. Neural networks are useful tools, but they should be validated against strong classical baselines and interpreted with care.

## AI Assistance And Validation

AI tools were used to accelerate coding, drafting, debugging, documentation, and manuscript refinement.

The project remains the author's responsibility. The actuarial methodology, data-source choice, validation design, result interpretation, and limitations should be understood, checked, and defended by the author. The important point is not whether every line was typed manually, but whether the methodology is understood and the results are verified honestly.

## Reproduction Checklist

Before presenting this project, I should be able to:

- explain Lee-Carter in simple words
- explain why HMD data were used
- explain the difference between classical, neural, and hybrid models
- explain rolling-origin validation
- explain the main results without reading the paper
- explain why neural networks do not always win
- explain the annuity case study
- explain the project limitations
- run the main scripts or explain the reproduction workflow
- show where the results are stored in the repository

## Key Files In The Repository

- `paper/main.pdf`: research paper / working manuscript
- `paper/main.tex`: LaTeX source for the paper
- `README.md`: project overview and reproduction instructions
- `scripts/run_benchmark.py`: benchmark entry point
- `scripts/run_case_study.py`: French annuity case-study entry point
- `results/benchmark.csv`: stored benchmark results
- `results/case_study.csv`: stored case-study results
- `src/mortality/`: implementation of data loading, models, evaluation, actuarial calculations, and visualization
