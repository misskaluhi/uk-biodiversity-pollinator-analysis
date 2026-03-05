# UK Biodiversity & Pollinator Analysis

A machine learning pipeline that predicts pollinator decline in the UK using Genetic Algorithm-optimised feature selection and Random Forest regression.

Built as a proof-of-concept AI application for **"Bee Positive"**, an NGO focused on supporting pollinator populations.

## Overview

This project analyses five UK Biodiversity Indicator datasets (JNCC 2025 release) to model and predict trends in pollinating insect populations. The pipeline:

1. **Preprocesses and merges** 5 datasets with different time ranges (1976–2024)
2. **Uses a Genetic Algorithm** to optimise feature selection, year alignment, and interpolation strategy
3. **Trains a Random Forest** model to predict the pollinator occupancy index
4. **Evaluates performance** with R², RMSE, MAE, and cross-validation
5. **Generates visualisations** and summary reports

## Datasets

| Dataset | Source | Years | Role |
|---------|--------|-------|------|
| Pollinating Insects | JNCC | 1980–2024 | **Target variable** (mean occupancy index) |
| Butterflies (Insects of Wider Countryside) | JNCC | 1976–2024 | Feature – insect ecosystem health proxy |
| Plants of Wider Countryside | JNCC | 2015–2024 | Feature – plant food sources for pollinators |
| Agri-Environment Schemes | JNCC | 1992–2022 | Feature – conservation intervention measure |
| Habitat Connectivity | JNCC | 1985–2012 | Feature – habitat fragmentation measure |

## Results

| Metric | Value |
|--------|-------|
| Training R² | 0.9972 |
| Test R² | **0.9928** |
| Cross-validation R² | **0.9815 ± 0.0130** |
| Test RMSE | 0.9557 |
| Test MAE | 0.7964 |

### GA-Optimised Configuration
- **Year range:** 1983–2023
- **Selected features:** Butterfly All Species, Butterfly Generalists, Habitat Connectivity, Year
- **Interpolation:** Spline

### Key Findings
- The UK pollinator index has declined from 100 (1980 baseline) to approximately 77.3 in 2024
- Year (temporal trend) is the most important predictor, followed by butterfly generalist abundance
- Habitat connectivity contributes meaningfully despite its limited data range (1985–2012)

## Project Structure

```
├── code/
│   ├── data_preprocessing.py    # Data loading, cleaning, merging
│   ├── genetic_algorithm.py     # GA for feature/year/interpolation optimisation
│   ├── random_forest_model.py   # RF training, evaluation, metrics
│   ├── main.py                  # Complete pipeline orchestrator
│   └── requirements.txt         # Python dependencies
├── data/
│   ├── cleaned_merged_dataset.csv
│   ├── ga_optimization_results.csv
│   ├── model_predictions.csv
│   └── feature_importance.csv
├── figures/
│   ├── pollinator_trend.png
│   ├── feature_correlation_heatmap.png
│   ├── ga_fitness_evolution.png
│   ├── feature_importance_plot.png
│   ├── predictions_vs_actual.png
│   └── residuals_plot.png
└── results_summary.txt
```

## How to Run

```bash
# Install dependencies
pip install -r code/requirements.txt

# Run the full pipeline
python code/main.py
```

> **Note:** The raw Excel datasets (UK-BDI-2025-*.xlsx) should be placed in `~/Downloads/` before running. These are available from [JNCC UK Biodiversity Indicators](https://jncc.gov.uk/our-work/uk-biodiversity-indicators/).

## Technologies

- **Python 3.12**
- **pandas** / **NumPy** – data manipulation
- **scikit-learn** – Random Forest, cross-validation, metrics
- **matplotlib** / **seaborn** – visualisation
- **Custom Genetic Algorithm** – binary-encoded chromosome optimisation
