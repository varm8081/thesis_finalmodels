# Thesis Final Models — Financial Distress Prediction (Tehran Stock Exchange)

Predicting next-year **cash-flow distress zone (Class 0)** for companies listed on
the Tehran Stock Exchange, using financial ratios, Altman Z-scores, and temporal
features. This repository holds a clean, leakage-free ML pipeline rebuilt on macOS.

## Project goal
Primary target: `ocf_ratio_zone_next_year`
- **Class 0 — Distress zone** (PRIMARY FOCUS)
- Class 1 — Gray zone
- Class 2 — Safe zone

Best published baseline (original Windows project): **SVM-RBF on Group_B
financial-health ratios → F1 = 0.632** for distress detection.

## What's new in this rebuild
The original project used XGBoost / LightGBM / SVM / RF / LogReg / GBM / KNN.
This version adds models and techniques **not previously tried**, all runnable on
macOS without OpenMP:
- **CatBoost** (gradient boosting, native categorical support) + **Optuna** tuning
- **HistGradientBoosting** (scikit-learn) + **Optuna** tuning
- **Probability calibration** (Isotonic) for reliable distress probabilities
- **Soft-voting ensembles** of tuned models
- Proper **temporal (out-of-time) split** — no panel-data leakage

## Repository structure
```
thesis_finalmodels/
├── data/
│   ├── raw/merged_clean.xlsx          # source (1,991 rows × 179 cols)
│   └── processed/                     # split artifacts (gitignored if large)
├── src/                               # pipeline modules
│   ├── common.py                      # paths, logging, feature-set definitions
│   ├── preprocess.py                  # leakage-free temporal split + SMOTE
│   ├── models.py                      # model factories + threshold tuning
│   ├── run_e8_benchmark.py            # benchmark + new-model sweep
│   ├── run_e9_new_approaches.py       # Optuna + calibration + ensemble
│   └── run_e10_dashboard_data.py      # export best-model metrics for dashboard
├── models/                            # saved trained models (joblib)
├── results/                           # experiment CSVs (E8, E9, ...)
├── reports/                           # figures + HTML outputs
├── dashboard/                         # Streamlit presentable dashboard
├── PROJECT_STATE.md                   # full project state (resume doc)
├── requirements.txt
└── README.md
```

## Quick start
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# place merged_clean.xlsx in data/raw/
PYTHONPATH=src python src/run_e8_benchmark.py
PYTHONPATH=src python src/run_e9_new_approaches.py
```

## Methodology (no data leakage)
| Split | Years | Samples |
|-------|-------|---------|
| Train | ≤ 1399 | 1,504 |
| Validate | 1400–1401 | 226 |
| Test | 1402–1403 | 261 |

Median imputation, StandardScaler, and SMOTE are all fit on **train only**;
threshold tuning is done on **validation only**; the test set is touched once.

## Results
See `results/` and `reports/` — the latest best distress-detection scores are
summarised in `PROJECT_STATE.md` and the Streamlit dashboard.
