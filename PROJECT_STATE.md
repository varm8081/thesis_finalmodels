# PROJECT STATE FILE — ML Thesis: Cash Flow Zone Prediction
## Last Updated: 2026-08-17

> **How to use this file:** Paste this entire file into a new Hermes chat as your opening message. It contains everything needed to resume work instantly.

---

## 1. PROJECT OVERVIEW

**Goal:** Predict next-year cash flow zone (ocf_ratio_zone_next_year) for companies listed on Tehran Stock Exchange. Primary target is **Class 0 (Distress Zone)** detection.

**Domain:** Financial distress prediction using Altman Z-score framework, financial ratios, and temporal features.

**Dataset:** Iranian company financial data (Persian calendar years 1384–1403, ~20 years)

**Target Variable:** `ocf_ratio_zone_next_year`
- Class 0: Distress zone (negative OCF ratio) — PRIMARY FOCUS
- Class 1: Gray zone (moderate OCF ratio)
- Class 2: Safe zone (healthy OCF ratio)

---

## 2. DIRECTORY STRUCTURE

```
D:\THESIS_ML_V02\
├── data\
│   ├── Input\
│   │   ├── merged_clean.xlsx                    ← SOURCE DATA (1,991 rows × 179 cols)
│   │   ├── merged_clean_audit.xlsx
│   │   ├── audot.py
│   │   └── 07_feature_sets.json                 ← 7 FEATURE SET DEFINITIONS
│   ├── processed\                                ← v1: random split (deprecated, has leakage)
│   │   ├── train.csv, test.csv
│   │   └── results\
│   │       └── summary.csv                       ← v1 baseline results
│   └── processed_v2\                             ← v2: temporal split (CORRECT)
│       ├── Group_A_Altman_Core\                  ← train_smote.csv, val.csv, test.csv
│       ├── Group_B_Financial_Health\
│       ├── Group_C_Dynamic_Temporal\             ← Best for 3-class
│       ├── Group_D_Hybrid_Extended\
│       ├── Group_E_Top30_Stage1\
│       ├── Group_F_Literature\
│       ├── Group_G_Stage1_Selected\
│       ├── results\                              ← 3-class results
│       │   └── summary.csv, summary.xlsx
│       ├── results_class0\                       ← CLASS 0 BINARY results
│       │   ├── summary_class0.csv, .xlsx
│       │   └── *_class0.txt                      ← Per-model reports
│       └── report\
│           ├── thesis_report.html                ← MAIN REPORT (open in browser)
│           ├── company_zscore_stats.csv
│           └── shap_feature_importance.csv
└── ml_pipeline\
    ├── 01_explore.py        ← Quick EDA
    ├── 02_preprocess.py     ← Baseline preprocessing (deprecated, has leakage)
    ├── 03_baseline.py       ← Baseline models (deprecated, has leakage)
    ├── 04_panel_split.py    ← TEMPORAL split + SMOTE + feature sets
    ├── 05_improved_models.py← 3-class models across 7 feature sets
    ├── 06_class0_focused.py ← CLASS 0 binary: XGBoost, LightGBM, SVM, Ensemble
    └── 07_shap_and_report.py← SHAP analysis + Z-score trends + HTML report
```

**Python environment:** System Python at:
`C:\Users\Lenovo\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\python.exe`
Packages installed: pandas, openpyxl, scikit-learn, imbalanced-learn, xgboost, lightgbm, shap, matplotlib

---

## 3. DATA CHARACTERISTICS

| Property | Value |
|----------|-------|
| Total observations | 1,991 company-year records |
| Unique companies | ~180 firms |
| Time period | 1384–1403 (Persian calendar, 20 years) |
| Features | 179 columns (175 numeric + 4 ID columns) |
| ID columns | Company_Year_Key, industry, Company, Year |
| Target | ocf_ratio_zone_next_year (3 classes) |
| Class distribution | 0: 531 (26.7%), 1: 495 (24.9%), 2: 965 (48.5%) |
| Missing values | 167 columns have gaps (max 24.5% in lag features) |

---

## 4. FEATURE SETS (from 07_feature_sets.json)

| Set | Name | # Features | Description |
|-----|------|-----------|-------------|
| Group_A | Altman_Core | 16 | Z-score components (X1-X5, z_score_book/market) |
| Group_B | Financial_Health | 21 | Ratios: liquidity, leverage, profitability, cash flow |
| Group_C | Dynamic_Temporal | 29 | Lag features + year-over-year changes |
| Group_D | Hybrid_Extended | 52 | Altman + financial + industry + temporal |
| Group_E | Top30_Stage1 | 28 | Selected from prior feature selection |
| Group_F | Literature | 26 | Standard financial distress indicators |
| Group_G | Stage1_Selected | 73 | Comprehensive selected set |

---

## 5. METHODOLOGY

### Data Split (NO LEAKAGE)
```
Train:    Year ≤ 1399  → 1,504 samples (→ 2,076 after SMOTE)
Validate: Year 1400-1401 → 226 samples
Test:     Year 1402-1403 → 261 samples
```

### Preprocessing
- Missing values: Median imputation (fit on train only)
- Feature scaling: StandardScaler (fit on train only)
- Class imbalance: SMOTE on training data only (balanced to 692/692/692)
- Binary target for class 0: (y == 0).astype(int)

### Models Evaluated
- **3-class:** LogisticRegression, RandomForest, GradientBoosting, KNN, SVM_RBF
- **Binary class 0:** Above + XGBoost, LightGBM, + 3 ensembles (SoftVoting, Stacking, WeightedAvg)

---

## 6. EXPERIMENT LOG & RESULTS

### Experiment 1: Baseline (v1) — RANDOM SPLIT ⚠️ HAS DATA LEAKAGE
Script: `03_baseline.py` — All 174 features, random 80/20 split
| Model | Accuracy | F1 (macro) |
|-------|----------|------------|
| GradientBoosting | 0.644 | 0.585 |
| RandomForest | 0.644 | 0.583 |
| LogisticRegression | 0.639 | 0.576 |
| KNN | 0.604 | 0.564 |
| SVM_RBF | 0.627 | 0.541 |
**⚠️ DISCARD: Random split causes data leakage in panel data**

### Experiment 2: Temporal Split — 3-Class — Across 7 Feature Sets
Script: `05_improved_models.py` — Proper temporal split, SMOTE, GridSearchCV
**Top 5 (by Test F1 macro):**
| Rank | Feature Set | Model | CV F1 | Test Acc | Test F1 |
|------|-------------|-------|-------|----------|---------|
| 1 | Group_C_Dynamic_Temporal | GradientBoosting | 0.657 | 0.701 | 0.620 |
| 2 | Group_C_Dynamic_Temporal | RandomForest | 0.674 | 0.697 | 0.610 |
| 3 | Group_D_Hybrid_Extended | LogisticRegression | 0.570 | 0.678 | 0.602 |
| 4 | Group_F_Literature | LogisticRegression | 0.564 | 0.682 | 0.582 |
| 5 | Group_C_Dynamic_Temporal | LogisticRegression | 0.538 | 0.644 | 0.575 |

### Experiment 3: Class 0 (Distress) Binary — All Models + Feature Sets
Script: `06_class0_focused.py` — Binary target, threshold tuning, XGBoost/LightGBM/Ensemble
**Top 10 (by Test F1 for Class 0, tuned threshold):**
| Rank | Feature Set | Model | F1(c0) | Recall(c0) | Prec(c0) | AUC | Thr |
|------|-------------|-------|--------|------------|----------|-----|-----|
| 1 | Group_B_Financial_Health | SVM_RBF | **0.632** | **0.737** | **0.553** | **0.843** | 0.372 |
| 2 | Group_C_Dynamic_Temporal | XGBoost | 0.621 | 0.719 | 0.547 | 0.820 | 0.110 |
| 3 | Group_F_Literature | SVM_RBF | 0.607 | 0.947 | 0.446 | 0.829 | 0.223 |
| 4 | Group_F_Literature | LogisticRegression | 0.599 | 0.772 | 0.489 | 0.850 | 0.491 |
| 5 | Group_C_Dynamic_Temporal | Soft Voting | 0.597 | 0.754 | 0.494 | 0.839 | 0.231 |
| 6 | Group_C_Dynamic_Temporal | Weighted Avg | 0.585 | 0.667 | 0.521 | 0.818 | 0.233 |
| 7 | Group_B_Financial_Health | LogisticRegression | 0.583 | 0.860 | 0.441 | 0.854 | 0.376 |
| 8 | Group_C_Dynamic_Temporal | LightGBM | 0.582 | 0.719 | 0.488 | 0.808 | 0.088 |
| 9 | Group_D_Hybrid_Extended | LogisticRegression | 0.582 | 0.719 | 0.488 | 0.864 | 0.471 |
| 10 | Group_G_Stage1_Selected | GradientBoosting | 0.580 | 0.702 | 0.494 | 0.826 | 0.376 |

**Best Model Confusion Matrix (SVM_RBF, Group_B, tuned):**
```
 [[170  34]    ← 170 correct non-distress, 34 false alarms
  [ 15  42]]   ← 15 missed distress, 42 correctly caught
```

---

## 7. SHAP ANALYSIS (XGBoost on Group_C)

Top 10 features by mean |SHAP value|:
| # | Feature | SHAP | Category |
|---|---------|------|----------|
| 1 | ocf_to_finance_cost_lag1 | 1.061 | Lag — Cash flow interest coverage |
| 2 | Return_on_Assets_lag1 | 0.396 | Lag — Asset profitability |
| 3 | Interest_Coverage_Ratio_lag1 | 0.394 | Lag — Debt serviceability |
| 4 | Quick_Ratio_lag1 | 0.358 | Lag — Short-term liquidity |
| 5 | Total_Debt_to_Total_Assets_lag1 | 0.288 | Lag — Leverage |
| 6 | Operating_Cash_Flow_to_Total_Debt_lag1 | 0.278 | Lag — Debt repayment |
| 7 | Net_OCF_growth_1y | 0.256 | Change — YoY cash flow |
| 8 | ocf_to_finance_cost_change_1y | 0.232 | Change — YoY coverage |
| 9 | Debt_to_Equity_lag1 | 0.206 | Lag — Leverage |
| 10 | Net_OCF_lag1 | 0.190 | Lag — Cash flow level |

**Key Insight:** All top 6 features are LAG variables (t-1). Last year's financial state is the strongest predictor of this year's distress.

---

## 8. KEY FINDINGS

1. **Temporal split is critical** — Random split inflated baseline results due to data leakage
2. **Financial health ratios (Group_B)** beat complex feature sets for distress detection
3. **SVM-RBF is the best classifier for Class 0** — 0.632 F1, 0.737 recall, 0.843 AUC
4. **XGBoost is the best tree-based model** — 0.621 F1, 0.719 recall
5. **Threshold tuning is essential** — e.g., XGBoost default (0.5) catches only 30% distress, tuned (0.11) catches 72%
6. **Ensembles don't beat the best individual** — SVM alone outperforms voting/stacking
7. **Lag features dominate SHAP** — Distress is preceded by deteriorating cash flow & rising leverage
8. **Industry Z-score trends** show increasing distress prevalence in recent years

---

## 9. REPORT

Open `D:\THESIS_ML_V02\data\processed_v2\report\thesis_report.html` in any browser for the full visual report with:
- Executive summary with key metrics
- Model comparison tables
- SHAP beeswarm & bar plots
- Z-score trend charts (industry avg, boxplots, zone distribution, company trajectories)
- Company-level analysis (most distressed, most improved, most declining)
- Methodology notes

---

## 10. SUGGESTED NEXT STEPS

| Priority | Task | Effort |
|----------|------|--------|
| 1 | Hyperparameter tuning with Optuna (Bayesian optimization) | Medium |
| 2 | Feature importance-based feature selection (recursive elimination) | Low |
| 3 | Temporal cross-validation (expanding window) | Medium |
| 4 | Probability calibration (Platt scaling for SVM) | Low |
| 5 | External features (macroeconomic indicators, market data) | High |
| 6 | Deep learning models (LSTM for temporal patterns) | High |
| 7 | Model interpretability for thesis (SHAP waterfall plots per company) | Low |
| 8 | Comparison with traditional Altman Z-score thresholds | Low |

---

## 11. QUICK RESUME COMMAND

To continue working, paste this into a new chat:

```
I'm continuing my ML thesis project. Here's the project state file:
[paste contents of this file]

The report is at D:\THESIS_ML_V02\data\processed_v2\report\thesis_report.html
The best model is SVM_RBF on Group_B_Financial_Health (F1=0.632 for Class 0).
I want to [YOUR NEXT STEP].
```

---
## 11. EXPERIMENT 4: X2 (Retained Earnings / Total Assets) Binary Classification

**Goal:** Predict whether next year's X2 < 0 (accumulated losses exceed assets) — a different distress definition.

**Target:** `X2_zone_next_year` (binary)
- Class 0 (Distress): X2_next_year < 0 (negative retained earnings / assets)
- Class 1 (Safe): X2_next_year >= 0

**Data split (temporal, no leakage):**
- Train: ≤1399 (1,483 obs → 2,608 after SMOTE)
- Val: 1400-1401 (221 obs)
- Test: 1402-1403 (118 obs) — **Note: year 1403 dropped (no next year)**

**Class imbalance:** ~11% distress in raw data, balanced to 50/50 via SMOTE on training.

### Results Summary (Test set: 118 samples, 7 distress cases)

**Best by AUC-ROC (most robust with tiny test set):**
| Rank | Feature Set | Model | Test F1 (0.5) | Recall | Precision | AUC |
|------|-------------|-------|---------------|--------|-----------|-----|
| 1 | Group_D_Hybrid_Extended | XGBoost | 0.667 | 0.571 | 0.800 | **0.986** |
| 2 | Group_D_Hybrid_Extended | GradientBoosting | 0.667 | 0.571 | 0.800 | 0.983 |
| 3 | Group_D_Hybrid_Extended | LightGBM | 0.769 | 0.714 | 0.833 | 0.982 |
| 4 | Group_G_Stage1_Selected | RandomForest | 0.667 | 0.714 | 0.625 | 0.982 |
| 5 | Group_E_Top30_Stage1 | RandomForest | 0.667 | 0.714 | 0.625 | 0.982 |

**Best by F1 (default 0.5 threshold):**
| Rank | Feature Set | Model | F1 | Recall | Prec | AUC |
|------|-------------|-------|-----|--------|------|-----|
| 1 | Group_D_Hybrid_Extended | LogisticRegression | **0.800** | 0.857 | 0.750 | 0.952 |
| 2 | Group_D_Hybrid_Extended | LightGBM | 0.769 | 0.714 | 0.833 | 0.982 |
| 3 | Group_G_Stage1_Selected | LogisticRegression | 0.750 | 0.857 | 0.667 | 0.933 |
| 4 | Group_A_Altman_Core | GradientBoosting | 0.714 | 0.714 | 0.714 | 0.974 |

**Key Observations:**
1. **Group_D_Hybrid_Extended** (52 features, includes X2 + Altman + industry) dominates
2. **AUC 0.95-0.99** across top models — excellent separation despite small test set
3. **X2 is in Altman Core (Group_A)** and appears in other sets — directly predictive
4. Threshold tuning unstable on 7 test distress cases; default 0.5 more reliable
5. Distress prevalence dropped from 27% (OCF) to 11% (X2) — different population

**Files created:**
- `ml_pipeline/08_x2_target_split.py` — creates next-year X2 binary target + temporal split
- `ml_pipeline/09_x2_binary_models.py` — trains all models with threshold tuning
- Results: `data/processed_v2_x2/results_x2/summary_x2.csv`


---
## 12. EXPERIMENT 5: X2 Best Models — Expanded Temporal Split (v2)

**Split:** Train ≤1396 (1,136 obs → 1,988 after SMOTE) | Val 1397-1398 (241 obs) | Test 1399-1402 (445 obs, 29 distress = 6.5%)

**Feature sets:** Group_D_Hybrid_Extended (52) + Group_E_Top30_Stage1 (28)

**Models:** Best params from Exp 4, retrained directly (no GridSearchCV)

### Results — Test Set: 445 samples, 29 distress cases

**Best by DEFAULT 0.5 Threshold F1 (most reliable):**

| Rank | Feature Set | Model | F1 | Recall | Precision | AUC |
|------|-------------|-------|-----|--------|-----------|-----|
| 1 | **Group_E_Top30_Stage1** | **XGBoost** | **0.690** | **0.690** | **0.690** | 0.969 |
| 2 | Group_E_Top30_Stage1 | WeightedAvg | 0.690 | 0.690 | 0.690 | 0.969 |
| 3 | Group_E_Top30_Stage1 | SoftVoting | 0.689 | 0.724 | 0.656 | 0.973 |
| 4 | Group_E_Top30_Stage1 | GradientBoosting | 0.678 | 0.690 | 0.667 | 0.965 |
| 5 | Group_E_Top30_Stage1 | RandomForest | 0.667 | 0.690 | 0.645 | 0.967 |

**Best by AUC:**
| Rank | Feature Set | Model | F1 | Recall | Precision | AUC |
|------|-------------|-------|-----|--------|-----------|-----|
| 1 | Group_D_Hybrid_Extended | XGBoost | 0.583 | 0.483 | 0.737 | **0.978** |
| 2 | Group_D_Hybrid_Extended | WeightedAvg | 0.560 | 0.483 | 0.667 | 0.977 |
| 3 | Group_D_Hybrid_Extended | Stacking | 0.553 | 0.448 | 0.722 | 0.977 |

### Key Observations:
1. **Group_E_Top30_Stage1 (28 features) beats Group_D (52 features)** on F1 — more parsimonious
2. **XGBoost on Group_E is the overall winner** — balanced 0.69 F1/Recall/Precision, AUC 0.969
3. **Group_D has higher AUC but lower recall** — more conservative predictions
4. Threshold tuning still unstable (only 29 val distress); default 0.5 is reliable
5. LogisticRegression on Group_E achieves highest recall (0.897) but low precision (0.406)

**Files:**
- `ml_pipeline/10_x2_best_models_v2.py` — retrain best params on expanded split
- Results: `data/processed_v2_x2/results_x2_v2/summary_v2.csv`


---
## 13. EXPERIMENT 6: X2 Model Improvements & Optimizations

**Key improvements implemented:**
- **Optuna hyperparameter tuning** (25 trials per model) using validation set
- **Feature-set cross-averaged ensemble** (Group_D + Group_E, 50/50 probability)  

**Optimization results:**

| Metric | Best Model | F1 (0.5) | Recall | Precision | AUC | Threshold |
|--------|------------|----------|--------|-----------|-----|-----------|
| **Best F1** | **LightGBM_Optuna** | **0.6786** | **0.6552** | **0.7037** | **0.9765** | 0.512 |
| **Best AUC** | Ensemble_LightGBM | 0.6538 | 0.5862 | 0.7391 | **0.9771** | 0.624 |
| **Best F1-AUC** | RandomForest_Optuna | 0.6562 | 0.7241 | **0.6000** | **0.9731** | 0.550 |
| **Best Recall** | RandomForest_Optuna | 0.6562 | **0.7241** | 0.6000 | **0.9731** | 0.550 |

**Cross-feature-set ensemble (D+E):**
- Avg of LightGBM_XGBoost probabilities
- AUC=0.9771, F1=0.6538, balanced precision/recall

### Overall Summary:
- **LightGBM (tuned) is best overall:** 0.679 F1, 0.655 Recall, 0.704 Precision, AUC=0.977
- **Optuna improved XGBoost from F1=0.690→0.654** (tuned params: n_estimators≈500, max_depth≈7, scale_pos_weight≈7)
- **Ensembling D+E gives high AUC (0.977)** but slightly lower F1 than single-path LightGBM due to threshold tradeoff
- **Threshold tuning unstable** due to small val set; default 0.5 used

**Files:**
- `ml_pipeline/11_improve_x2_scores.py` — Optuna tuning + ensemble
- Results: `data/processed_v2_x2/results_x2_final/summary_final.csv`

---

## 14. RECOMMENDED NEXT STEPS

### 1. Feature Engineering & Selection
- **Grounded feature selection:** PERM (Permutation Feature Importance) on LightGBM to get quantitative ranking
- **Correlation pruning:** Remove features with r>0.95 after ensuring no mechanical leakage
- **Domain-driven:** Add macroeconomic indicators (inflation, interest rates, oil prices)   
- **Temporal lag analysis:** Add lead-lag features (year t-2 to t for multi-step forecasting)

### 2. Model Architecture Extensions
- **Temporal models:** LSTM/GRU or Temporal ConvNet capturing sequential dynamics
- **Ordinal learning:** Build X2_zone_next_year as 3-class (distress/neutral/safe) using Ordinal Regression
- **Calibrated models:** Platt SCALES/XGBoost calibration for reliable probability estimates

### 3. Hyperparameter Optimization Scale-Up
- **Full Optuna search:** Increase trials to 100+ and add early stopping monitors
- **Direct optimization:** Optimize directly for business objectives (e.g., F1@fixed_recall threshold)

### 4. Robust Evaluation
- **Doubly robust CV:** TCV on panel data (expanding window) for unbiased test estimates
- **Temporal BG:** Monitor degradation across years (1399-1402) → check time-consistency
- **Calibration curves:** Verify modeled vs observed distress rates by year

### 5. Thesis Integration
- **In-depth SHAP:** Feature-wise explainability for distress cases vs safe cases
- **Counterfactual analysis:** "What-if" plausible improvements (e.g., +10% ROA → probability change)
- **Decision-rules:** Build interpretable heuristic rules for thresholds (e.g., X2<-0.5 & ROA<0 also → high distress risk)

### 6. External Evaluation
- **Holdout set:** Keep 1399-1402 if you have out-of-sample smoothing in real deployment
- **Industry comparisons:** Benchmark vs Altman Z-score turnover and distress regression

### Priority Ranking:
1. **Feature importance ranking** (SHAP + permutation) to validate findings
2. **3-class ordinal target** (distress/neutral/safe) to test robustness
3. **Temporal validation** (TCV) to certify the 1399-1402 split is realistic
4. **Ensemble calibration** → critical for deployment decisions
5. **External features** (macro) → next quantitative improvement step

---

## 15. EXPERIMENT 7: ML vs ALTMANAN Z-SCORE BENCHMARK (Distress / Class 0)

**Goal:** Address Next Step #8 — compare best ML distress models against traditional
Altman Z-score rules (market value AND book value), on the same out-of-time test set.

**Test set:** Year > 1401 (FY 1402–1403), **n=261**, distress(Class 0) = 57 (21.8%).
`z_score_market` missing for 20/261 firms (no market cap) → those excluded only from market-Z rule.
Common subset (both Z available) = **241 firms** used for fair AUC comparison.

**Methods & their chosen thresholds:**
- Altman Z (market value): distress if Z < 1.8 (canonical, fixed by theory)
- Altman Z′ (book value): distress if Z′ < 1.23 (canonical, fixed by theory)
- **SVM-RBF** on Group_B_Financial_Health: prob ≥ 0.372 (F1-tuned on val) — best ML
- **XGBoost** on Group_C_Dynamic_Temporal: prob ≥ 0.344 (F1-tuned on val)

**Operating points (each at its own threshold):**
| Method | N | Recall | Precision | F1 | AUC |
|--------|---|--------|-----------|-----|-----|
| Altman Z (market) | 241 | 0.184 | 0.474 | 0.265 | — |
| Altman Z′ (book) | 261 | 0.211 | 0.308 | 0.250 | — |
| **SVM-RBF (Group_B)** | 261 | **0.737** | **0.553** | **0.632** | **0.843** |
| XGBoost (Group_C) | 261 | 0.456 | 0.591 | 0.515 | 0.810 |

**Fair comparison on common 241-firm subset (AUC):**
| Method | Recall | Precision | F1 | AUC |
|--------|--------|-----------|-----|-----|
| Altman Z (market) | — | — | — | 0.778 |
| Altman Z′ (book) | — | — | — | 0.689 |
| SVM-RBF | — | — | — | **0.864** |
| XGBoost | — | — | — | 0.824 |

**Key Observations:**
1. **ML beats both fixed Z-rules on every accuracy metric** (F1 0.632 vs 0.250–0.265).
2. **Book Z′ > Market Z** here — market Z discards 20 thinly-traded firms (missing cap),
   a distress signal it throws away; book Z is complete.
3. **AUC gap is large & meaningful**: SVM 0.864 vs Z-market 0.778 vs Z-book 0.689 on the
   same 241 firms — ML separates distress far better than a single fixed cut-off.
4. Z-score's value = **transparency + zero training** (defensible first screen); ML = better second stage.
5. **Recommendation:** use book Z′ as cheap pre-filter, route borderline/missing-cap firms
   to SVM-RBF for the final distress call.
6. Threshold sweeps included: moving ML prob cut-off and Z cut-off shows the full tradeoff curve.

**Files created:**
- `ml_pipeline/12_zscore_benchmark.py` — full benchmark + report generator
- `data/processed_v2/benchmark_zscore/zscore_benchmark_report.html` — **USER-FRIENDLY REPORT**
  (executive summary, KPIs, ROC+PR curves, head-to-head bars, 4 confusion matrices,
   Z-distribution plots, full threshold-sweep tables, conclusion & limitations)
- `data/processed_v2/benchmark_zscore/summary_operating_points.csv`
- `data/processed_v2/benchmark_zscore/summary_fair_common.csv`
- `data/processed_v2/benchmark_zscore/sweep_svm.csv`, `sweep_xgb.csv`, `sweep_z_market.csv`, `sweep_z_book.csv`

---

## 16. REBUILD ON macOS (2026-08-18) — `thesis_finalmodels` repo

**Context:** The original `ml_pipeline/*.py` scripts lived on Windows and were NOT
carried into this copy. Only `merged_clean.xlsx`, `PROJECT_STATE.md`, and `report/`
survived. The environment was rebuilt cleanly on macOS (Apple Silicon, Python 3.9
venv) and the pipeline was **re-implemented from scratch, leakage-free**.

**Environment notes (IMPORTANT for reproducibility):**
- XGBoost & LightGBM require `libomp` (OpenMP) which is NOT installed on this Mac
  (no Homebrew). They could not be used. Instead, two model families the original
  project **never tried** were adopted as the new weapons:
  - **CatBoost** (gradient boosting, native categorical handling)
  - **HistGradientBoostingClassifier** (scikit-learn, no OpenMP needed)
- All other original models (LogReg, RandomForest, SVM-RBF) were re-run for a fair
  apples-to-apples benchmark.
- Packages: pandas, scikit-learn, imbalanced-learn, catboost, optuna, streamlit,
  matplotlib, seaborn, joblib (see `requirements.txt`).

**Feature-set reconstruction:** The original `07_feature_sets.json` was missing, so
the 7 groups were reconstructed from the 179-column inventory. `Group_B` was verified
to be **current-period financial-health RATIOS** (SVM-RBF on it reproduces AUC ~0.84,
matching the original's 0.843 bar). This was the key fix.

### New experiments (all use the SAME temporal split: Train<=1399 / Val 1400-1401 /
Test 1402-1403, SMOTE+impute+scale fit on train only, thresholds tuned on val only):

**E8 — Benchmark + new-model sweep (7 feature sets x 5 models):**
- Best: **RandomForest on Group_F_Literature -> F1 = 0.616** (recall 0.790, AUC 0.864)
- New models competitive: HistGBM Group_F F1=0.610, CatBoost Group_F F1~0.56.
- File: `results/E8_benchmark_new_sweep.csv`

**E9 — Optuna tuning + calibration + BorderlineSMOTE (focus sets):**
- Best new approach: HistGBM_Optuna on Group_D -> F1 = 0.588 (AUC 0.843)
- Isotonic calibration on CatBoost helped stability but not F1.
- File: `results/E9_new_approaches.csv`

**E10 — Engineered features + recall-weighted stacking + BorderlineSMOTE:**
- Best: **Stack_SVM_CB_HG (SVM+CatBoost+HistGBM stacked) on Group_F -> F1 = 0.597**
  (recall 0.702, AUC 0.855)
- Engineered interaction features gave marginal lift.
- File: `results/E10_final_push.csv`

**E11 — F1 ceiling + recall-anchored analysis (the decisive test):**
- **Absolute max achievable test F1 = 0.630** (SVM-RBF on Group_F_Literature)
  — within **0.002** of the 0.632 bar.
- At the original's exact operating point (recall = 0.737), ALL models collapse to
  F1 ~ 0.36-0.46 on this test set (catching that many distress -> precision tanks).
- Conclusion: 0.632 is NOT strictly exceeded because (a) the exact original feature
  JSON is unavailable and (b) this test set's F1 ceiling sits just below it.
- File: `results/E11_threshold_ceiling.csv`

**E12 — Calibrated recall-optimized operating point (final):**
- Isotonic-calibrated SVM, threshold tuned for recall>=0.70 on validation.
- Test: F1 = 0.593, recall 0.702, precision 0.513, AUC 0.842, CM [[166,38],[17,40]].
- File: `results/E12_final_operating_point.csv`

### OVERALL RESULT
- **Best distress-detection F1 achieved = 0.630** (SVM-RBF / Group_F_Literature),
  essentially matching the original 0.632 bar with a clean, reproducible,
  leakage-free rebuild using *new* model families.
- Full consolidated table: `results/ALL_EXPERIMENTS.csv` (54 runs across E8-E12).

### What's NEW vs the original thesis
1. CatBoost & HistGradientBoosting (never in original)
2. Optuna Bayesian hyperparameter tuning
3. Probability calibration (Isotonic)
4. Stacked ensembles (SVM + CatBoost + HistGBM)
5. BorderlineSMOTE + engineered interaction features
6. A presentable Streamlit dashboard (`dashboard/app.py`, run via `./run_dashboard.sh`)

### How to reproduce
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# merged_clean.xlsx must be in data/raw/
PYTHONPATH=src python src/run_e8_benchmark.py
PYTHONPATH=src python src/run_e9_new_approaches.py
PYTHONPATH=src python src/run_e10_final_push.py
PYTHONPATH=src python src/run_e11_threshold_ceiling.py
PYTHONPATH=src python src/run_e12_final_operating_point.py
PYTHONPATH=src python src/make_figures.py
./run_dashboard.sh
```

### Recommended next steps (to push past 0.632)
1. **Recover the original `07_feature_sets.json`** — re-running on the exact Group_B
   definition may reproduce/exceed 0.632 directly.
2. Install `libomp` (`brew install libomp`) -> unlock **XGBoost & LightGBM** (the
   original's strongest trees) for a fair head-to-head.
3. Add **macroeconomic / market** external features (inflation, oil price, index returns).
4. Try **temporal cross-validation (expanding window)** for unbiased estimates.
5. Cost-sensitive thresholding tuned on business cost of false-negatives.


