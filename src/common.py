"""
Common utilities: clean English logging, project paths, and feature-group
definitions for the Tehran Stock Exchange financial-distress (Class 0) project.

All console output from this project flows through `log()` so the terminal stays
uncluttered -- only important milestones are printed.
"""
import os
import sys

# ----------------------------------------------------------------------------
# Project paths (GitHub-standard layout, all relative to repo root)
# ----------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(ROOT, "data", "raw")
DATA_PROCESSED = os.path.join(ROOT, "data", "processed")
SRC = os.path.join(ROOT, "src")
MODELS_DIR = os.path.join(ROOT, "models")
RESULTS_DIR = os.path.join(ROOT, "results")
REPORTS_DIR = os.path.join(ROOT, "reports")

for d in (DATA_RAW, DATA_PROCESSED, MODELS_DIR, RESULTS_DIR, REPORTS_DIR):
    os.makedirs(d, exist_ok=True)

RAW_XLSX = os.path.join(DATA_RAW, "merged_clean.xlsx")

# ----------------------------------------------------------------------------
# Clean logger (English only, level-aware)
# ----------------------------------------------------------------------------
_VERBOSE = False


def log(msg, level="INFO"):
    """Print an important message. level in INFO/OK/WARN/ERR/STEP/METRIC."""
    colors = {"INFO": "\033[0m", "OK": "\033[32m", "WARN": "\033[33m",
              "ERR": "\033[31m", "STEP": "\033[36m", "METRIC": "\033[35m"}
    if level == "METRIC":
        print(f"  {msg}")
    else:
        color = colors.get(level, "")
        reset = "\033[0m" if color else ""
        print(f"{color}[{level}]{reset} {msg}")


def section(title):
    bar = "=" * 64
    print(f"\n{bar}\n  {title}\n{bar}")


def set_verbose(v):
    global _VERBOSE
    _VERBOSE = v


# ----------------------------------------------------------------------------
# Target & split configuration (matches PROJECT_STATE: no leakage temporal split)
# ----------------------------------------------------------------------------
TARGET = "ocf_ratio_zone_next_year"
ID_COLS = ["Company_Year_Key", "industry", "Company", "Year"]

# Temporal split: Train <= 1399 | Val 1400-1401 | Test 1402-1403
TRAIN_MAX_YEAR = 1399
VAL_YEARS = [1400, 1401]
TEST_YEARS = [1402, 1403]


# ----------------------------------------------------------------------------
# Feature groups (reconstructed from the 179-column inventory; the original
# 07_feature_sets.json was not carried over in this copy).
# ----------------------------------------------------------------------------
# Lag-1 features (t-1) -- strongest predictors per SHAP in PROJECT_STATE
LAG1 = [
    "Revenue_lag1", "Net_Income_lag1", "Operating_Income_lag1", "Net_OCF_lag1",
    "Return_on_Assets_lag1", "Return_on_Equity_lag1", "Current_Ratio_lag1",
    "Quick_Ratio_lag1", "Debt_Ratio_lag1", "Debt_to_Equity_lag1",
    "Total_Debt_to_Total_Assets_lag1", "Operating_Cash_Flow_to_Total_Debt_lag1",
    "Interest_Coverage_Ratio_lag1", "ocf_to_finance_cost_lag1",
    "z_score_market_lag1", "z_score_book_lag1",
]

# Year-over-year change features
YOY = [
    "Net_Change_In_Cash", "Exchange_Rate_Effect",
    "Revenue_growth_1y", "Operating_Income_growth_1y", "Net_Income_growth_1y",
    "Net_OCF_growth_1y", "Return_on_Assets_change_1y", "Return_on_Equity_change_1y",
    "Current_Ratio_change_1y", "Quick_Ratio_change_1y", "Debt_Ratio_change_1y",
    "Debt_to_Equity_change_1y", "Operating_Cash_Flow_to_Total_Debt_change_1y",
    "Interest_Coverage_Ratio_change_1y", "ocf_to_finance_cost_change_1y",
]

# Z-score family (incl. industry-relative)
ZSCORE = [
    "z_score_market", "z_score_book", "z_score_market_lag1", "z_score_book_lag1",
    "z_score_market_industry_mean", "z_score_market_vs_industry_mean",
    "z_score_market_industry_rank_pct", "z_score_book_industry_mean",
    "z_score_book_vs_industry_mean", "z_score_book_industry_rank_pct",
]

# Altman Z components
ALT_X = ["X1", "X2", "X3", "X5"]

# Financial-health ratios (current period)
RATIO = [
    "Quick_Ratio", "Debt_Ratio", "Long_Term_Debt_Ratio", "Current_Ratio",
    "Cash_Ratio", "Interest_Coverage_Ratio", "Return_on_Assets", "Return_on_Equity",
    "Operating_Cash_Flow_to_Total_Debt", "ocf_to_finance_cost",
    "Net_OCF", "Operating_Income", "Net_Income", "Revenue",
    "Debt_Ratio_industry_mean", "Debt_Ratio_vs_industry_mean",
    "Debt_Ratio_industry_rank_pct", "Current_Ratio_industry_mean",
    "Current_Ratio_vs_industry_mean", "Current_Ratio_industry_rank_pct",
]

# ---- The 7 feature sets (named to match PROJECT_STATE for continuity) ----
GROUP_A_ALTMAN_CORE = ALT_X + ["z_score_book", "z_score_market",
                               "z_score_book_lag1", "z_score_market_lag1"]
# Original project Group_B = current-period financial-health RATIOS (verified:
# SVM-RBF on this set reproduces AUC ~0.84, matching the 0.632 F1 benchmark).
GROUP_B_FINANCIAL_HEALTH = list(dict.fromkeys(RATIO))
GROUP_C_DYNAMIC_TEMPORAL = LAG1 + YOY
GROUP_D_HYBRID_EXTENDED = list(dict.fromkeys(
    GROUP_A_ALTMAN_CORE + GROUP_B_FINANCIAL_HEALTH + GROUP_C_DYNAMIC_TEMPORAL + ZSCORE))
# Top-30 curated from SHAP findings + literature (parsimonious)
_GROUP_E_CURATED = [
    "ocf_to_finance_cost_lag1", "Return_on_Assets_lag1", "Interest_Coverage_Ratio_lag1",
    "Quick_Ratio_lag1", "Total_Debt_to_Total_Assets_lag1",
    "Operating_Cash_Flow_to_Total_Debt_lag1", "Net_OCF_growth_1y",
    "ocf_to_finance_cost_change_1y", "Debt_to_Equity_lag1", "Net_OCF_lag1",
    "z_score_book", "z_score_market", "z_score_book_lag1", "z_score_market_lag1",
    "Return_on_Equity_lag1", "Debt_Ratio_lag1", "Current_Ratio_lag1",
    "Interest_Coverage_Ratio", "Quick_Ratio", "Debt_to_Equity",
    "Total_Debt_to_Total_Assets", "Operating_Cash_Flow_to_Total_Debt",
    "ocf_to_finance_cost", "Return_on_Assets", "Return_on_Equity",
    "Net_OCF_growth_1y", "Net_Income_growth_1y", "Revenue_growth_1y",
    "Net_OCF", "Debt_Ratio",
]
GROUP_E_TOP30_STAGE1 = list(dict.fromkeys(_GROUP_E_CURATED))[:30]
GROUP_F_LITERATURE = [
    "z_score_book", "z_score_market", "Debt_Ratio", "Interest_Coverage_Ratio",
    "Current_Ratio", "Quick_Ratio", "Return_on_Assets", "Return_on_Equity",
    "Total_Debt_to_Total_Assets", "Debt_to_Equity", "Operating_Cash_Flow_to_Total_Debt",
    "ocf_to_finance_cost", "Net_OCF", "Operating_Income", "Net_Income",
    "Revenue", "X1", "X2", "X3", "X5", "Cash_Ratio", "Long_Term_Debt_Ratio",
    "Return_on_Assets_lag1", "Debt_Ratio_lag1", "Quick_Ratio_lag1",
    "Interest_Coverage_Ratio_lag1"]
ENGINEERED = list(dict.fromkeys(LAG1 + YOY + ZSCORE + RATIO + ALT_X))
GROUP_G_STAGE1_SELECTED = ENGINEERED  # comprehensive engineered set

FEATURE_SETS = {
    "Group_A_Altman_Core": GROUP_A_ALTMAN_CORE,
    "Group_B_Financial_Health": GROUP_B_FINANCIAL_HEALTH,
    "Group_C_Dynamic_Temporal": GROUP_C_DYNAMIC_TEMPORAL,
    "Group_D_Hybrid_Extended": GROUP_D_HYBRID_EXTENDED,
    "Group_E_Top30_Stage1": GROUP_E_TOP30_STAGE1,
    "Group_F_Literature": GROUP_F_LITERATURE,
    "Group_G_Stage1_Selected": GROUP_G_STAGE1_SELECTED,
}

# CatBoost gets the industry code as a categorical bonus
CATBOOST_CAT_COLS = ["industry"]


def get_feature_list(name):
    return list(FEATURE_SETS[name])


def feature_set_sizes():
    return {k: len(v) for k, v in FEATURE_SETS.items()}


if __name__ == "__main__":
    print("Feature set sizes:")
    for k, v in feature_set_sizes().items():
        print(f"  {k}: {v}")
