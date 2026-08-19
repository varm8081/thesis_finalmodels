"""
Model training + evaluation for the distress (Class 0) binary task.

Benchmark models (re-run here for a fair, leakage-free comparison):
    LogisticRegression, RandomForest, SVM-RBF   (original project's best was SVM on Group_B)

NEW models never tried in the original project (its weapon was XGBoost/LightGBM):
    CatBoost, HistGradientBoostingClassifier      (both run natively on macOS, no OpenMP)

All threshold tuning is done on the VALIDATION set only (no test leakage).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (f1_score, recall_score, precision_score,
                             roc_auc_score, average_precision_score,
                             confusion_matrix)
from common import log


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------
def make_models(seed=42):
    """Return dict name -> untrained estimator (probabilistic)."""
    return {
        "LogReg": LogisticRegression(max_iter=2000, class_weight="balanced",
                                     random_state=seed),
        "RandomForest": RandomForestClassifier(n_estimators=400,
                                               class_weight="balanced",
                                               n_jobs=-1, random_state=seed),
        "SVM_RBF": SVC(probability=True, class_weight="balanced",
                       gamma="scale", random_state=seed),
        "CatBoost": None,   # built lazily (needs cat cols)
        "HistGBM": HistGradientBoostingClassifier(random_state=seed),
    }


def make_catboost(params=None, seed=42):
    from catboost import CatBoostClassifier
    p = dict(iterations=400, depth=6, learning_rate=0.05,
             loss_function="Logloss", eval_metric="F1",
             random_seed=seed, verbose=0, early_stopping_rounds=40)
    if params:
        p.update(params)
    return CatBoostClassifier(**p)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def _proba_pos(est, X, cat_cols=None):
    if hasattr(est, "predict_proba"):
        return est.predict_proba(X)[:, 1]
    raise RuntimeError("estimator has no predict_proba")


def tune_threshold(y_val, p_val, target_recall=0.70):
    """
    Pick the probability threshold that maximises F1 on validation, with a
    soft floor on recall (we care about catching distress). Returns threshold.
    """
    grids = np.linspace(0.05, 0.95, 91)
    best_f1, best_t, best_meta = -1, 0.5, None
    for t in grids:
        pred = (p_val >= t).astype(int)
        f1 = f1_score(y_val, pred, zero_division=0)
        rec = recall_score(y_val, pred, zero_division=0)
        # prefer F1, but require recall >= target*0.9 so we don't sacrifice catch-rate
        if rec >= target_recall * 0.85 and f1 > best_f1:
            best_f1, best_t = f1, t
    if best_t == 0.5 and best_f1 == -1:
        # fallback: best F1 ignoring recall floor
        for t in grids:
            pred = (p_val >= t).astype(int)
            f1 = f1_score(y_val, pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
    return float(round(best_t, 3))


def evaluate(est, split, model_name, feature_set, cat_cols=None,
             target_recall=0.70):
    cat_cols = cat_cols or []
    X_val, y_val = split["X_val"], split["y_val"]
    X_test, y_test = split["X_test"], split["y_test"]

    p_val = _proba_pos(est, X_val, cat_cols)
    p_test = _proba_pos(est, X_test, cat_cols)

    thr = tune_threshold(y_val, p_val, target_recall)
    pred_test = (p_test >= thr).astype(int)

    auc = roc_auc_score(y_test, p_test) if len(np.unique(y_test)) == 2 else float("nan")
    ap = average_precision_score(y_test, p_test) if len(np.unique(y_test)) == 2 else float("nan")

    cm = confusion_matrix(y_test, pred_test)
    res = {
        "feature_set": feature_set,
        "model": model_name,
        "threshold": thr,
        "test_f1": round(f1_score(y_test, pred_test), 4),
        "test_recall": round(recall_score(y_test, pred_test, zero_division=0), 4),
        "test_precision": round(precision_score(y_test, pred_test, zero_division=0), 4),
        "test_accuracy": round((pred_test == y_test).mean(), 4),
        "test_auc": round(auc, 4) if not np.isnan(auc) else None,
        "test_pr_auc": round(ap, 4) if not np.isnan(ap) else None,
        "confusion_matrix": cm.tolist(),
        "n_test": int(len(y_test)),
        "n_test_distress": int(y_test.sum()),
    }
    return res


def fit_model(name, split, seed=42):
    """Fit a single named model on the (SMOTE-balanced) training data."""
    if name == "CatBoost":
        est = make_catboost(seed=seed)
        if split["cat_cols"]:
            cat_idx = [split["feature_cols"].index(c) for c in split["cat_cols"]
                       if c in split["feature_cols"]]
            est.fit(split["X_res"], split["y_res"], cat_features=cat_idx)
        else:
            est.fit(split["X_res"], split["y_res"])
    else:
        est = make_models(seed)[name]
        est.fit(split["X_res"], split["y_res"])
    return est


if __name__ == "__main__":
    from preprocess import build_pipeline
    from common import feature_set_sizes
    df = __import__("preprocess").load_raw()
    fs = "Group_B_Financial_Health"
    split = build_pipeline(df, feature_set_sizes()[fs], cat_cols=[])
    for name in ["LogReg", "RandomForest", "SVM_RBF", "CatBoost", "HistGBM"]:
        est = fit_model(name, split)
        r = evaluate(est, split, name, fs)
        log(f"{name}: F1={r['test_f1']} recall={r['test_recall']} "
            f"prec={r['test_precision']} AUC={r['test_auc']} thr={r['threshold']}",
            "METRIC")
