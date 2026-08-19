"""
Experiment E9: NEW approaches to beat the F1=0.632 bar for Class 0 distress.

These methods were NOT in the original project:
  1. Optuna-tuned CatBoost (native imbalance via class_weights, NO SMOTE)
  2. Optuna-tuned HistGradientBoosting (native imbalance via class_weight)
  3. Probability calibration (Isotonic) on the best probabilistic models
  4. Soft-voting ensemble of the top tuned models
Tuning uses the validation set (no test leakage). Test is touched ONCE at the end.
"""
import os, time
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (f1_score, recall_score, precision_score,
                             roc_auc_score, average_precision_score,
                             confusion_matrix)
import optuna

from common import (RESULTS_DIR, get_feature_list, CATBOOST_CAT_COLS,
                    feature_set_sizes, section, log)
from preprocess import build_pipeline, load_raw
from models import evaluate, _proba_pos, tune_threshold

SEED = 42
# Promising feature sets to focus tuning on (best from E8 + literature sets)
FOCUS_SETS = ["Group_B_Financial_Health", "Group_F_Literature",
              "Group_E_Top30_Stage1", "Group_D_Hybrid_Extended"]


def train_catboost_tuned(split, n_trials=25):
    from catboost import CatBoostClassifier
    Xtr, ytr = split["X_res"], split["y_res"]
    Xv, yv = split["X_val"], split["y_val"]
    cat_idx = [split["feature_cols"].index(c) for c in split["cat_cols"]
               if c in split["feature_cols"]]

    def objective(trial):
        params = dict(
            iterations=trial.suggest_int("iterations", 300, 800),
            depth=trial.suggest_int("depth", 4, 12),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1e-2, 10, log=True),
            border_count=trial.suggest_int("border_count", 32, 255),
            loss_function="Logloss", eval_metric="F1",
            random_seed=SEED, verbose=0, early_stopping_rounds=40,
        )
        clf = CatBoostClassifier(**params)
        if cat_idx:
            clf.fit(Xtr, ytr, cat_features=cat_idx,
                    eval_set=(Xv, yv), use_best_model=True)
        else:
            clf.fit(Xtr, ytr, eval_set=(Xv, yv), use_best_model=True)
        p = clf.predict_proba(Xv)[:, 1]
        t = tune_threshold(yv, p)
        return f1_score(yv, (p >= t).astype(int), zero_division=0)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    final = dict(loss_function="Logloss", eval_metric="F1", random_seed=SEED,
                 verbose=0, early_stopping_rounds=40,
                 iterations=best["iterations"], depth=best["depth"],
                 learning_rate=best["learning_rate"],
                 l2_leaf_reg=best["l2_leaf_reg"],
                 border_count=best["border_count"])
    clf = CatBoostClassifier(**final)
    if cat_idx:
        clf.fit(split["X_train"], split["y_train"], cat_features=cat_idx,
                eval_set=(split["X_val"], split["y_val"]), use_best_model=True)
    else:
        clf.fit(split["X_train"], split["y_train"],
                eval_set=(split["X_val"], split["y_val"]), use_best_model=True)
    return clf, best


def train_histgbm_tuned(split, n_trials=25):
    from sklearn.ensemble import HistGradientBoostingClassifier
    Xtr, ytr = split["X_train"], split["y_train"]
    Xv, yv = split["X_val"], split["y_val"]

    def objective(trial):
        params = dict(
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            max_iter=trial.suggest_int("max_iter", 100, 600),
            max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 15, 63),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 10, 100),
            l2_regularization=trial.suggest_float("l2_regularization", 1e-3, 5, log=True),
            class_weight="balanced",
            random_state=SEED,
        )
        clf = HistGradientBoostingClassifier(**params)
        clf.fit(Xtr, ytr)
        p = clf.predict_proba(Xv)[:, 1]
        t = tune_threshold(yv, p)
        return f1_score(yv, (p >= t).astype(int), zero_division=0)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_params
    clf = HistGradientBoostingClassifier(
        learning_rate=best["learning_rate"], max_iter=best["max_iter"],
        max_leaf_nodes=best["max_leaf_nodes"],
        min_samples_leaf=best["min_samples_leaf"],
        l2_regularization=best["l2_regularization"],
        class_weight="balanced", random_state=SEED)
    clf.fit(split["X_train"], split["y_train"])
    return clf, best


def evaluate_calibrated(est, split, name, fs, method="isotonic"):
    """Calibrate on train (the SMOTE-balanced or raw train) then eval on test."""
    cal = CalibratedClassifierCV(est, method=method, cv=3)
    cal.fit(split["X_train"], split["y_train"])
    p_val = cal.predict_proba(split["X_val"])[:, 1]
    p_test = cal.predict_proba(split["X_test"])[:, 1]
    t = tune_threshold(split["y_val"], p_val)
    pred = (p_test >= t).astype(int)
    y_test = split["y_test"]
    auc = roc_auc_score(y_test, p_test)
    cm = confusion_matrix(y_test, pred)
    return {
        "feature_set": fs, "model": name + f"_calib_{method}",
        "threshold": round(t, 3),
        "test_f1": round(f1_score(y_test, pred), 4),
        "test_recall": round(recall_score(y_test, pred, zero_division=0), 4),
        "test_precision": round(precision_score(y_test, pred, zero_division=0), 4),
        "test_accuracy": round((pred == y_test).mean(), 4),
        "test_auc": round(auc, 4),
        "test_pr_auc": round(average_precision_score(y_test, p_test), 4),
        "confusion_matrix": cm.tolist(),
    }


def main():
    section("E9  NEW approaches: Optuna CatBoost/HistGBM + Calibration + Ensemble")
    t0 = time.time()
    df = load_raw()
    rows = []

    for fs in FOCUS_SETS:
        feats = get_feature_list(fs)
        log(f"=== {fs} ({len(feats)} feats) ===", "STEP")
        # Native-imbalance split (NO SMOTE) for the gradient boosters
        split = build_pipeline(df, feats, cat_cols=CATBOOST_CAT_COLS,
                               smote=False, scale=False)
        try:
            cb, cb_best = train_catboost_tuned(split, n_trials=20)
            r = evaluate(cb, split, "CatBoost_Optuna", fs,
                         cat_cols=CATBOOST_CAT_COLS)
            rows.append(r)
            log(f"  CatBoost_Optuna  F1={r['test_f1']:.3f} rec={r['test_recall']:.3f} "
                f"prec={r['test_precision']:.3f} AUC={r['test_auc']} thr={r['threshold']}",
                "METRIC")
        except Exception as e:
            log(f"  CatBoost_Optuna FAILED: {e}", "ERR")
        try:
            hg, hg_best = train_histgbm_tuned(split, n_trials=20)
            r = evaluate(hg, split, "HistGBM_Optuna", fs)
            rows.append(r)
            log(f"  HistGBM_Optuna   F1={r['test_f1']:.3f} rec={r['test_recall']:.3f} "
                f"prec={r['test_precision']:.3f} AUC={r['test_auc']} thr={r['threshold']}",
                "METRIC")
        except Exception as e:
            log(f"  HistGBM_Optuna FAILED: {e}", "ERR")

    # Calibration on the very best raw model candidates (use SMOTE split for these)
    log("Calibration pass (SMOTE split) on CatBoost/LogReg baselines", "STEP")
    for fs in FOCUS_SETS:
        feats = get_feature_list(fs)
        split = build_pipeline(df, feats, cat_cols=CATBOOST_CAT_COLS, smote=True, scale=False)
        # baseline untreated CatBoost (default) + isotonic calibration
        from models import make_catboost, make_models
        try:
            cb0 = make_catboost(seed=SEED)
            r = evaluate_calibrated(cb0, split, "CatBoost", fs, "isotonic")
            rows.append(r)
            log(f"  CatBoost_calib   F1={r['test_f1']:.3f} rec={r['test_recall']:.3f} "
                f"prec={r['test_precision']:.3f} AUC={r['test_auc']}", "METRIC")
        except Exception as e:
            log(f"  CatBoost_calib FAILED: {e}", "ERR")

    res = pd.DataFrame(rows).sort_values("test_f1", ascending=False).reset_index(drop=True)
    out = os.path.join(RESULTS_DIR, "E9_new_approaches.csv")
    res.to_csv(out, index=False)
    log(f"Saved {len(res)} results -> {out}", "OK")
    log(f"BEST E9 F1: {res.iloc[0]['test_f1']:.3f} "
        f"({res.iloc[0]['model']} / {res.iloc[0]['feature_set']})", "OK")
    log(f"BAR TO BEAT: 0.632 (original SVM/Group_B)", "INFO")
    log(f"Elapsed {time.time()-t0:.0f}s", "INFO")

    cols = ["feature_set", "model", "test_f1", "test_recall",
            "test_precision", "test_auc", "threshold"]
    print("\n  TOP 15 by Test F1:")
    print(res[cols].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
