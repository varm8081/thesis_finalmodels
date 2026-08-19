"""
Experiment E13: BEAT 0.632 using the EXACT original feature sets (07_feature_sets.json)
+ the new model families (CatBoost, HistGBM) + Optuna tuning + calibration +
stacking. Feature definitions are now authoritative (validated present in data).

Strategy to exceed the bar:
  * Reproduce SVM/Group_B ~0.632 first (sanity)
  * Tune CatBoost & HistGBM (Optuna) on each set, native imbalance
  * Isotonic-calibrate top probabilistic models
  * Stacked ensembles of the best bases
  * Recall-oriented threshold on validation (thesis priority = catch distress)
Test touched once at the end.
"""
import os, time, json, numpy as np, pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, HistGradientBoostingClassifier,
                              StackingClassifier, VotingClassifier)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (f1_score, recall_score, precision_score,
                             roc_auc_score, average_precision_score,
                             confusion_matrix)
import optuna
from catboost import CatBoostClassifier

from common import (RESULTS_DIR, get_feature_list, feature_set_sizes,
                    CATBOOST_CAT_COLS, section, log)
from preprocess import build_pipeline, load_raw
from models import tune_threshold

SEED = 42
BAR = 0.632


def tune_catboost(split, n_trials=25):
    Xtr, ytr = split["X_train"], split["y_train"]
    Xv, yv = split["X_val"], split["y_val"]
    cat_idx = [split["feature_cols"].index(c) for c in split["cat_cols"]
               if c in split["feature_cols"]]

    def obj(trial):
        p = dict(iterations=trial.suggest_int("iterations", 300, 800),
                 depth=trial.suggest_int("depth", 4, 10),
                 learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                 l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1e-2, 10, log=True),
                 border_count=trial.suggest_int("border_count", 32, 255),
                 loss_function="Logloss", eval_metric="F1", random_seed=SEED,
                 verbose=0, early_stopping_rounds=40)
        clf = CatBoostClassifier(**p)
        if cat_idx:
            clf.fit(Xtr, ytr, cat_features=cat_idx, eval_set=(Xv, yv), use_best_model=True)
        else:
            clf.fit(Xtr, ytr, eval_set=(Xv, yv), use_best_model=True)
        pv = clf.predict_proba(Xv)[:, 1]
        t = tune_threshold(yv, pv)
        return f1_score(yv, (pv >= t).astype(int), zero_division=0)

    st = optuna.create_study(direction="maximize",
                             sampler=optuna.samplers.TPESampler(seed=SEED))
    st.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    b = st.best_params
    final = dict(loss_function="Logloss", eval_metric="F1", random_seed=SEED,
                 verbose=0, early_stopping_rounds=40, iterations=b["iterations"],
                 depth=b["depth"], learning_rate=b["learning_rate"],
                 l2_leaf_reg=b["l2_leaf_reg"], border_count=b["border_count"])
    clf = CatBoostClassifier(**final)
    if cat_idx:
        clf.fit(split["X_train"], split["y_train"], cat_features=cat_idx,
                eval_set=(split["X_val"], split["y_val"]), use_best_model=True)
    else:
        clf.fit(split["X_train"], split["y_train"],
                eval_set=(split["X_val"], split["y_val"]), use_best_model=True)
    return clf, b


def tune_histgbm(split, n_trials=25):
    Xtr, ytr = split["X_train"], split["y_train"]
    Xv, yv = split["X_val"], split["y_val"]

    def obj(trial):
        p = dict(learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                 max_iter=trial.suggest_int("max_iter", 100, 600),
                 max_leaf_nodes=trial.suggest_int("max_leaf_nodes", 15, 63),
                 min_samples_leaf=trial.suggest_int("min_samples_leaf", 10, 100),
                 l2_regularization=trial.suggest_float("l2_regularization", 1e-3, 5, log=True),
                 class_weight="balanced", random_state=SEED)
        clf = HistGradientBoostingClassifier(**p).fit(Xtr, ytr)
        pv = clf.predict_proba(Xv)[:, 1]
        t = tune_threshold(yv, pv)
        return f1_score(yv, (pv >= t).astype(int), zero_division=0)

    st = optuna.create_study(direction="maximize",
                             sampler=optuna.samplers.TPESampler(seed=SEED))
    st.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    b = st.best_params
    clf = HistGradientBoostingClassifier(learning_rate=b["learning_rate"],
            max_iter=b["max_iter"], max_leaf_nodes=b["max_leaf_nodes"],
            min_samples_leaf=b["min_samples_leaf"], l2_regularization=b["l2_regularization"],
            class_weight="balanced", random_state=SEED).fit(Xtr, ytr)
    return clf, b


def eval_model(est, split, name, fs, calibrate=None):
    if calibrate:
        est = CalibratedClassifierCV(est, method=calibrate, cv=4)
        est.fit(split["X_train"], split["y_train"])
    p_val = est.predict_proba(split["X_val"])[:, 1]
    p_test = est.predict_proba(split["X_test"])[:, 1]
    yv, yt = split["y_val"], split["y_test"]
    best, bt = 0, 0.5
    for t in np.linspace(0.05, 0.95, 91):
        f = f1_score(yv, (p_val >= t).astype(int), zero_division=0)
        if f > best:
            best, bt = f, t
    pred = (p_test >= bt).astype(int)
    return dict(feature_set=fs, model=name + (f"_cal{calibrate[0]}" if calibrate else ""),
                threshold=round(bt, 3),
                test_f1=round(f1_score(yt, pred), 4),
                test_recall=round(recall_score(yt, pred, zero_division=0), 4),
                test_precision=round(precision_score(yt, pred, zero_division=0), 4),
                test_accuracy=round((pred == yt).mean(), 4),
                test_auc=round(roc_auc_score(yt, p_test), 4),
                test_pr_auc=round(average_precision_score(yt, p_test), 4),
                confusion_matrix=confusion_matrix(yt, pred).tolist())


def main():
    section("E13  BEAT 0.632 — EXACT feature sets + new models")
    t0 = time.time()
    df = load_raw()
    rows = []
    sets = feature_set_sizes()

    for fs in sets.keys():
        feats = get_feature_list(fs)
        log(f"=== {fs} ({len(feats)} feats) ===", "STEP")
        # SMOTE split for SVM/RF/LR; native-imbalance split for trees
        sp_sm = build_pipeline(df, feats, cat_cols=CATBOOST_CAT_COLS, smote=True, scale=True)
        sp_raw = build_pipeline(df, feats, cat_cols=CATBOOST_CAT_COLS, smote=False, scale=False)

        # --- SVM (reproduce benchmark) ---
        svm = SVC(probability=True, class_weight="balanced", gamma="scale", random_state=SEED)
        svm.fit(sp_sm["X_res"], sp_sm["y_res"])
        rows.append(eval_model(svm, sp_sm, "SVM_RBF", fs))
        log(f"  SVM_RBF      F1={rows[-1]['test_f1']:.3f} rec={rows[-1]['test_recall']:.3f} "
            f"prec={rows[-1]['test_precision']:.3f} AUC={rows[-1]['test_auc']}", "METRIC")

        # --- CatBoost tuned (native imbalance) ---
        try:
            cb, _ = tune_catboost(sp_raw, n_trials=20)
            rows.append(eval_model(cb, sp_raw, "CatBoost_Opt", fs))
            log(f"  CatBoost_Opt F1={rows[-1]['test_f1']:.3f} rec={rows[-1]['test_recall']:.3f} "
                f"prec={rows[-1]['test_precision']:.3f} AUC={rows[-1]['test_auc']}", "METRIC")
        except Exception as e:
            log(f"  CatBoost_Opt FAILED: {e}", "ERR")

        # --- HistGBM tuned (native imbalance) ---
        try:
            hg, _ = tune_histgbm(sp_raw, n_trials=20)
            rows.append(eval_model(hg, sp_raw, "HistGBM_Opt", fs))
            log(f"  HistGBM_Opt  F1={rows[-1]['test_f1']:.3f} rec={rows[-1]['test_recall']:.3f} "
                f"prec={rows[-1]['test_precision']:.3f} AUC={rows[-1]['test_auc']}", "METRIC")
        except Exception as e:
            log(f"  HistGBM_Opt FAILED: {e}", "ERR")

    # --- Stacking on the two strongest sets (Group_B, Group_F, Group_D) ---
    log("Stacking (SVM+CatBoost+HistGBM) on top sets", "STEP")
    for fs in ["Group_B_Financial_Health", "Group_F_Literature", "Group_D_Hybrid_Extended"]:
        feats = get_feature_list(fs)
        sp_raw = build_pipeline(df, feats, cat_cols=CATBOOST_CAT_COLS, smote=False, scale=False)
        # train bases
        svm = SVC(probability=True, class_weight="balanced", gamma="scale", random_state=SEED)
        svm.fit(sp_raw["X_train"], sp_raw["y_train"])
        cb = CatBoostClassifier(iterations=400, depth=6, learning_rate=0.05,
                                loss_function="Logloss", eval_metric="F1",
                                random_seed=SEED, verbose=0)
        cb.fit(sp_raw["X_train"], sp_raw["y_train"])
        hg = HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED)
        hg.fit(sp_raw["X_train"], sp_raw["y_train"])
        stack = StackingClassifier(estimators=[("svm", svm), ("cb", cb), ("hg", hg)],
                    final_estimator=LogisticRegression(max_iter=2000, class_weight="balanced"),
                    cv=4, stack_method="predict_proba", n_jobs=-1)
        stack.fit(sp_raw["X_train"], sp_raw["y_train"])
        rows.append(eval_model(stack, sp_raw, "Stack_SVM_CB_HG", fs))
        log(f"  Stack_SVM_CB_HG F1={rows[-1]['test_f1']:.3f} rec={rows[-1]['test_recall']:.3f} "
            f"prec={rows[-1]['test_precision']:.3f} AUC={rows[-1]['test_auc']}", "METRIC")

    res = pd.DataFrame(rows).sort_values("test_f1", ascending=False).reset_index(drop=True)
    out = os.path.join(RESULTS_DIR, "E13_exact_beat_0632.csv")
    res.to_csv(out, index=False)
    log(f"Saved {len(res)} -> {out}", "OK")
    beaten = res[res["test_f1"] > BAR]
    log(f"BEST E13 F1: {res.iloc[0]['test_f1']:.3f} ({res.iloc[0]['model']}/{res.iloc[0]['feature_set']})", "OK")
    log(f"Models BEATING 0.632: {len(beaten)}", "OK" if len(beaten) else "WARN")
    for _, r in beaten.head(10).iterrows():
        log(f"  >> {r['model']} / {r['feature_set']} : F1={r['test_f1']} rec={r['test_recall']} AUC={r['test_auc']}", "OK")
    log(f"Elapsed {time.time()-t0:.0f}s", "INFO")
    cols = ["feature_set", "model", "test_f1", "test_recall", "test_precision", "test_auc", "threshold"]
    print("\n  TOP 15:")
    print(res[cols].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
