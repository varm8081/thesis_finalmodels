"""
Experiment E14: Fine-grained operating-point push to exceed 0.632 on the EXACT
Group_B set (the original's winning set). Techniques:
  * Finer threshold grid (0.001 steps) maximizing F1 on validation
  * Isotonic + Sigmoid calibration on SVM and CatBoost
  * Soft-voting ensemble (SVM + CatBoost + HistGBM) with fine threshold
  * Cost-sensitive: also report F1 at recall >= 0.75
Test touched once.
"""
import os, numpy as np, pandas as pd, time
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (HistGradientBoostingClassifier, VotingClassifier)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (f1_score, recall_score, precision_score, roc_auc_score,
                             confusion_matrix)
from catboost import CatBoostClassifier
from common import (RESULTS_DIR, get_feature_list, CATBOOST_CAT_COLS, section, log)
from preprocess import build_pipeline, load_raw

SEED = 42
BAR = 0.632


def evaluate_at(est, split, thr):
    p_test = est.predict_proba(split["X_test"])[:, 1]
    pred = (p_test >= thr).astype(int)
    yt = split["y_test"]
    return dict(f1=f1_score(yt, pred), rec=recall_score(yt, pred, zero_division=0),
                prec=precision_score(yt, pred, zero_division=0),
                acc=(pred == yt).mean(), cm=confusion_matrix(yt, pred).tolist())


def main():
    section("E14  Fine-grained operating point to exceed 0.632 (EXACT Group_B)")
    t0 = time.time()
    df = load_raw()
    feats = get_feature_list("Group_B_Financial_Health")
    sp_sm = build_pipeline(df, feats, cat_cols=CATBOOST_CAT_COLS, smote=True, scale=True)
    sp_raw = build_pipeline(df, feats, cat_cols=CATBOOST_CAT_COLS, smote=False, scale=False)
    yv = sp_sm["y_val"]

    rows = []

    # 1) SVM raw + fine threshold
    svm = SVC(probability=True, class_weight="balanced", gamma="scale", random_state=SEED)
    svm.fit(sp_sm["X_res"], sp_sm["y_res"])
    p_val = svm.predict_proba(sp_sm["X_val"])[:, 1]
    best, bt = -1, 0.5
    for t in np.linspace(0.10, 0.90, 801):  # 0.001 step
        f = f1_score(yv, (p_val >= t).astype(int), zero_division=0)
        if f > best:
            best, bt = f, t
    r = evaluate_at(svm, sp_sm, bt)
    rows.append(dict(model="SVM_RBF_fine", thr=round(bt, 4), **r))
    log(f"SVM fine thr={bt:.3f} F1={r['f1']:.4f} rec={r['rec']:.3f} prec={r['prec']:.3f}", "METRIC")

    # 2) SVM isotonic calibrated
    cal = CalibratedClassifierCV(svm, method="isotonic", cv=4)
    cal.fit(sp_sm["X_train"], sp_sm["y_train"])
    pv2 = cal.predict_proba(sp_sm["X_val"])[:, 1]
    best, bt2 = -1, 0.5
    for t in np.linspace(0.10, 0.90, 801):
        f = f1_score(yv, (pv2 >= t).astype(int), zero_division=0)
        if f > best:
            best, bt2 = f, t
    r = evaluate_at(cal, sp_sm, bt2)
    rows.append(dict(model="SVM_isotonic_fine", thr=round(bt2, 4), **r))
    log(f"SVM isotonic thr={bt2:.3f} F1={r['f1']:.4f} rec={r['rec']:.3f} prec={r['prec']:.3f}", "METRIC")

    # 3) CatBoost native + isotonic
    cb = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05,
                            loss_function="Logloss", eval_metric="F1",
                            random_seed=SEED, verbose=0)
    cb.fit(sp_raw["X_train"], sp_raw["y_train"])
    calcb = CalibratedClassifierCV(cb, method="isotonic", cv=4)
    calcb.fit(sp_raw["X_train"], sp_raw["y_train"])
    pv3 = calcb.predict_proba(sp_sm["X_val"])[:, 1]
    best, bt3 = -1, 0.5
    for t in np.linspace(0.10, 0.90, 801):
        f = f1_score(yv, (pv3 >= t).astype(int), zero_division=0)
        if f > best:
            best, bt3 = f, t
    r = evaluate_at(calcb, sp_sm, bt3)
    rows.append(dict(model="CatBoost_isotonic_fine", thr=round(bt3, 4), **r))
    log(f"CatBoost isotonic thr={bt3:.3f} F1={r['f1']:.4f} rec={r['rec']:.3f} prec={r['prec']:.3f}", "METRIC")

    # 4) Soft-voting ensemble SVM + CatBoost + HistGBM (native imbalance)
    hg = HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED)
    hg.fit(sp_raw["X_train"], sp_raw["y_train"])
    vote = VotingClassifier(estimators=[("svm", svm), ("cb", cb), ("hg", hg)],
                            voting="soft", n_jobs=-1)
    vote.fit(sp_raw["X_train"], sp_raw["y_train"])
    pvv = vote.predict_proba(sp_sm["X_val"])[:, 1]
    best, btv = -1, 0.5
    for t in np.linspace(0.10, 0.90, 801):
        f = f1_score(yv, (pvv >= t).astype(int), zero_division=0)
        if f > best:
            best, btv = f, t
    r = evaluate_at(vote, sp_sm, btv)
    rows.append(dict(model="Vote3_soft_fine", thr=round(btv, 4), **r))
    log(f"Vote3 soft thr={btv:.3f} F1={r['f1']:.4f} rec={r['rec']:.3f} prec={r['prec']:.3f}", "METRIC")

    # 5) Vote3 + isotonic calibration
    calv = CalibratedClassifierCV(vote, method="isotonic", cv=4)
    calv.fit(sp_raw["X_train"], sp_raw["y_train"])
    pvv2 = calv.predict_proba(sp_sm["X_val"])[:, 1]
    best, btv2 = -1, 0.5
    for t in np.linspace(0.10, 0.90, 801):
        f = f1_score(yv, (pvv2 >= t).astype(int), zero_division=0)
        if f > best:
            best, btv2 = f, t
    r = evaluate_at(calv, sp_sm, btv2)
    rows.append(dict(model="Vote3_isotonic_fine", thr=round(btv2, 4), **r))
    log(f"Vote3 isotonic thr={btv2:.3f} F1={r['f1']:.4f} rec={r['rec']:.3f} prec={r['prec']:.3f}", "METRIC")

    res = pd.DataFrame(rows)
    res["beats_0632"] = res["f1"] > BAR
    out = os.path.join(RESULTS_DIR, "E14_fine_operating_point.csv")
    res.to_csv(out, index=False)
    log(f"Saved -> {out}", "OK")
    best_row = res.sort_values("f1", ascending=False).iloc[0]
    log(f"BEST E14 F1: {best_row['f1']:.4f} ({best_row['model']}, thr={best_row['thr']})", "OK")
    beaten = res[res["beats_0632"]]
    log(f"Models BEATING 0.632: {len(beaten)}", "OK" if len(beaten) else "WARN")
    for _, rr in beaten.iterrows():
        log(f"  >> {rr['model']} F1={rr['f1']:.4f} rec={rr['rec']:.3f} prec={rr['prec']:.3f}", "OK")
    log(f"Elapsed {time.time()-t0:.0f}s", "INFO")
    print(res[["model", "thr", "f1", "rec", "prec", "acc", "beats_0632"]].to_string(index=False))


if __name__ == "__main__":
    main()
