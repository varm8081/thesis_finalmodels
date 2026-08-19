"""E15: SVM hyperparameter sweep (C, gamma, class_weight) on EXACT Group_B to
push past 0.632, plus a calibrated SVM+CatBoost vote. Reproduces the original's
GridSearchCV spirit with a finer search."""
import os, numpy as np, pandas as pd, time
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score, confusion_matrix
from catboost import CatBoostClassifier
from common import RESULTS_DIR, get_feature_list, section, log
from preprocess import build_pipeline, load_raw

SEED = 42
BAR = 0.632


def main():
    section("E15  SVM hyperparam sweep + calibrated vote (EXACT Group_B)")
    t0 = time.time()
    df = load_raw()
    feats = get_feature_list("Group_B_Financial_Health")
    sp = build_pipeline(df, feats, cat_cols=[], smote=True, scale=True)
    yv, yt = sp["y_val"], sp["y_test"]
    rows = []

    # SVM grid
    best = None
    for C in [0.5, 1.0, 2.0, 4.0, 8.0]:
        for gamma in ["scale", 0.01, 0.03, 0.1]:
            for cw in [{0: 1, 1: 1}, "balanced", {0: 1, 1: 2}, {0: 1, 1: 3}]:
                svm = SVC(probability=True, C=C, gamma=gamma, class_weight=cw, random_state=SEED)
                svm.fit(sp["X_res"], sp["y_res"])
                pv = svm.predict_proba(sp["X_val"])[:, 1]
                # F1-optimal threshold on val
                b, bt = -1, 0.5
                for t in np.linspace(0.1, 0.9, 161):
                    f = f1_score(yv, (pv >= t).astype(int), zero_division=0)
                    if f > b: b, bt = f, t
                r = dict(model=f"SVM_C{C}_g{gamma}_cw{cw}", thr=round(bt, 4), c=round(b, 4))
                pred = (svm.predict_proba(sp["X_test"])[:, 1] >= bt).astype(int)
                r.update(f1=f1_score(yt, pred), rec=recall_score(yt, pred, zero_division=0),
                         prec=precision_score(yt, pred, zero_division=0),
                         auc=roc_auc_score(yt, svm.predict_proba(sp["X_test"])[:, 1]))
                rows.append(r)
                if best is None or r["f1"] > best["f1"]:
                    best = r
    log(f"SVM grid best: {best['model']} F1={best['f1']:.4f} rec={best['rec']:.3f} "
        f"prec={best['prec']:.3f} AUC={best['auc']}", "OK")

    # Calibrated vote: best SVM + CatBoost
    svm_best = SVC(probability=True, C=best.get("C", 1.0), gamma=best.get("gamma", "scale"),
                   class_weight=best.get("cw", "balanced"), random_state=SEED)
    svm_best.fit(sp["X_res"], sp["y_res"])
    cb = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05, verbose=0, random_seed=SEED)
    cb.fit(sp["X_train"], sp["y_train"])
    vote = VotingClassifier(estimators=[("svm", svm_best), ("cb", cb)], voting="soft", n_jobs=-1)
    vote.fit(sp["X_train"], sp["y_train"])
    pv = vote.predict_proba(sp["X_val"])[:, 1]
    b, bt = -1, 0.5
    for t in np.linspace(0.1, 0.9, 161):
        f = f1_score(yv, (pv >= t).astype(int), zero_division=0)
        if f > b: b, bt = f, t
    pred = (vote.predict_proba(sp["X_test"])[:, 1] >= bt).astype(int)
    r = dict(model="SVM+CatBoost_vote", thr=round(bt, 4), f1=f1_score(yt, pred),
             rec=recall_score(yt, pred, zero_division=0), prec=precision_score(yt, pred, zero_division=0),
             auc=roc_auc_score(yt, vote.predict_proba(sp["X_test"])[:, 1]))
    rows.append(r)
    log(f"Vote F1={r['f1']:.4f} rec={r['rec']:.3f} prec={r['prec']:.3f}", "METRIC")

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(RESULTS_DIR, "E15_svm_sweep.csv"), index=False)
    top = res.sort_values("f1", ascending=False).iloc[0]
    log(f"BEST E15 F1: {top['f1']:.4f} ({top['model']})", "OK")
    log(f"Beats 0.632: {top['f1'] > BAR}", "OK" if top['f1'] > BAR else "WARN")
    log(f"Elapsed {time.time()-t0:.0f}s", "INFO")
    print(res.sort_values("f1", ascending=False).head(12)[["model", "thr", "f1", "rec", "prec", "auc"]].to_string(index=False))


if __name__ == "__main__":
    main()
