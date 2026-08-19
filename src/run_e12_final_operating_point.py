"""
Experiment E12: Final attempt to exceed 0.632 -- calibrated, recall-optimized
operating point on the strongest candidate (SVM-RBF / Group_F_Literature).

Approach:
  * Isotonic-calibrate SVM probabilities (more reliable distress probs)
  * Choose threshold on VALIDATION that maximizes F1 while keeping recall>=0.70
  * Optionally use a cost-sensitive threshold (weight false-negatives 2x)
Report the single best honest operating point.
"""
import os, numpy as np, pandas as pd, time
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (f1_score, recall_score, precision_score, roc_auc_score,
                             confusion_matrix)
from common import get_feature_list, section, log
from preprocess import build_pipeline, load_raw

SEED = 42


def main():
    section("E12  Calibrated recall-optimized operating point (final)")
    t0 = time.time()
    df = load_raw()
    feats = get_feature_list("Group_F_Literature")
    split = build_pipeline(df, feats, cat_cols=[], smote=True, scale=True)

    # Calibrated SVM
    svm = SVC(probability=True, class_weight="balanced", gamma="scale", random_state=SEED)
    svm.fit(split["X_res"], split["y_res"])
    cal = CalibratedClassifierCV(svm, method="isotonic", cv=4)
    cal.fit(split["X_train"], split["y_train"])

    p_val = cal.predict_proba(split["X_val"])[:, 1]
    p_test = cal.predict_proba(split["X_test"])[:, 1]
    yv, yt = split["y_val"], split["y_test"]

    # Threshold selection: maximize F1 on val with recall floor, OR pick the F1-max
    grids = np.linspace(0.01, 0.99, 99)
    best, bt = -1, 0.5
    best_rec_first = -1
    for t in grids:
        pred_v = (p_val >= t).astype(int)
        f1 = f1_score(yv, pred_v, zero_division=0)
        rec = recall_score(yv, pred_v, zero_division=0)
        if rec >= 0.70 and f1 > best:
            best, bt = f1, t
        if rec >= 0.737 and best_rec_first < 0:
            best_rec_first = f1  # record F1 at first threshold meeting 0.737
    # absolute F1 max (no recall floor)
    abs_best, abs_t = -1, 0.5
    for t in grids:
        f1 = f1_score(yv, (p_val >= t).astype(int), zero_division=0)
        if f1 > abs_best:
            abs_best, abs_t = f1, t

    # Evaluate both operating points on test
    def eval_at(t):
        pred = (p_test >= t).astype(int)
        return dict(t=round(t, 3), f1=round(f1_score(yt, pred), 4),
                    rec=round(recall_score(yt, pred, zero_division=0), 4),
                    prec=round(precision_score(yt, pred, zero_division=0), 4),
                    acc=round((pred == yt).mean(), 4),
                    auc=round(roc_auc_score(yt, p_test), 4),
                    cm=confusion_matrix(yt, pred).tolist())
    r_rec = eval_at(bt)
    r_abs = eval_at(abs_t)
    log(f"Validation F1-max@rec>=0.70 -> test F1={r_rec['f1']} rec={r_rec['rec']} "
        f"prec={r_rec['prec']} AUC={r_rec['auc']} thr={r_rec['t']}", "METRIC")
    log(f"Validation absolute F1-max   -> test F1={r_abs['f1']} rec={r_abs['rec']} "
        f"prec={r_abs['prec']} AUC={r_abs['auc']} thr={r_abs['t']}", "METRIC")

    out = pd.DataFrame([{"op": "recall>=0.70_tuned", **r_rec},
                        {"op": "abs_F1max_tuned", **r_abs}])
    out.to_csv("results/E12_final_operating_point.csv", index=False)
    log("Saved results/E12_final_operating_point.csv", "OK")
    log(f"BAR = 0.632. Best test F1 here = {max(r_rec['f1'], r_abs['f1'])}", "INFO")
    log(f"Elapsed {time.time()-t0:.0f}s", "INFO")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
