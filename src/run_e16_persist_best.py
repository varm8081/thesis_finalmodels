"""
E16: Lock in the WINNING model and persist it.
Best = SVM(C=2.0, gamma=0.03, class_weight={0:1,1:3}) on EXACT Group_B_Financial_Health.
Saves:
  models/best_distress_model.joblib  - the fitted pipeline + SVM
  results/best_model_summary.json   - metrics + confusion matrix + params
  results/best_features_importance.csv - SVM coefficients (signed) as importance
Then regenerate figures.
"""
import os, json, numpy as np, pandas as pd, joblib
from sklearn.svm import SVC
from sklearn.metrics import (f1_score, recall_score, precision_score, roc_auc_score,
                             confusion_matrix)
from common import RESULTS_DIR, MODELS_DIR, get_feature_list, section, log
from preprocess import build_pipeline, load_raw

SEED = 42
WIN_PARAMS = dict(C=2.0, gamma=0.03, class_weight={0: 1, 1: 3}, probability=True,
                  random_state=SEED)


def main():
    section("E16  Persist winning model (SVM cost-sensitive / Group_B)")
    df = load_raw()
    feats = get_feature_list("Group_B_Financial_Health")
    split = build_pipeline(df, feats, cat_cols=[], smote=True, scale=True)

    svm = SVC(**WIN_PARAMS)
    svm.fit(split["X_res"], split["y_res"])

    p_test = svm.predict_proba(split["X_test"])[:, 1]
    # F1-optimal threshold on val
    yv = split["y_val"]
    best, bt = -1, 0.5
    for t in np.linspace(0.1, 0.9, 161):
        f = f1_score(yv, (svm.predict_proba(split["X_val"])[:, 1] >= t).astype(int), zero_division=0)
        if f > best: best, bt = f, t
    pred = (p_test >= bt).astype(int)
    yt = split["y_test"]
    cm = confusion_matrix(yt, pred)
    summary = dict(
        model="SVM_RBF_cost_sensitive",
        params=WIN_PARAMS,
        feature_set="Group_B_Financial_Health",
        n_features=len(feats),
        threshold=round(float(bt), 4),
        test_f1=round(float(f1_score(yt, pred)), 4),
        test_recall=round(float(recall_score(yt, pred, zero_division=0)), 4),
        test_precision=round(float(precision_score(yt, pred, zero_division=0)), 4),
        test_accuracy=round(float((pred == yt).mean()), 4),
        test_auc=round(float(roc_auc_score(yt, p_test)), 4),
        confusion_matrix=cm.tolist(),
        n_test=int(len(yt)), n_test_distress=int(yt.sum()),
        beats_baseline_0632=bool(f1_score(yt, pred) > 0.632),
    )
    # Save model bundle (transformers + svm + threshold)
    bundle = dict(imputer=split["imputer"], scaler=split["scaler"],
                  features=feats, svm=svm, threshold=float(bt))
    joblib.dump(bundle, os.path.join(MODELS_DIR, "best_distress_model.joblib"))
    with open(os.path.join(RESULTS_DIR, "best_model_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Feature importance: RBF has no coef_, so fit a LINEAR-kernel SVM on the
    # same scaled train to get signed, interpretable feature weights (proxy).
    from sklearn.svm import LinearSVC
    lin = LinearSVC(C=WIN_PARAMS["C"], class_weight=WIN_PARAMS["class_weight"],
                    random_state=SEED, max_iter=5000)
    lin.fit(split["X_res"], split["y_res"])
    coef = lin.coef_[0]
    imp = pd.DataFrame({"feature": feats, "svm_linear_coef": coef,
                        "abs_coef": np.abs(coef)}).sort_values("abs_coef", ascending=False)
    imp.to_csv(os.path.join(RESULTS_DIR, "best_features_importance.csv"), index=False)

    log(f"F1={summary['test_f1']} rec={summary['test_recall']} prec={summary['test_precision']} "
        f"AUC={summary['test_auc']} thr={summary['threshold']}", "OK")
    log(f"Beats 0.632: {summary['beats_baseline_0632']}", "OK" if summary['beats_baseline_0632'] else "WARN")
    log(f"Saved model -> models/best_distress_model.joblib", "OK")
    log("Top 8 features by |SVM weight|:", "INFO")
    print(imp.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
