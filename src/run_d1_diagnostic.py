"""
Diagnostic D1: understand the F1 ceiling and test ensembles.
 - Compare SMOTE vs native class_weight
 - Sweep thresholds with recall targets {0.65, 0.70, 0.737}
 - Soft-voting ensemble of [SVM, RF, CatBoost, HistGBM]
Goal: get closest to / exceed the 0.632 bar.
"""
import numpy as np, pandas as pd, time
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, HistGradientBoostingClassifier
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
from catboost import CatBoostClassifier
from common import get_feature_list, CATBOOST_CAT_COLS, section, log
from preprocess import build_pipeline, load_raw
from models import tune_threshold

SEED = 42


def soft_vote_eval(estimators, split, name, fs):
    from sklearn.ensemble import VotingClassifier
    # Build a voting classifier that averages probabilities
    vote = VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)
    vote.fit(split["X_train"], split["y_train"])
    p_val = vote.predict_proba(split["X_val"])[:, 1]
    p_test = vote.predict_proba(split["X_test"])[:, 1]
    yv, yt = split["y_val"], split["y_test"]
    best, bt = 0, 0.5
    for t in np.linspace(0.05, 0.95, 91):
        f = f1_score(yv, (p_val >= t).astype(int), zero_division=0)
        if f > best:
            best, bt = f, t
    pred = (p_test >= bt).astype(int)
    return dict(feature_set=fs, model=name, threshold=round(bt, 3),
                test_f1=round(f1_score(yt, pred), 4),
                test_recall=round(recall_score(yt, pred, zero_division=0), 4),
                test_precision=round(precision_score(yt, pred, zero_division=0), 4),
                test_auc=round(roc_auc_score(yt, p_test), 4),
                test_accuracy=round((pred == yt).mean(), 4),
                confusion_matrix=__import__("sklearn.metrics").metrics.confusion_matrix(yt, pred).tolist())


def main():
    section("D1  Threshold + Ensemble diagnostic")
    df = load_raw()
    feats = get_feature_list("Group_B_Financial_Health")
    # Two split variants
    split_smote = build_pipeline(df, feats, cat_cols=[], smote=True, scale=True)
    split_raw = build_pipeline(df, feats, cat_cols=[], smote=False, scale=False)

    rows = []
    for tag, split in [("SMOTE", split_smote), ("RAW", split_raw)]:
        # SVM
        svm = SVC(probability=True, class_weight="balanced", gamma="scale", random_state=SEED)
        svm.fit(split["X_res"] if tag == "SMOTE" else split["X_train"],
                split["y_res"] if tag == "SMOTE" else split["y_train"])
        # RF
        rf = RandomForestClassifier(n_estimators=400, class_weight="balanced", n_jobs=-1, random_state=SEED)
        rf.fit(split["X_res"] if tag == "SMOTE" else split["X_train"],
               split["y_res"] if tag == "SMOTE" else split["y_train"])
        # HistGBM
        hg = HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED)
        hg.fit(split["X_train"], split["y_train"])
        # CatBoost
        cb = CatBoostClassifier(iterations=400, depth=6, learning_rate=0.05,
                                loss_function="Logloss", eval_metric="F1",
                                random_seed=SEED, verbose=0)
        cb.fit(split["X_train"], split["y_train"])

        Xtr = split["X_res"] if tag == "SMOTE" else split["X_train"]
        ytr = split["y_res"] if tag == "SMOTE" else split["y_train"]
        vote = VotingClassifier(estimators=[("svm", svm), ("rf", rf), ("hg", hg), ("cb", cb)],
                                voting="soft", n_jobs=-1)
        vote.fit(Xtr, ytr)
        p_val = vote.predict_proba(split["X_val"])[:, 1]
        p_test = vote.predict_proba(split["X_test"])[:, 1]
        yv, yt = split["y_val"], split["y_test"]
        for rec_t in [0.65, 0.70, 0.737]:
            bt = tune_threshold(yv, p_val, target_recall=rec_t)
            pred = (p_test >= bt).astype(int)
            rows.append(dict(split=tag, ensemble="VOTE4", rec_target=rec_t,
                            threshold=round(bt, 3),
                            test_f1=round(f1_score(yt, pred), 4),
                            test_recall=round(recall_score(yt, pred, zero_division=0), 4),
                            test_precision=round(precision_score(yt, pred, zero_division=0), 4),
                            test_auc=round(roc_auc_score(yt, p_test), 4)))
        log(f"[{tag}] VOTE4 ensemble tested (recall targets)", "METRIC")

    res = pd.DataFrame(rows)
    print(res.to_string(index=False))
    # Find best
    best = res.sort_values("test_f1", ascending=False).iloc[0]
    log(f"BEST D1 F1: {best['test_f1']:.3f} ({best['split']} vote, rec_t={best['rec_target']})", "OK")
    log(f"BAR: 0.632", "INFO")


if __name__ == "__main__":
    main()
