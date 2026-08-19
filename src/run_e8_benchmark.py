"""
Experiment E8: Benchmark re-run + NEW model sweep for Class 0 distress.

Strategy:
  1. Re-run the original 3 benchmark models (LogReg, RF, SVM) across all 7
     feature sets to confirm we reproduce the F1~0.632 bar on SVM/Group_B.
  2. Introduce the NEW models (CatBoost, HistGBM) -- never tried in the
     original project -- across the same sets.
  3. Save a tidy results table to results/.
"""
import os, json, time
import pandas as pd
from common import (ROOT, RESULTS_DIR, feature_set_sizes, get_feature_list,
                    CATBOOST_CAT_COLS, section, log)
from preprocess import build_pipeline, load_raw
from models import fit_model, evaluate

MODEL_NAMES = ["LogReg", "RandomForest", "SVM_RBF", "CatBoost", "HistGBM"]
SEED = 42


def main():
    section("E8  Class-0 Distress: Benchmark + New-Model Sweep")
    t0 = time.time()
    df = load_raw()
    rows = []
    for fs_name in feature_set_sizes().keys():
        feats = get_feature_list(fs_name)
        log(f"Feature set: {fs_name} ({len(feats)} features)", "STEP")
        split = build_pipeline(df, feats, cat_cols=CATBOOST_CAT_COLS)
        for m in MODEL_NAMES:
            try:
                est = fit_model(m, split, seed=SEED)
            except Exception as e:
                log(f"  {m} FAILED: {e}", "ERR")
                continue
            r = evaluate(est, split, m, fs_name,
                         cat_cols=CATBOOST_CAT_COLS)
            rows.append(r)
            log(f"  {m:13s} F1={r['test_f1']:.3f} recall={r['test_recall']:.3f} "
                f"prec={r['test_precision']:.3f} AUC={r['test_auc']} thr={r['threshold']}",
                "METRIC")

    res = pd.DataFrame(rows)
    out_csv = os.path.join(RESULTS_DIR, "E8_benchmark_new_sweep.csv")
    res = res.sort_values("test_f1", ascending=False).reset_index(drop=True)
    res.to_csv(out_csv, index=False)
    log(f"Saved {len(res)} results -> {out_csv}", "OK")
    log(f"Best F1 this sweep: {res.iloc[0]['test_f1']:.3f} "
        f"({res.iloc[0]['model']} / {res.iloc[0]['feature_set']})", "OK")
    log(f"Baseline bar to beat: SVM_RBF on Group_B = 0.632", "INFO")
    log(f"Elapsed {time.time()-t0:.0f}s", "INFO")

    # Top 12 preview
    print("\n  TOP 12 by Test F1:")
    cols = ["feature_set", "model", "test_f1", "test_recall",
            "test_precision", "test_auc", "threshold"]
    print(res[cols].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
