"""Consolidate all experiment CSVs into results/ALL_EXPERIMENTS.csv with a
uniform schema: experiment, feature_set, model, test_f1, test_recall,
test_precision, test_auc, threshold, confusion_matrix. Handles the alternate
column names used by E14/E15 (f1, rec, prec, auc, thr)."""
import os
import pandas as pd
from common import RESULTS_DIR

CSV_FILES = [
    "E8_benchmark_new_sweep.csv", "E9_new_approaches.csv", "E10_final_push.csv",
    "E11_threshold_ceiling.csv", "E12_final_operating_point.csv",
    "E13_exact_beat_0632.csv", "E14_fine_operating_point.csv", "E15_svm_sweep.csv",
]


def norm(df, exp):
    out = pd.DataFrame()
    out["experiment"] = [exp] * len(df)
    # feature_set
    if "feature_set" in df.columns:
        out["feature_set"] = df["feature_set"]
    elif exp in ("E14", "E15"):
        out["feature_set"] = "Group_B_Financial_Health"
    else:
        out["feature_set"] = None
    # model
    out["model"] = df["model"] if "model" in df.columns else df.get("op")
    # metrics (handle both test_* and bare names)
    def col(*names):
        for n in names:
            if n in df.columns:
                return df[n]
        return None
    out["test_f1"] = col("test_f1", "f1")
    out["test_recall"] = col("test_recall", "rec")
    out["test_precision"] = col("test_precision", "prec")
    out["test_auc"] = col("test_auc", "auc")
    out["threshold"] = col("threshold", "thr")
    out["confusion_matrix"] = col("confusion_matrix", "cm")
    return out


def main():
    frames = []
    for fn in CSV_FILES:
        p = os.path.join(RESULTS_DIR, fn)
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p)
        exp = fn.split("_")[0]  # E8..E15
        frames.append(norm(d, exp))
    allp = pd.concat(frames, ignore_index=True)
    allp["test_f1"] = pd.to_numeric(allp["test_f1"], errors="coerce")
    allp = allp.sort_values("test_f1", ascending=False).reset_index(drop=True)
    out = os.path.join(RESULTS_DIR, "ALL_EXPERIMENTS.csv")
    allp.to_csv(out, index=False)
    print(f"Consolidated {len(allp)} rows -> {out}")
    print("\nTOP 8:")
    print(allp[["experiment", "feature_set", "model", "test_f1",
                "test_recall", "test_precision", "test_auc"]].head(8).to_string(index=False))


if __name__ == "__main__":
    main()
