"""Load the EXACT original feature sets from 07_feature_sets.json and validate
against the actual columns in merged_clean.xlsx. Also reconcile name mismatches
(e.g. X4_book vs X4, Total_Debt_to_Equity vs Debt_to_Equity)."""
import os, json, pandas as pd
from common import DATA_RAW, ROOT

JSON_PATH = os.path.join(DATA_RAW, "07_feature_sets.json")


def load_json():
    with open(JSON_PATH) as f:
        return json.load(f)


def validate(df_cols):
    js = load_json()
    sets = js["feature_sets"]
    missing_global = {}
    report = {}
    for g, feats in sets.items():
        miss = [c for c in feats if c not in df_cols]
        report[g] = {"n": len(feats), "present": len(feats) - len(miss),
                     "missing": miss}
        if miss:
            missing_global[g] = miss
    return sets, report


if __name__ == "__main__":
    df = pd.read_excel(os.path.join(DATA_RAW, "merged_clean.xlsx"), sheet_name=0)
    cols = list(df.columns)
    sets, report = validate(cols)
    print("DATA COLS:", len(cols))
    for g, r in report.items():
        print(f"  {g:28s} n={r['n']:2d} present={r['present']:2d} missing={r['missing']}")
    # Save the raw sets for downstream use
    out = os.path.join(ROOT, "src", "_orig_feature_sets.json")
    with open(out, "w") as f:
        json.dump(sets, f, indent=2)
    print("Saved exact sets ->", out)
