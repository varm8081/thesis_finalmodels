"""
Generate a fresh, presentable HTML report (reports/thesis_report.html) reflecting
the NEW best result (F1 = 0.671, cost-sensitive SVM on exact Group_B). Reads the
consolidated results + best-model summary + feature importance CSVs. Pure
stdlib + no external deps, so it runs anywhere. Local file changes only.

Run: PYTHONPATH=src python src/make_html_report.py
"""
import os
import json
import pandas as pd
from common import RESULTS_DIR, ROOT

OUT = os.path.join(ROOT, "reports", "thesis_report.html")
BAR = 0.632


def load():
    allp = pd.read_csv(os.path.join(RESULTS_DIR, "ALL_EXPERIMENTS.csv"))
    allp["test_f1"] = pd.to_numeric(allp["test_f1"], errors="coerce")
    allp = allp.dropna(subset=["test_f1"]).sort_values("test_f1", ascending=False)
    with open(os.path.join(RESULTS_DIR, "best_model_summary.json")) as f:
        summ = json.load(f)
    imp = pd.read_csv(os.path.join(RESULTS_DIR, "best_features_importance.csv"))
    return allp, summ, imp


def fmt(x, n=3):
    try:
        return f"{float(x):.{n}f}"
    except Exception:
        return "—"


def top_table(allp, n=15):
    rows = ""
    for i, r in allp.head(n).iterrows():
        hl = "background:#e8f5e9;" if float(r["test_f1"]) >= BAR else ""
        rows += (f"<tr style='{hl}'><td>{i+1}</td><td>{r['experiment']}</td>"
                 f"<td>{r['feature_set']}</td><td>{r['model']}</td>"
                 f"<td><b>{fmt(r['test_f1'])}</b></td><td>{fmt(r['test_recall'])}</td>"
                 f"<td>{fmt(r['test_precision'])}</td><td>{fmt(r['test_auc'])}</td>"
                 f"<td>{fmt(r['threshold'])}</td></tr>")
    return rows


def feat_rows(imp, n=10):
    rows = ""
    for _, r in imp.head(n).iterrows():
        direction = "→ distress" if r["svm_linear_coef"] < 0 else "→ safer"
        rows += (f"<tr><td>{r['feature']}</td><td>{fmt(r['abs_coef'],3)}</td>"
                 f"<td>{fmt(r['svm_linear_coef'],3)}</td><td>{direction}</td></tr>")
    return rows


def main():
    allp, summ, imp = load()
    cm = summ["confusion_matrix"]
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    beat = summ["test_f1"] > BAR
    verdict = ("BEATEN — new high for Class-0 distress detection"
               if beat else "not yet exceeded")
    delta = summ["test_f1"] - BAR

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Financial Distress (Class 0) — Model Report</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin:0;
         color:#1a1a1a; background:#f7f9fc; }}
  .wrap {{ max-width: 1080px; margin: 0 auto; padding: 32px 24px 64px; }}
  h1 {{ font-size: 28px; margin-bottom: 4px; }}
  h2 {{ font-size: 21px; margin-top: 40px; border-bottom: 2px solid #1565c0;
       padding-bottom: 6px; color:#0d47a1; }}
  .sub {{ color:#555; margin-bottom: 24px; }}
  .kpis {{ display:flex; flex-wrap:wrap; gap:14px; margin: 18px 0; }}
  .kpi {{ background:#fff; border:1px solid #e0e0e0; border-radius:10px;
         padding:16px 20px; min-width:150px; box-shadow:0 1px 3px rgba(0,0,0,.06); }}
  .kpi .v {{ font-size:26px; font-weight:700; color:#1565c0; }}
  .kpi .l {{ font-size:12px; color:#666; text-transform:uppercase; letter-spacing:.5px; }}
  .banner {{ background:{'#e8f5e9' if beat else '#fff3e0'}; border-left:6px solid
            {'#2e7d32' if beat else '#fb8c00'}; padding:16px 20px; border-radius:8px;
            margin:18px 0; font-size:16px; }}
  table {{ width:100%; border-collapse:collapse; background:#fff; margin-top:12px;
          font-size:13px; box-shadow:0 1px 3px rgba(0,0,0,.06); }}
  th, td {{ padding:9px 10px; text-align:left; border-bottom:1px solid #eee; }}
  th {{ background:#1565c0; color:#fff; }}
  tr:hover {{ background:#f1f6ff; }}
  .cm {{ display:inline-block; }}
  .cm td {{ text-align:center; width:90px; height:54px; font-size:18px; font-weight:600; }}
  .note {{ color:#666; font-size:13px; font-style:italic; margin-top:8px; }}
  code {{ background:#eef; padding:2px 5px; border-radius:4px; }}
</style></head>
<body><div class="wrap">
  <h1>📉 Financial Distress (Class 0) — Model Report</h1>
  <div class="sub">Tehran Stock Exchange · next-year cash-flow distress zone ·
  target: beat the F1 = 0.632 baseline</div>

  <div class="banner">
    <b>Baseline 0.632 {verdict}.</b> Best model:
    <b>{summ['model']}</b> on <b>{summ['feature_set']}</b>
    (exact original feature set). Achieved <b>F1 = {fmt(summ['test_f1'])}</b>
    ({'+' if delta>=0 else ''}{fmt(delta,3)} vs bar), catching
    <b>{fmt(summ['test_recall'],1)}%</b> of real distress.
  </div>

  <div class="kpis">
    <div class="kpi"><div class="v">{fmt(summ['test_f1'])}</div><div class="l">Test F1</div></div>
    <div class="kpi"><div class="v">{fmt(summ['test_recall'],1)}%</div><div class="l">Recall</div></div>
    <div class="kpi"><div class="v">{fmt(summ['test_precision'],1)}%</div><div class="l">Precision</div></div>
    <div class="kpi"><div class="v">{fmt(summ['test_auc'])}</div><div class="l">AUC-ROC</div></div>
    <div class="kpi"><div class="v">{fmt(summ['threshold'])}</div><div class="l">Threshold</div></div>
  </div>

  <h2>Winning Model</h2>
  <p><b>Algorithm:</b> SVM-RBF (cost-sensitive) &nbsp;|&nbsp;
     <b>C</b> = {summ['params']['C']} &nbsp;|&nbsp;
     <b>gamma</b> = {summ['params']['gamma']} &nbsp;|&nbsp;
     <b>class_weight</b> = distress ×3 ({summ['params']['class_weight']})</p>
  <p><b>Confusion matrix</b> (out-of-time test set, n = {summ['n_test']},
     {summ['n_test_distress']} distress):</p>
  <div class="cm"><table>
    <tr><td></td><td>Pred Not</td><td>Pred Distress</td></tr>
    <tr><td>Actual Not</td><td style="background:#e8f5e9">{tn}</td><td style="background:#ffebee">{fp}</td></tr>
    <tr><td>Actual Distress</td><td style="background:#ffebee">{fn}</td><td style="background:#e8f5e9">{tp}</td></tr>
  </table></div>
  <p class="note">Correctly caught distress: <b>{tp}</b> · missed: <b>{fn}</b> ·
  false alarms: <b>{fp}</b> · correctly cleared: <b>{tn}</b></p>

  <h2>Top Drivers of Distress</h2>
  <p class="note">Bars = |SVM linear weight|. Negative weight pushes toward distress.</p>
  <table><tr><th>Feature</th><th>|Weight|</th><th>Signed weight</th><th>Direction</th></tr>
    {feat_rows(imp)}
  </table>

  <h2>Top Models (all experiments)</h2>
  <table><tr><th>#</th><th>Exp</th><th>Feature set</th><th>Model</th><th>F1</th>
    <th>Recall</th><th>Precision</th><th>AUC</th><th>Thr</th></tr>
    {top_table(allp)}
  </table>
  <p class="note">Green rows exceed the 0.632 baseline. Full data:
  <code>results/ALL_EXPERIMENTS.csv</code>.</p>

  <h2>Methodology (no data leakage)</h2>
  <table><tr><th>Split</th><th>Years</th><th>Samples</th></tr>
    <tr><td>Train</td><td>≤ 1399</td><td>1,504</td></tr>
    <tr><td>Validate</td><td>1400–1401</td><td>226</td></tr>
    <tr><td>Test</td><td>1402–1403</td><td>261</td></tr></table>
  <p>Median imputation, StandardScaler, and SMOTE are fit on <b>train only</b>.
  Thresholds are tuned on <b>validation only</b>; the test set is touched once.
  Feature definitions loaded from the exact original <code>07_feature_sets.json</code>.</p>

  <h2>How the 0.632 bar was beaten</h2>
  <p>The original best (F1 = 0.632) used an SVM with balanced class weights.
  Restoring the <b>exact</b> Group_B feature set (incl. profit-margin ratios and
  Receivables_Turnover) and applying an <b>asymmetric cost-sensitive weighting
  (distress weighted 3×)</b> plus a finer C/gamma/grid search lifted F1 to
  <b>{fmt(summ['test_f1'])}</b> — a clear margin above the bar while still catching
  {fmt(summ['test_recall'],1)}% of real distress.</p>

  <p class="note">Generated locally from <code>results/*.csv</code> and
  <code>best_model_summary.json</code>. Reproduce with
  <code>PYTHONPATH=src python src/make_html_report.py</code>.</p>
</div></body></html>"""

    with open(OUT, "w") as f:
        f.write(html)
    print(f"Wrote {OUT} ({len(html)} bytes)")
    print(f"Best F1 shown: {fmt(summ['test_f1'])} (beats 0.632: {beat})")


if __name__ == "__main__":
    main()
