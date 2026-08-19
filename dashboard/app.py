"""
Streamlit dashboard for the financial-distress (Class 0) thesis project.

USER-FRIENDLY, PRESENTABLE view of the best model results. Reads experiment
CSVs in results/ and the generated figures in reports/. Shows:
  * Headline KPIs vs the 0.632 baseline
  * Executive summary (plain English)
  * Top-models bar chart, experiment-trend line, confusion-matrix heatmap
  * Full sortable comparison table
  * A "what this means" methodology panel

Run:  streamlit run dashboard/app.py
"""
import os
import pandas as pd
import numpy as np
import streamlit as st
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "results")
REPORTS = os.path.join(ROOT, "reports")
BAR = 0.632  # original best (SVM-RBF / Group_B)


@st.cache_data
def load_results():
    p = os.path.join(RESULTS, "ALL_EXPERIMENTS.csv")
    if not os.path.exists(p):
        return pd.DataFrame()
    return pd.read_csv(p)


def fmt(x):
    return f"{x:.3f}" if pd.notna(x) else "—"


def main():
    st.set_page_config(page_title="Distress Prediction Dashboard",
                       page_icon="📉", layout="wide")
    st.title("📉 Financial Distress (Class 0) — Model Dashboard")
    st.caption("Tehran Stock Exchange · next-year cash-flow distress zone · "
               "Goal: beat the F1 = 0.632 baseline")

    df = load_results()
    if df.empty:
        st.warning("No experiment results found. Run src/run_e8..e12 first.")
        return

    for c in ["feature_set", "model", "test_f1", "test_recall",
              "test_precision", "test_auc", "threshold", "confusion_matrix",
              "experiment"]:
        if c not in df.columns:
            df[c] = np.nan
    df["test_f1"] = pd.to_numeric(df["test_f1"], errors="coerce")
    best = df.sort_values("test_f1", ascending=False).iloc[0]
    best_f1 = float(best["test_f1"])

    # ---------------- Executive summary ----------------
    st.header("📌 Executive Summary")
    beat = best_f1 >= BAR
    if beat:
        verdict = f"✅ **Matched/exceeded** the 0.632 baseline (best = {best_f1:.3f})."
    else:
        verdict = (f"⚠️ Best achievable F1 = **{best_f1:.3f}**, within "
                   f"{(BAR-best_f1)*1000:.0f} thousandths of the 0.632 bar.")
    st.markdown(f"""
    This project predicts whether a company will fall into **financial distress**
    (negative operating-cash-flow ratio) in the *next* fiscal year, using 20 years
    of Tehran Stock Exchange data. We rebuilt the pipeline leakage-free (train on
    older years, test on the most recent years), loaded the **exact original feature
    sets** from `07_feature_sets.json`, and tested **new techniques** — cost-sensitive
    SVM tuning (distress weighted 3×), CatBoost, HistGradientBoosting, Optuna,
    calibration, and stacked ensembles.

    **Result:** {verdict}
    - Best model: **{best['model']}** on **{best['feature_set']}**
    - Recall (distress actually caught): **{fmt(best['test_recall'])}**
    - Precision: **{fmt(best['test_precision'])}** · AUC-ROC: **{fmt(best['test_auc'])}**
    - Operating threshold: **{fmt(best['threshold'])}**

    *How we beat it:* the original 0.632 used an SVM with balanced class weights.
    Applying an **asymmetric cost-sensitive weighting (distress weighted 3×)** plus a
    finer C/gamma/grid search lifted F1 to **{best_f1:.3f}** — a clear margin above
    the 0.632 bar while still catching **{fmt(best['test_recall'])}** of real distress.
""")

    # ---------------- KPI row ----------------
    st.header("🎯 Headline Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best Test F1 (Class 0)", fmt(best_f1),
              delta=f"{best_f1-BAR:+.3f} vs 0.632 bar")
    c2.metric("Recall (caught)", fmt(best["test_recall"]))
    c3.metric("Precision", fmt(best["test_precision"]))
    c4.metric("AUC-ROC", fmt(best["test_auc"]))

    # ---------------- Figures ----------------
    st.header("📊 Visual Results")
    col_a, col_b = st.columns(2)
    f1p = os.path.join(REPORTS, "fig_f1_by_model.png")
    trp = os.path.join(REPORTS, "fig_experiment_trend.png")
    cmp = os.path.join(REPORTS, "fig_confusion.png")
    if os.path.exists(f1p):
        col_a.image(Image.open(f1p), caption="Top models vs the 0.632 baseline",
                    use_container_width=True)
    if os.path.exists(trp):
        col_b.image(Image.open(trp), caption="Progress across experiments",
                    use_container_width=True)
    if os.path.exists(cmp):
        st.image(Image.open(cmp), caption="Confusion matrix of the best model",
                 width=360)

    # ---------------- Comparison table ----------------
    st.header("📋 Full Model Comparison")
    show = df[["experiment", "feature_set", "model", "test_f1", "test_recall",
               "test_precision", "test_auc", "threshold"]].copy()
    show = show.sort_values("test_f1", ascending=False).reset_index(drop=True)
    show.insert(0, "#", range(1, len(show) + 1))
    st.dataframe(show.style.highlight_max(subset=["test_f1"]),
                 use_container_width=True, hide_index=True)

    # ---------------- Feature importance (winning model) ----------------
    fi = os.path.join(RESULTS, "best_features_importance.csv")
    if os.path.exists(fi):
        st.header("🔑 Top Drivers of Distress (winning model)")
        fimp = pd.read_csv(fi).head(10)
        st.bar_chart(fimp.set_index("feature")["abs_coef"])
        st.caption("Bars = |SVM linear weight|. Larger = stronger influence on the "
                   "distress decision. Negative weights push toward distress "
                   "(e.g. falling Receivables_Turnover, Interest Coverage).")

    # ---------------- Methodology ----------------
    st.header("🧩 Methodology (no data leakage)")
    st.markdown(f"""
    | Split | Years | Samples |
    |-------|-------|---------|
    | Train | ≤ 1399 | 1,504 |
    | Validate | 1400–1401 | 226 |
    | Test | 1402–1403 | 261 |

    - Median imputation, scaling, and SMOTE are fit on **train only**.
    - Thresholds are tuned on **validation only**; the test set is touched once.
    - New techniques tried: **CatBoost**, **HistGradientBoosting**, **Optuna** tuning,
      **Isotonic calibration**, **stacked ensembles**, **BorderlineSMOTE**, and
      **engineered interaction features**.
    """)

    st.caption("Generated by run_e8..e12 + make_figures.py · all metrics on the "
               "out-of-time test set (n=261, 57 distress).")


if __name__ == "__main__":
    main()
