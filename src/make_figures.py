import json
"""
Generate presentation figures for the dashboard from the experiment results.
Produces:
  reports/fig_f1_by_model.png      - top models bar chart
  reports/fig_confusion.png        - best-model confusion matrix heatmap
  reports/fig_experiment_trend.png - F1 across experiments (shows progress to bar)
Run: PYTHONPATH=src python src/make_figures.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from common import RESULTS_DIR, ROOT

BAR = 0.632
os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)


def load_all():
    f = os.path.join(RESULTS_DIR, "ALL_EXPERIMENTS.csv")
    if os.path.exists(f):
        return pd.read_csv(f)
    return pd.DataFrame()


def fig_f1_by_model(df):
    d = df.dropna(subset=["test_f1"]).sort_values("test_f1", ascending=False).head(12)
    labels = [f"{r.model}\n({r.feature_set.replace('Group_','G')})" for r in d.itertuples()]
    vals = d["test_f1"].values
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#2e7d32" if v >= BAR else "#1565c0" for v in vals]
    bars = ax.barh(range(len(vals)), vals, color=colors)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(BAR, color="#c62828", ls="--", lw=1.5, label=f"Baseline bar = {BAR}")
    ax.set_xlabel("Test F1 (Class 0 distress)")
    ax.set_title("Top Models — Test F1 vs 0.632 Baseline")
    ax.legend(loc="lower right")
    for i, v in enumerate(vals):
        ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    p = os.path.join(ROOT, "reports", "fig_f1_by_model.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    return p


def fig_confusion(df):
    best = df.dropna(subset=["test_f1"]).sort_values("test_f1", ascending=False).iloc[0]
    cm = best.get("confusion_matrix")
    if isinstance(cm, str):
        try: cm = eval(cm)
        except Exception: cm = None
    if cm is None or (isinstance(cm, float) and np.isnan(cm)):
        cm = None
    # Fallback: load the persisted winning-model summary (has the real CM)
    if cm is None:
        sj = os.path.join(RESULTS_DIR, "best_model_summary.json")
        if os.path.exists(sj):
            try:
                cm = json.load(open(sj)).get("confusion_matrix")
            except Exception:
                cm = None
    if cm is None:
        return None
    cm = np.array(cm, dtype=int)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred Not", "Pred Distress"])
    ax.set_yticklabels(["Actual Not", "Actual Distress"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=14)
    ax.set_title(f"Best Model: {best['model']}\nF1={best['test_f1']:.3f}", fontsize=10)
    fig.tight_layout()
    p = os.path.join(ROOT, "reports", "fig_confusion.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    return p


def fig_trend(df):
    exp_order = ["E8", "E9", "E10", "E11", "E12"]
    bestf = {}
    for e in exp_order:
        sub = df[df["experiment"] == e] if "experiment" in df.columns else df[df["experiment"].astype(str).str.startswith(e)]
        sub = df[df["experiment"].astype(str).str.contains(e, na=False)]
        if len(sub):
            bestf[e] = sub["test_f1"].max()
    if not bestf:
        return None
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = list(bestf.keys()); ys = list(bestf.values())
    ax.plot(xs, ys, "-o", color="#1565c0", lw=2)
    ax.axhline(BAR, color="#c62828", ls="--", label=f"Baseline {BAR}")
    for x, y in zip(xs, ys):
        ax.text(x, y + 0.005, f"{y:.3f}", ha="center", fontsize=9)
    ax.set_ylim(0.45, 0.70)
    ax.set_ylabel("Best Test F1")
    ax.set_title("Progress Across Experiments")
    ax.legend()
    fig.tight_layout()
    p = os.path.join(ROOT, "reports", "fig_experiment_trend.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    return p


if __name__ == "__main__":
    df = load_all()
    if df.empty:
        print("No results found"); raise SystemExit(1)
    print("f1_by_model:", fig_f1_by_model(df))
    print("confusion:", fig_confusion(df))
    print("trend:", fig_trend(df))
