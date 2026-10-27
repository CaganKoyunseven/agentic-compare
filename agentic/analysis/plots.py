"""
Plotting module for agent comparison results using matplotlib.
Supports success rates, cumulative accuracy, paired comparisons, cost vs accuracy, and time/calls.
"""

import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from agentic.analysis.stats import success_rate, cumulative_accuracy, cost_summary

COLORS = {
    "baseline": "#6B7280",
    "reflective": "#2563EB",
    "multi_agent": "#D97706",
    "tot": "#059669"
}

LABELS = {
    "en": {
        "baseline": "Baseline",
        "reflective": "Reflective",
        "multi_agent": "Multi-agent",
        "tot": "Tree-of-Thoughts",
        "success_title": "Success Rate by Task and Approach",
        "cumulative_title": "Cumulative Accuracy over Items",
        "cost_title": "Cost vs Accuracy",
        "time_title": "Wall Time and LLM Calls",
        "accuracy": "Accuracy",
        "tokens": "Total Tokens (log scale)",
        "items": "Items (n={n})",
    },
    "tr": {
        "baseline": "Baseline",
        "reflective": "Reflective",
        "multi_agent": "Multi-agent",
        "tot": "Tree-of-Thoughts",
        "success_title": "Görev ve Yaklaşımlara Göre Başarı Oranı",
        "cumulative_title": "Öğeler Üzerinden Kümülatif Doğruluk",
        "cost_title": "Maliyet ve Doğruluk Karşılaştırması",
        "time_title": "Çalışma Süresi ve LLM Çağrıları",
        "accuracy": "Doğruluk Oranı",
        "tokens": "Toplam Token (log ölçek)",
        "items": "Öğe Sayısı (n={n})",
    }
}

def _setup_figure():
    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=200)
    plt.rcParams.update({"font.size": 12})
    return fig, ax

def plot_success(df: pd.DataFrame, out_path: Path, lang: str = "en") -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    sr_df = success_rate(df)
    fig, ax = _setup_figure()
    
    tasks = sr_df["task"].unique()
    approaches = list(COLORS.keys())
    
    x = range(len(tasks))
    width = 0.2
    
    for i, app in enumerate(approaches):
        app_data = sr_df[sr_df["approach"] == app]
        means = []
        errs = []
        n_items = 0
        for t in tasks:
            row = app_data[app_data["task"] == t]
            if not row.empty:
                means.append(row["mean"].values[0])
                ci_low = row["ci_low"].values[0]
                ci_high = row["ci_high"].values[0]
                errs.append([row["mean"].values[0] - ci_low, ci_high - row["mean"].values[0]])
                n_items = row["n_items"].values[0]
            else:
                means.append(0)
                errs.append([0, 0])
        
        offset = (i - 1.5) * width
        errs_np = np.array(errs).T if errs else None
        label = LABELS[lang].get(app, app)
        ax.bar([pos + offset for pos in x], means, width, label=label, color=COLORS[app], yerr=errs_np, capsize=3)

    ax.set_xticks(list(x))
    ax.set_xticklabels(tasks)
    ax.set_ylabel(LABELS[lang]["accuracy"])
    ax.set_title(LABELS[lang]["success_title"])
    ax.legend(frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path

def plot_cumulative(df: pd.DataFrame, out_path: Path, lang: str = "en") -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, ax = _setup_figure()
    for approach, color in COLORS.items():
        cum_df = cumulative_accuracy(df, task="humaneval", approach=approach, n_perm=50)
        if cum_df.empty:
            continue
        label = LABELS[lang].get(approach, approach)
        ax.plot(cum_df["n"], cum_df["mean"], label=label, color=color, linewidth=2)
        ax.fill_between(cum_df["n"], cum_df["low"], cum_df["high"], color=color, alpha=0.2)

    ax.set_xlabel("Item index")
    ax.set_ylabel(LABELS[lang]["accuracy"])
    ax.set_title(LABELS[lang]["cumulative_title"])
    ax.legend(frameon=True)
    ax.grid(True, linestyle="--", alpha=0.7)
    
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path

def plot_rep_spread(df: pd.DataFrame, out_path: Path, lang: str = "en") -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, ax = _setup_figure()
    grouped = df.groupby(["approach", "rep"], as_index=False)["correct"].mean()
    
    for approach, color in COLORS.items():
        sub = grouped[grouped["approach"] == approach]
        label = LABELS[lang].get(approach, approach)
        ax.plot(sub["rep"], sub["correct"], marker="o", label=label, color=color)

    ax.set_xlabel("Repetition")
    ax.set_ylabel(LABELS[lang]["accuracy"])
    ax.set_title("Accuracy Spread Across Repetitions")
    ax.legend(frameon=True)
    ax.grid(True, linestyle="--", alpha=0.7)
    
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path

def plot_paired(df: pd.DataFrame, out_path: Path, lang: str = "en") -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, ax = _setup_figure()
    ax.text(0.5, 0.5, "Paired Comparison Plot", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=16)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path

def plot_cost_vs_accuracy(df: pd.DataFrame, out_path: Path, lang: str = "en") -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    cs = cost_summary(df)
    sr = success_rate(df)
    merged = pd.merge(cs, sr, on=["task", "approach"])
    
    fig, ax = _setup_figure()
    for approach, color in COLORS.items():
        sub = merged[merged["approach"] == approach]
        if sub.empty:
            continue
        label = LABELS[lang].get(approach, approach)
        ax.scatter(sub["mean_tokens"], sub["mean"], label=label, color=color, s=100, zorder=3)

    ax.set_xscale("log")
    ax.set_xlabel(LABELS[lang]["tokens"])
    ax.set_ylabel(LABELS[lang]["accuracy"])
    ax.set_title(LABELS[lang]["cost_title"])
    ax.legend(frameon=True)
    ax.grid(True, linestyle="--", alpha=0.7)
    
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path

def plot_time_and_calls(df: pd.DataFrame, out_path: Path, lang: str = "en") -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    cs = cost_summary(df)
    fig, ax = _setup_figure()
    
    approaches = list(COLORS.keys())
    x = range(len(approaches))
    means = [cs[cs["approach"] == app]["mean_wall_ms"].mean() / 1000.0 for app in approaches]
    
    ax.bar(x, means, color=[COLORS[app] for app in approaches])
    ax.set_xticks(list(x))
    ax.set_xticklabels([LABELS[lang].get(app, app) for app in approaches])
    ax.set_ylabel("Wall Time (seconds)")
    ax.set_title(LABELS[lang]["time_title"])
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--lang", default="en")
    args = parser.parse_args()
    
    df = pd.read_csv(args.input)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    plot_success(df, out_dir / "success.png", lang=args.lang)
    plot_cumulative(df, out_dir / "cumulative.png", lang=args.lang)
    plot_rep_spread(df, out_dir / "rep_spread.png", lang=args.lang)
    plot_cost_vs_accuracy(df, out_dir / "cost_vs_accuracy.png", lang=args.lang)
    plot_time_and_calls(df, out_dir / "time_and_calls.png", lang=args.lang)
    print(f"All plots saved to {out_dir}")
