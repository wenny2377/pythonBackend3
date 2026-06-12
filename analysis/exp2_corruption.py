"""
exp2_corruption.py
──────────────────
Experiment 2: Corruption Robustness

Compares two DBs:
  robot_exp_baseline   (Run 1: no corruption)
  robot_exp_corruption (Run 2 or 3: with corruption)

Run: python3 analysis/exp2_corruption.py

Outputs:
  analysis/results/exp2_accuracy_comparison.png
  analysis/results/exp2_per_class_comparison.png
  analysis/results/exp2_summary.txt
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from pymongo import MongoClient

MONGO_URI  = "mongodb://127.0.0.1:27017/"
DB_BASELINE   = "robot_exp_baseline"
DB_CORRUPTION = "robot_exp_corruption"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

LABELS = [
    "Drinking", "SittingDrink", "Sitting", "Eating", "Cooking",
    "Opening", "Laying", "Watching", "Reading", "Cleaning",
    "PhoneUse", "Typing",
]


def load_docs(db_name: str) -> list:
    db = MongoClient(MONGO_URI)[db_name]
    return list(db.eval_logs.find(
        {"ground_truth": {"$exists": True, "$ne": ""},
         "spatial_action": {"$exists": True}},
        {"ground_truth": 1, "spatial_action": 1, "vlm_output": 1}
    ))


def compute_metrics(docs: list) -> dict:
    total   = len(docs)
    correct = sum(1 for d in docs
                  if (d.get("spatial_action") or d.get("vlm_output","")) ==
                  d.get("ground_truth",""))
    acc = correct / total if total > 0 else 0

    by_class = defaultdict(lambda: {"tp": 0, "total": 0})
    for d in docs:
        gt   = d.get("ground_truth","")
        pred = d.get("spatial_action") or d.get("vlm_output","")
        if gt in LABELS:
            by_class[gt]["total"] += 1
            if gt == pred:
                by_class[gt]["tp"] += 1

    per_class = {}
    for label in LABELS:
        info = by_class.get(label, {"tp":0,"total":0})
        per_class[label] = info["tp"] / info["total"] if info["total"] > 0 else None

    return {
        "total":     total,
        "correct":   correct,
        "accuracy":  acc,
        "per_class": per_class,
    }


def plot_accuracy_comparison(m_base: dict, m_corr: dict):
    fig, ax = plt.subplots(figsize=(8, 5))

    labels = ["Baseline\n(no corruption)", "Corruption\n(pickup + putdown + confusion)"]
    accs   = [m_base["accuracy"] * 100, m_corr["accuracy"] * 100]
    colors = ["#5C6BC0", "#EF5350"]

    bars = ax.bar(range(2), accs, color=colors, alpha=0.85,
                  width=0.45, edgecolor="white")
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{acc:.1f}%", ha="center", fontsize=13, fontweight="bold")

    ax.set_xticks(range(2))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_ylim(0, 105)
    delta = accs[1] - accs[0]
    ax.set_title(
        f"Experiment 2 — Corruption robustness\n"
        f"Δ = {delta:+.1f}%  "
        f"(baseline n={m_base['total']}, corruption n={m_corr['total']})",
        fontsize=11, fontweight="bold")
    ax.axhline(y=accs[0], color="#5C6BC0", linestyle="--",
               alpha=0.4, linewidth=1)
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    path = os.path.join(OUT, "exp2_accuracy_comparison.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[exp2] Saved: {path}")


def plot_per_class_comparison(m_base: dict, m_corr: dict):
    labels_present = [l for l in LABELS
                      if m_base["per_class"].get(l) is not None
                      or m_corr["per_class"].get(l) is not None]

    base_accs = [m_base["per_class"].get(l, 0) or 0 for l in labels_present]
    corr_accs = [m_corr["per_class"].get(l, 0) or 0 for l in labels_present]

    x     = np.arange(len(labels_present))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(12, len(labels_present)), 5))
    ax.bar(x - width/2, [a*100 for a in base_accs], width,
           label="Baseline", color="#5C6BC0", alpha=0.85)
    ax.bar(x + width/2, [a*100 for a in corr_accs], width,
           label="Corruption", color="#EF5350", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels_present, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 110)
    ax.legend(fontsize=10)
    ax.set_title("Experiment 2 — Per-class accuracy: baseline vs corruption",
                 fontweight="bold")
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    path = os.path.join(OUT, "exp2_per_class_comparison.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[exp2] Saved: {path}")


def save_summary(m_base: dict, m_corr: dict):
    delta = m_corr["accuracy"] - m_base["accuracy"]
    lines = [
        "Experiment 2: Corruption Robustness",
        f"",
        f"{'':20} {'Baseline':>12} {'Corruption':>12} {'Delta':>8}",
        f"{'Overall':20} {m_base['accuracy']:>11.1%} "
        f"{m_corr['accuracy']:>11.1%} "
        f"{delta:>+7.1%}",
        f"",
        f"Per-class:",
        f"{'Action':<16} {'Baseline':>10} {'Corruption':>12} {'Delta':>8}",
        "-" * 50,
    ]
    for label in LABELS:
        b = m_base["per_class"].get(label)
        c = m_corr["per_class"].get(label)
        if b is None and c is None:
            continue
        b_str = f"{b:.1%}" if b is not None else "N/A"
        c_str = f"{c:.1%}" if c is not None else "N/A"
        d_str = f"{c-b:+.1%}" if (b is not None and c is not None) else ""
        flag  = ("XX" if (c is not None and c < 0.40) else
                 "WN" if (c is not None and c < 0.70) else "OK")
        lines.append(f"{flag} {label:<14} {b_str:>10} {c_str:>12} {d_str:>8}")

    path = os.path.join(OUT, "exp2_summary.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"[exp2] Saved: {path}")
    print("\n".join(lines))


def main():
    os.makedirs(OUT, exist_ok=True)

    docs_base = load_docs(DB_BASELINE)
    docs_corr = load_docs(DB_CORRUPTION)

    if not docs_base:
        print(f"[exp2] No data in {DB_BASELINE}")
        return
    if not docs_corr:
        print(f"[exp2] No data in {DB_CORRUPTION}")
        return

    print(f"[exp2] Baseline: {len(docs_base)} episodes")
    print(f"[exp2] Corruption: {len(docs_corr)} episodes")

    m_base = compute_metrics(docs_base)
    m_corr = compute_metrics(docs_corr)

    plot_accuracy_comparison(m_base, m_corr)
    plot_per_class_comparison(m_base, m_corr)
    save_summary(m_base, m_corr)


if __name__ == "__main__":
    main()