import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from pymongo import MongoClient

MONGO_URI     = "mongodb://127.0.0.1:27017/"
DB_BASELINE   = "robot_exp_baseline"
DB_CORRUPTION = "robot_exp_corruption"
OUT           = os.path.join(_ROOT, "analysis", "results")

LABELS = [
    "Drinking", "SittingDrink", "Sitting", "Eating", "Cooking",
    "Opening", "Laying", "Watching", "Reading", "Cleaning",
    "PhoneUse", "Typing",
]

COLOR_BASE = "#5C6BC0"
COLOR_CORR = "#EF5350"

def load_docs(db_name):
    db = MongoClient(MONGO_URI)[db_name]
    return list(db.eval_logs.find(
        {"ground_truth": {"$exists": True, "$ne": ""},
         "spatial_action": {"$exists": True}},
        {"ground_truth": 1, "spatial_action": 1, "vlm_output": 1}
    ))

def compute_metrics(docs):
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
    return {"total": total, "correct": correct, "accuracy": acc, "per_class": per_class}

def plot_overall_comparison(m_base, m_corr):
    fig, ax = plt.subplots(figsize=(7, 6))

    categories = ["Baseline\n(No Corruption)", "Corruption\n(Realistic Noise)"]
    accs       = [m_base["accuracy"] * 100, m_corr["accuracy"] * 100]
    colors     = [COLOR_BASE, COLOR_CORR]

    bars = ax.bar(range(2), accs, color=colors, alpha=0.88,
                  width=0.45, edgecolor="white", linewidth=1.5)

    for bar, acc, n in zip(bars, accs,
                            [m_base["total"], m_corr["total"]]):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.8,
                f"{acc:.1f}%\n(n={n})",
                ha="center", fontsize=12, fontweight="bold")

    delta = accs[1] - accs[0]
    mid_x = 0.5
    mid_y = (accs[0] + accs[1]) / 2
    ax.annotate("", xy=(1, accs[1] + 2), xytext=(0, accs[0] + 2),
                arrowprops=dict(arrowstyle="<->", color="#333", lw=1.5))
    ax.text(mid_x, max(accs) + 5,
            f"Δ = {delta:+.1f}%",
            ha="center", fontsize=12, color="#C62828", fontweight="bold")

    ax.set_xticks(range(2))
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylabel("Recognition Accuracy (%)", fontsize=12)
    ax.set_ylim(0, 110)
    ax.set_title(
        "HAR Accuracy: Baseline vs Corruption",
        fontsize=13, fontweight="bold", pad=12)
    ax.grid(axis="y", alpha=0.25)

    base_p = plt.Rectangle((0,0),1,1, fc=COLOR_BASE, alpha=0.88)
    corr_p = plt.Rectangle((0,0),1,1, fc=COLOR_CORR, alpha=0.88)
    ax.legend([base_p, corr_p],
              ["Baseline", "Corruption (EPIC-KITCHENS + YOLOv8)"],
              fontsize=9, loc="upper right")

    plt.tight_layout()
    path = os.path.join(OUT, "exp2_overall_comparison.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[exp2] Saved: {path}")

def plot_per_class_drop(m_base, m_corr):
    data = []
    for label in LABELS:
        b = m_base["per_class"].get(label)
        c = m_corr["per_class"].get(label)
        if b is None or c is None:
            continue
        data.append((label, b * 100, c * 100, (c - b) * 100))

    data.sort(key=lambda x: x[3])

    labels     = [d[0] for d in data]
    base_accs  = [d[1] for d in data]
    corr_accs  = [d[2] for d in data]
    deltas     = [d[3] for d in data]
    n          = len(labels)

    fig, ax = plt.subplots(figsize=(11, 7))
    y = np.arange(n)
    h = 0.35

    bars_b = ax.barh(y + h/2, base_accs, h,
                     color=COLOR_BASE, alpha=0.85,
                     label="Baseline", edgecolor="white")
    bars_c = ax.barh(y - h/2, corr_accs, h,
                     color=COLOR_CORR, alpha=0.85,
                     label="Corruption", edgecolor="white")

    for i, (b, c, delta) in enumerate(zip(base_accs, corr_accs, deltas)):
        ax.text(max(b, c) + 1, y[i] + h/2,
                f"{b:.0f}%", va="center", fontsize=8, color=COLOR_BASE)
        ax.text(max(b, c) + 1, y[i] - h/2,
                f"{c:.0f}%", va="center", fontsize=8, color=COLOR_CORR)
        ax.text(max(b, c) + 8, y[i],
                f"Δ{delta:+.0f}%",
                va="center", fontsize=9,
                color="#C62828" if delta < -10 else "#555",
                fontweight="bold" if delta < -10 else "normal")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Accuracy (%)", fontsize=12)
    ax.set_xlim(0, 125)
    ax.set_title(
        "Per-class Accuracy: Baseline vs Corruption\n"
        "(Sorted by accuracy drop)",
        fontsize=12, fontweight="bold", pad=10)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(axis="x", alpha=0.25)
    ax.axvline(x=80, color="gray", linestyle="--", alpha=0.4, lw=1)

    plt.tight_layout()
    path = os.path.join(OUT, "exp2_per_class_drop.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[exp2] Saved: {path}")

def save_summary(m_base, m_corr):
    delta = m_corr["accuracy"] - m_base["accuracy"]
    lines = [
        "Experiment 2: Corruption Robustness",
        f"Baseline DB   : {DB_BASELINE}  (n={m_base['total']})",
        f"Corruption DB : {DB_CORRUPTION}  (n={m_corr['total']})",
        "",
        f"Overall Accuracy:",
        f"  Baseline   : {m_base['accuracy']:.1%}",
        f"  Corruption : {m_corr['accuracy']:.1%}",
        f"  Delta      : {delta:+.1%}",
        "",
        f"Per-class Results:",
        f"{'Action':<16} {'Baseline':>10} {'Corruption':>12} {'Delta':>8} {'Impact':>8}",
        "-" * 58,
    ]
    for label in LABELS:
        b = m_base["per_class"].get(label)
        c = m_corr["per_class"].get(label)
        if b is None and c is None:
            continue
        b_str = f"{b:.1%}" if b is not None else "N/A"
        c_str = f"{c:.1%}" if c is not None else "N/A"
        d_val = (c - b) if (b is not None and c is not None) else None
        d_str = f"{d_val:+.1%}" if d_val is not None else ""
        impact = ("HIGH" if d_val is not None and d_val < -0.15 else
                  "MED"  if d_val is not None and d_val < -0.05 else "LOW")
        lines.append(
            f"{label:<16} {b_str:>10} {c_str:>12} {d_str:>8} {impact:>8}")

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
        print(f"[exp2] No data in {DB_CORRUPTION} — run corruption experiment first")
        return
    print(f"[exp2] Baseline: {len(docs_base)} | Corruption: {len(docs_corr)}")
    m_base = compute_metrics(docs_base)
    m_corr = compute_metrics(docs_corr)
    plot_overall_comparison(m_base, m_corr)
    plot_per_class_drop(m_base, m_corr)
    save_summary(m_base, m_corr)

if __name__ == "__main__":
    main()