import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from pymongo import MongoClient

from exp_config import (
    MONGO_URI, DB_BASELINE, DB_CORRUPTION, ADL_LABELS,
    C, FONT_TITLE, FONT_AXIS, FONT_ANNOT, FONT_TICK,
    LINE_WIDTH, FIG_DPI, RESULTS_DIR, apply_style
)

apply_style()

MAX_DAY = 21   # truncate both DBs to first 21 virtual days


def load_docs(db_name):
    db = MongoClient(MONGO_URI)[db_name]
    return list(db.eval_logs.find(
        {"ground_truth": {"$exists": True, "$ne": ""},
         "spatial_action": {"$exists": True},
         "virtual_day": {"$lte": MAX_DAY}},
        {"ground_truth": 1, "spatial_action": 1, "vlm_output": 1}
    ))


def compute_metrics(docs):
    total   = len(docs)
    correct = sum(1 for d in docs
                  if (d.get("spatial_action") or d.get("vlm_output", "")) ==
                  d.get("ground_truth", ""))
    acc = correct / total if total else 0

    by_class = defaultdict(lambda: {"tp": 0, "total": 0})
    for d in docs:
        gt   = d.get("ground_truth", "")
        pred = d.get("spatial_action") or d.get("vlm_output", "")
        if gt in ADL_LABELS:
            by_class[gt]["total"] += 1
            if gt == pred:
                by_class[gt]["tp"] += 1

    per_class = {
        label: (by_class[label]["tp"] / by_class[label]["total"]
                if by_class[label]["total"] > 0 else None)
        for label in ADL_LABELS
    }
    return {"total": total, "correct": correct, "accuracy": acc,
            "per_class": per_class}


def plot_overall(m_base, m_corr, save_path):
    labels = ["Baseline\n(Clean Sensors)", "Corruption\n(Noisy Sensors)"]
    accs   = [m_base["accuracy"] * 100, m_corr["accuracy"] * 100]
    colors = [C["baseline"], C["corruption"]]
    delta  = accs[1] - accs[0]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(range(2), accs, color=colors, width=0.45,
                  alpha=0.88, edgecolor="white")

    for bar, acc, n in zip(bars, accs, [m_base["total"], m_corr["total"]]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                f"{acc:.1f}%\n(n={n})",
                ha="center", fontsize=FONT_TICK + 1, fontweight="bold")

    ax.annotate("", xy=(1, max(accs) + 5), xytext=(0, max(accs) + 5),
                arrowprops=dict(arrowstyle="<->", color="#555", lw=LINE_WIDTH))
    ax.text(0.5, max(accs) + 7, f"Δ = {delta:+.1f}%",
            ha="center", fontsize=FONT_TICK + 1,
            color=C["corruption"] if delta < 0 else C["pass"],
            fontweight="bold")

    ax.set_xticks(range(2))
    ax.set_xticklabels(labels, fontsize=FONT_TICK)
    ax.set_ylabel("Recognition Accuracy (%)", fontsize=FONT_AXIS)
    ax.set_ylim(0, 115)
    ax.set_title(
        f"HAR Accuracy: Baseline vs Sensor Noise (Day 1–{MAX_DAY})",
        fontsize=FONT_TITLE, fontweight="bold", pad=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"[exp2] Saved: {save_path}")


def plot_per_class_drop(m_base, m_corr, save_path):
    data = []
    for label in ADL_LABELS:
        b = m_base["per_class"].get(label)
        c = m_corr["per_class"].get(label)
        if b is None or c is None:
            continue
        data.append((label, b * 100, c * 100, (c - b) * 100))

    if not data:
        print("[exp2] No per-class data to plot"); return

    data.sort(key=lambda x: x[3])
    labels    = [d[0] for d in data]
    base_accs = [d[1] for d in data]
    corr_accs = [d[2] for d in data]
    deltas    = [d[3] for d in data]
    n = len(labels)
    y = np.arange(n)
    h = 0.35

    fig, ax = plt.subplots(figsize=(11, max(5, n * 0.7)))
    ax.barh(y + h / 2, base_accs, h, color=C["baseline"],
            alpha=0.85, label="Baseline", edgecolor="white")
    ax.barh(y - h / 2, corr_accs, h, color=C["corruption"],
            alpha=0.85, label="Corruption", edgecolor="white")

    for i, (b, c, delta) in enumerate(zip(base_accs, corr_accs, deltas)):
        right = max(b, c) + 1
        ax.text(right, y[i] + h / 2, f"{b:.0f}%",
                va="center", fontsize=FONT_ANNOT, color=C["baseline"])
        ax.text(right, y[i] - h / 2, f"{c:.0f}%",
                va="center", fontsize=FONT_ANNOT, color=C["corruption"])
        if abs(delta) > 5:
            ax.text(right + 9, y[i], f"Δ{delta:+.0f}%",
                    va="center", fontsize=FONT_ANNOT,
                    color=C["highlight"], fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=FONT_TICK)
    ax.set_xlabel("Accuracy (%)", fontsize=FONT_AXIS)
    ax.set_xlim(0, 130)
    ax.set_title("Per-class Accuracy: Baseline vs Sensor Noise",
                 fontsize=FONT_TITLE, fontweight="bold", pad=10)
    ax.legend(fontsize=FONT_TICK, loc="lower right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"[exp2] Saved: {save_path}")


def save_summary(m_base, m_corr, save_path):
    delta = m_corr["accuracy"] - m_base["accuracy"]
    lines = [
        "Experiment 2: Robustness Under Sensor Noise",
        f"Both DBs truncated to first {MAX_DAY} virtual days",
        f"Baseline   : {DB_BASELINE}   (n={m_base['total']})",
        f"Corruption : {DB_CORRUPTION} (n={m_corr['total']})",
        "",
        f"Overall Accuracy:",
        f"  Baseline   : {m_base['accuracy']:.1%}",
        f"  Corruption : {m_corr['accuracy']:.1%}",
        f"  Delta      : {delta:+.1%}",
        "",
        f"{'Action':<16} {'Baseline':>10} {'Corruption':>12} {'Delta':>8}",
        "-" * 50,
    ]
    for label in ADL_LABELS:
        b = m_base["per_class"].get(label)
        c = m_corr["per_class"].get(label)
        b_s = f"{b:.1%}" if b is not None else "N/A"
        c_s = f"{c:.1%}" if c is not None else "N/A"
        d_s = f"{(c-b):+.1%}" if (b is not None and c is not None) else ""
        lines.append(f"{label:<16} {b_s:>10} {c_s:>12} {d_s:>8}")

    with open(save_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[exp2] Saved: {save_path}")
    print("\n".join(lines))


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    docs_base = load_docs(DB_BASELINE)
    docs_corr = load_docs(DB_CORRUPTION)

    if not docs_base:
        print(f"[exp2] No data in {DB_BASELINE}"); return
    if not docs_corr:
        print(f"[exp2] No data in {DB_CORRUPTION}"); return

    print(f"[exp2] Baseline (day 1-{MAX_DAY}): {len(docs_base)} | "
          f"Corruption (day 1-{MAX_DAY}): {len(docs_corr)}")

    m_base = compute_metrics(docs_base)
    m_corr = compute_metrics(docs_corr)

    plot_overall(m_base, m_corr,
                 os.path.join(RESULTS_DIR, "exp2_overall_comparison.png"))
    plot_per_class_drop(m_base, m_corr,
                        os.path.join(RESULTS_DIR, "exp2_per_class_drop.png"))
    save_summary(m_base, m_corr,
                 os.path.join(RESULTS_DIR, "exp2_summary.txt"))


if __name__ == "__main__":
    main()