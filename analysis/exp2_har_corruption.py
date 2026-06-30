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
    MONGO_URI, DB_BASELINE, DB_CORRUPTION,
    COL_SEMANTIC, COL_CORRUPTION_LIGHT, COL_CORRUPTION_MEDIUM, COL_CORRUPTION_HEAVY,
    ADL_LABELS, USERS, C,
    FONT_TITLE, FONT_AXIS, FONT_ANNOT, FONT_TICK,
    FIG_DPI, RESULTS_DIR,
    apply_style, load_docs, compute_accuracy,
)

apply_style()

CONDITIONS = [
    ("Baseline（System A）", DB_BASELINE,   COL_SEMANTIC,          C["baseline"]),
    ("Light Corruption",    DB_CORRUPTION, COL_CORRUPTION_LIGHT,  C["corruption_light"]),
    ("Medium Corruption",   DB_CORRUPTION, COL_CORRUPTION_MEDIUM, C["corruption_medium"]),
    ("Heavy Corruption",    DB_CORRUPTION, COL_CORRUPTION_HEAVY,  C["corruption_heavy"]),
]


def per_class_accuracy(docs: list) -> dict:
    by_class = defaultdict(lambda: {"tp": 0, "total": 0})
    for d in docs:
        gt = d.get("ground_truth", "")
        if gt in ADL_LABELS:
            by_class[gt]["total"] += 1
            if gt == d.get("_pred", ""):
                by_class[gt]["tp"] += 1
    return by_class


NOISE_DETAIL = {
    "Baseline（System A）": "",
    "Light Corruption":    "pickup 15% / putdown 5% / obj 10% / skel 5°",
    "Medium Corruption":   "pickup 25% / putdown 10% / obj 15% / skel 10°",
    "Heavy Corruption":    "pickup 35% / putdown 15% / obj 20% / skel 15°",
}


def plot_accuracy_drop(results: dict, save_path: str):
    names = list(results.keys())
    accs  = [results[n]["acc"] * 100 for n in names]
    colors = [r[3] for r in CONDITIONS if r[0] in names]

    fig, ax = plt.subplots(figsize=(13, 6.5))
    bars = ax.bar(range(len(names)), accs, color=colors,
                  alpha=0.88, width=0.55, edgecolor="white")

    baseline_acc = results.get("Baseline", {}).get("acc", 0) * 100
    for i, (bar, acc) in enumerate(zip(bars, accs)):
        drop = baseline_acc - acc
        label = f"{acc:.1f}%"
        if drop > 0.1:
            label += f"\n(−{drop:.1f}%)"
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5, label,
                ha="center", va="bottom", fontsize=FONT_ANNOT,
                color=C["highlight"] if drop > 5 else "#333",
                fontweight="bold" if drop > 5 else "normal")

    tick_labels = []
    for n in names:
        detail = NOISE_DETAIL.get(n, "")
        detail_lines = detail.replace(" / ", "\n") if detail else ""
        tick_labels.append(f"{n}\n{detail_lines}".rstrip())

    ax.axhline(baseline_acc, color=C["baseline"], linestyle="--",
               lw=1.5, alpha=0.6, label=f"Baseline ({baseline_acc:.1f}%)")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(tick_labels, fontsize=FONT_TICK - 2)
    ax.set_ylabel("Accuracy (%)", fontsize=FONT_AXIS)
    ax.set_ylim(0, 110)
    ax.set_title("HAR Accuracy vs Sensor Corruption Level",
                 fontsize=FONT_TITLE, fontweight="bold", pad=10)
    ax.legend(fontsize=FONT_TICK)

    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"[exp2] Saved: {save_path}")


def save_summary(results: dict, save_path: str):
    lines = [
        "Experiment 2: HAR Corruption Robustness",
        f"DB: {DB_CORRUPTION}",
        "",
        f"{'Condition':<22} {'Acc':>6} {'Correct':>8} {'Total':>7}",
        "-" * 48,
    ]
    baseline_acc = results.get("Baseline", {}).get("acc", 0)
    for name, data in results.items():
        acc  = data["acc"]
        drop = baseline_acc - acc
        drop_str = f"(−{drop:.1%})" if drop > 0.001 else ""
        lines.append(
            f"{name:<22} {acc:>5.1%} {data['correct']:>8} "
            f"{data['total']:>7}  {drop_str}")

    lines += ["", "Worst drop by class (Baseline → Heavy):"]
    if "Baseline" in results and "Heavy Corruption" in results:
        for label in ADL_LABELS:
            b_info = results["Baseline"]["by_class"].get(label, {"tp": 0, "total": 0})
            h_info = results["Heavy Corruption"]["by_class"].get(label, {"tp": 0, "total": 0})
            if b_info["total"] == 0 or h_info["total"] == 0:
                continue
            b_acc  = b_info["tp"] / b_info["total"]
            h_acc  = h_info["tp"] / h_info["total"]
            drop   = b_acc - h_acc
            if drop > 0.1:
                lines.append(f"  {label:<16} {b_acc:.1%} → {h_acc:.1%}  (−{drop:.1%})")

    with open(save_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[exp2] Saved: {save_path}")
    print("\n".join(lines))


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    client  = MongoClient(MONGO_URI)
    results = {}

    for name, db_name, col_name, _ in CONDITIONS:
        db   = client[db_name]
        docs = load_docs(db, col_name)
        if not docs:
            print(f"[exp2] Skipping {name}: no data in {db_name}.{col_name}")
            continue
        acc, correct, total = compute_accuracy(docs)
        by_class = per_class_accuracy(docs)
        results[name] = {
            "acc": acc, "correct": correct, "total": total,
            "by_class": by_class, "docs": docs,
        }
        print(f"[exp2] {name}: {acc:.1%} ({correct}/{total})")

    if not results:
        print("[exp2] No data found. Run experiments first.")
        return

    plot_accuracy_drop(results, os.path.join(RESULTS_DIR, "exp2_accuracy_drop.png"))
    save_summary(results, os.path.join(RESULTS_DIR, "exp2_summary.txt"))

    print("\n[exp2] Done.")


if __name__ == "__main__":
    main()
