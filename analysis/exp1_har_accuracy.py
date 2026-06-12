"""
exp1_har_accuracy.py
────────────────────
Experiment 1: HAR Accuracy

Run: DB_NAME=robot_exp_baseline python3 analysis/exp1_har_accuracy.py

Outputs:
  analysis/results/exp1_confusion_matrix.png
  analysis/results/exp1_layer_breakdown.png
  analysis/results/exp1_summary.txt
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from pymongo import MongoClient

MONGO_URI = "mongodb://127.0.0.1:27017/"
DB_NAME   = os.environ.get("DB_NAME", "robot_rag_db")
OUT       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

LABELS = [
    "Drinking", "SittingDrink", "Sitting", "Eating", "Cooking",
    "Opening", "Laying", "Watching", "Reading", "Cleaning",
    "PhoneUse", "Typing",
]


def connect():
    return MongoClient(MONGO_URI)[DB_NAME]


def dominant_layer(reason: str) -> str:
    r = (reason or "").lower()
    if any(k in r for k in ("skeleton_lying", "head(", "strong:skeleton")):
        return "skeleton"
    if any(k in r for k in ("strong:held", "strong:head")):
        return "skeleton"
    if "pmi_llm" in r or "llm:" in r:
        return "llm"
    if "held:" in r:
        return "held"
    if "affordance:" in r:
        return "affordance"
    if "vlm(" in r:
        return "vlm"
    return "other"


def load_docs(db):
    return list(db.eval_logs.find(
        {"ground_truth": {"$exists": True, "$ne": ""},
         "spatial_action": {"$exists": True}},
        {"ground_truth": 1, "spatial_action": 1,
         "vlm_output": 1, "upgrade_reason": 1}
    ))


def plot_confusion(docs, labels):
    present = [l for l in labels
               if any(d.get("ground_truth") == l for d in docs)]
    n = len(present)
    if n == 0:
        print("[exp1] No labeled data")
        return None

    matrix = np.zeros((n, n), dtype=int)
    for d in docs:
        gt   = d.get("ground_truth", "")
        pred = d.get("spatial_action") or d.get("vlm_output", "")
        if gt in present and pred in present:
            matrix[present.index(gt)][present.index(pred)] += 1

    total   = int(matrix.sum())
    correct = int(np.trace(matrix))
    acc     = correct / total if total > 0 else 0

    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    norm = matrix / row_sums

    fig, ax = plt.subplots(figsize=(max(9, n), max(7, n)))
    im = ax.imshow(norm.T, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Recall rate")

    for i in range(n):
        for j in range(n):
            v = norm[i][j]
            if matrix[i][j] > 0:
                ax.text(i, j, f"{v:.2f}\n({matrix[i][j]})",
                        ha="center", va="center", fontsize=7,
                        color="white" if v > 0.55 else "black",
                        fontweight="bold" if i == j else "normal")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(present, rotation=40, ha="right", fontsize=9)
    ax.set_yticklabels(present, fontsize=9)
    ax.set_xlabel("Ground truth", fontsize=11)
    ax.set_ylabel("Predicted", fontsize=11)
    ax.set_title(
        f"Experiment 1 — HAR confusion matrix\n"
        f"Accuracy = {acc:.1%}  ({correct}/{total})  |  {DB_NAME}",
        fontsize=11, fontweight="bold", pad=10)

    plt.tight_layout()
    path = os.path.join(OUT, "exp1_confusion_matrix.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[exp1] Saved: {path}")
    return acc, correct, total


def plot_layer_breakdown(docs):
    total = len(docs)
    if total == 0:
        return

    correct_vlm  = sum(1 for d in docs
                       if d.get("vlm_output","") == d.get("ground_truth",""))
    correct_full = sum(1 for d in docs
                       if (d.get("spatial_action") or d.get("vlm_output","")) ==
                       d.get("ground_truth",""))
    layer_counts = Counter(
        dominant_layer(d.get("upgrade_reason","")) for d in docs)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    configs = [
        ("VLM only",    correct_vlm  / total, "#B0BEC5"),
        ("Full system", correct_full / total, "#5C6BC0"),
    ]
    bars = ax.bar(range(2), [a * 100 for _, a, _ in configs],
                  color=[c for _, _, c in configs],
                  alpha=0.85, width=0.5, edgecolor="white")
    for bar, (_, acc, _) in zip(bars, configs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f"{acc:.1%}", ha="center", fontsize=11, fontweight="bold")
    ax.set_xticks(range(2))
    ax.set_xticklabels(["VLM only", "Full system"], fontsize=11)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    delta = configs[1][1] - configs[0][1]
    ax.set_title(f"VLM only vs full system\n+{delta:.1%} from spatial reasoning",
                 fontweight="bold")
    ax.grid(axis="y", alpha=0.25)

    ax2 = axes[1]
    items  = layer_counts.most_common()
    llabels = [l for l, _ in items]
    lvals   = [c for _, c in items]
    colors  = ["#5C6BC0","#26A69A","#EF5350","#FFA726","#AB47BC","#78909C"]
    ax2.bar(range(len(llabels)), lvals,
            color=colors[:len(llabels)], alpha=0.85,
            width=0.6, edgecolor="white")
    for i, v in enumerate(lvals):
        ax2.text(i, v + 0.5, f"{v/total:.0%}", ha="center", fontsize=10)
    ax2.set_xticks(range(len(llabels)))
    ax2.set_xticklabels(llabels, fontsize=10)
    ax2.set_ylabel("Episodes")
    ax2.set_title("Recognition layer distribution", fontweight="bold")
    ax2.grid(axis="y", alpha=0.25)

    fig.suptitle(f"Experiment 1 — Layer breakdown  |  {DB_NAME}",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUT, "exp1_layer_breakdown.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"[exp1] Saved: {path}")


def save_summary(docs, acc, correct, total):
    by_class = defaultdict(lambda: {"tp": 0, "total": 0})
    for d in docs:
        gt   = d.get("ground_truth","")
        pred = d.get("spatial_action") or d.get("vlm_output","")
        if gt in LABELS:
            by_class[gt]["total"] += 1
            if gt == pred:
                by_class[gt]["tp"] += 1

    lines = [
        f"Experiment 1: HAR Accuracy",
        f"DB: {DB_NAME}",
        f"",
        f"Overall: {acc:.1%}  ({correct}/{total})",
        f"",
        f"Per-class:",
    ]
    for label in LABELS:
        info = by_class.get(label, {"tp":0, "total":0})
        if info["total"] == 0:
            continue
        a    = info["tp"] / info["total"]
        flag = "OK" if a >= 0.70 else ("WN" if a >= 0.40 else "XX")
        lines.append(
            f"  {flag} {label:<14} {a:.1%}  ({info['tp']}/{info['total']})")

    path = os.path.join(OUT, "exp1_summary.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"[exp1] Saved: {path}")
    print("\n".join(lines))


def main():
    os.makedirs(OUT, exist_ok=True)
    db   = connect()
    docs = load_docs(db)

    if not docs:
        print(f"[exp1] No eval_logs in {DB_NAME}")
        return

    print(f"[exp1] {len(docs)} episodes from {DB_NAME}")
    result = plot_confusion(docs, LABELS)
    if result:
        acc, correct, total = result
        plot_layer_breakdown(docs)
        save_summary(docs, acc, correct, total)


if __name__ == "__main__":
    main()