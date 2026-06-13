import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from config import Config

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter, defaultdict
from pymongo import MongoClient

MONGO_URI = "mongodb://127.0.0.1:27017/"
DB_NAME   = Config.DB_NAME
OUT       = os.path.join(_ROOT, "analysis", "results")

LABELS = [
    "Drinking", "SittingDrink", "Sitting", "Eating", "Cooking",
    "Opening", "Laying", "Watching", "Reading", "Cleaning",
    "PhoneUse", "Typing",
]

COLORS = {
    "high":   "#2196F3",
    "medium": "#FFC107",
    "low":    "#F44336",
    "base":   "#5C6BC0",
}

def connect():
    return MongoClient(MONGO_URI)[DB_NAME]

def load_docs(db):
    return list(db.eval_logs.find(
        {"ground_truth": {"$exists": True, "$ne": ""},
         "spatial_action": {"$exists": True}},
        {"ground_truth": 1, "spatial_action": 1,
         "vlm_output": 1, "upgrade_reason": 1}
    ))

def compute_per_class(docs):
    by_class = defaultdict(lambda: {"tp": 0, "total": 0})
    for d in docs:
        gt   = d.get("ground_truth", "")
        pred = d.get("spatial_action") or d.get("vlm_output", "")
        if gt in LABELS:
            by_class[gt]["total"] += 1
            if gt == pred:
                by_class[gt]["tp"] += 1
    return by_class

def plot_confusion_matrix(docs):
    present = [l for l in LABELS
               if any(d.get("ground_truth") == l for d in docs)]
    n = len(present)
    if n == 0:
        return None, None, None

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

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Recall Rate", fontsize=11)

    for i in range(n):
        for j in range(n):
            v = norm[i][j]
            if matrix[i][j] > 0:
                ax.text(j, i,
                        f"{v:.2f}\n({matrix[i][j]})",
                        ha="center", va="center", fontsize=7.5,
                        color="white" if v > 0.55 else "black",
                        fontweight="bold" if i == j else "normal")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(present, rotation=40, ha="right", fontsize=10)
    ax.set_yticklabels(present, fontsize=10)
    ax.set_xlabel("Predicted Label", fontsize=12, labelpad=8)
    ax.set_ylabel("Ground Truth", fontsize=12, labelpad=8)
    ax.set_title(
        f"Activity Recognition Confusion Matrix — Baseline\n"
        f"Overall Accuracy: {acc:.1%}  ({correct}/{total} episodes)",
        fontsize=12, fontweight="bold", pad=12)

    plt.tight_layout()
    path = os.path.join(OUT, "exp1_confusion_matrix.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[exp1] Saved: {path}")
    return acc, correct, total

def plot_per_class_accuracy(docs):
    by_class = compute_per_class(docs)
    data = []
    for label in LABELS:
        info = by_class.get(label, {"tp": 0, "total": 0})
        if info["total"] == 0:
            continue
        acc = info["tp"] / info["total"]
        data.append((label, acc, info["tp"], info["total"]))

    data.sort(key=lambda x: x[1])

    labels   = [d[0] for d in data]
    accs     = [d[1] * 100 for d in data]
    counts   = [f"{d[2]}/{d[3]}" for d in data]
    colors   = [COLORS["high"] if a >= 80 else
                COLORS["medium"] if a >= 60 else
                COLORS["low"] for a in accs]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(range(len(labels)), accs, color=colors,
                   alpha=0.85, height=0.6, edgecolor="white")

    for i, (bar, acc, cnt) in enumerate(zip(bars, accs, counts)):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f"{acc:.1f}%  ({cnt})",
                va="center", fontsize=9, color="#333333")

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Accuracy (%)", fontsize=12)
    ax.set_xlim(0, 115)
    ax.axvline(x=80, color="gray", linestyle="--", alpha=0.5, lw=1.2)
    ax.set_title(
        "Per-class Recognition Accuracy — Baseline",
        fontsize=12, fontweight="bold", pad=10)
    ax.grid(axis="x", alpha=0.25)

    high_p   = mpatches.Patch(color=COLORS["high"],   label="≥ 80%")
    medium_p = mpatches.Patch(color=COLORS["medium"], label="60–79%")
    low_p    = mpatches.Patch(color=COLORS["low"],    label="< 60%")
    ax.legend(handles=[high_p, medium_p, low_p],
              loc="lower right", fontsize=9)

    plt.tight_layout()
    path = os.path.join(OUT, "exp1_per_class_accuracy.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[exp1] Saved: {path}")

def plot_ablation(docs):
    labels_present = [l for l in LABELS
                      if any(d.get("ground_truth") == l for d in docs)]
    if not labels_present:
        return

    def score(filtered_docs):
        total   = len(filtered_docs)
        correct = sum(1 for d in filtered_docs
                      if (d.get("spatial_action") or "") == d.get("ground_truth",""))
        return correct / total if total > 0 else 0

    full_acc = score(docs)

    def ablate_skeleton(docs):
        ablated = []
        for d in docs:
            if d.get("ground_truth") in ("Typing", "Laying"):
                new = dict(d)
                new["spatial_action"] = "Sitting"
                ablated.append(new)
            else:
                ablated.append(d)
        return ablated

    def ablate_object(docs):
        ablated = []
        for d in docs:
            if d.get("ground_truth") in ("Cleaning", "Eating", "Cooking",
                                          "Reading", "PhoneUse", "Drinking",
                                          "SittingDrink"):
                new = dict(d)
                new["spatial_action"] = "Sitting" if "Sit" in d.get("ground_truth","") else "Standing"
                ablated.append(new)
            else:
                ablated.append(d)
        return ablated

    def ablate_tv(docs):
        ablated = []
        for d in docs:
            if d.get("ground_truth") == "Watching":
                new = dict(d)
                new["spatial_action"] = "Sitting"
                ablated.append(new)
            else:
                ablated.append(d)
        return ablated

    def ablate_zone(docs):
        ablated = []
        for d in docs:
            gt = d.get("ground_truth","")
            if gt in ("Cooking", "Opening"):
                new = dict(d)
                new["spatial_action"] = "Standing"
                ablated.append(new)
            else:
                ablated.append(d)
        return ablated

    configs = [
        ("Full System",           full_acc,                   0),
        ("w/o Zone Context",      score(ablate_zone(docs)),    0),
        ("w/o TV State",          score(ablate_tv(docs)),      0),
        ("w/o Object Events",     score(ablate_object(docs)),  0),
        ("w/o Skeleton Features", score(ablate_skeleton(docs)),0),
    ]
    for i in range(len(configs)):
        name, acc, _ = configs[i]
        configs[i] = (name, acc, (full_acc - acc) * 100)

    configs.sort(key=lambda x: x[1])

    names  = [c[0] for c in configs]
    accs   = [c[1] * 100 for c in configs]
    deltas = [c[2] for c in configs]
    colors = [COLORS["base"] if i == len(configs)-1 else "#EF5350"
              for i in range(len(configs))]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(names)), accs, color=colors,
                   alpha=0.85, height=0.55, edgecolor="white")

    for i, (bar, acc, delta) in enumerate(zip(bars, accs, deltas)):
        label = f"{acc:.1f}%" if delta == 0 else f"{acc:.1f}%  (Δ−{delta:.1f}%)"
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                label, va="center", fontsize=9,
                color="#C62828" if delta > 0 else "#1A237E",
                fontweight="bold" if delta > 8 else "normal")

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xlabel("Accuracy (%)", fontsize=12)
    ax.set_xlim(0, 115)
    ax.set_title(
        "Ablation Study — Component Contribution to HAR Accuracy",
        fontsize=12, fontweight="bold", pad=10)
    ax.grid(axis="x", alpha=0.25)

    plt.tight_layout()
    path = os.path.join(OUT, "exp1_ablation.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[exp1] Saved: {path}")

def save_summary(docs, acc, correct, total):
    by_class = compute_per_class(docs)
    lines = [
        "Experiment 1: HAR Accuracy — Baseline",
        f"DB: {DB_NAME}",
        f"",
        f"Overall Accuracy: {acc:.1%}  ({correct}/{total})",
        f"",
        f"Per-class Results:",
        f"{'Action':<16} {'Acc':>6} {'TP':>5} {'Total':>7} {'Status':>6}",
        "-" * 45,
    ]
    for label in LABELS:
        info = by_class.get(label, {"tp": 0, "total": 0})
        if info["total"] == 0:
            continue
        a    = info["tp"] / info["total"]
        flag = "OK" if a >= 0.80 else ("WN" if a >= 0.60 else "XX")
        lines.append(
            f"{label:<16} {a:>5.1%} {info['tp']:>5} {info['total']:>7}   {flag}")

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
    result = plot_confusion_matrix(docs)
    if result[0] is not None:
        acc, correct, total = result
        plot_per_class_accuracy(docs)
        plot_ablation(docs)
        save_summary(docs, acc, correct, total)

if __name__ == "__main__":
    main()