"""
analysis/layer1_recognition.py
Layer 1: Behaviour Recognition Analysis
Outputs:
  results/Fig1_confusion.png
  results/Fig2_ablation.png
"""

import os
import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
from pymongo import MongoClient

MONGO_URI = "mongodb://127.0.0.1:27017/"
DB_NAME   = "robot_rag_db"
OUT = os.path.join(os.path.dirname(__file__), "results")

BEHAVIORS = [
    "Drinking", "SittingDrink", "Sitting", "Eating", "Cooking",
    "Opening", "Laying", "Watching", "Reading", "Cleaning",
    "PhoneUse", "Typing",
]
HIGH   = {"Cooking", "Opening", "Laying", "Cleaning", "PhoneUse", "Typing"}
MEDIUM = {"Eating", "Drinking"}
LOW    = {"SittingDrink", "Sitting", "Reading", "Watching"}
NO_WEIGHT = {"PickingUp", "PuttingDown", "Walking", "Standing", "StandUp"}

NORMALIZE = {
    "drinking":"Drinking","sittingdrink":"SittingDrink","sitting":"Sitting",
    "eating":"Eating","cooking":"Cooking","opening":"Opening","laying":"Laying",
    "watching":"Watching","reading":"Reading","cleaning":"Cleaning",
    "phoneuse":"PhoneUse","typing":"Typing","standing":"Standing",
    "walking":"Walking","unknown":"Unknown",
}

def norm(s):
    if not s: return "Unknown"
    return NORMALIZE.get(s.lower().strip().replace(" ","").replace("_",""), s.strip())

def group_color(b):
    if b in HIGH:   return "#F44336"
    if b in MEDIUM: return "#FF9800"
    return "#2196F3"

def connect():
    return MongoClient(MONGO_URI)[DB_NAME]

def plot_fig1_confusion(db):
    print("Fig1: Confusion Matrix...")
    docs = list(db.eval_logs.find(
        {"ground_truth": {"$exists": True, "$ne": ""},
         "spatial_action": {"$exists": True}},
        {"ground_truth": 1, "spatial_action": 1}
    ))
    if not docs:
        print("  No eval_logs found.")
        return

    labels = [b for b in BEHAVIORS
              if any(norm(d["ground_truth"]) == b for d in docs)]
    n = len(labels)
    if n == 0:
        print("  No valid labels.")
        return

    matrix = np.zeros((n, n), dtype=int)
    for d in docs:
        gt   = norm(d.get("ground_truth", ""))
        pred = norm(d.get("spatial_action", ""))
        if gt in labels and pred in labels:
            matrix[labels.index(gt)][labels.index(pred)] += 1

    total   = int(matrix.sum())
    correct = int(np.trace(matrix))
    overall = correct / total if total > 0 else 0

    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    matrix_norm = matrix / row_sums

    def group_acc(group):
        idxs = [i for i, b in enumerate(labels) if b in group]
        if not idxs: return 0
        sub = matrix[np.ix_(idxs, idxs)]
        c, t = int(np.trace(sub)), int(sub.sum())
        return c / t if t > 0 else 0

    high_acc   = group_acc(HIGH)
    medium_acc = group_acc(MEDIUM)
    low_acc    = group_acc(LOW)

    fig, ax = plt.subplots(figsize=(max(10, n), max(8, n)))
    im = ax.imshow(matrix_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Recall Rate")

    for i in range(n):
        for j in range(n):
            v = matrix_norm[i][j]
            if matrix[i][j] > 0:
                ax.text(j, i, f"{v:.2f}\n({matrix[i][j]})",
                        ha="center", va="center", fontsize=7,
                        color="white" if v > 0.55 else "black",
                        fontweight="bold" if i == j else "normal")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    for tick, b in zip(ax.get_xticklabels(), labels):
        tick.set_color(group_color(b))
    for tick, b in zip(ax.get_yticklabels(), labels):
        tick.set_color(group_color(b))

    ax.set_title(
        f"Fig1  Behaviour Recognition Confusion Matrix\n"
        f"Overall = {overall:.1%} ({correct}/{total})  |  "
        f"High: {high_acc:.1%}  Medium: {medium_acc:.1%}  Low: {low_acc:.1%}\n"
        f"[Red=High-specificity  Orange=Medium  Blue=Low-specificity]",
        fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Ground Truth", fontsize=11)

    plt.tight_layout()
    path = os.path.join(OUT, "Fig1_confusion.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    print(f"  Overall: {overall:.1%}  High: {high_acc:.1%}  Medium: {medium_acc:.1%}  Low: {low_acc:.1%}")


def plot_fig2_ablation(db):
    print("Fig2: Ablation Study...")
    docs = list(db.eval_logs.find(
        {"ground_truth":  {"$exists": True, "$ne": ""},
         "spatial_action": {"$exists": True},
         "vlm_output":    {"$exists": True}},
        {"ground_truth": 1, "spatial_action": 1,
         "vlm_output": 1, "upgrade_reason": 1}
    ))
    if not docs:
        print("  No eval_logs found.")
        return

    total = len(docs)

    def dominant_layer(reason):
        r = (reason or "").lower()
        if r.startswith("skeleton") or r.startswith("head("):
            return "skeleton"
        if "skeleton" in r: return "skeleton"
        if "held:"    in r: return "held"
        if "nearby:"  in r: return "nearby"
        if "prox:"    in r or "ray:" in r or "zone:" in r:
            return "geometry"
        return "temporal"

    for d in docs:
        d["_layer"] = dominant_layer(d.get("upgrade_reason", ""))

    def simulate(available):
        return sum(
            1 for d in docs
            if norm(d["_layer"] in available
                    and d.get("spatial_action", "")
                    or d.get("vlm_output", ""))
            == norm(d["ground_truth"])
        )

    c1 = sum(1 for d in docs
             if norm(d.get("vlm_output", "")) == norm(d["ground_truth"]))
    c2 = sum(1 for d in docs
             if (norm(d.get("spatial_action","")) if d["_layer"] == "skeleton"
                 else norm(d.get("vlm_output",""))) == norm(d["ground_truth"]))
    c3 = sum(1 for d in docs
             if (norm(d.get("spatial_action","")) if d["_layer"] in {"skeleton","geometry"}
                 else norm(d.get("vlm_output",""))) == norm(d["ground_truth"]))
    c4 = sum(1 for d in docs
             if (norm(d.get("spatial_action","")) if d["_layer"] in {"skeleton","geometry","held","nearby"}
                 else norm(d.get("vlm_output",""))) == norm(d["ground_truth"]))
    c5 = sum(1 for d in docs
             if norm(d.get("spatial_action","")) == norm(d["ground_truth"]))

    configs = [
        ("VLM Only\n(Baseline)",            c1, "#BDBDBD"),
        ("+ Skeleton\n(hip+head)",           c2, "#2196F3"),
        ("+ Geometry\n(affinity+ray)",       c3, "#4CAF50"),
        ("+ Object Context\n(held+nearby)",  c4, "#FF9800"),
        ("Full System\n(+temporal)",         c5, "#F44336"),
    ]

    accs   = [c / total for _, c, _ in configs]
    labels = [l for l, _, _ in configs]
    colors = [col for _, _, col in configs]
    counts = [c for _, c, _ in configs]
    deltas = [0] + [accs[i] - accs[i-1] for i in range(1, len(accs))]

    print(f"  Config 1 VLM only:     {accs[0]:.1%}")
    print(f"  Config 2 +Skeleton:    {accs[1]:.1%}  Δ={deltas[1]:+.1%}")
    print(f"  Config 3 +Geometry:    {accs[2]:.1%}  Δ={deltas[2]:+.1%}")
    print(f"  Config 4 +Object:      {accs[3]:.1%}  Δ={deltas[3]:+.1%}")
    print(f"  Config 5 Full:         {accs[4]:.1%}  Δ={deltas[4]:+.1%}")

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.bar(range(len(configs)), [a * 100 for a in accs],
                  color=colors, alpha=0.85, edgecolor="white", width=0.6)

    for i, (bar, acc, cnt, delta) in enumerate(
            zip(bars, accs, counts, deltas)):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                f"{acc:.1%}\n(n={cnt})",
                ha="center", fontsize=10, fontweight="bold")
        if i > 0:
            sign = "+" if delta >= 0 else ""
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() / 2,
                    f"{sign}{delta:.1%}",
                    ha="center", fontsize=9,
                    color="white", fontweight="bold")

    for i in range(1, len(accs)):
        x1 = i - 1 + 0.32
        x2 = i     - 0.32
        y  = max(accs[i-1], accs[i]) * 100 + 8
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="->", color="#616161", lw=1.5))

    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_ylim(0, 115)
    ax.set_title(
        f"Fig2  Ablation Study — Incremental Layer Contribution\n"
        f"Total = {total} episodes  |  "
        f"VLM-only = {accs[0]:.1%}  →  Full system = {accs[4]:.1%}",
        fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    path = os.path.join(OUT, "Fig2_ablation.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    db = connect()
    print(f"Connected → {DB_NAME}")
    plot_fig1_confusion(db)
    plot_fig2_ablation(db)
    print("Done.")