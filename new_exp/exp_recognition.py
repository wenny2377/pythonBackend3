"""
exp_recognition.py
Activity Recognition Accuracy Analysis
Reads from eval_logs in MongoDB.

Usage:
  python3 exp_recognition.py
  python3 exp_recognition.py --out results/
"""

import os
import datetime
import argparse
from collections import defaultdict, Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pymongo import MongoClient

MONGO_URI = "mongodb://127.0.0.1:27017/"
DB_NAME   = "robot_rag_db"

BEHAVIOR_ORDER = [
    "Eating", "Drinking", "SittingDrink", "Cooking", "Opening",
    "Laying", "Watching", "Reading", "Cleaning", "PhoneUse", "Typing",
]

UPGRADE_LABELS = {
    "VLM_hint":   "VLM Direct",
    "L3a":        "SayCan (L3a)",
    "L3b_heading":"Heading (L3b)",
    "L3b5":       "Proximity (L3b5)",
    "L3c_zone":   "Zone (L3c)",
    "MinPrior":   "Body Prior",
    "zone_not_ready": "Zone Not Ready",
}


def connect():
    return MongoClient(MONGO_URI)[DB_NAME]


def load_eval_logs(db):
    docs = list(db.eval_logs.find(
        {"ground_truth": {"$exists": True, "$ne": ""}},
        {
            "ground_truth":   1,
            "vlm_output":     1,
            "spatial_action": 1,
            "upgrade_reason": 1,
            "vlm_confidence": 1,
            "infer_source":   1,
            "body_orientation": 1,
        }
    ))
    print(f"  eval_logs loaded: {len(docs)}")
    return docs


def norm(s):
    return (s or "").strip()


def get_layer(reason):
    if not reason:
        return "VLM Direct"
    r = reason.split(":")[0]
    for key, label in UPGRADE_LABELS.items():
        if key in r:
            return label
    return "VLM Direct"


def plot_overall_accuracy(docs, out):
    """
    Figure 1: Overall accuracy comparison
    VLM only vs Full system (spatial_action)
    """
    vlm_correct     = 0
    spatial_correct = 0
    total           = len(docs)

    for d in docs:
        gt  = norm(d.get("ground_truth", ""))
        vlm = norm(d.get("vlm_output", ""))
        spa = norm(d.get("spatial_action", ""))
        if vlm == gt:
            vlm_correct += 1
        if spa == gt:
            spatial_correct += 1

    vlm_acc     = vlm_correct     / total * 100
    spatial_acc = spatial_correct / total * 100
    improvement = spatial_acc - vlm_acc

    fig, ax = plt.subplots(figsize=(7, 5))

    bars = ax.bar(
        ["VLM Only\n(Stage 1)", "Full System\n(Stage 2)"],
        [vlm_acc, spatial_acc],
        color=["#FF9800", "#2196F3"],
        alpha=0.85, edgecolor="white", width=0.5
    )

    for bar, val in zip(bars, [vlm_acc, spatial_acc]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1f}%",
            ha="center", fontsize=14, fontweight="bold"
        )

    ax.annotate(
        f"+{improvement:.1f}%",
        xy=(1, spatial_acc),
        xytext=(1.3, (vlm_acc + spatial_acc) / 2),
        fontsize=12, color="#E53935", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#E53935", lw=1.5)
    )

    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_ylim(0, min(100, max(vlm_acc, spatial_acc) * 1.25))
    ax.set_title(
        f"Figure 1: Overall Activity Recognition Accuracy\n"
        f"N={total} | VLM={vlm_acc:.1f}% | System={spatial_acc:.1f}%",
        fontsize=12, fontweight="bold"
    )
    ax.grid(axis="y", alpha=0.25)
    ax.axhline(y=70, color="#9C27B0", linewidth=1.5,
               linestyle="--", alpha=0.7, label="70% threshold")
    ax.legend(fontsize=10)

    plt.tight_layout()
    path = os.path.join(out, "fig1_overall_accuracy.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return vlm_acc, spatial_acc


def plot_per_class_accuracy(docs, out):
    """
    Figure 2: Per-class accuracy (VLM vs Full System)
    """
    class_gt     = defaultdict(int)
    class_vlm    = defaultdict(int)
    class_spatial = defaultdict(int)

    for d in docs:
        gt  = norm(d.get("ground_truth", ""))
        vlm = norm(d.get("vlm_output", ""))
        spa = norm(d.get("spatial_action", ""))
        if not gt:
            continue
        class_gt[gt] += 1
        if vlm == gt:
            class_vlm[gt] += 1
        if spa == gt:
            class_spatial[gt] += 1

    classes = [c for c in BEHAVIOR_ORDER if c in class_gt]
    classes += [c for c in class_gt if c not in BEHAVIOR_ORDER]

    vlm_accs     = [class_vlm[c]     / class_gt[c] * 100 for c in classes]
    spatial_accs = [class_spatial[c] / class_gt[c] * 100 for c in classes]
    counts       = [class_gt[c] for c in classes]

    x = np.arange(len(classes))
    w = 0.38

    fig, ax = plt.subplots(figsize=(max(10, len(classes) * 1.3), 6))

    b1 = ax.bar(x - w/2, vlm_accs,     w,
                color="#FF9800", alpha=0.85,
                edgecolor="white", label="VLM Only (Stage 1)")
    b2 = ax.bar(x + w/2, spatial_accs, w,
                color="#2196F3", alpha=0.85,
                edgecolor="white", label="Full System (Stage 2)")

    for bar, val in zip(b1, vlm_accs):
        if val > 5:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{val:.0f}%",
                    ha="center", fontsize=8, color="#E65100")

    for bar, val in zip(b2, spatial_accs):
        if val > 5:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{val:.0f}%",
                    ha="center", fontsize=8, color="#1565C0")

    ax2 = ax.twinx()
    ax2.plot(x, counts, "D--", color="#9C27B0",
             linewidth=1.5, markersize=6,
             markerfacecolor="white", markeredgewidth=2,
             label="Sample count", alpha=0.7)
    ax2.set_ylabel("Sample Count", fontsize=11, color="#9C27B0")
    ax2.tick_params(axis="y", labelcolor="#9C27B0")

    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=10)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_ylim(0, 120)
    ax.axhline(y=70, color="#9C27B0", linewidth=1,
               linestyle=":", alpha=0.5)
    ax.set_title(
        "Figure 2: Per-Class Activity Recognition Accuracy\n"
        "VLM Only vs Full System with Spatial Reasoning",
        fontsize=12, fontweight="bold"
    )

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2,
              fontsize=10, loc="upper right")
    ax.grid(axis="y", alpha=0.2)

    plt.tight_layout()
    path = os.path.join(out, "fig2_per_class_accuracy.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_confusion_matrix(docs, out):
    """
    Figure 3: Confusion matrix (Full System)
    """
    classes = [c for c in BEHAVIOR_ORDER
               if any(norm(d.get("ground_truth")) == c for d in docs)]

    n = len(classes)
    matrix = np.zeros((n, n), dtype=int)
    idx_map = {c: i for i, c in enumerate(classes)}

    for d in docs:
        gt  = norm(d.get("ground_truth", ""))
        spa = norm(d.get("spatial_action", ""))
        if gt in idx_map and spa in idx_map:
            matrix[idx_map[gt]][idx_map[spa]] += 1

    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    matrix_norm = matrix / row_sums

    fig, ax = plt.subplots(figsize=(max(8, n * 0.9), max(7, n * 0.8)))
    im = ax.imshow(matrix_norm, cmap="Blues", vmin=0, vmax=1)

    for i in range(n):
        for j in range(n):
            v = matrix_norm[i, j]
            if v > 0.02:
                ax.text(j, i, f"{v:.2f}",
                        ha="center", va="center",
                        fontsize=8,
                        color="white" if v > 0.55 else "black")

    plt.colorbar(im, ax=ax, label="Normalised Count")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(classes, rotation=40, ha="right", fontsize=9)
    ax.set_yticklabels(classes, fontsize=9)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Ground Truth", fontsize=11)
    ax.set_title(
        "Figure 3: Confusion Matrix — Full System\n"
        "(diagonal = correct, off-diagonal = misclassification)",
        fontsize=12, fontweight="bold"
    )

    plt.tight_layout()
    path = os.path.join(out, "fig3_confusion_matrix.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_spatial_reasoning_contribution(docs, out):
    """
    Figure 4: Which reasoning layer contributed to correct predictions
    Ablation study: shows how much each layer improved accuracy
    """
    layer_total   = defaultdict(int)
    layer_correct = defaultdict(int)

    for d in docs:
        gt     = norm(d.get("ground_truth", ""))
        spa    = norm(d.get("spatial_action", ""))
        reason = norm(d.get("upgrade_reason", ""))
        layer  = get_layer(reason)

        layer_total[layer]   += 1
        if spa == gt:
            layer_correct[layer] += 1

    layers = sorted(layer_total.keys(),
                    key=lambda x: -layer_total[x])
    totals  = [layer_total[l]   for l in layers]
    corrects = [layer_correct[l] for l in layers]
    accs    = [layer_correct[l] / layer_total[l] * 100
               for l in layers]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    colors = ["#2196F3", "#4CAF50", "#FF9800",
              "#9C27B0", "#E53935", "#795548", "#607D8B"]

    bars = ax1.bar(layers, totals,
                   color=colors[:len(layers)],
                   alpha=0.85, edgecolor="white")
    for bar, t, c in zip(bars, totals, corrects):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.5,
                 f"n={t}",
                 ha="center", fontsize=9)
    ax1.set_ylabel("Number of Predictions", fontsize=11)
    ax1.set_title("Reasoning Layer Usage Count",
                  fontsize=11, fontweight="bold")
    ax1.set_xticklabels(layers, rotation=20, ha="right", fontsize=9)
    ax1.grid(axis="y", alpha=0.25)

    bars2 = ax2.bar(layers, accs,
                    color=colors[:len(layers)],
                    alpha=0.85, edgecolor="white")
    for bar, acc in zip(bars2, accs):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.5,
                 f"{acc:.1f}%",
                 ha="center", fontsize=9, fontweight="bold")
    ax2.axhline(y=70, color="#9C27B0", linewidth=1.5,
                linestyle="--", alpha=0.7, label="70% threshold")
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.set_ylim(0, 115)
    ax2.set_title("Accuracy per Reasoning Layer",
                  fontsize=11, fontweight="bold")
    ax2.set_xticklabels(layers, rotation=20, ha="right", fontsize=9)
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.25)

    fig.suptitle(
        "Figure 4: Spatial Reasoning Layer Contribution — Ablation Study\n"
        "Left: usage count | Right: accuracy per layer",
        fontsize=12, fontweight="bold"
    )

    plt.tight_layout()
    path = os.path.join(out, "fig4_layer_contribution.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_confidence_vs_accuracy(docs, out):
    """
    Figure 5: VLM confidence vs accuracy
    Shows that confidence gate at 0.4 is justified
    """
    bins       = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    bin_labels = ["0-0.2","0.2-0.3","0.3-0.4","0.4-0.5",
                  "0.5-0.6","0.6-0.7","0.7-0.8","0.8-0.9","0.9-1.0"]

    bin_total   = defaultdict(int)
    bin_correct = defaultdict(int)

    for d in docs:
        gt   = norm(d.get("ground_truth", ""))
        spa  = norm(d.get("spatial_action", ""))
        conf = float(d.get("vlm_confidence", 0.5))

        for i in range(len(bins) - 1):
            if bins[i] <= conf < bins[i+1]:
                bin_total[bin_labels[i]]   += 1
                if spa == gt:
                    bin_correct[bin_labels[i]] += 1
                break

    labels  = [l for l in bin_labels if bin_total[l] > 0]
    totals  = [bin_total[l]   for l in labels]
    accs    = [bin_correct[l] / bin_total[l] * 100
               if bin_total[l] > 0 else 0 for l in labels]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(labels, accs, color="#2196F3", alpha=0.75,
           edgecolor="white", label="Accuracy")

    ax2 = ax.twinx()
    ax2.plot(labels, totals, "D--", color="#FF9800",
             linewidth=2, markersize=7,
             markerfacecolor="white", markeredgewidth=2,
             label="Sample count")
    ax2.set_ylabel("Sample Count", fontsize=11, color="#FF9800")
    ax2.tick_params(axis="y", labelcolor="#FF9800")

    gate_x = bin_labels.index("0.4-0.5") if "0.4-0.5" in bin_labels else 3
    ax.axvline(x=gate_x - 0.5, color="#E53935", linewidth=2,
               linestyle="--", label="Confidence gate (0.40)")

    for i, (acc, tot) in enumerate(zip(accs, totals)):
        if tot > 0:
            ax.text(i, acc + 1, f"{acc:.0f}%",
                    ha="center", fontsize=8, color="#1565C0")

    ax.set_xlabel("VLM Confidence Range", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_ylim(0, 115)
    ax.set_title(
        "Figure 5: VLM Confidence vs Recognition Accuracy\n"
        "Justification for confidence gate at threshold=0.40",
        fontsize=12, fontweight="bold"
    )

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2,
              fontsize=10, loc="upper left")
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    path = os.path.join(out, "fig5_confidence_accuracy.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def save_summary(docs, vlm_acc, spatial_acc, out):
    total = len(docs)

    class_gt      = defaultdict(int)
    class_spatial = defaultdict(int)

    for d in docs:
        gt  = norm(d.get("ground_truth", ""))
        spa = norm(d.get("spatial_action", ""))
        if gt:
            class_gt[gt] += 1
            if spa == gt:
                class_spatial[gt] += 1

    layer_counts = Counter(
        get_layer(norm(d.get("upgrade_reason", "")))
        for d in docs
    )

    lines = [
        "=" * 65,
        "Activity Recognition Accuracy Summary",
        f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 65, "",
        f"Total eval samples : {total}",
        f"VLM Only accuracy  : {vlm_acc:.1f}%",
        f"Full System accuracy: {spatial_acc:.1f}%",
        f"Improvement        : +{spatial_acc - vlm_acc:.1f}%", "",
        "Per-class accuracy (Full System):",
    ]

    for cls in BEHAVIOR_ORDER:
        if cls in class_gt:
            acc = class_spatial[cls] / class_gt[cls] * 100
            lines.append(
                f"  {cls:15}: {acc:5.1f}%"
                f"  ({class_spatial[cls]}/{class_gt[cls]})")

    lines += ["", "Reasoning layer distribution:"]
    for layer, count in layer_counts.most_common():
        lines.append(f"  {layer:20}: {count:4} ({count/total:.1%})")

    path = os.path.join(out, "recognition_summary.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {path}")

    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    db = connect()
    print(f"Connected -> {DB_NAME}")

    print("\n" + "=" * 60)
    print("Activity Recognition Accuracy Analysis")
    print("=" * 60)

    docs = load_eval_logs(db)
    if not docs:
        print("No eval_logs found.")
        exit(1)

    vlm_acc, spatial_acc = plot_overall_accuracy(docs, args.out)
    plot_per_class_accuracy(docs, args.out)
    plot_confusion_matrix(docs, args.out)
    plot_spatial_reasoning_contribution(docs, args.out)
    plot_confidence_vs_accuracy(docs, args.out)
    save_summary(docs, vlm_acc, spatial_acc, args.out)

    print("\nDone. Check", args.out)