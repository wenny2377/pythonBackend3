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
    MONGO_URI, DB_BASELINE,
    COL_BASELINE, COL_SEMANTIC, COL_VLM,
    ADL_LABELS, USERS, C,
    FONT_TITLE, FONT_AXIS, FONT_ANNOT, FONT_TICK,
    FIG_DPI, RESULTS_DIR,
    apply_style, load_docs, compute_accuracy,
)

apply_style()


def plot_confusion_matrix(docs: list, save_path: str, system_label: str = "Baseline") -> tuple:
    gt_labels   = [l for l in ADL_LABELS if any(d.get("ground_truth") == l for d in docs)]
    extra_preds = sorted(set(d.get("_pred", "") for d in docs if d.get("_pred") and d.get("_pred") not in gt_labels))
    all_labels  = gt_labels + extra_preds
    n_gt        = len(gt_labels)
    n_all       = len(all_labels)
    matrix = np.zeros((n_gt, n_all), dtype=int)

    for d in docs:
        gt   = d.get("ground_truth", "")
        pred = d.get("_pred", "")
        if gt in gt_labels and pred in all_labels:
            matrix[gt_labels.index(gt)][all_labels.index(pred)] += 1

    total   = int(matrix.sum())
    correct = sum(matrix[i][i] for i in range(n_gt))
    acc     = correct / total if total else 0

    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    norm = matrix / row_sums

    fig, ax = plt.subplots(figsize=(max(11, n_all), n_gt))
    im   = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Recall Rate", fontsize=FONT_AXIS)

    for i in range(n_gt):
        for j in range(n_all):
            v = norm[i][j]
            if matrix[i][j] > 0:
                ax.text(j, i,
                        f"{v:.2f}\n({matrix[i][j]})",
                        ha="center", va="center", fontsize=7.5,
                        color="white" if v > 0.55 else "black",
                        fontweight="bold" if gt_labels[i] == all_labels[j] else "normal")

    ax.set_xticks(range(n_all)); ax.set_yticks(range(n_gt))
    ax.set_xticklabels(all_labels, rotation=40, ha="right", fontsize=FONT_TICK)
    ax.set_yticklabels(gt_labels, fontsize=FONT_TICK)
    ax.set_xlabel("Predicted", fontsize=FONT_AXIS)
    ax.set_ylabel("Ground Truth", fontsize=FONT_AXIS)
    ax.set_title(
        f"HAR Confusion Matrix — {system_label}\n"
        f"Overall Accuracy: {acc:.1%}",
        fontsize=FONT_TITLE, fontweight="bold", pad=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"[exp1] Saved: {save_path}")
    return acc, correct, total


def plot_system_comparison(docs_a: list, docs_b: list, save_path: str):
    labels_a = [l for l in ADL_LABELS if any(d.get("ground_truth") == l for d in docs_a)]
    labels   = labels_a

    def get_class_accs(docs):
        accs = []
        for label in labels:
            ld = [d for d in docs if d.get("ground_truth") == label]
            if not ld:
                accs.append(0.0)
                continue
            correct = sum(1 for d in ld if d.get("_pred") == label)
            accs.append(correct / len(ld) * 100)
        return accs

    accs_a = get_class_accs(docs_a)
    accs_b = get_class_accs(docs_b) if docs_b else [0.0] * len(labels)

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - w/2, accs_a, w,
           label="Semantic System",
           color=C["baseline"], alpha=0.85)
    ax.bar(x + w/2, accs_b, w,
           label="VLM System",
           color=C["ablation"], alpha=0.85)

    # Add value labels on each bar
    for i, (a, b) in enumerate(zip(accs_a, accs_b)):
        if a > 0:
            ax.text(i - w/2, a + 1, f"{a:.0f}%",
                    ha="center", va="bottom", fontsize=8, color="#333")
        else:
            ax.text(i - w/2, 2, "0%",
                    ha="center", va="bottom", fontsize=8, color="red", fontweight="bold")
        if b > 0:
            ax.text(i + w/2, b + 1, f"{b:.0f}%",
                    ha="center", va="bottom", fontsize=8, color="#333")
        else:
            ax.text(i + w/2, 2, "0%",
                    ha="center", va="bottom", fontsize=8, color="red", fontweight="bold")

    acc_a, c_a, t_a = compute_accuracy(docs_a)
    acc_b, c_b, t_b = compute_accuracy(docs_b) if docs_b else (0, 0, 0)

    ax.axhline(80, color="#999", linestyle="--", lw=1.0, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=FONT_TICK)
    ax.set_ylabel("Accuracy (%)", fontsize=FONT_AXIS)
    ax.set_ylim(0, 115)
    ax.set_title(
        f"Semantic System vs VLM System — Per-class Accuracy\n"
        f"Semantic: {acc_a:.1%}  |  VLM: {acc_b:.1%}",
        fontsize=FONT_TITLE, fontweight="bold", pad=10)
    ax.legend(fontsize=FONT_TICK)

    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"[exp1] Saved: {save_path}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    db = MongoClient(MONGO_URI)[DB_BASELINE]

    docs_a = load_docs(db, COL_SEMANTIC)
    docs_b = load_docs(db, COL_VLM)

    if not docs_a:
        print(f"[exp1] No Semantic System data in {DB_BASELINE}.{COL_SEMANTIC}")
        return

    print(f"[exp1] Semantic System: {len(docs_a)} episodes")
    print(f"[exp1] VLM System: {len(docs_b)} episodes")

    # Confusion matrix — Semantic System
    acc_a, correct_a, total_a = plot_confusion_matrix(
        docs_a,
        os.path.join(RESULTS_DIR, "exp1_confusion_matrix_semantic.png"),
        system_label="Semantic System (Skeleton+Object+Spatiotemporal)")

    # Confusion matrix — VLM System
    if docs_b:
        acc_b, correct_b, total_b = plot_confusion_matrix(
            docs_b,
            os.path.join(RESULTS_DIR, "exp1_confusion_matrix_vlm.png"),
            system_label="VLM System")

    # System comparison bar chart
    plot_system_comparison(
        docs_a, docs_b,
        os.path.join(RESULTS_DIR, "exp1_system_comparison.png"))

    print(f"\n[exp1] Semantic System: {acc_a:.1%} ({correct_a}/{total_a})")
    if docs_b:
        acc_b, correct_b, total_b = compute_accuracy(docs_b)
        print(f"[exp1] VLM System: {acc_b:.1%} ({correct_b}/{total_b})")
        delta = acc_a - acc_b
        winner = "Semantic System" if delta > 0 else "VLM System"
        print(f"[exp1] {winner} wins by {abs(delta):.1%}")


if __name__ == "__main__":
    main()