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
    COL_BASELINE, COL_SEMANTIC, COL_VLM_SOM,
    ADL_LABELS, USERS, C,
    FONT_TITLE, FONT_AXIS, FONT_ANNOT, FONT_TICK,
    FIG_DPI, RESULTS_DIR,
    apply_style, load_docs, compute_accuracy,
)

apply_style()


def per_class_accuracy(docs: list) -> dict:
    by_class = defaultdict(lambda: {"tp": 0, "total": 0})
    for d in docs:
        gt = d.get("ground_truth", "")
        if gt in ADL_LABELS:
            by_class[gt]["total"] += 1
            if gt == d.get("_pred", ""):
                by_class[gt]["tp"] += 1
    return by_class


def plot_confusion_matrix(docs: list, save_path: str, system_label: str = "Baseline") -> tuple:
    present = [l for l in ADL_LABELS
               if any(d.get("ground_truth") == l for d in docs)]
    n      = len(present)
    matrix = np.zeros((n, n), dtype=int)

    for d in docs:
        gt   = d.get("ground_truth", "")
        pred = d.get("_pred", "")
        if gt in present and pred in present:
            matrix[present.index(gt)][present.index(pred)] += 1

    total   = int(matrix.sum())
    correct = int(np.trace(matrix))
    acc     = correct / total if total else 0

    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    norm = matrix / row_sums

    fig, ax = plt.subplots(figsize=(11, 9))
    im   = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Recall Rate", fontsize=FONT_AXIS)

    for i in range(n):
        for j in range(n):
            v = norm[i][j]
            if matrix[i][j] > 0:
                ax.text(j, i,
                        f"{v:.2f}\n({matrix[i][j]})",
                        ha="center", va="center", fontsize=7.5,
                        color="white" if v > 0.55 else "black",
                        fontweight="bold" if i == j else "normal")

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(present, rotation=40, ha="right", fontsize=FONT_TICK)
    ax.set_yticklabels(present, fontsize=FONT_TICK)
    ax.set_xlabel("Predicted", fontsize=FONT_AXIS)
    ax.set_ylabel("Ground Truth", fontsize=FONT_AXIS)
    ax.set_title(
        f"HAR Confusion Matrix — {system_label}\n"
        f"Overall Accuracy: {acc:.1%}  ({correct}/{total} episodes)",
        fontsize=FONT_TITLE, fontweight="bold", pad=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"[exp1] Saved: {save_path}")
    return acc, correct, total


def plot_per_class_bar(docs: list, save_path: str, system_label: str = "Baseline"):
    by_class = per_class_accuracy(docs)
    present  = [(l, by_class[l]) for l in ADL_LABELS
                if by_class[l]["total"] > 0]
    present.sort(key=lambda x: x[1]["tp"] / x[1]["total"])

    labels = [p[0] for p in present]
    accs   = [p[1]["tp"] / p[1]["total"] * 100 for p in present]
    totals = [p[1]["total"] for p in present]
    n      = len(labels)

    colors = []
    for a in accs:
        if a >= 80:   colors.append(C["baseline"])
        elif a >= 60: colors.append("#F5A623")
        else:         colors.append(C["corruption_heavy"])

    fig, ax = plt.subplots(figsize=(10, max(5, n * 0.65)))
    bars = ax.barh(range(n), accs, color=colors,
                   alpha=0.88, height=0.55, edgecolor="white")

    for i, (bar, acc, tot) in enumerate(zip(bars, accs, totals)):
        ax.text(min(bar.get_width() + 0.8, 101),
                bar.get_y() + bar.get_height() / 2,
                f"{acc:.1f}%  ({int(acc/100*tot)}/{tot})",
                va="center", fontsize=FONT_ANNOT,
                color="#333", fontweight="bold" if acc < 80 else "normal")

    ax.axvline(80, color="#999", linestyle="--", lw=1.2, alpha=0.6)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=FONT_TICK)
    ax.set_xlabel("Accuracy (%)", fontsize=FONT_AXIS)
    ax.set_xlim(0, 120)
    ax.set_title(f"Per-class Recognition Accuracy — {system_label}",
                 fontsize=FONT_TITLE, fontweight="bold", pad=10)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color=C["baseline"],          label="≥ 80%"),
        Patch(color="#F5A623",              label="60–79%"),
        Patch(color=C["corruption_heavy"],  label="< 60%"),
    ], fontsize=FONT_TICK, loc="lower right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"[exp1] Saved: {save_path}")


def plot_user_breakdown(docs: list, save_path: str, system_label: str = "Baseline"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, uid in zip(axes, USERS):
        user_docs = [d for d in docs if d.get("user") == uid]
        if not user_docs:
            ax.set_title(f"{uid}\n(no data)")
            continue
        by_class = per_class_accuracy(user_docs)
        present  = [(l, by_class[l]) for l in ADL_LABELS
                    if by_class[l]["total"] > 0]
        present.sort(key=lambda x: x[1]["tp"] / x[1]["total"])
        labels = [p[0] for p in present]
        accs   = [p[1]["tp"] / p[1]["total"] * 100 for p in present]
        colors = [C["mom"] if uid == "User_Mom" else C["dad"]] * len(labels)
        bars = ax.barh(range(len(labels)), accs, color=colors,
                       alpha=0.80, height=0.55, edgecolor="white")
        for bar, acc in zip(bars, accs):
            ax.text(min(bar.get_width() + 0.5, 101),
                    bar.get_y() + bar.get_height() / 2,
                    f"{acc:.0f}%", va="center", fontsize=FONT_ANNOT)
        ax.axvline(80, color="#999", linestyle="--", lw=1.0, alpha=0.6)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=FONT_TICK)
        ax.set_xlim(0, 115)
        acc_all, _, _ = compute_accuracy(user_docs)
        ax.set_title(f"{uid}\nOverall: {acc_all:.1%}",
                     fontsize=FONT_TITLE, fontweight="bold")

    plt.suptitle(f"Per-class Accuracy by User — {system_label}",
                 fontsize=FONT_TITLE + 1, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"[exp1] Saved: {save_path}")


def save_summary(docs: list, acc: float, correct: int, total: int, save_path: str,
                  system_label: str = "Baseline", collection_name: str = ""):
    by_class = per_class_accuracy(docs)
    lines = [
        f"Experiment 1: HAR {system_label}",
        f"DB: {DB_BASELINE} | Collection: {collection_name}",
        f"Episodes: {total}  Correct: {correct}  Overall: {acc:.1%}",
        "",
        f"{'Action':<16} {'Acc':>6} {'TP':>5} {'Total':>7}",
        "-" * 38,
    ]
    for label in ADL_LABELS:
        info = by_class.get(label, {"tp": 0, "total": 0})
        if info["total"] == 0:
            continue
        a = info["tp"] / info["total"]
        lines.append(f"{label:<16} {a:>5.1%} {info['tp']:>5} {info['total']:>7}")

    with open(save_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[exp1] Saved: {save_path}")
    print("\n".join(lines))


def plot_system_comparison(docs_a: list, docs_b: list, save_path: str):
    from exp_config import C, FONT_TITLE, FONT_AXIS, FONT_ANNOT, FONT_TICK, FIG_DPI
    import matplotlib.pyplot as plt
    import numpy as np

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
    ax.bar(x - w/2, accs_a, w, label="System A: Skeleton+Object+Spatial→LLM",
           color=C["baseline"], alpha=0.85)
    ax.bar(x + w/2, accs_b, w, label="System B: VLM+SoM→LLM",
           color=C["ablation"], alpha=0.85)

    acc_a, c_a, t_a = compute_accuracy(docs_a)
    acc_b, c_b, t_b = compute_accuracy(docs_b) if docs_b else (0, 0, 0)

    ax.axhline(80, color="#999", linestyle="--", lw=1.0, alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=FONT_TICK)
    ax.set_ylabel("Accuracy (%)", fontsize=FONT_AXIS)
    ax.set_ylim(0, 115)
    ax.set_title(
        f"System A vs System B — Per-class Accuracy\n"
        f"System A: {acc_a:.1%} ({c_a}/{t_a})  |  "
        f"System B: {acc_b:.1%} ({c_b}/{t_b})",
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
    docs_b = load_docs(db, COL_VLM_SOM)

    if not docs_a:
        print(f"[exp1] No System A data in {DB_BASELINE}.{COL_SEMANTIC}")
        return

    print(f"[exp1] System A: {len(docs_a)} episodes")
    print(f"[exp1] System B: {len(docs_b)} episodes")

    # System A analysis
    acc_a, correct_a, total_a = plot_confusion_matrix(
        docs_a, os.path.join(RESULTS_DIR, "exp1_confusion_matrix_semantic.png"),
        system_label="System A (Skeleton+Object+Spatial)")

    plot_per_class_bar(
        docs_a, os.path.join(RESULTS_DIR, "exp1_per_class_bar_semantic.png"),
        system_label="System A (Skeleton+Object+Spatial)")

    plot_user_breakdown(
        docs_a, os.path.join(RESULTS_DIR, "exp1_user_breakdown_semantic.png"),
        system_label="System A (Skeleton+Object+Spatial)")

    save_summary(docs_a, acc_a, correct_a, total_a,
                 os.path.join(RESULTS_DIR, "exp1_summary_semantic.txt"),
                 system_label="System A (Skeleton+Object+Spatial)",
                 collection_name=COL_SEMANTIC)

    # System B analysis（if data exists）
    if docs_b:
        acc_b, correct_b, total_b = plot_confusion_matrix(
            docs_b, os.path.join(RESULTS_DIR, "exp1_confusion_matrix_vlm_som.png"),
            system_label="System B (VLM+SoM)")
        plot_per_class_bar(
            docs_b, os.path.join(RESULTS_DIR, "exp1_per_class_bar_vlm_som.png"),
            system_label="System B (VLM+SoM)")
        save_summary(docs_b, acc_b, correct_b, total_b,
                     os.path.join(RESULTS_DIR, "exp1_summary_vlm_som.txt"),
                     system_label="System B (VLM+SoM)",
                     collection_name=COL_VLM_SOM)

    # Comparison
    plot_system_comparison(
        docs_a, docs_b,
        os.path.join(RESULTS_DIR, "exp1_system_comparison.png"))

    print(f"\n[exp1] System A accuracy: {acc_a:.1%} ({correct_a}/{total_a})")
    if docs_b:
        acc_b, correct_b, total_b = compute_accuracy(docs_b)
        print(f"[exp1] System B accuracy: {acc_b:.1%} ({correct_b}/{total_b})")
        delta = acc_a - acc_b
        winner = "System A" if delta > 0 else "System B"
        print(f"[exp1] {winner} wins by {abs(delta):.1%}")


if __name__ == "__main__":
    main()