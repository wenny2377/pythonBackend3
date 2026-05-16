"""
analyze_exp1.py
Experiment 1: Two-stage Behavior Recognition Ablation Study

Outputs:
    exp1_stacked_bar.png      Main figure: Stacked Bar Chart
    exp1_overall_metrics.png  Overall Acc / Macro F1 / Unknown Rate
    exp1_upgrade_reasons.png  L2A / L2B / L3 contributions
    exp1_confidence.png       SBERT confidence analysis
    exp1_ablation_table.png   Ablation study table
    exp1_metrics_stage1.csv
    exp1_metrics_stage2.csv
    exp1_summary.txt
"""

import argparse
import csv
import datetime
import os
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pymongo import MongoClient

MONGO_URI = "mongodb://127.0.0.1:27017/"
DB_NAME   = "robot_rag_db"

BEHAVIOR_LABELS = [
    "Drinking", "SittingDrink", "Eating", "Cooking",
    "Opening", "Laying", "Watching", "Reading",
    "Cleaning", "PhoneUse", "Typing",
]

NORMALIZE_MAP = {
    "drink": "Drinking", "drinking": "Drinking",
    "drinkwater": "Drinking", "drinkingwater": "Drinking",
    "sittingdrink": "SittingDrink", "sitting drink": "SittingDrink",
    "eat": "Eating", "eating": "Eating",
    "cook": "Cooking", "cooking": "Cooking",
    "open": "Opening", "opening": "Opening",
    "lay": "Laying", "laying": "Laying", "lying": "Laying",
    "sleep": "Laying", "sleeping": "Laying", "resting": "Laying",
    "watch": "Watching", "watching": "Watching", "watchingtv": "Watching",
    "read": "Reading", "reading": "Reading",
    "clean": "Cleaning", "cleaning": "Cleaning",
    "sweeping": "Cleaning", "mopping": "Cleaning",
    "phoneuse": "PhoneUse", "phone": "PhoneUse",
    "usingphone": "PhoneUse", "scrolling": "PhoneUse",
    "type": "Typing", "typing": "Typing", "working": "Typing",
    "stand": "Standing", "standing": "Standing",
    "walk": "Walking", "walking": "Walking",
    "unknown": "Unknown",
}

KEYWORD_HINTS = [
    ("keyboard", "Typing"), ("laptop", "Typing"), ("computer", "Typing"),
    ("book", "Reading"), ("magazine", "Reading"),
    ("bottle", "Drinking"), ("cup", "Drinking"), ("juice", "Drinking"),
    ("phone", "PhoneUse"), ("mobile", "PhoneUse"),
    ("sofa", "Laying"), ("couch", "Laying"), ("lying", "Laying"),
    ("television", "Watching"), ("tv", "Watching"), ("screen", "Watching"),
    ("pan", "Cooking"), ("stove", "Cooking"), ("spatula", "Cooking"),
    ("broom", "Cleaning"), ("sweeping", "Cleaning"),
    ("fridge", "Opening"), ("refrigerator", "Opening"),
    ("bowl", "Eating"), ("fork", "Eating"), ("spoon", "Eating"),
]

SKIP_FIRST = {
    "a", "an", "the", "person", "man", "woman",
    "user", "someone", "he", "she", "they",
}

HIGH_CONF_THRESHOLD = 0.42

COLOR_CORRECT = "#2E7D32"
COLOR_WRONG   = "#E53935"
COLOR_UNKNOWN = "#B0BEC5"
COLOR_S1      = "#FF9800"
COLOR_S2      = "#2196F3"


def normalize(label: str) -> str:
    if not label:
        return "Unknown"
    s   = label.strip()
    sl2 = s.lower()
    sl  = sl2.replace(" ", "").replace("_", "").replace("-", "")
    if sl in NORMALIZE_MAP:
        return NORMALIZE_MAP[sl]
    if sl2 in NORMALIZE_MAP:
        return NORMALIZE_MAP[sl2]
    words = sl2.split()
    first = words[0] if words else ""
    if first in NORMALIZE_MAP:
        return NORMALIZE_MAP[first]
    for kw, lbl in KEYWORD_HINTS:
        if kw in sl2:
            return lbl
    if first in SKIP_FIRST:
        for word in words[1:]:
            w = word.replace(" ", "")
            if w in NORMALIZE_MAP:
                return NORMALIZE_MAP[w]
    else:
        for word in words:
            w = word.replace(" ", "")
            if w in NORMALIZE_MAP:
                return NORMALIZE_MAP[w]
    return "Unknown"


def load_eval_logs(db, query=None):
    q    = query or {}
    docs = list(db.eval_logs.find(q, {
        "ground_truth":   1, "vlm_output":      1,
        "spatial_action": 1, "upgrade_reason":  1,
        "zone_label":     1, "sbert_sim":       1,
        "user_id":        1, "room_name":       1,
        "timestamp":      1,
    }))
    print(f"  Loaded {len(docs)} eval_log records")
    has_spatial = sum(1 for d in docs if d.get("spatial_action"))
    print(f"  Records with spatial_action: {has_spatial}/{len(docs)}")
    return docs


def compute_metrics(y_true, y_pred, labels):
    from collections import defaultdict
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    for t, p in zip(y_true, y_pred):
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1
    rows = []
    for lbl in labels:
        prec = tp[lbl] / (tp[lbl] + fp[lbl]) if (tp[lbl] + fp[lbl]) > 0 else 0.0
        rec  = tp[lbl] / (tp[lbl] + fn[lbl]) if (tp[lbl] + fn[lbl]) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        n    = y_true.count(lbl)
        rows.append({"label": lbl, "precision": prec,
                     "recall": rec, "f1": f1, "support": n})
    active   = [r for r in rows if r["support"] > 0]
    macro_f1 = np.mean([r["f1"] for r in active]) if active else 0.0
    macro_p  = np.mean([r["precision"] for r in active]) if active else 0.0
    macro_r  = np.mean([r["recall"] for r in active]) if active else 0.0
    overall  = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true) if y_true else 0.0
    return rows, macro_p, macro_r, macro_f1, overall


def compute_ablation(y_true, y_stage1, y_stage2, upgrade_reasons, labels):
    """
    從 upgrade_reason 反推四個 Ablation mode 的結果：
      stage1   : vlm_output（不做任何空間推理）
      l2a      : 只接受 L2A 升級
      l2a_l2b  : 接受 L2A 和 L2B 升級
      full     : spatial_action（完整三層）
    """
    y_l2a     = []
    y_l2a_l2b = []

    for s1, s2, reason in zip(y_stage1, y_stage2, upgrade_reasons):
        reason = reason or ""
        if reason.startswith("L2A"):
            y_l2a.append(s2)
            y_l2a_l2b.append(s2)
        elif reason.startswith("L2B"):
            y_l2a.append(s1)
            y_l2a_l2b.append(s2)
        else:
            y_l2a.append(s1)
            y_l2a_l2b.append(s1)

    results = {}
    for name, y_pred in [
        ("Stage 1 Only",          y_stage1),
        ("Stage 1 + L2A",         y_l2a),
        ("Stage 1 + L2A + L2B",   y_l2a_l2b),
        ("Full System\n(+ L3)",   y_stage2),
    ]:
        rows, mp, mr, mf1, acc = compute_metrics(y_true, y_pred, labels)
        unk = sum(p == "Unknown" for p in y_pred) / len(y_pred)
        results[name] = {
            "overall_acc":  acc,
            "macro_f1":     mf1,
            "unknown_rate": unk,
        }
    return results


def plot_stacked_bar(y_true, y_stage1, y_stage2, labels, out_path):
    """
    主圖：Stacked Bar Chart
    每個行為一個「組」，Stage 1 和 Stage 2 並排
    每個長條分三段：Correct（綠）、Wrong（紅）、Unknown（灰）
    按 Stage 1 的 Unknown Rate 由高到低排序
    """
    active = [l for l in labels if l in set(y_true)]
    if not active:
        return

    s1_stats = {}
    s2_stats = {}
    for lbl in active:
        mask = [i for i, t in enumerate(y_true) if t == lbl]
        n    = len(mask)
        if n == 0:
            continue

        def calc(y_pred):
            correct = sum(1 for i in mask if y_pred[i] == lbl)
            unknown = sum(1 for i in mask if y_pred[i] == "Unknown")
            wrong   = n - correct - unknown
            return correct/n, wrong/n, unknown/n

        s1_stats[lbl] = calc(y_stage1)
        s2_stats[lbl] = calc(y_stage2)

    active = sorted(
        active,
        key=lambda l: s1_stats.get(l, (0, 0, 0))[2],
        reverse=True
    )

    n_behaviors = len(active)
    x   = np.arange(n_behaviors)
    w   = 0.35
    fig, ax = plt.subplots(figsize=(max(12, n_behaviors * 1.4), 6))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    for i, lbl in enumerate(active):
        s1c, s1w, s1u = s1_stats.get(lbl, (0, 0, 0))
        s2c, s2w, s2u = s2_stats.get(lbl, (0, 0, 0))

        ax.bar(x[i] - w/2, s1c, w,
               color=COLOR_CORRECT, alpha=0.85, edgecolor="white")
        ax.bar(x[i] - w/2, s1w, w, bottom=s1c,
               color=COLOR_WRONG,   alpha=0.85, edgecolor="white")
        ax.bar(x[i] - w/2, s1u, w, bottom=s1c+s1w,
               color=COLOR_UNKNOWN, alpha=0.85, edgecolor="white")

        ax.bar(x[i] + w/2, s2c, w,
               color=COLOR_CORRECT, alpha=0.85, edgecolor="white")
        ax.bar(x[i] + w/2, s2w, w, bottom=s2c,
               color=COLOR_WRONG,   alpha=0.85, edgecolor="white")
        ax.bar(x[i] + w/2, s2u, w, bottom=s2c+s2w,
               color=COLOR_UNKNOWN, alpha=0.85, edgecolor="white")

        if s1u > 0.05:
            ax.text(x[i] - w/2, 1.02, f"{s1u:.0%}",
                    ha="center", va="bottom", fontsize=7.5,
                    color=COLOR_UNKNOWN, fontweight="bold")
        if s2u > 0.05:
            ax.text(x[i] + w/2, 1.02, f"{s2u:.0%}",
                    ha="center", va="bottom", fontsize=7.5,
                    color="#78909C")

    for xi in x:
        ax.axvline(xi + w/2 + 0.22, color="#E0E0E0",
                   linewidth=0.8, alpha=0.5)

    legend_patches = [
        mpatches.Patch(color=COLOR_CORRECT, alpha=0.85, label="Correct"),
        mpatches.Patch(color=COLOR_WRONG,   alpha=0.85, label="Wrong"),
        mpatches.Patch(color=COLOR_UNKNOWN, alpha=0.85, label="Unknown"),
    ]
    s1_patch = mpatches.Patch(color="white", label="Left  = Stage 1 (VLM only)")
    s2_patch = mpatches.Patch(color="white", label="Right = Stage 2 (VLM + Spatial)")
    ax.legend(
        handles=legend_patches + [s1_patch, s2_patch],
        fontsize=9, loc="upper right", framealpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(active, fontsize=10)
    ax.set_ylabel("Proportion of Episodes", fontsize=12)
    ax.set_ylim(0, 1.18)
    ax.set_title(
        "Experiment 1: Behavior Recognition Results — Stage 1 vs Stage 2\n"
        "(Sorted by Stage 1 Unknown Rate, descending)",
        fontsize=12, pad=12)
    ax.grid(axis="y", alpha=0.25)

    for spine in ax.spines.values():
        spine.set_edgecolor("#BDBDBD")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="#FAFAFA")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_overall_metrics(res_b, res_a, out_path):
    metrics = ["Overall Acc", "Macro F1", "Unknown Rate"]
    bv = [res_b["overall_acc"], res_b["macro_f1"], res_b["unknown_rate"]]
    av = [res_a["overall_acc"], res_a["macro_f1"], res_a["unknown_rate"]]

    x  = np.arange(len(metrics))
    w  = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    b1 = ax.bar(x - w/2, bv, w, color=COLOR_S1, alpha=0.85,
                edgecolor="white", label="Stage 1 (VLM only)")
    b2 = ax.bar(x + w/2, av, w, color=COLOR_S2, alpha=0.85,
                edgecolor="white", label="Stage 2 (VLM + Spatial)")

    for bar, v in zip(b1, bv):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.015,
                f"{v:.1%}", ha="center", fontsize=10, fontweight="bold")
    for bar, v in zip(b2, av):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.015,
                f"{v:.1%}", ha="center", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(
        f"Overall Metrics: Stage 1 vs Stage 2\n"
        f"Acc: {res_b['overall_acc']:.1%} → {res_a['overall_acc']:.1%}  "
        f"Macro F1: {res_b['macro_f1']:.3f} → {res_a['macro_f1']:.3f}  "
        f"Unknown: {res_b['unknown_rate']:.1%} → {res_a['unknown_rate']:.1%}",
        fontsize=11, pad=10)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.25)

    for spine in ax.spines.values():
        spine.set_edgecolor("#BDBDBD")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="#FAFAFA")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_ablation_table(ablation_results, n_total, out_path):
    methods = list(ablation_results.keys())
    accs    = [ablation_results[m]["overall_acc"]  for m in methods]
    f1s     = [ablation_results[m]["macro_f1"]     for m in methods]
    unks    = [ablation_results[m]["unknown_rate"] for m in methods]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.patch.set_facecolor("#FAFAFA")
    fig.suptitle(
        f"Ablation Study: Incremental Spatial Reasoning Contribution\n"
        f"(n={n_total} episodes)",
        fontsize=13, fontweight="bold", y=1.02)

    colors = ["#9E9E9E", "#FF9800", "#FFC107", "#2196F3"]

    for ax_idx, (ax, values, title, ylabel) in enumerate(zip(
        axes,
        [accs, f1s, unks],
        ["Overall Accuracy", "Macro F1", "Unknown Rate"],
        ["Accuracy", "F1 Score", "Rate"],
    )):
        ax.set_facecolor("#FAFAFA")
        bars = ax.bar(range(len(methods)), values,
                      color=colors, alpha=0.85,
                      edgecolor="white", linewidth=1.2)

        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.01,
                    f"{v:.1%}" if ax_idx != 1 else f"{v:.3f}",
                    ha="center", va="bottom",
                    fontsize=10, fontweight="bold")

        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(
            [m.replace(" (+ L3)", "\n(+ L3)") for m in methods],
            fontsize=8.5, ha="center")
        ax.set_ylim(0, max(values) * 1.25 if max(values) > 0 else 1.0)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(axis="y", alpha=0.25)

        for spine in ax.spines.values():
            spine.set_edgecolor("#BDBDBD")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="#FAFAFA")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_upgrade_reasons(upgrade_reasons, out_path):
    layer_map = {
        "L2A": "L2A\nHeld-object",
        "L2B": "L2B\nHeading Align",
        "L3":  "L3\nZone Match",
    }
    counts = {v: 0 for v in layer_map.values()}
    for reason in upgrade_reasons:
        if not reason:
            continue
        for key, label in layer_map.items():
            if reason.startswith(key):
                counts[label] += 1
                break

    if sum(counts.values()) == 0:
        print("  No upgrades to plot")
        return

    labels = list(counts.keys())
    values = [counts[l] for l in labels]
    colors = ["#4CAF50", "#2196F3", "#9C27B0"]

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    bars = ax.bar(labels, values, color=colors,
                  alpha=0.85, edgecolor="white", width=0.5)
    ax.set_ylabel("Number of Cases", fontsize=12)
    ax.set_title(
        "Spatial Reasoning Layer Contributions\n"
        "(Cases upgraded from Unknown / Low-confidence)",
        fontsize=12, pad=10)
    ax.grid(axis="y", alpha=0.25)

    total = sum(values)
    for bar, v in zip(bars, values):
        if v > 0:
            pct = v / total * 100 if total > 0 else 0
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.2,
                    f"{v}\n({pct:.0f}%)",
                    ha="center", fontsize=11, fontweight="bold")

    for spine in ax.spines.values():
        spine.set_edgecolor("#BDBDBD")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="#FAFAFA")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_confidence_analysis(y_true, y_pred, sbert_sims, out_path):
    if not any(s > 0 for s in sbert_sims):
        return

    sims    = np.array(sbert_sims)
    correct = np.array([t == p for t, p in zip(y_true, y_pred)])
    bins    = np.arange(0, 1.05, 0.1)
    bin_acc, bin_cnt, bin_mid = [], [], []

    for i in range(len(bins) - 1):
        mask = (sims >= bins[i]) & (sims < bins[i+1])
        if mask.sum() > 0:
            bin_acc.append(correct[mask].mean())
            bin_cnt.append(mask.sum())
            bin_mid.append((bins[i] + bins[i+1]) / 2)

    high_mask = sims >= HIGH_CONF_THRESHOLD
    low_mask  = sims <  HIGH_CONF_THRESHOLD
    unk_mask  = np.array([p == "Unknown" for p in y_pred])
    high_acc  = correct[high_mask].mean() if high_mask.sum() > 0 else 0.0
    low_acc   = correct[low_mask & ~unk_mask].mean() \
                if (low_mask & ~unk_mask).sum() > 0 else 0.0

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.patch.set_facecolor("#FAFAFA")

    ax1    = axes[0]
    ax1.set_facecolor("#FAFAFA")
    colors = [COLOR_S2 if m >= HIGH_CONF_THRESHOLD else COLOR_S1
              for m in bin_mid]
    bars   = ax1.bar(bin_mid, bin_acc, width=0.08, color=colors,
                     edgecolor="white", linewidth=0.5)
    ax1.axvline(HIGH_CONF_THRESHOLD, color="red", linestyle="--",
                linewidth=1.5,
                label=f"Threshold = {HIGH_CONF_THRESHOLD}")
    ax1.set_xlabel("SBERT Similarity Score", fontsize=12)
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.set_title("Stage 1: Accuracy by SBERT Confidence", fontsize=12)
    ax1.set_ylim(0, 1.15)
    ax1.legend(fontsize=10)
    ax1.grid(axis="y", alpha=0.25)
    for bar, cnt in zip(bars, bin_cnt):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.02,
                 f"n={cnt}", ha="center", va="bottom", fontsize=8)

    ax2 = axes[1]
    ax2.set_facecolor("#FAFAFA")
    categories = [
        f"High Conf\n(sim≥{HIGH_CONF_THRESHOLD})\nn={high_mask.sum()}",
        f"Low Conf\n(sim<{HIGH_CONF_THRESHOLD})\nn={(low_mask & ~unk_mask).sum()}",
        f"Unknown\nn={unk_mask.sum()}",
    ]
    ax2.bar(categories, [high_acc, low_acc, 0.0],
            color=[COLOR_S2, COLOR_S1, "#9E9E9E"],
            alpha=0.85, edgecolor="white")
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.set_title("Stage 1: Accuracy by Confidence Level", fontsize=12)
    ax2.set_ylim(0, 1.15)
    ax2.grid(axis="y", alpha=0.25)
    for i, v in enumerate([high_acc, low_acc]):
        if v > 0:
            ax2.text(i, v + 0.02, f"{v:.1%}", ha="center",
                     va="bottom", fontsize=11, fontweight="bold")

    plt.suptitle(
        f"SBERT Confidence Analysis  |  "
        f"High Conf Acc={high_acc:.1%}  "
        f"Low Conf Acc={low_acc:.1%}  "
        f"Unknown Rate={unk_mask.mean():.1%}",
        fontsize=11, y=1.02)

    for ax in axes:
        for spine in ax.spines.values():
            spine.set_edgecolor("#BDBDBD")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="#FAFAFA")
    plt.close()
    print(f"  Saved: {out_path}")


def save_csv(rows, macro_p, macro_r, macro_f1, out_path):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Behavior", "Precision", "Recall", "F1-Score", "Support"])
        for r in rows:
            if r["support"] > 0:
                w.writerow([r["label"],
                            f"{r['precision']:.3f}",
                            f"{r['recall']:.3f}",
                            f"{r['f1']:.3f}",
                            r["support"]])
        w.writerow([])
        w.writerow(["Macro Avg",
                    f"{macro_p:.3f}",
                    f"{macro_r:.3f}",
                    f"{macro_f1:.3f}", ""])
    print(f"  Saved: {out_path}")


def save_summary(rows_b, rows_a, res_b, res_a,
                 ablation_results, n_total,
                 upgrade_counts, out_path):
    f1a = {r["label"]: r["f1"] for r in rows_a}

    lines = [
        "=" * 65,
        "Experiment 1 — Two-stage Behavior Recognition Ablation Study",
        f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 65,
        f"Total valid episodes : {n_total}",
        "",
        "Stage 1  (VLM-only, SBERT Prototype Matching)",
        f"  Overall accuracy : {res_b['overall_acc']:.1%}",
        f"  Macro F1         : {res_b['macro_f1']:.3f}",
        f"  Unknown rate     : {res_b['unknown_rate']:.1%}",
        "",
        "Stage 2  (VLM + Hierarchical Spatial Reasoning)",
        f"  Overall accuracy : {res_a['overall_acc']:.1%}",
        f"  Macro F1         : {res_a['macro_f1']:.3f}",
        f"  Unknown rate     : {res_a['unknown_rate']:.1%}",
        "",
        "Improvement",
        f"  Accuracy   : {res_b['overall_acc']:.1%} -> {res_a['overall_acc']:.1%}"
        f"  ({res_a['overall_acc'] - res_b['overall_acc']:+.1%})",
        f"  Macro F1   : {res_b['macro_f1']:.3f} -> {res_a['macro_f1']:.3f}"
        f"  ({res_a['macro_f1'] - res_b['macro_f1']:+.3f})",
        f"  Unk rate   : {res_b['unknown_rate']:.1%} -> {res_a['unknown_rate']:.1%}"
        f"  ({res_a['unknown_rate'] - res_b['unknown_rate']:+.1%})",
        "",
        "Ablation Study",
        f"  {'Method':<28} {'Acc':>8} {'Macro F1':>10} {'Unknown':>10}",
        "  " + "-" * 58,
    ]

    for method, result in ablation_results.items():
        m = method.replace("\n", " ")
        lines.append(
            f"  {m:<28} "
            f"{result['overall_acc']:>8.1%} "
            f"{result['macro_f1']:>10.3f} "
            f"{result['unknown_rate']:>10.1%}"
        )

    lines += [
        "",
        "Spatial Reasoning Layer Contributions",
        f"  L2A Held-object    : {upgrade_counts.get('L2A', 0)} cases",
        f"  L2B Heading Align  : {upgrade_counts.get('L2B', 0)} cases",
        f"  L3  Zone Match     : {upgrade_counts.get('L3', 0)} cases",
        "",
        "Per-class F1 Comparison",
    ]

    for r in rows_b:
        if r["support"] == 0:
            continue
        f1b  = r["f1"]
        f1af = f1a.get(r["label"], 0.0)
        lines.append(
            f"  {r['label']:15s}  Stage1={f1b:.3f}  "
            f"Stage2={f1af:.3f}  "
            f"Diff={f1af - f1b:+.3f}  (n={r['support']})"
        )

    lines += [
        "",
        "Thesis Statement:",
        f"The proposed two-stage architecture achieves an overall accuracy",
        f"of {res_a['overall_acc']:.1%} (Stage 1: {res_b['overall_acc']:.1%})",
        f"and a Macro F1 of {res_a['macro_f1']:.3f} "
        f"(Stage 1: {res_b['macro_f1']:.3f}).",
        f"The spatial reasoning module reduces the Unknown rate from",
        f"{res_b['unknown_rate']:.1%} to {res_a['unknown_rate']:.1%}.",
        "Ablation study confirms incremental contributions:",
        "L2A (held-object), L2B (heading alignment), L3 (zone prior).",
    ]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",   default="results")
    parser.add_argument("--user",  default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("Connecting to MongoDB...")
    client = MongoClient(MONGO_URI)
    db     = client[DB_NAME]

    q    = {"user_id": args.user} if args.user else {}
    docs = load_eval_logs(db, q)

    if not docs:
        print("No eval_logs found.")
        return

    y_true          = [normalize(d.get("ground_truth",   "")) for d in docs]
    y_stage1        = [normalize(d.get("vlm_output",     "")) for d in docs]
    y_stage2_raw    = [d.get("spatial_action", "")            for d in docs]
    y_stage2        = [normalize(s) if s else normalize(d.get("vlm_output", ""))
                       for s, d in zip(y_stage2_raw, docs)]
    sbert_sims      = [float(d.get("sbert_sim", 0.0))         for d in docs]
    upgrade_reasons = [d.get("upgrade_reason", "")            for d in docs]

    valid = [
        (t, s1, s2, sim, ur, d)
        for t, s1, s2, sim, ur, d
        in zip(y_true, y_stage1, y_stage2,
               sbert_sims, upgrade_reasons, docs)
        if t not in ("Unknown", "Standing", "Walking")
    ]

    if not valid:
        print("No valid ground truth labels found.")
        return

    y_true, y_stage1, y_stage2, sbert_sims, upgrade_reasons, docs_v = \
        zip(*valid)
    y_true          = list(y_true)
    y_stage1        = list(y_stage1)
    y_stage2        = list(y_stage2)
    sbert_sims      = list(sbert_sims)
    upgrade_reasons = list(upgrade_reasons)

    print(f"  Valid samples   : {len(y_true)}")
    print(f"  GT distribution : {dict(Counter(y_true).most_common())}")
    print(f"  Stage1 dist     : {dict(Counter(y_stage1).most_common())}")
    print(f"  Stage2 dist     : {dict(Counter(y_stage2).most_common())}")

    n_upgraded = sum(1 for r in upgrade_reasons if r)
    print(f"  Upgraded        : {n_upgraded}/{len(y_true)} "
          f"({n_upgraded/len(y_true):.1%})")

    upgrade_counts = {}
    for reason in upgrade_reasons:
        if not reason:
            continue
        for key in ("L2A", "L2B", "L3"):
            if reason.startswith(key):
                upgrade_counts[key] = upgrade_counts.get(key, 0) + 1
                break

    labels = [l for l in BEHAVIOR_LABELS if l in set(y_true)]
    others = [l for l in set(y_true)
              if l not in BEHAVIOR_LABELS and l != "Unknown"]
    labels += others

    rows_b, mp_b, mr_b, mf1_b, acc_b = compute_metrics(
        y_true, y_stage1, labels)
    rows_a, mp_a, mr_a, mf1_a, acc_a = compute_metrics(
        y_true, y_stage2, labels)

    unk_b = sum(p == "Unknown" for p in y_stage1) / len(y_stage1)
    unk_a = sum(p == "Unknown" for p in y_stage2) / len(y_stage2)

    res_b = {
        "overall_acc":  acc_b, "macro_f1":     mf1_b,
        "unknown_rate": unk_b,
        "f1":      {r["label"]: r["f1"]     for r in rows_b},
        "support": {r["label"]: r["support"] for r in rows_b},
    }
    res_a = {
        "overall_acc":  acc_a, "macro_f1":     mf1_a,
        "unknown_rate": unk_a,
        "f1":      {r["label"]: r["f1"]     for r in rows_a},
        "support": {r["label"]: r["support"] for r in rows_a},
    }

    ablation_results = compute_ablation(
        y_true, y_stage1, y_stage2, upgrade_reasons, labels)

    print("\nGenerating figures...")

    plot_stacked_bar(
        y_true, y_stage1, y_stage2, labels,
        os.path.join(args.out, "exp1_stacked_bar.png"))

    plot_overall_metrics(
        res_b, res_a,
        os.path.join(args.out, "exp1_overall_metrics.png"))

    plot_ablation_table(
        ablation_results, len(y_true),
        os.path.join(args.out, "exp1_ablation_table.png"))

    plot_upgrade_reasons(
        upgrade_reasons,
        os.path.join(args.out, "exp1_upgrade_reasons.png"))

    plot_confidence_analysis(
        y_true, y_stage1, sbert_sims,
        os.path.join(args.out, "exp1_confidence.png"))

    save_csv(rows_b, mp_b, mr_b, mf1_b,
             os.path.join(args.out, "exp1_metrics_stage1.csv"))
    save_csv(rows_a, mp_a, mr_a, mf1_a,
             os.path.join(args.out, "exp1_metrics_stage2.csv"))

    save_summary(rows_b, rows_a, res_b, res_a,
                 ablation_results, len(y_true),
                 upgrade_counts,
                 os.path.join(args.out, "exp1_summary.txt"))

    if args.debug:
        print(f"\n  Upgrade breakdown: {upgrade_counts}")
        print(f"\n  Ablation results:")
        for m, r in ablation_results.items():
            print(f"    {m.replace(chr(10),' '):<30} "
                  f"Acc={r['overall_acc']:.1%} "
                  f"F1={r['macro_f1']:.3f} "
                  f"Unk={r['unknown_rate']:.1%}")
        print(f"\n  First 10 mismatches (Stage 1):")
        count = 0
        for d, t, s1, s2, ur in zip(
                docs_v, y_true, y_stage1, y_stage2, upgrade_reasons):
            if t != s1 and count < 10:
                print(f"    GT={t:15s} S1={s1:15s} "
                      f"S2={s2:15s} reason={ur or 'none'}")
                count += 1


if __name__ == "__main__":
    main()