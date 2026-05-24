import argparse
import csv
import datetime
import os
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    ("bowl", "Eating"), ("fork", "Eating"), ("spoon", "Eating"),("saladbowl", "Eating"),
]

SKIP_FIRST = {
    "a", "an", "the", "person", "man", "woman",
    "user", "someone", "he", "she", "they",
}

HIGH_CONF_THRESHOLD = 0.42


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
        "ground_truth":  1, "vlm_output":    1,
        "spatial_action": 1, "upgrade_reason": 1,
        "zone_label":    1, "sbert_sim":     1,
        "user_id":       1, "room_name":     1,
        "timestamp":     1,
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


def build_confusion_matrix(y_true, y_pred, labels):
    idx = {l: i for i, l in enumerate(labels)}
    n   = len(labels)
    cm  = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            cm[idx[t]][idx[p]] += 1
    return cm


def plot_confusion_matrix(cm, labels, out_path, overall_acc, macro_f1, title_prefix=""):
    active  = [l for l in labels
               if cm[labels.index(l)].sum() > 0 or
               cm[:, labels.index(l)].sum() > 0]
    if not active:
        print("  No data to plot")
        return
    act_idx = [labels.index(l) for l in active]
    cm_sub  = cm[np.ix_(act_idx, act_idx)]
    n       = len(active)
    fig, ax = plt.subplots(figsize=(max(8, n * 1.1), max(7, n * 1.0)))
    im      = ax.imshow(cm_sub, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(active, rotation=40, ha="right", fontsize=10)
    ax.set_yticklabels(active, fontsize=10)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("Ground Truth Label", fontsize=12)
    ax.set_title(
        f"{title_prefix}Confusion Matrix\n"
        f"Overall Accuracy = {overall_acc:.1%}   Macro F1 = {macro_f1:.3f}",
        fontsize=12, pad=12,
    )
    thresh = cm_sub.max() / 2.0 if cm_sub.max() > 0 else 1
    for i in range(n):
        for j in range(n):
            v = cm_sub[i, j]
            if v > 0:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=9,
                        color="white" if v > thresh else "black",
                        fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_comparison(res_b, res_a, labels, out_path):
    active    = [l for l in labels if res_b["support"].get(l, 0) > 0]
    if not active:
        return
    x         = np.arange(len(active))
    w         = 0.35
    f1_before = [res_b["f1"].get(l, 0) for l in active]
    f1_after  = [res_a["f1"].get(l, 0)  for l in active]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax1 = axes[0]
    ax1.bar(x - w/2, f1_before, w, label="Stage 1 (VLM only)",
            color="#FF9800", alpha=0.85, edgecolor="white")
    ax1.bar(x + w/2, f1_after,  w, label="Stage 2 (VLM + Spatial)",
            color="#2196F3", alpha=0.85, edgecolor="white")
    ax1.set_xticks(x)
    ax1.set_xticklabels(active, rotation=40, ha="right", fontsize=10)
    ax1.set_ylabel("F1 Score", fontsize=12)
    ax1.set_ylim(0, 1.1)
    ax1.set_title("Per-class F1: Stage 1 vs Stage 2", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(axis="y", alpha=0.3)

    ax2     = axes[1]
    metrics = ["Overall Acc", "Macro F1", "Unknown Rate"]
    bv      = [res_b["overall_acc"], res_b["macro_f1"], res_b["unknown_rate"]]
    av      = [res_a["overall_acc"], res_a["macro_f1"], res_a["unknown_rate"]]
    xi      = np.arange(len(metrics))
    ax2.bar(xi - w/2, bv, w, label="Stage 1 (VLM only)",
            color="#FF9800", alpha=0.85, edgecolor="white")
    ax2.bar(xi + w/2, av, w, label="Stage 2 (VLM + Spatial)",
            color="#2196F3", alpha=0.85, edgecolor="white")
    ax2.set_xticks(xi)
    ax2.set_xticklabels(metrics, fontsize=11)
    ax2.set_ylim(0, 1.1)
    ax2.set_title("Overall Metrics Comparison", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(axis="y", alpha=0.3)
    for i, (b, a) in enumerate(zip(bv, av)):
        ax2.text(i - w/2, b + 0.02, f"{b:.1%}", ha="center", fontsize=9)
        ax2.text(i + w/2, a + 0.02, f"{a:.1%}", ha="center", fontsize=9)

    plt.suptitle(
        f"Spatial Reasoning Impact  |  "
        f"Acc: {res_b['overall_acc']:.1%} -> {res_a['overall_acc']:.1%}  "
        f"Macro F1: {res_b['macro_f1']:.3f} -> {res_a['macro_f1']:.3f}",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_upgrade_reasons(upgrade_reasons, out_path):
    layer_map = {
        "L2A": "L2A Held-object",
        "L2B": "L2B Heading Alignment",
        "L3":  "L3 Zone Match",
    }
    counts = {"L2A Held-object": 0, "L2B Heading Alignment": 0, "L3 Zone Match": 0}
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

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors, alpha=0.85, edgecolor="white")
    ax.set_ylabel("Number of Cases", fontsize=12)
    ax.set_title("Spatial Reasoning Upgrade Sources\n"
                 "(Which layer contributed to upgrading Unknown/Low-confidence predictions)",
                 fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, values):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.3,
                    str(v), ha="center", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
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

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax1    = axes[0]
    colors = ["#2196F3" if m >= HIGH_CONF_THRESHOLD else "#FF9800" for m in bin_mid]
    bars   = ax1.bar(bin_mid, bin_acc, width=0.08, color=colors,
                     edgecolor="white", linewidth=0.5)
    ax1.axvline(HIGH_CONF_THRESHOLD, color="red", linestyle="--",
                linewidth=1.5, label=f"Threshold={HIGH_CONF_THRESHOLD}")
    ax1.set_xlabel("SBERT Similarity Score", fontsize=12)
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.set_title("Stage 1 Accuracy by SBERT Confidence Score", fontsize=12)
    ax1.set_ylim(0, 1.1)
    ax1.legend(fontsize=10)
    for bar, cnt in zip(bars, bin_cnt):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.02,
                 f"n={cnt}", ha="center", va="bottom", fontsize=8)

    ax2 = axes[1]
    categories = [
        f"High Conf\n(sim>={HIGH_CONF_THRESHOLD})\nn={high_mask.sum()}",
        f"Low Conf\n(sim<{HIGH_CONF_THRESHOLD})\nn={(low_mask & ~unk_mask).sum()}",
        f"Unknown\nn={unk_mask.sum()}",
    ]
    ax2.bar(categories, [high_acc, low_acc, 0.0],
            color=["#2196F3", "#FF9800", "#9E9E9E"],
            edgecolor="white")
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.set_title("Stage 1 Accuracy by Confidence Level", fontsize=12)
    ax2.set_ylim(0, 1.1)
    for i, v in enumerate([high_acc, low_acc]):
        if v > 0:
            ax2.text(i, v + 0.02, f"{v:.1%}", ha="center",
                     va="bottom", fontsize=11, fontweight="bold")
    plt.suptitle(
        f"SBERT Confidence Analysis  |  "
        f"High={high_acc:.1%}  Low={low_acc:.1%}  "
        f"Unknown={unk_mask.mean():.1%}",
        fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def save_csv(rows, macro_p, macro_r, macro_f1, out_path):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Behavior", "Precision", "Recall", "F1-Score", "Support"])
        for r in rows:
            if r["support"] > 0:
                w.writerow([r["label"], f"{r['precision']:.3f}",
                            f"{r['recall']:.3f}", f"{r['f1']:.3f}",
                            r["support"]])
        w.writerow([])
        w.writerow(["Macro Avg", f"{macro_p:.3f}",
                    f"{macro_r:.3f}", f"{macro_f1:.3f}", ""])
    print(f"  Saved: {out_path}")


def save_summary(rows_b, rows_a, res_b, res_a, n_total,
                 upgrade_counts, out_path):
    f1a = {r["label"]: r["f1"] for r in rows_a}
    lines = [
        "=" * 65,
        "Experiment 1 — Two-stage Behavior Recognition Ablation Study",
        f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 65,
        f"Total valid episodes : {n_total}",
        "",
        "Stage 1  VLM-only (SBERT Prototype Matching)",
        f"  Overall accuracy : {res_b['overall_acc']:.1%}",
        f"  Macro F1         : {res_b['macro_f1']:.3f}",
        f"  Unknown rate     : {res_b['unknown_rate']:.1%}",
        "",
        "Stage 2  VLM + Hierarchical Spatial Reasoning",
        f"  Overall accuracy : {res_a['overall_acc']:.1%}",
        f"  Macro F1         : {res_a['macro_f1']:.3f}",
        f"  Unknown rate     : {res_a['unknown_rate']:.1%}",
        "",
        "Improvement",
        f"  Accuracy   : {res_b['overall_acc']:.1%} -> {res_a['overall_acc']:.1%}"
        f" ({res_a['overall_acc']-res_b['overall_acc']:+.1%})",
        f"  Macro F1   : {res_b['macro_f1']:.3f} -> {res_a['macro_f1']:.3f}"
        f" ({res_a['macro_f1']-res_b['macro_f1']:+.3f})",
        f"  Unk rate   : {res_b['unknown_rate']:.1%} -> {res_a['unknown_rate']:.1%}"
        f" ({res_a['unknown_rate']-res_b['unknown_rate']:+.1%})",
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
            f"Diff={f1af-f1b:+.3f}  (n={r['support']})"
        )
    lines += [
        "",
        "Thesis Statement:",
        "The proposed two-stage architecture achieves an overall accuracy",
        f"of {res_a['overall_acc']:.1%} (Stage 1: {res_b['overall_acc']:.1%})",
        f"and a Macro F1 of {res_a['macro_f1']:.3f} "
        f"(Stage 1: {res_b['macro_f1']:.3f}) across",
        f"{len([r for r in rows_b if r['support']>0])} behavioral categories.",
        f"The spatial reasoning module reduces the Unknown rate from",
        f"{res_b['unknown_rate']:.1%} to {res_a['unknown_rate']:.1%} by applying",
        "three-tier evidence fusion: held-object priority (L2A),",
        "heading alignment with zone matching (L2B), and",
        "zone-level affordance prior (L3).",
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",   default=".", help="Output directory")
    parser.add_argument("--user",  default=None, help="Filter by user_id")
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

    valid = [(t, s1, s2, sim, ur, d)
             for t, s1, s2, sim, ur, d
             in zip(y_true, y_stage1, y_stage2, sbert_sims, upgrade_reasons, docs)
             if t not in ("Unknown", "Standing", "Walking")]

    if not valid:
        print("No valid ground truth labels found.")
        return

    y_true, y_stage1, y_stage2, sbert_sims, upgrade_reasons, docs_v = zip(*valid)
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

    rows_b, mp_b, mr_b, mf1_b, acc_b = compute_metrics(y_true, y_stage1, labels)
    rows_a, mp_a, mr_a, mf1_a, acc_a = compute_metrics(y_true, y_stage2, labels)

    unk_b = sum(p == "Unknown" for p in y_stage1) / len(y_stage1)
    unk_a = sum(p == "Unknown" for p in y_stage2) / len(y_stage2)

    res_b = {
        "overall_acc": acc_b, "macro_f1": mf1_b, "unknown_rate": unk_b,
        "f1":      {r["label"]: r["f1"]     for r in rows_b},
        "support": {r["label"]: r["support"] for r in rows_b},
    }
    res_a = {
        "overall_acc": acc_a, "macro_f1": mf1_a, "unknown_rate": unk_a,
        "f1":      {r["label"]: r["f1"]     for r in rows_a},
        "support": {r["label"]: r["support"] for r in rows_a},
    }

    cm_b = build_confusion_matrix(y_true, y_stage1, labels)
    cm_a = build_confusion_matrix(y_true, y_stage2, labels)

    plot_confusion_matrix(
        cm_b, labels,
        os.path.join(args.out, "exp1_cm_stage1.png"),
        acc_b, mf1_b, "[Stage 1: VLM Only] ")

    plot_confusion_matrix(
        cm_a, labels,
        os.path.join(args.out, "exp1_cm_stage2.png"),
        acc_a, mf1_a, "[Stage 2: VLM + Spatial] ")

    plot_comparison(res_b, res_a, labels,
                    os.path.join(args.out, "exp1_comparison.png"))

    plot_confidence_analysis(
        y_true, y_stage1, sbert_sims,
        os.path.join(args.out, "exp1_confidence.png"))

    plot_upgrade_reasons(
        upgrade_reasons,
        os.path.join(args.out, "exp1_upgrade_reasons.png"))

    save_csv(rows_b, mp_b, mr_b, mf1_b,
             os.path.join(args.out, "exp1_metrics_stage1.csv"))
    save_csv(rows_a, mp_a, mr_a, mf1_a,
             os.path.join(args.out, "exp1_metrics_stage2.csv"))

    save_summary(rows_b, rows_a, res_b, res_a,
                 len(y_true), upgrade_counts,
                 os.path.join(args.out, "exp1_summary.txt"))

    if args.debug:
        print(f"\n  Upgrade reason breakdown: {upgrade_counts}")
        print(f"\n  === First 10 mismatches (Stage 1) ===")
        count = 0
        for d, t, s1, s2, ur in zip(docs_v, y_true, y_stage1, y_stage2, upgrade_reasons):
            if t != s1 and count < 10:
                print(f"    GT={t:15s} S1={s1:15s} S2={s2:15s} "
                      f"reason={ur or 'none'}")
                count += 1


if __name__ == "__main__":
    main()