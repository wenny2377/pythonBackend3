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
    "drink":           "Drinking",
    "drinking":        "Drinking",
    "drinkwater":      "Drinking",
    "drinkingwater":   "Drinking",
    "sittingdrink":    "SittingDrink",
    "sitting drink":   "SittingDrink",
    "sittingdrinking": "SittingDrink",
    "eat":             "Eating",
    "eating":          "Eating",
    "cook":            "Cooking",
    "cooking":         "Cooking",
    "open":            "Opening",
    "opening":         "Opening",
    "lay":             "Laying",
    "laying":          "Laying",
    "lying":           "Laying",
    "sleep":           "Laying",
    "sleeping":        "Laying",
    "resting":         "Laying",
    "watch":           "Watching",
    "watching":        "Watching",
    "watchingtv":      "Watching",
    "read":            "Reading",
    "reading":         "Reading",
    "clean":           "Cleaning",
    "cleaning":        "Cleaning",
    "sweeping":        "Cleaning",
    "mopping":         "Cleaning",
    "phoneuse":        "PhoneUse",
    "phone":           "PhoneUse",
    "usingphone":      "PhoneUse",
    "phonecall":       "PhoneUse",
    "scrolling":       "PhoneUse",
    "type":            "Typing",
    "typing":          "Typing",
    "working":         "Typing",
    "usingcomputer":   "Typing",
    "stand":           "Standing",
    "standing":        "Standing",
    "walk":            "Walking",
    "walking":         "Walking",
    "unknown":         "Unknown",
}

KEYWORD_HINTS = [
    ("keyboard",     "Typing"),
    ("laptop",       "Typing"),
    ("computer",     "Typing"),
    ("typing",       "Typing"),
    ("book",         "Reading"),
    ("reading",      "Reading"),
    ("magazine",     "Reading"),
    ("bottle",       "Drinking"),
    ("cup",          "Drinking"),
    ("drinking",     "Drinking"),
    ("juice",        "Drinking"),
    ("mug",          "Drinking"),
    ("phone",        "PhoneUse"),
    ("mobile",       "PhoneUse"),
    ("scrolling",    "PhoneUse"),
    ("smartphone",   "PhoneUse"),
    ("sofa",         "Laying"),
    ("couch",        "Laying"),
    ("lying",        "Laying"),
    ("sleeping",     "Laying"),
    ("television",   "Watching"),
    ("tv",           "Watching"),
    ("screen",       "Watching"),
    ("pan",          "Cooking"),
    ("stove",        "Cooking"),
    ("cooking",      "Cooking"),
    ("spatula",      "Cooking"),
    ("broom",        "Cleaning"),
    ("sweeping",     "Cleaning"),
    ("mopping",      "Cleaning"),
    ("fridge",       "Opening"),
    ("refrigerator", "Opening"),
    ("bowl",         "Eating"),
    ("eating",       "Eating"),
    ("fork",         "Eating"),
    ("spoon",        "Eating"),
    ("utensil",      "Eating"),
]

SKIP_FIRST = {
    "a", "an", "the", "person", "man", "woman",
    "user", "someone", "he", "she", "they", "two",
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
        "ground_truth": 1, "vlm_output": 1,
        "user_id": 1, "room": 1,
        "sbert_sim": 1, "timestamp": 1,
    }))
    print(f"  Loaded {len(docs)} eval_log records")
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
    macro_p  = np.mean([r["precision"] for r in active]) if active else 0.0
    macro_r  = np.mean([r["recall"]    for r in active]) if active else 0.0
    macro_f1 = np.mean([r["f1"]        for r in active]) if active else 0.0
    overall  = sum(t == p for t, p in zip(y_true, y_pred)) / len(y_true) \
               if y_true else 0.0

    return rows, macro_p, macro_r, macro_f1, overall


def build_confusion_matrix(y_true, y_pred, labels):
    idx = {l: i for i, l in enumerate(labels)}
    n   = len(labels)
    cm  = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            cm[idx[t]][idx[p]] += 1
    return cm


def plot_confusion_matrix(cm, labels, out_path, overall_acc, macro_f1):
    active = [l for l in labels
              if cm[labels.index(l)].sum() > 0 or
                 cm[:, labels.index(l)].sum() > 0]
    if not active:
        print("  No data to plot")
        return

    act_idx = [labels.index(l) for l in active]
    cm_sub  = cm[np.ix_(act_idx, act_idx)]

    n = len(active)
    fig, ax = plt.subplots(figsize=(max(8, n * 1.1), max(7, n * 1.0)))
    im = ax.imshow(cm_sub, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(active, rotation=40, ha="right", fontsize=10)
    ax.set_yticklabels(active, fontsize=10)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("Ground Truth Label", fontsize=12)
    ax.set_title(
        f"Experiment 1: VLM Action Recognition — Confusion Matrix\n"
        f"Overall Accuracy = {overall_acc:.1%}   Macro F1 = {macro_f1:.3f}",
        fontsize=12, pad=12,
    )

    thresh = cm_sub.max() / 2.0
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


def plot_confidence_analysis(y_true, y_pred, sbert_sims, out_path):
    if not any(s > 0 for s in sbert_sims):
        print("  No sbert_sim data, skipping confidence plot")
        return

    sims = np.array(sbert_sims)
    correct = np.array([t == p for t, p in zip(y_true, y_pred)])

    bins = np.arange(0, 1.05, 0.1)
    bin_acc  = []
    bin_cnt  = []
    bin_mid  = []

    for i in range(len(bins) - 1):
        mask = (sims >= bins[i]) & (sims < bins[i+1])
        if mask.sum() > 0:
            bin_acc.append(correct[mask].mean())
            bin_cnt.append(mask.sum())
            bin_mid.append((bins[i] + bins[i+1]) / 2)

    high_mask = sims >= HIGH_CONF_THRESHOLD
    low_mask  = sims <  HIGH_CONF_THRESHOLD
    unk_mask  = np.array([p == "Unknown" for p in y_pred])

    high_acc = correct[high_mask].mean() if high_mask.sum() > 0 else 0.0
    low_acc  = correct[low_mask & ~unk_mask].mean() \
               if (low_mask & ~unk_mask).sum() > 0 else 0.0
    unk_rate = unk_mask.mean()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    colors = ["#2196F3" if m >= HIGH_CONF_THRESHOLD else "#FF9800"
              for m in bin_mid]
    bars = ax1.bar(bin_mid, bin_acc, width=0.08, color=colors,
                   edgecolor="white", linewidth=0.5)
    ax1.axvline(HIGH_CONF_THRESHOLD, color="red", linestyle="--",
                linewidth=1.5, label=f"Threshold = {HIGH_CONF_THRESHOLD}")
    ax1.set_xlabel("SBERT Similarity Score", fontsize=12)
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.set_title("Accuracy by SBERT Confidence Score", fontsize=12)
    ax1.set_ylim(0, 1.1)
    ax1.legend(fontsize=10)

    for bar, cnt in zip(bars, bin_cnt):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.02,
                 f"n={cnt}", ha="center", va="bottom", fontsize=8)

    ax2 = axes[1]
    categories = [
        f"High Conf\n(sim≥{HIGH_CONF_THRESHOLD})\nn={high_mask.sum()}",
        f"Low Conf\n(sim<{HIGH_CONF_THRESHOLD})\nn={(low_mask & ~unk_mask).sum()}",
        f"Unknown\n(below thresh)\nn={unk_mask.sum()}",
    ]
    values = [high_acc, low_acc, 0.0]
    bar_colors = ["#2196F3", "#FF9800", "#9E9E9E"]
    ax2.bar(categories, values, color=bar_colors,
            edgecolor="white", linewidth=0.5)
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.set_title("Accuracy: High vs Low Confidence vs Unknown", fontsize=12)
    ax2.set_ylim(0, 1.1)

    for i, v in enumerate(values):
        if v > 0:
            ax2.text(i, v + 0.02, f"{v:.1%}",
                     ha="center", va="bottom", fontsize=11,
                     fontweight="bold")

    plt.suptitle(
        f"SBERT Confidence Analysis  |  "
        f"High conf acc={high_acc:.1%}  "
        f"Low conf acc={low_acc:.1%}  "
        f"Unknown rate={unk_rate:.1%}",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def print_mismatch_samples(y_true, y_pred, docs, n=20):
    print(f"\n  === First {n} mismatches ===")
    count = 0
    for d, t, p in zip(docs, y_true, y_pred):
        if t != p:
            raw_gt  = d.get("ground_truth", "")
            raw_vlm = d.get("vlm_output",   "")
            sim     = d.get("sbert_sim", -1.0)
            print(f"    GT={t:15s} VLM={p:15s} sim={sim:.2f} "
                  f"| raw_gt='{raw_gt}' raw_vlm='{raw_vlm[:60]}'")
            count += 1
            if count >= n:
                break


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
                    f"{macro_p:.3f}", f"{macro_r:.3f}", f"{macro_f1:.3f}", ""])
    print(f"  Saved: {out_path}")


def save_summary(rows, macro_f1, overall_acc, n_total,
                 high_acc, low_acc, unk_rate, out_path):
    active = [r for r in rows if r["support"] > 0]
    best   = max(active, key=lambda x: x["f1"]) if active else None
    worst  = min(active, key=lambda x: x["f1"]) if active else None

    lines = [
        "=" * 60,
        "Experiment 1: VLM Action Recognition Accuracy",
        f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 60,
        f"Total episodes     : {n_total}",
        f"Overall accuracy   : {overall_acc:.1%}",
        f"Macro F1-score     : {macro_f1:.3f}",
        "",
        f"SBERT Confidence Analysis:",
        f"  High conf (sim>={HIGH_CONF_THRESHOLD}) accuracy : {high_acc:.1%}",
        f"  Low conf  (sim< {HIGH_CONF_THRESHOLD}) accuracy : {low_acc:.1%}",
        f"  Unknown rate                      : {unk_rate:.1%}",
        "",
        "Per-class results:",
    ]
    for r in active:
        lines.append(
            f"  {r['label']:15s}  P={r['precision']:.3f}  "
            f"R={r['recall']:.3f}  F1={r['f1']:.3f}  (n={r['support']})"
        )
    if best:
        lines += [
            "",
            f"Best behavior  : {best['label']} (F1={best['f1']:.3f})",
            f"Worst behavior : {worst['label']} (F1={worst['f1']:.3f})",
        ]
    lines += [
        "",
        "For thesis:",
        f"The VLM-based perception pipeline achieves an overall accuracy of",
        f"{overall_acc:.1%} and a macro F1-score of {macro_f1:.3f} across",
        f"{len(active)} behavioral categories in the Unity 3D simulation.",
        f"Among high-confidence predictions (SBERT sim >= {HIGH_CONF_THRESHOLD}),",
        f"accuracy reaches {high_acc:.1%}.",
    ]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",   default=".", help="Output directory")
    parser.add_argument("--user",  default=None, help="Filter by user_id")
    parser.add_argument("--debug", action="store_true",
                        help="Print mismatch samples")
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

    y_true     = [normalize(d.get("ground_truth", "")) for d in docs]
    y_pred     = [normalize(d.get("vlm_output",   "")) for d in docs]
    sbert_sims = [float(d.get("sbert_sim", 0.0))       for d in docs]

    # Filter out Unknown ground truth
    valid = [(t, p, s) for t, p, s in zip(y_true, y_pred, sbert_sims)
             if t != "Unknown"]
    if not valid:
        print("No valid ground truth labels found.")
        return

    y_true, y_pred, sbert_sims = zip(*valid)
    y_true     = list(y_true)
    y_pred     = list(y_pred)
    sbert_sims = list(sbert_sims)

    observed = sorted(set(y_true))
    print(f"  GT labels:       {observed}")
    print(f"  GT distribution: {dict(Counter(y_true).most_common())}")
    print(f"  VLM distribution:{dict(Counter(y_pred).most_common())}")

    unknown_rate = sum(p == "Unknown" for p in y_pred) / len(y_pred)
    print(f"  Unknown rate:    {unknown_rate:.1%}")

    sims = np.array(sbert_sims)
    high_mask = sims >= HIGH_CONF_THRESHOLD
    low_mask  = sims <  HIGH_CONF_THRESHOLD
    unk_mask  = np.array([p == "Unknown" for p in y_pred])
    correct   = np.array([t == p for t, p in zip(y_true, y_pred)])

    high_acc = correct[high_mask].mean() if high_mask.sum() > 0 else 0.0
    low_acc  = correct[low_mask & ~unk_mask].mean() \
               if (low_mask & ~unk_mask).sum() > 0 else 0.0

    labels = [l for l in BEHAVIOR_LABELS if l in observed]
    others = [l for l in observed
              if l not in BEHAVIOR_LABELS and l != "Unknown"]
    if others:
        print(f"  Extra labels: {others}")
    labels += others

    if args.debug:
        print_mismatch_samples(y_true, y_pred, docs)

    rows, mp, mr, mf1, acc = compute_metrics(y_true, y_pred, labels)
    cm = build_confusion_matrix(y_true, y_pred, labels)

    plot_confusion_matrix(
        cm, labels,
        out_path=os.path.join(args.out, "exp1_confusion_matrix.png"),
        overall_acc=acc, macro_f1=mf1,
    )
    plot_confidence_analysis(
        y_true, y_pred, sbert_sims,
        out_path=os.path.join(args.out, "exp1_confidence_analysis.png"),
    )
    save_csv(rows, mp, mr, mf1,
             out_path=os.path.join(args.out, "exp1_metrics.csv"))
    save_summary(rows, mf1, acc, len(docs),
                 high_acc, low_acc, unknown_rate,
                 out_path=os.path.join(args.out, "exp1_summary.txt"))


if __name__ == "__main__":
    main()