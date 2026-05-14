import argparse
import csv
import datetime
import math
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
    "sittingdrinking": "SittingDrink",
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
    "usingphone": "PhoneUse", "phonecall": "PhoneUse", "scrolling": "PhoneUse",
    "type": "Typing", "typing": "Typing",
    "working": "Typing", "usingcomputer": "Typing",
    "stand": "Standing", "standing": "Standing",
    "walk": "Walking", "walking": "Walking",
    "unknown": "Unknown",
}

KEYWORD_HINTS = [
    ("keyboard", "Typing"), ("laptop", "Typing"), ("computer", "Typing"),
    ("book", "Reading"), ("reading", "Reading"), ("magazine", "Reading"),
    ("bottle", "Drinking"), ("cup", "Drinking"), ("drinking", "Drinking"),
    ("juice", "Drinking"), ("mug", "Drinking"),
    ("phone", "PhoneUse"), ("mobile", "PhoneUse"), ("smartphone", "PhoneUse"),
    ("sofa", "Laying"), ("couch", "Laying"), ("lying", "Laying"),
    ("television", "Watching"), ("tv", "Watching"), ("screen", "Watching"),
    ("pan", "Cooking"), ("stove", "Cooking"), ("spatula", "Cooking"),
    ("broom", "Cleaning"), ("sweeping", "Cleaning"), ("mopping", "Cleaning"),
    ("fridge", "Opening"), ("refrigerator", "Opening"),
    ("bowl", "Eating"), ("fork", "Eating"), ("spoon", "Eating"),
]

SKIP_FIRST = {
    "a", "an", "the", "person", "man", "woman",
    "user", "someone", "he", "she", "they", "two",
}

HIGH_CONF_THRESHOLD = 0.42

ROOM_AFFORDANCE = {
    "LivingRoom": {
        "primary":   ["Watching", "Laying", "Reading", "PhoneUse"],
        "secondary": ["Drinking", "SittingDrink", "Cleaning"],
    },
    "Kitchen": {
        "primary":   ["Cooking", "Eating", "Drinking", "SittingDrink"],
        "secondary": ["Opening", "Cleaning"],
    },
    "BedRoom":  {
        "primary":   ["Laying", "Reading", "Typing"],
        "secondary": ["PhoneUse", "Drinking"],
    },
    "DadRoom": {
        "primary":   ["Laying", "Reading", "Typing"],
        "secondary": ["PhoneUse", "Drinking"],
    },
}

ITEM_TO_ACTION = {
    "bowl":      "Eating",   "fork":    "Eating",
    "spoon":     "Eating",   "plate":   "Eating",
    "food":      "Eating",
    "cell phone": "PhoneUse", "phone":  "PhoneUse",
    "book":      "Reading",  "magazine": "Reading",
    "laptop":    "Typing",   "keyboard": "Typing",
    "cup":       "Drinking", "bottle":  "Drinking",
    "mug":       "Drinking", "juice":   "Drinking",
    "broom":     "Cleaning", "mop":     "Cleaning",
    "pan":       "Cooking",  "spatula": "Cooking",
    "remote":    "Watching",
}

TV_KEYWORDS    = {"tv", "television", "screen", "monitor", "电视"}
SOFA_KEYWORDS  = {"sofa", "couch", "沙发"}
DESK_KEYWORDS  = {"desk", "laptop", "keyboard", "computer", "monitor"}
BED_KEYWORDS   = {"bed", "pillow", "mattress"}
STOVE_KEYWORDS = {"stove", "oven", "pan", "spatula", "counter"}


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


def _heading_alignment(user_forward, user_pos, furniture_pos) -> float:
    if not user_forward or not user_pos or not furniture_pos:
        return 0.0
    try:
        if isinstance(furniture_pos, list) and len(furniture_pos) >= 2:
            fx, fz = float(furniture_pos[0]), float(furniture_pos[1])
        else:
            return 0.0
        ux = float(user_pos.get("x", 0))
        uz = float(user_pos.get("z", 0))
        dx, dz = fx - ux, fz - uz
        dist = math.sqrt(dx * dx + dz * dz)
        if dist < 0.01:
            return 0.0
        dx /= dist
        dz /= dist
        fwd_x = float(user_forward.get("x", 0))
        fwd_z = float(user_forward.get("z", 0))
        fwd_len = math.sqrt(fwd_x * fwd_x + fwd_z * fwd_z)
        if fwd_len < 0.01:
            return 0.0
        fwd_x /= fwd_len
        fwd_z /= fwd_len
        dot = fwd_x * dx + fwd_z * dz
        return max(0.0, dot)
    except Exception:
        return 0.0


def _find_furniture_by_semantic(db, keywords, room_name="", max_results=3):
    query = {}
    if room_name:
        query["room"] = {"$regex": room_name, "$options": "i"}
    docs = list(db.scene_snapshots.find(query, {"label": 1, "pos": 1, "room": 1}))
    if not docs:
        docs = list(db.scene_snapshots.find({}, {"label": 1, "pos": 1, "room": 1}))
    results = []
    for doc in docs:
        label = doc.get("label", "").lower()
        if any(kw in label for kw in keywords):
            results.append(doc)
        if len(results) >= max_results:
            break
    return results


def apply_spatial_reasoning(vlm_action, doc, db) -> tuple:
    ground_truth = doc.get("ground_truth", "")
    user_pos     = doc.get("user_pos")
    user_forward = doc.get("user_forward")
    room_name    = doc.get("room_name", "") or doc.get("room", "")
    items        = doc.get("interacting_items", [])

    if isinstance(items, str):
        items = [items] if items else []

    upgraded_action = vlm_action
    upgrade_reason  = ""

    # L1: Held-object evidence (highest priority)
    for item in items:
        item_lower = item.lower().strip()
        for obj_key, action in ITEM_TO_ACTION.items():
            if obj_key in item_lower:
                if upgraded_action != action:
                    upgraded_action = action
                    upgrade_reason  = f"held_object:{item}->{action}"
                return upgraded_action, upgrade_reason

    # L2: Heading Alignment — only when VLM is uncertain
    if vlm_action in ("Unknown", "Laying", "Standing") and user_forward and user_pos:
        best_score  = 0.0
        best_action = vlm_action

        tv_docs = _find_furniture_by_semantic(db, TV_KEYWORDS, room_name)
        for tv_doc in tv_docs:
            score = _heading_alignment(user_forward, user_pos, tv_doc.get("pos"))
            if score > 0.6 and score > best_score:
                best_score  = score
                best_action = "Watching"
                upgrade_reason = f"heading_tv:{score:.2f}"

        desk_docs = _find_furniture_by_semantic(db, DESK_KEYWORDS, room_name)
        for desk_doc in desk_docs:
            score = _heading_alignment(user_forward, user_pos, desk_doc.get("pos"))
            if score > 0.6 and score > best_score:
                best_score  = score
                best_action = "Typing"
                upgrade_reason = f"heading_desk:{score:.2f}"

        if best_action != vlm_action:
            upgraded_action = best_action

    # L3: Room affordance prior — lowest priority, only for Unknown
    if upgraded_action == "Unknown" and room_name:
        room_key = None
        for key in ROOM_AFFORDANCE:
            if key.lower() in room_name.lower():
                room_key = key
                break
        if room_key:
            primary = ROOM_AFFORDANCE[room_key]["primary"]
            if primary:
                upgraded_action = primary[0]
                upgrade_reason  = f"room_prior:{room_key}->{upgraded_action}"

    return upgraded_action, upgrade_reason


def load_eval_logs(db, query=None):
    q    = query or {}
    docs = list(db.eval_logs.find(q, {
        "ground_truth": 1, "vlm_output": 1,
        "user_id": 1, "room": 1, "room_name": 1,
        "sbert_sim": 1, "user_pos": 1, "user_forward": 1,
        "interacting_items": 1, "timestamp": 1,
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
        rows.append({"label": lbl, "precision": prec, "recall": rec,
                     "f1": f1, "support": n})
    active   = [r for r in rows if r["support"] > 0]
    macro_p  = np.mean([r["precision"] for r in active]) if active else 0.0
    macro_r  = np.mean([r["recall"]    for r in active]) if active else 0.0
    macro_f1 = np.mean([r["f1"]        for r in active]) if active else 0.0
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
               if cm[labels.index(l)].sum() > 0 or cm[:, labels.index(l)].sum() > 0]
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
        f"{title_prefix}VLM Action Recognition — Confusion Matrix\n"
        f"Overall Accuracy = {overall_acc:.1%}   Macro F1 = {macro_f1:.3f}",
        fontsize=12, pad=12,
    )
    thresh = cm_sub.max() / 2.0
    for i in range(n):
        for j in range(n):
            v = cm_sub[i, j]
            if v > 0:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=9,
                        color="white" if v > thresh else "black", fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_comparison(results_before, results_after, labels, out_path):
    active = [l for l in labels
              if results_before["support"].get(l, 0) > 0]
    if not active:
        return

    x   = np.arange(len(active))
    w   = 0.35
    f1_before = [results_before["f1"].get(l, 0) for l in active]
    f1_after  = [results_after["f1"].get(l, 0)  for l in active]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    ax1 = axes[0]
    ax1.bar(x - w/2, f1_before, w, label="VLM Only",
            color="#FF9800", alpha=0.85, edgecolor="white")
    ax1.bar(x + w/2, f1_after,  w, label="VLM + Spatial Reasoning",
            color="#2196F3", alpha=0.85, edgecolor="white")
    ax1.set_xticks(x)
    ax1.set_xticklabels(active, rotation=40, ha="right", fontsize=10)
    ax1.set_ylabel("F1 Score", fontsize=12)
    ax1.set_ylim(0, 1.1)
    ax1.set_title("Per-class F1: VLM Only vs VLM + Spatial Reasoning", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(axis="y", alpha=0.3)

    ax2 = axes[1]
    metrics = ["Overall Acc", "Macro F1", "Unknown Rate"]
    before_vals = [
        results_before["overall_acc"],
        results_before["macro_f1"],
        results_before["unknown_rate"],
    ]
    after_vals = [
        results_after["overall_acc"],
        results_after["macro_f1"],
        results_after["unknown_rate"],
    ]
    xi = np.arange(len(metrics))
    ax2.bar(xi - w/2, before_vals, w, label="VLM Only",
            color="#FF9800", alpha=0.85, edgecolor="white")
    ax2.bar(xi + w/2, after_vals,  w, label="VLM + Spatial Reasoning",
            color="#2196F3", alpha=0.85, edgecolor="white")
    ax2.set_xticks(xi)
    ax2.set_xticklabels(metrics, fontsize=11)
    ax2.set_ylim(0, 1.1)
    ax2.set_title("Overall Metrics Comparison", fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(axis="y", alpha=0.3)

    for i, (bv, av) in enumerate(zip(before_vals, after_vals)):
        ax2.text(i - w/2, bv + 0.02, f"{bv:.1%}", ha="center", fontsize=9)
        ax2.text(i + w/2, av + 0.02, f"{av:.1%}", ha="center", fontsize=9)

    plt.suptitle(
        f"Spatial Reasoning Impact  |  "
        f"Acc: {results_before['overall_acc']:.1%} → {results_after['overall_acc']:.1%}  "
        f"Macro F1: {results_before['macro_f1']:.3f} → {results_after['macro_f1']:.3f}",
        fontsize=11, y=1.01,
    )
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
    unk_rate  = unk_mask.mean()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax1    = axes[0]
    colors = ["#2196F3" if m >= HIGH_CONF_THRESHOLD else "#FF9800" for m in bin_mid]
    bars   = ax1.bar(bin_mid, bin_acc, width=0.08, color=colors,
                     edgecolor="white", linewidth=0.5)
    ax1.axvline(HIGH_CONF_THRESHOLD, color="red", linestyle="--",
                linewidth=1.5, label=f"Threshold={HIGH_CONF_THRESHOLD}")
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
        f"Unknown\nn={unk_mask.sum()}",
    ]
    ax2.bar(categories, [high_acc, low_acc, 0.0],
            color=["#2196F3", "#FF9800", "#9E9E9E"],
            edgecolor="white", linewidth=0.5)
    ax2.set_ylabel("Accuracy", fontsize=12)
    ax2.set_title("Accuracy: High vs Low Confidence vs Unknown", fontsize=12)
    ax2.set_ylim(0, 1.1)
    for i, v in enumerate([high_acc, low_acc, 0.0]):
        if v > 0:
            ax2.text(i, v + 0.02, f"{v:.1%}", ha="center",
                     va="bottom", fontsize=11, fontweight="bold")
    plt.suptitle(
        f"SBERT Confidence  |  High={high_acc:.1%}  Low={low_acc:.1%}  Unknown={unk_rate:.1%}",
        fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def print_mismatch_samples(y_true, y_pred, docs, n=20):
    print(f"\n  === First {n} mismatches ===")
    count = 0
    for d, t, p in zip(docs, y_true, y_pred):
        if t != p:
            sim = d.get("sbert_sim", -1.0)
            raw = d.get("vlm_output", "")
            print(f"    GT={t:15s} VLM={p:15s} sim={sim:.2f} | raw='{raw[:50]}'")
            count += 1
            if count >= n:
                break


def save_csv(rows, macro_p, macro_r, macro_f1, out_path):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Behavior", "Precision", "Recall", "F1-Score", "Support"])
        for r in rows:
            if r["support"] > 0:
                w.writerow([r["label"], f"{r['precision']:.3f}",
                            f"{r['recall']:.3f}", f"{r['f1']:.3f}", r["support"]])
        w.writerow([])
        w.writerow(["Macro Avg", f"{macro_p:.3f}", f"{macro_r:.3f}", f"{macro_f1:.3f}", ""])
    print(f"  Saved: {out_path}")


def save_summary(rows_before, rows_after, res_before, res_after, n_total, out_path):
    lines = [
        "=" * 65,
        "Experiment 1: VLM Action Recognition — Spatial Reasoning Ablation",
        f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 65,
        f"Total episodes     : {n_total}",
        "",
        "Stage 1: VLM-only Pipeline (SBERT Prototype Matching)",
        f"  Overall accuracy : {res_before['overall_acc']:.1%}",
        f"  Macro F1-score   : {res_before['macro_f1']:.3f}",
        f"  Unknown rate     : {res_before['unknown_rate']:.1%}",
        "",
        "Stage 2: VLM + Spatial Reasoning",
        f"  Overall accuracy : {res_after['overall_acc']:.1%}",
        f"  Macro F1-score   : {res_after['macro_f1']:.3f}",
        f"  Unknown rate     : {res_after['unknown_rate']:.1%}",
        "",
        f"Improvement:",
        f"  Accuracy:    {res_before['overall_acc']:.1%} -> {res_after['overall_acc']:.1%} "
        f"({res_after['overall_acc'] - res_before['overall_acc']:+.1%})",
        f"  Macro F1:    {res_before['macro_f1']:.3f} -> {res_after['macro_f1']:.3f} "
        f"({res_after['macro_f1'] - res_before['macro_f1']:+.3f})",
        f"  Unknown rate:{res_before['unknown_rate']:.1%} -> {res_after['unknown_rate']:.1%} "
        f"({res_after['unknown_rate'] - res_before['unknown_rate']:+.1%})",
        "",
        "Per-class F1 comparison:",
    ]
    active = [r for r in rows_before if r["support"] > 0]
    f1_after_map = {r["label"]: r["f1"] for r in rows_after}
    for r in active:
        f1b = r["f1"]
        f1a = f1_after_map.get(r["label"], 0.0)
        diff = f1a - f1b
        lines.append(
            f"  {r['label']:15s}  Before={f1b:.3f}  After={f1a:.3f}  "
            f"Diff={diff:+.3f}  (n={r['support']})"
        )
    lines += [
        "",
        "For thesis:",
        "The proposed two-stage architecture — VLM-based perceptual",
        "description followed by hierarchical spatial reasoning —",
        f"improves overall accuracy from {res_before['overall_acc']:.1%} to",
        f"{res_after['overall_acc']:.1%} and Macro F1 from {res_before['macro_f1']:.3f}",
        f"to {res_after['macro_f1']:.3f}. The spatial reasoning module",
        f"reduces the Unknown rate from {res_before['unknown_rate']:.1%} to",
        f"{res_after['unknown_rate']:.1%} by applying three-tier evidence",
        "fusion: held-object priority, heading alignment, and room-level",
        "affordance prior.",
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

    y_true     = [normalize(d.get("ground_truth", "")) for d in docs]
    y_pred_raw = [normalize(d.get("vlm_output",   "")) for d in docs]
    sbert_sims = [float(d.get("sbert_sim", 0.0))       for d in docs]

    valid = [(t, p, s, d)
             for t, p, s, d in zip(y_true, y_pred_raw, sbert_sims, docs)
             if t != "Unknown"]
    if not valid:
        print("No valid ground truth.")
        return

    y_true, y_pred_raw, sbert_sims, docs_valid = zip(*valid)
    y_true      = list(y_true)
    y_pred_raw  = list(y_pred_raw)
    sbert_sims  = list(sbert_sims)
    docs_valid  = list(docs_valid)

    print(f"  GT labels:    {sorted(set(y_true))}")
    print(f"  GT dist:      {dict(Counter(y_true).most_common())}")
    print(f"  VLM dist:     {dict(Counter(y_pred_raw).most_common())}")

    y_pred_spatial = []
    upgrade_log    = []
    for pred, doc in zip(y_pred_raw, docs_valid):
        upgraded, reason = apply_spatial_reasoning(pred, doc, db)
        y_pred_spatial.append(upgraded)
        upgrade_log.append(reason)

    n_upgraded = sum(1 for r in upgrade_log if r)
    print(f"  Spatial upgrades: {n_upgraded}/{len(docs_valid)} "
          f"({n_upgraded/len(docs_valid):.1%})")

    labels = [l for l in BEHAVIOR_LABELS if l in set(y_true)]
    others = [l for l in set(y_true) if l not in BEHAVIOR_LABELS and l != "Unknown"]
    labels += others

    rows_b, mp_b, mr_b, mf1_b, acc_b = compute_metrics(y_true, y_pred_raw,    labels)
    rows_a, mp_a, mr_a, mf1_a, acc_a = compute_metrics(y_true, y_pred_spatial, labels)

    unk_b = sum(p == "Unknown" for p in y_pred_raw)    / len(y_pred_raw)
    unk_a = sum(p == "Unknown" for p in y_pred_spatial) / len(y_pred_spatial)

    res_before = {
        "overall_acc":  acc_b, "macro_f1": mf1_b, "unknown_rate": unk_b,
        "f1":      {r["label"]: r["f1"]     for r in rows_b},
        "support": {r["label"]: r["support"] for r in rows_b},
    }
    res_after = {
        "overall_acc":  acc_a, "macro_f1": mf1_a, "unknown_rate": unk_a,
        "f1":      {r["label"]: r["f1"]     for r in rows_a},
        "support": {r["label"]: r["support"] for r in rows_a},
    }

    cm_b = build_confusion_matrix(y_true, y_pred_raw,    labels)
    cm_a = build_confusion_matrix(y_true, y_pred_spatial, labels)

    plot_confusion_matrix(cm_b, labels,
        os.path.join(args.out, "exp1_cm_vlm_only.png"),
        acc_b, mf1_b, title_prefix="[VLM Only] ")
    plot_confusion_matrix(cm_a, labels,
        os.path.join(args.out, "exp1_cm_spatial.png"),
        acc_a, mf1_a, title_prefix="[VLM + Spatial] ")
    plot_comparison(res_before, res_after, labels,
        os.path.join(args.out, "exp1_comparison.png"))
    plot_confidence_analysis(y_true, y_pred_raw, sbert_sims,
        os.path.join(args.out, "exp1_confidence.png"))

    save_csv(rows_b, mp_b, mr_b, mf1_b,
             os.path.join(args.out, "exp1_metrics_vlm_only.csv"))
    save_csv(rows_a, mp_a, mr_a, mf1_a,
             os.path.join(args.out, "exp1_metrics_spatial.csv"))
    save_summary(rows_b, rows_a, res_before, res_after,
                 len(docs_valid),
                 os.path.join(args.out, "exp1_summary.txt"))

    if args.debug:
        print_mismatch_samples(y_true, y_pred_raw, docs_valid)
        print(f"\n  Upgrade reasons: {dict(Counter(upgrade_log).most_common(10))}")


if __name__ == "__main__":
    main()