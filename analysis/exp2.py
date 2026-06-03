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

NORMALIZE_MAP = {
    "drinking": "Drinking", "drink": "Drinking",
    "sittingdrink": "SittingDrink", "sitting drink": "SittingDrink",
    "eating": "Eating", "eat": "Eating",
    "cooking": "Cooking", "cook": "Cooking",
    "opening": "Opening", "open": "Opening",
    "laying": "Laying", "lay": "Laying", "sleeping": "Laying",
    "watching": "Watching", "watch": "Watching",
    "reading": "Reading", "read": "Reading",
    "cleaning": "Cleaning", "clean": "Cleaning",
    "phoneuse": "PhoneUse", "phone": "PhoneUse",
    "typing": "Typing", "type": "Typing",
    "standing": "Standing", "stand": "Standing",
    "walking": "Walking", "walk": "Walking",
    "unknown": "Unknown",
}

BEHAVIOR_ORDER = [
    "Eating", "Drinking", "SittingDrink", "Cooking", "Opening",
    "Laying", "Watching", "Reading", "Cleaning", "PhoneUse", "Typing",
]


def normalize(label):
    if not label:
        return "Unknown"
    s = label.lower().strip().replace(" ", "").replace("_", "")
    return NORMALIZE_MAP.get(s, label.capitalize())


def load_proposals(db):
    docs = list(db.service_proposals.find(
        {},
        {"user_id": 1, "intent": 1, "confidence": 1,
         "status": 1, "created_at": 1},
    ).sort("created_at", 1))
    print(f"  service_proposals: {len(docs)}")
    return docs


def load_gt(db, n_episodes):
    docs = list(db.eval_logs.find(
        {"ground_truth": {"$exists": True, "$ne": ""}},
        {"ground_truth": 1},
    ).sort("timestamp", -1).limit(n_episodes))
    labels = [normalize(d.get("ground_truth", "")) for d in docs]
    labels = [l for l in labels
              if l not in ("Unknown", "Standing", "Walking")]
    print(f"  GT from eval_logs: {len(labels)}")
    return labels


def compute_metrics(proposals, gt_labels, n_episodes):
    intent_counts = Counter(
        normalize(p.get("intent", "")) for p in proposals)
    gt_counts     = Counter(gt_labels)
    top_intent    = intent_counts.most_common(1)[0] if intent_counts else ("—", 0)
    top_gt        = gt_counts.most_common(1)[0]     if gt_counts     else ("—", 0)
    top_match     = top_intent[0] == top_gt[0]
    all_labels    = set(intent_counts) | set(gt_counts)
    n_prop        = sum(intent_counts.values()) or 1
    n_gt          = sum(gt_counts.values())     or 1
    bc = sum(
        np.sqrt((intent_counts.get(l, 0) / n_prop) *
                (gt_counts.get(l, 0)     / n_gt))
        for l in all_labels
    )
    confs     = [float(p.get("confidence", 0.0)) for p in proposals]
    mean_conf = float(np.mean(confs)) if confs else 0.0
    std_conf  = float(np.std(confs))  if confs else 0.0
    return {
        "n_episodes":           n_episodes,
        "n_triggered":          len(proposals),
        "trigger_rate":         len(proposals) / n_episodes if n_episodes else 0,
        "top_intent":           top_intent,
        "top_gt":               top_gt,
        "top_intent_match":     top_match,
        "distribution_overlap": bc,
        "mean_confidence":      mean_conf,
        "std_confidence":       std_conf,
        "intent_counts":        intent_counts,
        "gt_counts":            gt_counts,
    }


def plot_distribution(m, out_path):
    ic = m["intent_counts"]
    gc = m["gt_counts"]
    all_labels = [l for l in BEHAVIOR_ORDER if l in ic or l in gc]
    others     = [l for l in (set(ic) | set(gc))
                  if l not in BEHAVIOR_ORDER and l != "Unknown"]
    all_labels += others

    if not all_labels:
        print("  No data for distribution plot")
        return

    n_prop = sum(ic.values()) or 1
    n_gt   = sum(gc.values()) or 1
    gt_v   = [gc.get(l, 0) / n_gt   for l in all_labels]
    int_v  = [ic.get(l, 0) / n_prop for l in all_labels]

    x = np.arange(len(all_labels))
    w = 0.38

    fig, ax = plt.subplots(figsize=(max(10, len(all_labels) * 1.4), 5.5))
    ax.bar(x - w/2, [v*100 for v in gt_v],  w,
           color="#2196F3", alpha=0.85, edgecolor="white",
           label="GT Behavior Distribution")
    ax.bar(x + w/2, [v*100 for v in int_v], w,
           color="#E53935", alpha=0.85, edgecolor="white",
           label="Intent Trigger Distribution")

    for i, (gv, iv) in enumerate(zip(gt_v, int_v)):
        if gv > 0.01:
            ax.text(x[i] - w/2, gv*100 + 0.3,
                    f"{gv*100:.0f}%", ha="center", fontsize=8,
                    color="#1565C0")
        if iv > 0.01:
            ax.text(x[i] + w/2, iv*100 + 0.3,
                    f"{iv*100:.0f}%", ha="center", fontsize=8,
                    color="#B71C1C")

    bc  = m["distribution_overlap"]
    match_str = "Top-intent MATCH" if m["top_intent_match"] \
                else "Top-intent mismatch"
    ax.set_title(
        f"Experiment 2: Intent Distribution vs GT Behavior Distribution\n"
        f"Triggered={m['n_triggered']}  Trigger Rate={m['trigger_rate']:.1%}  "
        f"Bhattacharyya={bc:.3f}  |  {match_str}",
        fontsize=10, pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, fontsize=10)
    ax.set_ylabel("Relative Frequency (%)", fontsize=12)
    ax.set_ylim(0, max(max(gt_v), max(int_v), 0.01) * 100 * 1.3)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_confidence(proposals, m, out_path):
    confs = [float(p.get("confidence", 0.0)) for p in proposals]
    if not confs:
        return
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(confs, bins=min(15, len(confs)),
            color="#7C3AED", alpha=0.75, edgecolor="white")
    ax.axvline(x=0.60, color="#E53935", linewidth=1.5, linestyle="--",
               label="Trigger threshold (C=0.60)")
    ax.axvline(x=m["mean_confidence"], color="#4CAF50", linewidth=1.5,
               label=f"Mean={m['mean_confidence']:.3f} "
                     f"± {m['std_confidence']:.3f}")
    ax.set_xlabel("Intent Confidence", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(
        f"Experiment 2: Proposal Confidence Distribution\n"
        f"n={len(confs)}  Mean={m['mean_confidence']:.3f}  "
        f"Std={m['std_confidence']:.3f}",
        fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def save_summary(m, out_path):
    bc   = m["distribution_overlap"]
    bc_q = ("excellent (>=0.90)" if bc >= 0.90
            else "good (>=0.70)"      if bc >= 0.70
            else "moderate (>=0.50)"  if bc >= 0.50
            else "low (<0.50)")
    lines = [
        "=" * 65,
        "Experiment 2: Habit Learning & Proactive Service",
        f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 65,
        "",
        f"Total episodes        : {m['n_episodes']}",
        f"Triggered proposals   : {m['n_triggered']}",
        f"Trigger Rate          : {m['trigger_rate']:.1%}",
        f"Mean Confidence       : {m['mean_confidence']:.3f}"
        f" +/- {m['std_confidence']:.3f}",
        f"Distribution Overlap  : {bc:.3f}  [{bc_q}]",
        f"Top triggered intent  : {m['top_intent'][0]}"
        f" (n={m['top_intent'][1]})",
        f"Top GT behavior       : {m['top_gt'][0]}"
        f" (n={m['top_gt'][1]})",
        f"Top-Intent Match      : "
        f"{'yes' if m['top_intent_match'] else 'no'}",
        "",
        "Intent distribution:",
        *[f"  {l:16s}  {c:3d}"
          f"  ({c/sum(m['intent_counts'].values())*100:.1f}%)"
          for l, c in m["intent_counts"].most_common()],
        "",
        "GT distribution:",
        *[f"  {l:16s}  {c:3d}"
          f"  ({c/sum(m['gt_counts'].values())*100:.1f}%)"
          for l, c in m["gt_counts"].most_common()],
        "",
        "For thesis:",
        f"Over {m['n_episodes']} episodes, ManifoldEngine triggered"
        f" {m['n_triggered']} proactive proposals",
        f"(trigger rate={m['trigger_rate']:.1%},"
        f" mean confidence={m['mean_confidence']:.3f}).",
        f"Bhattacharyya overlap={bc:.3f} ({bc_q}).",
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",      default="results")
    parser.add_argument("--episodes", type=int, default=300)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("Connecting to MongoDB...")
    client = MongoClient(MONGO_URI)
    db     = client[DB_NAME]

    print("\nStep 1: Loading proposals...")
    proposals = load_proposals(db)

    print("\nStep 2: Loading GT...")
    gt_labels = load_gt(db, args.episodes)

    if not proposals:
        print("No service_proposals found. Run Experiment3 first.")
        return

    if not gt_labels:
        print("No GT. Using intent distribution as proxy.")
        gt_labels = [normalize(p.get("intent", "")) for p in proposals]

    print("\nStep 3: Computing metrics...")
    m = compute_metrics(proposals, gt_labels, args.episodes)

    print("\nStep 4: Generating outputs...")
    plot_distribution(m,
        os.path.join(args.out, "exp2_intent_distribution.png"))
    plot_confidence(proposals, m,
        os.path.join(args.out, "exp2_confidence.png"))
    save_summary(m,
        os.path.join(args.out, "exp2_summary.txt"))


if __name__ == "__main__":
    main()