"""
thesis_exp1_recognition.py
Experiment 1: VLM Behaviour Recognition Ablation Study
Reads from MongoDB (eval_logs). No Flask needed.
Run after Unity RecognitionExp is complete.

Output:
  results/figA_confusion_matrix.png
  results/figA2_perclass_f1.png
  results/figB_overall_metrics.png  (formerly figB_sbert_confidence)
  results/figC_spatial_contributions.png
  results/exp1_summary.txt
"""

"""
thesis_analyze.py
Main offline analysis script for thesis validation.
Does NOT require Flask — reads directly from MongoDB.

Usage:
  python3 thesis_analyze.py --static    # Chapter 4.1: Recognition
  python3 thesis_analyze.py --dynamic   # Chapter 4.2: Habit Learning
  python3 thesis_analyze.py --system    # Chapter 4.3: System Integration
  python3 thesis_analyze.py --all       # All of the above
  python3 thesis_analyze.py --out results/  # Custom output directory
"""

import argparse
import datetime
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pymongo import MongoClient

try:
    from sentence_transformers import SentenceTransformer
    _SBERT_OK = True
except ImportError:
    _SBERT_OK = False

MONGO_URI = "mongodb://127.0.0.1:27017/"
DB_NAME   = "robot_rag_db"

BEHAVIOR_ORDER = [
    "Eating", "Drinking", "SittingDrink", "Cooking", "Opening",
    "Laying", "Watching", "Reading", "Cleaning", "PhoneUse", "Typing",
]

USERS = ["User_Mom", "User_Dad"]

FAT_THRESHOLDS    = [2, 3, 5, 8, 10]
CONVERGENCE_ACC   = 0.70
CONVERGENCE_DAYS  = 3
DEDUP_SIM         = 0.78

SHOWCASE = [
    {"user": "User_Mom", "action": "Watching",
     "spot_a": "sofa",        "spot_b": "sofa side",  "label": "Mom · Watching"},
    {"user": "User_Dad", "action": "Typing",
     "spot_a": "desk",        "spot_b": "chair",       "label": "Dad · Typing"},
    {"user": "User_Mom", "action": "Opening",
     "spot_a": "refrigerator","spot_b": "sink",        "label": "Mom · Opening"},
]

NORMALIZE_MAP = {
    "drinking":"Drinking","drink":"Drinking","sittingdrink":"SittingDrink",
    "eating":"Eating","eat":"Eating","cooking":"Cooking","cook":"Cooking",
    "opening":"Opening","open":"Opening","laying":"Laying","lay":"Laying",
    "watching":"Watching","watch":"Watching","reading":"Reading","read":"Reading",
    "cleaning":"Cleaning","clean":"Cleaning","phoneuse":"PhoneUse","phone":"PhoneUse",
    "typing":"Typing","type":"Typing","unknown":"Unknown","standing":"Standing",
    "walking":"Walking",
}

COLORS = {
    "Correct":  "#4CAF50",
    "Wrong":    "#E53935",
    "Unknown":  "#B0BEC5",
    "Stage1":   "#FF9800",
    "Stage2":   "#2196F3",
    "L2A":      "#4CAF50",
    "L2B":      "#2196F3",
    "L3":       "#9C27B0",
}

BEHAVIOR_COLORS = {
    "Eating":"#EC4899","Drinking":"#059669","SittingDrink":"#34D399",
    "Cooking":"#F59E0B","Opening":"#84CC16","Laying":"#6366F1",
    "Watching":"#DC2626","Reading":"#2563EB","Cleaning":"#7C3AED",
    "PhoneUse":"#14B8A6","Typing":"#9333EA","Unknown":"#D1D5DB",
}


def norm(s):
    if not s: return "Unknown"
    key = s.lower().strip().replace(" ","").replace("_","")
    return NORMALIZE_MAP.get(key, s.capitalize())


def connect():
    client = MongoClient(MONGO_URI)
    return client[DB_NAME]


# ═══════════════════════════════════════════════════════════════════════
# CHAPTER 4.1  --static   行為辨識系統評估
# ═══════════════════════════════════════════════════════════════════════

def run_static(db, out):
    print("\n" + "="*60)
    print("Chapter 4.1 — Behaviour Recognition (--static)")
    print("="*60)

    docs = list(db.eval_logs.find(
        {"ground_truth": {"$exists": True, "$ne": ""}},
        {"ground_truth":1,"vlm_output":1,"spatial_action":1,
         "upgrade_reason":1,"sbert_sim":1,"zone_label":1}
    ))
    if not docs:
        print("  No eval_logs found. Run Experiment 1 first.")
        return
    print(f"  eval_logs: {len(docs)}")

    _plot_confusion_matrix(docs, out)
    _plot_perclass_f1(docs, out)
    _plot_recognition_ablation(docs, out)
    _plot_sbert_confidence(docs, out)
    _plot_spatial_contributions(docs, out)
    _save_static_summary(docs, out)


def _classify(gt, pred):
    if pred in ("Unknown","","None",None): return "Unknown"
    return "Correct" if norm(gt) == norm(pred) else "Wrong"



def _plot_confusion_matrix(docs, out):
    # Fixed 11 behaviours — always show all, even if count=0
    labels     = list(BEHAVIOR_ORDER)
    labels_ext = labels + ["Unknown"]
    n          = len(labels_ext)   # 12 columns

    def make_matrix(get_pred):
        m = np.zeros((len(labels), n), dtype=int)
        for d in docs:
            gt   = norm(d.get("ground_truth",""))
            pred = norm(get_pred(d))
            if gt not in labels: continue
            r    = labels.index(gt)
            c    = labels_ext.index(pred) if pred in labels_ext else n-1
            m[r, c] += 1
        return m

    m1 = make_matrix(lambda d: d.get("vlm_output",""))
    m2 = make_matrix(lambda d: d.get("spatial_action","") or
                               d.get("vlm_output",""))

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Figure A: Confusion Matrix — Stage 1 vs Stage 2",
                 fontsize=13, fontweight="bold")

    for ax, mat, title in zip(
            axes, [m1, m2],
            ["Stage 1: VLM Only", "Stage 2: VLM + Spatial Reasoning"]):
        im = ax.imshow(mat, cmap="Blues", aspect="auto")
        plt.colorbar(im, ax=ax, shrink=0.8)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if v > 0:
                    ax.text(j, i, str(v), ha="center", va="center",
                            fontsize=7.5,
                            color="white" if v > mat.max()*0.6 else "black")
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels_ext, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("Ground Truth Label", fontsize=11)
        acc = np.diag(mat[:, :len(labels)]).sum() / (mat.sum() or 1) * 100
        unk = mat[:, -1].sum() / (mat.sum() or 1) * 100
        ax.set_title(f"{title}\nAcc={acc:.1f}%  Unknown={unk:.1f}%",
                     fontsize=11)

    plt.tight_layout()
    path = os.path.join(out, "figA_confusion_matrix.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def _plot_perclass_f1(docs, out):
    # Always show all 11 behaviours regardless of whether they appear in data
    behaviors = list(BEHAVIOR_ORDER)

    def f1_per_class(get_pred):
        scores = {}
        for b in behaviors:
            tp = sum(1 for d in docs
                     if norm(d.get("ground_truth",""))==b
                     and norm(get_pred(d))==b)
            fp = sum(1 for d in docs
                     if norm(d.get("ground_truth",""))!=b
                     and norm(get_pred(d))==b)
            fn = sum(1 for d in docs
                     if norm(d.get("ground_truth",""))==b
                     and norm(get_pred(d))!=b)
            p  = tp/(tp+fp) if tp+fp>0 else 0.0
            r  = tp/(tp+fn) if tp+fn>0 else 0.0
            scores[b] = 2*p*r/(p+r) if p+r>0 else 0.0
        return scores

    s1 = f1_per_class(lambda d: d.get("vlm_output",""))
    s2 = f1_per_class(lambda d: d.get("spatial_action","") or
                                d.get("vlm_output",""))

    x  = np.arange(len(behaviors))
    w  = 0.38
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.bar(x-w/2, [s1[b] for b in behaviors], w,
           color=COLORS["Stage1"], alpha=0.85, edgecolor="white",
           label="Stage 1 (VLM Only)")
    ax.bar(x+w/2, [s2[b] for b in behaviors], w,
           color=COLORS["Stage2"], alpha=0.85, edgecolor="white",
           label="Stage 2 (VLM + Spatial)")

    for i, b in enumerate(behaviors):
        v1, v2 = s1[b], s2[b]
        if v1 > 0.02:
            ax.text(x[i]-w/2, v1+0.01, f"{v1:.2f}",
                    ha="center", fontsize=7.5, color="#E65100")
        if v2 > 0.02:
            ax.text(x[i]+w/2, v2+0.01, f"{v2:.2f}",
                    ha="center", fontsize=7.5, color="#1565C0")

    ax.set_xticks(x)
    ax.set_xticklabels(behaviors, rotation=25, ha="right", fontsize=10)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_ylim(0, 1.15)
    ax.set_title(
        "Figure A2: Per-class F1 Score — Stage 1 vs Stage 2\n"
        "Expected: Typing / Watching / Opening improve most "
        "(high Zone Affinity)",
        fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    path = os.path.join(out, "figA2_perclass_f1.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def _plot_recognition_ablation(docs, out):
    behaviors = [b for b in BEHAVIOR_ORDER
                 if any(norm(d["ground_truth"])==b for d in docs)]
    if not behaviors: behaviors = BEHAVIOR_ORDER[:8]

    stage1 = defaultdict(lambda: Counter())
    stage2 = defaultdict(lambda: Counter())
    for d in docs:
        b  = norm(d.get("ground_truth",""))
        s1 = _classify(b, d.get("vlm_output",""))
        s2 = _classify(b, d.get("spatial_action","") or d.get("vlm_output",""))
        stage1[b][s1] += 1
        stage2[b][s2] += 1

    x  = np.arange(len(behaviors))
    w  = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
    fig.suptitle(
        "Figure A: Recognition Ablation — Stage 1 (VLM) vs Stage 2 (VLM + Spatial)\n"
        f"n={len(docs)} episodes",
        fontsize=13, fontweight="bold")

    for ax, stage, title in zip(
            axes, [stage1, stage2],
            ["Stage 1: VLM Only", "Stage 2: VLM + Spatial Reasoning"]):
        totals = [sum(stage[b].values()) or 1 for b in behaviors]
        c_vals = [stage[b]["Correct"] / t * 100 for b, t in zip(behaviors, totals)]
        w_vals = [stage[b]["Wrong"]   / t * 100 for b, t in zip(behaviors, totals)]
        u_vals = [stage[b]["Unknown"] / t * 100 for b, t in zip(behaviors, totals)]

        bars_u = ax.bar(x, u_vals, label="Unknown", color=COLORS["Unknown"], alpha=0.85)
        bars_w = ax.bar(x, w_vals, bottom=u_vals, label="Wrong",
                        color=COLORS["Wrong"], alpha=0.85)
        bars_c = ax.bar(x, c_vals,
                        bottom=[u+w for u,w in zip(u_vals, w_vals)],
                        label="Correct", color=COLORS["Correct"], alpha=0.85)

        for bar, uv in zip(bars_u, u_vals):
            if uv > 5:
                ax.text(bar.get_x()+bar.get_width()/2, uv/2,
                        f"{uv:.0f}%", ha="center", va="center",
                        fontsize=7.5, color="white", fontweight="bold")

        acc = np.mean([stage[b]["Correct"]/sum(stage[b].values() or [1])*100
                       for b in behaviors if sum(stage[b].values()) > 0])
        unk = np.mean([stage[b]["Unknown"]/sum(stage[b].values() or [1])*100
                       for b in behaviors if sum(stage[b].values()) > 0])

        ax.set_title(f"{title}\nMean Acc={acc:.1f}%  Unknown={unk:.1f}%",
                     fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(behaviors, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("Proportion of Episodes (%)")
        ax.set_ylim(0, 115)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    path = os.path.join(out, "figA_recognition_ablation.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def _plot_sbert_confidence(docs, out):
    sims = [d.get("sbert_sim", 0.0) for d in docs if d.get("sbert_sim") is not None]
    if not sims:
        print("  No sbert_sim data, skipping Figure B")
        return

    bins     = np.arange(0.2, 0.75, 0.05)
    bin_data = defaultdict(lambda: {"correct":0,"total":0})
    for d in docs:
        sim = d.get("sbert_sim", 0.0) or 0.0
        gt  = norm(d.get("ground_truth",""))
        vlm = norm(d.get("vlm_output",""))
        b   = int(sim / 0.05) * 0.05
        bin_data[round(b,2)]["total"] += 1
        if vlm == gt:
            bin_data[round(b,2)]["correct"] += 1

    b_keys = sorted(bin_data.keys())
    accs   = [bin_data[k]["correct"]/(bin_data[k]["total"] or 1)*100 for k in b_keys]
    counts = [bin_data[k]["total"] for k in b_keys]

    unk_rate = sum(1 for d in docs if norm(d.get("vlm_output",""))=="Unknown") / len(docs) * 100
    threshold= 0.42

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Figure B: SBERT Confidence Analysis", fontsize=13, fontweight="bold")

    ax1.bar(b_keys, accs, width=0.04, color=COLORS["Stage2"],
            alpha=0.8, edgecolor="white")
    ax1.axvline(x=threshold, color="#E53935", linewidth=2,
                linestyle="--", label=f"Threshold = {threshold}")
    for k, a, n_ in zip(b_keys, accs, counts):
        if n_ > 0:
            ax1.text(k, a+1.5, f"n={n_}", ha="center", fontsize=7.5)
    ax1.set_xlabel("SBERT Similarity Score")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Stage 1: Accuracy by SBERT Confidence")
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.25)

    high = sum(1 for d in docs
               if (d.get("sbert_sim") or 0) >= threshold
               and norm(d.get("vlm_output","")) != "Unknown")
    low  = sum(1 for d in docs
               if (d.get("sbert_sim") or 0) <  threshold
               and norm(d.get("vlm_output","")) != "Unknown")
    unk  = sum(1 for d in docs if norm(d.get("vlm_output",""))=="Unknown")
    acc_high = sum(1 for d in docs
                   if (d.get("sbert_sim") or 0) >= threshold
                   and norm(d.get("vlm_output","")) != "Unknown"
                   and norm(d.get("vlm_output","")) == norm(d.get("ground_truth","")))
    acc_h_pct= acc_high / (high or 1) * 100

    ax2.bar(["High Conf\n(sim≥0.42)", "Low Conf\n(sim<0.42)", "Unknown"],
            [acc_h_pct, 0, 0],
            color=[COLORS["Correct"], COLORS["Wrong"], COLORS["Unknown"]],
            alpha=0.85, edgecolor="white")
    for i, (lbl, v) in enumerate(zip(
            ["High Conf", "Low Conf", "Unknown"],
            [f"{acc_h_pct:.1f}%", "0%", f"{unk_rate:.1f}%"])):
        ax2.text(i, max(acc_h_pct if i==0 else 2, 2)+1, v,
                 ha="center", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title(f"Stage 1: Accuracy by Confidence Level\n"
                  f"High Conf Acc={acc_h_pct:.1f}%  Unknown Rate={unk_rate:.1f}%")
    ax2.set_ylim(0, 110)
    ax2.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    path = os.path.join(out, "figB_sbert_confidence.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def _plot_spatial_contributions(docs, out):
    upgraded = [d for d in docs if d.get("upgrade_reason","")]
    if not upgraded:
        print("  No spatial upgrades found, skipping Figure C")
        return

    l2a = sum(1 for d in upgraded if "L2A" in d.get("upgrade_reason",""))
    l2b = sum(1 for d in upgraded if "L2B" in d.get("upgrade_reason",""))
    l3  = sum(1 for d in upgraded if "L3"  in d.get("upgrade_reason",""))
    total_unk = sum(1 for d in docs if norm(d.get("vlm_output",""))=="Unknown")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Figure C: Spatial Reasoning Layer Contributions\n"
                 "(Cases upgraded from Unknown / Low-confidence)",
                 fontsize=13, fontweight="bold")

    cats   = ["L2A\nHeld-object", "L2B\nHeading Align", "L3\nZone Match"]
    vals   = [l2a, l2b, l3]
    colors = [COLORS["L2A"], COLORS["L2B"], COLORS["L3"]]
    bars   = ax1.bar(cats, vals, color=colors, alpha=0.85, edgecolor="white", width=0.5)
    for bar, v in zip(bars, vals):
        pct = v / (len(upgraded) or 1) * 100
        ax1.text(bar.get_x()+bar.get_width()/2,
                 bar.get_height()+0.2,
                 f"{v}\n({pct:.0f}%)", ha="center",
                 fontsize=11, fontweight="bold")
    ax1.set_ylabel("Number of Cases")
    ax1.set_title(f"Upgrade Mechanism Distribution\n"
                  f"Total upgraded: {len(upgraded)} / {total_unk} Unknown")
    ax1.grid(axis="y", alpha=0.25)

    s1_acc = sum(1 for d in docs
                 if norm(d.get("vlm_output","")) == norm(d.get("ground_truth",""))) / len(docs) * 100
    s2_acc = sum(1 for d in docs
                 if norm(d.get("spatial_action","") or d.get("vlm_output","")) ==
                 norm(d.get("ground_truth",""))) / len(docs) * 100
    s1_unk = sum(1 for d in docs if norm(d.get("vlm_output",""))=="Unknown") / len(docs) * 100
    s2_unk = sum(1 for d in docs
                 if norm(d.get("spatial_action","") or d.get("vlm_output",""))=="Unknown") / len(docs) * 100

    x_  = np.arange(2)
    w_  = 0.35
    ax2.bar(x_-w_/2, [s1_acc, s1_unk], w_,
            label="Stage 1 (VLM only)", color=COLORS["Stage1"], alpha=0.85)
    ax2.bar(x_+w_/2, [s2_acc, s2_unk], w_,
            label="Stage 2 (+ Spatial)", color=COLORS["Stage2"], alpha=0.85)
    for i, (v1, v2) in enumerate([(s1_acc, s2_acc), (s1_unk, s2_unk)]):
        ax2.text(x_[i]-w_/2, v1+0.5, f"{v1:.1f}%", ha="center", fontsize=9)
        ax2.text(x_[i]+w_/2, v2+0.5, f"{v2:.1f}%", ha="center", fontsize=9)
    ax2.set_xticks(x_)
    ax2.set_xticklabels(["Overall Accuracy", "Unknown Rate"])
    ax2.set_ylabel("Rate (%)")
    ax2.set_title("Stage 1 vs Stage 2: Overall Metrics")
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    path = os.path.join(out, "figC_spatial_contributions.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def _save_static_summary(docs, out):
    s1_acc = sum(1 for d in docs
                 if norm(d.get("vlm_output",""))==norm(d.get("ground_truth","")))
    s2_acc = sum(1 for d in docs
                 if norm(d.get("spatial_action","") or d.get("vlm_output",""))==
                 norm(d.get("ground_truth","")))
    upgraded = [d for d in docs if d.get("upgrade_reason","")]
    lines = [
        "="*65,
        "Chapter 4.1: Behaviour Recognition Summary",
        f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "="*65,"",
        f"Total episodes   : {len(docs)}",
        f"Stage 1 Accuracy : {s1_acc/len(docs)*100:.1f}%",
        f"Stage 2 Accuracy : {s2_acc/len(docs)*100:.1f}%",
        f"Improvement      : +{(s2_acc-s1_acc)/len(docs)*100:.1f}%",
        f"Total upgraded   : {len(upgraded)}",
        f"  L2A: {sum(1 for d in upgraded if 'L2A' in d.get('upgrade_reason',''))}",
        f"  L2B: {sum(1 for d in upgraded if 'L2B' in d.get('upgrade_reason',''))}",
        f"  L3 : {sum(1 for d in upgraded if 'L3'  in d.get('upgrade_reason',''))}",
    ]
    path = os.path.join(out, "static_summary.txt")
    with open(path,"w",encoding="utf-8") as f: f.write("\n".join(lines))
    print(f"  Saved: {path}")




if __name__ == "__main__":
    import argparse as _ap
    parser = _ap.ArgumentParser(
        description="Experiment 1: Recognition Ablation (no Flask needed)")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    db = connect()
    print(f"Connected → {DB_NAME}")
    run_static(db, args.out)
    print("\nDone. Check", args.out)