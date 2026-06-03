"""
exp2_habit.py — Behaviour Recognition Experiment Analysis v3.1
Figures: Fig.A / Fig.B / Fig.C / Fig.D / Fig.F / Fig.G

Usage:
  python3 exp2_habit.py
  python3 exp2_habit.py --out results/
  python3 exp2_habit.py --only A
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
import matplotlib.font_manager as _fm

def _setup_cjk_font():
    cjk_candidates = [
        "Noto Sans CJK TC", "Noto Sans CJK JP", "Noto Sans CJK SC",
        "WenQuanYi Micro Hei", "Source Han Sans TC", "AR PL UMing TW",
    ]
    available = {f.name for f in _fm.fontManager.ttflist}
    for name in cjk_candidates:
        if name in available:
            plt.rcParams["font.family"]  = name
            plt.rcParams["axes.unicode_minus"] = False
            return name
    plt.rcParams["axes.unicode_minus"] = False
    return None

_cjk = _setup_cjk_font()
from pymongo import MongoClient

try:
    from sentence_transformers import SentenceTransformer
    _SBERT_OK = True
except ImportError:
    _SBERT_OK = False

MONGO_URI = "mongodb://127.0.0.1:27017/"
DB_NAME   = "robot_rag_db"

BEHAVIOR_LABELS = [
    "Drinking", "SittingDrink", "Sitting", "Eating", "Cooking", "Opening",
    "Laying", "Watching", "Reading", "Cleaning", "PhoneUse",
    "Typing", "StandUp", "PickingUp", "PuttingDown", "Standing", "Walking",
]

BEHAVIOR_VIZ = [
    "Drinking", "SittingDrink", "Sitting", "Eating", "Cooking", "Opening",
    "Laying", "Watching", "Reading", "Cleaning", "PhoneUse", "Typing",
]

BEHAVIOR_HIGH   = {"Cooking", "Opening", "Laying", "Watching", "Typing", "Cleaning"}
BEHAVIOR_MEDIUM = {"Eating", "Drinking", "Standing", "Walking", "StandUp"}
BEHAVIOR_LOW    = {"SittingDrink", "Sitting", "Reading", "PhoneUse"}

USERS            = ["User_Mom", "User_Dad"]
FAT_THRESHOLDS   = [2, 3, 5, 8, 10]
CONVERGENCE_ACC  = 0.70
CONVERGENCE_DAYS = 3
DEDUP_SIM        = 0.78

NORMALIZE_MAP = {
    "drinking":     "Drinking",    "sittingdrink": "SittingDrink",
    "sitting":      "Sitting",     "eating":       "Eating",
    "cooking":      "Cooking",     "opening":      "Opening",
    "laying":       "Laying",      "watching":     "Watching",
    "reading":      "Reading",     "cleaning":     "Cleaning",
    "phoneuse":     "PhoneUse",    "typing":       "Typing",
    "standup":      "StandUp",     "pickingup":    "PickingUp",
    "puttingdown":  "PuttingDown", "standing":     "Standing",
    "walking":      "Walking",     "unknown":      "Unknown",
}

COLOR_MAP = {
    "Drinking":    "#2196F3", "SittingDrink": "#03A9F4",
    "Sitting":     "#B3E5FC", "Eating":       "#FF9800",
    "Cooking":     "#F44336", "Opening":      "#9C27B0",
    "Laying":      "#4CAF50", "Watching":     "#00BCD4",
    "Reading":     "#8BC34A", "Cleaning":     "#795548",
    "PhoneUse":    "#E91E63", "Typing":       "#607D8B",
    "StandUp":     "#BDBDBD",
}


def norm(s):
    if not s: return "Unknown"
    key = s.lower().strip().replace(" ", "").replace("_", "")
    return NORMALIZE_MAP.get(key, s.strip())


def connect():
    return MongoClient(MONGO_URI)[DB_NAME]


def _convergence_day(sm):
    consec = 0
    for i, acc in enumerate(sm):
        consec = consec + 1 if acc >= CONVERGENCE_ACC else 0
        if consec >= CONVERGENCE_DAYS:
            return i - CONVERGENCE_DAYS + 2
    return None


def _learning_curve_for(snaps, user, action):
    daily = defaultdict(lambda: defaultdict(int))
    for d in snaps:
        if d.get("user") != user or norm(d.get("action", "")) != action:
            continue
        key = d.get("canonical_key") or d.get("instance", "")
        daily[d.get("date", "")][key] += d.get("daily_count", 0)
    if not daily:
        return [], []
    dates = sorted(daily.keys())
    cum   = defaultdict(int)
    tops  = []
    for date in dates:
        for key, cnt in daily[date].items():
            cum[key] += cnt
        top = sorted(cum.items(), key=lambda x: x[1], reverse=True)
        tops.append(top[0][0] if top else "")
    final_top = tops[-1] if tops else ""
    accs = [1.0 if t == final_top and t != "" else 0.0 for t in tops]
    sm   = [float(np.mean(accs[max(0, i-2):i+1])) for i in range(len(accs))]
    return list(range(1, len(dates) + 1)), sm


def _zone_weight_ranking(obs, user, action):
    agg = defaultdict(int)
    for d in obs:
        if d.get("user") != user or norm(d.get("action", "")) != action:
            continue
        zone = d.get("zone_name", "") or d.get("instance", "Unknown")
        agg[zone] += d.get("weight", 0)
    return sorted(agg.items(), key=lambda x: -x[1])


def _get_showcase_combos(db):
    pipeline = [
        {"$group": {"_id": {"user": "$user", "action": "$action"},
                    "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
    ]
    results = list(db.habit_snapshots.aggregate(pipeline))
    combos  = []
    for r in results:
        if len(combos) >= 3: break
        user   = r["_id"]["user"]
        action = r["_id"]["action"]
        label  = f"{user.replace('User_', '')} · {action}"
        combos.append({"user": user, "action": action, "label": label})
    if not combos:
        combos = [
            {"user": "User_Mom", "action": "Watching", "label": "Mom · Watching"},
            {"user": "User_Dad", "action": "Typing",   "label": "Dad · Typing"},
            {"user": "User_Mom", "action": "Opening",  "label": "Mom · Opening"},
        ]
    return combos


# ══════════════════════════════════════════════════════════════
# Fig.A  Recognition Confusion Matrix (Paper 4.1)
# ══════════════════════════════════════════════════════════════

def run_recognition(db, out):
    print("\n" + "=" * 60)
    print("Fig.A — Behaviour Recognition Confusion Matrix (Paper 4.1)")
    print("=" * 60)

    docs = list(db.eval_logs.find(
        {"ground_truth": {"$exists": True, "$ne": ""},
         "spatial_action": {"$exists": True}},
        {"ground_truth": 1, "spatial_action": 1,
         "vlm_output": 1, "upgrade_reason": 1}
    ))
    print(f"  eval_logs: {len(docs)}")
    if not docs:
        print("  No eval_logs. Run RecognitionExp first.")
        return

    valid_labels = list(BEHAVIOR_VIZ)
    gt_list   = [norm(d["ground_truth"])   for d in docs]
    pred_list = [norm(d["spatial_action"]) for d in docs]

    labels_present = sorted(
        {l for l in gt_list + pred_list if l in valid_labels},
        key=lambda x: valid_labels.index(x) if x in valid_labels else 99
    )
    n = len(labels_present)
    if n == 0:
        print("  No valid labels found.")
        return

    matrix = np.zeros((n, n), dtype=int)
    for gt, pred in zip(gt_list, pred_list):
        if gt in labels_present and pred in labels_present:
            i = labels_present.index(gt)
            j = labels_present.index(pred)
            matrix[i][j] += 1

    correct     = int(np.trace(matrix))
    total       = int(matrix.sum())
    overall_acc = correct / total if total > 0 else 0.0

    group_acc = {}
    for group_name, group_set in [
        ("High",   BEHAVIOR_HIGH),
        ("Medium", BEHAVIOR_MEDIUM),
        ("Low",    BEHAVIOR_LOW),
    ]:
        idxs = [i for i, l in enumerate(labels_present) if l in group_set]
        if idxs:
            sub = matrix[np.ix_(idxs, idxs)]
            c   = int(np.trace(sub))
            t   = int(sub.sum())
            group_acc[group_name] = c / t if t > 0 else 0.0

    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    matrix_norm = matrix / row_sums

    def _group_color(label):
        if label in BEHAVIOR_HIGH:   return "#F44336"
        if label in BEHAVIOR_MEDIUM: return "#FF9800"
        return "#2196F3"

    fig, ax = plt.subplots(figsize=(max(10, n * 0.9), max(8, n * 0.9)))
    im = ax.imshow(matrix_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Recall Rate")

    for i in range(n):
        for j in range(n):
            v   = matrix_norm[i][j]
            raw = matrix[i][j]
            if raw > 0:
                ax.text(j, i, f"{v:.2f}\n({raw})",
                        ha="center", va="center", fontsize=7,
                        color="white" if v > 0.55 else "black",
                        fontweight="bold" if i == j else "normal")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels_present, rotation=40, ha="right", fontsize=9)
    ax.set_yticklabels(labels_present, fontsize=9)
    for tick, label in zip(ax.get_xticklabels(), labels_present):
        tick.set_color(_group_color(label))
    for tick, label in zip(ax.get_yticklabels(), labels_present):
        tick.set_color(_group_color(label))

    group_str = "  ".join([f"{k}: {v:.1%}" for k, v in group_acc.items()])
    ax.set_title(
        f"Fig.A  Behaviour Recognition Confusion Matrix (Paper 4.1)\n"
        f"Overall Acc = {overall_acc:.1%} ({correct}/{total})  |  {group_str}\n"
        f"[Red=High-specificity  Orange=Medium  Blue=Low-specificity]",
        fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Ground Truth", fontsize=11)
    plt.tight_layout()
    path = os.path.join(out, "FigA_recognition_confusion_matrix.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    lines = [
        "=" * 60, "Fig.A  Recognition Summary",
        f"Generated: {datetime.datetime.now():%Y-%m-%d %H:%M}", "=" * 60, "",
        f"Total samples : {total}",
        f"Correct       : {correct}",
        f"Overall Acc   : {overall_acc:.1%}", "",
        "Per-group Accuracy:",
        *[f"  {k:8}: {v:.1%}" for k, v in group_acc.items()], "",
        "Per-class Recall:",
    ]
    for i, label in enumerate(labels_present):
        row_total = int(matrix[i].sum())
        recall    = matrix_norm[i][i]
        lines.append(f"  {label:15}: {recall:.1%} ({matrix[i][i]}/{row_total})")
    txt_path = os.path.join(out, "FigA_recognition_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {txt_path}")


# ══════════════════════════════════════════════════════════════
# Fig.B  VLM Confidence Calibration (Paper 4.1.1)
# ══════════════════════════════════════════════════════════════

def run_vlm_confidence(db, out):
    print("\n" + "=" * 60)
    print("Fig.B — VLM Confidence Calibration (Paper 4.1.1)")
    print("=" * 60)

    docs = list(db.eval_logs.find(
        {"ground_truth":    {"$exists": True, "$ne": ""},
         "vlm_confidence":  {"$exists": True},
         "spatial_action":  {"$exists": True}},
        {"ground_truth": 1, "spatial_action": 1,
         "vlm_confidence": 1, "upgrade_reason": 1}
    ))
    print(f"  eval_logs with vlm_confidence: {len(docs)}")
    if len(docs) < 10:
        print("  Insufficient data.")
        return

    confs   = np.array([float(d["vlm_confidence"]) for d in docs])
    correct = np.array([
        1 if norm(d["spatial_action"]) == norm(d["ground_truth"]) else 0
        for d in docs
    ])

    bins        = np.linspace(0, 1, 11)
    bin_acc     = []
    bin_count   = []
    bin_centers = []
    for i in range(len(bins) - 1):
        mask = (confs >= bins[i]) & (confs < bins[i+1])
        cnt  = mask.sum()
        acc  = correct[mask].mean() if cnt > 0 else np.nan
        bin_acc.append(acc)
        bin_count.append(cnt)
        bin_centers.append((bins[i] + bins[i+1]) / 2)

    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    ax2 = ax1.twinx()

    valid = [(c, a, n) for c, a, n in zip(bin_centers, bin_acc, bin_count)
             if not np.isnan(a)]
    if valid:
        xs, ys, ns = zip(*valid)
        ax1.plot(xs, [y * 100 for y in ys], "o-",
                 color="#2196F3", linewidth=2.5, markersize=8,
                 markerfacecolor="white", markeredgewidth=2.5,
                 label="Actual Accuracy")
        ax1.plot([0, 1], [0, 100], "--", color="#BDBDBD",
                 linewidth=1.5, label="Perfect Calibration")

    ax2.bar(bin_centers, bin_count, width=0.08,
            color="#FF9800", alpha=0.4, label="Sample Count")
    ax1.axvline(x=0.60, color="#E53935", linewidth=1.5,
                linestyle=":", label="VLM gate = 0.60")

    ax1.set_xlabel("VLM Self-Reported Confidence", fontsize=12)
    ax1.set_ylabel("Actual Accuracy (%)", fontsize=12, color="#2196F3")
    ax2.set_ylabel("Sample Count", fontsize=11, color="#FF9800")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 110)
    ax1.set_title(
        "Fig.B  VLM Self-Confidence vs Actual Accuracy (Paper 4.1.1)\n"
        "Gap between curve and diagonal = self-calibration error",
        fontsize=11, fontweight="bold")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
    ax1.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    path = os.path.join(out, "FigB_vlm_confidence_calibration.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════
# Fig.C  Ablation Study (Paper 4.1.2)
#
# Ablation logic:
#   reason field contains the dominant scoring signal, e.g.:
#     "head(21°≈21°)+0.45 prox:table2(0.70)+0.21 zone:..."
#   We classify each episode by which layer's signal was decisive:
#     Skeleton : reason contains "skeleton" OR starts with "head("
#     Geometry : reason contains "prox:" OR "ray:" OR "zone:"
#     Held     : reason contains "held:"
#     Temporal : anything else (inertia, time prior, VLM fallback)
#
#   For each Config we simulate "what would the output be if
#   only layers up to Config N were available":
#     If the decisive layer is within Config N → use spatial_action
#     Else → fall back to vlm_output
# ══════════════════════════════════════════════════════════════

def run_ablation(db, out):
    print("\n" + "=" * 60)
    print("Fig.C — Ablation Study (Paper 4.1.2)")
    print("=" * 60)

    docs = list(db.eval_logs.find(
        {"ground_truth":  {"$exists": True, "$ne": ""},
         "spatial_action": {"$exists": True},
         "vlm_output":    {"$exists": True}},
        {"ground_truth": 1, "spatial_action": 1,
         "vlm_output": 1, "upgrade_reason": 1}
    ))
    if not docs:
        print("  No eval_logs.")
        return

    total = len(docs)
    print(f"  eval_logs: {total}")

    def _dominant_layer(reason):
        """
        Classify the dominant reasoning layer for this episode.

        Priority order (highest to lowest):
          skeleton : explicit skeleton tag or head-pitch profile match
          held     : held_object item-to-action mapping
          nearby   : nearby dynamic object (e.g. keyboard on desk)
          geometry : spatial proximity / ray-cast / zone affinity
          temporal : time prior, inertia, or VLM fallback
        """
        r = (reason or "").lower()
        if r.startswith("skeleton") or r.startswith("head("):
            return "skeleton"
        if "skeleton" in r:
            return "skeleton"
        if "held:" in r:
            return "held"
        if "nearby:" in r:
            return "nearby"
        if "prox:" in r or "ray:" in r or "zone:" in r:
            return "geometry"
        return "temporal"

    # Tag each episode
    for d in docs:
        d["_layer"] = _dominant_layer(d.get("upgrade_reason", ""))

    LAYER_ORDER = ["skeleton", "geometry", "held", "temporal"]

    def _simulate(docs, available_layers):
        """
        Simulate accuracy when only `available_layers` are active.
        If the decisive layer is in available_layers → use spatial_action.
        Otherwise → fall back to vlm_output.
        """
        correct = 0
        for d in docs:
            if d["_layer"] in available_layers:
                pred = norm(d.get("spatial_action", ""))
            else:
                pred = norm(d.get("vlm_output", ""))
            if pred == norm(d["ground_truth"]):
                correct += 1
        return correct

    # Config 1: VLM only (no reasoning layers)
    c1 = sum(1 for d in docs
             if norm(d.get("vlm_output", "")) == norm(d["ground_truth"]))

    # Config 2: VLM + Skeleton (hip + head-pitch profile)
    c2 = _simulate(docs, {"skeleton"})

    # Config 3: VLM + Skeleton + Geometry (proximity + ray-cast + zone)
    c3 = _simulate(docs, {"skeleton", "geometry"})

    # Config 4: VLM + Skeleton + Geometry + Object Context
    #   (held_object item-to-action + nearby dynamic objects)
    c4 = _simulate(docs, {"skeleton", "geometry", "held", "nearby"})

    # Config 5: Full System (all layers including temporal inertia)
    c5 = sum(1 for d in docs
             if norm(d.get("spatial_action", "")) == norm(d["ground_truth"]))

    # Layer distribution stats
    layer_counts = Counter(d["_layer"] for d in docs)
    print(f"  Layer distribution: {dict(layer_counts)}")
    print(f"  Config 1 (VLM only):          {c1}/{total} = {c1/total:.1%}")
    print(f"  Config 2 (+Skeleton):         {c2}/{total} = {c2/total:.1%}")
    print(f"  Config 3 (+Geometry):         {c3}/{total} = {c3/total:.1%}")
    print(f"  Config 4 (+Held Object):      {c4}/{total} = {c4/total:.1%}")
    print(f"  Config 5 (Full System):       {c5}/{total} = {c5/total:.1%}")

    configs = [
        ("VLM Only\n(Baseline)",          c1, "#BDBDBD"),
        ("+ Skeleton\n(hip+head)",        c2, "#2196F3"),
        ("+ Geometry\n(affinity+ray)",    c3, "#4CAF50"),
        ("+ Object Context\n(held+nearby)",  c4, "#FF9800"),
        ("Full System\n(+temporal)",      c5, "#F44336"),
    ]

    accs   = [c / total for _, c, _ in configs]
    labels = [l for l, _, _ in configs]
    colors = [c for _, _, c in configs]
    counts = [c for _, c, _ in configs]
    deltas = [0] + [accs[i] - accs[i-1] for i in range(1, len(accs))]

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
                    ha="center", fontsize=9, color="white", fontweight="bold")

    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_ylim(0, 115)
    ax.set_title(
        "Fig.C  Ablation Study — Incremental Layer Contribution (Paper 4.1.2)\n"
        f"Total episodes = {total} | each bar activates one additional reasoning layer",
        fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)

    for i in range(1, len(accs)):
        x1 = i - 1 + 0.32
        x2 = i     - 0.32
        y  = max(accs[i-1], accs[i]) * 100 + 8
        ax.annotate("",
                    xy=(x2, y), xytext=(x1, y),
                    arrowprops=dict(arrowstyle="->", color="#616161", lw=1.5))

    plt.tight_layout()
    path = os.path.join(out, "FigC_ablation_layer_contribution.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════
# Fig.D  Habit Learning Curve (Paper 4.2)
# ══════════════════════════════════════════════════════════════

def run_dynamic(db, out):
    print("\n" + "=" * 60)
    print("Fig.D — Habit Learning Curve (Paper 4.2)")
    print("=" * 60)

    snaps = list(db.habit_snapshots.find({}))
    print(f"  habit_snapshots: {len(snaps)}")
    if not snaps:
        print("  No data. Run HabitExp first.")
        return

    showcase = _get_showcase_combos(db)
    print(f"  showcase combos: {[s['label'] for s in showcase]}")

    n   = len(showcase)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=True)
    if n == 1: axes = [axes]
    fig.suptitle(
        "Fig.D  Habit Learning Convergence Curve (Paper 4.2)\n"
        "3-day rolling mean accuracy; convergence = 70% for 3 consecutive days",
        fontsize=12, fontweight="bold")

    colors = ["#2196F3", "#4CAF50", "#E53935"]
    for ax, sc, c in zip(axes, showcase, colors):
        days, sm = _learning_curve_for(snaps, sc["user"], sc["action"])
        conv     = _convergence_day(sm)
        if not days:
            ax.set_title(sc["label"] + "\n(no data)")
            ax.text(0.5, 0.5, "No habit_snapshots data",
                    ha="center", va="center", transform=ax.transAxes)
            continue
        ax.plot(days, [s * 100 for s in sm], "o-", color=c,
                linewidth=2.2, markersize=6,
                markerfacecolor="white", markeredgewidth=2,
                label="Accuracy (3-day rolling)")
        ax.axhline(y=CONVERGENCE_ACC * 100, color="#FF9800",
                   linewidth=1.5, linestyle="--",
                   label=f"Threshold {CONVERGENCE_ACC:.0%}")
        if conv:
            ax.axvline(x=conv, color="#9C27B0", linewidth=2,
                       linestyle=":", label=f"Converged Day {conv}")
        ax.set_xlabel("Day", fontsize=11)
        ax.set_ylabel("Accuracy (%)", fontsize=11)
        ax.set_ylim(-5, 110)
        ax.set_title(sc["label"], fontsize=12, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    path = os.path.join(out, "FigD_habit_learning_curve.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════
# Fig.F  FAT Threshold Sensitivity (Paper 4.2)
# ══════════════════════════════════════════════════════════════

def run_fat(db, out):
    print("\n" + "=" * 60)
    print("Fig.F — FAT Threshold Sensitivity (Paper 4.2)")
    print("=" * 60)

    if not _SBERT_OK:
        print("  SBERT not available, skipping Fig.F")
        return

    obs = list(db.observation_logs.find(
        {}, {"user": 1, "action": 1, "instance": 1, "weight": 1, "zone_name": 1}))
    print(f"  observation_logs: {len(obs)}")
    if not obs:
        print("  No data. Run HabitExp first.")
        return

    print("  Loading SBERT (CPU)...")
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    fig, axes = plt.subplots(
        1, len(USERS), figsize=(6 * len(USERS), 5.5), sharey=True)
    if len(USERS) == 1: axes = [axes]
    fig.suptitle(
        "Fig.F  FAT Threshold Sensitivity (Paper 4.2)\n"
        "Precision / Recall / F1 across FAT values; selected threshold = FAT=5",
        fontsize=12, fontweight="bold")

    for ax, user_id in zip(axes, USERS):
        agg = defaultdict(int)
        for d in obs:
            if d.get("user") != user_id: continue
            key = (norm(d.get("action", "")),
                   d.get("zone_name") or d.get("instance", ""))
            agg[key] += d.get("weight", 0)

        grouped = [
            {"action": act, "instance": inst, "weight": w}
            for (act, inst), w in sorted(agg.items(), key=lambda x: -x[1])
        ]
        gt_habits = [g for g in grouped if g["weight"] >= 3]

        results = []
        for thr in FAT_THRESHOLDS:
            habits = [g for g in grouped if g["weight"] >= thr]
            if len(habits) < 2:
                redundancy = 0.0
            else:
                texts = [f"{h['action']} near {h['instance']}" for h in habits]
                vecs  = model.encode(texts, normalize_embeddings=True)
                pairs = redundant = 0
                for i in range(len(vecs)):
                    for j in range(i + 1, len(vecs)):
                        pairs += 1
                        if float(np.dot(vecs[i], vecs[j])) >= DEDUP_SIM:
                            redundant += 1
                redundancy = redundant / pairs if pairs > 0 else 0.0

            precision = 1.0 - redundancy
            gt_keys   = {f"{h['action']}@{h['instance']}" for h in gt_habits}
            lk        = {f"{h['action']}@{h['instance']}" for h in habits}
            recall    = len(gt_keys & lk) / len(gt_keys) if gt_keys else 0.0
            f1 = (2 * precision * recall / (precision + recall)
                  if precision + recall > 0 else 0.0)
            results.append({"thr": thr, "precision": precision,
                             "recall": recall, "f1": f1})

        x_    = list(range(len(FAT_THRESHOLDS)))
        precs = [r["precision"] for r in results]
        recs  = [r["recall"]    for r in results]
        f1s   = [r["f1"]        for r in results]

        if 5 in FAT_THRESHOLDS:
            ax.axvline(x=FAT_THRESHOLDS.index(5), color="#E53935",
                       linewidth=1.8, linestyle="--", alpha=0.7,
                       label="FAT=5 (selected)")

        ax.plot(x_, recs,  "o-", color="#2196F3", linewidth=2.2, markersize=8,
                markerfacecolor="white", markeredgewidth=2, label="Recall")
        ax.plot(x_, precs, "s-", color="#FF9800", linewidth=2.2, markersize=8,
                markerfacecolor="white", markeredgewidth=2, label="Precision")
        ax.plot(x_, f1s,   "^-", color="#4CAF50", linewidth=2.5, markersize=9,
                markerfacecolor="white", markeredgewidth=2.5, label="F1")

        for i, (r, p) in enumerate(zip(recs, precs)):
            ax.text(i, r + 0.02, f"{r:.2f}", ha="center",
                    fontsize=8, color="#1565C0")
            ax.text(i, p - 0.06, f"{p:.2f}", ha="center",
                    fontsize=8, color="#E65100")

        ax.set_xticks(x_)
        ax.set_xticklabels([f"FAT={v}" for v in FAT_THRESHOLDS], fontsize=10)
        ax.set_ylim(0, 1.25)
        ax.set_xlabel("Fast Adaptation Threshold", fontsize=11)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_title(user_id.replace("_", " "), fontsize=12, fontweight="bold")
        ax.legend(loc="lower left", fontsize=9)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    path = os.path.join(out, "FigF_fat_sensitivity.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════
# Fig.G  Habit Affinity Convergence (Paper 4.2.1)
# ══════════════════════════════════════════════════════════════

def run_convergence(db, out):
    print("\n" + "=" * 60)
    print("Fig.G — Habit Affinity Convergence (Paper 4.2.1)")
    print("=" * 60)

    docs = list(db.affinity_history.find({}))
    print(f"  affinity_history: {len(docs)}")
    if not docs:
        print("  No data. Run HabitExp first.")
        return

    combo_counts = defaultdict(int)
    for d in docs:
        key = (d.get("user_id", ""), d.get("action", ""))
        combo_counts[key] += 1

    top_combos = sorted(combo_counts.items(), key=lambda x: -x[1])
    SHOW = []
    for (user, action), cnt in top_combos:
        if len(SHOW) >= 2: break
        SHOW.append({
            "user": user, "action": action,
            "label": f"{user.replace('User_', '')} · {action}"
        })

    if not SHOW:
        print("  No valid combos.")
        return

    FAT_THR  = 5
    L3_PRIOR = 0.10

    fat_days = {}
    for sc in SHOW:
        cum = 0
        for r in db.observation_logs.aggregate([
            {"$match": {"user": sc["user"], "action": sc["action"]}},
            {"$group": {"_id": "$last_date", "daily": {"$sum": "$weight"}}},
            {"$sort": {"_id": 1}},
        ]):
            cum += r["daily"]
            if cum >= FAT_THR:
                fat_days[sc["user"] + "_" + sc["action"]] = r["_id"]
                break

    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors  = ["#2196F3", "#4CAF50", "#E53935"]

    for sc, c in zip(SHOW, colors):
        by_date = {}
        for d in docs:
            if d.get("user_id") != sc["user"]: continue
            if d.get("action")  != sc["action"]: continue
            date = d.get("date", "")
            aff  = d.get("affinity", 0.0)
            if date:
                by_date[date] = max(by_date.get(date, 0.0), aff)
        if not by_date: continue

        dates = sorted(by_date.keys())
        days  = list(range(1, len(dates) + 1))
        affs  = [by_date[d] for d in dates]
        stds  = [float(np.std(affs[max(0, i-2):i+1]))
                 if i > 0 else 0.0 for i in range(len(affs))]

        ax.plot(days, affs, "o-", color=c, linewidth=2.2,
                markersize=6, markerfacecolor="white", markeredgewidth=2,
                label=sc["label"])
        ax.fill_between(days,
                        [a - s for a, s in zip(affs, stds)],
                        [a + s for a, s in zip(affs, stds)],
                        color=c, alpha=0.15)

        key = sc["user"] + "_" + sc["action"]
        if key in fat_days and fat_days[key] in dates:
            fd = dates.index(fat_days[key]) + 1
            ax.axvline(x=fd, color=c, linewidth=1.5,
                       linestyle=":", alpha=0.7)
            ax.annotate(f"FAT triggered\nDay {fd}",
                        xy=(fd, affs[fd - 1]),
                        xytext=(fd + 0.3, affs[fd - 1] + 0.05),
                        fontsize=8, color=c,
                        arrowprops=dict(arrowstyle="->", color=c, lw=1))

    ax.axhline(y=L3_PRIOR, color="#FF9800", linewidth=1.5,
               linestyle="--", label=f"L3 Static Prior ({L3_PRIOR})")
    ax.axhline(y=0.70, color="#4CAF50", linewidth=1,
               linestyle="--", alpha=0.5,
               label="Personalised threshold (0.70)")
    ax.set_xlabel("Day", fontsize=12)
    ax.set_ylabel("Affinity Score", fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(
        "Fig.G  Zone × Behaviour Affinity Convergence (Paper 4.2.1)\n"
        "Shaded = 3-day rolling std; dotted vertical = FAT trigger day",
        fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    path = os.path.join(out, "FigG_affinity_convergence.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Behaviour Recognition Experiment Analysis v3.1")
    parser.add_argument("--out",  default="results")
    parser.add_argument("--only", default="",
                        help="Run only one figure: A/B/C/D/F/G")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    db = connect()
    print(f"Connected -> {DB_NAME}")
    print(f"Output   -> {args.out}/")

    only = args.only.upper()

    if not only or only == "A": run_recognition(db, args.out)
    if not only or only == "B": run_vlm_confidence(db, args.out)
    if not only or only == "C": run_ablation(db, args.out)
    if not only or only == "D": run_dynamic(db, args.out)
    if not only or only == "F": run_fat(db, args.out)
    if not only or only == "G": run_convergence(db, args.out)

    print(f"\nDone. Check {args.out}/")
    print("\nFigure index:")
    print("  FigA  Recognition Confusion Matrix      (Paper 4.1)")
    print("  FigB  VLM Confidence Calibration         (Paper 4.1.1)")
    print("  FigC  Ablation Study Layer Contribution  (Paper 4.1.2)")
    print("  FigD  Habit Learning Curve               (Paper 4.2)")
    print("  FigF  FAT Threshold Sensitivity          (Paper 4.2)")
    print("  FigG  Affinity Convergence               (Paper 4.2.1)")