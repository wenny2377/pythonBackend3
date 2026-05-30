"""
exp2_habit.py — 論文實驗分析腳本 v2.0
支援 17 個行為標籤（含 Sitting / StandUp）
圖表命名：Fig.A ~ Fig.H + 論文章節對應

Usage:
  python3 exp2_habit.py
  python3 exp2_habit.py --out results/
  python3 exp2_habit.py --skip-entropy
  python3 exp2_habit.py --saycan
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

# 17 個行為標籤（與系統一致）
BEHAVIOR_LABELS = [
    "Drinking", "SittingDrink", "Sitting", "Eating", "Cooking", "Opening",
    "Laying", "Watching", "Reading", "Cleaning", "PhoneUse",
    "Typing", "StandUp", "PickingUp", "PuttingDown", "Standing", "Walking",
]

# 可視化使用的子集（排除過渡動作）
BEHAVIOR_VIZ = [
    "Drinking", "SittingDrink", "Sitting", "Eating", "Cooking", "Opening",
    "Laying", "Watching", "Reading", "Cleaning", "PhoneUse", "Typing",
]

# 論文行為分類
BEHAVIOR_HIGH    = {"Cooking", "Opening", "Laying", "Watching", "Typing", "Cleaning"}
BEHAVIOR_MEDIUM  = {"Eating", "Drinking", "Standing", "Walking", "StandUp"}
BEHAVIOR_LOW     = {"SittingDrink", "Sitting", "Reading", "PhoneUse"}

N_BEHAVIORS = len(BEHAVIOR_LABELS)
USERS       = ["User_Mom", "User_Dad"]

FAT_THRESHOLDS  = [2, 3, 5, 8, 10]
CONVERGENCE_ACC = 0.70
CONVERGENCE_DAYS = 3
DEDUP_SIM       = 0.78

NORMALIZE_MAP = {
    "drinking":     "Drinking",
    "sittingdrink": "SittingDrink",
    "sitting":      "Sitting",
    "eating":       "Eating",
    "cooking":      "Cooking",
    "opening":      "Opening",
    "laying":       "Laying",
    "watching":     "Watching",
    "reading":      "Reading",
    "cleaning":     "Cleaning",
    "phoneuse":     "PhoneUse",
    "typing":       "Typing",
    "standup":      "StandUp",
    "pickingup":    "PickingUp",
    "puttingdown":  "PuttingDown",
    "standing":     "Standing",
    "walking":      "Walking",
    "unknown":      "Unknown",
}

COLOR_MAP = {
    "Drinking":    "#2196F3",
    "SittingDrink":"#03A9F4",
    "Sitting":     "#B3E5FC",
    "Eating":      "#FF9800",
    "Cooking":     "#F44336",
    "Opening":     "#9C27B0",
    "Laying":      "#4CAF50",
    "Watching":    "#00BCD4",
    "Reading":     "#8BC34A",
    "Cleaning":    "#795548",
    "PhoneUse":    "#E91E63",
    "Typing":      "#607D8B",
    "StandUp":     "#BDBDBD",
}


def norm(s):
    if not s:
        return "Unknown"
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

    dates     = sorted(daily.keys())
    cum       = defaultdict(int)
    tops      = []

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
        if len(combos) >= 3:
            break
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
# Fig.A  Recognition Confusion Matrix（論文 4.1）
# ══════════════════════════════════════════════════════════════

def run_recognition(db, out):
    print("\n" + "=" * 60)
    print("Fig.A — Recognition Experiment Confusion Matrix（論文 4.1）")
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

    valid_labels = [b for b in BEHAVIOR_VIZ]
    gt_list  = [norm(d["ground_truth"])   for d in docs]
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

    # 整體準確率
    correct = int(np.trace(matrix))
    total   = int(matrix.sum())
    overall_acc = correct / total if total > 0 else 0.0

    # 按特異性分組準確率
    group_acc = {}
    for group_name, group_set in [
        ("High", BEHAVIOR_HIGH),
        ("Medium", BEHAVIOR_MEDIUM),
        ("Low", BEHAVIOR_LOW),
    ]:
        idxs = [i for i, l in enumerate(labels_present) if l in group_set]
        if idxs:
            sub = matrix[np.ix_(idxs, idxs)]
            c   = int(np.trace(sub))
            t   = int(sub.sum())
            group_acc[group_name] = c / t if t > 0 else 0.0

    # 正規化混淆矩陣（行加總=1）
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    matrix_norm = matrix / row_sums

    # 顏色邊框（按特異性分組）
    def _group_color(label):
        if label in BEHAVIOR_HIGH:   return "#F44336"
        if label in BEHAVIOR_MEDIUM: return "#FF9800"
        return "#2196F3"

    fig, ax = plt.subplots(figsize=(max(10, n * 0.9), max(8, n * 0.9)))
    im = ax.imshow(matrix_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Recall Rate")

    for i in range(n):
        for j in range(n):
            v = matrix_norm[i][j]
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

    # 標籤顏色
    for tick, label in zip(ax.get_xticklabels(), labels_present):
        tick.set_color(_group_color(label))
    for tick, label in zip(ax.get_yticklabels(), labels_present):
        tick.set_color(_group_color(label))

    group_str = "  ".join([f"{k}: {v:.1%}" for k, v in group_acc.items()])
    ax.set_title(
        f"Fig.A  Behaviour Recognition Confusion Matrix（論文 4.1）\n"
        f"Overall Acc = {overall_acc:.1%}  ({correct}/{total})  |  {group_str}\n"
        f"[Red=High-specificity  Orange=Medium  Blue=Low-specificity]",
        fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Ground Truth", fontsize=11)

    plt.tight_layout()
    path = os.path.join(out, "FigA_recognition_confusion_matrix.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    # 文字摘要
    lines = [
        "=" * 60,
        "Fig.A  Recognition Summary",
        f"Generated: {datetime.datetime.now():%Y-%m-%d %H:%M}",
        "=" * 60, "",
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
# Fig.B  VLM Confidence vs Actual Accuracy（論文 4.1.1）
# ══════════════════════════════════════════════════════════════

def run_vlm_confidence(db, out):
    print("\n" + "=" * 60)
    print("Fig.B — VLM Confidence vs Actual Accuracy（論文 4.1.1）")
    print("=" * 60)

    docs = list(db.eval_logs.find(
        {"ground_truth": {"$exists": True, "$ne": ""},
         "vlm_confidence": {"$exists": True},
         "spatial_action":  {"$exists": True}},
        {"ground_truth": 1, "spatial_action": 1,
         "vlm_confidence": 1, "upgrade_reason": 1}
    ))
    print(f"  eval_logs with vlm_confidence: {len(docs)}")
    if len(docs) < 10:
        print("  Insufficient data for confidence analysis.")
        return

    confs   = np.array([float(d["vlm_confidence"]) for d in docs])
    correct = np.array([
        1 if norm(d["spatial_action"]) == norm(d["ground_truth"]) else 0
        for d in docs
    ])

    bins = np.linspace(0, 1, 11)
    bin_acc   = []
    bin_count = []
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
        # 對角線（理想 self-calibration）
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
        "Fig.B  VLM Self-Confidence vs Actual Accuracy（論文 4.1.1）\n"
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
# Fig.C  Ablation Study：各推理層的貢獻（論文 4.1.2）
# ══════════════════════════════════════════════════════════════

def run_ablation(db, out):
    print("\n" + "=" * 60)
    print("Fig.C — Ablation Study：各推理層貢獻（論文 4.1.2）")
    print("=" * 60)

    docs = list(db.eval_logs.find(
        {"ground_truth": {"$exists": True, "$ne": ""},
         "upgrade_reason": {"$exists": True}},
        {"ground_truth": 1, "spatial_action": 1,
         "vlm_output": 1, "upgrade_reason": 1}
    ))
    if not docs:
        print("  No eval_logs with upgrade_reason.")
        return

    def _layer(reason):
        r = (reason or "").lower()
        if "vlm_hint:"       in r and "fallback" not in r: return "① VLM_hint"
        if "ray_cast"        in r:                          return "③ Ray-cast"
        if "l3b5_proximity"  in r:                          return "② Proximity"
        if "l3a_saycan"      in r:                          return "④ SayCan"
        if "l3b_heading"     in r:                          return "⑤ Heading"
        if "l3c_zone"        in r:                          return "⑥ Zone"
        if "vlm_hint_fallback" in r:                        return "⑦ VLM_fallback"
        if "minprior"        in r:                          return "⑧ MinPrior"
        return "Other"

    LAYER_ORDER = ["① VLM_hint", "② Proximity", "③ Ray-cast",
                   "④ SayCan", "⑤ Heading", "⑥ Zone",
                   "⑦ VLM_fallback", "⑧ MinPrior", "Other"]

    layer_correct = defaultdict(int)
    layer_total   = defaultdict(int)

    for d in docs:
        layer = _layer(d.get("upgrade_reason", ""))
        hit   = 1 if norm(d["spatial_action"]) == norm(d["ground_truth"]) else 0
        layer_correct[layer] += hit
        layer_total[layer]   += 1

    layers  = [l for l in LAYER_ORDER if layer_total[l] > 0]
    accs    = [layer_correct[l] / layer_total[l] for l in layers]
    counts  = [layer_total[l] for l in layers]

    total_ep = len(docs)
    usage    = [layer_total[l] / total_ep for l in layers]

    x = np.arange(len(layers))
    w = 0.4

    fig, ax1 = plt.subplots(figsize=(max(10, len(layers) * 1.5), 5.5))
    ax2 = ax1.twinx()

    bars1 = ax1.bar(x - w/2, [a * 100 for a in accs], w,
                    color=[COLOR_MAP.get(l.split()[-1], "#607D8B")
                           for l in layers],
                    alpha=0.85, edgecolor="white", label="Accuracy (%)")
    bars2 = ax2.bar(x + w/2, [u * 100 for u in usage], w,
                    color="#BDBDBD", alpha=0.70,
                    edgecolor="white", label="Usage Rate (%)")

    for bar, acc, cnt in zip(bars1, accs, counts):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.8,
                 f"{acc:.0%}\n(n={cnt})",
                 ha="center", fontsize=8, fontweight="bold")
    for bar, u in zip(bars2, usage):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.3,
                 f"{u:.0%}",
                 ha="center", fontsize=8, color="#616161")

    ax1.set_xticks(x)
    ax1.set_xticklabels(layers, fontsize=10)
    ax1.set_ylabel("Layer Accuracy (%)", fontsize=12)
    ax2.set_ylabel("Usage Rate (% of all episodes)", fontsize=11,
                   color="#616161")
    ax1.set_ylim(0, 130)
    ax2.set_ylim(0, 130)
    ax1.set_title(
        "Fig.C  Ablation Study：決策鏈各層準確率與使用率（論文 4.1.2）\n"
        "Accuracy = correct / episodes hitting this layer",
        fontsize=11, fontweight="bold")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
    ax1.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    path = os.path.join(out, "FigC_ablation_layer_contribution.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════
# Fig.D  習慣學習收斂曲線（論文 4.2）
# ══════════════════════════════════════════════════════════════

def run_dynamic(db, out):
    print("\n" + "=" * 60)
    print("Fig.D/E/F — Habit Learning（論文 4.2）")
    print("=" * 60)

    snaps = list(db.habit_snapshots.find({}))
    obs   = list(db.observation_logs.find(
        {}, {"user": 1, "action": 1, "instance": 1,
             "weight": 1, "zone_name": 1}))
    print(f"  habit_snapshots : {len(snaps)}")
    print(f"  observation_logs: {len(obs)}")

    if not snaps and not obs:
        print("  No dynamic data. Run HabitExp first.")
        return

    showcase = _get_showcase_combos(db)
    print(f"  showcase combos : {[s['label'] for s in showcase]}")

    _plot_figD_learning_curve(snaps, out, showcase)
    _plot_figE_zone_discrimination(obs, out, showcase)
    _plot_figF_fat_sensitivity(obs, out)


def _plot_figD_learning_curve(snaps, out, showcase):
    n   = len(showcase)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]
    fig.suptitle(
        "Fig.D  習慣學習收斂曲線（論文 4.2）\n"
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


def _plot_figE_zone_discrimination(obs, out, showcase):
    n   = len(showcase)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5.5))
    if n == 1:
        axes = [axes]
    fig.suptitle(
        "Fig.E  Zone 辨別力（論文 4.2）\n"
        "Zone 1 = system-learned preferred location; ratio > 3× = strong discrimination",
        fontsize=12, fontweight="bold")

    for ax, sc in zip(axes, showcase):
        zones = _zone_weight_ranking(obs, sc["user"], sc["action"])
        total = sum(w for _, w in zones) or 1

        if not zones:
            ax.set_title(f"{sc['label']}\n(no data)")
            ax.text(0.5, 0.5, "No observation data",
                    ha="center", va="center", transform=ax.transAxes)
            continue

        z1_name, w1 = zones[0] if len(zones) > 0 else ("None", 0)
        z2_name, w2 = zones[1] if len(zones) > 1 else ("None", 0)
        w_other     = sum(w for _, w in zones[2:])

        lbls = [
            f"Zone 1\n({z1_name.replace('_Zone','').replace('_',' ')})",
            f"Zone 2\n({z2_name.replace('_Zone','').replace('_',' ')})",
            "Other",
        ]
        vals = [w1, w2, w_other]
        cols = ["#2196F3", "#FF9800", "#BDBDBD"]
        bars = ax.bar(lbls, vals, color=cols, alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.2,
                    f"{v}\n({v / total:.0%})",
                    ha="center", fontsize=10, fontweight="bold")

        ratio = w1 / (w2 + 1e-9)
        ratio_str = f"{min(ratio, 99.9):.1f}×" if ratio < 99.9 else "∞"
        ax.set_title(
            f"{sc['label']}\nZone 1/2 ratio = {ratio_str}",
            fontsize=11, fontweight="bold")
        ax.set_ylabel("Cumulative Weight")
        ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    path = os.path.join(out, "FigE_zone_discrimination.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def _plot_figF_fat_sensitivity(obs, out):
    if not _SBERT_OK:
        print("  SBERT not available, skipping Fig.F")
        return

    print("  Loading SBERT for FAT analysis (CPU)...")
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    fig, axes = plt.subplots(
        1, len(USERS), figsize=(6 * len(USERS), 5.5), sharey=True)
    if len(USERS) == 1:
        axes = [axes]
    fig.suptitle(
        "Fig.F  FAT Threshold Sensitivity（論文 4.2）\n"
        "Precision / Recall / F1 across FAT values; selected = FAT=5",
        fontsize=12, fontweight="bold")

    for ax, user_id in zip(axes, USERS):
        agg = defaultdict(int)
        for d in obs:
            if d.get("user") != user_id:
                continue
            key = (norm(d.get("action", "")), d.get("instance", ""))
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
# Fig.G  習慣收斂曲線（affinity_history）（論文 4.2.1）
# ══════════════════════════════════════════════════════════════

def run_convergence(db, out):
    print("\n" + "=" * 60)
    print("Fig.G — Habit Affinity Convergence（論文 4.2.1）")
    print("=" * 60)

    docs = list(db.affinity_history.find({}))
    print(f"  affinity_history: {len(docs)}")
    if not docs:
        print("  No affinity_history. Run HabitExp first.")
        return

    combo_counts = defaultdict(int)
    for d in docs:
        key = (d.get("user_id", ""), d.get("action", ""))
        combo_counts[key] += 1

    top_combos = sorted(combo_counts.items(), key=lambda x: -x[1])
    SHOW = []
    for (user, action), cnt in top_combos:
        if len(SHOW) >= 2:
            break
        label = f"{user.replace('User_', '')} · {action}"
        SHOW.append({"user": user, "action": action, "label": label})

    if not SHOW:
        print("  No valid combos found.")
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
    colors   = ["#2196F3", "#4CAF50", "#E53935"]

    for sc, c in zip(SHOW, colors):
        by_date = {}
        for d in docs:
            if d.get("user_id") != sc["user"]:
                continue
            if d.get("action") != sc["action"]:
                continue
            date = d.get("date", "")
            aff  = d.get("affinity", 0.0)
            if date:
                by_date[date] = max(by_date.get(date, 0.0), aff)

        if not by_date:
            continue

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
               linestyle="--", alpha=0.5, label="Personalised threshold (0.70)")
    ax.set_xlabel("Day", fontsize=12)
    ax.set_ylabel("Affinity Score", fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(
        "Fig.G  Zone × Behaviour Affinity Convergence（論文 4.2.1）\n"
        "Shaded = 3-day rolling std; dotted = FAT trigger day",
        fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    path = os.path.join(out, "FigG_affinity_convergence.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════
# Fig.H  Behaviour × Zone Heatmap（論文 4.2.2）
# ══════════════════════════════════════════════════════════════

def run_system(db, out):
    print("\n" + "=" * 60)
    print("Fig.H — Behaviour × Zone Heatmap（論文 4.2.2）")
    print("=" * 60)

    obs = list(db.observation_logs.find(
        {}, {"user": 1, "action": 1,
             "zone_name": 1, "weight": 1}))
    print(f"  observation_logs: {len(obs)}")
    if not obs:
        print("  No observation_logs. Run HabitExp first.")
        return

    _plot_figH_behavior_zone_heatmap(obs, out)


def _plot_figH_behavior_zone_heatmap(obs, out):
    grid      = defaultdict(lambda: defaultdict(int))
    zones_all = set()
    for d in obs:
        act  = norm(d.get("action", ""))
        zone = d.get("zone_name", "")
        if act in ("Unknown", "Standing", "Walking", "StandUp",
                   "PickingUp", "PuttingDown", "", "None"):
            continue
        if not zone:
            continue
        grid[act][zone] += d.get("weight", 0)
        zones_all.add(zone)

    if not zones_all:
        print("  No zone_name in observation_logs — skipping Fig.H")
        return

    behaviors = [b for b in BEHAVIOR_VIZ if b in grid]
    zones     = sorted(zones_all)
    if not behaviors or not zones:
        return

    matrix = np.zeros((len(behaviors), len(zones)))
    for i, b in enumerate(behaviors):
        for j, z in enumerate(zones):
            matrix[i, j] = grid[b][z]

    col_sums = matrix.sum(axis=0)
    col_sums[col_sums == 0] = 1
    matrix_norm = matrix / col_sums[np.newaxis, :]

    fig, ax = plt.subplots(
        figsize=(max(10, len(zones) * 1.2), max(6, len(behaviors) * 0.7)))
    im = ax.imshow(matrix_norm, aspect="auto", cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(range(len(zones)))
    ax.set_xticklabels([z.replace("_Zone", "") for z in zones],
                       rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(len(behaviors)))
    ax.set_yticklabels(behaviors, fontsize=10)

    for i in range(len(behaviors)):
        for j in range(len(zones)):
            v = matrix_norm[i, j]
            if v > 0.05:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=8,
                        color="white" if v > 0.55 else "black")

    plt.colorbar(im, ax=ax, label="Normalised Weight (per Zone)")
    ax.set_title(
        "Fig.H  Behaviour × Zone Affinity Heatmap（論文 4.2.2）\n"
        "Normalised — high diagonal = correct spatial association learned",
        fontsize=12, fontweight="bold")
    ax.set_xlabel("Zone", fontsize=11)
    ax.set_ylabel("Behaviour", fontsize=11)
    plt.tight_layout()
    path = os.path.join(out, "FigH_behavior_zone_heatmap.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════
# Fig.I  Spatiotemporal Entropy Heatmap（論文 4.3）
# ══════════════════════════════════════════════════════════════

def run_entropy(db, out):
    print("\n" + "=" * 60)
    print("Fig.I — Spatiotemporal Entropy Heatmap（論文 4.3）")
    print("=" * 60)

    try:
        _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _root not in sys.path:
            sys.path.insert(0, _root)
        from modules.manifold_engine import ManifoldEngine
        me = ManifoldEngine(db)
        print("  ManifoldEngine loaded")
    except Exception as e:
        print(f"  Cannot load ManifoldEngine: {e}")
        return

    zone_doc = db.scene_snapshots.find_one({"label": "sofa"})
    if zone_doc and zone_doc.get("pos"):
        cx = zone_doc["pos"][0] / 10.0
        cz = zone_doc["pos"][1] / 10.0
    else:
        cx, cz = 0.25, -0.12

    for user_id in USERS:
        hours, matrix = me.probe_spatiotemporal(
            user_id, pos=[cx, cz], prev_action="Standing", n_hours=48)

        if matrix.max() < 1e-6:
            print(f"  {user_id}: no model — run manifold_train first")
            continue

        try:
            from scipy.ndimage import gaussian_filter
            matrix_smooth = gaussian_filter(matrix.astype(float), sigma=1.0)
        except ImportError:
            matrix_smooth = matrix.astype(float)

        entropies = []
        for j in range(matrix_smooth.shape[1]):
            p = matrix_smooth[:, j].copy()
            p = p / (p.sum() + 1e-9)
            entropies.append(-float(np.sum(p * np.log2(p + 1e-9))))

        viz_labels = [b for b in BEHAVIOR_VIZ
                      if b in BEHAVIOR_LABELS]
        viz_idx    = [BEHAVIOR_LABELS.index(b) for b in viz_labels]
        matrix_viz = matrix_smooth[viz_idx, :]

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(14, 8),
            gridspec_kw={"height_ratios": [3, 1]},
            sharex=True)

        im = ax1.imshow(
            matrix_viz, aspect="auto", origin="lower",
            cmap="YlOrRd", vmin=0, vmax=matrix_viz.max(),
            interpolation="bicubic",
            extent=[0, 24, -0.5, len(viz_labels) - 0.5])
        plt.colorbar(im, ax=ax1, label="Intent Probability")
        ax1.set_yticks(range(len(viz_labels)))
        ax1.set_yticklabels(viz_labels, fontsize=8)
        ax1.set_ylabel("Behaviour")
        ax1.set_title(
            f"Fig.I  Spatiotemporal Intent Heatmap — {user_id}（論文 4.3）\n"
            f"Fixed pos=Sofa_Zone, prev=Standing, sweep 0–24h",
            fontsize=11, fontweight="bold")

        n_pts   = matrix_smooth.shape[1]
        x_ticks = np.linspace(0, 24, n_pts, endpoint=False)
        ax2.plot(x_ticks, entropies, color="#E53935", linewidth=2)
        ax2.fill_between(x_ticks, entropies, color="#E53935", alpha=0.15)
        max_h = float(np.log2(N_BEHAVIORS))
        ax2.axhline(y=max_h, color="#BDBDBD", linewidth=1, linestyle="--",
                    label=f"Max entropy = {max_h:.2f} bits")
        ax2.set_xlabel("Time of Day (hour)")
        ax2.set_ylabel("Shannon Entropy H (bits)")
        ax2.set_xlim(0, 24)
        ax2.legend(fontsize=8)
        ax2.grid(axis="y", alpha=0.25)
        ax2.set_title("Intent Entropy — low = confident prediction", fontsize=9)

        plt.tight_layout()
        path = os.path.join(out, f"FigI_entropy_heatmap_{user_id}.png")
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════
# Fig.J  User Topology Comparison（論文 4.3.1）
# ══════════════════════════════════════════════════════════════

def run_topology(db, out):
    print("\n" + "=" * 60)
    print("Fig.J — User Topology Comparison（論文 4.3.1）")
    print("=" * 60)

    try:
        _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if _root not in sys.path:
            sys.path.insert(0, _root)
        from modules.manifold_engine import ManifoldEngine
        me = ManifoldEngine(db)
    except Exception as e:
        print(f"  Cannot load ManifoldEngine: {e}")
        return

    zone_pos = defaultdict(list)
    for d in db.observation_logs.find(
            {"zone_name": {"$exists": True, "$ne": ""},
             "pos": {"$exists": True}},
            {"zone_name": 1, "pos": 1}):
        zn  = d.get("zone_name", "")
        pos = d.get("pos")
        if zn and isinstance(pos, list) and len(pos) == 2:
            zone_pos[zn].append(pos)

    if zone_pos:
        zone_centers = {
            zn: [float(np.mean([p[0] for p in positions])) / 10.0,
                 float(np.mean([p[1] for p in positions])) / 10.0]
            for zn, positions in zone_pos.items()
        }
    else:
        zone_centers = {
            "Watching_Zone":    [0.25, -0.15],
            "Typing_Zone":      [-0.85, -0.48],
            "Cooking_Zone":     [0.34,  0.01],
            "Laying_Zone":      [-0.70, 0.30],
        }
        print("  Using default zone centres")

    zone_names_ord = list(zone_centers.keys())
    n_zones = len(zone_names_ord)

    viz_labels = [b for b in BEHAVIOR_VIZ if b in BEHAVIOR_LABELS]
    n_beh      = len(viz_labels)

    fig, axes = plt.subplots(
        1, 2, figsize=(13, max(5, n_zones * 0.8 + 2)))
    fig.suptitle(
        "Fig.J  Behaviour-Zone Topology Comparison（論文 4.3.1）\n"
        "Per-user isolated MLP avoids habit cross-contamination",
        fontsize=12, fontweight="bold")

    for ax, uid, ulabel in zip(
            axes, USERS, ["Mom's MLP", "Dad's MLP"]):
        zn_list, matrix = me.probe_zone_behavior(
            uid, zone_centers, virtual_hour=20.0, prev_action="Standing")

        row_order  = [zn_list.index(z) if z in zn_list else 0
                      for z in zone_names_ord]
        matrix_ord = matrix[row_order, :]

        col_idx     = [BEHAVIOR_LABELS.index(b) if b in BEHAVIOR_LABELS else 0
                       for b in viz_labels]
        matrix_show = matrix_ord[:, col_idx]

        if matrix_show.max() < 1e-6:
            ax.set_title(f"{ulabel}\n(no model — run manifold_train first)")
            continue

        vmax_val = max(float(matrix_show.max()), 0.8)
        im = ax.imshow(matrix_show, aspect="auto",
                       cmap="Blues", vmin=0, vmax=vmax_val)
        plt.colorbar(im, ax=ax, label="Probability")

        for i in range(n_zones):
            for j in range(n_beh):
                v = matrix_show[i, j]
                if v > 0.08:
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=7,
                            color="white" if v > 0.5 else "black")

        ax.set_xticks(range(n_beh))
        ax.set_xticklabels(viz_labels, rotation=40,
                           ha="right", fontsize=8)
        ax.set_yticks(range(n_zones))
        ax.set_yticklabels(
            [z.replace("_Zone", "") for z in zone_names_ord], fontsize=9)
        ax.set_xlabel("Behaviour")
        ax.set_ylabel("Zone")
        ax.set_title(ulabel, fontsize=12, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(out, "FigJ_topology_comparison.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════
# Fig.K  Say × Can Gate Analysis（論文 4.4）
# ══════════════════════════════════════════════════════════════

def run_saycan(db, out):
    print("\n" + "=" * 60)
    print("Fig.K — Say × Can Gate Analysis（論文 4.4）")
    print("=" * 60)

    logs = list(db.saycan_logs.find(
        {}, {"query": 1, "best_action": 1, "best_score": 1,
             "say_scores": 1, "habit_probs": 1,
             "env_scores": 1, "skill_scores": 1,
             "final_scores": 1, "user_id": 1}
    ).sort("timestamp", 1))

    if not logs:
        print("  No saycan_logs found.")
        print("  Generating test queries via POST /interact...")
        _generate_test_saycan_queries()
        logs = list(db.saycan_logs.find(
            {}, {"query": 1, "best_action": 1, "best_score": 1,
                 "say_scores": 1, "habit_probs": 1,
                 "env_scores": 1, "skill_scores": 1,
                 "final_scores": 1, "user_id": 1}
        ).sort("timestamp", 1))
        if not logs:
            print("  Still no logs. POST /interact first.")
            return

    print(f"  saycan_logs: {len(logs)}")
    _plot_figK_saycan_scores(logs, out)
    _plot_figK2_component_breakdown(logs, out)
    _save_saycan_summary(logs, out)


def _generate_test_saycan_queries():
    import requests
    queries = [
        ("I am hungry", "User_Mom"),
        ("I want something to drink", "User_Dad"),
        ("I am tired", "User_Mom"),
        ("I want to watch TV", "User_Dad"),
        ("I feel like reading", "User_Mom"),
        ("bring me water", "User_Dad"),
        ("I need to cook dinner", "User_Mom"),
        ("I want to use my phone", "User_Dad"),
    ]
    for query, user_id in queries:
        try:
            import requests as _r, time
            resp = _r.post(
                "http://localhost:5000/interact",
                json={"query": query, "userID": user_id},
                timeout=30)
            print(f"  [SayCan] '{query}' → {resp.status_code}")
            time.sleep(1)
        except Exception as e:
            print(f"  [SayCan] failed: {e}")


def _plot_figK_saycan_scores(logs, out):
    n_queries = min(len(logs), 6)
    fig, axes = plt.subplots(1, n_queries, figsize=(5 * n_queries, 5.5))
    if n_queries == 1:
        axes = [axes]

    fig.suptitle(
        "Fig.K  Say × Can Gate — Final Fused Scores（論文 4.4）\n"
        "(Say=LLM × Can_habit=MLP × Can_env=DB × Can_skill=SKILL.md)",
        fontsize=12, fontweight="bold")

    colors = ["#2196F3", "#FF9800", "#9C27B0",
              "#E53935", "#4CAF50", "#795548"]

    for ax, log, c in zip(axes, logs[:n_queries], colors):
        final = log.get("final_scores", {})
        if not final:
            ax.set_title("No scores")
            continue

        top5 = sorted(final.items(), key=lambda x: -x[1])[:5]
        lbls = [b for b, _ in top5]
        vals = [v for _, v in top5]

        bars = ax.bar(range(len(lbls)), vals,
                      color=c, alpha=0.80, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002,
                    f"{v:.3f}", ha="center", fontsize=8, fontweight="bold")

        ax.set_xticks(range(len(lbls)))
        ax.set_xticklabels(lbls, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("Say × Can Score")
        ax.set_ylim(0, max(vals) * 1.35 if vals else 0.5)
        query_short = log.get("query", "")[:30]
        best = log.get("best_action", "")
        ax.set_title(f'"{query_short}"\n→ {best}',
                     fontsize=9, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    path = os.path.join(out, "FigK_saycan_scores.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def _plot_figK2_component_breakdown(logs, out):
    if not logs:
        return

    queries      = []
    say_vals     = []
    habit_vals   = []
    env_vals     = []
    skill_vals   = []
    best_actions = []

    for log in logs[:8]:
        best  = log.get("best_action", "")
        say_s = log.get("say_scores",   {}).get(best, 0.0)
        hab_s = log.get("habit_probs",  {}).get(best, 0.0)
        env_s = log.get("env_scores",   {}).get(best, 0.0)
        skl_s = log.get("skill_scores", {}).get(best, 1.0)
        queries.append(log.get("query", "")[:20])
        say_vals.append(say_s)
        habit_vals.append(hab_s)
        env_vals.append(env_s)
        skill_vals.append(skl_s)
        best_actions.append(best)

    x   = np.arange(len(queries))
    w   = 0.6
    fig, ax = plt.subplots(figsize=(max(10, len(queries) * 1.6), 5.5))

    ax.bar(x, say_vals,   w, label="Say (LLM semantic)",
           color="#2196F3", alpha=0.85)
    ax.bar(x, habit_vals, w, bottom=say_vals,
           label="Can_habit (MLP prior)", color="#FF9800", alpha=0.85)
    env_bottom = [s + h for s, h in zip(say_vals, habit_vals)]
    ax.bar(x, env_vals, w, bottom=env_bottom,
           label="Can_env (object feasibility)", color="#4CAF50", alpha=0.85)
    skl_bottom = [e + b for e, b in zip(env_bottom, env_vals)]
    ax.bar(x, skill_vals, w, bottom=skl_bottom,
           label="Can_skill (preference filter)", color="#9C27B0", alpha=0.85)

    for i, (q, b) in enumerate(zip(queries, best_actions)):
        ax.text(i, -0.08, f"→{b}", ha="center", fontsize=7.5,
                color="#1A237E", fontweight="bold",
                transform=ax.get_xaxis_transform())

    ax.set_xticks(x)
    ax.set_xticklabels([f'"{q}"' for q in queries],
                       rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Component Score (raw, before product)")
    ax.set_title(
        "Fig.K2  Say × Can Component Breakdown（論文 4.4）\n"
        "Each bar = raw component scores for the winning action",
        fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    path = os.path.join(out, "FigK2_saycan_breakdown.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def _save_saycan_summary(logs, out):
    action_counts = defaultdict(int)
    for log in logs:
        action_counts[log.get("best_action", "Unknown")] += 1

    lines = [
        "=" * 65,
        "Fig.K  Say × Can Gate Summary（論文 4.4）",
        f"Generated: {datetime.datetime.now():%Y-%m-%d %H:%M}",
        "=" * 65, "",
        f"Total queries resolved: {len(logs)}", "",
        "Action distribution:",
        *[f"  {a:15}: {c} times"
          for a, c in sorted(action_counts.items(), key=lambda x: -x[1])],
    ]
    path = os.path.join(out, "FigK_saycan_summary.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="論文實驗分析腳本 v2.0")
    parser.add_argument("--out",           default="results")
    parser.add_argument("--skip-entropy",  action="store_true",
                        help="Skip Fig.I and Fig.J (requires trained MLP)")
    parser.add_argument("--saycan",        action="store_true",
                        help="Run Fig.K Say×Can analysis")
    parser.add_argument("--only",          default="",
                        help="Run only one figure: A/B/C/D/E/F/G/H/I/J/K")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    db = connect()
    print(f"Connected → {DB_NAME}")
    print(f"Output   → {args.out}/")

    only = args.only.upper()

    if not only or only == "A":
        run_recognition(db, args.out)
    if not only or only == "B":
        run_vlm_confidence(db, args.out)
    if not only or only == "C":
        run_ablation(db, args.out)
    if not only or only in ("D", "E", "F"):
        run_dynamic(db, args.out)
    if not only or only == "G":
        run_convergence(db, args.out)
    if not only or only == "H":
        run_system(db, args.out)

    if not args.skip_entropy:
        if not only or only == "I":
            run_entropy(db, args.out)
        if not only or only == "J":
            run_topology(db, args.out)
    else:
        print("\n[Skipped] Fig.I + Fig.J — run without --skip-entropy after MLP training")

    if args.saycan or only == "K":
        run_saycan(db, args.out)
    else:
        print("\n[Skipped] Fig.K — run with --saycan after calling /interact")

    print(f"\nDone. Check {args.out}/")
    print("\nFigure index:")
    print("  FigA  Recognition Confusion Matrix          （論文 4.1）")
    print("  FigB  VLM Confidence Calibration            （論文 4.1.1）")
    print("  FigC  Ablation Study Layer Contribution     （論文 4.1.2）")
    print("  FigD  Habit Learning Curve                  （論文 4.2）")
    print("  FigE  Zone Discrimination                   （論文 4.2）")
    print("  FigF  FAT Threshold Sensitivity             （論文 4.2）")
    print("  FigG  Affinity Convergence                  （論文 4.2.1）")
    print("  FigH  Behaviour × Zone Heatmap              （論文 4.2.2）")
    print("  FigI  Spatiotemporal Entropy Heatmap        （論文 4.3）")
    print("  FigJ  User Topology Comparison              （論文 4.3.1）")
    print("  FigK  Say × Can Gate Analysis               （論文 4.4）")