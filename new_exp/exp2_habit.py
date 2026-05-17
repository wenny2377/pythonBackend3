"""
thesis_exp2_habit.py
Experiment 2, 3, 4: Habit Learning, System Integration, Manifold Engine
Reads from MongoDB. No Flask needed.
Run after Unity HabitExp (30 days) is complete and MLP models are trained.

Output:
  results/figD_habit_learning_curve.png
  results/figE_spot_discrimination.png
  results/figF_fat_sensitivity.png
  results/figG_intent_distribution.png
  results/figH_behavior_zone_heatmap.png
  results/expA_habit_convergence.png
  results/expB_entropy_heatmap_User_Mom.png
  results/expB_entropy_heatmap_User_Dad.png
  results/expC_topology_comparison.png
  results/exp2_summary.txt
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
# CHAPTER 4.2  --dynamic   習慣學習效果評估
# ═══════════════════════════════════════════════════════════════════════

def run_dynamic(db, out):
    print("\n" + "="*60)
    print("Chapter 4.2 — Habit Learning (--dynamic)")
    print("="*60)

    snaps   = list(db.habit_snapshots.find({}))
    obs     = list(db.observation_logs.find(
        {}, {"user":1,"action":1,"instance":1,"weight":1,"zone_name":1}))
    print(f"  habit_snapshots : {len(snaps)}")
    print(f"  observation_logs: {len(obs)}")

    if not snaps and not obs:
        print("  No dynamic data. Run Experiment 3 first.")
        return

    _plot_habit_learning_curve(snaps, out)
    _plot_spot_discrimination(obs, out)
    _plot_fat_sensitivity(obs, out)
    _save_dynamic_summary(snaps, obs, out)


def _learning_curve_for(snaps, user, action, spot_a):
    daily = defaultdict(lambda: defaultdict(int))
    for d in snaps:
        if d.get("user")!=user or norm(d.get("action",""))!=action: continue
        daily[d.get("date","")][d.get("instance","")] += d.get("daily_count",0)
    if not daily: return [], []
    dates = sorted(daily.keys())
    cum   = defaultdict(int)
    accs  = []
    for date in dates:
        for inst, cnt in daily[date].items(): cum[inst] += cnt
        if cum:
            top = sorted(cum.items(), key=lambda x: x[1], reverse=True)[0][0]
            hit = spot_a.lower() in top.lower() or top.lower() in spot_a.lower()
            accs.append(1.0 if hit else 0.0)
        else: accs.append(0.0)
    sm = [float(np.mean(accs[max(0,i-2):i+1])) for i in range(len(accs))]
    return list(range(1, len(dates)+1)), sm


def _convergence_day(sm):
    consec = 0
    for i, acc in enumerate(sm):
        consec = consec+1 if acc >= CONVERGENCE_ACC else 0
        if consec >= CONVERGENCE_DAYS: return i - CONVERGENCE_DAYS + 2
    return None


def _plot_habit_learning_curve(snaps, out):
    n   = len(SHOWCASE)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 5), sharey=True)
    if n == 1: axes = [axes]
    fig.suptitle(
        "Figure D: Habit Learning Curve — Prediction Accuracy over Days\n"
        "(3-day rolling mean; accuracy = Spot_A ranked #1 in cumulative weight)",
        fontsize=12, fontweight="bold")
    colors = ["#2196F3","#4CAF50","#E53935"]
    for ax, sc, c in zip(axes, SHOWCASE, colors):
        days, sm = _learning_curve_for(snaps, sc["user"], sc["action"], sc["spot_a"])
        conv     = _convergence_day(sm)
        if not days:
            ax.set_title(sc["label"]+"\n(no data)")
            ax.text(0.5,0.5,"No habit_snapshots data",
                    ha="center",va="center",transform=ax.transAxes); continue
        ax.plot(days, [s*100 for s in sm], "o-", color=c, linewidth=2.2,
                markersize=6, markerfacecolor="white", markeredgewidth=2,
                label="Accuracy (3-day rolling)")
        ax.axhline(y=CONVERGENCE_ACC*100, color="#FF9800",
                   linewidth=1.5, linestyle="--",
                   label=f"Threshold {CONVERGENCE_ACC:.0%}")
        if conv:
            ax.axvline(x=conv, color="#9C27B0", linewidth=2,
                       linestyle=":", label=f"Converged Day {conv}")
        ax.set_xlabel("Day", fontsize=11)
        ax.set_ylabel("Accuracy (%)", fontsize=11)
        ax.set_ylim(-5, 110)
        ax.set_title(sc["label"], fontsize=12, fontweight="bold")
        ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    path = os.path.join(out, "figD_habit_learning_curve.png")
    plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")


def _plot_spot_discrimination(obs, out):
    n   = len(SHOWCASE)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5))
    if n == 1: axes = [axes]
    fig.suptitle(
        "Figure E: Spot Discrimination — Cumulative Weight after 30 Days\n"
        "(Spot_A = user's preferred location)",
        fontsize=12, fontweight="bold")
    for ax, sc in zip(axes, SHOWCASE):
        wa = wb = wo = 0
        for d in obs:
            if d.get("user")!=sc["user"] or norm(d.get("action",""))!=sc["action"]: continue
            inst = d.get("instance","").lower()
            w    = d.get("weight", 0)
            if sc["spot_a"].lower() in inst or inst in sc["spot_a"].lower(): wa += w
            elif sc["spot_b"].lower() in inst or inst in sc["spot_b"].lower(): wb += w
            else: wo += w
        total = wa+wb+wo or 1
        lbls  = [f"Spot_A\n({sc['spot_a']})", f"Spot_B\n({sc['spot_b']})", "Other"]
        vals  = [wa, wb, wo]
        cols  = ["#2196F3","#FF9800","#BDBDBD"]
        bars  = ax.bar(lbls, vals, color=cols, alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+0.2,
                    f"{v}\n({v/total:.0%})",
                    ha="center", fontsize=10, fontweight="bold")
        ratio = wa/(wb+1e-9)
        ax.set_title(f"{sc['label']}\nA/B ratio = {ratio:.1f}×",
                     fontsize=11, fontweight="bold")
        ax.set_ylabel("Cumulative Weight"); ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    path = os.path.join(out, "figE_spot_discrimination.png")
    plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")


def _plot_fat_sensitivity(obs, out):
    if not _SBERT_OK:
        print("  SBERT not available, skipping Figure F")
        return

    print("  Loading SBERT for FAT analysis (CPU)...")
    model = SentenceTransformer("all-MiniLM-L6-v2",
                                device="cpu")

    fig, axes = plt.subplots(1, len(USERS), figsize=(6*len(USERS), 5.5), sharey=True)
    if len(USERS)==1: axes=[axes]
    fig.suptitle("Figure F: FAT Threshold Sensitivity\n"
                 "Precision / Recall / F1 across FAT values",
                 fontsize=13, fontweight="bold")

    for ax, user_id in zip(axes, USERS):
        pipeline = [
            {"$match": {"user": user_id}},
            {"$group": {"_id": {"action":"$action","instance":"$instance"},
                        "total_weight": {"$sum":"$weight"}}},
            {"$sort": {"total_weight":-1}},
        ]
        grouped = [{"action": r["_id"]["action"],
                    "instance": r["_id"]["instance"],
                    "weight": r["total_weight"]}
                   for r in obs
                   if isinstance(obs, list)]

        from collections import defaultdict as _dd
        agg = _dd(int)
        for d in obs:
            if d.get("user") != user_id: continue
            key = (norm(d.get("action","")), d.get("instance",""))
            agg[key] += d.get("weight", 0)
        grouped = [
            {"action": act, "instance": inst, "weight": w}
            for (act, inst), w in sorted(agg.items(), key=lambda x: -x[1])
        ]

        gt_min   = 3
        gt_habits= [g for g in grouped if g["weight"] >= gt_min]

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
                    for j in range(i+1, len(vecs)):
                        pairs += 1
                        if float(np.dot(vecs[i], vecs[j])) >= DEDUP_SIM:
                            redundant += 1
                redundancy = redundant/pairs if pairs > 0 else 0.0
            precision = 1.0 - redundancy
            gt_keys   = {f"{h['action']}@{h['instance']}" for h in gt_habits}
            lk        = {f"{h['action']}@{h['instance']}" for h in habits}
            recall    = len(gt_keys & lk)/len(gt_keys) if gt_keys else 0.0
            f1 = 2*precision*recall/(precision+recall) if precision+recall > 0 else 0.0
            results.append({"thr":thr,"precision":precision,"recall":recall,"f1":f1})

        x_     = list(range(len(FAT_THRESHOLDS)))
        precs  = [r["precision"] for r in results]
        recs   = [r["recall"]    for r in results]
        f1s    = [r["f1"]        for r in results]

        if 5 in FAT_THRESHOLDS:
            fi = FAT_THRESHOLDS.index(5)
            ax.axvline(x=fi, color="#E53935", linewidth=1.8,
                       linestyle="--", alpha=0.7, label="FAT=5 (selected)")

        ax.plot(x_, recs,  "o-", color="#2196F3", linewidth=2.2,
                markersize=8, markerfacecolor="white", markeredgewidth=2,
                label="Recall")
        ax.plot(x_, precs, "s-", color="#FF9800", linewidth=2.2,
                markersize=8, markerfacecolor="white", markeredgewidth=2,
                label="Precision")
        ax.plot(x_, f1s,   "^-", color="#4CAF50", linewidth=2.5,
                markersize=9, markerfacecolor="white", markeredgewidth=2.5,
                label="F1")

        for i, (r, p, f) in enumerate(zip(recs, precs, f1s)):
            ax.text(i, r+0.02, f"{r:.2f}", ha="center", fontsize=8, color="#1565C0")
            ax.text(i, p-0.06, f"{p:.2f}", ha="center", fontsize=8, color="#E65100")

        ax.set_xticks(x_)
        ax.set_xticklabels([f"FAT={v}" for v in FAT_THRESHOLDS], fontsize=10)
        ax.set_ylim(0, 1.25)
        ax.set_xlabel("Fast Adaptation Threshold", fontsize=11)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_title(user_id.replace("_"," "), fontsize=12, fontweight="bold")
        ax.legend(loc="lower left", fontsize=9)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    path = os.path.join(out, "figF_fat_sensitivity.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")


def _save_dynamic_summary(snaps, obs, out):
    lines = [
        "="*65,
        "Chapter 4.2: Habit Learning Summary",
        f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "="*65,"",
        f"habit_snapshots : {len(snaps)}",
        f"observation_logs: {len(obs)}","",
        "── Habit Learning Metrics ──",
    ]
    for sc in SHOWCASE:
        days, sm = _learning_curve_for(snaps, sc["user"], sc["action"], sc["spot_a"])
        conv     = _convergence_day(sm)
        final    = f"{sm[-1]*100:.1f}%" if sm else "N/A"
        wa = wb  = 0
        for d in obs:
            if d.get("user")!=sc["user"] or norm(d.get("action",""))!=sc["action"]: continue
            inst = d.get("instance","").lower()
            w    = d.get("weight", 0)
            if sc["spot_a"].lower() in inst: wa += w
            elif sc["spot_b"].lower() in inst: wb += w
        lines += [
            f"  [{sc['label']}]",
            f"    Convergence Day = {'Day '+str(conv) if conv else 'Not converged'}",
            f"    Final Accuracy  = {final}",
            f"    Spot_A weight   = {wa}",
            f"    Spot_B weight   = {wb}",
            f"    A/B ratio       = {wa/(wb+1e-9):.1f}x","",
        ]
    path = os.path.join(out, "dynamic_summary.txt")
    with open(path,"w",encoding="utf-8") as f: f.write("\n".join(lines))
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════
# CHAPTER 4.3  --system   系統整合效果評估
# ═══════════════════════════════════════════════════════════════════════

def run_system(db, out):
    print("\n" + "="*60)
    print("Chapter 4.3 — System Integration (--system)")
    print("="*60)

    proposals = list(db.service_proposals.find(
        {}, {"user_id":1,"intent":1,"confidence":1,"created_at":1}
    ).sort("created_at",1))
    obs = list(db.observation_logs.find(
        {}, {"user":1,"action":1,"zone_name":1,"weight":1}
    ))
    gt_docs = list(db.eval_logs.find(
        {"ground_truth":{"$exists":True,"$ne":""}},
        {"ground_truth":1}
    ).sort("timestamp",-1).limit(300))
    gt_labels = [norm(d.get("ground_truth","")) for d in gt_docs]
    gt_labels = [l for l in gt_labels if l not in ("Unknown","Standing","Walking")]

    print(f"  service_proposals : {len(proposals)}")
    print(f"  observation_logs  : {len(obs)}")
    print(f"  GT labels         : {len(gt_labels)}")

    if not proposals and not obs:
        print("  No system data. Run Experiment 3 first.")
        return

    if proposals:
        _plot_intent_distribution(proposals, gt_labels, out)
        _plot_confidence(proposals, out)
    _plot_behavior_zone_heatmap(obs, out)
    _save_system_summary(proposals, obs, out)


def _plot_intent_distribution(proposals, gt_labels, out):
    ic = Counter(norm(p.get("intent","")) for p in proposals)
    gc = Counter(gt_labels)
    lbls = [l for l in BEHAVIOR_ORDER if l in ic or l in gc]
    lbls += [l for l in (set(ic)|set(gc))
             if l not in BEHAVIOR_ORDER and l not in ("Unknown","Standing","Walking")]
    if not lbls: return
    np_ = sum(ic.values()) or 1
    ng  = sum(gc.values()) or 1
    gt_v  = [gc.get(l,0)/ng   for l in lbls]
    int_v = [ic.get(l,0)/np_  for l in lbls]
    x = np.arange(len(lbls)); w = 0.38
    fig, ax = plt.subplots(figsize=(max(10,len(lbls)*1.4), 5.5))
    ax.bar(x-w/2, [v*100 for v in gt_v],  w, color="#2196F3", alpha=0.85,
           edgecolor="white", label="GT Behaviour Distribution")
    ax.bar(x+w/2, [v*100 for v in int_v], w, color="#E53935", alpha=0.85,
           edgecolor="white", label="Service Proposal Intent Distribution")
    for i,(gv,iv) in enumerate(zip(gt_v,int_v)):
        if gv>0.01: ax.text(x[i]-w/2, gv*100+0.3, f"{gv*100:.0f}%",
                            ha="center", fontsize=8, color="#1565C0")
        if iv>0.01: ax.text(x[i]+w/2, iv*100+0.3, f"{iv*100:.0f}%",
                            ha="center", fontsize=8, color="#B71C1C")
    bc = sum(np.sqrt((ic.get(l,0)/np_)*(gc.get(l,0)/ng))
             for l in set(ic)|set(gc))
    n_ep = 300
    ax.set_title(
        f"Figure G: Service Proposal Intent vs GT Behaviour Distribution\n"
        f"Triggered={len(proposals)}  Trigger Rate={len(proposals)/n_ep:.1%}  "
        f"Bhattacharyya={bc:.3f}",
        fontsize=11, pad=10)
    ax.set_xticks(x); ax.set_xticklabels(lbls, fontsize=10)
    ax.set_ylabel("Relative Frequency (%)", fontsize=12)
    ax.set_ylim(0, max(max(gt_v), max(int_v), 0.01)*100*1.35)
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    path = os.path.join(out, "figG_intent_distribution.png")
    plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")


def _plot_confidence(proposals, out):
    confs = [float(p.get("confidence",0.0)) for p in proposals]
    if not confs: return
    mc, sc = float(np.mean(confs)), float(np.std(confs))
    fig, ax = plt.subplots(figsize=(7,4.5))
    ax.hist(confs, bins=min(15,len(confs)), color="#7C3AED",
            alpha=0.75, edgecolor="white")
    ax.axvline(x=0.60, color="#E53935", linewidth=1.5,
               linestyle="--", label="Trigger threshold C=0.60")
    ax.axvline(x=mc, color="#4CAF50", linewidth=1.5,
               label=f"Mean={mc:.3f} ± {sc:.3f}")
    ax.set_xlabel("Intent Confidence", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Proposal Confidence Distribution\n"
                 f"n={len(confs)}  Mean={mc:.3f}  Std={sc:.3f}", fontsize=11)
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    path = os.path.join(out, "figG2_confidence.png")
    plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")


def _plot_behavior_zone_heatmap(obs, out):
    """
    Figure H: Behaviour × Zone Heatmap
    X axis: Zone names
    Y axis: Behaviour labels
    Colour: cumulative weight (normalised per zone)
    Replaces UMAP — directly interpretable
    """
    from collections import defaultdict as _dd
    grid = _dd(lambda: _dd(int))
    zones_all = set()
    for d in obs:
        act  = norm(d.get("action",""))
        zone = d.get("zone_name","") or "Unknown_Zone"
        if act in ("Unknown","Standing","Walking","","None"): continue
        if not zone or zone == "Unknown_Zone": continue
        grid[act][zone] += d.get("weight",0)
        zones_all.add(zone)

    if not zones_all:
        print("  No zone_name data in observation_logs, skipping Figure H")
        print("  (zone_name is populated after perception.py is updated and Exp3 is run)")
        return

    behaviors = [b for b in BEHAVIOR_ORDER if b in grid]
    zones     = sorted(zones_all)
    if not behaviors or not zones: return

    matrix = np.zeros((len(behaviors), len(zones)))
    for i, b in enumerate(behaviors):
        for j, z in enumerate(zones):
            matrix[i, j] = grid[b][z]

    # Normalise per zone (column) so each zone sums to 1
    col_sums = matrix.sum(axis=0)
    col_sums[col_sums == 0] = 1
    matrix_norm = matrix / col_sums[np.newaxis, :]

    fig, ax = plt.subplots(
        figsize=(max(10, len(zones)*1.2), max(6, len(behaviors)*0.7)))
    im = ax.imshow(matrix_norm, aspect="auto", cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(range(len(zones)))
    ax.set_xticklabels([z.replace("_Zone","") for z in zones],
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
        "Figure H: Behaviour × Zone Affinity Heatmap\n"
        "(Normalised cumulative observation weight — diagonal = system learned correctly)",
        fontsize=12, fontweight="bold")
    ax.set_xlabel("Zone", fontsize=11)
    ax.set_ylabel("Behaviour", fontsize=11)
    plt.tight_layout()
    path = os.path.join(out, "figH_behavior_zone_heatmap.png")
    plt.savefig(path, dpi=200, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")


def _save_system_summary(proposals, obs, out):
    confs = [float(p.get("confidence",0.0)) for p in proposals]
    bc_str = "N/A"
    lines = [
        "="*65,
        "Chapter 4.3: System Integration Summary",
        f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "="*65,"",
        f"service_proposals : {len(proposals)}",
        f"Trigger Rate      : {len(proposals)/300:.1%}" if proposals else "",
        f"Mean Confidence   : {np.mean(confs):.3f}" if confs else "",
        f"observation_logs  : {len(obs)}","",
    ]
    path = os.path.join(out, "system_summary.txt")
    with open(path,"w",encoding="utf-8") as f: f.write("\n".join(lines))
    print(f"  Saved: {path}")




# ═══════════════════════════════════════════════════════════════════════
# EXPERIMENT A  --convergence   習慣收斂曲線
# ═══════════════════════════════════════════════════════════════════════

def run_convergence(db, out):
    print("\n" + "="*60)
    print("Experiment A — Habit Convergence Curve (--convergence)")
    print("="*60)

    docs = list(db.affinity_history.find({}))
    print(f"  affinity_history: {len(docs)}")
    if not docs:
        print("  No affinity_history. Run Experiment 3 first.")
        return

    SHOW = [
        {"user":"User_Mom","action":"PhoneUse","zone":"Watching_Zone",
         "label":"Mom · PhoneUse @ Sofa"},
        {"user":"User_Dad","action":"PhoneUse","zone":"Typing_Zone",
         "label":"Dad · PhoneUse @ Desk (Control)"},
    ]

    L3_PRIOR = 0.10   # Gemma3 static prior for PhoneUse@Sofa
    FAT_THR  = 5      # FAT threshold weight

    # find FAT trigger day from observation_logs
    fat_days = {}
    for sc in SHOW:
        pipeline = [
            {"$match": {"user": sc["user"], "action": sc["action"]}},
            {"$group": {"_id": "$last_date",
                        "daily": {"$sum": "$weight"}}},
            {"$sort": {"_id": 1}},
        ]
        cum = 0
        for r in db.observation_logs.aggregate(pipeline):
            cum += r["daily"]
            if cum >= FAT_THR:
                fat_days[sc["user"]+"_"+sc["action"]] = r["_id"]
                break

    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = ["#2196F3", "#BDBDBD"]

    for sc, c in zip(SHOW, colors):
        # collect daily affinity
        by_date = {}
        for d in docs:
            if d.get("user_id") != sc["user"]: continue
            if d.get("action")  != sc["action"]: continue
            date = d.get("date","")
            aff  = d.get("affinity", 0.0)
            if date:
                by_date[date] = max(by_date.get(date, 0.0), aff)

        if not by_date:
            continue

        dates  = sorted(by_date.keys())
        days   = list(range(1, len(dates)+1))
        affs   = [by_date[d] for d in dates]

        # compute 3-day rolling std for variance band
        stds = []
        for i in range(len(affs)):
            window = affs[max(0,i-2):i+1]
            stds.append(float(np.std(window)) if len(window) > 1 else 0.0)

        ax.plot(days, affs, "o-", color=c, linewidth=2.2,
                markersize=5, markerfacecolor="white", markeredgewidth=2,
                label=sc["label"])
        ax.fill_between(days,
                         [a-s for a,s in zip(affs,stds)],
                         [a+s for a,s in zip(affs,stds)],
                         color=c, alpha=0.15)

        # mark FAT trigger day
        key = sc["user"]+"_"+sc["action"]
        if key in fat_days:
            fat_date = fat_days[key]
            if fat_date in dates:
                fd = dates.index(fat_date) + 1
                ax.axvline(x=fd, color=c, linewidth=1.5,
                           linestyle=":", alpha=0.7)
                ax.annotate(f"FAT triggered\nDay {fd}",
                            xy=(fd, affs[fd-1]),
                            xytext=(fd+0.5, affs[fd-1]+0.05),
                            fontsize=8, color=c,
                            arrowprops=dict(arrowstyle="->",
                                            color=c, lw=1))

    ax.axhline(y=L3_PRIOR, color="#FF9800", linewidth=1.5,
               linestyle="--", label=f"L3 Static Prior ({L3_PRIOR})")
    ax.axhline(y=0.70, color="#4CAF50", linewidth=1,
               linestyle="--", alpha=0.5, label="Personalised threshold (0.70)")

    ax.set_xlabel("Day", fontsize=12)
    ax.set_ylabel("Affinity Score", fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(
        "Experiment A: Habit Convergence Curve\n"
        "Sofa × PhoneUse Affinity Score over 30 Days\n"
        "(shaded band = 3-day rolling std; dotted line = FAT trigger)",
        fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    path = os.path.join(out, "expA_habit_convergence.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════
# EXPERIMENT B  --entropy   時空意圖熵值熱力圖
# ═══════════════════════════════════════════════════════════════════════

def run_entropy(db, out, manifold_engine=None):
    print("\n" + "="*60)
    print("Experiment B — Spatiotemporal Entropy Heatmap (--entropy)")
    print("="*60)

    if manifold_engine is None:
        try:
            sys.path.insert(0, ".")
            from manifold_engine import ManifoldEngine
            manifold_engine = ManifoldEngine(db)
            print("  ManifoldEngine loaded")
        except Exception as e:
            print(f"  Cannot load ManifoldEngine: {e}")
            return

    # find sofa center from zone graph or use default
    zone_doc = db.scene_snapshots.find_one({"label": "sofa"})
    if zone_doc and zone_doc.get("pos"):
        cx = zone_doc["pos"][0] / 10.0
        cz = zone_doc["pos"][1] / 10.0
    else:
        cx, cz = 0.25, -0.12   # default sofa centre (normalised)

    for user_id in ["User_Mom", "User_Dad"]:
        hours, matrix = manifold_engine.probe_spatiotemporal(
            user_id, pos=[cx, cz], prev_action="Standing", n_hours=48)

        if matrix.max() < 1e-6:
            print(f"  {user_id}: no model yet, skipping")
            continue

        # Gaussian smoothing
        try:
            from scipy.ndimage import gaussian_filter
            matrix_smooth = gaussian_filter(matrix.astype(float), sigma=1.0)
        except ImportError:
            matrix_smooth = matrix

        # Shannon entropy per time step
        entropies = []
        for j in range(matrix_smooth.shape[1]):
            p = matrix_smooth[:, j].copy()
            p = p / (p.sum() + 1e-9)
            H = -float(np.sum(p * np.log2(p + 1e-9)))   # bits
            entropies.append(H)

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(14, 8),
            gridspec_kw={"height_ratios": [3, 1]},
            sharex=True)

        im = ax1.imshow(
            matrix_smooth, aspect="auto", origin="lower",
            cmap="YlOrRd", vmin=0, vmax=matrix_smooth.max(),
            interpolation="bicubic",
            extent=[0, 24, -0.5, len(BEHAVIOR_ORDER)-0.5])
        plt.colorbar(im, ax=ax1, label="Intent Probability")
        ax1.set_yticks(range(len(BEHAVIOR_ORDER)))
        ax1.set_yticklabels(BEHAVIOR_ORDER, fontsize=8)
        ax1.set_ylabel("Behaviour")
        ax1.set_title(
            f"Experiment B: Spatiotemporal Intent Heatmap — {user_id}\n"
            f"Fixed pos=Sofa_Zone, prev=Standing, sweep 0-24h",
            fontsize=11, fontweight="bold")

        n_pts   = matrix_smooth.shape[1]
        x_ticks = np.linspace(0, 24, n_pts, endpoint=False)
        ax2.plot(x_ticks, entropies, color="#E53935", linewidth=2)
        ax2.fill_between(x_ticks, entropies,
                          color="#E53935", alpha=0.15)
        max_entropy = float(np.log2(N_BEHAVIORS))
        ax2.axhline(y=max_entropy, color="#BDBDBD",
                    linewidth=1, linestyle="--",
                    label=f"Max entropy (uniform) = {max_entropy:.2f} bits")
        ax2.set_xlabel("Time of Day (hour)")
        ax2.set_ylabel("Shannon Entropy H (bits)")
        ax2.set_xlim(0, 24)
        ax2.legend(fontsize=8)
        ax2.grid(axis="y", alpha=0.25)
        ax2.set_title("Intent Entropy — low = confident prediction",
                       fontsize=9)

        plt.tight_layout()
        path = os.path.join(out,
            f"expB_entropy_heatmap_{user_id}.png")
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════
# EXPERIMENT C  --topology   雙熱力圖拓撲對比
# ═══════════════════════════════════════════════════════════════════════

def run_topology(db, out, manifold_engine=None):
    print("\n" + "="*60)
    print("Experiment C — User Topology Comparison (--topology)")
    print("="*60)

    if manifold_engine is None:
        try:
            sys.path.insert(0, ".")
            from manifold_engine import ManifoldEngine
            manifold_engine = ManifoldEngine(db)
        except Exception as e:
            print(f"  Cannot load ManifoldEngine: {e}")
            return

    # collect zone centers from zone_graph (observation_logs zone_name)
    # fall back to hard-coded defaults if no data
    zone_defaults = {
        "Sofa_Zone":   [0.25, -0.15],
        "Typing_Zone": [-0.85, -0.48],
        "Kitchen_Zone":[ 0.34,  0.01],
        "Bed_Zone":    [-0.70,  0.30],
    }

    # try to get real centres from observation_logs
    from collections import defaultdict as _dd
    zone_pos = _dd(list)
    for d in db.observation_logs.find(
            {"zone_name": {"$exists": True, "$ne": ""},
             "pos":       {"$exists": True}},
            {"zone_name":1, "pos":1}):
        zn  = d.get("zone_name","")
        pos = d.get("pos")
        if zn and isinstance(pos, list) and len(pos) == 2:
            zone_pos[zn].append(pos)

    zone_centers = {}
    for zn, positions in zone_pos.items():
        cx = float(np.mean([p[0] for p in positions])) / 10.0
        cz = float(np.mean([p[1] for p in positions])) / 10.0
        zone_centers[zn] = [cx, cz]
    if not zone_centers:
        zone_centers = zone_defaults
        print("  Using default zone centres")

    users       = ["User_Mom", "User_Dad"]
    user_labels = ["Mom's MLP", "Dad's MLP"]

    n_zones = len(zone_centers)
    n_beh   = len(BEHAVIOR_ORDER)
    zone_names_ord = list(zone_centers.keys())

    fig, axes = plt.subplots(1, 2, figsize=(6*2, max(5, n_zones*0.8+2)))
    fig.suptitle(
        "Experiment C: Behaviour-Zone Topology Comparison\n"
        "Proves per-user isolated MLP avoids habit cross-contamination",
        fontsize=12, fontweight="bold")

    for ax, uid, ulabel in zip(axes, users, user_labels):
        zn_list, matrix = manifold_engine.probe_zone_behavior(
            uid, zone_centers, virtual_hour=20.0,
            prev_action="Standing")

        # reorder rows to match zone_names_ord
        row_order = [zn_list.index(z) if z in zn_list else 0
                     for z in zone_names_ord]
        matrix_ord = matrix[row_order, :]

        # only show BEHAVIOR_ORDER columns
        col_idx = [BEHAVIOR_LABELS.index(b) if b in BEHAVIOR_LABELS else 0
                   for b in BEHAVIOR_ORDER]
        matrix_show = matrix_ord[:, col_idx]

        if matrix_show.max() < 1e-6:
            ax.set_title(f"{ulabel}\n(no model yet)")
            continue

        vmax_val = max(float(matrix_show.max()), 0.8)
        im = ax.imshow(matrix_show, aspect="auto",
                       cmap="Blues", vmin=0, vmax=vmax_val)
        plt.colorbar(im, ax=ax, label="Probability")
        for i in range(n_zones):
            for j in range(n_beh):
                v = matrix_show[i, j]
                if v > 0.08:
                    ax.text(j, i, f"{v:.2f}", ha="center",
                            va="center", fontsize=7,
                            color="white" if v > 0.5 else "black")

        ax.set_xticks(range(n_beh))
        ax.set_xticklabels(BEHAVIOR_ORDER, rotation=40,
                           ha="right", fontsize=8)
        ax.set_yticks(range(n_zones))
        ax.set_yticklabels(
            [z.replace("_Zone","") for z in zone_names_ord],
            fontsize=9)
        ax.set_xlabel("Behaviour")
        ax.set_ylabel("Zone")
        ax.set_title(ulabel, fontsize=12, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(out, "expC_topology_comparison.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


if __name__ == "__main__":
    import argparse as _ap
    parser = _ap.ArgumentParser(
        description="Experiment 2/3/4: Habit + System + Manifold (no Flask needed)")
    parser.add_argument("--out",          default="results")
    parser.add_argument("--skip-entropy", action="store_true",
                        help="Skip entropy/topology plots (require trained MLP)")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    db = connect()
    print(f"Connected → {DB_NAME}")

    run_dynamic(db, args.out)
    run_system(db, args.out)
    run_convergence(db, args.out)

    if not args.skip_entropy:
        run_entropy(db, args.out)
        run_topology(db, args.out)
    else:
        print("\n[Skipped] --entropy and --topology "
              "(run without --skip-entropy after training MLP)")

    print("\nDone. Check", args.out)