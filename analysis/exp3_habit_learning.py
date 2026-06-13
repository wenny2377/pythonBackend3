import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from config import Config

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from pymongo import MongoClient

MONGO_URI     = "mongodb://127.0.0.1:27017/"
DB_BASELINE   = "robot_exp_baseline"
DB_CORRUPTION = "robot_exp_corruption"
OUT           = os.path.join(_ROOT, "analysis", "results")

USERS      = ["User_Mom", "User_Dad"]
FAT        = 5
NO_WEIGHT  = {"PickingUp", "PuttingDown", "Walking", "Standing", "StandUp"}
BINS       = [10, 20, 30, 40, 50, 60, 80, 100, 120, 150, 200, 250, 300]

ACTION_LABELS = [
    "Eating", "Drinking", "SittingDrink", "Cooking", "Opening",
    "Laying", "Watching", "Reading", "Cleaning", "PhoneUse",
    "Typing", "Sitting",
]

COLOR_BASE = "#5C6BC0"
COLOR_CORR = "#EF5350"
USER_COLORS = {"User_Mom": "#E91E63", "User_Dad": "#1976D2"}

def connect(db_name):
    return MongoClient(MONGO_URI)[db_name]

def get_observations(db, user_id):
    return list(db.observation_logs.find(
        {"user": user_id, "action": {"$nin": list(NO_WEIGHT)}},
        {"action": 1, "zone_name": 1, "instance": 1,
         "weight": 1, "last_seen": 1}
    ).sort("last_seen", 1))

def get_transitions(db, user_id):
    return list(db.transition_counts.find(
        {"user_id": user_id},
        {"from_action": 1, "to_action": 1, "count": 1, "time_slot": 1}
    ).sort("count", -1))

def compute_habits(obs, fat):
    agg = defaultdict(float)
    for d in obs:
        key = (d.get("action",""), d.get("zone_name") or d.get("instance",""))
        agg[key] += d.get("weight", 1)
    return [(a, z, w) for (a, z), w in agg.items() if w >= fat]

def compute_hsi(obs, fat, gt_habits):
    habits    = compute_habits(obs, fat)
    gt_keys   = {f"{a}@{z}" for a, z, _ in gt_habits}
    h_keys    = {f"{a}@{z}" for a, z, _ in habits}
    stable    = len(h_keys & gt_keys)
    return stable / len(gt_keys) if gt_keys else 0.0

def plot_hsi_curve():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    fig.suptitle(
        "Habit Stability Index (HSI) — Learning Convergence\n"
        "Baseline vs Corruption (FAT=5)",
        fontsize=13, fontweight="bold")

    has_corruption = False
    try:
        db_corr = connect(DB_CORRUPTION)
        test    = db_corr.observation_logs.count_documents({})
        has_corruption = test > 0
    except Exception:
        pass

    for ax, uid in zip(axes, USERS):
        db_base = connect(DB_BASELINE)
        obs_b   = get_observations(db_base, uid)

        if not obs_b:
            ax.set_title(f"{uid.replace('_',' ')}\n(no data)")
            continue

        gt_habits = compute_habits(obs_b, fat=3)

        xs_b, ys_b = [], []
        for n in BINS:
            if n > len(obs_b): break
            hsi = compute_hsi(obs_b[:n], FAT, gt_habits)
            xs_b.append(n)
            ys_b.append(hsi)

        ax.plot(xs_b, ys_b, "o-", color=COLOR_BASE, lw=2.5,
                ms=6, mfc="white", mew=2, label="Baseline")

        if has_corruption:
            db_corr = connect(DB_CORRUPTION)
            obs_c   = get_observations(db_corr, uid)
            xs_c, ys_c = [], []
            for n in BINS:
                if n > len(obs_c): break
                hsi = compute_hsi(obs_c[:n], FAT, gt_habits)
                xs_c.append(n)
                ys_c.append(hsi)
            ax.plot(xs_c, ys_c, "s--", color=COLOR_CORR, lw=2.5,
                    ms=6, mfc="white", mew=2, label="Corruption")

        conv_idx = next((i for i, y in enumerate(ys_b) if y >= 0.80), None)
        if conv_idx is not None:
            ax.axvline(x=xs_b[conv_idx], color="gray",
                       linestyle=":", lw=1.5, alpha=0.7)
            ax.text(xs_b[conv_idx] + 5, 0.05,
                    f"Converge\n~{xs_b[conv_idx]} ep",
                    fontsize=8, color="gray")

        ax.axhline(y=0.80, color="gray", linestyle="--",
                   alpha=0.4, lw=1.2, label="80% target")
        ax.set_xlabel("Episodes", fontsize=11)
        ax.set_ylabel("Habit Stability Index (HSI)", fontsize=11)
        ax.set_ylim(0, 1.1)
        ax.set_xlim(0, max(BINS) + 15)
        ax.set_title(uid.replace("_"," "), fontsize=12, fontweight="bold")
        ax.legend(fontsize=9, loc="lower right")
        ax.grid(alpha=0.25)

    plt.tight_layout()
    path = os.path.join(OUT, "exp3_hsi_curve.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[exp3] Saved: {path}")

def plot_personalization_gap():
    db = connect(DB_BASELINE)

    user_profiles = {}
    for uid in USERS:
        obs  = get_observations(db, uid)
        agg  = defaultdict(float)
        for d in obs:
            agg[d.get("action","")] += d.get("weight", 1)
        total = sum(agg.values()) or 1
        user_profiles[uid] = {a: w/total*100 for a, w in agg.items()
                               if a in ACTION_LABELS}

    actions_present = [a for a in ACTION_LABELS
                       if any(user_profiles[u].get(a, 0) > 0
                              for u in USERS)]
    if not actions_present:
        print("[exp3] No personalization data")
        return

    n   = len(actions_present)
    x   = np.arange(n)
    w   = 0.35

    fig, ax = plt.subplots(figsize=(13, 6))

    for i, uid in enumerate(USERS):
        vals   = [user_profiles[uid].get(a, 0) for a in actions_present]
        offset = (i - 0.5) * w
        bars   = ax.bar(x + offset, vals, w,
                        color=USER_COLORS[uid], alpha=0.85,
                        label=uid.replace("_"," "), edgecolor="white")
        for bar, v in zip(bars, vals):
            if v > 1.5:
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.3,
                        f"{v:.1f}%",
                        ha="center", fontsize=7.5, rotation=0)

    ax.set_xticks(x)
    ax.set_xticklabels(actions_present, rotation=35, ha="right", fontsize=10)
    ax.set_ylabel("Relative Frequency (%)", fontsize=12)
    ax.set_title(
        "Personalization Gap — Learned Habit Profiles\n"
        "User Mom vs User Dad (Baseline, n=300)",
        fontsize=12, fontweight="bold", pad=10)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    path = os.path.join(OUT, "exp3_personalization_gap.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[exp3] Saved: {path}")

def plot_timeslot_heatmap():
    db      = connect(DB_BASELINE)
    slots   = ["Morning", "Noon", "Afternoon", "Evening", "Night"]
    actions = [a for a in ACTION_LABELS
               if db.observation_logs.count_documents({"action": a}) > 0]
    if not actions:
        print("[exp3] No timeslot data")
        return

    matrix = np.zeros((len(actions), len(slots)))
    for i, action in enumerate(actions):
        for j, slot in enumerate(slots):
            docs = list(db.observation_logs.find(
                {"action": action, "time_slot": slot},
                {"weight": 1}))
            matrix[i][j] = sum(d.get("weight", 1) for d in docs)

    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    norm = matrix / row_sums

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(norm, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Relative Frequency", fontsize=10)

    for i in range(len(actions)):
        for j in range(len(slots)):
            if norm[i][j] > 0.05:
                ax.text(j, i, f"{norm[i][j]:.2f}",
                        ha="center", va="center", fontsize=9,
                        color="white" if norm[i][j] > 0.6 else "black",
                        fontweight="bold" if norm[i][j] > 0.5 else "normal")

    ax.set_xticks(range(len(slots)))
    ax.set_yticks(range(len(actions)))
    ax.set_xticklabels(slots, fontsize=11)
    ax.set_yticklabels(actions, fontsize=11)
    ax.set_xlabel("Time Slot", fontsize=12)
    ax.set_ylabel("Activity", fontsize=12)
    ax.set_title(
        "Temporal Activity Distribution\n"
        "Learned from Observation Logs (Baseline)",
        fontsize=12, fontweight="bold", pad=10)

    plt.tight_layout()
    path = os.path.join(OUT, "exp3_timeslot_heatmap.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[exp3] Saved: {path}")

def plot_transition_heatmap():
    db = connect(DB_BASELINE)

    all_transitions = []
    for uid in USERS:
        all_transitions += get_transitions(db, uid)

    evening = [d for d in all_transitions if d.get("time_slot") == "Evening"]
    docs    = evening if evening else all_transitions

    present = [l for l in ACTION_LABELS
               if any(d.get("from_action") == l or
                      d.get("to_action")   == l for d in docs)]
    if not present:
        print("[exp3] No transition data")
        return

    n      = len(present)
    matrix = np.zeros((n, n))
    for d in docs:
        fi = present.index(d["from_action"]) if d["from_action"] in present else -1
        ti = present.index(d["to_action"])   if d["to_action"]   in present else -1
        if fi >= 0 and ti >= 0:
            matrix[fi][ti] += d.get("count", 0)

    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    norm = matrix / row_sums

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Transition Probability", fontsize=10)

    for i in range(n):
        for j in range(n):
            if norm[i][j] > 0.05:
                ax.text(j, i, f"{norm[i][j]:.2f}",
                        ha="center", va="center", fontsize=8,
                        color="white" if norm[i][j] > 0.6 else "black")

    slot_label = "Evening" if evening else "All Slots"
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(present, rotation=40, ha="right", fontsize=9)
    ax.set_yticklabels(present, fontsize=9)
    ax.set_xlabel("To Activity", fontsize=12)
    ax.set_ylabel("From Activity", fontsize=12)
    ax.set_title(
        f"Learned Behavior Transition Matrix ({slot_label})\n"
        "Normalized by Row (Transition Probability)",
        fontsize=12, fontweight="bold", pad=10)

    plt.tight_layout()
    path = os.path.join(OUT, "exp3_transition_heatmap.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[exp3] Saved: {path}")

def save_summary():
    db = connect(DB_BASELINE)
    lines = [
        "Experiment 3: Habit Learning",
        f"Baseline DB: {DB_BASELINE}",
        "",
        f"FAT threshold: {FAT}",
        "",
        "Habit profiles per user:",
    ]
    for uid in USERS:
        obs = get_observations(db, uid)
        habits = compute_habits(obs, FAT)
        lines.append(f"\n  {uid} ({len(obs)} observations, {len(habits)} stable habits):")
        for a, z, w in sorted(habits, key=lambda x: -x[2])[:8]:
            lines.append(f"    {a:<14} @ {z:<20} weight={w:.1f}")

    path = os.path.join(OUT, "exp3_summary.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"[exp3] Saved: {path}")
    print("\n".join(lines))

def main():
    os.makedirs(OUT, exist_ok=True)
    db = connect(DB_BASELINE)
    total = db.observation_logs.count_documents(
        {"action": {"$nin": list(NO_WEIGHT)}})
    if total == 0:
        print(f"[exp3] No observation_logs in {DB_BASELINE}")
        return
    print(f"[exp3] {total} observations")
    plot_hsi_curve()
    plot_personalization_gap()
    plot_timeslot_heatmap()
    plot_transition_heatmap()
    save_summary()

if __name__ == "__main__":
    main()