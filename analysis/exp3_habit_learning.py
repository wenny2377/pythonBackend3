"""
exp3_habit_learning.py
──────────────────────
Experiment 3: Habit Learning

Uses observation_logs from DB_NAME (baseline DB).
No additional experiment run needed.

Run: DB_NAME=robot_exp_baseline python3 analysis/exp3_habit_learning.py

Outputs:
  analysis/results/exp3_fat_sensitivity.png
  analysis/results/exp3_learning_curve.png
  analysis/results/exp3_transition_heatmap.png
  analysis/results/exp3_summary.txt
"""

import os
import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from pymongo import MongoClient

MONGO_URI = "mongodb://127.0.0.1:27017/"
DB_NAME   = os.environ.get("DB_NAME", "robot_rag_db")
OUT       = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

USERS      = ["User_Mom", "User_Dad"]
FAT_VALUES = [2, 3, 5, 8, 10]
FAT_SELECT = 5
NO_WEIGHT  = {"PickingUp", "PuttingDown", "Walking", "Standing", "StandUp"}

EPISODE_BINS = [10, 20, 30, 40, 50, 60, 80, 100, 120, 150, 200, 250, 300]

ACTION_LABELS = [
    "Eating", "Drinking", "SittingDrink", "Cooking", "Opening",
    "Laying", "Watching", "Reading", "Cleaning", "PhoneUse",
    "Typing", "Sitting",
]


def connect():
    return MongoClient(MONGO_URI)[DB_NAME]


def get_observations(db, user_id: str) -> list:
    return list(db.observation_logs.find(
        {"user": user_id, "action": {"$nin": list(NO_WEIGHT)}},
        {"action": 1, "zone_name": 1, "instance": 1, "weight": 1, "last_seen": 1}
    ).sort("last_seen", 1))


def get_transition_counts(db, user_id: str) -> list:
    return list(db.transition_counts.find(
        {"user_id": user_id},
        {"from_action": 1, "to_action": 1, "count": 1, "time_slot": 1}
    ).sort("count", -1))


def compute_habits(obs: list, fat: int) -> list:
    agg = defaultdict(int)
    for d in obs:
        key = (d.get("action",""), d.get("zone_name") or d.get("instance",""))
        agg[key] += d.get("weight", 1)
    return [(act, inst, w) for (act, inst), w in agg.items() if w >= fat]


def compute_precision_recall(habits: list, gt_habits: list) -> tuple:
    if not habits:
        return 0.0, 0.0, 0.0
    gt_keys     = {f"{a}@{i}" for a, i, _ in gt_habits}
    habit_keys  = {f"{a}@{i}" for a, i, _ in habits}
    precision   = len(habit_keys & gt_keys) / len(habit_keys) if habit_keys else 0.0
    recall      = len(habit_keys & gt_keys) / len(gt_keys) if gt_keys else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def plot_fat_sensitivity(db):
    fig, axes = plt.subplots(1, len(USERS),
                             figsize=(6 * len(USERS), 5), sharey=True)
    if len(USERS) == 1:
        axes = [axes]
    fig.suptitle("Experiment 3 — FAT threshold sensitivity",
                 fontsize=12, fontweight="bold")

    summary_lines = []

    for ax, uid in zip(axes, USERS):
        obs = get_observations(db, uid)
        if not obs:
            ax.set_title(f"{uid}\n(no data)")
            continue

        gt_habits = compute_habits(obs, fat=3)
        results   = []

        for fat in FAT_VALUES:
            habits = compute_habits(obs, fat)
            p, r, f1 = compute_precision_recall(habits, gt_habits)
            results.append({"fat": fat, "p": p, "r": r, "f1": f1})

        x = range(len(FAT_VALUES))
        ax.plot(x, [r["r"]  for r in results], "o-", color="#2196F3",
                lw=2, ms=8, mfc="white", mew=2, label="Recall")
        ax.plot(x, [r["p"]  for r in results], "s-", color="#FF9800",
                lw=2, ms=8, mfc="white", mew=2, label="Precision")
        ax.plot(x, [r["f1"] for r in results], "^-", color="#4CAF50",
                lw=2.5, ms=9, mfc="white", mew=2.5, label="F1")

        if FAT_SELECT in FAT_VALUES:
            sel_idx = FAT_VALUES.index(FAT_SELECT)
            ax.axvline(x=sel_idx, color="#E53935", linestyle="--",
                       alpha=0.7, lw=1.8, label=f"FAT={FAT_SELECT} selected")
            sel = results[sel_idx]
            summary_lines.append(
                f"{uid}  FAT={FAT_SELECT}  P={sel['p']:.3f}  R={sel['r']:.3f}  F1={sel['f1']:.3f}")

        ax.set_xticks(x)
        ax.set_xticklabels([f"FAT={v}" for v in FAT_VALUES], fontsize=10)
        ax.set_ylim(0, 1.2)
        ax.set_xlabel("FAT threshold")
        ax.set_ylabel("Score")
        ax.set_title(uid.replace("_"," "), fontsize=11, fontweight="bold")
        ax.legend(loc="lower left", fontsize=9)
        ax.grid(alpha=0.2)

    plt.tight_layout()
    path = os.path.join(OUT, "exp3_fat_sensitivity.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[exp3] Saved: {path}")
    return summary_lines


def plot_learning_curve(db):
    fig, axes = plt.subplots(1, len(USERS),
                             figsize=(7 * len(USERS), 5))
    if len(USERS) == 1:
        axes = [axes]
    fig.suptitle("Experiment 3 — Learning curve (F1 vs observations)",
                 fontsize=12, fontweight="bold")

    colors = {2:"#1E88E5", 3:"#43A047", 5:"#FB8C00", 8:"#E53935", 10:"#8E24AA"}

    for ax, uid in zip(axes, USERS):
        obs = get_observations(db, uid)
        if not obs:
            ax.set_title(f"{uid}\n(no data)")
            continue

        gt_habits = compute_habits(obs, fat=3)

        for fat in FAT_VALUES:
            f1s, xs = [], []
            for n in EPISODE_BINS:
                if n > len(obs):
                    break
                h = compute_habits(obs[:n], fat)
                _, _, f1 = compute_precision_recall(h, gt_habits)
                f1s.append(f1)
                xs.append(n)
            if f1s:
                ax.plot(xs, f1s, marker="o", color=colors[fat],
                        lw=2, ms=5, label=f"FAT={fat}")

        ax.axhline(y=0.85, color="gray", linestyle="--",
                   alpha=0.5, label="85% target")
        ax.set_xlabel("Observations")
        ax.set_ylabel("F1")
        ax.set_title(uid.replace("_"," "), fontsize=11, fontweight="bold")
        ax.set_xlim(0, max(EPISODE_BINS) + 10)
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT, "exp3_learning_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[exp3] Saved: {path}")


def plot_transition_heatmap(db):
    fig, axes = plt.subplots(1, len(USERS),
                             figsize=(8 * len(USERS), 7))
    if len(USERS) == 1:
        axes = [axes]
    fig.suptitle("Experiment 3 — Learned transition matrix (Evening)",
                 fontsize=12, fontweight="bold")

    for ax, uid in zip(axes, USERS):
        docs = [d for d in get_transition_counts(db, uid)
                if d.get("time_slot") == "Evening"]
        if not docs:
            docs = get_transition_counts(db, uid)

        labels_present = [l for l in ACTION_LABELS
                         if any(d["from_action"] == l or
                                d["to_action"] == l for d in docs)]
        if not labels_present:
            ax.set_title(f"{uid}\n(no transitions)")
            continue

        n      = len(labels_present)
        matrix = np.zeros((n, n))
        for d in docs:
            fi = labels_present.index(d["from_action"]) \
                 if d["from_action"] in labels_present else -1
            ti = labels_present.index(d["to_action"]) \
                 if d["to_action"] in labels_present else -1
            if fi >= 0 and ti >= 0:
                matrix[fi][ti] = d.get("count", 0)

        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        norm = matrix / row_sums

        im = ax.imshow(norm, cmap="YlOrRd", vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, label="Transition probability")

        for i in range(n):
            for j in range(n):
                if norm[i][j] > 0.05:
                    ax.text(j, i, f"{norm[i][j]:.2f}",
                            ha="center", va="center", fontsize=7,
                            color="white" if norm[i][j] > 0.6 else "black")

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels_present, rotation=40, ha="right", fontsize=8)
        ax.set_yticklabels(labels_present, fontsize=8)
        ax.set_xlabel("To action")
        ax.set_ylabel("From action")
        ax.set_title(uid.replace("_"," "), fontsize=11, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(OUT, "exp3_transition_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[exp3] Saved: {path}")


def save_summary(db, fat_lines: list):
    lines = [
        "Experiment 3: Habit Learning",
        f"DB: {DB_NAME}",
        "",
        "FAT sensitivity (FAT=5 selected):",
    ] + fat_lines + ["", "Top learned transitions:"]

    for uid in USERS:
        docs = get_transition_counts(db, uid)[:5]
        if not docs: continue
        lines.append(f"\n  {uid}:")
        for d in docs:
            lines.append(
                f"    {d['from_action']:14} → {d['to_action']:14} "
                f"count={d['count']} ({d.get('time_slot','?')})")

    path = os.path.join(OUT, "exp3_summary.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"[exp3] Saved: {path}")
    print("\n".join(lines))


def main():
    os.makedirs(OUT, exist_ok=True)
    db = connect()

    total_obs = db.observation_logs.count_documents(
        {"action": {"$nin": list(NO_WEIGHT)}})
    if total_obs == 0:
        print(f"[exp3] No observation_logs in {DB_NAME}")
        return

    print(f"[exp3] {total_obs} observations from {DB_NAME}")

    fat_lines = plot_fat_sensitivity(db)
    plot_learning_curve(db)
    plot_transition_heatmap(db)
    save_summary(db, fat_lines)


if __name__ == "__main__":
    main()