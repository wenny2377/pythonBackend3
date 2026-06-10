"""
analysis/layer2_habit.py
Layer 2: Habit Learning Analysis
Outputs:
  results/Fig3_fat_sensitivity.png
  results/Fig4_learning_curve.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from pymongo import MongoClient
import datetime

try:
    from sentence_transformers import SentenceTransformer
    _SBERT_OK = True
except ImportError:
    _SBERT_OK = False

MONGO_URI = "mongodb://127.0.0.1:27017/"
DB_NAME   = "robot_rag_db"
OUT = os.path.join(os.path.dirname(__file__), "results")

USERS         = ["User_Mom", "User_Dad"]
FAT_VALUES    = [2, 3, 5, 8, 10]
NO_WEIGHT     = {"PickingUp", "PuttingDown", "Walking", "Standing", "StandUp"}
DEDUP_SIM     = 0.78
SELECTED_FAT  = 5

# Learning curve 參數
EPISODE_BINS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200]
ACCURACY_THRESHOLD = 0.85  # 達到 85% 準確率所需的 episodes

def connect():
    return MongoClient(MONGO_URI)[DB_NAME]


def get_observations_in_order(db):
    """取得按時間排序的 observations，用於 learning curve"""
    obs = list(db.observation_logs.find(
        {"action": {"$nin": list(NO_WEIGHT)}},
        {"user": 1, "action": 1, "zone_name": 1,
         "instance": 1, "weight": 1, "last_seen": 1}
    ))
    # 按時間排序
    obs.sort(key=lambda x: x.get("last_seen", datetime.datetime.min))
    return obs


def compute_habits_at_episode(obs_upto, fat):
    """計算到某個時間點為止的 habits"""
    agg = defaultdict(int)
    for d in obs_upto:
        key = (d.get("action", ""),
               d.get("zone_name") or d.get("instance", ""))
        agg[key] += d.get("weight", 0)
    
    habits = []
    for (act, inst), w in agg.items():
        if w >= fat:
            habits.append({"action": act, "instance": inst, "weight": w})
    return habits


def compute_ground_truth_habits(obs_all, min_weight=3):
    """用全部數據 + min_weight 定義 ground truth habits"""
    agg = defaultdict(int)
    for d in obs_all:
        key = (d.get("action", ""),
               d.get("zone_name") or d.get("instance", ""))
        agg[key] += d.get("weight", 0)
    
    habits = []
    for (act, inst), w in agg.items():
        if w >= min_weight:
            habits.append({"action": act, "instance": inst, "weight": w})
    return habits


def compute_metrics(habits, gt_habits, model):
    """計算 precision, recall, f1"""
    if not habits:
        return 0.0, 0.0, 0.0
    
    # 計算 redundancy (precision)
    if len(habits) < 2:
        redundancy = 0.0
    else:
        texts = [f"{h['action']} near {h['instance']}" for h in habits]
        vecs = model.encode(texts, normalize_embeddings=True)
        pairs = redundant = 0
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                pairs += 1
                if float(np.dot(vecs[i], vecs[j])) >= DEDUP_SIM:
                    redundant += 1
        redundancy = redundant / pairs if pairs > 0 else 0.0
    
    precision = 1.0 - redundancy
    
    # 計算 recall
    gt_keys = {f"{h['action']}@{h['instance']}" for h in gt_habits}
    habit_keys = {f"{h['action']}@{h['instance']}" for h in habits}
    recall = len(gt_keys & habit_keys) / len(gt_keys) if gt_keys else 0.0
    
    # 計算 F1
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    
    return precision, recall, f1


def plot_fig3_fat_sensitivity(db):
    print("Fig3: FAT Sensitivity...")

    obs = list(db.observation_logs.find(
        {"action": {"$nin": list(NO_WEIGHT)}},
        {"user": 1, "action": 1, "zone_name": 1,
         "instance": 1, "weight": 1}
    ))
    print(f"  observation_logs: {len(obs)}")

    if not obs:
        print("  No data. Run HabitExp first.")
        return

    if not _SBERT_OK:
        print("  SBERT not available.")
        return

    print("  Loading SBERT...")
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    fig, axes = plt.subplots(1, len(USERS), figsize=(6 * len(USERS), 5.5),
                              sharey=True)
    if len(USERS) == 1:
        axes = [axes]
    fig.suptitle(
        "Fig3  FAT Threshold Sensitivity\n"
        "Precision / Recall / F1 across FAT values; selected threshold = FAT=5",
        fontsize=12, fontweight="bold"
    )

    for ax, user_id in zip(axes, USERS):
        agg = defaultdict(int)
        for d in obs:
            if d.get("user") != user_id:
                continue
            key = (d.get("action", ""),
                   d.get("zone_name") or d.get("instance", ""))
            agg[key] += d.get("weight", 0)

        grouped = [
            {"action": act, "instance": inst, "weight": w}
            for (act, inst), w in sorted(agg.items(), key=lambda x: -x[1])
        ]
        gt_habits = [g for g in grouped if g["weight"] >= 3]

        results = []
        for fat in FAT_VALUES:
            habits = [g for g in grouped if g["weight"] >= fat]
            precision, recall, f1 = compute_metrics(habits, gt_habits, model)
            results.append({"fat": fat, "precision": precision,
                             "recall": recall, "f1": f1})

        x_    = list(range(len(FAT_VALUES)))
        precs = [r["precision"] for r in results]
        recs  = [r["recall"]    for r in results]
        f1s   = [r["f1"]        for r in results]

        if SELECTED_FAT in FAT_VALUES:
            ax.axvline(x=FAT_VALUES.index(SELECTED_FAT),
                       color="#E53935", linewidth=1.8,
                       linestyle="--", alpha=0.7,
                       label=f"FAT={SELECTED_FAT} (selected)")

        ax.plot(x_, recs,  "o-", color="#2196F3", linewidth=2.2,
                markersize=8, markerfacecolor="white",
                markeredgewidth=2, label="Recall")
        ax.plot(x_, precs, "s-", color="#FF9800", linewidth=2.2,
                markersize=8, markerfacecolor="white",
                markeredgewidth=2, label="Precision")
        ax.plot(x_, f1s,   "^-", color="#4CAF50", linewidth=2.5,
                markersize=9, markerfacecolor="white",
                markeredgewidth=2.5, label="F1")

        for i, (r, p, f) in enumerate(zip(recs, precs, f1s)):
            ax.text(i, r + 0.02, f"{r:.2f}", ha="center",
                    fontsize=8, color="#1565C0")
            ax.text(i, p - 0.06, f"{p:.2f}", ha="center",
                    fontsize=8, color="#E65100")

        ax.set_xticks(x_)
        ax.set_xticklabels([f"FAT={v}" for v in FAT_VALUES], fontsize=10)
        ax.set_ylim(0, 1.25)
        ax.set_xlabel("Fast Adaptation Threshold", fontsize=11)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_title(user_id.replace("_", " "), fontsize=12, fontweight="bold")
        ax.legend(loc="lower left", fontsize=9)
        ax.grid(True, alpha=0.2)

        fat5_idx = FAT_VALUES.index(SELECTED_FAT)
        print(f"  {user_id}  FAT={SELECTED_FAT}: "
              f"P={precs[fat5_idx]:.3f} R={recs[fat5_idx]:.3f} F1={f1s[fat5_idx]:.3f}")

    plt.tight_layout()
    path = os.path.join(OUT, "Fig3_fat_sensitivity.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_fig4_learning_curve(db):
    """新增：Learning Curve - 不同 FAT 下，F1 隨 observation 數量變化"""
    print("\nFig4: Learning Curve...")

    obs_all = get_observations_in_order(db)
    if not obs_all:
        print("  No observations found.")
        return

    if not _SBERT_OK:
        print("  SBERT not available.")
        return

    print("  Loading SBERT...")
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    # 計算 ground truth habits（用全部數據，weight >= 3）
    gt_habits_all = compute_ground_truth_habits(obs_all, min_weight=3)
    print(f"  Ground truth habits: {len(gt_habits_all)}")

    fig, axes = plt.subplots(1, len(USERS), figsize=(7 * len(USERS), 5.5))
    if len(USERS) == 1:
        axes = [axes]

    fig.suptitle(
        "Fig4  Learning Curve\n"
        "F1 Score vs Number of Observations for Different FAT Thresholds",
        fontsize=12, fontweight="bold"
    )

    colors = {2: "#1E88E5", 3: "#43A047", 5: "#FB8C00", 8: "#E53935", 10: "#8E24AA"}
    markers = {2: "o", 3: "s", 5: "^", 8: "D", 10: "v"}

    for ax, user_id in zip(axes, USERS):
        # 篩選該用戶的 observations，按時間排序
        user_obs = [o for o in obs_all if o.get("user") == user_id]
        
        if not user_obs:
            print(f"  No observations for {user_id}")
            continue

        # 計算該用戶的 ground truth habits
        gt_habits = compute_ground_truth_habits(user_obs, min_weight=3)
        print(f"  {user_id}: {len(user_obs)} obs, {len(gt_habits)} ground truth habits")

        # 對每個 FAT 值，計算 learning curve
        for fat in FAT_VALUES:
            f1_scores = []
            episodes = []
            
            for n in EPISODE_BINS:
                if n > len(user_obs):
                    break
                
                # 取前 n 個 observations
                obs_upto = user_obs[:n]
                habits = compute_habits_at_episode(obs_upto, fat)
                
                _, _, f1 = compute_metrics(habits, gt_habits, model)
                f1_scores.append(f1)
                episodes.append(n)
            
            if f1_scores:
                ax.plot(episodes, f1_scores, 
                       marker=markers.get(fat, "o"),
                       color=colors.get(fat, "#333"),
                       linewidth=2, markersize=6,
                       label=f"FAT={fat}")
                
                # 標記達到閾值的點
                for i, (ep, f1) in enumerate(zip(episodes, f1_scores)):
                    if f1 >= ACCURACY_THRESHOLD:
                        ax.scatter([ep], [f1], color=colors.get(fat, "#333"), 
                                  s=80, zorder=5, edgecolor="white", linewidth=1.5)
                        ax.text(ep, f1 + 0.03, f"{ep}eps", 
                               ha="center", fontsize=7, color=colors.get(fat, "#333"))
                        break

        ax.axhline(y=ACCURACY_THRESHOLD, color="gray", linestyle="--", 
                  alpha=0.5, label=f"{ACCURACY_THRESHOLD*100:.0f}% target")
        ax.set_xlabel("Number of Observations", fontsize=11)
        ax.set_ylabel("F1 Score", fontsize=11)
        ax.set_title(user_id.replace("_", " "), fontsize=12, fontweight="bold")
        ax.set_xlim(0, max(EPISODE_BINS) + 10)
        ax.set_ylim(0, 1.05)
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT, "Fig4_learning_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def print_learning_speed_summary(db):
    """輸出每個 FAT 達到目標準確率所需的 episodes"""
    print("\n" + "="*60)
    print("Learning Speed Summary (Episodes to reach 85% F1)")
    print("="*60)

    obs_all = get_observations_in_order(db)
    if not obs_all:
        return

    if not _SBERT_OK:
        print("  SBERT not available.")
        return

    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    for user_id in USERS:
        user_obs = [o for o in obs_all if o.get("user") == user_id]
        if not user_obs:
            continue
            
        gt_habits = compute_ground_truth_habits(user_obs, min_weight=3)
        
        print(f"\n{user_id}:")
        print(f"  {'FAT':<6} {'Episodes to 85%':<18} {'Final F1':<10}")
        print(f"  {'-'*35}")
        
        for fat in FAT_VALUES:
            episodes_needed = None
            final_f1 = 0.0
            
            for n in EPISODE_BINS:
                if n > len(user_obs):
                    break
                obs_upto = user_obs[:n]
                habits = compute_habits_at_episode(obs_upto, fat)
                _, _, f1 = compute_metrics(habits, gt_habits, model)
                final_f1 = f1
                
                if f1 >= ACCURACY_THRESHOLD and episodes_needed is None:
                    episodes_needed = n
            
            if episodes_needed:
                print(f"  {fat:<6} {episodes_needed:<18} {final_f1:.3f}")
            else:
                print(f"  {fat:<6} {'>200 (not reached)':<18} {final_f1:.3f}")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    db = connect()
    print(f"Connected → {DB_NAME}")
    
    plot_fig3_fat_sensitivity(db)
    plot_fig4_learning_curve(db)
    print_learning_speed_summary(db)
    
    print("\nDone.")