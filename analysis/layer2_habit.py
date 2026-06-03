"""
analysis/layer2_habit.py
Layer 2: Habit Learning Analysis
Outputs:
  results/Fig3_fat_sensitivity.png
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from pymongo import MongoClient

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

def connect():
    return MongoClient(MONGO_URI)[DB_NAME]


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

            if len(habits) < 2:
                redundancy = 0.0
            else:
                texts = [f"{h['action']} near {h['instance']}"
                         for h in habits]
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


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    db = connect()
    print(f"Connected → {DB_NAME}")
    plot_fig3_fat_sensitivity(db)
    print("Done.")