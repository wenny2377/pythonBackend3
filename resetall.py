"""
resetall.py
Complete reset for RobotBrain experiments.

Usage:
  python3 resetall.py                        # full reset
  python3 resetall.py --keep-scene           # keep furniture positions
  python3 resetall.py --keep-affinity        # keep affinity matrix
  python3 resetall.py --keep-scene --keep-affinity
"""

import os
import shutil
import argparse
from datetime import datetime
from pymongo import MongoClient

MONGO_URI = "mongodb://127.0.0.1:27017/"
DB_NAME   = "robot_rag_db"

CLEAN_SKILL_TEMPLATE = """# {user_id} Skill Profile
*Version 1 | Updated: {date}*

## Behavior Patterns
<!-- Observed: action + location + frequency only -->

## Preferences
<!-- No confirmed preferences yet -->

## How to Handle Requests
- Check object availability before recommending
- If requested item is unavailable, suggest nearest alternative

## What NOT to do
- Do not invent object locations
- Do not recommend items not in the environment snapshot
"""

SKILL_USERS = ["User_Mom", "User_Dad"]

COLLECTIONS_TO_CLEAR = [
    # 舊版資料（必須清除）
    "robot_memory",
    # Evaluation
    "eval_logs",
    "exp_checkpoint_logs",
    "exp_checkpoints",
    # Habit learning（新版）
    "observation_logs",
    "habit_snapshots",
    "activity_sequences",
    # Manifold
    "manifold_training_data",
    "manifold_points",
    # Affinity（動態部分）
    "affinity_history",
    "user_spatial_affinity",
    # Scene
    "dynamic_objects",
    "raw_objects",
    "scene_snapshots",
    # Memory
    "semantic_memories",
    "conversation_logs",
    "behavior_clusters",
    # Service
    "service_proposals",
    "service_results",
    "intent_stats",
    # Skill chunks
    "skill_chunks",
    "episodic_summaries",
    # SayCan
    "saycan_logs",
    "saycan_behavior_objects",
    # Navigation
    "navigation_logs",
    "user_positions",
    # SayCan behavior objects（讓系統重新蒸餾）
    "saycan_behavior_objects",
]

AFFINITY_COLLECTIONS = ["affinity_matrix"]

FAISS_FILES = [
    "robot_memory.index",   "robot_memory_meta.json",
    "dynamic_memory.index", "dynamic_memory_meta.json",
    "skill_chunks.index",   "skill_chunks_meta.json",
    # 新版 FAISS 路徑
    "faiss_habit.index",    "faiss_habit_meta.json",
    "faiss_dynamic.index",  "faiss_dynamic_meta.json",
]

MANIFOLD_MODEL_DIR = "manifold_models"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep-scene",    action="store_true",
                        help="Keep scene_snapshots (furniture positions)")
    parser.add_argument("--keep-affinity", action="store_true",
                        help="Keep affinity_matrix (saves ~60s rebuild)")
    args = parser.parse_args()

    client = MongoClient(MONGO_URI)
    db     = client[DB_NAME]

    print(f"\n{'='*55}")
    print(f"  RobotBrain Reset  |  DB: {DB_NAME}")
    if args.keep_scene:    print("  --keep-scene:    scene_snapshots preserved")
    if args.keep_affinity: print("  --keep-affinity: affinity_matrix preserved")
    print(f"{'='*55}\n")

    print("Before reset:")
    for col in sorted(db.list_collection_names()):
        n = db[col].count_documents({})
        if n > 0:
            print(f"  {n:5} | {col}")

    to_clear = list(COLLECTIONS_TO_CLEAR)
    if args.keep_scene:
        to_clear = [c for c in to_clear if c != "scene_snapshots"]
    if not args.keep_affinity:
        to_clear += AFFINITY_COLLECTIONS

    print("\nClearing collections...")
    total = 0
    for col in to_clear:
        try:
            n = db[col].delete_many({}).deleted_count
            total += n
            if n > 0:
                print(f"  [{col}] deleted {n}")
        except Exception as e:
            print(f"  [{col}] ERROR: {e}")
    print(f"  Total deleted: {total}")

    print("\nResetting SKILL.md...")
    date = datetime.now().strftime("%Y-%m-%d")
    for uid in SKILL_USERS:
        db.user_skills.update_one(
            {"user_id": uid},
            {"$set": {
                "skill_md":   CLEAN_SKILL_TEMPLATE.format(
                    user_id=uid, date=date),
                "version":    1,
                "updated_at": datetime.utcnow(),
                "is_stale":   False,
            }},
            upsert=True,
        )
        print(f"  [{uid}] skill reset")

    print("\nRemoving FAISS index files...")
    for path in FAISS_FILES:
        if os.path.exists(path):
            os.remove(path)
            print(f"  removed {path}")
        else:
            # 也檢查 data/ 子目錄
            alt = os.path.join("data", path)
            if os.path.exists(alt):
                os.remove(alt)
                print(f"  removed {alt}")

    print("\nRemoving Manifold models...")
    if os.path.exists(MANIFOLD_MODEL_DIR):
        removed = 0
        for f in os.listdir(MANIFOLD_MODEL_DIR):
            if f.endswith(".pkl"):
                os.remove(os.path.join(MANIFOLD_MODEL_DIR, f))
                removed += 1
        print(f"  removed {removed} .pkl model files")
    else:
        print(f"  {MANIFOLD_MODEL_DIR}/ not found, skipping")

    if os.path.exists("debug_images"):
        shutil.rmtree("debug_images")
        print("  removed debug_images/")

    print(f"\nAfter reset:")
    for col in sorted(db.list_collection_names()):
        n = db[col].count_documents({})
        if n > 0:
            print(f"  {n:5} | {col}")

    print(f"\n{'='*55}")
    print("  Reset complete.")
    print("  Next steps:")
    print("    1. python3 app.py")
    print("    2. Press Play in Unity (HabitExp mode)")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()