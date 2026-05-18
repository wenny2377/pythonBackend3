"""
resetall.py
Complete reset script for RobotBrain experiments.
Clears ALL experiment-related collections before each run.

Usage:
  python3 resetall.py              # full reset
  python3 resetall.py --keep-scene # keep scene_snapshots (skip re-scan)
  python3 resetall.py --keep-affinity # keep Gemma affinity_matrix (slow to rebuild)
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

# All collections that must be cleared between experiments
COLLECTIONS_TO_CLEAR = [
    # Experiment evaluation
    "eval_logs",
    "exp_checkpoint_logs",
    "exp_checkpoints",

    # Habit learning
    "observation_logs",
    "habit_snapshots",
    "activity_sequences",

    # Manifold engine
    "manifold_training_data",
    "manifold_points",

    # Affinity (conditionally kept)
    "affinity_history",
    "user_spatial_affinity",

    # Scene and objects
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

    # Skill
    "skill_chunks",
    "episodic_summaries",

    # SayCan
    "saycan_logs",
    "saycan_behavior_objects",

    # Navigation
    "navigation_logs",
    "user_positions",
]

# These are slow to rebuild — offer option to keep
AFFINITY_COLLECTIONS = [
    "affinity_matrix",       # Gemma distillation (~60s to rebuild)
]

FAISS_FILES = [
    "robot_memory.index",
    "robot_memory_meta.json",
    "dynamic_memory.index",
    "dynamic_memory_meta.json",
    "skill_chunks.index",
    "skill_chunks_meta.json",
]

MANIFOLD_MODEL_DIR = "manifold_models"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--keep-scene", action="store_true",
        help="Keep scene_snapshots (furniture positions, skip re-scan)")
    parser.add_argument(
        "--keep-affinity", action="store_true",
        help="Keep affinity_matrix (Gemma distillation, ~60s to rebuild)")
    args = parser.parse_args()

    client = MongoClient(MONGO_URI)
    db     = client[DB_NAME]

    print(f"\n{'='*55}")
    print(f"  RobotBrain Reset  |  DB: {DB_NAME}")
    if args.keep_scene:
        print(f"  --keep-scene: scene_snapshots preserved")
    if args.keep_affinity:
        print(f"  --keep-affinity: affinity_matrix preserved")
    print(f"{'='*55}\n")

    # Show current state
    print("Record counts BEFORE reset:")
    for col in sorted(db.list_collection_names()):
        n = db[col].count_documents({})
        if n > 0:
            print(f"  {n:5} | {col}")

    # Build clear list
    to_clear = list(COLLECTIONS_TO_CLEAR)
    if args.keep_scene:
        to_clear = [c for c in to_clear if c != "scene_snapshots"]
    if not args.keep_affinity:
        to_clear += AFFINITY_COLLECTIONS

    print("\nClearing collections...")
    total_deleted = 0
    for col in to_clear:
        try:
            n = db[col].delete_many({}).deleted_count
            total_deleted += n
            status = f"deleted {n}" if n > 0 else "already empty"
            print(f"  [{col}] {status}")
        except Exception as e:
            print(f"  [{col}] ERROR: {e}")

    print(f"\n  Total records deleted: {total_deleted}")

    # Reset SKILL.md
    print("\nResetting SKILL.md...")
    date = datetime.now().strftime("%Y-%m-%d")
    for user_id in SKILL_USERS:
        clean = CLEAN_SKILL_TEMPLATE.format(
            user_id=user_id, date=date)
        db.user_skills.update_one(
            {"user_id": user_id},
            {"$set": {
                "skill_md":   clean,
                "version":    1,
                "updated_at": datetime.utcnow(),
                "is_stale":   False,
            }},
            upsert=True,
        )
        print(f"  [{user_id}] SKILL.md reset to v1")

    # Remove FAISS index files
    print("\nRemoving FAISS index files...")
    for path in FAISS_FILES:
        if os.path.exists(path):
            os.remove(path)
            print(f"  [FAISS] removed {path}")
        else:
            print(f"  [FAISS] not found (skip): {path}")

    # Remove manifold model files
    print("\nRemoving ManifoldEngine models...")
    if os.path.exists(MANIFOLD_MODEL_DIR):
        for fname in os.listdir(MANIFOLD_MODEL_DIR):
            if fname.endswith(".pkl"):
                path = os.path.join(MANIFOLD_MODEL_DIR, fname)
                os.remove(path)
                print(f"  [Manifold] removed {path}")
    else:
        print(f"  [Manifold] directory not found (skip)")

    # Remove debug images
    if os.path.exists("debug_images"):
        shutil.rmtree("debug_images")
        print("\n  [debug_images] removed")

    # Final state
    print(f"\nRecord counts AFTER reset:")
    for col in sorted(db.list_collection_names()):
        n = db[col].count_documents({})
        if n > 0:
            print(f"  {n:5} | {col}")

    print(f"\n{'='*55}")
    print(f"  Done.")
    print(f"  Next: python3 app.py")
    if args.keep_affinity:
        print(f"  Note: affinity_matrix kept "
              f"({db.affinity_matrix.count_documents({})} entries)")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()