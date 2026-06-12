import os
import shutil
import argparse
from datetime import datetime
from pymongo import MongoClient

MONGO_URI = "mongodb://127.0.0.1:27017/"
DB_NAME   = os.environ.get("DB_NAME", "robot_rag_db")

CLEAN_SKILL_TEMPLATE = """# {user_id} Skill Profile
*Version 1 | Updated: {date}*

## Behavior Patterns

## Preferences

## How to Handle Requests
- Check object availability before recommending
- If requested item is unavailable, suggest nearest alternative

## What NOT to do
- Do not invent object locations
- Do not recommend items not in the environment snapshot
"""

SKILL_USERS = ["User_Mom", "User_Dad"]

COLLECTIONS_TO_CLEAR = [
    "robot_memory", "eval_logs", "exp_checkpoint_logs",
    "observation_logs", "habit_snapshots", "activity_sequences",
    "transition_counts", "manifold_training_data", "manifold_points",
    "affinity_history", "user_spatial_affinity", "dynamic_objects",
    "raw_objects", "semantic_memories", "conversation_logs",
    "service_proposals", "service_results", "intent_stats",
    "skill_chunks", "episodic_summaries", "saycan_logs",
    "navigation_logs", "user_positions", "object_events",
]

KEEP_COLLECTIONS = [
    "scene_snapshots",
    "transition_matrix",
    "charades_affinity",
    "charades_affinity_normalized",
]

FAISS_FILES = [
    "robot_memory.index",   "robot_memory_meta.json",
    "dynamic_memory.index", "dynamic_memory_meta.json",
    "skill_chunks.index",   "skill_chunks_meta.json",
]

MANIFOLD_MODEL_DIR = "manifold_models"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keep-scene",    action="store_true",
                        help="Keep scene_snapshots (furniture layout)")
    parser.add_argument("--keep-charades", action="store_true",
                        help="Keep transition_matrix from Charades")
    args = parser.parse_args()

    client = MongoClient(MONGO_URI)
    db     = client[DB_NAME]

    print(f"\n{'='*50}")
    print(f"  Reset: {DB_NAME}")
    if args.keep_scene:    print("  --keep-scene:    scene_snapshots preserved")
    if args.keep_charades: print("  --keep-charades: transition_matrix preserved")
    print(f"{'='*50}\n")

    to_clear = list(COLLECTIONS_TO_CLEAR)

    if not args.keep_scene:
        to_clear.append("scene_snapshots")
    if not args.keep_charades:
        to_clear += ["transition_matrix", "charades_affinity",
                     "charades_affinity_normalized"]

    print("Clearing collections...")
    total = 0
    for col in to_clear:
        try:
            n = db[col].delete_many({}).deleted_count
            total += n
            if n > 0:
                print(f"  [{col}] deleted {n}")
        except Exception as e:
            print(f"  [{col}] error: {e}")
    print(f"  Total deleted: {total}")

    print("\nResetting SKILL.md...")
    date = datetime.now().strftime("%Y-%m-%d")
    for uid in SKILL_USERS:
        db.user_skills.update_one(
            {"user_id": uid},
            {"$set": {
                "skill_md":   CLEAN_SKILL_TEMPLATE.format(user_id=uid, date=date),
                "version":    1,
                "updated_at": datetime.utcnow(),
                "is_stale":   False,
            }},
            upsert=True,
        )
        print(f"  [{uid}] reset")

    print("\nRemoving FAISS files...")
    for path in FAISS_FILES:
        for base in [".", "data"]:
            full = os.path.join(base, path)
            if os.path.exists(full):
                os.remove(full)
                print(f"  removed {full}")

    print("\nRemoving Manifold models...")
    if os.path.exists(MANIFOLD_MODEL_DIR):
        removed = 0
        for f in os.listdir(MANIFOLD_MODEL_DIR):
            if f.endswith(".pkl"):
                os.remove(os.path.join(MANIFOLD_MODEL_DIR, f))
                removed += 1
        print(f"  removed {removed} model files")

    if os.path.exists("debug_images"):
        shutil.rmtree("debug_images")
        print("  removed debug_images/")

    print(f"\n{'='*50}")
    print(f"  Reset complete: {DB_NAME}")
    print(f"  Next: bash run.sh baseline (or corruption/demo)")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()