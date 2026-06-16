import os
import shutil
from datetime import datetime
from pymongo import MongoClient

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _ask_db() -> str:
    print("\nWhich DB to reset?")
    print("  1) Baseline   (robot_exp_baseline)")
    print("  2) Corruption (robot_exp_corruption)")
    try:
        choice = input("Choice [1]: ").strip() or "1"
    except EOFError:
        choice = "1"
    return {
        "1": "robot_exp_baseline",
        "2": "robot_exp_corruption",
    }.get(choice, "robot_exp_baseline")


MONGO_URI = "mongodb://127.0.0.1:27017/"
DB_NAME   = os.environ.get("DB_NAME") or _ask_db()

SKILL_USERS = ["User_Mom", "User_Dad"]

CLEAN_SKILL = """# {user_id} Skill Profile
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

COLLECTIONS_TO_CLEAR = [
    "robot_memory", "eval_logs", "exp_checkpoint_logs",
    "observation_logs", "habit_snapshots", "activity_sequences",
    "transition_counts", "manifold_training_data", "manifold_points",
    "affinity_history", "user_spatial_affinity", "affinity_matrix",
    "dynamic_objects", "raw_objects", "semantic_memories",
    "conversation_logs", "proposals", "service_proposals",
    "service_results", "intent_stats", "skill_chunks",
    "episodic_summaries", "saycan_logs", "navigation_logs",
    "user_positions", "object_events", "device_states",
    "system_config",
]

FAISS_FILES = [
    "robot_memory.index",   "robot_memory_meta.json",
    "dynamic_memory.index", "dynamic_memory_meta.json",
    "skill_chunks.index",   "skill_chunks_meta.json",
]


def main():
    client = MongoClient(MONGO_URI)
    db     = client[DB_NAME]

    print(f"\n{'='*45}")
    print(f"  Reset: {DB_NAME}")
    print(f"{'='*45}\n")

    print("Clearing collections...")
    total = 0
    for col in COLLECTIONS_TO_CLEAR:
        n = db[col].delete_many({}).deleted_count
        total += n
        if n > 0:
            print(f"  {col}: {n} deleted")
    print(f"  Total: {total} records deleted")

    print("\nResetting SKILL.md...")
    date = datetime.now().strftime("%Y-%m-%d")
    for uid in SKILL_USERS:
        db.user_skills.update_one(
            {"user_id": uid},
            {"$set": {
                "skill_md":   CLEAN_SKILL.format(user_id=uid, date=date),
                "version":    1,
                "updated_at": datetime.utcnow(),
                "is_stale":   False,
            }},
            upsert=True,
        )
        print(f"  {uid} reset")

    print("\nRemoving FAISS files...")
    for fname in FAISS_FILES:
        full = os.path.join(_ROOT, fname)
        if os.path.exists(full):
            os.remove(full)
            print(f"  removed {full}")

    print("\nRemoving manifold models...")
    manifold_dir = os.path.join(_ROOT, "modules", "manifold_models")
    if os.path.exists(manifold_dir):
        removed = 0
        for f in os.listdir(manifold_dir):
            if f.endswith(".pkl"):
                os.remove(os.path.join(manifold_dir, f))
                removed += 1
        if removed:
            print(f"  removed {removed} manifold .pkl files")

    debug_dir = os.path.join(_ROOT, "debug_images")
    if os.path.exists(debug_dir):
        shutil.rmtree(debug_dir)
        print(f"  removed debug_images/")

    print(f"\n  Done. DB={DB_NAME} is clean.")
    print(f"  Next: python3 app.py\n")


if __name__ == "__main__":
    main()