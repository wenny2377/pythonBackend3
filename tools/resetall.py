import os
import shutil
from datetime import datetime
from pymongo import MongoClient


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
    "affinity_history", "user_spatial_affinity", "dynamic_objects",
    "raw_objects", "semantic_memories", "conversation_logs",
    "service_proposals", "service_results", "intent_stats",
    "skill_chunks", "episodic_summaries", "saycan_logs",
    "navigation_logs", "user_positions", "object_events",
    "device_states",
]

KEEP_ALWAYS = [
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
    for path in FAISS_FILES:
        for base in [".", "data"]:
            full = os.path.join(base, path)
            if os.path.exists(full):
                os.remove(full)
                print(f"  removed {full}")

    if os.path.exists("manifold_models"):
        removed = sum(
            1 for f in os.listdir("manifold_models")
            if f.endswith(".pkl") and
            not os.remove(os.path.join("manifold_models", f))
        )
        if removed:
            print(f"  removed {removed} manifold models")

    if os.path.exists("debug_images"):
        shutil.rmtree("debug_images")
        print("  removed debug_images/")

    print(f"\n  Done. DB={DB_NAME} is clean.")
    print(f"  Next: python3 app.py\n")


if __name__ == "__main__":
    main()