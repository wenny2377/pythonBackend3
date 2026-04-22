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
    "eval_logs",
    "observation_logs",
    "exp_checkpoint_logs",
    "activity_sequences",
    "conversation_logs",
    "raw_objects",
    "dynamic_objects",
    "scene_snapshots",
    "semantic_memories",
    "navigation_logs",
    "manifold_points",
    "behavior_clusters",
    "service_proposals",
    "intent_stats",
    "skill_chunks",
    "episodic_summaries",
]

FAISS_FILES = [
    "robot_memory.index",
    "robot_memory_meta.json",
    "dynamic_memory.index",
    "dynamic_memory_meta.json",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--keep-scene", action="store_true",
        help="Keep scene_snapshots (furniture positions, skip re-scan)"
    )
    args = parser.parse_args()

    client = MongoClient(MONGO_URI)
    db     = client[DB_NAME]

    print(f"\n{'='*50}")
    print(f"  Reset Script  |  DB: {DB_NAME}")
    if args.keep_scene:
        print(f"  Mode: keep scene_snapshots")
    print(f"{'='*50}\n")

    print("Record counts before reset:")
    for col in sorted(db.list_collection_names()):
        n = db[col].count_documents({})
        if n > 0:
            print(f"  [{col}] {n} records")

    print("\nClearing collections...")
    to_clear = [
        c for c in COLLECTIONS_TO_CLEAR
        if not (args.keep_scene and c == "scene_snapshots")
    ]
    for col in to_clear:
        n = db[col].delete_many({}).deleted_count
        if n > 0:
            print(f"  [{col}] deleted {n}")
        else:
            print(f"  [{col}] already empty")

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
        print(f"  [{user_id}] reset to v1")

    print("\nRemoving FAISS index files...")
    for path in FAISS_FILES:
        if os.path.exists(path):
            os.remove(path)
            print(f"  [FAISS] removed {path}")
        else:
            print(f"  [FAISS] not found (skip): {path}")

    if os.path.exists("debug_images"):
        shutil.rmtree("debug_images")
        print("\n  [debug_images] removed")

    print(f"\n{'='*50}")
    print(f"  Done. Next: python3 app.py")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()