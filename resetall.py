import os
import shutil
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

client = MongoClient(MONGO_URI)
db     = client[DB_NAME]

print(f"\nReset script — target DB: {DB_NAME}\n")

print("Record counts before reset:")
for col in db.list_collection_names():
    print(f"  [{col}] {db[col].count_documents({})} records")

print("\nClearing collections...")
for col in COLLECTIONS_TO_CLEAR:
    n = db[col].delete_many({}).deleted_count
    print(f"  [{col}] deleted {n} records")

print("\nResetting SKILL.md for all users...")
date = datetime.now().strftime("%Y-%m-%d")
for user_id in SKILL_USERS:
    clean = CLEAN_SKILL_TEMPLATE.format(user_id=user_id, date=date)
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
    print(f"  [{user_id}] SKILL.md reset to version 1")

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

print("\nDone. Restart Flask: python3 app.py\n")