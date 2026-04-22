import json
import os
from datetime import datetime
from pymongo import MongoClient
from bson import ObjectId

MONGO_URI  = "mongodb://127.0.0.1:27017/"
DB_NAME    = "robot_rag_db"
BACKUP_DIR = "db_backups"
OUTPUT     = os.path.join(BACKUP_DIR, "halfway.json")

COLLECTIONS_PARTIAL = ["observation_logs"]
COLLECTIONS_FULL    = ["skill_chunks", "user_skills",
                       "dynamic_objects", "scene_snapshots"]
COLLECTIONS_SKIP    = ["manifold_points", "behavior_clusters", "intent_stats"]

HALFWAY_COUNT = 20


def serialize(obj):
    if isinstance(obj, ObjectId):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Not serializable: {type(obj)}")


def main():
    os.makedirs(BACKUP_DIR, exist_ok=True)
    client = MongoClient(MONGO_URI)
    db     = client[DB_NAME]

    print("Backing up HALFWAY state (first 20 observations, no manifold)...")
    data = {}

    for col in COLLECTIONS_PARTIAL:
        docs = list(db[col].find({}))[:HALFWAY_COUNT]
        data[col] = docs
        print(f"  [{col}] {len(docs)} records (capped at {HALFWAY_COUNT})")

    for col in COLLECTIONS_FULL:
        docs = list(db[col].find({}))
        data[col] = docs
        print(f"  [{col}] {len(docs)} records")

    for col in COLLECTIONS_SKIP:
        data[col] = []
        print(f"  [{col}] skipped (no manifold in halfway state)")

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(data, f, default=serialize,
                  ensure_ascii=False, indent=2)

    size_mb = os.path.getsize(OUTPUT) / 1024 / 1024
    print(f"\nSaved: {OUTPUT} ({size_mb:.1f} MB)")
    print("Done.")


if __name__ == "__main__":
    main()