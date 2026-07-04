import os
import sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from pymongo import MongoClient

DEMO_DB   = "robot_exp_demo"
MONGO_URI = "mongodb://127.0.0.1:27017/"

ZONE_MAP = {
    "Cooking_Zone":   "stove_Zone",
    "Opening_Zone_2": "refrigerator_Zone",
    "Opening_Zone_3": "cabinet_Zone",
    "Sitting_Zone":   "sofa_Zone",
    "Sitting_Zone_2": "sofa side_Zone",
    "Sitting_Zone_4": "sofa side 2_Zone",
    "Typing_Zone":    "monitor_Zone",
}

COLLECTIONS = [
    "observation_logs",
    "behavior_patterns",
    "user_spatial_affinity",
    "transition_counts",
    "activity_sequences",
]

def migrate(db):
    for col_name in COLLECTIONS:
        col   = db[col_name]
        total = 0
        for old, new in ZONE_MAP.items():
            result = col.update_many(
                {"zone_name": old},
                {"$set": {"zone_name": new}}
            )
            if result.modified_count > 0:
                print(f"  [{col_name}] {old} → {new}: {result.modified_count} docs")
                total += result.modified_count
        if total == 0:
            print(f"  [{col_name}] no changes")
        print()

def main():
    client = MongoClient(MONGO_URI)
    db     = client[DEMO_DB]

    if not db.list_collection_names():
        print(f"[!] {DEMO_DB} does not exist. Run setup_demo_db.py first.")
        client.close()
        return

    print(f"Migrating zone names in {DEMO_DB}")
    print("=" * 50)
    migrate(db)
    print("=" * 50)
    print("Done.")
    client.close()

if __name__ == "__main__":
    main()