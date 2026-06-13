import os
import sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from pymongo import MongoClient

MONGO_URI = "mongodb://127.0.0.1:27017/"
DBS       = ["robot_exp_baseline", "robot_exp_corruption", "robot_exp_demo"]

def main():
    client = MongoClient(MONGO_URI)

    for db_name in DBS:
        db = client[db_name]
        if db_name not in client.list_database_names():
            print(f"  {db_name}: not found, skip")
            continue

        print(f"\n=== {db_name} ===")

        r = db.dynamic_objects.update_many(
            {"label": "bottle"}, {"$set": {"label": "water"}})
        print(f"  dynamic_objects: {r.modified_count} updated")

        r = db.object_events.update_many(
            {"object": {"$in": ["bottle","waterbottle","water bottle"]}},
            {"$set": {"object": "water"}})
        print(f"  object_events:   {r.modified_count} updated")

        r = db.observation_logs.update_many(
            {"interacting_items": {"$in": ["bottle","waterbottle"]}},
            {"$set": {"interacting_items.$[el]": "water"}},
            array_filters=[{"el": {"$in": ["bottle","waterbottle"]}}])
        print(f"  observation_logs: {r.modified_count} updated")

        for uid in ["User_Mom","User_Dad"]:
            doc = db.user_skills.find_one({"user_id":uid})
            if not doc:
                continue
            new_md = doc["skill_md"].replace("bottle","water")
            if new_md != doc["skill_md"]:
                db.user_skills.update_one(
                    {"user_id":uid}, {"$set":{"skill_md":new_md}})
                print(f"  user_skills {uid}: bottle → water")

    print("\nDone. Unity GameObject should also be renamed to 'water'.")

if __name__ == "__main__":
    main()