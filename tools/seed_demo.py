import os
import sys
import datetime
from pymongo import MongoClient

MONGO_URI = "mongodb://127.0.0.1:27017/"
DB_NAME   = os.environ.get("DB_NAME", "robot_demo")

client = MongoClient(MONGO_URI)
db     = client[DB_NAME]

USERS = ["User_Mom", "User_Dad"]

CLEAN_SKILL = """# {user_id} Skill Profile
*Version 3 | Updated: {date}*

## Behavior Patterns
- Eating near dining_table in Evening (42 times)
- Watching near sofa in Evening (38 times)
- SittingDrink near sofa in Evening (31 times)
- Reading near sofa in Morning (18 times)

## Preferences
- User enjoys cola during Watching in Evening
- User prefers cola over juice

## How to Handle Requests
- Check object availability before recommending
- If requested item is unavailable, suggest nearest alternative

## What NOT to do
- Do not invent object locations
- Do not recommend items not in the environment snapshot
"""

SKILL_MOM = """# User_Mom Skill Profile
*Version 4 | Updated: {date}*

## Behavior Patterns
- Eating near dining_table in Evening (42 times)
- Watching near sofa in Evening (38 times)
- SittingDrink near sofa in Evening (31 times)
- Reading near sofa in Morning (18 times)
- Typing near desk in Morning (12 times)

## Preferences
- User enjoys cola during Watching in Evening
- User prefers to drink cola while watching TV

## How to Handle Requests
- Check object availability before recommending
- If requested item is unavailable, suggest nearest alternative

## What NOT to do
- Do not invent object locations
- Do not recommend items not in the environment snapshot
"""


def clear_and_seed():
    print(f"Seeding demo database: {DB_NAME}")

    collections_to_clear = [
        "transition_counts", "observation_logs", "habit_snapshots",
        "activity_sequences", "dynamic_objects", "user_skills",
        "service_proposals", "eval_logs",
    ]
    for col in collections_to_clear:
        db[col].delete_many({})
    print("  Cleared existing data")

    # ── 1. Transition counts ──────────────────────────────────────────────────
    now = datetime.datetime.utcnow()
    transitions = [
        ("User_Mom", "Eating",      "Watching",     "Evening", 42, 38.5),
        ("User_Mom", "Watching",    "SittingDrink", "Evening", 31, 28.2),
        ("User_Mom", "SittingDrink","Watching",     "Evening", 18, 16.4),
        ("User_Mom", "Watching",    "Laying",       "Night",   22, 20.1),
        ("User_Mom", "Reading",     "Sitting",      "Morning", 15, 13.8),
        ("User_Mom", "Sitting",     "Typing",       "Morning", 12, 11.0),
        ("User_Mom", "Eating",      "Cleaning",     "Morning", 8,  7.3),
        ("User_Dad", "Eating",      "Watching",     "Evening", 35, 32.0),
        ("User_Dad", "Watching",    "SittingDrink", "Evening", 28, 25.5),
        ("User_Dad", "Sitting",     "Reading",      "Evening", 20, 18.2),
        ("User_Dad", "Reading",     "Laying",       "Night",   16, 14.6),
    ]
    db.transition_counts.insert_many([
        {
            "user_id":     t[0],
            "from_action": t[1],
            "to_action":   t[2],
            "time_slot":   t[3],
            "count":       t[4],
            "weight":      t[5],
            "last_updated": now,
            "created_at":  now,
        }
        for t in transitions
    ])
    print(f"  Inserted {len(transitions)} transition_counts")

    # ── 2. Dynamic objects (available items for reactive service) ─────────────
    objects = [
        {"label": "cola",          "category": "drink", "room": "living_room",
         "last_seen_on": "dining_table", "interact_count": 28, "seen_count": 45},
        {"label": "juice",         "category": "drink", "room": "living_room",
         "last_seen_on": "dining_table", "interact_count": 12, "seen_count": 22},
        {"label": "water bottle",  "category": "drink", "room": "kitchen",
         "last_seen_on": "kitchen_counter", "interact_count": 8, "seen_count": 18},
        {"label": "apple",         "category": "food",  "room": "kitchen",
         "last_seen_on": "kitchen_counter", "interact_count": 5, "seen_count": 10},
        {"label": "sandwich",      "category": "food",  "room": "kitchen",
         "last_seen_on": "dining_table", "interact_count": 3, "seen_count": 6},
        {"label": "remote",        "category": "device", "room": "living_room",
         "last_seen_on": "sofa", "interact_count": 35, "seen_count": 55},
        {"label": "phone",         "category": "device", "room": "living_room",
         "last_seen_on": "sofa", "interact_count": 20, "seen_count": 40},
    ]
    for obj in objects:
        obj["last_seen"]   = now
        obj["first_seen"]  = now - datetime.timedelta(days=14)
        obj["source"]      = "unity"
        obj["status"]      = "active"
        obj["sensor_pos"]  = [3.5, 2.1]
        obj["spatial_rel"] = "on"
        obj["held_by"]     = ""
    db.dynamic_objects.insert_many(objects)
    print(f"  Inserted {len(objects)} dynamic_objects")

    # ── 3. Scene snapshots (furniture) ────────────────────────────────────────
    furniture = [
        {"label": "sofa",            "pos": [2.0, 1.5], "room": "living_room"},
        {"label": "dining_table",    "pos": [5.0, 3.0], "room": "living_room"},
        {"label": "desk",            "pos": [8.0, 2.0], "room": "study"},
        {"label": "bed",             "pos": [3.0, 8.0], "room": "bedroom"},
        {"label": "kitchen_counter", "pos": [1.0, 6.0], "room": "kitchen"},
        {"label": "tv_stand",        "pos": [0.5, 1.5], "room": "living_room"},
    ]
    for f in furniture:
        f["x"] = f["pos"][0]
        f["y"] = 0.0
        f["z"] = f["pos"][1]
        f["is_static"]    = True
        f["source"]       = "sensor"
        f["last_updated"] = now
    db.scene_snapshots.insert_many(furniture)
    print(f"  Inserted {len(furniture)} scene_snapshots")

    # ── 4. User skills (SKILL.md) ─────────────────────────────────────────────
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    db.user_skills.update_one(
        {"user_id": "User_Mom"},
        {"$set": {
            "skill_md":   SKILL_MOM.format(date=date_str),
            "version":    4,
            "updated_at": now,
            "last_used":  now,
            "is_stale":   False,
        }},
        upsert=True,
    )
    db.user_skills.update_one(
        {"user_id": "User_Dad"},
        {"$set": {
            "skill_md":   CLEAN_SKILL.format(user_id="User_Dad", date=date_str),
            "version":    2,
            "updated_at": now,
            "last_used":  now,
            "is_stale":   False,
        }},
        upsert=True,
    )
    print("  Inserted user_skills for User_Mom + User_Dad")

    # ── 5. Observation logs (habit weights) ───────────────────────────────────
    obs = [
        ("User_Mom", "Watching",    "sofa",         "Evening", 38),
        ("User_Mom", "SittingDrink","sofa",         "Evening", 31),
        ("User_Mom", "Eating",      "dining_table", "Evening", 42),
        ("User_Mom", "Reading",     "sofa",         "Morning", 18),
        ("User_Mom", "Typing",      "desk",         "Morning", 12),
        ("User_Dad", "Watching",    "sofa",         "Evening", 35),
        ("User_Dad", "SittingDrink","sofa",         "Evening", 28),
        ("User_Dad", "Eating",      "dining_table", "Evening", 38),
        ("User_Dad", "Reading",     "sofa",         "Evening", 20),
    ]
    db.observation_logs.insert_many([
        {
            "user":              o[0],
            "action":            o[1],
            "zone_name":         o[2],
            "instance":          o[2],
            "time_slot":         o[3],
            "weight":            o[4],
            "interacting_items": [],
            "room":              "living_room",
            "last_seen":         now,
            "last_date":         date_str,
        }
        for o in obs
    ])
    print(f"  Inserted {len(obs)} observation_logs")

    # ── 6. Charades pipeline (reuse if exists, else skip) ─────────────────────
    n_charades = db.transition_matrix.count_documents({})
    if n_charades == 0:
        print("  [warn] transition_matrix empty — run charades_pipeline.py first")
    else:
        print(f"  transition_matrix: {n_charades} records (already present)")

    print(f"\nDemo database ready: {DB_NAME}")
    print("Start demo backend:")
    print(f"  DB_NAME={DB_NAME} python3 app.py")


if __name__ == "__main__":
    clear_and_seed()