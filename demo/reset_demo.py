import os
import sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from pymongo import MongoClient
from collections import defaultdict
from datetime import datetime

DEMO_DB   = "robot_exp_demo"
MONGO_URI = "mongodb://127.0.0.1:27017/"
USERS     = ["User_Mom", "User_Dad"]

DRINK_ACTIONS = {"Drinking", "SittingDrink"}
FOOD_ACTIONS  = {"Eating", "Cooking"}

RESET_COLLECTIONS = [
    "user_skills",
    "skill_chunks",
    "service_proposals",
    "conversation_logs",
]

def _get_top_item(db, user_id, actions, category):
    obs = list(db.observation_logs.find(
        {"user": user_id, "action": {"$in": list(actions)}},
        {"zone_name": 1, "weight": 1}
    ).sort("weight", -1).limit(5))

    zone_counts = defaultdict(float)
    for d in obs:
        zone_counts[d.get("zone_name", "")] += d.get("weight", 1)

    top_zone = max(zone_counts, key=zone_counts.get) if zone_counts else ""

    obj = db.dynamic_objects.find_one(
        {"category": category, "last_seen_on": top_zone},
        sort=[("interact_count", -1)]
    )
    if obj:
        return obj["label"]

    obj = db.dynamic_objects.find_one(
        {"category": category},
        sort=[("interact_count", -1)]
    )
    return obj["label"] if obj else None


def _generate_skill(db, user_id):
    obs = list(db.observation_logs.find(
        {"user": user_id},
        {"action": 1, "zone_name": 1, "weight": 1, "time_slot": 1}
    ).sort("weight", -1).limit(8))

    obs_lines = "\n".join(
        f"- {d['action']} at {d.get('zone_name','?')} "
        f"({d.get('weight',0):.0f}x, {d.get('time_slot','?')})"
        for d in obs
    ) or "- No habits recorded yet"

    drink_item = _get_top_item(db, user_id, DRINK_ACTIONS, "drink")
    food_item  = _get_top_item(db, user_id, FOOD_ACTIONS,  "food")

    drink_bullet = f"- {user_id} frequently drinks {drink_item}" if drink_item else ""
    food_bullet  = f"- {user_id} frequently eats {food_item}" if food_item else ""

    skill_md = (
        f"# {user_id} Skill Profile\n"
        f"*Version 1 | Updated: {datetime.now().strftime('%Y-%m-%d')}*\n\n"
        f"## Behavior Patterns\n"
        f"{obs_lines}\n\n"
        f"## Preferences\n"
        f"{drink_bullet}\n"
        f"{food_bullet}\n\n"
        f"## How to Handle Requests\n"
        f"- Check object availability before recommending\n"
        f"- If weight >= 10, fetch directly without asking\n"
        f"- If weight < 10, ask for confirmation first\n\n"
        f"## What NOT to do\n"
        f"- Do not invent object locations\n"
        f"- Do not recommend items not in the environment\n"
    )

    db.user_skills.update_one(
        {"user_id": user_id},
        {"$set": {
            "user_id":  user_id,
            "skill_md": skill_md,
            "version":  1,
        }},
        upsert=True,
    )

    print(f"  SKILL.md regenerated for {user_id}")
    if drink_item:
        print(f"    → drink: {drink_item}")
    if food_item:
        print(f"    → food:  {food_item}")

    return drink_item, food_item


def main():
    client = MongoClient(MONGO_URI)
    db     = client[DEMO_DB]

    existing = db.list_collection_names()
    if not existing:
        print(f"[!] {DEMO_DB} does not exist.")
        print("    Run setup_demo_db.py first.")
        return

    print(f"Resetting demo DB: {DEMO_DB}")
    print("=" * 50)

    for col in RESET_COLLECTIONS:
        result = db[col].delete_many({})
        if result.deleted_count > 0:
            print(f"  Cleared {col:<25} ({result.deleted_count} docs)")
        else:
            print(f"  Cleared {col:<25} (already empty)")

    print()
    print("Regenerating SKILL.md from observation_logs...")
    for uid in USERS:
        _generate_skill(db, uid)

    print()
    print("=" * 50)
    print("Demo DB reset complete.")
    print()
    print("State after reset:")
    for col in RESET_COLLECTIONS:
        n = db[col].count_documents({})
        print(f"  {col:<25} {n} docs")

    print()
    print("Observation data untouched:")
    for col in ["observation_logs", "transition_counts",
                "dynamic_objects", "scene_snapshots"]:
        n = db[col].count_documents({})
        print(f"  {col:<25} {n} docs")

    print()
    print("Ready to demo. Start Flask:")
    print("  python3 app.py  → choose 3 (Demo)")

if __name__ == "__main__":
    main()