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

LABEL_NORMALIZE = {
    "waterbottle":  "water",
    "water bottle": "water",
    "juicebottle":  "juice",
    "juice bottle": "juice",
    "orange juice": "juice",
    "cola can":     "cola",
    "coca cola":    "cola",
    "coke":         "cola",
    "soda can":     "cola",
    "cell phone":   "phone",
    "mobile phone": "phone",
    "smartphone":   "phone",
    "iphone":       "phone",
    "frying pan":   "pan",
    "cooking pan":  "pan",
    "skillet":      "pan",
    "mug":          "cup",
    "coffee cup":   "cup",
    "tea cup":      "cup",
}


def _normalize_label(label: str) -> str:
    return LABEL_NORMALIZE.get(label.lower().strip(), label.lower().strip())


def _get_top_item(db, user_id, category):
    item_counts = defaultdict(int)
    for d in db.object_events.find(
        {"user": user_id, "pickup_time": {"$exists": True}},
        {"object": 1}
    ):
        label = _normalize_label(d.get("object", "").strip())
        if db.dynamic_objects.find_one({"label": label, "category": category}):
            item_counts[label] += 1
    if item_counts:
        return max(item_counts, key=item_counts.get)
    obj = db.dynamic_objects.find_one(
        {"category": category},
        sort=[("interact_count", -1)]
    )
    return obj["label"] if obj else None


def _refresh_last_seen(db):
    now = datetime.utcnow()
    r   = db.dynamic_objects.update_many({}, {"$set": {"last_seen": now}})
    print(f"  Refreshed last_seen: {r.modified_count} objects → now")


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

    drink_item = _get_top_item(db, user_id, "drink")
    food_item  = _get_top_item(db, user_id, "food")

    drink_bullet = f"- User enjoys {drink_item} during Drinking" if drink_item else ""
    food_bullet  = f"- User enjoys {food_item} during Eating"    if food_item else ""

    skill_md = (
        f"# {user_id} Skill Profile\n"
        f"*Version 1 | Updated: {datetime.now().strftime('%Y-%m-%d')}*\n\n"
        f"## Behavior Patterns\n"
        f"{obs_lines}\n\n"
        f"## Preferences\n"
        f"{drink_bullet}\n"
        f"{food_bullet}\n\n"
        f"## How to Handle Requests\n"
        f"- If weight >= 10, fetch directly without asking\n"
        f"- If weight < 10, ask for confirmation first\n\n"
        f"## What NOT to do\n"
        f"- Do not invent object locations\n"
        f"- Do not recommend items not in the environment\n"
    )

    db.user_skills.update_one(
        {"user_id": user_id},
        {"$set": {"user_id": user_id, "skill_md": skill_md, "version": 1}},
        upsert=True,
    )
    print(f"  SKILL.md regenerated for {user_id}")
    if drink_item: print(f"    drink: {drink_item}")
    if food_item:  print(f"    food:  {food_item}")


def main():
    client = MongoClient(MONGO_URI)
    db     = client[DEMO_DB]

    if not db.list_collection_names():
        print(f"[!] {DEMO_DB} does not exist. Run setup_demo_db.py first.")
        return

    print(f"Resetting demo DB: {DEMO_DB}")
    print("=" * 50)

    for col in RESET_COLLECTIONS:
        result = db[col].delete_many({})
        print(f"  Cleared {col:<25} ({result.deleted_count} docs)")

    print()
    print("Refreshing object timestamps...")
    _refresh_last_seen(db)

    print()
    print("Regenerating SKILL.md...")
    for uid in USERS:
        _generate_skill(db, uid)

    print()
    print("=" * 50)
    print("Demo DB reset complete.")
    print()
    print("State after reset:")
    for col in RESET_COLLECTIONS:
        print(f"  {col:<25} {db[col].count_documents({})} docs")
    print()
    print("Untouched:")
    for col in ["observation_logs", "transition_counts",
                "dynamic_objects", "scene_snapshots", "object_events"]:
        print(f"  {col:<25} {db[col].count_documents({})} docs")
    print()
    print("Ready. Start Flask: python3 app.py → choose 3 (Demo)")


if __name__ == "__main__":
    main()