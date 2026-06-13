import os
import sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from pymongo import MongoClient
from config import Config

SRC_DB    = "robot_exp_baseline"
DST_DB    = "robot_exp_demo"
MONGO_URI = "mongodb://127.0.0.1:27017/"

COLLECTIONS = [
    "observation_logs", "transition_counts",
    "scene_snapshots",  "dynamic_objects",
    "user_positions",   "device_states",
    "habit_snapshots",  "activity_sequences",
    "user_spatial_affinity", "zone_anchors",
    "user_skills", "skill_chunks",
]

USERS = ["User_Mom", "User_Dad"]

DRINK_ACTIONS = {"Drinking", "SittingDrink"}
FOOD_ACTIONS  = {"Eating", "Cooking"}


def _get_top_item(db, user_id, actions, category):
    from collections import defaultdict
    obs = list(db.observation_logs.find(
        {"user": user_id, "action": {"$in": list(actions)}},
        {"zone_name": 1, "weight": 1, "interacting_items": 1}
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


def _skill_template(user_id, drink_item, food_item, obs_lines):
    from datetime import datetime
    drink_bullet = f"- {user_id} frequently drinks {drink_item}" if drink_item else ""
    food_bullet  = f"- {user_id} frequently eats {food_item}" if food_item else ""
    return (
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


def generate_skill(db, user_id):
    from collections import defaultdict

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

    skill_md = _skill_template(user_id, drink_item, food_item, obs_lines)

    db.user_skills.update_one(
        {"user_id": user_id},
        {"$set": {
            "user_id":    user_id,
            "skill_md":   skill_md,
            "version":    1,
        }},
        upsert=True,
    )
    print(f"  SKILL.md generated for {user_id}")
    if drink_item:
        print(f"    → drink preference: {drink_item}")
    if food_item:
        print(f"    → food preference:  {food_item}")


def main():
    client = MongoClient(MONGO_URI)
    src    = client[SRC_DB]
    dst    = client[DST_DB]

    print(f"Copying {SRC_DB} → {DST_DB}")
    print("=" * 50)

    existing = dst.list_collection_names()
    if existing:
        confirm = input(f"[!] {DST_DB} already exists. Overwrite? [y/N]: ").strip().lower()
        if confirm != "y":
            print("Aborted.")
            return
        for col in existing:
            dst[col].drop()
        print("  Cleared existing demo DB.")

    total = 0
    for col in COLLECTIONS:
        docs = list(src[col].find({}))
        if not docs:
            print(f"  {col:<30} skipped (empty)")
            continue
        for d in docs:
            d.pop("_id", None)
        dst[col].insert_many(docs)
        print(f"  {col:<30} {len(docs)} docs")
        total += len(docs)

    print()
    print("Generating SKILL.md from observation_logs...")
    for uid in USERS:
        generate_skill(dst, uid)

    print()
    print("=" * 50)
    print(f"Done. {total} documents copied to {DST_DB}")
    print()
    print("Next steps:")
    print("  python3 app.py        → choose 3 (Demo)")
    print("  open demo/index.html  in browser")
    print()
    print("Rules:")
    print("  NEVER run Unity against robot_exp_demo")
    print("  NEVER run resetall.py on robot_exp_demo")
    print("  NEVER run analysis scripts against robot_exp_demo")

if __name__ == "__main__":
    main()