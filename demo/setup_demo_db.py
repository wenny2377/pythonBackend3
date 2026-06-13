import os
import sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from pymongo import MongoClient
from collections import defaultdict
from datetime import datetime

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
    "object_events",
    "object_events",
]

USERS = ["User_Mom", "User_Dad"]

DRINK_ACTIONS = {"Drinking", "SittingDrink"}
FOOD_ACTIONS  = {"Eating", "Cooking"}

LABEL_NORMALIZE = {
    "waterbottle":  "bottle",
    "water bottle": "bottle",
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

PREF_KEYWORD = ["enjoys", "likes", "frequently", "drinks", "eats", "prefers"]


def _normalize_label(label: str) -> str:
    return LABEL_NORMALIZE.get(label.lower().strip(), label.lower().strip())


def _get_top_item_from_events(db, user_id, actions, category):
    item_counts = defaultdict(int)
    for d in db.object_events.find(
        {"user": user_id, "pickup_time": {"$exists": True}},
        {"object": 1}
    ):
        raw   = d.get("object", "")
        label = _normalize_label(raw)
        obj   = db.dynamic_objects.find_one({"label": label, "category": category})
        if obj:
            item_counts[label] += 1

    if item_counts:
        return max(item_counts, key=item_counts.get)

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


def _build_skill_md(user_id, drink_item, food_item, obs_lines):
    drink_bullet = f"- User enjoys {drink_item} during Drinking" if drink_item else ""
    food_bullet  = f"- User enjoys {food_item} during Eating"    if food_item else ""
    return (
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


def generate_skill(db, user_id):
    obs = list(db.observation_logs.find(
        {"user": user_id},
        {"action": 1, "zone_name": 1, "weight": 1, "time_slot": 1}
    ).sort("weight", -1).limit(8))

    obs_lines = "\n".join(
        f"- {d['action']} at {d.get('zone_name','?')} "
        f"({d.get('weight',0):.0f}x, {d.get('time_slot','?')})"
        for d in obs
    ) or "- No habits recorded yet"

    drink_item = _get_top_item_from_events(db, user_id, DRINK_ACTIONS, "drink")
    food_item  = _get_top_item_from_events(db, user_id, FOOD_ACTIONS,  "food")

    skill_md = _build_skill_md(user_id, drink_item, food_item, obs_lines)

    db.user_skills.update_one(
        {"user_id": user_id},
        {"$set": {"user_id": user_id, "skill_md": skill_md, "version": 1}},
        upsert=True,
    )
    print(f"  SKILL.md generated for {user_id}")
    if drink_item:
        print(f"    drink: {drink_item}")
    if food_item:
        print(f"    food:  {food_item}")


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
    print("Generating SKILL.md from object_events + observation_logs...")
    for uid in USERS:
        generate_skill(dst, uid)

    print()
    print("=" * 50)
    print(f"Done. {total} documents copied to {DST_DB}")
    print()
    print("Next steps:")
    print("  python3 app.py  → choose 3 (Demo)")
    print("  open demo/index.html in browser")
    print()
    print("Rules:")
    print("  NEVER run Unity against robot_exp_demo")
    print("  NEVER run resetall.py on robot_exp_demo")
    print("  NEVER run analysis scripts against robot_exp_demo")


if __name__ == "__main__":
    main()