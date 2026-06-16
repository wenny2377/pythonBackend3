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
]

USERS = ["User_Mom", "User_Dad"]

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
    print(f"  SKILL.md generated for {user_id}")
    if drink_item: print(f"    drink: {drink_item}")
    if food_item:  print(f"    food:  {food_item}")


def rebuild_skill_chunks(client, user_id, db_name):
    try:
        from modules.memory.skill_manager import SkillManager
        db = client[db_name]
        sm = SkillManager(
            db_client=client,
            ollama_url="http://localhost:11434",
            model_name="llama3.1:8b",
            db_name=db_name,
        )
        doc = db.user_skills.find_one({"user_id": user_id})
        if doc:
            sm._chunk_skill_md(doc["skill_md"], user_id)
            print(f"  skill_chunks rebuilt for {user_id}")
    except Exception as e:
        print(f"  [warn] skill_chunks rebuild failed: {e}")


def main():
    client = MongoClient(MONGO_URI)
    src    = client[SRC_DB]
    dst    = client[DST_DB]

    print(f"Copying {SRC_DB} → {DST_DB}")
    print("=" * 50)

    existing = dst.list_collection_names()
    if existing:
        confirm = input(
            f"[!] {DST_DB} already exists. Overwrite? [y/N]: "
        ).strip().lower()
        if confirm != "y":
            print("Aborted.")
            return
        for col in existing:
            dst[col].drop()
        print("  Cleared existing demo DB.")

    total = 0
    seen  = set()
    for col in COLLECTIONS:
        if col in seen:
            continue
        seen.add(col)
        docs = list(src[col].find({}))
        if not docs:
            print(f"  {col:<30} skipped (empty)")
            continue
        for d in docs:
            d.pop("_id", None)
        dst[col].insert_many(docs)
        print(f"  {col:<30} {len(docs)} docs")
        total += len(docs)

    now = datetime.utcnow()
    dst.dynamic_objects.update_many({}, {"$set": {"last_seen": now}})
    print(f"  Refreshed last_seen on dynamic_objects")

    print()
    print("Generating SKILL.md from object_events...")
    dst.user_skills.drop()
    dst.skill_chunks.drop()

    for uid in USERS:
        generate_skill(dst, uid)

    print()
    print("Rebuilding skill_chunks FAISS index...")
    for uid in USERS:
        rebuild_skill_chunks(client, uid, DST_DB)

    print()
    print("=" * 50)
    print(f"Done. {total} documents copied to {DST_DB}")

    print()
    print("Preferences inferred:")
    import re
    for uid in USERS:
        doc = dst.user_skills.find_one({"user_id": uid})
        if doc:
            m = re.search(
                r"## Preferences\n(.*?)(?=\n## )",
                doc["skill_md"], re.DOTALL
            )
            prefs = m.group(1).strip() if m else "N/A"
            print(f"  {uid}: {prefs}")

    print()
    print("Next steps:")
    print("  python3 app.py  → choose 3 (Demo)")
    print("  open demo/index.html in browser")
    print()
    print("Rules:")
    print("  NEVER run Unity Experiment mode against robot_exp_demo")
    print("  NEVER run analysis scripts against robot_exp_demo")


if __name__ == "__main__":
    main()