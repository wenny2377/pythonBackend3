import os
import sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from pymongo import MongoClient
from datetime import datetime

from config import Config
from modules.memory.pattern_analyzer import PatternAnalyzer
from modules.memory.skill_manager import SkillManager

SRC_DB    = "robot_exp_baseline"
DST_DB    = "robot_exp_demo"
MONGO_URI = "mongodb://127.0.0.1:27017/"

COLLECTIONS = [
    "observation_logs", "transition_counts", "behavior_patterns",
    "scene_snapshots",  "dynamic_objects",
    "user_positions",   "device_states",
    "activity_sequences",
    "user_spatial_affinity", "zone_anchors",
    "user_skills", "skill_chunks",
    "object_events",
]

USERS = ["User_Mom", "User_Dad"]


def regenerate_skills(db_name: str):
    client = MongoClient(MONGO_URI)
    db = client[db_name]

    analyzer = PatternAnalyzer(db)
    skill_manager = SkillManager(
        db_client=client,
        ollama_url=Config.OLLAMA_URL,
        model_name=Config.LLM_MODEL,
        db_name=db_name,
    )

    for uid in USERS:
        db.user_skills.delete_one({"user_id": uid})
        skill_manager.generate(uid)
        patterns = analyzer.analyze_user(uid)
        if patterns:
            skill_manager.sync_from_patterns(uid, patterns)
        skill_doc = db.user_skills.find_one({"user_id": uid})
        print(f"  SKILL.md generated for {uid}")
        if skill_doc:
            print(f"    version: {skill_doc.get('version', 1)}")

    client.close()


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
    print("Regenerating SKILL.md via PatternAnalyzer + SkillManager...")
    dst.user_skills.drop()
    dst.skill_chunks.drop()

    regenerate_skills(DST_DB)

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
