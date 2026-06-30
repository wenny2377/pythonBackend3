import os
import sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from pymongo import MongoClient
from datetime import datetime

from config import Config
from modules.memory.pattern_analyzer import PatternAnalyzer
from modules.memory.skill_manager import SkillManager

DEMO_DB   = "robot_exp_demo"
MONGO_URI = "mongodb://127.0.0.1:27017/"
USERS     = ["User_Mom", "User_Dad"]

RESET_COLLECTIONS = [
    "user_skills",
    "skill_chunks",
    "service_proposals",
    "conversation_logs",
]


def _refresh_last_seen(db):
    now = datetime.utcnow()
    r   = db.dynamic_objects.update_many({}, {"$set": {"last_seen": now}})
    print(f"  Refreshed last_seen: {r.modified_count} objects → now")


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
    print("Regenerating SKILL.md via PatternAnalyzer + SkillManager...")

    analyzer = PatternAnalyzer(db)
    skill_manager = SkillManager(
        db_client=client,
        ollama_url=Config.OLLAMA_URL,
        model_name=Config.LLM_MODEL,
        db_name=DEMO_DB,
    )

    for uid in USERS:
        skill_manager.generate(uid)
        patterns = analyzer.analyze_user(uid)
        if patterns:
            skill_manager.sync_from_patterns(uid, patterns)
        print(f"  SKILL.md regenerated for {uid}")

    print()
    print("=" * 50)
    print("Demo DB reset complete.")
    print()
    print("State after reset:")
    for col in RESET_COLLECTIONS:
        print(f"  {col:<25} {db[col].count_documents({})} docs")
    print()
    print("Untouched:")
    for col in ["observation_logs", "transition_counts", "behavior_patterns",
                "dynamic_objects", "scene_snapshots", "object_events"]:
        print(f"  {col:<25} {db[col].count_documents({})} docs")
    print()
    print("Ready. Start Flask: python3 app.py → choose 3 (Demo)")


if __name__ == "__main__":
    main()
