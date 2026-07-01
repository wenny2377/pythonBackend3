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

# Cleared every reset — state accumulated during a demo run.
# Learning data (observation_logs, transition_counts, etc.) is NOT cleared
# because demo mode now skips all learning writes entirely.
CLEAR_COLLECTIONS = [
    "user_skills",
    "skill_chunks",
    "service_proposals",
    "activity_sequences",
    "conversation_logs",
]

# Never touched — baseline learning data and static scene data.
# These are safe because demo mode does not write to them.
PRESERVE_COLLECTIONS = [
    "observation_logs",
    "transition_counts",
    "behavior_patterns",
    "user_spatial_affinity",
    "scene_snapshots",
    "dynamic_objects",
    "object_events",
    "affinity_matrix",
    "affinity_history",
    "zone_anchors",
    "user_positions",
]


def _refresh_timestamps(db):
    now = datetime.utcnow()
    r = db.dynamic_objects.update_many({}, {"$set": {"last_seen": now}})
    print(f"  Refreshed dynamic_objects.last_seen  ({r.modified_count} docs)")
    r = db.user_positions.update_many({}, {"$set": {"updated_at": now}})
    print(f"  Refreshed user_positions.updated_at  ({r.modified_count} docs)")


def main():
    client = MongoClient(MONGO_URI)
    db     = client[DEMO_DB]

    if not db.list_collection_names():
        print(f"[!] {DEMO_DB} does not exist. Run setup_demo_db.py first.")
        client.close()
        return

    print(f"Resetting demo DB: {DEMO_DB}")
    print("=" * 50)

    print("Clearing demo-run state...")
    for col in CLEAR_COLLECTIONS:
        result = db[col].delete_many({})
        print(f"  Cleared {col:<30} ({result.deleted_count} docs)")

    print()
    print("Refreshing timestamps...")
    _refresh_timestamps(db)

    print()
    print("Regenerating SKILL.md from baseline learning data...")
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
    print("Cleared (demo run state):")
    for col in CLEAR_COLLECTIONS:
        print(f"  {col:<30} {db[col].count_documents({})} docs")
    print()
    print("Preserved (untouched by demo mode):")
    for col in PRESERVE_COLLECTIONS:
        print(f"  {col:<30} {db[col].count_documents({})} docs")
    print()
    print("Ready. Start Flask with DB_NAME=robot_exp_demo, then run Unity in Demo mode.")

    client.close()


if __name__ == "__main__":
    main()