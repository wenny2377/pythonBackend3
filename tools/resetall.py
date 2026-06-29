import os
import shutil
from datetime import datetime
from pymongo import MongoClient

_ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MONGO_URI = "mongodb://127.0.0.1:27017/"

SKILL_USERS = ["User_Mom", "User_Dad"]

CLEAN_SKILL = """# {user_id} Skill Profile
*Version 1 | Updated: {date}*

## Behavior Patterns

## Preferences

## How to Handle Requests
- Check object availability before recommending
- If requested item is unavailable, suggest nearest alternative

## What NOT to do
- Do not invent object locations
- Do not recommend items not in the environment snapshot
"""

COLLECTIONS_LEARNING = [
    "observation_logs", "activity_sequences", "transition_counts",
    "affinity_history", "user_spatial_affinity", "affinity_matrix",
    "robot_memory", "semantic_memories", "habit_snapshots",
]

COLLECTIONS_EXPERIMENT = [
    "experiment_logs",
    "experiment_logs_corruption_light",
    "experiment_logs_corruption_medium",
    "experiment_logs_corruption_heavy",
    "eval_logs", "exp_checkpoint_logs",
]

COLLECTIONS_ABLATION = [
    "ablation_no_skeleton", "ablation_no_vlm",
    "ablation_no_object",   "ablation_no_spatial",
]

COLLECTIONS_RUNTIME = [
    "dynamic_objects", "raw_objects", "user_positions",
    "object_events", "device_states", "system_config",
    "service_proposals", "navigation_logs", "scene_graph",
    "scene_snapshots", "user_spatial_affinity",
]

COLLECTIONS_LEGACY = [
    "manifold_training_data", "manifold_points",
    "saycan_logs", "saycan_behavior_objects",
    "skill_chunks", "episodic_summaries",
    "conversation_logs", "intent_stats",
]

MODES = {
    "1": {
        "label":      "Full reset — everything (re-run from scratch)",
        "collections": (COLLECTIONS_LEARNING + COLLECTIONS_EXPERIMENT +
                        COLLECTIONS_ABLATION + COLLECTIONS_RUNTIME + COLLECTIONS_LEGACY),
        "reset_skill": True,
        "clean_files": True,
    },
    "2": {
        "label":      "Experiment + Ablation — keep learning, re-run experiments",
        "collections": COLLECTIONS_EXPERIMENT + COLLECTIONS_ABLATION,
        "reset_skill": False,
        "clean_files": False,
    },
    "3": {
        "label":      "Ablation only — keep experiment_logs, re-run ablation",
        "collections": COLLECTIONS_ABLATION,
        "reset_skill": False,
        "clean_files": False,
    },
    "4": {
        "label":      "Learning only — reset obs/habits/skills",
        "collections": COLLECTIONS_LEARNING,
        "reset_skill": True,
        "clean_files": False,
    },
    "5": {
        "label":      "Legacy cleanup — remove old collections from previous runs",
        "collections": COLLECTIONS_LEGACY,
        "reset_skill": False,
        "clean_files": False,
    },
    "6": {
        "label":      "Custom — choose collections interactively",
        "collections": [],
        "reset_skill": False,
        "clean_files": False,
    },
    "7": {
        "label":      "Wipe ALL — delete every collection in the DB",
        "collections": [],
        "reset_skill": True,
        "clean_files": True,
    },
    "8": {
        "label":      "Single experiment — pick one collection to clear",
        "collections": [],
        "reset_skill": False,
        "clean_files": False,
    },
}



EXPERIMENT_COLLECTIONS = [
    ("experiment_logs",                    "Baseline"),
    ("experiment_logs_corruption_light",   "Corruption Light"),
    ("experiment_logs_corruption_medium",  "Corruption Medium"),
    ("experiment_logs_corruption_heavy",   "Corruption Heavy"),
    ("ablation_no_skeleton",               "Ablation: no skeleton"),
    ("ablation_no_vlm",                    "Ablation: no VLM"),
    ("ablation_no_object",                 "Ablation: no object"),
    ("ablation_no_spatial",               "Ablation: no spatial"),
]

def _ask_single_experiment(db) -> list:
    print("\nWhich experiment collection to clear?")
    existing = set(db.list_collection_names())
    available = [(col, label) for col, label in EXPERIMENT_COLLECTIONS if col in existing]
    if not available:
        print("  No experiment collections found in DB.")
        return []
    for i, (col, label) in enumerate(available, 1):
        n = db[col].count_documents({})
        print(f"  {i}) {label:35} ({col}) — {n} docs")
    try:
        choice = input("Choice: ").strip()
        idx = int(choice) - 1
        if 0 <= idx < len(available):
            return [available[idx][0]]
    except (ValueError, EOFError):
        pass
    return []


def _ask_mode() -> str:
    print("\nWhat to reset?")
    for k, v in MODES.items():
        print(f"  {k}) {v['label']}")
    try:
        choice = input("Choice [3]: ").strip() or "3"
    except EOFError:
        choice = "3"
    return choice if choice in MODES else "3"


def _ask_db() -> str:
    print("\nWhich DB?")
    print("  1) robot_exp_baseline")
    print("  2) robot_exp_corruption")
    print("  3) Both")
    try:
        choice = input("Choice [1]: ").strip() or "1"
    except EOFError:
        choice = "1"
    return choice


def _ask_custom(db) -> list:
    all_cols = sorted(db.list_collection_names())
    if not all_cols:
        print("  No collections found.")
        return []
    print("\nAvailable collections:")
    for i, col in enumerate(all_cols, 1):
        n = db[col].count_documents({})
        print(f"  {i:2}) {col:45} ({n} docs)")
    try:
        raw = input("\nSelect numbers to delete (e.g. 1 3 5): ").strip()
    except EOFError:
        return []
    selected = []
    for part in raw.split():
        try:
            idx = int(part) - 1
            if 0 <= idx < len(all_cols):
                selected.append(all_cols[idx])
        except ValueError:
            pass
    return selected


def _confirm(prompt: str) -> bool:
    try:
        ans = input(f"{prompt} [y/N]: ").strip().lower()
        return ans in ("y", "yes")
    except EOFError:
        return False


def reset_db(db_name: str, mode_key: str):
    client = MongoClient(MONGO_URI)
    db     = client[db_name]
    mode   = MODES[mode_key]

    print(f"\n{'='*55}")
    print(f"  DB: {db_name}")
    print(f"  Mode: {mode['label']}")
    print(f"{'='*55}")

    if mode_key == "8":
        to_clear = _ask_single_experiment(db)
        if not to_clear:
            print("  Nothing selected.")
            client.close()
            return
        col = to_clear[0]
        n   = db[col].count_documents({})
        print(f"\n  Will delete: {col} ({n} docs)")
        if not _confirm(f"  Proceed on {db_name}?"):
            print("  Cancelled.")
            client.close()
            return
        deleted = db[col].delete_many({}).deleted_count
        print(f"  cleared {col}: {deleted} docs")
        client.close()
        return

    if mode_key == "7":
        all_cols = db.list_collection_names()
        print(f"\n  Will delete ALL {len(all_cols)} collections:")
        for col in sorted(all_cols):
            n = db[col].count_documents({})
            print(f"    {col:45} {n:6} docs")
        if not _confirm(f"\n  WIPE EVERYTHING in {db_name}?"):
            print("  Cancelled.")
            client.close()
            return
        total = 0
        for col in all_cols:
            n = db[col].delete_many({}).deleted_count
            total += n
            if n > 0:
                print(f"  cleared {col}: {n}")
        print(f"  Total deleted: {total} docs")
        if mode.get("reset_skill"):
            print("\n  Resetting SKILL.md...")
            date = datetime.now().strftime("%Y-%m-%d")
            for uid in SKILL_USERS:
                db.user_skills.update_one(
                    {"user_id": uid},
                    {"$set": {
                        "skill_md":   CLEAN_SKILL.format(user_id=uid, date=date),
                        "version":    1,
                        "updated_at": datetime.utcnow(),
                        "is_stale":   False,
                    }},
                    upsert=True,
                )
                print(f"  {uid} SKILL.md reset")
        client.close()
        return

    to_clear = _ask_custom(db) if mode_key == "6" else mode["collections"]

    if not to_clear:
        print("  Nothing selected.")
        client.close()
        return

    existing = set(db.list_collection_names())
    to_clear = [c for c in to_clear if c in existing]

    if not to_clear:
        print("  No matching collections found in DB.")
        client.close()
        return

    print(f"\n  Will delete from {len(to_clear)} collection(s):")
    for col in to_clear:
        n = db[col].count_documents({})
        print(f"    {col:45} {n:6} docs")

    if not _confirm(f"\n  Proceed on {db_name}?"):
        print("  Cancelled.")
        client.close()
        return

    total = 0
    for col in to_clear:
        n = db[col].delete_many({}).deleted_count
        total += n
        if n > 0:
            print(f"  cleared {col}: {n}")
    print(f"  Total deleted: {total} docs")

    if mode.get("reset_skill"):
        print("\n  Resetting SKILL.md...")
        date = datetime.now().strftime("%Y-%m-%d")
        for uid in SKILL_USERS:
            db.user_skills.update_one(
                {"user_id": uid},
                {"$set": {
                    "skill_md":   CLEAN_SKILL.format(user_id=uid, date=date),
                    "version":    1,
                    "updated_at": datetime.utcnow(),
                    "is_stale":   False,
                }},
                upsert=True,
            )
            print(f"  {uid} SKILL.md reset")

    client.close()


def clean_files():
    print("\n  Cleaning local files...")
    debug_dir = os.path.join(_ROOT, "debug_images")
    if os.path.exists(debug_dir):
        shutil.rmtree(debug_dir)
        print(f"  removed debug_images/")
    for fname in ["robot_memory.index", "robot_memory_meta.json",
                  "dynamic_memory.index", "dynamic_memory_meta.json"]:
        full = os.path.join(_ROOT, fname)
        if os.path.exists(full):
            os.remove(full)
            print(f"  removed {fname}")


def print_next_steps(mode_key: str):
    steps = {
        "1": ["python app.py", "Run Baseline in Unity (autoRunAll=false, experimentType=Baseline)",
              "Then run Corruption in Unity"],
        "2": ["python app.py", "Re-run experiments in Unity"],
        "3": ["DB_NAME=robot_exp_baseline python analysis/exp3_modality_ablation.py"],
        "4": ["python app.py", "Re-run Baseline in Unity to re-learn habits"],
        "5": ["Legacy collections removed, existing experiment data preserved"],
        "6": ["Selected collections cleared"],
        "7": ["DB is now empty", "python app.py  (then run experiments in Unity)"],
        "8": ["Collection cleared",
              "python app.py",
              "Unity: autoRunAll=false, set experimentType to the one you want to re-run"],
    }
    print(f"\n  Next steps:")
    for step in steps.get(mode_key, []):
        print(f"    {step}")
    print()


def main():
    env_db   = os.environ.get("DB_NAME", "").strip()
    env_mode = os.environ.get("RESET_MODE", "").strip()

    if env_db and env_mode and env_mode in MODES:
        mode_key = env_mode
        if env_db == "both":
            dbs = ["robot_exp_baseline", "robot_exp_corruption"]
        elif "corruption" in env_db:
            dbs = ["robot_exp_corruption"]
        else:
            dbs = ["robot_exp_baseline"]
    else:
        mode_key = _ask_mode()
        db_raw   = _ask_db()
        dbs = (["robot_exp_baseline", "robot_exp_corruption"] if db_raw == "3"
               else ["robot_exp_corruption"]                   if db_raw == "2"
               else ["robot_exp_baseline"])

    for db_name in dbs:
        reset_db(db_name, mode_key)

    if MODES[mode_key].get("clean_files"):
        clean_files()

    print_next_steps(mode_key)


if __name__ == "__main__":
    main()