"""
train_manifold.py
─────────────────
Offline tool — Train ManifoldEngine MLP models after collecting data.

The ManifoldEngine trains automatically during experiments (every 20 samples),
but this script lets you manually trigger training after an experiment run,
or retrain from scratch with all collected data.

Usage:
  # Train all users
  python3 tools/train_manifold.py

  # Train specific user
  python3 tools/train_manifold.py --user User_Mom

  # Show training data stats only (no training)
  python3 tools/train_manifold.py --stats

  # Force retrain even if sample count is low
  python3 tools/train_manifold.py --force
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

MONGO_URI = "mongodb://127.0.0.1:27017/"
DB_NAME   = "robot_rag_db"
USERS     = ["User_Mom", "User_Dad"]


def print_stats(db):
    print("\n=== ManifoldEngine Training Data Stats ===")
    for uid in USERS:
        n = db.manifold_training_data.count_documents({"user_id": uid})
        model_path = os.path.join("manifold_models", f"{uid}.pkl")
        model_exists = os.path.exists(model_path)

        print(f"\n  {uid}:")
        print(f"    Training samples : {n}")
        print(f"    Model exists     : {'Yes' if model_exists else 'No'}")

        if n > 0:
            # Show action distribution
            from collections import Counter
            docs    = list(db.manifold_training_data.find(
                {"user_id": uid}, {"action": 1}))
            counter = Counter(d.get("action", "?") for d in docs)
            print(f"    Action distribution:")
            for action, count in sorted(counter.items(), key=lambda x: -x[1]):
                bar = "█" * min(count, 20)
                print(f"      {action:<16} {count:>4}  {bar}")


def train_user(db, sbert, user_id: str, force: bool = False):
    from modules.memory.manifold_engine import ManifoldEngine, MIN_TRAIN_SAMPLE

    n = db.manifold_training_data.count_documents({"user_id": user_id})
    print(f"\n  {user_id}: {n} training samples")

    if n < MIN_TRAIN_SAMPLE and not force:
        print(f"  Skipping — need at least {MIN_TRAIN_SAMPLE} samples "
              f"(use --force to override)")
        return False

    if n == 0:
        print(f"  No data — skipping")
        return False

    print(f"  Training MLP...")
    engine = ManifoldEngine(db=db, sbert_model=sbert)
    engine.train_model(user_id)
    print(f"  Done — model saved to manifold_models/{user_id}.pkl")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Train ManifoldEngine MLP models")
    parser.add_argument("--user",  type=str, default=None,
                        help="User ID to train (default: all)")
    parser.add_argument("--stats", action="store_true",
                        help="Show stats only, no training")
    parser.add_argument("--force", action="store_true",
                        help="Force training even with few samples")
    args = parser.parse_args()

    client = MongoClient(MONGO_URI)
    db     = client[DB_NAME]

    print_stats(db)

    if args.stats:
        return

    print("\n=== Training ===")

    # Load SBERT (lightweight for feature encoding)
    print("Loading SBERT...")
    try:
        sbert = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    except Exception as e:
        print(f"SBERT load failed: {e}")
        sbert = None

    users = [args.user] if args.user else USERS
    trained = 0

    for uid in users:
        ok = train_user(db, sbert, uid, force=args.force)
        if ok:
            trained += 1

    print(f"\n=== Done: {trained}/{len(users)} models trained ===")

    if trained > 0:
        print("\nNext steps:")
        print("  1. Restart app.py to load new models")
        print("  2. Or call POST /manifold_train to reload without restart")


if __name__ == "__main__":
    main()