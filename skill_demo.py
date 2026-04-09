import os
import time
import argparse
from datetime import datetime
from pymongo import MongoClient

MONGO_URI = "mongodb://127.0.0.1:27017/"
DB_NAME   = "robot_rag_db"


def clear():
    os.system("clear")


def get_doc(db, user_id):
    return db.user_skills.find_one({"user_id": user_id})


def display(doc, user_id, last_version):
    clear()
    now = datetime.now().strftime("%H:%M:%S")

    if not doc:
        print(f"[{now}] No SKILL.md found for {user_id}")
        return last_version

    version  = doc.get("version", 0)
    skill_md = doc.get("skill_md", "")
    changed  = version != last_version and last_version != -1

    print("=" * 60)
    print(f"  SKILL.md Monitor — {user_id}")
    print(f"  Version: {version}  |  Updated: {now}")
    if changed:
        print(f"  *** VERSION CHANGED: {last_version} -> {version} ***")
    print("=" * 60)
    print(skill_md)
    print("=" * 60)
    print(f"  [Ctrl+C to stop]  polling every 2s")

    return version


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", default="User_Mom")
    parser.add_argument("--interval", type=float, default=2.0)
    args = parser.parse_args()

    client = MongoClient(MONGO_URI)
    db     = client[DB_NAME]

    last_version = -1

    print(f"Monitoring SKILL.md for {args.user}...")
    time.sleep(1)

    try:
        while True:
            doc          = get_doc(db, args.user)
            last_version = display(doc, args.user, last_version)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()