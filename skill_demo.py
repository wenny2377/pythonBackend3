import os
import time
import argparse
from datetime import datetime
from pymongo import MongoClient

MONGO_URI = "mongodb://127.0.0.1:27017/"
DB_NAME   = "robot_rag_db"
USERS     = ["User_Mom", "User_Dad"]
WIDTH     = 62

SECTIONS = [
    "Behavior Patterns",
    "Preferences",
    "How to Handle",
    "What NOT to do",
]


def clear():
    os.system("clear")


def get_doc(db, user_id):
    return db.user_skills.find_one({"user_id": user_id})


def count_bullets(skill_md: str) -> dict:
    counts  = {s: 0 for s in SECTIONS}
    current = None
    for line in skill_md.split('\n'):
        for s in SECTIONS:
            if f"## {s}" in line:
                current = s
                break
        if current and line.strip().startswith('-'):
            counts[current] += 1
    return counts


def get_section_bullets(skill_md: str, section: str,
                         max_items: int = 10) -> list:
    idx = skill_md.find(f"## {section}")
    if idx == -1:
        return []
    block = skill_md[idx:]
    end   = len(block)
    for s in SECTIONS:
        if s == section:
            continue
        i = block.find(f"## {s}", 3)
        if i != -1 and i < end:
            end = i
    block = block[:end]
    return [
        l.strip() for l in block.split('\n')
        if l.strip().startswith('-')
    ][:max_items]


def display_user(db, user_id, last_version, show_full):
    doc = get_doc(db, user_id)
    if not doc:
        print(f"  [{user_id}] No SKILL.md found")
        return last_version

    version  = doc.get("version", 0)
    skill_md = doc.get("skill_md", "")
    changed  = (version != last_version and last_version != -1)
    counts   = count_bullets(skill_md)
    total    = sum(counts.values())

    print("─" * WIDTH)
    marker = "  *** UPDATED ***" if changed else ""
    print(f"  {user_id}  |  v{version}  |  Rules: {total}{marker}")
    print("─" * WIDTH)

    if changed:
        print(f"  Version changed: v{last_version} -> v{version}")
        print()

    for sec, count in counts.items():
        bar   = "█" * count + "░" * max(0, 8 - count)
        short = sec[:18]
        print(f"  {short:<18} [{bar}] {count}")

    print()

    if show_full:
        for section in SECTIONS:
            bullets = get_section_bullets(skill_md, section, 10)
            print(f"  ## {section}")
            if bullets:
                for b in bullets:
                    print(f"    {b}")
            else:
                print(f"    (empty)")
            print()
    else:
        not_todos = get_section_bullets(skill_md, "What NOT to do", 6)
        if not_todos:
            print("  What NOT to do:")
            for b in not_todos:
                print(f"    {b}")
            print()

        prefs = get_section_bullets(skill_md, "Preferences", 4)
        if prefs:
            print("  Preferences:")
            for b in prefs:
                print(f"    {b}")
            print()

        patterns = get_section_bullets(skill_md, "Behavior Patterns", 4)
        if patterns:
            print("  Behavior Patterns:")
            for b in patterns:
                print(f"    {b}")
            print()

    return version


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--users",    default=",".join(USERS))
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--full",     action="store_true",
                        help="Show all bullets in every section")
    args  = parser.parse_args()
    users = args.users.split(",")

    client = MongoClient(MONGO_URI)
    db     = client[DB_NAME]

    last_versions = {u: -1 for u in users}

    print(f"Monitoring: {', '.join(users)}")
    time.sleep(1)

    try:
        while True:
            clear()
            now = datetime.now().strftime("%H:%M:%S")

            print("=" * WIDTH)
            print(f"  SKILL.md Monitor  |  {now}")
            if args.full:
                print(f"  Mode: FULL")
            print("=" * WIDTH)
            print()

            for user_id in users:
                last_versions[user_id] = display_user(
                    db, user_id, last_versions[user_id], args.full
                )
                print()

            print("─" * WIDTH)
            print(f"  Ctrl+C to stop  |  every {args.interval}s")

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()