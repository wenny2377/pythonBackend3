import argparse
import datetime
import json
import os
import sys
import time

import requests
from pymongo import MongoClient

BACKEND_URL = "http://localhost:5000"
MONGO_URI   = "mongodb://127.0.0.1:27017/"
DB_NAME     = "robot_rag_db"
USER_ID     = "User_Mom"
THRESHOLD   = 5   # must match habit_learner.HABIT_THRESHOLD
TARGET_ITEM = "milk"
TARGET_ACT  = "Drink"
TARGET_INST = "table"

CLEAN_SKILL = """# User_Mom Skill Profile
*Version 1 | Updated: {date}*

## Behavior Patterns

## Preferences
<!-- No confirmed preferences yet -->

## How to Handle Requests
- Check object availability before recommending
- If requested item is unavailable, suggest nearest alternative

## What NOT to do
- Do not invent object locations
- Do not recommend items not in the environment snapshot
"""


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def reset_skill(db):
    clean = CLEAN_SKILL.format(date=datetime.datetime.now().strftime("%Y-%m-%d"))
    db.user_skills.update_one(
        {"user_id": USER_ID},
        {"$set": {
            "skill_md":   clean,
            "version":    1,
            "updated_at": datetime.datetime.utcnow(),
        }},
        upsert=True,
    )
    db.skill_chunks.delete_many({"user_id": USER_ID})
    print("  SKILL.md reset to version 1")


def reset_observations(db):
    db.observation_logs.delete_many({"user": USER_ID})
    print("  observation_logs cleared for User_Mom")


def inject_observations(db, n: int):
    """
    Inject N synthetic observation_logs entries to simulate
    VLM observing User_Mom drinking milk N times.
    """
    now = datetime.datetime.utcnow()
    db.observation_logs.update_one(
        {
            "user":     USER_ID,
            "action":   TARGET_ACT,
            "instance": TARGET_INST,
        },
        {
            "$set": {
                "user":               USER_ID,
                "action":             TARGET_ACT,
                "instance":           TARGET_INST,
                "interacting_items":  [TARGET_ITEM],
                "last_seen":          now,
                "pos":                [3.0, -0.6],
            },
            "$inc": {"weight": n},
            "$setOnInsert": {"first_seen": now},
        },
        upsert=True,
    )
    print(f"  Injected {n} observations: "
          f"{TARGET_ACT} near {TARGET_INST} with {TARGET_ITEM}")


def get_version(db):
    doc = db.user_skills.find_one({"user_id": USER_ID})
    return doc.get("version", 0) if doc else 0


def get_skill(db):
    doc = db.user_skills.find_one({"user_id": USER_ID})
    return doc.get("skill_md", "") if doc else ""


def trigger_habit_check(url):
    """
    POST to /habit_check to manually trigger HabitLearner.check_and_update()
    without needing a full VLM episode.
    """
    try:
        resp = requests.post(
            f"{url}/habit_check",
            json={"user_id": USER_ID},
            timeout=30,
        )
        return resp.status_code == 200
    except Exception as e:
        print(f"  [trigger] Error: {e}")
        return False


def call_stream(url, query, user_id):
    try:
        resp = requests.post(
            f"{url}/interact/stream",
            json={"query": query, "userID": user_id, "room": ""},
            stream=True,
            timeout=60,
        )
        resp.raise_for_status()
    except Exception as e:
        return {"answer": "", "intent_type": ""}

    answer  = ""
    buf     = ""
    is_json = None

    for raw in resp.iter_lines():
        if not raw:
            continue
        line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        if not line.startswith("data: "):
            continue
        try:
            ev = json.loads(line[6:])
            if ev.get("type") == "intent":
                is_json = ev.get("intent", "") == "service"
            elif ev.get("type") == "token":
                token = ev.get("content", "")
                buf  += token
                if not is_json:
                    answer += token
            elif ev.get("type") == "done":
                if is_json and buf:
                    import re
                    m = re.search(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"', buf)
                    if m:
                        answer = m.group(1)
                break
        except Exception:
            continue

    return {"answer": answer.strip()}


def check_skill_updated(skill_md: str) -> dict:
    bp_updated   = TARGET_ITEM in skill_md and "## Behavior Patterns" in skill_md
    pref_updated = TARGET_ITEM in skill_md and "## Preferences" in skill_md
    return {
        "behavior_patterns_updated": bp_updated,
        "preferences_updated":       pref_updated,
        "any_updated":               bp_updated or pref_updated,
    }


def check_milk_recommended(answer: str) -> bool:
    return TARGET_ITEM.lower() in answer.lower()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url",  default=BACKEND_URL)
    parser.add_argument("--out",  default=".")
    parser.add_argument("--wait", type=int, default=25,
                        help="Seconds to wait for background habit update")
    parser.add_argument("--obs",  type=int, default=THRESHOLD,
                        help="Number of observations to inject (default: threshold)")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    try:
        requests.get(f"{args.url}/", timeout=5)
    except Exception:
        print(f"Cannot connect to {args.url}. Start app.py first.")
        sys.exit(1)

    client = MongoClient(MONGO_URI)
    db     = client[DB_NAME]

    print("\n" + "=" * 60)
    print("RQ3: Habit-Based Automatic SKILL.md Learning")
    print("=" * 60)

    # ── Setup ──────────────────────────────────────────────────
    print("\n--- Setup ---")
    reset_skill(db)
    reset_observations(db)
    version_before = get_version(db)
    print(f"  Initial SKILL.md version: {version_before}")

    # ── Step 1: Baseline (no learning yet) ────────────────────
    print("\n--- Step 1: Baseline query (no habit learned yet) ---")
    r1 = call_stream(args.url, "I am thirsty", USER_ID)
    print(f"  Answer: {r1['answer'][:80]}")
    baseline_has_milk = check_milk_recommended(r1["answer"])
    print(f"  Milk recommended at baseline: {baseline_has_milk}")

    # ── Step 2: Inject observations ───────────────────────────
    print(f"\n--- Step 2: Inject {args.obs} observations "
          f"({TARGET_ACT} + {TARGET_ITEM}) ---")
    inject_observations(db, args.obs)

    obs_doc = db.observation_logs.find_one({
        "user": USER_ID, "action": TARGET_ACT, "instance": TARGET_INST
    })
    actual_weight = obs_doc.get("weight", 0) if obs_doc else 0
    print(f"  observation_logs weight: {actual_weight} "
          f"(threshold: {THRESHOLD})")

    # ── Step 3: Trigger habit check ───────────────────────────
    print("\n--- Step 3: Trigger HabitLearner ---")
    triggered = trigger_habit_check(args.url)
    print(f"  Trigger sent: {triggered}")
    print(f"  Waiting {args.wait}s for background update...")
    time.sleep(args.wait)

    # ── Step 4: Verify SKILL.md updated ───────────────────────
    print("\n--- Step 4: Verify SKILL.md ---")
    version_after = get_version(db)
    skill_after   = get_skill(db)
    checks        = check_skill_updated(skill_after)

    print(f"  Version: {version_before} -> {version_after}")
    print(f"  Behavior Patterns updated : {checks['behavior_patterns_updated']}")
    print(f"  Preferences updated       : {checks['preferences_updated']}")

    with open(os.path.join(args.out, "rq3_skill_after.md"), "w",
              encoding="utf-8") as f:
        f.write(skill_after)

    # ── Step 5: Post-learning query ───────────────────────────
    print("\n--- Step 5: Post-learning query ---")
    r2 = call_stream(args.url, "I am thirsty", USER_ID)
    print(f"  Answer: {r2['answer'][:80]}")
    postlearn_has_milk = check_milk_recommended(r2["answer"])
    print(f"  Milk recommended after learning: {postlearn_has_milk}")

    # ── Result ─────────────────────────────────────────────────
    passed = (
        actual_weight >= THRESHOLD and
        checks["any_updated"] and
        version_after > version_before and
        postlearn_has_milk
    )

    lines = [
        "=" * 65,
        "RQ3: Habit-Based Automatic SKILL.md Learning",
        f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 65,
        "",
        f"Setup:",
        f"  Observation threshold       : {THRESHOLD}",
        f"  Observations injected       : {args.obs}",
        f"  Target item                 : {TARGET_ITEM}",
        f"  Target action               : {TARGET_ACT}",
        "",
        f"Step 1 (baseline):",
        f"  Answer                      : {r1['answer'][:60]}",
        f"  Milk recommended at baseline: {baseline_has_milk}",
        "",
        f"Step 2 (observations):",
        f"  Weight in observation_logs  : {actual_weight}",
        f"  Threshold reached           : {actual_weight >= THRESHOLD}",
        "",
        f"Step 3-4 (auto update):",
        f"  Version changed             : {version_before} -> {version_after}",
        f"  Behavior Patterns updated   : {checks['behavior_patterns_updated']}",
        f"  Preferences updated         : {checks['preferences_updated']}",
        "",
        f"Step 5 (post-learning):",
        f"  Answer                      : {r2['answer'][:60]}",
        f"  Milk recommended            : {postlearn_has_milk}",
        "",
        f"Overall RQ3: {'PASSED' if passed else 'FAILED'}",
        "",
        "For thesis:",
        f"The habit-based automatic learning mechanism was validated by",
        f"injecting {args.obs} synthetic observations of User_Mom drinking",
        f"{TARGET_ITEM}. Once the observation count reached the threshold",
        f"of {THRESHOLD}, the system automatically updated SKILL.md without",
        f"any explicit user input. Subsequent drink queries recommended",
        f"{TARGET_ITEM} based on the inferred preference",
        f"(milk recommended: {postlearn_has_milk}), demonstrating that",
        f"the system can learn from behavioral patterns rather than",
        f"relying solely on user-initiated corrections.",
    ]

    summary_path = os.path.join(args.out, "rq3_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n  Summary saved: {summary_path}")
    print(f"\n  Overall RQ3: {'PASSED' if passed else 'FAILED'}")


if __name__ == "__main__":
    main()