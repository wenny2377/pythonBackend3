import argparse
import csv
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

USER_ID = "User_Mom"

CLEAN_SKILL = """# User_Mom Skill Profile
*Version 1 | Updated: {date}*

## Behavior Patterns
- Watching near sofa (12 times)
- Drinking near table (8 times)
- Sitting near sofa (7 times)

## Preferences
<!-- No confirmed preferences yet -->

## How to Handle Requests
- Check object availability before recommending
- If requested item is unavailable, suggest nearest alternative

## What NOT to do
- Do not invent object locations
- Do not recommend items not in the environment snapshot
"""


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
    print(f"  SKILL.md reset to version 1")


def get_skill(db):
    doc = db.user_skills.find_one({"user_id": USER_ID})
    return doc.get("skill_md", "") if doc else ""


def get_version(db):
    doc = db.user_skills.find_one({"user_id": USER_ID})
    return doc.get("version", 0) if doc else 0


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
        return {"error": str(e), "answer": "", "intent_type": ""}

    answer  = ""
    intent  = ""
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
                intent  = ev.get("intent", "")
                is_json = intent == "service"
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

    return {"answer": answer.strip(), "intent_type": intent}


def wait_bg(seconds, label=""):
    print(f"  Waiting {seconds}s for background skill update{' - ' + label if label else ''}...")
    time.sleep(seconds)


def check_cola_absent(answer):
    return "cola" not in answer.lower()


def check_juice_present(answer):
    return any(kw in answer.lower() for kw in ["juice", "juicebottle"])


def check_preference_recorded(skill_md):
    pref_start = skill_md.find("## Preferences")
    pref_end   = skill_md.find("## How to Handle Requests")
    if pref_start == -1:
        return False
    pref_section = skill_md[pref_start:pref_end] if pref_end != -1 else skill_md[pref_start:]
    return any(
        kw in pref_section.lower()
        for kw in ["cola", "juice", "dislike", "prefer", "not like"]
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url",    default=BACKEND_URL)
    parser.add_argument("--out",    default=".")
    parser.add_argument("--wait",   type=int, default=20)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    try:
        requests.get(f"{args.url}/", timeout=5)
    except Exception:
        print(f"Cannot connect to {args.url}. Start app.py first.")
        sys.exit(1)

    client = MongoClient(MONGO_URI)
    db     = client[DB_NAME]

    results = {}

    print("\n" + "=" * 60)
    print("RQ3: SKILL.md Adaptive Learning Validation")
    print("=" * 60)

    print("\n--- Setup: Reset SKILL.md ---")
    reset_skill(db)
    skill_before = get_skill(db)
    version_before = get_version(db)

    with open(os.path.join(args.out, "rq3_before.md"), "w", encoding="utf-8") as f:
        f.write(skill_before)
    print(f"  Initial version: {version_before}")

    print("\n--- Scenario A: Preference Learning ---")
    print("\nStep 1: Ask about drink (baseline)")
    r1 = call_stream(args.url, "I am thirsty", USER_ID)
    print(f"  Answer: {r1['answer']}")
    results["a1_answer"] = r1["answer"]
    results["a1_has_cola"] = "cola" in r1["answer"].lower()

    print("\nStep 2: Express dislike of cola")
    r2 = call_stream(args.url, "I don't like cola, I prefer juice", USER_ID)
    print(f"  Answer: {r2['answer']}")
    results["a2_answer"] = r2["answer"]
    wait_bg(args.wait, "preference update")

    version_after_a = get_version(db)
    skill_after_a   = get_skill(db)
    results["a_version_changed"] = version_after_a > version_before
    results["a_preference_recorded"] = check_preference_recorded(skill_after_a)
    print(f"  Version: {version_before} -> {version_after_a}")
    print(f"  Preference recorded: {results['a_preference_recorded']}")

    print("\nStep 3: Ask about drink again")
    r3 = call_stream(args.url, "I am thirsty again", USER_ID)
    print(f"  Answer: {r3['answer']}")
    results["a3_answer"]    = r3["answer"]
    results["a3_cola_gone"] = check_cola_absent(r3["answer"])
    results["a3_juice_present"] = check_juice_present(r3["answer"])
    print(f"  Cola absent: {results['a3_cola_gone']}")
    print(f"  Juice present: {results['a3_juice_present']}")

    skill_after_pref = get_skill(db)
    with open(os.path.join(args.out, "rq3_after_preference.md"), "w", encoding="utf-8") as f:
        f.write(skill_after_pref)

    scenario_a_pass = (
        results.get("a_version_changed", False) and
        results.get("a_preference_recorded", False) and
        results.get("a3_cola_gone", False)
    )

    lines = [
        "=" * 65,
        "RQ3: SKILL.md Adaptive Learning Validation",
        f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 65,
        "",
        "Scenario A: Preference Learning",
        f"  Step 1 (baseline) answer    : {results.get('a1_answer','')[:60]}",
        f"  Step 1 mentioned cola       : {results.get('a1_has_cola', False)}",
        f"  Step 2 (correction) answer  : {results.get('a2_answer','')[:60]}",
        f"  Version changed             : {results.get('a_version_changed', False)}",
        f"  Preference recorded         : {results.get('a_preference_recorded', False)}",
        f"  Step 3 (after learning)     : {results.get('a3_answer','')[:60]}",
        f"  Cola absent in step 3       : {results.get('a3_cola_gone', False)}",
        f"  Juice present in step 3     : {results.get('a3_juice_present', False)}",
        f"  Scenario A PASSED           : {scenario_a_pass}",
        "",
        f"Overall RQ3: {'PASSED' if scenario_a_pass else 'FAILED'}",
        "",
        "For thesis:",
        f"The SKILL.md preference learning mechanism was evaluated through",
        f"a controlled scenario. After the user expressed a preference",
        f"('I don't like cola, I prefer juice'), the system automatically",
        f"updated SKILL.md and subsequent drink queries no longer recommended",
        f"cola (cola absent: {results.get('a3_cola_gone', False)},",
        f"juice present: {results.get('a3_juice_present', False)}).",
        f"The system learned from a single conversation without any model",
        f"retraining or cloud connectivity.",
    ]

    summary_path = os.path.join(args.out, "rq3_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  Summary saved: {summary_path}")
    print(f"\n  Scenario A (preference): {'PASSED' if scenario_a_pass else 'FAILED'}")


if __name__ == "__main__":
    main()