import requests
import json
import time
from pymongo import MongoClient

BACKEND  = "http://127.0.0.1:5000"
USER_ID  = "User_Mom"
MONGO    = "mongodb://127.0.0.1:27017/"
DB_NAME  = "robot_rag_db"

db = MongoClient(MONGO)[DB_NAME]


def divider(title=""):
    width = 60
    if title:
        pad = (width - len(title) - 2) // 2
        print("\n" + "=" * pad + f" {title} " + "=" * pad)
    else:
        print("\n" + "=" * width)


def show_skill(label="Current SKILL.md"):
    doc = db.user_skills.find_one({"user_id": USER_ID})
    if not doc:
        print(f"  [{label}] No skill profile found for {USER_ID}")
        return
    print(f"\n[{label}]  version={doc.get('version', 0)}")
    print("-" * 60)
    print(doc["skill_md"])
    print("-" * 60)


def say(query, wait=True):
    print(f"\n>>> {query}")
    try:
        resp = requests.post(
            f"{BACKEND}/interact/stream",
            json={"query": query, "userID": USER_ID, "room": ""},
            stream=True,
            timeout=60,
        )
        resp.raise_for_status()
    except Exception as e:
        print(f"[error] {e}")
        return

    print("[robot]  ", end="", flush=True)
    answer_buf = ""
    last_len   = 0

    for raw_line in resp.iter_lines():
        if not raw_line:
            continue
        line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
        if not line.startswith("data: "):
            continue
        try:
            event = json.loads(line[6:])
        except json.JSONDecodeError:
            continue

        etype = event.get("type", "")

        if etype == "intent":
            intent = event.get("intent", "")
            if intent in ("chat", "query", "interrupt"):
                for raw2 in resp.iter_lines():
                    if not raw2:
                        continue
                    line2 = raw2.decode("utf-8") if isinstance(raw2, bytes) else raw2
                    if not line2.startswith("data: "):
                        continue
                    try:
                        ev2 = json.loads(line2[6:])
                    except json.JSONDecodeError:
                        continue
                    if ev2.get("type") == "token":
                        print(ev2.get("content", ""), end="", flush=True)
                    elif ev2.get("type") == "done":
                        break
                break

        elif etype == "token":
            answer_buf += event.get("content", "")
            marker = '"answer":'
            idx = answer_buf.find(marker)
            if idx != -1:
                start = answer_buf.find('"', idx + len(marker))
                if start != -1:
                    start += 1
                    visible = []
                    i = start
                    while i < len(answer_buf):
                        c = answer_buf[i]
                        if c == '\\' and i + 1 < len(answer_buf):
                            nc = answer_buf[i + 1]
                            if nc == '"':
                                visible.append('"'); i += 2; continue
                            elif nc == 'n':
                                visible.append('\n'); i += 2; continue
                        elif c == '"':
                            break
                        visible.append(c)
                        i += 1
                    visible_str = "".join(visible)
                    new_part    = visible_str[last_len:]
                    if new_part:
                        print(new_part, end="", flush=True)
                        last_len = len(visible_str)

        elif etype == "done":
            break

    print()
    if wait:
        time.sleep(3)


def wait_for_bg(seconds=6):
    print(f"  [waiting {seconds}s for background skill update...]")
    time.sleep(seconds)


def run_demo():
    divider("SKILL.MD DEMO")
    print(f"User: {USER_ID}")
    print("This script demonstrates the three ways SKILL.md evolves.")

    input("\nPress Enter to start...\n")

    divider("BEFORE — initial state")
    show_skill("Before any interaction")

    divider("SCENARIO A — preference learning via update()")
    print("Step 1: robot recommends cola")
    say("I am thirsty")

    print("\nStep 2: user expresses dislike")
    say("I don't like cola, I prefer juice")
    wait_for_bg(8)

    print("\nStep 3: robot should no longer recommend cola")
    say("I am thirsty again")

    divider("SKILL.MD AFTER SCENARIO A")
    show_skill("After preference update")
    input("\nPress Enter to continue to Scenario B...\n")

    divider("SCENARIO B — fill_gap: AI generates new rule")
    print("Step 1: user requests unavailable item")
    say("I want cheese please")
    wait_for_bg(8)

    print("\nStep 2: check that a new rule was inserted")
    show_skill("After fill_gap")

    print("\nStep 3: ask again — robot should now handle it better")
    say("I want cheese")

    divider("SCENARIO C — FAISS chunk compression")
    doc = db.user_skills.find_one({"user_id": USER_ID})
    if doc:
        skill_md    = doc["skill_md"]
        full_tokens = len(skill_md.split()) * 1.3
        chunks      = list(db.skill_chunks.find({"user_id": USER_ID}).limit(2))
        if chunks:
            top2_content = "\n\n".join(c["content"] for c in chunks)
            top2_tokens  = len(top2_content.split()) * 1.3
            compression  = (1 - top2_tokens / full_tokens) * 100
            print(f"\n  Full SKILL.md : {len(skill_md):4d} chars  ~{full_tokens:.0f} tokens")
            print(f"  Top-2 chunks  : {len(top2_content):4d} chars  ~{top2_tokens:.0f} tokens")
            print(f"  Compression   : {compression:.0f}%")
            print(f"\n  Top-2 chunks injected into LLM prompt:")
            print("-" * 60)
            print(top2_content)
            print("-" * 60)
        else:
            print("  No chunks found. Run migrate_skills.py first.")
    else:
        print("  No SKILL.md found.")

    divider("DEMO COMPLETE")
    print(f"  SKILL.md version: {doc.get('version', 0) if doc else 'N/A'}")
    print("  MongoDB collection: robot_rag_db.user_skills")
    print("  FAISS chunks: robot_rag_db.skill_chunks")


if __name__ == "__main__":
    run_demo()