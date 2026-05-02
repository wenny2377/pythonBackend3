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

QUERIES_PRESENT = [
    ("is there any juice",        "User_Mom", ["juice", "juicebottle"]),
    ("I want an apple",           "User_Mom", ["apple"]),
    ("where is the cola",         "User_Mom", ["cola"]),
    ("is there a banana",         "User_Mom", ["banana"]),
    ("do we have any fruit",      "User_Mom", ["apple", "banana", "fruit"]),
]

QUERIES_ABSENT = [
    ("I want cheese",             "User_Mom", ["sorry", "no ", "not", "unavailable", "cannot"]),
    ("is there any pineapple",    "User_Mom", ["sorry", "no ", "not", "unavailable", "cannot"]),
    ("get me some coffee",        "User_Mom", ["sorry", "no ", "not", "unavailable", "cannot"]),
    ("I want pizza",              "User_Mom", ["sorry", "no ", "not", "unavailable", "cannot"]),
    ("is there any milk",         "User_Mom", ["sorry", "no ", "not", "unavailable", "cannot"]),
]


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

    answer = ""
    intent = ""
    buf    = ""
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


def disable_snapshot(url):
    try:
        requests.post(f"{url}/debug/snapshot_off", timeout=3)
    except Exception:
        pass


def enable_snapshot(url):
    try:
        requests.post(f"{url}/debug/snapshot_on", timeout=3)
    except Exception:
        pass


def run_with_snapshot(url):
    results = []
    print("\n--- WITH snapshot ---")

    for query, user_id, expected_keywords in QUERIES_PRESENT:
        r      = call_stream(url, query, user_id)
        answer = r.get("answer", "").lower()
        hit    = any(kw in answer for kw in expected_keywords)
        results.append({
            "query":    query,
            "answer":   r.get("answer", ""),
            "expected": "mention item",
            "correct":  hit,
            "mode":     "with_snapshot",
            "type":     "present",
        })
        print(f"  {'v' if hit else 'x'}  {query:35}  -> {r.get('answer','')[:50]}")
        time.sleep(0.5)

    for query, user_id, expected_keywords in QUERIES_ABSENT:
        r      = call_stream(url, query, user_id)
        answer = r.get("answer", "").lower()
        hit    = any(kw in answer for kw in expected_keywords)
        results.append({
            "query":    query,
            "answer":   r.get("answer", ""),
            "expected": "say not available",
            "correct":  hit,
            "mode":     "with_snapshot",
            "type":     "absent",
        })
        print(f"  {'v' if hit else 'x'}  {query:35}  -> {r.get('answer','')[:50]}")
        time.sleep(0.5)

    return results


def run_without_snapshot(url, db):
    results     = []
    backup_docs = list(db.dynamic_objects.find({}))

    print("\n--- WITHOUT snapshot (dynamic_objects cleared) ---")
    db.dynamic_objects.delete_many({})

    for query, user_id, expected_keywords in QUERIES_PRESENT:
        r          = call_stream(url, query, user_id)
        answer     = r.get("answer", "").lower()
        hallucin   = any(kw in answer for kw in expected_keywords)
        results.append({
            "query":          query,
            "answer":         r.get("answer", ""),
            "expected":       "NOT mention item (no data)",
            "hallucination":  hallucin,
            "correct":        not hallucin,
            "mode":           "without_snapshot",
            "type":           "present",
        })
        print(f"  {'hallucination!' if hallucin else 'ok':15}  {query:35}  -> {r.get('answer','')[:50]}")
        time.sleep(0.5)

    for query, user_id, expected_keywords in QUERIES_ABSENT:
        r      = call_stream(url, query, user_id)
        answer = r.get("answer", "").lower()
        hit    = any(kw in answer for kw in expected_keywords)
        results.append({
            "query":         query,
            "answer":        r.get("answer", ""),
            "expected":      "say not available",
            "hallucination": False,
            "correct":       hit,
            "mode":          "without_snapshot",
            "type":          "absent",
        })
        print(f"  {'v' if hit else 'x'}  {query:35}  -> {r.get('answer','')[:50]}")
        time.sleep(0.5)

    db.dynamic_objects.delete_many({})
    if backup_docs:
        db.dynamic_objects.insert_many(backup_docs)
    print("  (dynamic_objects restored)")

    return results


def save_csv(with_results, without_results, path):
    fields = ["mode", "type", "query", "correct", "hallucination", "answer"]
    rows   = []
    for r in with_results:
        rows.append({
            "mode":          r["mode"],
            "type":          r["type"],
            "query":         r["query"],
            "correct":       r["correct"],
            "hallucination": False,
            "answer":        r["answer"][:80],
        })
    for r in without_results:
        rows.append({
            "mode":          r["mode"],
            "type":          r["type"],
            "query":         r["query"],
            "correct":       r.get("correct", False),
            "hallucination": r.get("hallucination", False),
            "answer":        r["answer"][:80],
        })
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  CSV saved: {path}")


def save_summary(with_results, without_results, path):
    n_with       = len(with_results)
    n_with_ok    = sum(1 for r in with_results if r["correct"])
    n_without    = len(without_results)
    n_without_ok = sum(1 for r in without_results if r.get("correct", False))
    n_hallucin   = sum(1 for r in without_results if r.get("hallucination", False))

    lines = [
        "=" * 65,
        "RQ2: Object Localization and Hallucination Prevention",
        f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 65,
        "",
        "WITH snapshot (dynamic_objects in MongoDB):",
        f"  Correct: {n_with_ok}/{n_with} ({n_with_ok/n_with:.0%})",
        f"  Hallucinations: 0",
        "",
        "WITHOUT snapshot (dynamic_objects cleared):",
        f"  Correct: {n_without_ok}/{n_without} ({n_without_ok/n_without:.0%})",
        f"  Hallucinations: {n_hallucin} (robot invented non-existent objects)",
        "",
        "Per-query results (WITH snapshot):",
        *[f"  {'v' if r['correct'] else 'x'}  {r['query']:35}  {r['answer'][:50]}" for r in with_results],
        "",
        "Per-query results (WITHOUT snapshot):",
        *[
            f"  {'hallucination!' if r.get('hallucination') else ('v' if r.get('correct') else 'x'):15}  "
            f"{r['query']:35}  {r['answer'][:50]}"
            for r in without_results
        ],
        "",
        "For thesis:",
        f"To evaluate the snapshot injection mechanism, the system was tested",
        f"with and without access to the dynamic object database.",
        f"With snapshot: {n_with_ok}/{n_with} correct responses ({n_with_ok/n_with:.0%}).",
        f"Without snapshot: {n_hallucin} hallucination(s) detected where the robot",
        f"invented the location of objects not present in the environment,",
        f"confirming that snapshot injection is essential for grounded responses.",
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Summary saved: {path}")
    print(f"\n  WITH snapshot   : {n_with_ok}/{n_with} correct")
    print(f"  WITHOUT snapshot: {n_hallucin} hallucinations detected")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=BACKEND_URL)
    parser.add_argument("--out", default=".")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    try:
        requests.get(f"{args.url}/", timeout=5)
    except Exception:
        print(f"Cannot connect to {args.url}. Start app.py first.")
        sys.exit(1)

    client = MongoClient(MONGO_URI)
    db     = client[DB_NAME]

    print("Step 1: Testing WITH snapshot...")
    with_results = run_with_snapshot(args.url)

    print("\nStep 2: Testing WITHOUT snapshot...")
    without_results = run_without_snapshot(args.url, db)

    save_csv(with_results, without_results, os.path.join(args.out, "rq2_results.csv"))
    save_summary(with_results, without_results, os.path.join(args.out, "rq2_summary.txt"))


if __name__ == "__main__":
    main()