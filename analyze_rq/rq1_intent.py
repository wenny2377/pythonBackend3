import argparse
import csv
import datetime
import json
import os
import sys
import time

import requests

BACKEND_URL = "http://localhost:5000"

QUERIES = [
    ("I am thirsty",                          "User_Mom", "service"),
    ("I want something to drink",              "User_Mom", "service"),
    ("I am tired, I need to sit down",         "User_Mom", "service"),
    ("I am hungry",                            "User_Mom", "service"),
    ("get me something to eat",                "User_Mom", "service"),
    ("bring me the remote",                    "User_Mom", "service"),
    ("I wanna drink",                          "User_Dad", "service"),
    ("something refreshing please",            "User_Dad", "service"),
    ("are there any fruits",                   "User_Mom", "query"),
    ("what drinks do we have",                 "User_Mom", "query"),
    ("where is the remote",                    "User_Mom", "query"),
    ("is there any cola",                      "User_Dad", "query"),
    ("where is mom",                           "User_Dad", "query"),
    ("I hate my boss",                         "User_Mom", "chat"),
    ("I feel great today",                     "User_Dad", "chat"),
    ("good morning",                           "User_Mom", "chat"),
    ("thank you so much",                      "User_Mom", "chat"),
    ("I am exhausted",                         "User_Dad", "chat"),
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
        return {"error": str(e), "intent_type": ""}

    intent_type = ""
    for raw in resp.iter_lines():
        if not raw:
            continue
        line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        if not line.startswith("data: "):
            continue
        try:
            ev = json.loads(line[6:])
            if ev.get("type") == "intent":
                intent_type = ev.get("intent", "")
            if ev.get("type") == "done":
                break
        except Exception:
            continue

    return {"intent_type": intent_type}


def run(url):
    results = []
    print(f"\nRunning {len(QUERIES)} queries\n")
    print(f"{'#':>3}  {'User':10}  {'Query':40}  {'Expected':8}  {'Actual':8}  {'OK':4}")
    print("-" * 80)

    for i, (query, user_id, expected) in enumerate(QUERIES):
        result   = call_stream(url, query, user_id)
        actual   = result.get("intent_type", "error")
        correct  = actual == expected

        results.append({
            "index":    i + 1,
            "query":    query,
            "user_id":  user_id,
            "expected": expected,
            "actual":   actual,
            "correct":  correct,
        })

        ok = "v" if correct else "x"
        print(f"{i+1:>3}  {user_id:10}  {query:40}  {expected:8}  {actual:8}  {ok:4}")
        time.sleep(0.5)

    return results


def save_csv(results, path):
    fields = ["index", "query", "user_id", "expected", "actual", "correct"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)
    print(f"  CSV saved: {path}")


def save_summary(results, path):
    n        = len(results)
    n_ok     = sum(1 for r in results if r["correct"])
    acc      = n_ok / n if n > 0 else 0.0

    by_type = {}
    for et in ("service", "query", "chat"):
        sub = [r for r in results if r["expected"] == et]
        if sub:
            ok = sum(1 for r in sub if r["correct"])
            by_type[et] = (ok, len(sub))

    errors = [r for r in results if not r["correct"]]

    lines = [
        "=" * 65,
        "RQ1: Intent Classification Accuracy",
        f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 65,
        "",
        f"Total queries    : {n}",
        f"Correct          : {n_ok}",
        f"Overall accuracy : {acc:.0%}",
        "",
        "By intent class:",
        *[f"  {et:8s}: {ok}/{total} ({ok/total:.0%})" for et, (ok, total) in by_type.items()],
        "",
        "Errors:",
        *(
            [f"  x  {r['user_id']:10}  {r['query']:40}  expected={r['expected']}  actual={r['actual']}"
             for r in errors]
            if errors else ["  none"]
        ),
        "",
        "For thesis:",
        f"The LLM-based intent classifier correctly classified {n_ok} of {n}",
        f"test queries (accuracy = {acc:.0%}) across three intent classes",
        f"(service, query, chat), without any handcrafted keyword rules.",
        f"Target threshold: 85%. Result: {'PASSED' if acc >= 0.85 else 'BELOW THRESHOLD'}.",
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Summary saved: {path}")
    print(f"\n  Overall accuracy: {n_ok}/{n} = {acc:.0%}  ({'PASSED' if acc >= 0.85 else 'BELOW 85%'})")


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

    results = run(args.url)
    save_csv(results,     os.path.join(args.out, "rq1_results.csv"))
    save_summary(results, os.path.join(args.out, "rq1_summary.txt"))


if __name__ == "__main__":
    main()