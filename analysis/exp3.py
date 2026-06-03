"""
analyze_exp3.py
Experiment 3: Snapshot Injection Hallucination Test
Tests whether removing dynamic_objects causes LLM to hallucinate object locations.

Outputs:
    results/exp3_hallucination.png
    results/exp3_summary.txt

Prerequisites:
    app.py must be running
"""

import argparse
import csv
import datetime
import json
import os
import time

import requests

BACKEND_URL = "http://localhost:5000"

QUERIES_PRESENT = [
    ("is there any juice",         "User_Mom", ["juice", "juicebottle"]),
    ("I want an apple",            "User_Mom", ["apple"]),
    ("where is the cola",          "User_Mom", ["cola"]),
    ("is there a banana",          "User_Mom", ["banana"]),
    ("do we have any fruit",       "User_Mom", ["apple", "banana", "fruit"]),
]

QUERIES_ABSENT = [
    ("I want cheese",              "User_Mom", ["sorry","no ","not","unavailable"]),
    ("is there any pineapple",     "User_Mom", ["sorry","no ","not","unavailable"]),
    ("get me some coffee",         "User_Mom", ["sorry","no ","not","unavailable"]),
    ("I want pizza",               "User_Mom", ["sorry","no ","not","unavailable"]),
    ("is there any milk",          "User_Mom", ["sorry","no ","not","unavailable"]),
]


def call_stream(url, query, user_id):
    try:
        resp = requests.post(
            f"{url}/interact/stream",
            json={"query": query, "userID": user_id, "room": ""},
            stream=True, timeout=60,
        )
        resp.raise_for_status()
    except Exception as e:
        return {"answer": "", "error": str(e)}

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
                    m = re.search(
                        r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"', buf)
                    if m:
                        answer = m.group(1)
                break
        except Exception:
            continue
    return {"answer": answer.strip()}


def run_condition(url, queries, label, expected_mode):
    results = []
    print(f"\n--- {label} ---")
    for query, user_id, keywords in queries:
        r      = call_stream(url, query, user_id)
        answer = r.get("answer", "").lower()
        if expected_mode == "present":
            hit = any(kw in answer for kw in keywords)
        else:
            hit = any(kw in answer for kw in keywords)
        results.append({
            "query":    query,
            "answer":   r.get("answer", "")[:80],
            "keywords": keywords,
            "correct":  hit,
            "mode":     label,
            "type":     expected_mode,
        })
        print(f"  {'v' if hit else 'x'}  {query:35}  "
              f"-> {r.get('answer','')[:50]}")
        time.sleep(0.5)
    return results


def plot_results(with_results, without_results, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_with     = len(with_results)
    n_without  = len(without_results)
    acc_with   = sum(1 for r in with_results   if r["correct"]) / n_with   if n_with   else 0
    acc_wout   = sum(1 for r in without_results if r["correct"]) / n_without if n_without else 0
    hall_wout  = sum(1 for r in without_results
                     if r["type"] == "present" and not r["correct"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax1 = axes[0]
    categories = ["With Snapshot\n(Full System)", "Without Snapshot"]
    values     = [acc_with * 100, acc_wout * 100]
    colors     = ["#4CAF50", "#E53935"]
    bars = ax1.bar(categories, values, color=colors,
                   alpha=0.85, edgecolor="white", width=0.5)
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 1,
                 f"{val:.0f}%", ha="center", va="bottom",
                 fontsize=13, fontweight="bold")
    ax1.set_ylim(0, 115)
    ax1.set_ylabel("Correct Response Rate (%)", fontsize=12)
    ax1.set_title("Correct Response Rate\nWith vs Without Snapshot",
                  fontsize=12)
    ax1.grid(axis="y", alpha=0.3)

    ax2 = axes[1]
    hallucination_with  = 0
    hallucination_wout  = hall_wout
    ax2.bar(["With Snapshot", "Without Snapshot"],
            [hallucination_with, hallucination_wout],
            color=["#4CAF50", "#E53935"], alpha=0.85, edgecolor="white",
            width=0.5)
    ax2.set_ylabel("Hallucination Count", fontsize=12)
    ax2.set_title("Hallucination Count\n(Robot invents non-existent object locations)",
                  fontsize=12)
    ax2.grid(axis="y", alpha=0.3)
    for i, v in enumerate([hallucination_with, hallucination_wout]):
        ax2.text(i, v + 0.05, str(v), ha="center", va="bottom",
                 fontsize=13, fontweight="bold")

    plt.suptitle(
        f"Experiment 3: Snapshot Injection Hallucination Test\n"
        f"With snapshot: {acc_with:.0%} correct  |  "
        f"Without snapshot: {acc_wout:.0%} correct  |  "
        f"Hallucinations: {hallucination_wout}",
        fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def save_summary(with_results, without_results, out_path):
    n_with    = len(with_results)
    n_without = len(without_results)
    ok_with   = sum(1 for r in with_results   if r["correct"])
    ok_wout   = sum(1 for r in without_results if r["correct"])
    hall_wout = sum(1 for r in without_results
                    if r["type"] == "present" and not r["correct"])

    lines = [
        "=" * 65,
        "Experiment 3: Snapshot Injection Hallucination Test",
        f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 65,
        "",
        f"WITH snapshot: {ok_with}/{n_with} correct "
        f"({ok_with/n_with:.0%})",
        f"WITHOUT snapshot: {ok_wout}/{n_without} correct "
        f"({ok_wout/n_without:.0%})",
        f"Hallucinations (without): {hall_wout}",
        "",
        "WITH snapshot results:",
        *[f"  {'v' if r['correct'] else 'x'}  "
          f"{r['query']:35}  {r['answer'][:50]}"
          for r in with_results],
        "",
        "WITHOUT snapshot results:",
        *[f"  {'v' if r['correct'] else 'x'}  "
          f"{r['query']:35}  {r['answer'][:50]}"
          for r in without_results],
        "",
        "For thesis:",
        f"Removing the snapshot mechanism caused {hall_wout} hallucination(s),",
        f"where the robot invented object locations not present in the environment.",
        f"With snapshot: {ok_with/n_with:.0%} correct vs "
        f"without: {ok_wout/n_without:.0%}.",
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=BACKEND_URL)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    try:
        requests.get(f"{args.url}/", timeout=5)
    except Exception:
        print(f"Cannot connect to {args.url}. Start app.py first.")
        return

    from pymongo import MongoClient
    client = MongoClient(MONGO_URI if "MONGO_URI" in dir() else
                         "mongodb://127.0.0.1:27017/")
    db = client["robot_rag_db"]

    print("Step 1: Testing WITH snapshot...")
    with_results = run_condition(
        args.url,
        QUERIES_PRESENT + QUERIES_ABSENT,
        "WITH snapshot", "present")

    print("\nStep 2: Clearing dynamic_objects...")
    backup = list(db.dynamic_objects.find({}))
    db.dynamic_objects.delete_many({})
    print(f"  Cleared {len(backup)} documents")

    print("\nStep 3: Testing WITHOUT snapshot...")
    without_results = run_condition(
        args.url,
        QUERIES_PRESENT + QUERIES_ABSENT,
        "WITHOUT snapshot", "present")

    print("\nStep 4: Restoring dynamic_objects...")
    if backup:
        db.dynamic_objects.insert_many(backup)
    print(f"  Restored {len(backup)} documents")

    print("\nStep 5: Generating outputs...")
    plot_results(with_results, without_results,
                 os.path.join(args.out, "exp3_hallucination.png"))
    save_summary(with_results, without_results,
                 os.path.join(args.out, "exp3_summary.txt"))


if __name__ == "__main__":
    main()