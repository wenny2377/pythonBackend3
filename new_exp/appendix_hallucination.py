"""
thesis_appendix_hallucination.py
Appendix 2: Snapshot Injection Hallucination Test

Validates the necessity of the dynamic_objects snapshot mechanism:
  WITH snapshot : LLM answers correctly using real object locations
  WITHOUT snapshot : LLM hallucinates object locations

Requires: Flask (app.py) running with dynamic_objects populated.

Usage:
  python3 thesis_appendix_hallucination.py
  python3 thesis_appendix_hallucination.py --url http://localhost:5000 --out results/

Output:
  results/appendix_hallucination.png
  results/appendix_hallucination_detail.png
  results/appendix_hallucination_summary.txt
"""

import argparse
import datetime
import json
import os
import re
import time

import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pymongo import MongoClient

BACKEND_URL = "http://localhost:5000"
MONGO_URI   = "mongodb://127.0.0.1:27017/"
DB_NAME     = "robot_rag_db"

# Objects that EXIST in the scene (dynamic_objects should have these)
QUERIES_PRESENT = [
    ("is there any juice",       "User_Mom",
     ["juice", "juicebottle", "fridge", "kitchen"]),
    ("I want an apple",          "User_Mom",
     ["apple", "kitchen", "fridge"]),
    ("where is the cola",        "User_Mom",
     ["cola", "fridge", "kitchen"]),
    ("is there a banana",        "User_Mom",
     ["banana", "kitchen"]),
    ("do we have any fruit",     "User_Mom",
     ["apple", "banana", "fruit", "kitchen"]),
]

# Objects that DO NOT EXIST in the scene
QUERIES_ABSENT = [
    ("I want some cheese",       "User_Mom",
     ["sorry", "no ", "not ", "unavailable", "don't have", "cannot find"]),
    ("is there any pineapple",   "User_Mom",
     ["sorry", "no ", "not ", "unavailable", "don't have", "cannot find"]),
    ("can I have some coffee",   "User_Mom",
     ["sorry", "no ", "not ", "unavailable", "don't have", "cannot find"]),
    ("I want pizza",             "User_Mom",
     ["sorry", "no ", "not ", "unavailable", "don't have", "cannot find"]),
    ("is there any milk",        "User_Mom",
     ["sorry", "no ", "not ", "unavailable", "don't have", "cannot find"]),
]


def call_stream(url, query, user_id):
    """Call Flask /interact/stream and return robot answer."""
    try:
        resp = requests.post(
            f"{url}/interact/stream",
            json={"query": query, "userID": user_id, "room": ""},
            stream=True, timeout=60)
        resp.raise_for_status()
    except Exception as e:
        return {"answer": "", "error": str(e)}

    answer = buf = ""
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
                    m = re.search(
                        r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"', buf)
                    if m:
                        answer = m.group(1)
                break
        except Exception:
            continue
    return {"answer": answer.strip()}


def run_condition(url, queries, label, query_type):
    """Run all queries under one condition and collect results."""
    results = []
    print(f"\n{'─'*55}")
    print(f"Condition: {label}")
    print(f"{'─'*55}")

    for query, user_id, keywords in queries:
        r      = call_stream(url, query, user_id)
        answer = r.get("answer", "").lower()
        hit    = any(kw in answer for kw in keywords)

        results.append({
            "query":      query,
            "answer":     r.get("answer", "")[:100],
            "keywords":   keywords,
            "correct":    hit,
            "type":       query_type,   # "present" or "absent"
            "condition":  label,
        })
        status = "✓" if hit else "✗"
        print(f"  {status}  [{query_type.upper():7}] {query:35}"
              f" → {r.get('answer','')[:50]}")
        time.sleep(0.6)

    return results


def is_hallucination(result):
    """
    A hallucination occurs when:
      - query_type is 'absent' (object does not exist)
      - system claims the object IS present (correct=False means
        it did NOT say sorry/unavailable → it hallucinated a location)
    """
    return result["type"] == "absent" and not result["correct"]


def plot_main(with_r, without_r, out):
    """Main comparison figure."""
    n_w    = len(with_r)
    n_wo   = len(without_r)
    acc_w  = sum(1 for r in with_r    if r["correct"]) / (n_w  or 1)
    acc_wo = sum(1 for r in without_r if r["correct"]) / (n_wo or 1)
    hall_w  = sum(1 for r in with_r    if is_hallucination(r))
    hall_wo = sum(1 for r in without_r if is_hallucination(r))

    # Break down by query type
    present_w  = [r for r in with_r    if r["type"] == "present"]
    present_wo = [r for r in without_r if r["type"] == "present"]
    absent_w   = [r for r in with_r    if r["type"] == "absent"]
    absent_wo  = [r for r in without_r if r["type"] == "absent"]

    acc_present_w  = sum(1 for r in present_w  if r["correct"]) / (len(present_w)  or 1)
    acc_present_wo = sum(1 for r in present_wo if r["correct"]) / (len(present_wo) or 1)
    acc_absent_w   = sum(1 for r in absent_w   if r["correct"]) / (len(absent_w)   or 1)
    acc_absent_wo  = sum(1 for r in absent_wo  if r["correct"]) / (len(absent_wo)  or 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    fig.suptitle(
        "Appendix 2: Snapshot Injection Hallucination Test\n"
        "Dynamic Object Snapshot vs No Snapshot",
        fontsize=13, fontweight="bold")

    # Panel 1: Overall accuracy
    ax = axes[0]
    cats   = ["With Snapshot\n(Full System)", "Without\nSnapshot"]
    vals   = [acc_w * 100, acc_wo * 100]
    colors = ["#4CAF50", "#E53935"]
    bars   = ax.bar(cats, vals, color=colors, alpha=0.85,
                    edgecolor="white", width=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f"{v:.0f}%", ha="center", fontsize=14,
                fontweight="bold")
    ax.set_ylim(0, 120)
    ax.set_ylabel("Correct Response Rate (%)", fontsize=11)
    ax.set_title("Overall Accuracy", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Panel 2: By query type
    ax = axes[1]
    x  = np.array([0, 1])
    w  = 0.3
    ax.bar(x - w/2,
           [acc_present_w * 100, acc_absent_w * 100],
           w, color="#4CAF50", alpha=0.85, edgecolor="white",
           label="With Snapshot")
    ax.bar(x + w/2,
           [acc_present_wo * 100, acc_absent_wo * 100],
           w, color="#E53935", alpha=0.85, edgecolor="white",
           label="Without Snapshot")
    for i, (vw, vwo) in enumerate(
            [(acc_present_w, acc_present_wo),
             (acc_absent_w,  acc_absent_wo)]):
        ax.text(x[i] - w/2, vw*100  + 1.5, f"{vw*100:.0f}%",
                ha="center", fontsize=9)
        ax.text(x[i] + w/2, vwo*100 + 1.5, f"{vwo*100:.0f}%",
                ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(["Object EXISTS\n(should confirm)",
                        "Object ABSENT\n(should deny)"],
                       fontsize=9)
    ax.set_ylim(0, 130)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("Accuracy by Query Type", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Panel 3: Hallucination count
    ax = axes[2]
    bars = ax.bar(["With Snapshot", "Without\nSnapshot"],
                  [hall_w, hall_wo],
                  color=["#4CAF50", "#E53935"],
                  alpha=0.85, edgecolor="white", width=0.5)
    for bar, v in zip(bars, [hall_w, hall_wo]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                str(v), ha="center", fontsize=14, fontweight="bold")
    ax.set_ylabel("Hallucination Count", fontsize=11)
    ax.set_title("Hallucinations\n(invents non-existent locations)",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(hall_wo + 1, 2))
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(out, "appendix_hallucination.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_detail(with_r, without_r, out):
    """Detailed per-query result comparison."""
    all_queries = [r["query"] for r in with_r]
    n           = len(all_queries)
    w_correct   = [r["correct"] for r in with_r]
    wo_correct  = [r["correct"] for r in without_r]

    fig, ax = plt.subplots(figsize=(13, max(4, n * 0.55 + 2)))

    y = np.arange(n)
    h = 0.35

    for i, (wc, woc, q) in enumerate(
            zip(w_correct, wo_correct, all_queries)):
        ax.barh(y[i] + h/2, 1, h,
                color="#4CAF50" if wc  else "#E53935",
                alpha=0.75, edgecolor="white")
        ax.barh(y[i] - h/2, 1, h,
                color="#4CAF50" if woc else "#E53935",
                alpha=0.75, edgecolor="white")
        ax.text(1.05, y[i] + h/2,
                "✓" if wc  else "✗ Hallucinated",
                va="center", fontsize=9,
                color="#1B5E20" if wc  else "#B71C1C")
        ax.text(1.05, y[i] - h/2,
                "✓" if woc else "✗ Hallucinated",
                va="center", fontsize=9,
                color="#1B5E20" if woc else "#B71C1C")

    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"Q{i+1}: {q[:40]}" for i, q in enumerate(all_queries)],
        fontsize=9)
    ax.set_xlim(0, 2.5)
    ax.set_xticks([])
    ax.set_title(
        "Appendix 2 — Per-query Result Detail\n"
        "(upper bar = With Snapshot, lower bar = Without Snapshot)",
        fontsize=11, fontweight="bold")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4CAF50", alpha=0.75, label="Correct"),
        Patch(facecolor="#E53935", alpha=0.75, label="Hallucinated / Wrong"),
        Patch(facecolor="#4CAF50", alpha=0.75, label="With Snapshot (upper)"),
        Patch(facecolor="#E53935", alpha=0.75, label="Without Snapshot (lower)"),
    ]
    ax.legend(handles=legend_elements, fontsize=8,
              loc="lower right", ncol=2)
    ax.grid(axis="x", alpha=0.2)
    plt.tight_layout()

    path = os.path.join(out, "appendix_hallucination_detail.png")
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def save_summary(with_r, without_r, out):
    """Save text summary."""
    ok_w    = sum(1 for r in with_r    if r["correct"])
    ok_wo   = sum(1 for r in without_r if r["correct"])
    hall_wo = sum(1 for r in without_r if is_hallucination(r))
    hall_w  = sum(1 for r in with_r    if is_hallucination(r))

    lines = [
        "=" * 65,
        "Appendix 2: Snapshot Injection Hallucination Test",
        f"Generated : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 65, "",
        "Summary:",
        f"  WITH snapshot    : {ok_w}/{len(with_r)} correct"
        f" ({ok_w/len(with_r):.0%})"
        f"  Hallucinations={hall_w}",
        f"  WITHOUT snapshot : {ok_wo}/{len(without_r)} correct"
        f" ({ok_wo/len(without_r):.0%})"
        f"  Hallucinations={hall_wo}",
        "",
        "WITH snapshot results:",
    ]
    for r in with_r:
        status = "✓" if r["correct"] else "✗"
        hall   = " [HALLUCINATED]" if is_hallucination(r) else ""
        lines.append(
            f"  {status}  [{r['type'].upper():7}]"
            f" {r['query']:35} → {r['answer'][:55]}{hall}")

    lines += ["", "WITHOUT snapshot results:"]
    for r in without_r:
        status = "✓" if r["correct"] else "✗"
        hall   = " [HALLUCINATED]" if is_hallucination(r) else ""
        lines.append(
            f"  {status}  [{r['type'].upper():7}]"
            f" {r['query']:35} → {r['answer'][:55]}{hall}")

    lines += [
        "",
        "Thesis interpretation:",
        f"  Removing the dynamic_objects snapshot mechanism caused",
        f"  {hall_wo} hallucination(s), where the robot invented",
        f"  object locations not present in the environment.",
        f"  With snapshot: {ok_w/len(with_r):.0%} correct.",
        f"  Without snapshot: {ok_wo/len(without_r):.0%} correct.",
        f"  This validates the necessity of the real-time snapshot",
        f"  injection mechanism for grounded home robot responses.",
    ]

    path = os.path.join(out, "appendix_hallucination_summary.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {path}")


def check_dynamic_objects(mongo_uri, db_name):
    """Check if dynamic_objects has data."""
    try:
        db  = MongoClient(mongo_uri)[db_name]
        cnt = db.dynamic_objects.count_documents({})
        print(f"  dynamic_objects: {cnt} documents")
        if cnt == 0:
            print("  [Warning] dynamic_objects is empty.")
            print("  Run Unity HabitExp first to populate dynamic objects.")
        return db, cnt
    except Exception as e:
        print(f"  [Error] MongoDB connection failed: {e}")
        return None, 0


def main():
    parser = argparse.ArgumentParser(
        description="Appendix 2: Hallucination Test (requires Flask)")
    parser.add_argument("--url",  default=BACKEND_URL,
                        help="Flask backend URL")
    parser.add_argument("--out",  default="results",
                        help="Output directory")
    parser.add_argument("--mongo", default=MONGO_URI,
                        help="MongoDB URI")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Check Flask
    try:
        requests.get(f"{args.url}/", timeout=5)
    except Exception:
        print(f"[Error] Cannot connect to {args.url}")
        print("  Start Flask first: python3 app.py")
        return

    print(f"[Appendix 2] Hallucination Test")
    print(f"  Backend : {args.url}")
    print(f"  Output  : {args.out}/")

    # Check MongoDB
    db, cnt = check_dynamic_objects(args.mongo, DB_NAME)
    if db is None:
        return

    # Backup dynamic_objects
    backup = list(db.dynamic_objects.find({}))
    print(f"\n  Backed up {len(backup)} dynamic_objects documents")

    try:
        # Step 1: Test WITH snapshot
        print("\n[Step 1] Testing WITH snapshot (full system)...")
        with_r = run_condition(
            args.url,
            QUERIES_PRESENT + QUERIES_ABSENT,
            "WITH snapshot", "present")

        # Re-classify absent queries
        n_present = len(QUERIES_PRESENT)
        for i, r in enumerate(with_r):
            r["type"] = "present" if i < n_present else "absent"

        # Step 2: Clear dynamic_objects
        print("\n[Step 2] Clearing dynamic_objects...")
        db.dynamic_objects.delete_many({})
        print(f"  Cleared {len(backup)} documents")
        time.sleep(1.0)

        # Step 3: Test WITHOUT snapshot
        print("\n[Step 3] Testing WITHOUT snapshot...")
        without_r = run_condition(
            args.url,
            QUERIES_PRESENT + QUERIES_ABSENT,
            "WITHOUT snapshot", "present")

        for i, r in enumerate(without_r):
            r["type"] = "present" if i < n_present else "absent"

    finally:
        # Always restore
        print("\n[Step 4] Restoring dynamic_objects...")
        if backup:
            db.dynamic_objects.insert_many(backup)
        print(f"  Restored {len(backup)} documents")

    # Generate outputs
    print("\n[Step 5] Generating plots...")
    plot_main(with_r, without_r, args.out)
    plot_detail(with_r, without_r, args.out)
    save_summary(with_r, without_r, args.out)

    # Print quick stats
    ok_w  = sum(1 for r in with_r    if r["correct"])
    ok_wo = sum(1 for r in without_r if r["correct"])
    hall  = sum(1 for r in without_r if is_hallucination(r))
    print(f"\n[Result]")
    print(f"  With snapshot    : {ok_w}/{len(with_r)} correct")
    print(f"  Without snapshot : {ok_wo}/{len(without_r)} correct")
    print(f"  Hallucinations   : {hall}")
    print(f"\n[Done]")
    print(f"  appendix_hallucination.png")
    print(f"  appendix_hallucination_detail.png")
    print(f"  appendix_hallucination_summary.txt")


if __name__ == "__main__":
    main()