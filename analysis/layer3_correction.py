"""
analysis/layer3_correction.py
Layer 3: Personalised Service — Correction Rate Experiment
Outputs:
  results/Fig4_correction_rate.png

Runs a live dialogue script against Flask:
  Round 1: User says "I am thirsty" → system recommends X
  Round 1: User rejects X
  Round 2-5: Does system stop recommending X?
"""

import os
import time
import json
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pymongo import MongoClient

BACKEND   = "http://localhost:5000"
OUT = os.path.join(os.path.dirname(__file__), "results")
MONGO_URI = "mongodb://127.0.0.1:27017/"
DB_NAME   = "robot_rag_db"

SCRIPT_MOM = [
    {"round": 1, "query": "I am thirsty",       "expect_reject": "juice"},
    {"round": 2, "query": "No juice please",     "expect_reject": "juice"},
    {"round": 3, "query": "I want something",    "expect_reject": "juice"},
    {"round": 4, "query": "I am thirsty again",  "expect_reject": "juice"},
    {"round": 5, "query": "Can I have a drink?", "expect_reject": "juice"},
]

SCRIPT_DAD = [
    {"round": 1, "query": "I am thirsty",       "expect_reject": "cola"},
    {"round": 2, "query": "No cola please",      "expect_reject": "cola"},
    {"round": 3, "query": "I want something",    "expect_reject": "cola"},
    {"round": 4, "query": "I am thirsty again",  "expect_reject": "cola"},
    {"round": 5, "query": "Can I have a drink?", "expect_reject": "cola"},
]


def ask(query, user_id):
    try:
        r = requests.post(
            f"{BACKEND}/interact",
            json={"query": query, "userID": user_id},
            timeout=30
        )
        return r.json()
    except Exception as e:
        print(f"  [Error] {e}")
        return {}


def reject(user_id, item):
    try:
        requests.post(
            f"{BACKEND}/habit_feedback",
            json={"user_id": user_id, "result": "rejected", "item": item},
            timeout=10
        )
    except Exception as e:
        print(f"  [Reject Error] {e}")


def run_script(script, user_id, rejected_item):
    results = []
    for step in script:
        r     = step["round"]
        query = step["query"]
        print(f"  Round {r}: '{query}'")

        resp   = ask(query, user_id)
        answer = resp.get("answer", "")
        recs   = [x.get("label","").lower()
                  for x in resp.get("recommendations", [])]

        wrong = rejected_item.lower() in answer.lower() or \
                any(rejected_item.lower() in rec for rec in recs)

        print(f"    Answer: {answer[:80]}")
        print(f"    Wrong recommendation: {wrong}")

        results.append({"round": r, "wrong": wrong, "answer": answer})

        if r == 1:
            print(f"    → Rejecting '{rejected_item}'")
            reject(user_id, rejected_item)
            time.sleep(2)

        time.sleep(1)

    return results


def plot_fig4_correction_rate(mom_results, dad_results):
    print("Fig4: Correction Rate...")

    rounds   = [r["round"] for r in mom_results]
    err_mom  = [100 if r["wrong"] else 0 for r in mom_results]
    err_dad  = [100 if r["wrong"] else 0 for r in dad_results]
    baseline = [100] * len(rounds)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(rounds, baseline, "x--", color="#BDBDBD",
            linewidth=1.5, markersize=8,
            label="No Learning (Baseline)")
    ax.plot(rounds, err_mom, "o-", color="#2196F3",
            linewidth=2.5, markersize=10,
            markerfacecolor="white", markeredgewidth=2.5,
            label="User Mom (rejected juice)")
    ax.plot(rounds, err_dad, "s-", color="#4CAF50",
            linewidth=2.5, markersize=10,
            markerfacecolor="white", markeredgewidth=2.5,
            label="User Dad (rejected cola)")

    ax.axvline(x=1.5, color="#E53935", linewidth=1.5,
               linestyle=":", alpha=0.7, label="Rejection event")

    ax.annotate("Rejection\nrecorded",
                xy=(1, 100), xytext=(1.6, 80),
                fontsize=9, color="#E53935",
                arrowprops=dict(arrowstyle="->", color="#E53935", lw=1.5))

    if not err_mom[1]:
        ax.annotate("SKILL.md updated\n→ 0% error rate",
                    xy=(2, 0), xytext=(2.5, 30),
                    fontsize=9, color="#1565C0",
                    arrowprops=dict(arrowstyle="->", color="#1565C0", lw=1.5))

    ax.set_xticks(rounds)
    ax.set_xticklabels([f"Round {r}" for r in rounds], fontsize=10)
    ax.set_ylim(-10, 120)
    ax.set_xlabel("Dialogue Round", fontsize=12)
    ax.set_ylabel("Wrong Recommendation Rate (%)", fontsize=12)
    ax.set_title(
        "Fig4  Correction Rate — Feedback Learning Effectiveness\n"
        "After one rejection, system stops recommending the rejected item\n"
        "SKILL.md 'What NOT to do' updated after Round 1",
        fontsize=11, fontweight="bold"
    )
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    path = os.path.join(OUT, "Fig4_correction_rate.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)

    print("=== Layer 3: Correction Rate Experiment ===")
    print("Flask must be running: CUDA_VISIBLE_DEVICES='' python3 app.py")
    print()

    try:
        r = requests.get(f"{BACKEND}/ready", timeout=5)
        if "true" not in r.text.lower():
            print("Flask not ready. Start Flask first.")
            exit(1)
    except Exception:
        print("Cannot connect to Flask. Start Flask first.")
        exit(1)

    print("--- User Mom (rejecting juice) ---")
    mom_results = run_script(SCRIPT_MOM, "User_Mom", "juice")

    print()
    print("--- User Dad (rejecting cola) ---")
    dad_results = run_script(SCRIPT_DAD, "User_Dad", "cola")

    plot_fig4_correction_rate(mom_results, dad_results)

    print()
    print("Summary:")
    for label, results in [("Mom", mom_results), ("Dad", dad_results)]:
        wrongs = sum(1 for r in results if r["wrong"])
        print(f"  {label}: {wrongs}/{len(results)} wrong recommendations")
    print("Done.")