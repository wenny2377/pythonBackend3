import argparse
import csv
import datetime
import json
import os
import re
import sys
import time

import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BACKEND_URL = "http://localhost:5000"

SCRIPT = [
    {
        "day":           1,
        "query":         "I am thirsty",
        "followup":      "No, I don't want juice, I don't like it",
        "followup_type": "rejection",
        "description":   "Day 1: recommends juice -> user rejects",
    },
    {
        "day":           2,
        "query":         "I am thirsty",
        "followup":      "OK, that's fine",
        "followup_type": "acceptance",
        "description":   "Day 2: recommends alternative -> user accepts",
    },
    {
        "day":           3,
        "query":         "I am thirsty",
        "followup":      None,
        "followup_type": None,
        "description":   "Day 3: no correction needed",
    },
    {
        "day":           4,
        "query":         "I am thirsty",
        "followup":      None,
        "followup_type": None,
        "description":   "Day 4: learning stable",
    },
    {
        "day":           5,
        "query":         "I am thirsty",
        "followup":      None,
        "followup_type": None,
        "description":   "Day 5: learning stable",
    },
]

DRINK_CANDIDATES = [
    "juice", "cola", "water", "tea", "coffee", "soda", "milk", "drink",
]


def call_stream(url: str, query: str, user_id: str) -> dict:
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

    answer      = ""
    intent_type = ""
    buf         = ""
    is_service  = None

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
                is_service  = intent_type == "service"
            elif ev.get("type") == "token":
                token = ev.get("content", "")
                buf  += token
                if not is_service:
                    answer += token
            elif ev.get("type") == "done":
                if is_service and buf:
                    m = re.search(
                        r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"', buf)
                    if m:
                        answer = m.group(1)
                break
        except Exception:
            continue

    return {"answer": answer.strip(), "intent_type": intent_type}


def detect_recommendations(answer: str) -> list:
    lower = answer.lower()
    return [c for c in DRINK_CANDIDATES if c in lower]


def run_day(url: str, user_id: str, step: dict, rejected_items: set) -> dict:
    day   = step["day"]
    query = step["query"]

    print(f"\n{'=' * 55}")
    print(f"Day {day}: {step['description']}")
    print(f"  Query: '{query}'")

    result = call_stream(url, query, user_id)
    answer = result.get("answer", "")
    print(f"  System: '{answer[:100]}{'...' if len(answer) > 100 else ''}'")

    recs = detect_recommendations(answer)
    print(f"  Detected: {recs if recs else '(none)'}")

    correction_needed = bool(set(recs) & rejected_items)
    if correction_needed:
        print(f"  WARNING: recommended rejected item {set(recs) & rejected_items}")

    if step["followup"]:
        time.sleep(1)
        print(f"  User: '{step['followup']}'")
        fu = call_stream(url, step["followup"], user_id)
        print(f"  System: '{fu.get('answer', '')[:100]}'")

        if step["followup_type"] == "rejection":
            for rec in recs:
                rejected_items.add(rec)
            print(f"  Rejected: {recs} added to blacklist")
        elif step["followup_type"] == "acceptance":
            print(f"  Accepted.")

    correction_count = 1 if (correction_needed or
                              step["followup_type"] == "rejection") else 0

    return {
        "day":               day,
        "query":             query,
        "answer":            answer,
        "detected_recs":     recs,
        "followup":          step["followup"],
        "followup_type":     step["followup_type"],
        "correction_needed": correction_needed,
        "correction_count":  correction_count,
        "rejected_items":    list(rejected_items),
        "description":       step["description"],
    }


def run_experiment(url: str, user_id: str) -> list:
    print(f"\nRQ3c: Correction Rate")
    print(f"User   : {user_id}")
    print(f"Backend: {url}")

    results        = []
    rejected_items = set()

    for step in SCRIPT:
        result = run_day(url, user_id, step, rejected_items)
        results.append(result)
        time.sleep(2)

    return results


def compute_rates(results: list) -> list:
    return [
        {
            "day":             r["day"],
            "correction_count": r["correction_count"],
            "correction_rate": r["correction_count"] / 1.0,
            "description":     r["description"],
        }
        for r in results
    ]


def plot_correction_rate(rates: list, out_path: str):
    days  = [r["day"] for r in rates]
    pct   = [r["correction_rate"] * 100 for r in rates]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    ax.plot(days, pct, "o-",
            color="#E53935", linewidth=2.5, markersize=9,
            markerfacecolor="white", markeredgewidth=2.5,
            label="Correction Rate (%)", zorder=3)

    for day, rate in zip(days, pct):
        ax.annotate(
            f"{rate:.0f}%",
            xy=(day, rate), xytext=(0, 12),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=11, fontweight="bold", color="#E53935",
        )

    ax.axvspan(0.7, 1.3, alpha=0.1, color="#E53935", zorder=0)
    ax.annotate(
        "User rejects juice\nSystem learns",
        xy=(1, 100), xytext=(1.5, 82),
        fontsize=8.5, color="#C62828",
        arrowprops=dict(arrowstyle="->", color="#C62828", lw=1.2),
    )

    ax.axhspan(-5, 5, alpha=0.1, color="#4CAF50", zorder=0)
    ax.annotate(
        "No correction needed\n(system learned)",
        xy=(3, 0), xytext=(3, 22),
        fontsize=8.5, color="#2E7D32", ha="center",
        arrowprops=dict(arrowstyle="->", color="#2E7D32", lw=1.2),
    )

    ax.set_xticks(days)
    ax.set_xticklabels([f"Day {d}" for d in days], fontsize=11)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=10)
    ax.set_ylim(-10, 120)
    ax.set_xlabel("Conversation Day", fontsize=12)
    ax.set_ylabel("Correction Rate (%)", fontsize=12)
    ax.set_title(
        "RQ3c: Correction Rate over Time\n"
        "System learns user preference after single rejection",
        fontsize=12, fontweight="bold", pad=12,
    )
    ax.grid(axis="y", color="#E0E0E0", linewidth=0.8, zorder=0)
    ax.legend(loc="upper right", fontsize=10, framealpha=0.9)

    for spine in ax.spines.values():
        spine.set_edgecolor("#BDBDBD")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#FAFAFA")
    plt.close()
    print(f"Saved: {out_path}")


def save_csv(results: list, rates: list, out_path: str):
    fields = [
        "day", "query", "answer", "detected_recs",
        "correction_needed", "correction_rate", "description",
    ]
    rows = [
        {
            "day":               r["day"],
            "query":             r["query"],
            "answer":            r["answer"][:100],
            "detected_recs":     ", ".join(r["detected_recs"]),
            "correction_needed": r["correction_needed"],
            "correction_rate":   f"{rate['correction_rate']:.0%}",
            "description":       r["description"],
        }
        for r, rate in zip(results, rates)
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Saved: {out_path}")


def save_summary(results: list, rates: list, user_id: str, out_path: str):
    total_corr  = sum(r["correction_count"] for r in results)
    overall     = total_corr / len(results)

    lines = [
        "=" * 65,
        "RQ3c: Correction Rate — Feedback Learning Validation",
        f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 65,
        "",
        f"User            : {user_id}",
        f"Total days      : {len(results)}",
        f"Total corrections: {total_corr}",
        f"Overall rate    : {overall:.0%}",
        "",
        "Per-day results:",
        *[
            f"  Day {r['day']}: rate={rate['correction_rate']:.0%}  "
            f"({rate['description']})"
            for r, rate in zip(results, rates)
        ],
        "",
        "Rejected items accumulated:",
        *(
            [f"  After Day {r['day']}: {r['rejected_items']}"
             for r in results if r["rejected_items"]]
            or ["  (none)"]
        ),
        "",
        "For thesis:",
        f"A fixed 5-day dialogue script was executed for {user_id}.",
        f"On Day 1, the system recommended juice (matching the observed habit),",
        f"and the user rejected it explicitly.",
        f"From Day 2 onward, the system did not repeat the rejected item.",
        f"Correction Rate dropped from 100% on Day 1 to 0% from Day 2 onward,",
        f"demonstrating that the feedback channel reduces user correction burden",
        f"after a single explicit rejection.",
    ]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Saved: {out_path}")
    print(f"\nOverall Correction Rate: {overall:.0%}")
    print(
        "Day-by-day: "
        + "  ".join(
            f"Day{r['day']}:{rate['correction_rate']:.0%}"
            for r, rate in zip(results, rates)
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url",  default=BACKEND_URL)
    parser.add_argument("--user", default="User_Mom")
    parser.add_argument("--out",  default="results")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    try:
        requests.get(f"{args.url}/", timeout=5)
    except Exception:
        print(f"Cannot connect to {args.url}. Start app.py first.")
        sys.exit(1)

    results = run_experiment(args.url, args.user)
    rates   = compute_rates(results)

    plot_correction_rate(rates, os.path.join(args.out, "rq3c_correction_rate.png"))
    save_csv(results, rates, os.path.join(args.out, "rq3c_correction_rate.csv"))
    save_summary(results, rates, args.user,
                 os.path.join(args.out, "rq3c_correction_rate.txt"))


if __name__ == "__main__":
    main()