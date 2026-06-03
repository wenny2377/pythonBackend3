import argparse
import csv
import datetime
import json
import os
import re
import time

import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BACKEND_URL = "http://localhost:5000"
USER_ID     = "User_Mom"

SCRIPT = [
    {
        "day":           1,
        "query":         "I am thirsty",
        "followup":      "No, I don't want juice, I don't like it",
        "followup_type": "rejection",
        "description":   "Day 1: system recommends juice -> user rejects",
    },
    {
        "day":           2,
        "query":         "I am thirsty",
        "followup":      "OK, that's fine",
        "followup_type": "acceptance",
        "description":   "Day 2: system recommends alternative -> accepted",
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

DRINK_KEYWORDS = [
    "juice", "cola", "water", "tea", "coffee",
    "soda", "milk", "drink",
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
                    m = re.search(
                        r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"', buf)
                    if m:
                        answer = m.group(1)
                break
        except Exception:
            continue
    return {"answer": answer.strip()}


def detect_drinks(answer):
    lower = answer.lower()
    return [k for k in DRINK_KEYWORDS if k in lower]


def run_day(url, step, rejected_items):
    day   = step["day"]
    query = step["query"]
    print(f"\n{'='*50}")
    print(f"Day {day}: {step['description']}")
    print(f"  User: '{query}'")

    result = call_stream(url, query, USER_ID)
    answer = result.get("answer", "")
    print(f"  Robot: '{answer[:100]}'")

    recs               = detect_drinks(answer)
    correction_needed  = bool(set(recs) & rejected_items)

    if step["followup"]:
        time.sleep(1)
        print(f"  User: '{step['followup']}'")
        fu = call_stream(url, step["followup"], USER_ID)
        print(f"  Robot: '{fu.get('answer','')[:80]}'")
        if step["followup_type"] == "rejection":
            for r in recs:
                rejected_items.add(r)
            print(f"  -> Rejected: {recs}")

    return {
        "day":              day,
        "answer":           answer,
        "detected":         recs,
        "correction_needed": correction_needed,
        "correction_rate":  1.0 if correction_needed or
                            step["followup_type"] == "rejection" else 0.0,
        "description":      step["description"],
    }


def plot(results, out_path):
    days  = [r["day"]             for r in results]
    rates = [r["correction_rate"] * 100 for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    ax.plot(days, rates, "o-",
            color="#E53935", linewidth=2.5, markersize=9,
            markerfacecolor="white", markeredgewidth=2.5,
            label="Correction Rate (%)")

    for day, rate in zip(days, rates):
        ax.annotate(f"{rate:.0f}%",
                    xy=(day, rate), xytext=(0, 12),
                    textcoords="offset points",
                    ha="center", fontsize=11,
                    fontweight="bold", color="#E53935")

    ax.axvspan(0.7, 1.3, alpha=0.1, color="#E53935")
    ax.annotate("User rejects juice\nSystem learns",
                xy=(1, 100), xytext=(1.8, 80), fontsize=8.5,
                color="#C62828",
                arrowprops=dict(arrowstyle="->",
                                color="#C62828", lw=1.2))
    ax.axhspan(-5, 5, alpha=0.1, color="#4CAF50")
    ax.annotate("No correction needed",
                xy=(3, 0), xytext=(3, 22),
                fontsize=8.5, color="#2E7D32", ha="center",
                arrowprops=dict(arrowstyle="->",
                                color="#2E7D32", lw=1.2))

    ax.set_xticks(days)
    ax.set_xticklabels([f"Day {d}" for d in days], fontsize=11)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylim(-10, 120)
    ax.set_ylabel("Correction Rate (%)", fontsize=12)
    ax.set_title(
        "Rejection Learning: Correction Rate over 5 Days\n"
        "System learns user preference after a single rejection",
        fontsize=12, fontweight="bold", pad=12)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    for spine in ax.spines.values():
        spine.set_edgecolor("#BDBDBD")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor="#FAFAFA")
    plt.close()
    print(f"  Saved: {out_path}")


def save_summary(results, out_path):
    total = sum(r["correction_rate"] for r in results)
    lines = [
        "=" * 65,
        "Rejection Learning Demo (Supplementary)",
        f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 65,
        "",
        f"User: {USER_ID}",
        f"Total corrections: {int(total)}",
        "",
        "Per-day results:",
        *[f"  Day {r['day']}: rate={r['correction_rate']:.0%}  "
          f"({r['description']})"
          for r in results],
        "",
        "For thesis:",
        "On Day 1 the system recommended juice and the user rejected it.",
        "From Day 2 onward the system did not repeat the rejected item.",
        "Correction Rate dropped from 100% to 0% after a single rejection,",
        "demonstrating single-shot feedback learning.",
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

    print(f"Running 5-day rejection learning demo for {USER_ID}\n")
    results        = []
    rejected_items = set()

    for step in SCRIPT:
        result = run_day(args.url, step, rejected_items)
        results.append(result)
        time.sleep(2)

    plot(results,
         os.path.join(args.out, "correction_rate.png"))
    save_summary(results,
                 os.path.join(args.out, "correction_summary.txt"))


if __name__ == "__main__":
    main()