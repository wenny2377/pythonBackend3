"""
eval_runner.py
──────────────
Experiment evaluation tool.

Reads eval_logs from MongoDB and generates:
  - Confusion matrix
  - Per-class accuracy
  - Overall accuracy
  - Layer breakdown (skeleton / llm / zone_fallback)

Usage:
  python3 tools/eval_runner.py
  python3 tools/eval_runner.py --user User_Mom --last 100
"""

import argparse
from collections import defaultdict, Counter
from datetime import datetime
from pymongo import MongoClient

MONGO_URI = "mongodb://127.0.0.1:27017/"
DB_NAME   = "robot_rag_db"

LABELS = [
    "Drinking", "SittingDrink", "Sitting", "Eating", "Cooking", "Opening",
    "Laying", "Watching", "Reading", "Cleaning", "PhoneUse", "Typing",
]


def load_eval_logs(db, user_id: str = None, last_n: int = None) -> list:
    query = {
        "ground_truth": {"$exists": True, "$ne": ""},
        "spatial_action": {"$exists": True},
    }
    if user_id:
        query["user"] = user_id

    cursor = db.eval_logs.find(query).sort("timestamp", -1)
    if last_n:
        cursor = cursor.limit(last_n)

    return list(cursor)


def compute_metrics(docs: list) -> dict:
    correct  = 0
    total    = 0
    by_class = defaultdict(lambda: {"tp": 0, "total": 0})
    confused = defaultdict(Counter)
    by_layer = Counter()

    for doc in docs:
        gt     = doc.get("ground_truth", "")
        pred   = doc.get("spatial_action", "")
        reason = doc.get("upgrade_reason", "")

        if not gt or not pred:
            continue

        total += 1
        by_class[gt]["total"] += 1

        # Layer breakdown
        if "pmi_llm" in reason:
            layer = "llm"
        elif reason == "zone_affinity_fallback":
            layer = "zone_fallback"
        elif reason in ("skeleton_laying", "skeleton_opening",
                        "skeleton_watching", "skeleton_typing"):
            layer = "skeleton"
        else:
            layer = "other"
        by_layer[layer] += 1

        if gt == pred:
            correct += 1
            by_class[gt]["tp"] += 1
        else:
            confused[gt][pred] += 1

    if total == 0:
        return {}

    accuracy = correct / total

    per_class = {}
    for label in LABELS:
        info  = by_class.get(label, {"tp": 0, "total": 0})
        tp    = info["tp"]
        tot   = info["total"]
        acc   = tp / tot if tot > 0 else None
        per_class[label] = {
            "accuracy": round(acc, 3) if acc is not None else "N/A",
            "correct":  tp,
            "total":    tot,
            "confused": dict(confused.get(label, {}).most_common(3)),
        }

    layer_pct = {
        k: round(v / total, 3) for k, v in by_layer.items()
    }

    return {
        "total":     total,
        "correct":   correct,
        "accuracy":  round(accuracy, 3),
        "per_class": per_class,
        "by_layer":  layer_pct,
    }


def print_report(metrics: dict, user_id: str = None):
    if not metrics:
        print("No eval logs found.")
        return

    print("\n" + "=" * 70)
    title = f"Evaluation Report"
    if user_id:
        title += f" — {user_id}"
    print(f"  {title}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    print(f"\nOverall:  {metrics['accuracy']*100:.1f}%  "
          f"({metrics['correct']}/{metrics['total']} correct)")

    print("\nBy layer:")
    for layer, pct in sorted(metrics["by_layer"].items(),
                             key=lambda x: -x[1]):
        bar = "█" * int(pct * 20)
        print(f"  {layer:20s} {pct*100:5.1f}%  {bar}")

    print("\nPer-class accuracy:")
    print(f"  {'Action':<16} {'Acc':>6}  {'Correct':>8}  {'Top confusion'}")
    print("  " + "-" * 60)

    for label in LABELS:
        info    = metrics["per_class"].get(label, {})
        acc     = info.get("accuracy", "N/A")
        correct = info.get("correct", 0)
        total   = info.get("total", 0)
        confused = info.get("confused", {})

        if total == 0:
            continue

        acc_str = f"{acc*100:.0f}%" if isinstance(acc, float) else acc
        flag    = "✅" if isinstance(acc, float) and acc >= 0.7 else \
                  "⚠️ " if isinstance(acc, float) and acc >= 0.4 else "❌"

        conf_str = ""
        if confused:
            top = list(confused.items())[:2]
            conf_str = ", ".join(f"{k}({v})" for k, v in top)

        print(f"  {flag} {label:<14} {acc_str:>5}  "
              f"{correct:>3}/{total:<3}    {conf_str}")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Eval runner for Robot Brain")
    parser.add_argument("--user",  type=str, default=None,
                        help="Filter by user_id (e.g. User_Mom)")
    parser.add_argument("--last",  type=int, default=None,
                        help="Only use last N eval logs")
    args = parser.parse_args()

    client = MongoClient(MONGO_URI)
    db     = client[DB_NAME]

    print(f"[eval_runner] Loading eval_logs from {DB_NAME}...")

    if args.user:
        users = [args.user]
    else:
        users = [None]  # All users combined

        # Also print per-user if multiple users exist
        distinct_users = db.eval_logs.distinct("user")
        if len(distinct_users) > 1:
            users = [None] + distinct_users

    for uid in users:
        docs    = load_eval_logs(db, user_id=uid, last_n=args.last)
        metrics = compute_metrics(docs)
        print_report(metrics, user_id=uid or "All Users")


if __name__ == "__main__":
    main()