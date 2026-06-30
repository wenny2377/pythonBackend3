import os
from collections import defaultdict
from pymongo import MongoClient

MONGO_URI  = "mongodb://127.0.0.1:27017/"
DB_NAME    = os.environ.get("DB_NAME", "robot_exp_baseline")
COLLECTION = os.environ.get("COLLECTION", "experiment_logs_semantic")

GT_NORMALIZE_MAP = {
    "seateddrinking": "Drinking",
    "dadreading":     "Reading",
    "dadcleaning":    "Cleaning",
    "dadphone":       "UsingPhone",
}


def normalize_gt(label: str) -> str:
    if not label:
        return label
    key = label.lower().strip().replace(" ", "").replace("_", "")
    return GT_NORMALIZE_MAP.get(key, label)


def main():
    client = MongoClient(MONGO_URI)
    db     = client[DB_NAME]
    col    = db[COLLECTION]

    docs = list(col.find(
        {"ground_truth": {"$exists": True, "$ne": ""}},
        {"episode_id": 1, "user": 1, "ground_truth": 1, "spatial_action": 1,
         "vlm_output": 1, "upgrade_reason": 1, "held_event": 1,
         "vlm_key_object": 1, "zone_label": 1, "room_name": 1,
         "vlm_confidence": 1, "vlm_timed_out": 1, "time_slot": 1,
         "virtual_day": 1, "timestamp": 1},
    ).sort("timestamp", 1))

    print(f"DB={DB_NAME} | Collection={COLLECTION}")
    print(f"Total episodes with ground_truth: {len(docs)}")
    print("=" * 90)

    wrong = []
    for d in docs:
        gt   = normalize_gt(d.get("ground_truth", ""))
        pred = d.get("spatial_action") or d.get("vlm_output", "")
        pred = normalize_gt(pred)
        if gt and pred and gt != pred:
            d["_gt"]   = gt
            d["_pred"] = pred
            wrong.append(d)

    print(f"Misclassified episodes: {len(wrong)} / {len(docs)}")
    print("=" * 90)

    if not wrong:
        print("No misclassified episodes found.")
        return

    by_pair = defaultdict(list)
    for d in wrong:
        by_pair[(d["_gt"], d["_pred"])].append(d)

    pair_counts = sorted(by_pair.items(), key=lambda x: -len(x[1]))

    print("\nConfusion pairs (ground_truth -> predicted), sorted by frequency:")
    print("-" * 90)
    for (gt, pred), eps in pair_counts:
        print(f"  {gt:16} -> {pred:16}  {len(eps)} episode(s)")

    print("\n" + "=" * 90)
    print("Detail per episode:")
    print("=" * 90)

    for (gt, pred), eps in pair_counts:
        print(f"\n### {gt} -> {pred} ({len(eps)} episode(s)) ###")
        for i, d in enumerate(eps, 1):
            print(f"\n  [{i}] episode_id={d.get('episode_id','')} "
                  f"user={d.get('user','')} "
                  f"day={d.get('virtual_day','')} "
                  f"slot={d.get('time_slot','')} "
                  f"room={d.get('room_name','')} "
                  f"zone={d.get('zone_label','')}")
            print(f"      held_event:     {d.get('held_event','')}")
            print(f"      vlm_key_object: {d.get('vlm_key_object','')}")
            print(f"      vlm_confidence: {d.get('vlm_confidence','')}  "
                  f"vlm_timed_out: {d.get('vlm_timed_out','')}")
            print(f"      upgrade_reason: {d.get('upgrade_reason','')}")

    print("\n" + "=" * 90)
    print(f"Done. {len(wrong)} misclassified episode(s) across "
          f"{len(pair_counts)} confusion pair(s).")


if __name__ == "__main__":
    main()