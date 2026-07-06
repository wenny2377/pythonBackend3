"""
Diagnostic script: dump full raw fields for the remaining unexplained
misclassification cells in the Semantic System confusion matrix:

  - Laying    -> Reading   (27%, 8 cases)
  - Drinking  -> Cooking   (21%, 5 cases)
  - Opening   -> Eating    (43%, 3 cases, small sample)

Same approach as diagnose_laying_eating_confusion.py: print every stored
field for each misclassified episode so we can see what held_event,
skeleton values, and LLM reasoning actually looked like, instead of
guessing.

Run this from the same environment as your other analysis scripts.
"""
import os, sys, json
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from pymongo import MongoClient
from exp_config import MONGO_URI, DB_BASELINE, COL_SEMANTIC, load_docs

# (ground_truth, predicted) pairs to investigate
TARGET_PAIRS = [
    ("Laying", "Reading"),
    ("Drinking", "Cooking"),
    ("Opening", "Eating"),
]


def dump_episodes(docs, gt_label, pred_label):
    matches = [
        d for d in docs
        if d.get("ground_truth") == gt_label and d.get("_pred") == pred_label
    ]
    print("\n" + "=" * 80)
    print(f"{gt_label} -> {pred_label}: {len(matches)} episode(s)")
    print("=" * 80)

    for i, d in enumerate(matches, 1):
        printable = {}
        for k, v in d.items():
            try:
                json.dumps(v)
                printable[k] = v
            except TypeError:
                printable[k] = str(v)
        print(f"\n--- {gt_label}->{pred_label} Episode {i} ---")
        print(json.dumps(printable, indent=2, ensure_ascii=False))

    return matches


def main():
    db = MongoClient(MONGO_URI)[DB_BASELINE]
    docs = load_docs(db, COL_SEMANTIC)

    all_results = {}
    for gt_label, pred_label in TARGET_PAIRS:
        all_results[(gt_label, pred_label)] = dump_episodes(docs, gt_label, pred_label)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for (gt_label, pred_label), matches in all_results.items():
        print(f"{gt_label:10} -> {pred_label:10}: {len(matches)} episode(s)")
        # Quick peek at upgrade_reason and held_event across all matches
        reasons = sorted(set(m.get("upgrade_reason", "") for m in matches))
        held    = sorted(set(m.get("held_event", "") for m in matches))
        print(f"  upgrade_reason values: {reasons}")
        print(f"  held_event values:     {held}")


if __name__ == "__main__":
    main()