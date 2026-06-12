"""
check_system.py
───────────────
Monitor experiment progress in real time.

Usage:
  DB_NAME=robot_exp_baseline python3 tools/check_system.py          # full check
  DB_NAME=robot_exp_baseline python3 tools/check_system.py --watch  # live monitor
  DB_NAME=robot_exp_baseline python3 tools/check_system.py --quick  # summary only

Or via run.sh:
  bash run.sh results  (runs eval_runner.py + analysis scripts)
"""

import argparse
import datetime
import os
import time
from collections import Counter
from pymongo import MongoClient

MONGO_URI = "mongodb://127.0.0.1:27017/"
DB_NAME   = os.environ.get("DB_NAME", "robot_rag_db")


def connect():
    return MongoClient(MONGO_URI)[DB_NAME]


def section(title):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")


def ok(msg):   print(f"  OK    {msg}")
def warn(msg): print(f"  WARN  {msg}")
def err(msg):  print(f"  ERR   {msg}")


def _dominant_layer(reason: str) -> str:
    r = (reason or "").lower()
    if "strong:skeleton_lying" in r: return "skeleton"
    if any(k in r for k in ("strong:held:", "strong:head(")): return "strong"
    if any(k in r for k in ("skeleton_lying", "head(", "skeleton")): return "skeleton"
    if "pmi_llm" in r: return "llm"
    if "held:" in r: return "held"
    if "affordance:" in r: return "affordance"
    if "nearby:" in r: return "nearby"
    if "llm:" in r: return "llm"
    if any(k in r for k in ("prox:", "ray:", "zone:")): return "geometry"
    if "vlm(" in r: return "vlm"
    if "inertia(" in r: return "temporal"
    return "other"


def check_watch(db, n=15, interval=3, acc_threshold=0.50, window=20):
    print(f"\n[Watch] DB={DB_NAME} | refresh={interval}s | window={window}")
    print("Ctrl+C to stop.\n")
    seen_ids = set()

    while True:
        try:
            docs     = list(db.eval_logs.find({}).sort("timestamp", -1).limit(n))
            all_docs = list(db.eval_logs.find({}).sort("timestamp", -1).limit(window))
            total    = db.eval_logs.count_documents({})

            os.system("clear")
            now_str = datetime.datetime.now().strftime("%H:%M:%S")

            correct_window = sum(
                1 for d in all_docs
                if (d.get("spatial_action") or d.get("vlm_output", "")) ==
                   d.get("ground_truth", "")
            )
            acc_window = correct_window / len(all_docs) if all_docs else 0.0

            correct_total = sum(
                1 for d in db.eval_logs.find({})
                if (d.get("spatial_action") or d.get("vlm_output", "")) ==
                   d.get("ground_truth", "")
            )
            acc_total  = correct_total / total if total > 0 else 0.0
            pause_flag = acc_window < acc_threshold and len(all_docs) >= window

            obs_total = db.observation_logs.count_documents({})
            trans_cnt = db.transition_counts.count_documents({})

            print(f"[{now_str}] DB={DB_NAME} | "
                  f"episodes={total} | "
                  f"overall={acc_total:.0%} | "
                  f"last{window}={acc_window:.0%} | "
                  f"obs={obs_total} | trans={trans_cnt} | "
                  f"{'>>> CONSIDER PAUSING <<<' if pause_flag else 'running'}")
            print("-" * 95)
            print(f"  {'GT':14} {'VLM':14} {'Spatial':14} {'':3} "
                  f"{'Layer':10} {'Reason':28}")
            print("-" * 95)

            for d in reversed(docs):
                gt      = d.get("ground_truth", "?")
                vlm     = d.get("vlm_output", "?")
                spatial = d.get("spatial_action", "?") or vlm
                reason  = (d.get("upgrade_reason", "") or "")[:28]
                layer   = _dominant_layer(d.get("upgrade_reason", ""))
                is_new  = str(d.get("_id", "")) not in seen_ids
                seen_ids.add(str(d.get("_id", "")))
                ok_tag  = "OK" if spatial == gt else "XX"
                new_tag = "*" if is_new else " "
                print(f"{new_tag} {gt:14} {vlm:14} {spatial:14} {ok_tag:3} "
                      f"{layer:10} {reason}")

            print("-" * 95)

            wrong = [
                d for d in docs
                if (d.get("spatial_action") or d.get("vlm_output", "")) !=
                   d.get("ground_truth", "")
            ]

            if wrong:
                layer_wrong = Counter(
                    _dominant_layer(d.get("upgrade_reason", ""))
                    for d in wrong)
                print(f"\n  Errors (last {n}): {len(wrong)} | by layer: "
                      + " ".join(f"{l}:{c}" for l, c in layer_wrong.most_common()))

                gt_cnt = Counter(d.get("ground_truth", "?") for d in wrong)
                for gt_l, c in gt_cnt.most_common(4):
                    preds = Counter(
                        d.get("spatial_action") or d.get("vlm_output", "?")
                        for d in wrong if d.get("ground_truth") == gt_l
                    )
                    print(f"  GT={gt_l:12} {c}x -> {dict(preds.most_common(2))}")

            if pause_flag and wrong:
                top_layer = Counter(
                    _dominant_layer(d.get("upgrade_reason", ""))
                    for d in wrong).most_common(1)[0][0]
                hints = {
                    "geometry": "Check affinity_matrix proximity/zone scores",
                    "skeleton": "Check head_pitch thresholds in behavior_config.yaml",
                    "held":     "Check strong_held_items and held_by in dynamic_objects",
                    "llm":      "Check LLM prompt and scene_text quality",
                }
                print(f"\n  Hint: {hints.get(top_layer, f'Main error: {top_layer}')}")

            time.sleep(interval)

        except KeyboardInterrupt:
            print("\n[Watch] Stopped.")
            break


def check_scene(db):
    section("Scene & Zone Graph")
    n_scene = db.scene_snapshots.count_documents({})
    n_aff   = db.affinity_matrix.count_documents({})
    n_trans = db.transition_matrix.count_documents({})
    n_tcount = db.transition_counts.count_documents({})

    if n_scene == 0: err("scene_snapshots EMPTY")
    else: ok(f"scene_snapshots: {n_scene} objects")

    if n_aff   == 0: warn("affinity_matrix EMPTY")
    else: ok(f"affinity_matrix: {n_aff}")

    if n_trans == 0: warn("transition_matrix EMPTY (run charades_pipeline.py)")
    else: ok(f"transition_matrix: {n_trans} (Charades prior)")

    if n_tcount == 0: warn("transition_counts EMPTY (no experiments run yet)")
    else: ok(f"transition_counts: {n_tcount} (personal learning)")

    try:
        import requests
        r = requests.get("http://localhost:5000/ready", timeout=2)
        d = r.json()
        if d.get("ready"): ok(f"Flask /ready zones={d.get('zone_count',0)}")
        else: warn(f"Flask not ready zones={d.get('zone_count',0)}")
    except Exception:
        warn("Flask offline")


def check_observations(db):
    section("Observation Logs & Habits")
    total = db.observation_logs.count_documents({})
    if total == 0:
        warn("observation_logs EMPTY")
        return
    ok(f"Total: {total}")

    for uid in ["User_Mom", "User_Dad"]:
        docs = list(db.observation_logs.find(
            {"user": uid}, {"action": 1, "weight": 1, "zone_name": 1, "time_slot": 1}
        ).sort("weight", -1).limit(5))
        if not docs: continue
        print(f"\n  {uid} top habits:")
        for d in docs:
            print(f"    w={d.get('weight',0):3} | "
                  f"{d.get('action','?'):14} @ "
                  f"{d.get('zone_name','?'):20} "
                  f"({d.get('time_slot','?')})")

    n_trans = db.transition_counts.count_documents({})
    if n_trans > 0:
        print(f"\n  Learned transitions ({n_trans} total):")
        for uid in ["User_Mom", "User_Dad"]:
            docs = list(db.transition_counts.find(
                {"user_id": uid}
            ).sort("count", -1).limit(4))
            if not docs: continue
            print(f"  {uid}:")
            for d in docs:
                print(f"    {d['from_action']:14} → "
                      f"{d['to_action']:14} "
                      f"count={d['count']} "
                      f"({d.get('time_slot','?')})")


def check_eval(db):
    section(f"Evaluation Results — {DB_NAME}")
    total = db.eval_logs.count_documents({})
    if total == 0:
        warn("eval_logs EMPTY")
        return

    s2_correct = sum(
        1 for d in db.eval_logs.find({})
        if (d.get("spatial_action") or d.get("vlm_output", "")) ==
           d.get("ground_truth", "")
    )
    s1_correct = sum(
        1 for d in db.eval_logs.find({})
        if d.get("vlm_output", "") == d.get("ground_truth", "")
    )

    ok(f"Episodes: {total}")
    print(f"\n  Accuracy:")
    print(f"    VLM only    : {s1_correct}/{total} = {s1_correct/total:.1%}")
    print(f"    Full system : {s2_correct}/{total} = {s2_correct/total:.1%}  "
          f"(+{(s2_correct-s1_correct)/total:.1%})")

    all_docs   = list(db.eval_logs.find({}, {"upgrade_reason": 1}))
    reasons    = Counter(_dominant_layer(d.get("upgrade_reason","")) for d in all_docs)
    print(f"\n  Layer distribution:")
    for r, n in reasons.most_common():
        bar = "█" * int(n/total*20)
        print(f"    {r:12} {n:4} ({n/total:.0%})  {bar}")

    wrong = [
        d for d in db.eval_logs.find({})
        if (d.get("spatial_action") or d.get("vlm_output","")) !=
           d.get("ground_truth","")
    ]
    if not wrong:
        ok("No errors")
        return

    print(f"\n  Per-class accuracy:")
    labels = [
        "Drinking","SittingDrink","Sitting","Eating","Cooking",
        "Opening","Laying","Watching","Reading","Cleaning","PhoneUse","Typing",
    ]
    all_gt = list(db.eval_logs.find({}, {"ground_truth":1,"spatial_action":1,"vlm_output":1}))
    for label in labels:
        docs_label = [d for d in all_gt if d.get("ground_truth") == label]
        if not docs_label: continue
        correct = sum(
            1 for d in docs_label
            if (d.get("spatial_action") or d.get("vlm_output","")) == label
        )
        acc  = correct / len(docs_label)
        flag = "OK" if acc >= 0.70 else ("WN" if acc >= 0.40 else "XX")
        print(f"    {flag} {label:14} {acc:.0%} ({correct}/{len(docs_label)})")


def check_manifold(db):
    section("ManifoldEngine")
    total = db.manifold_training_data.count_documents({})
    ok(f"Total samples: {total}")
    for uid in ["User_Mom", "User_Dad"]:
        n = db.manifold_training_data.count_documents({"user_id": uid})
        if n >= 20: ok(f"{uid}: {n} samples")
        elif n > 0: warn(f"{uid}: {n} samples (need {20-n} more to train)")
        else: err(f"{uid}: 0 samples")

    model_dir = "manifold_models"
    if os.path.exists(model_dir):
        pkls = [f for f in os.listdir(model_dir) if f.endswith(".pkl")]
        if pkls: ok(f"Models: {pkls}")
        else: warn("No models trained yet")
    else:
        warn(f"manifold_models/ not found")


def summary(db):
    section(f"Summary — {DB_NAME}")
    checks = [
        ("scene_snapshots",        db.scene_snapshots.count_documents({}),        1),
        ("transition_matrix",      db.transition_matrix.count_documents({}),      1),
        ("transition_counts",      db.transition_counts.count_documents({}),      1),
        ("observation_logs",       db.observation_logs.count_documents({}),       1),
        ("eval_logs",              db.eval_logs.count_documents({}),              1),
        ("manifold_training_data", db.manifold_training_data.count_documents({}), 20),
    ]
    for name, count, threshold in checks:
        if count >= threshold: ok(f"{name:26}: {count}")
        else: warn(f"{name:26}: {count} (need >= {threshold})")

    total = db.eval_logs.count_documents({})
    if total > 0:
        correct = sum(
            1 for d in db.eval_logs.find({})
            if (d.get("spatial_action") or d.get("vlm_output","")) ==
               d.get("ground_truth","")
        )
        print(f"\n  Overall accuracy: {correct}/{total} = {correct/total:.1%}")


def main():
    parser = argparse.ArgumentParser(
        description=f"Check Robot Brain system (DB={DB_NAME})")
    parser.add_argument("--quick",     action="store_true",
                        help="Summary only")
    parser.add_argument("--watch",     action="store_true",
                        help="Live monitor mode")
    parser.add_argument("--n",         type=int,   default=15,
                        help="Episodes to show in watch mode")
    parser.add_argument("--interval",  type=int,   default=3,
                        help="Refresh interval (seconds)")
    parser.add_argument("--threshold", type=float, default=0.50,
                        help="Accuracy threshold for pause warning")
    parser.add_argument("--window",    type=int,   default=20,
                        help="Window size for rolling accuracy")
    args = parser.parse_args()

    db = connect()

    print(f"\nSystem check | DB={DB_NAME} | {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")

    if args.watch:
        check_watch(db, n=args.n, interval=args.interval,
                    acc_threshold=args.threshold, window=args.window)
        return

    if args.quick:
        summary(db)
        return

    check_scene(db)
    check_observations(db)
    check_manifold(db)
    check_eval(db)
    summary(db)


if __name__ == "__main__":
    main()