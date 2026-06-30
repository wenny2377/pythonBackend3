import argparse
import datetime
import os
import time
from collections import Counter
from pymongo import MongoClient

MONGO_URI = "mongodb://127.0.0.1:27017/"


def _ask_db() -> str:
    print("\nWhich DB?")
    print("  1) robot_exp_baseline")
    print("  2) robot_exp_corruption")
    try:
        choice = input("Choice [1]: ").strip() or "1"
    except EOFError:
        choice = "1"
    return {
        "1": "robot_exp_baseline",
        "2": "robot_exp_corruption",
    }.get(choice, "robot_exp_baseline")


DB_NAME = os.environ.get("DB_NAME") or _ask_db()


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
    if "pmi_llm" in r:              return "llm"
    if "zone_affinity_fallback" in r: return "zone_fallback"
    if "coord_priority" in r:       return "coord"
    if "coord_only" in r:           return "coord"
    if "sbert" in r:                return "sbert"
    if "llm_error_fallback" in r:   return "llm_fallback"
    if "zone_not_ready" in r:       return "not_ready"
    return "other"


def check_watch(db, n=15, interval=3, acc_threshold=0.50, window=20):
    print(f"\n[Watch] DB={DB_NAME} | refresh={interval}s | window={window}")
    print("Ctrl+C to stop.\n")
    seen_ids = set()

    while True:
        try:
            docs     = list(db.experiment_logs.find({}).sort("timestamp", -1).limit(n))
            all_docs = list(db.experiment_logs.find({}).sort("timestamp", -1).limit(window))
            total    = db.experiment_logs.count_documents({})

            os.system("clear")
            now_str = datetime.datetime.now().strftime("%H:%M:%S")

            correct_window = sum(
                1 for d in all_docs
                if (d.get("spatial_action") or d.get("vlm_output", "")) ==
                   d.get("ground_truth", "")
            )
            acc_window = correct_window / len(all_docs) if all_docs else 0.0

            total_sample  = list(db.experiment_logs.find(
                {}, {"spatial_action":1,"vlm_output":1,"ground_truth":1}
            ).sort("timestamp",-1).limit(200))
            correct_total = sum(
                1 for d in total_sample
                if (d.get("spatial_action") or d.get("vlm_output", "")) ==
                   d.get("ground_truth", "")
            )
            if total_sample: acc_total = correct_total / len(total_sample)
            acc_total  = correct_total / total if total > 0 else 0.0
            pause_flag = acc_window < acc_threshold and len(all_docs) >= window

            obs_total = db.observation_logs.count_documents({})
            trans_cnt = db.transition_counts.count_documents({})

            print(f"[{now_str}] DB={DB_NAME} | "
                  f"episodes={total} | "
                  f"overall={acc_total:.0%} | "
                  f"last{window}={acc_window:.0%} | "
                  f"obs={obs_total} | trans={trans_cnt} | "
                  f"{'>>> LOW ACCURACY <<<' if pause_flag else 'running'}")
            print("-" * 100)
            print(f"  {'GT':16} {'Spatial':16} {'':3} "
                  f"{'Layer':16} {'Conf':6} {'Reason':30}")
            print("-" * 100)

            for d in reversed(docs):
                gt      = d.get("ground_truth", "?")
                spatial = d.get("spatial_action", "?") or d.get("vlm_output", "?")
                reason  = (d.get("upgrade_reason", "") or "")[:30]
                layer   = _dominant_layer(d.get("upgrade_reason", ""))
                conf    = d.get("vlm_confidence", 0.0)
                is_new  = str(d.get("_id", "")) not in seen_ids
                seen_ids.add(str(d.get("_id", "")))
                ok_tag  = "OK" if spatial == gt else "XX"
                new_tag = "*" if is_new else " "
                timed   = " [VLM-TO]" if d.get("vlm_timed_out") else ""
                print(f"{new_tag} {gt:16} {spatial:16} {ok_tag:3} "
                      f"{layer:16} {conf:5.2f} {reason}{timed}")

            print("-" * 100)

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
                    print(f"  GT={gt_l:14} {c}x -> {dict(preds.most_common(2))}")

            timed_out = sum(1 for d in docs if d.get("vlm_timed_out"))
            if timed_out > 0:
                print(f"\n  VLM timeouts in last {n}: {timed_out}")

            if pause_flag and wrong:
                top_layer = Counter(
                    _dominant_layer(d.get("upgrade_reason", ""))
                    for d in wrong).most_common(1)[0][0]
                hints = {
                    "llm":          "Check LLM prompt and scene_text quality",
                    "zone_fallback": "LLM failed, check Ollama connectivity",
                    "coord":        "Check scene_snapshots positions",
                    "sbert":        "Check furniture label normalization",
                    "not_ready":    "Scene engine not ready, check /ready endpoint",
                }
                print(f"\n  Hint: {hints.get(top_layer, f'Main error: {top_layer}')}")

            time.sleep(interval)

        except KeyboardInterrupt:
            print("\n[Watch] Stopped.")
            break


def check_scene(db):
    section("Scene & Zone Graph")
    n_scene  = db.scene_snapshots.count_documents({})
    n_aff    = db.affinity_matrix.count_documents({})
    n_tcount = db.transition_counts.count_documents({})

    if n_scene == 0: err("scene_snapshots EMPTY")
    else:            ok(f"scene_snapshots: {n_scene} objects")

    if n_aff == 0: warn("affinity_matrix EMPTY")
    else:          ok(f"affinity_matrix: {n_aff}")

    if n_tcount == 0: warn("transition_counts EMPTY (no baseline run yet)")
    else:             ok(f"transition_counts: {n_tcount} (personal learning)")

    try:
        import requests
        r = requests.get("http://localhost:5000/ready", timeout=2)
        d = r.json()
        if d.get("ready"): ok(f"Flask /ready zones={d.get('zone_count',0)}")
        else:              warn(f"Flask not ready zones={d.get('zone_count',0)}")
    except Exception:
        warn("Flask offline")


def check_observations(db):
    section("Behavioral Pattern Accumulation (BPA)")
    total = db.observation_logs.count_documents({})
    if total == 0:
        warn("observation_logs EMPTY")
        return
    ok(f"Total: {total}")

    for uid in ["User_Mom", "User_Dad"]:
        docs = list(db.observation_logs.find(
            {"user": uid}, {"action": 1, "weight": 1, "zone_name": 1, "time_slot": 1}
        ).sort("weight", -1).limit(5))
        if not docs:
            continue
        print(f"\n  {uid} top habits:")
        for d in docs:
            print(f"    w={d.get('weight',0):5.1f} | "
                  f"{d.get('action','?'):16} @ "
                  f"{d.get('zone_name','?'):20} "
                  f"({d.get('time_slot','?')})")

    n_trans = db.transition_counts.count_documents({})
    if n_trans > 0:
        print(f"\n  Learned transitions ({n_trans} total):")
        for uid in ["User_Mom", "User_Dad"]:
            docs = list(db.transition_counts.find(
                {"user_id": uid}
            ).sort("count", -1).limit(5))
            if not docs:
                continue
            print(f"  {uid}:")
            for d in docs:
                print(f"    {d['from_action']:16} → "
                      f"{d['to_action']:16} "
                      f"count={d['count']} "
                      f"({d.get('time_slot','?')})")


def check_eval(db):
    section(f"Evaluation Results — {DB_NAME}")
    all_docs_combined = []
    col_names = [
        "experiment_logs",
        "experiment_logs_corruption_light",
        "experiment_logs_corruption_medium",
        "experiment_logs_corruption_heavy",
    ]
    for col in col_names:
        docs = list(db[col].find(
            {"ground_truth": {"$exists": True, "$ne": ""}},
            {"ground_truth":1,"spatial_action":1,"vlm_output":1,
             "upgrade_reason":1,"vlm_timed_out":1,"experiment_mode":1}
        ))
        if docs:
            print(f"  {col}: {len(docs)} episodes")
        all_docs_combined.extend(docs)
    total = len(all_docs_combined)
    if total == 0:
        warn("experiment_logs EMPTY")
        return

    s2_correct = sum(
        1 for d in all_docs_combined
        if (d.get("spatial_action") or d.get("vlm_output", "")) ==
           d.get("ground_truth", "")
    )
    timed_out = sum(1 for d in all_docs_combined if d.get("vlm_timed_out"))

    ok(f"Episodes total: {total}")
    if timed_out > 0:
        warn(f"VLM timeouts: {timed_out} ({timed_out/total:.1%})")

    print(f"\n  Accuracy:")
    print(f"    Full system : {s2_correct}/{total} = {s2_correct/total:.1%}")

    reasons  = Counter(_dominant_layer(d.get("upgrade_reason", "")) for d in all_docs_combined)
    print(f"\n  Layer distribution:")
    for r, n in reasons.most_common():
        bar = "█" * int(n / total * 20)
        print(f"    {r:18} {n:4} ({n/total:.0%})  {bar}")

    print(f"\n  Per-class accuracy:")
    labels = [
        "Drinking", "SeatedDrinking", "Sitting", "Eating", "Cooking",
        "Opening", "Laying", "Watching", "Reading", "Cleaning", "UsingPhone", "Typing",
    ]
    all_gt = all_docs_combined
    for label in labels:
        docs_label = [d for d in all_gt if d.get("ground_truth") == label]
        if not docs_label:
            continue
        correct = sum(
            1 for d in docs_label
            if (d.get("spatial_action") or d.get("vlm_output", "")) == label
        )
        acc  = correct / len(docs_label)
        flag = "OK" if acc >= 0.70 else ("WN" if acc >= 0.40 else "XX")
        print(f"    {flag} {label:16} {acc:.0%} ({correct}/{len(docs_label)})")


def check_proactive(db):
    section("Proactive Service Readiness")
    n_trans = db.transition_counts.count_documents({})
    if n_trans == 0:
        warn("transition_counts EMPTY — proactive cannot trigger")
        return
    ok(f"transition_counts: {n_trans}")

    MIN_COUNT = 3
    for uid in ["User_Mom", "User_Dad"]:
        docs = list(db.transition_counts.find(
            {"user_id": uid, "count": {"$gte": MIN_COUNT}},
            {"from_action": 1, "to_action": 1, "count": 1, "time_slot": 1}
        ).sort("count", -1))

        if not docs:
            warn(f"{uid}: no transitions with count >= {MIN_COUNT}")
            continue

        ok(f"{uid}: {len(docs)} actionable transitions")
        key_transitions = [
            d for d in docs
            if d["from_action"] in ("Watching", "Eating", "Cooking")
        ]
        for d in key_transitions[:3]:
            print(f"    {d['from_action']:16} → {d['to_action']:16} "
                  f"count={d['count']} ({d.get('time_slot','?')})")

    n_objs = db.dynamic_objects.count_documents({"category": {"$in": ["drink", "food"]}})
    if n_objs == 0:
        warn("dynamic_objects: no drink/food items (proactive has nothing to offer)")
    else:
        ok(f"dynamic_objects: {n_objs} drink/food items available")
        for obj in db.dynamic_objects.find(
                {"category": {"$in": ["drink", "food"]}},
                {"label": 1, "category": 1, "last_seen_on": 1}
        ).limit(5):
            print(f"    {obj['label']:12} [{obj['category']:5}] "
                  f"at {obj.get('last_seen_on','?')}")


def check_skill(db):
    section("SKILL.md Status")
    for uid in ["User_Mom", "User_Dad"]:
        doc = db.user_skills.find_one({"user_id": uid})
        if not doc:
            warn(f"{uid}: no SKILL.md")
            continue
        skill_md = doc.get("skill_md", "")
        n_bullets = skill_md.count("\n- ")
        ok(f"{uid}: v{doc.get('version',1)} | {n_bullets} bullets")
        pref_start = skill_md.find("## Preferences")
        pref_end   = skill_md.find("## How to Handle", pref_start)
        if pref_start > 0:
            prefs = [l.strip() for l in
                     skill_md[pref_start:pref_end].split("\n")
                     if l.strip().startswith("-")]
            for p in prefs[:3]:
                print(f"    {p}")


def summary(db):
    section(f"Summary — {DB_NAME}")
    checks = [
        ("scene_snapshots",  db.scene_snapshots.count_documents({}),  1),
        ("transition_counts", db.transition_counts.count_documents({}), 1),
        ("observation_logs", db.observation_logs.count_documents({}),  1),
        ("experiment_logs", db.experiment_logs.count_documents({}),         1),
        ("dynamic_objects",  db.dynamic_objects.count_documents({}),   1),
        ("user_skills",      db.user_skills.count_documents({}),       2),
    ]
    for name, count, threshold in checks:
        if count >= threshold: ok(f"{name:26}: {count}")
        else:                  warn(f"{name:26}: {count} (need >= {threshold})")

    total = db.experiment_logs.count_documents({})
    if total > 0:
        correct = sum(
            1 for d in db.experiment_logs.find({})
            if (d.get("spatial_action") or d.get("vlm_output", "")) ==
               d.get("ground_truth", "")
        )
        print(f"\n  Overall accuracy: {correct}/{total} = {correct/total:.1%}")

    n_trans  = db.transition_counts.count_documents({"count": {"$gte": 3}})
    n_skills = db.user_skills.count_documents({})
    print(f"  Actionable transitions (count>=3): {n_trans}")
    print(f"  SKILL.md profiles: {n_skills}")


def main():
    parser = argparse.ArgumentParser(
        description=f"Check Robot Brain system (DB={DB_NAME})")
    parser.add_argument("--quick",     action="store_true", help="Summary only")
    parser.add_argument("--watch",     action="store_true", help="Live monitor mode")
    parser.add_argument("--n",         type=int,   default=15)
    parser.add_argument("--interval",  type=int,   default=3)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--window",    type=int,   default=20)
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
    check_eval(db)
    check_proactive(db)
    check_skill(db)
    summary(db)


if __name__ == "__main__":
    main()