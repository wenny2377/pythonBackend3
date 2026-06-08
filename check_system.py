import argparse
import math
import datetime
import os
import time
import sys
from collections import Counter, deque
from pymongo import MongoClient

MONGO_URI = "mongodb://127.0.0.1:27017/"
DB_NAME   = "robot_rag_db"

def connect():
    return MongoClient(MONGO_URI)[DB_NAME]

def section(title):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")

def ok(msg):   print(f"  OK  {msg}")
def warn(msg): print(f"  WARN  {msg}")
def err(msg):  print(f"  ERR  {msg}")


def _match_debug_image(t_capture: str, user_id: str, activity: str) -> list:
    debug_dir = "debug_images"
    if not os.path.exists(debug_dir):
        return []
    try:
        ts_part = t_capture.replace(":", "").replace("-", "").replace("T", "")[:13]
        ts_hhmmss = ts_part[8:14] if len(ts_part) >= 14 else ""
        matches = []
        for fname in sorted(os.listdir(debug_dir)):
            if not fname.endswith((".jpg", ".png")):
                continue
            if ts_hhmmss and ts_hhmmss[:6] in fname:
                matches.append(fname)
            elif user_id.lower().replace("_", "") in fname.lower():
                matches.append(fname)
        return matches[:3]
    except Exception:
        return []


def _dominant_layer(reason: str) -> str:
    r = (reason or "").lower()
    if "strong:skeleton_lying" in r:
        return "skeleton"
    if any(k in r for k in ("strong:held:", "strong:head(")):
        return "strong"
    if any(k in r for k in ("skeleton_lying", "head(", "skeleton")):
        return "skeleton"
    if "pmi_llm" in r:
        return "llm"
    if "held:" in r:
        return "held"
    if "affordance:" in r:
        return "affordance"
    if "nearby:" in r:
        return "nearby"
    if "llm:" in r:
        return "llm"
    if any(k in r for k in ("prox:", "ray:", "zone:")):
        return "geometry"
    if "vlm(" in r:
        return "vlm"
    if "inertia(" in r:
        return "temporal"
    return "other"


def check_watch(db, n=15, interval=3, acc_threshold=0.50, window=20):
    print(f"\n[Watch Mode] Refreshing every {interval}s | showing last {n} episodes")
    print(f"  Auto-pause suggestion if acc < {acc_threshold:.0%} over last {window} episodes")
    print("Press Ctrl+C to stop.\n")

    seen_ids      = set()
    pause_alerted = False

    while True:
        try:
            docs     = list(db.eval_logs.find({}).sort("timestamp", -1).limit(n))
            all_docs = list(db.eval_logs.find({}).sort("timestamp", -1).limit(window))

            os.system("clear")
            now_str = datetime.datetime.now().strftime("%H:%M:%S")
            total   = db.eval_logs.count_documents({})

            correct_window = sum(
                1 for d in all_docs
                if (d.get("spatial_action") or d.get("vlm_output", "")) == d.get("ground_truth", "")
            )
            acc_window_val = correct_window / len(all_docs) if all_docs else 0.0

            correct_total = sum(
                1 for d in db.eval_logs.find({})
                if (d.get("spatial_action") or d.get("vlm_output", "")) == d.get("ground_truth", "")
            )
            acc_total  = correct_total / total if total > 0 else 0.0
            pause_flag = acc_window_val < acc_threshold and len(all_docs) >= window

            print(f"[{now_str}] total={total} | "
                  f"overall={acc_total:.0%} | "
                  f"last{window}={acc_window_val:.0%} | "
                  f"{'CONSIDER PAUSING' if pause_flag else 'running'}")
            print("-" * 95)
            print(f"  {'GT':12} {'VLM':12} {'Spatial':12} {'OK':3} "
                  f"{'Layer':10} {'Reason':30} {'img'}")
            print("-" * 95)

            for d in reversed(docs):
                gt      = d.get("ground_truth", "?")
                vlm     = d.get("vlm_output", "?")
                spatial = d.get("spatial_action", "?") or vlm
                reason  = d.get("upgrade_reason", "") or ""
                layer   = _dominant_layer(reason)
                reason_s = reason[:30]
                t_cap   = d.get("t_capture", "")
                user    = d.get("user", "")

                is_new  = str(d.get("_id", "")) not in seen_ids
                seen_ids.add(str(d.get("_id", "")))

                ok_tag  = "OK" if spatial == gt else "XX"
                new_tag = "*" if is_new else " "

                imgs    = _match_debug_image(t_cap, user, spatial)
                img_str = imgs[0] if imgs else ""

                print(f"{new_tag} {gt:12} {vlm:12} {spatial:12} {ok_tag:3} "
                      f"{layer:10} {reason_s:30} {img_str}")

            print("-" * 95)

            wrong_docs = [
                d for d in docs
                if (d.get("spatial_action") or d.get("vlm_output", "")) != d.get("ground_truth", "")
            ]

            if wrong_docs:
                print(f"\n  Error analysis (last {n})")
                layer_wrong = Counter(_dominant_layer(d.get("upgrade_reason", "")) for d in wrong_docs)
                print(f"  Wrong by layer:")
                for layer, c in layer_wrong.most_common():
                    print(f"    {c}x  [{layer}]")

                gt_cnt = Counter(d.get("ground_truth", "?") for d in wrong_docs)
                print(f"  Most confused GT:")
                for gt_l, c in gt_cnt.most_common(5):
                    preds = Counter(
                        d.get("spatial_action") or d.get("vlm_output", "?")
                        for d in wrong_docs if d.get("ground_truth") == gt_l
                    )
                    print(f"    GT={gt_l:12} -> {dict(preds.most_common(3))}")

                print(f"  Wrong reason samples:")
                shown = set()
                for d in wrong_docs[:5]:
                    r = d.get("upgrade_reason", "")[:50]
                    if r not in shown:
                        shown.add(r)
                        gt  = d.get("ground_truth", "?")
                        sp  = d.get("spatial_action", "?")
                        print(f"    GT={gt:12} Spatial={sp:12} reason={r}")

            if pause_flag and wrong_docs:
                layer_wrong = Counter(_dominant_layer(d.get("upgrade_reason", "")) for d in wrong_docs)
                top_layer   = layer_wrong.most_common(1)[0][0]
                print(f"\n  Last {window} acc {acc_window_val:.0%} < {acc_threshold:.0%}")
                if top_layer == "geometry":
                    print(f"  -> Check affinity_matrix proximity/zone scores")
                elif top_layer == "skeleton":
                    print(f"  -> Check head_pitch thresholds in behavior_config.yaml")
                elif top_layer == "held":
                    print(f"  -> Check strong_held_items and held_by in dynamic_objects")
                elif top_layer == "llm":
                    print(f"  -> Check LLM prompt and scene_text quality")
                else:
                    print(f"  -> Main error layer: {top_layer}")

            obs_total = db.observation_logs.count_documents({})
            man_total = db.manifold_training_data.count_documents({})
            print(f"\n  obs_logs={obs_total}  manifold_samples={man_total}")

            time.sleep(interval)

        except KeyboardInterrupt:
            print("\n[Watch] Stopped.")
            break


def check_scene(db):
    section("SceneEngine Zone Graph")

    n_scene = db.scene_snapshots.count_documents({})
    n_aff   = db.affinity_matrix.count_documents({})

    if n_scene == 0:
        err("scene_snapshots is EMPTY")
        return
    ok(f"scene_snapshots: {n_scene} furniture objects")

    if n_aff == 0:
        warn("affinity_matrix is EMPTY")
    else:
        ok(f"affinity_matrix: {n_aff} entries")

    try:
        import requests
        r    = requests.get("http://localhost:5000/ready", timeout=3)
        data = r.json()
        if data.get("ready"):
            ok(f"Flask /ready TRUE | zones={data.get('zone_count', 0)}")
        else:
            warn(f"Flask /ready FALSE | zones={data.get('zone_count', 0)}")
    except Exception as e:
        warn(f"/ready unreachable: {e}")

    print("\n  Zone Graph:")
    for doc in db.scene_snapshots.find(
            {}, {"label": 1, "room": 1, "pos": 1}).sort("room", 1):
        pos = doc.get("pos", [0, 0])
        print(f"    {doc.get('room','?'):15} | {doc.get('label','?'):20} "
              f"pos=({pos[0]:.1f},{pos[1]:.1f})")


def check_observations(db):
    section("HabitEngine observation_logs")

    total = db.observation_logs.count_documents({})
    if total == 0:
        warn("observation_logs is EMPTY")
        return
    ok(f"Total: {total}")

    pipeline    = [{"$group": {"_id": "$zone_name", "count": {"$sum": 1}}}]
    zone_groups = list(db.observation_logs.aggregate(pipeline))

    semantic = sum(g["count"] for g in zone_groups if g["_id"] and "_Zone" in str(g["_id"]))
    instance = sum(g["count"] for g in zone_groups if g["_id"] and "_Zone" not in str(g["_id"]))
    empty    = sum(g["count"] for g in zone_groups if not g["_id"])

    print(f"\n  Zone quality:")
    if semantic > 0: ok(f"Semantic (_Zone): {semantic} ({semantic/total:.0%})")
    if instance > 0:
        err(f"Raw instances (dirty): {instance} ({instance/total:.0%})")
        dirty = sorted(
            [g for g in zone_groups if g["_id"] and "_Zone" not in str(g["_id"])],
            key=lambda x: -x["count"])
        for g in dirty[:5]:
            print(f"    {g['count']:3}x | '{g['_id']}'")
    if empty > 0: warn(f"Empty zone_name: {empty}")

    print(f"\n  Top zones:")
    for g in sorted(zone_groups, key=lambda x: -x["count"])[:8]:
        zname = g["_id"] or "(empty)"
        tag   = "OK" if "_Zone" in str(zname) else "XX"
        print(f"    {tag} {g['count']:3}x | {zname}")

    print(f"\n  Per-user top habits:")
    for uid in ["User_Mom", "User_Dad"]:
        docs = list(db.observation_logs.find(
            {"user": uid}, {"action": 1, "weight": 1, "zone_name": 1}))
        if not docs:
            continue
        top = sorted(docs, key=lambda x: -x.get("weight", 0))[:5]
        print(f"    {uid}:")
        for d in top:
            print(f"      w={d.get('weight',0):3} | "
                  f"{d.get('action','?'):12} @ {d.get('zone_name','?')}")


def check_manifold(db):
    section("ManifoldEngine")

    total = db.manifold_training_data.count_documents({})
    ok(f"Total samples: {total}")

    for uid in ["User_Mom", "User_Dad"]:
        n = db.manifold_training_data.count_documents({
            "$or": [{"user_id": uid}, {"user": uid}]
        })
        if n >= 20:
            ok(f"{uid}: {n} samples (>= 20, can train)")
        elif n > 0:
            warn(f"{uid}: {n} samples (need {20-n} more)")
        else:
            err(f"{uid}: 0 samples")

    model_dir = "manifold_models"
    if os.path.exists(model_dir):
        pkls = [f for f in os.listdir(model_dir) if f.endswith(".pkl")]
        if pkls:
            ok(f"Models: {pkls}")
        else:
            warn("No .pkl models yet")
    else:
        warn(f"'{model_dir}/' not found")


def check_eval(db):
    section("PerceptionEngine eval_logs")

    total = db.eval_logs.count_documents({})
    if total == 0:
        warn("eval_logs is EMPTY")
        return

    upgraded = db.eval_logs.count_documents({
        "upgrade_reason": {"$ne": "", "$exists": True}
    })
    ok(f"Total episodes: {total}")
    ok(f"Spatial override rate: {upgraded}/{total} = {upgraded/total:.0%}")

    s1_correct = db.eval_logs.count_documents(
        {"$expr": {"$eq": ["$vlm_output", "$ground_truth"]}})
    s2_correct = sum(
        1 for d in db.eval_logs.find({})
        if (d.get("spatial_action") or d.get("vlm_output", "")) == d.get("ground_truth", "")
    )

    print(f"\n  Accuracy:")
    print(f"    VLM only   (L1-VLM) : {s1_correct}/{total} = {s1_correct/total:.0%}")
    print(f"    Full system (L1-L3)  : {s2_correct}/{total} = {s2_correct/total:.0%}")
    print(f"    Improvement          : +{(s2_correct-s1_correct)/total:.0%}")

    all_docs = list(db.eval_logs.find({}, {"upgrade_reason": 1}))
    reasons  = Counter(_dominant_layer(d.get("upgrade_reason", "")) for d in all_docs)
    if reasons:
        print(f"\n  Layer distribution:")
        for r, n in reasons.most_common():
            print(f"    {n:3}x | {r}")

    print(f"\n  Wrong predictions analysis:")
    wrong = [
        d for d in db.eval_logs.find({})
        if (d.get("spatial_action") or d.get("vlm_output", "")) != d.get("ground_truth", "")
    ]
    if not wrong:
        ok("No wrong predictions")
    else:
        gt_cnt = Counter(d.get("ground_truth", "?") for d in wrong)
        for gt_l, c in gt_cnt.most_common(8):
            preds = Counter(
                d.get("spatial_action") or d.get("vlm_output", "?")
                for d in wrong if d.get("ground_truth") == gt_l
            )
            print(f"    GT={gt_l:12}: {c}x wrong -> {dict(preds.most_common(3))}")

        print(f"\n  Wrong by layer:")
        layer_wrong = Counter(_dominant_layer(d.get("upgrade_reason", "")) for d in wrong)
        for layer, c in layer_wrong.most_common():
            pct = c / len(wrong) * 100
            print(f"    {c:3}x ({pct:.0f}%) | [{layer}]")

    print(f"\n  Latest 10 decisions:")
    print(f"  {'GT':12} {'VLM':12} {'Spatial':12} {'OK':3} {'Layer':10} {'Reason':25}")
    print(f"  {'-'*85}")
    for d in db.eval_logs.find({}).sort("timestamp", -1).limit(10):
        gt      = d.get("ground_truth", "?")
        vlm     = d.get("vlm_output", "?")
        spatial = d.get("spatial_action", "?") or vlm
        reason  = (d.get("upgrade_reason", "") or "")[:25]
        layer   = _dominant_layer(d.get("upgrade_reason", ""))
        ok_tag  = "OK" if spatial == gt else "XX"
        print(f"  {ok_tag} {gt:12} {vlm:12} {spatial:12}     {layer:10} {reason:25}")

    print(f"\n  debug_images folder:")
    debug_dir = "debug_images"
    if os.path.exists(debug_dir):
        files = sorted(os.listdir(debug_dir))[-10:]
        for f in files:
            print(f"    {f}")
    else:
        warn("debug_images/ not found")


def check_proposals(db):
    section("ServiceProposalEngine")

    total = db.service_proposals.count_documents({})
    ok(f"Total proposals: {total}")
    if total == 0:
        return

    intents = Counter(d.get("intent", "?") for d in db.service_proposals.find({}))
    for intent, n in intents.most_common():
        print(f"    {n:3}x | {intent}")


def summary(db):
    section("System Health Summary")

    checks = [
        ("scene_snapshots",        db.scene_snapshots.count_documents({}),        1),
        ("affinity_matrix",        db.affinity_matrix.count_documents({}),        1),
        ("transition_matrix",      db.transition_matrix.count_documents({}),      1),
        ("observation_logs",       db.observation_logs.count_documents({}),       1),
        ("eval_logs",              db.eval_logs.count_documents({}),              1),
        ("manifold_training_data", db.manifold_training_data.count_documents({}), 20),
        ("habit_snapshots",        db.habit_snapshots.count_documents({}),        1),
        ("service_proposals",      db.service_proposals.count_documents({}),      0),
    ]

    for name, count, threshold in checks:
        if count >= threshold:
            ok(f"{name:25}: {count}")
        else:
            warn(f"{name:25}: {count} (need >= {threshold})")

    total_obs = db.observation_logs.count_documents({})
    if total_obs > 0:
        semantic = db.observation_logs.count_documents(
            {"zone_name": {"$regex": "_Zone"}})
        pct = semantic / total_obs
        if pct >= 0.95:
            ok(f"Zone purity: {pct:.1%} (excellent)")
        elif pct >= 0.70:
            warn(f"Zone purity: {pct:.1%} (dirty leaks present)")
        else:
            err(f"Zone purity: {pct:.1%} (critical)")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick",     action="store_true")
    parser.add_argument("--watch",     action="store_true")
    parser.add_argument("--n",         type=int,   default=15)
    parser.add_argument("--interval",  type=int,   default=3)
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--window",    type=int,   default=20)
    args = parser.parse_args()

    db = connect()

    if args.watch:
        check_watch(db, n=args.n, interval=args.interval,
                    acc_threshold=args.threshold, window=args.window)
        return

    print(f"\nSystem check: {DB_NAME}")
    print(f"Time: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}")

    if args.quick:
        summary(db)
        return

    check_scene(db)
    check_observations(db)
    check_manifold(db)
    check_eval(db)
    check_proposals(db)
    summary(db)


if __name__ == "__main__":
    main()