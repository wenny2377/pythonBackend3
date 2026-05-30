"""
check_system.py
執行中的系統健康檢查 + 即時監控模式。

Usage:
  python3 check_system.py              # 完整檢查
  python3 check_system.py --quick      # 只看關鍵指標
  python3 check_system.py --watch      # 即時監控（每 3s 刷新）
  python3 check_system.py --watch --n 20  # 監控最新 20 筆
"""

import argparse
import math
import datetime
import os
import time
import sys
from collections import Counter
from pymongo import MongoClient

MONGO_URI = "mongodb://127.0.0.1:27017/"
DB_NAME   = "robot_rag_db"

def connect():
    return MongoClient(MONGO_URI)[DB_NAME]

def section(title):
    print(f"\n{'='*55}")
    print(f"  {title}")
    print(f"{'='*55}")

def ok(msg):   print(f"  ✅ {msg}")
def warn(msg): print(f"  ⚠️  {msg}")
def err(msg):  print(f"  ❌ {msg}")


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


def check_watch(db, n=15, interval=3):
    print(f"\n[Watch Mode] Refreshing every {interval}s | showing last {n} episodes")
    print("Press Ctrl+C to stop.\n")

    seen_ids = set()
    while True:
        try:
            docs = list(
                db.eval_logs.find({}).sort("timestamp", -1).limit(n)
            )

            os.system("clear")
            now_str = datetime.datetime.now().strftime("%H:%M:%S")
            total   = db.eval_logs.count_documents({})
            correct = sum(
                1 for d in docs
                if (d.get("spatial_action") or d.get("vlm_output", "")) == d.get("ground_truth", "")
            )
            acc = correct / len(docs) if docs else 0.0

            print(f"[{now_str}] eval_logs total={total} | "
                  f"last {len(docs)} acc={acc:.0%} | Ctrl+C to stop")
            print("-" * 80)
            print(f"  {'GT':12} {'VLM':12} {'Spatial':12} {'OK':3} "
                  f"{'Reason':22} {'Zone':20} {'img'}")
            print("-" * 80)

            for d in reversed(docs):
                gt      = d.get("ground_truth", "?")
                vlm     = d.get("vlm_output", "?")
                spatial = d.get("spatial_action", "?") or vlm
                reason  = d.get("upgrade_reason", "")[:22]
                zone    = (d.get("zone_label", "") or "")[:20]
                t_cap   = d.get("t_capture", "")
                user    = d.get("user", "")

                is_new  = str(d.get("_id", "")) not in seen_ids
                seen_ids.add(str(d.get("_id", "")))

                ok_tag  = "✅" if spatial == gt else "❌"
                new_tag = "★" if is_new else " "

                imgs = _match_debug_image(t_cap, user, spatial)
                img_str = imgs[0] if imgs else ""

                print(f"{new_tag} {gt:12} {vlm:12} {spatial:12} {ok_tag:3} "
                      f"{reason:22} {zone:20} {img_str}")

            print("-" * 80)

            wrong_docs = [
                d for d in docs
                if (d.get("spatial_action") or d.get("vlm_output", "")) != d.get("ground_truth", "")
            ]
            if wrong_docs:
                print(f"\n  Wrong predictions breakdown ({len(wrong_docs)}):")
                reason_cnt = Counter(
                    d.get("upgrade_reason", "none")[:30]
                    for d in wrong_docs
                )
                for r, c in reason_cnt.most_common(5):
                    print(f"    {c}x  reason={r}")

                gt_cnt = Counter(d.get("ground_truth", "?") for d in wrong_docs)
                print(f"  Most confused GT labels:")
                for gt_l, c in gt_cnt.most_common(5):
                    preds = Counter(
                        d.get("spatial_action") or d.get("vlm_output", "?")
                        for d in wrong_docs if d.get("ground_truth") == gt_l
                    )
                    print(f"    GT={gt_l:12} misclassified as: {dict(preds.most_common(3))}")

            obs_total = db.observation_logs.count_documents({})
            man_total = db.manifold_training_data.count_documents({})
            print(f"\n  obs_logs={obs_total}  manifold_samples={man_total}")

            time.sleep(interval)

        except KeyboardInterrupt:
            print("\n[Watch] Stopped.")
            break


def check_scene(db):
    section("SceneEngine — Zone Graph")

    n_scene = db.scene_snapshots.count_documents({})
    n_aff   = db.affinity_matrix.count_documents({})

    if n_scene == 0:
        err("scene_snapshots is EMPTY — Unity has not sent /scene yet")
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
            ok(f"Flask /ready = TRUE | zones={data.get('zone_count', 0)}")
        else:
            warn(f"Flask /ready = FALSE | zones={data.get('zone_count', 0)}")
    except Exception as e:
        warn(f"/ready unreachable: {e}")

    print("\n  Zone Graph:")
    for doc in db.scene_snapshots.find(
            {}, {"label": 1, "room": 1, "pos": 1}).sort("room", 1):
        pos = doc.get("pos", [0, 0])
        print(f"    {doc.get('room','?'):15} | {doc.get('label','?'):20} "
              f"pos=({pos[0]:.1f},{pos[1]:.1f})")


def check_observations(db):
    section("HabitEngine — observation_logs")

    total = db.observation_logs.count_documents({})
    if total == 0:
        warn("observation_logs is EMPTY")
        return
    ok(f"Total: {total}")

    pipeline = [{"$group": {"_id": "$zone_name", "count": {"$sum": 1}}}]
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
        tag   = "✅" if "_Zone" in str(zname) else "❌"
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
    section("PerceptionEngine — eval_logs")

    total = db.eval_logs.count_documents({})
    if total == 0:
        warn("eval_logs is EMPTY — run RecognitionExp first")
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
    print(f"    VLM only  (Stage 1): {s1_correct}/{total} = {s1_correct/total:.0%}")
    print(f"    Full system (Stage 2): {s2_correct}/{total} = {s2_correct/total:.0%}")
    print(f"    Improvement: +{(s2_correct-s1_correct)/total:.0%}")

    reasons = Counter(
        d.get("upgrade_reason", "").split(":")[0]
        for d in db.eval_logs.find({"upgrade_reason": {"$ne": "", "$exists": True}})
    )
    if reasons:
        print(f"\n  Override layer breakdown:")
        for r, n in reasons.most_common():
            print(f"    {n:3}x | {r}")

    print(f"\n  Wrong predictions analysis:")
    wrong = [
        d for d in db.eval_logs.find({})
        if (d.get("spatial_action") or d.get("vlm_output", "")) != d.get("ground_truth", "")
    ]
    if not wrong:
        ok("No wrong predictions!")
    else:
        gt_cnt = Counter(d.get("ground_truth", "?") for d in wrong)
        for gt_l, c in gt_cnt.most_common(8):
            preds = Counter(
                d.get("spatial_action") or d.get("vlm_output", "?")
                for d in wrong if d.get("ground_truth") == gt_l
            )
            print(f"    GT={gt_l:12}: {c}x wrong → {dict(preds.most_common(3))}")

    print(f"\n  Latest 10 decisions:")
    print(f"  {'GT':12} {'VLM':12} {'Spatial':12} {'OK':3} {'Reason':25} {'debug_img'}")
    print(f"  {'-'*80}")
    for d in db.eval_logs.find({}).sort("timestamp", -1).limit(10):
        gt      = d.get("ground_truth", "?")
        vlm     = d.get("vlm_output", "?")
        spatial = d.get("spatial_action", "?") or vlm
        reason  = (d.get("upgrade_reason", "") or "")[:25]
        t_cap   = d.get("t_capture", "")
        user    = d.get("user", "")
        ok_tag  = "✅" if spatial == gt else "❌"
        imgs    = _match_debug_image(t_cap, user, spatial)
        img_str = imgs[0] if imgs else ""
        print(f"  {ok_tag} {gt:12} {vlm:12} {spatial:12}     {reason:25} {img_str}")

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
    parser.add_argument("--quick",    action="store_true", help="Only summary")
    parser.add_argument("--watch",    action="store_true", help="Live monitor mode")
    parser.add_argument("--n",        type=int, default=15, help="Lines in watch mode")
    parser.add_argument("--interval", type=int, default=3,  help="Refresh seconds")
    args = parser.parse_args()

    db = connect()

    if args.watch:
        check_watch(db, n=args.n, interval=args.interval)
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