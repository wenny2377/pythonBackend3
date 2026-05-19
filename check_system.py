"""
check_system.py
執行中的系統健康檢查。不需要停止 Flask。

Usage:
  python3 check_system.py           # 完整檢查
  python3 check_system.py --quick   # 只看關鍵指標
"""

import argparse
import math
import datetime
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

def ok(msg):  print(f"  ✅ {msg}")
def warn(msg): print(f"  ⚠️  {msg}")
def err(msg):  print(f"  ❌ {msg}")

# ── 1. SceneEngine 狀態 ───────────────────────────────────────────────
def check_scene(db):
    section("SceneEngine — Zone Graph")

    n_scene = db.scene_snapshots.count_documents({})
    n_aff   = db.affinity_matrix.count_documents({})

    if n_scene == 0:
        err(f"scene_snapshots is EMPTY — Unity has not sent /scene yet")
        return
    ok(f"scene_snapshots: {n_scene} furniture objects")

    if n_aff == 0:
        warn("affinity_matrix is EMPTY — Gemma distillation not done yet")
    else:
        ok(f"affinity_matrix: {n_aff} entries")

    # Check /ready endpoint
    try:
        import requests
        r = requests.get("http://localhost:5000/ready", timeout=3)
        data = r.json()
        if data.get("ready"):
            ok(f"Flask /ready = TRUE | zones={data.get('zone_count',0)}")
        else:
            warn(f"Flask /ready = FALSE | zones={data.get('zone_count',0)}")
    except Exception as e:
        warn(f"/ready unreachable: {e}")

    # Show zone distribution
    print("\n  Zone Graph:")
    for doc in db.scene_snapshots.find(
        {}, {"label":1,"room":1,"pos":1}).sort("room",1):
        pos = doc.get("pos", [0,0])
        print(f"    {doc.get('room','?'):15} | {doc.get('label','?'):20} "
              f"pos=({pos[0]:.1f},{pos[1]:.1f})")

# ── 2. observation_logs 品質 ──────────────────────────────────────────
def check_observations(db):
    section("HabitEngine — observation_logs")

    total = db.observation_logs.count_documents({})
    if total == 0:
        warn("observation_logs is EMPTY")
        return
    ok(f"Total entries: {total}")

    # zone_name 品質
    zones = [d.get("zone_name","") for d in db.observation_logs.find({})]
    zone_counts = Counter(zones)
    semantic = sum(1 for z in zones if "_Zone" in z)
    instance = sum(1 for z in zones if "_Zone" not in z and z)
    empty    = sum(1 for z in zones if not z)

    print(f"\n  zone_name 品質:")
    if semantic > 0:
        ok(f"Semantic zones (_Zone): {semantic} ({semantic/total:.0%})")
    if instance > 0:
        warn(f"Instance names (dirty): {instance} ({instance/total:.0%})")
        # Show which instances
        for z, n in zone_counts.most_common(10):
            if z and "_Zone" not in z:
                print(f"    {n:3} | '{z}' ← dirty")
    if empty > 0:
        warn(f"Empty zone_name: {empty}")

    print(f"\n  Top zones:")
    for z, n in zone_counts.most_common(10):
        tag = "✅" if "_Zone" in z else "⚠️ "
        print(f"    {tag} {n:3} | {z or '(empty)'}")

    # Per-user per-action weight
    print(f"\n  Per-user weight summary:")
    for uid in ["User_Mom", "User_Dad"]:
        docs = list(db.observation_logs.find(
            {"user": uid}, {"action":1,"weight":1,"zone_name":1}))
        if not docs:
            continue
        top = sorted(docs, key=lambda x: -x.get("weight",0))[:5]
        print(f"\n  {uid}:")
        for d in top:
            print(f"    w={d.get('weight',0):3} | "
                  f"{d.get('action','?'):12} @ {d.get('zone_name','?')}")

# ── 3. ManifoldEngine 狀態 ────────────────────────────────────────────
def check_manifold(db):
    section("ManifoldEngine — Training Data")

    total = db.manifold_training_data.count_documents({})
    ok(f"Total samples: {total}")

    for uid in ["User_Mom", "User_Dad"]:
        n = db.manifold_training_data.count_documents({"user_id": uid})
        if n >= 20:
            ok(f"{uid}: {n} samples (>= 20, can train)")
        elif n > 0:
            warn(f"{uid}: {n} samples (need 20 to train)")
        else:
            err(f"{uid}: 0 samples")

    # Check model files
    import os
    model_dir = "manifold_models"
    if os.path.exists(model_dir):
        pkls = [f for f in os.listdir(model_dir) if f.endswith(".pkl")]
        if pkls:
            ok(f"Model files: {pkls}")
        else:
            warn("No .pkl model files — training not done yet")
    else:
        warn("manifold_models/ directory not found")

# ── 4. eval_logs 品質 ─────────────────────────────────────────────────
def check_eval(db):
    section("PerceptionEngine — eval_logs")

    total    = db.eval_logs.count_documents({})
    upgraded = db.eval_logs.count_documents({
        "$expr": {"$ne": [
            {"$ifNull": ["$spatial_action", "Unknown"]},
            {"$ifNull": ["$vlm_output", "Unknown"]}
        ]}
    })

    if total == 0:
        warn("eval_logs is EMPTY — no episodes processed yet")
        return

    ok(f"Total episodes: {total}")
    rate = upgraded / total
    if rate > 0:
        ok(f"Stage 2 upgrade rate: {upgraded}/{total} = {rate:.0%}")
    else:
        warn(f"Stage 2 upgrade rate: 0% — spatial reasoning not triggering")

    # Stage 1 vs Stage 2 accuracy
    s1_correct = sum(1 for d in db.eval_logs.find({})
                     if d.get("vlm_output") == d.get("ground_truth"))
    s2_correct = sum(1 for d in db.eval_logs.find({})
                     if (d.get("spatial_action") or d.get("vlm_output"))
                     == d.get("ground_truth"))

    print(f"\n  Accuracy:")
    print(f"    Stage 1 (VLM):     {s1_correct}/{total} = {s1_correct/total:.0%}")
    print(f"    Stage 2 (Spatial): {s2_correct}/{total} = {s2_correct/total:.0%}")

    # Upgrade reason distribution
    reasons = Counter(
        d.get("upgrade_reason","")
        for d in db.eval_logs.find({})
        if d.get("upgrade_reason","")
    )
    if reasons:
        print(f"\n  Upgrade reasons:")
        for r, n in reasons.most_common(10):
            l2a = "L2A" in r
            l2b = "L2B" in r
            l3  = "L3" in r
            tag = "L2A" if l2a else ("L2B" if l2b else ("L3" if l3 else "?"))
            print(f"    [{tag}] {n}x | {r[:60]}")

    # Recent episodes
    print(f"\n  Last 5 episodes:")
    for d in db.eval_logs.find({}).sort("timestamp",-1).limit(5):
        gt  = d.get("ground_truth","?")
        vlm = d.get("vlm_output","?")
        spa = d.get("spatial_action","?") or vlm
        zn  = d.get("zone_label","?")
        tag = "✅" if spa == gt else "❌"
        print(f"    {tag} GT={gt:12} VLM={vlm:12} → {spa:12} zone={zn}")

# ── 5. Service Proposals ──────────────────────────────────────────────
def check_proposals(db):
    section("ServiceProposalEngine")

    total = db.service_proposals.count_documents({})
    ok(f"Total proposals: {total}")
    if total == 0:
        return

    intents = Counter(
        d.get("intent","?")
        for d in db.service_proposals.find({})
    )
    print("  Intent distribution:")
    for intent, n in intents.most_common():
        print(f"    {n:3} | {intent}")

# ── 6. 整體健康摘要 ───────────────────────────────────────────────────
def summary(db):
    section("Summary")

    checks = [
        ("scene_snapshots",      db.scene_snapshots.count_documents({}),      1),
        ("affinity_matrix",      db.affinity_matrix.count_documents({}),      1),
        ("observation_logs",     db.observation_logs.count_documents({}),     1),
        ("eval_logs",            db.eval_logs.count_documents({}),            1),
        ("manifold_training_data", db.manifold_training_data.count_documents({}), 20),
        ("habit_snapshots",      db.habit_snapshots.count_documents({}),      1),
        ("service_proposals",    db.service_proposals.count_documents({}),    0),
    ]

    for name, count, threshold in checks:
        if count >= threshold:
            ok(f"{name}: {count}")
        else:
            warn(f"{name}: {count} (expected >= {threshold})")

    # Zone name quality
    total_obs = db.observation_logs.count_documents({})
    if total_obs > 0:
        semantic = db.observation_logs.count_documents(
            {"zone_name": {"$regex": "_Zone"}})
        pct = semantic / total_obs
        if pct >= 0.9:
            ok(f"Zone name quality: {pct:.0%} semantic")
        elif pct >= 0.5:
            warn(f"Zone name quality: {pct:.0%} semantic (some dirty data)")
        else:
            err(f"Zone name quality: {pct:.0%} semantic (mostly dirty!)")

    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Only show summary")
    args = parser.parse_args()

    db = connect()
    print(f"\nChecking DB: {DB_NAME}")
    print(f"Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

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