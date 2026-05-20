"""
check_system.py
執行中的系統健康檢查。安全對齊 PerceptionEngine 2.0 混合感知架構。
不需要停止 Flask。

Usage:
  python3 check_system.py           # 完整檢查
  python3 check_system.py --quick   # 只看關鍵指標
"""

import argparse
import math
import datetime
import os
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
        err("scene_snapshots is EMPTY — Unity has not sent /scene yet")
        return
    ok(f"scene_snapshots: {n_scene} furniture objects")

    if n_aff == 0:
        warn("affinity_matrix is EMPTY — SBERT SSP matrix compilation pending")
    else:
        ok(f"affinity_matrix: {n_aff} entries")

    # Check Flask App Route ready status
    try:
        import requests
        r = requests.get("http://localhost:5000/ready", timeout=3)
        data = r.json()
        if data.get("ready"):
            ok(f"Flask /ready = TRUE | zones={data.get('zone_count',0)}")
        else:
            warn(f"Flask /ready = FALSE | zones={data.get('zone_count',0)}")
    except Exception as e:
        warn(f"/ready unreachable (Flask offline): {e}")

    # Show zone distribution
    print("\n  Zone Graph Asset Deployment:")
    for doc in db.scene_snapshots.find({}, {"label":1,"room":1,"pos":1}).sort("room",1):
        pos = doc.get("pos", [0,0])
        print(f"    {doc.get('room','?'):15} | {doc.get('label','?'):20} pos=({pos[0]:.1f},{pos[1]:.1f})")

# ── 2. observation_logs 品質 ──────────────────────────────────────────
def check_observations(db):
    section("HabitEngine — observation_logs")

    total = db.observation_logs.count_documents({})
    if total == 0:
        warn("observation_logs is EMPTY")
        return
    ok(f"Total canonical log items: {total}")

    # 優化：利用 MongoDB Aggregate 聚合管道，取代全表 find() 造成的記憶體慢性自殺
    pipeline = [{"$group": {"_id": "$zone_name", "count": {"$sum": 1}}}]
    zone_groups = list(db.observation_logs.aggregate(pipeline))
    
    semantic = sum(g["count"] for g in zone_groups if g["_id"] and "_Zone" in str(g["_id"]))
    instance = sum(g["count"] for g in zone_groups if g["_id"] and "_Zone" not in str(g["_id"]))
    empty    = sum(g["count"] for g in zone_groups if not g["_id"])

    print(f"\n  Spatial Grounding Quality Control (zone_name品質):")
    if semantic > 0:
        ok(f"Semantic Grounded Zones (_Zone): {semantic} ({semantic/total:.0%})")
    if instance > 0:
        err(f"Raw Unity Instances detected (Dirty Context): {instance} ({instance/total:.0%})")
        # 顯示前幾名未被清洗乾淨的髒資料標籤
        dirty_items = sorted([g for g in zone_groups if g["_id"] and "_Zone" not in str(g["_id"])], key=lambda x: -x["count"])
        for g in dirty_items[:5]:
            print(f"    {g['count']:3}x | '{g['_id']}' ← Error: Leaked raw asset string")
    if empty > 0:
        warn(f"Empty zone_name fields: {empty}")

    print(f"\n  Top Hotspot Zones:")
    sorted_zones = sorted(zone_groups, key=lambda x: -x["count"])[:8]
    for g in sorted_zones:
        zname = g["_id"] or "(empty)"
        tag = "✅" if "_Zone" in str(zname) else "❌"
        print(f"    {tag} {g['count']:3}x | {zname}")

    # Per-user habit weight summaries
    print(f"\n  User Behavioral Weight Summary:")
    for uid in ["User_Mom", "User_Dad"]:
        docs = list(db.observation_logs.find({"user": uid}, {"action":1,"weight":1,"zone_name":1}))
        if not docs: continue
        top = sorted(docs, key=lambda x: -x.get("weight",0))[:5]
        print(f"    ▶️ {uid}:")
        for d in top:
            print(f"      w={d.get('weight',0):3} | {d.get('action','?'):12} @ {d.get('zone_name','?')}")

# ── 3. ManifoldEngine 狀態 ────────────────────────────────────────────
def check_manifold(db):
    section("ManifoldEngine — Continuous Latent Samples")

    total = db.manifold_training_data.count_documents({})
    ok(f"Total multi-dimensional samples: {total}")

    # 兼容性修正：同時檢索 user_id 與 user 欄位防禦錯位
    for uid in ["User_Mom", "User_Dad"]:
        n = db.manifold_training_data.count_documents({
            "$or": [{"user_id": uid}, {"user": uid}]
        })
        if n >= 20:
            ok(f"{uid}: {n} samples (>= 20 threshold, Model optimization enabled)")
        elif n > 0:
            warn(f"{uid}: {n} samples (Accumulating... need 20 to unlock backprop)")
        else:
            err(f"{uid}: 0 samples found in database")

    # Check local serializations
    model_dir = "manifold_models"
    if os.path.exists(model_dir):
        pkls = [f for f in os.listdir(model_dir) if f.endswith(".pkl")]
        if pkls:
            ok(f"Serialized user manifolds: {pkls}")
        else:
            warn("No user-specific .pkl models serialized yet")
    else:
        warn(f"Model directory '{model_dir}/' does not exist")

# ── 4. eval_logs 品質 ─────────────────────────────────────────────────
def check_eval(db):
    section("PerceptionEngine — Neural-Symbolic Evaluation")

    total = db.eval_logs.count_documents({})
    if total == 0:
        warn("eval_logs is EMPTY — no simulation cycles processed yet")
        return

    # 關鍵修正：依據 upgrade_reason 存在與否判斷 Stage 2 幾何防火牆是否有效介入
    upgraded = db.eval_logs.count_documents({
        "upgrade_reason": {"$ne": "", "$exists": True}
    })

    ok(f"Total simulation episodes recorded: {total}")
    rate = upgraded / total
    if rate > 0:
        ok(f"Stage 2 (Spatial Firewall) Intervention Rate: {upgraded}/{total} = {rate:.0%}")
    else:
        warn("Stage 2 Intervention Rate: 0% — Geometry engines idling. Check thresholds.")

    # Calculate exact Stage 1 vs Stage 2 Ground-Truth Accuracies
    s1_correct = db.eval_logs.count_documents({"$expr": {"$eq": ["$vlm_output", "$ground_truth"]}})
    
    # Stage 2 最終輸出邏輯判定
    s2_correct = 0
    for d in db.eval_logs.find({}):
        final_decision = d.get("spatial_action") if d.get("spatial_action") and d.get("spatial_action") != "Unknown" else d.get("vlm_output")
        if final_decision == d.get("ground_truth"):
            s2_correct += 1

    print(f"\n  Ablation Study Accuracy Assessment:")
    print(f"    Stage 1 Baseline (Raw VLM Hint):   {s1_correct}/{total} = {s1_correct/total:.0%}")
    print(f"    Stage 2 Complete (Neuro-Symbolic): {s2_correct}/{total} = {s2_correct/total:.0%}")

    # Extract upgrade reason subcomponents (L2A SayCan vs L2B Heading vs L3 Zone)
    reasons = Counter(
        d.get("upgrade_reason", "").split(":")[0] 
        for d in db.eval_logs.find({"upgrade_reason": {"$ne": "", "$exists": True}})
    )
    if reasons:
        print(f"\n  Intervention Layer Routing Stats:")
        for r, n in reasons.most_common():
            tag = "L2A (SayCan)" if "L2A" in r else ("L2B (Heading)" if "L2B" in r else ("L3 (Spatial Field)" if "L3" in r else r))
            print(f"    ⚙️  {n:3}x | {tag}")

    # Print real-time prediction streams
    print(f"\n  Latest 5 Streaming Decisions:")
    for d in db.eval_logs.find({}).sort("timestamp", -1).limit(5):
        gt  = d.get("ground_truth", "?")
        vlm = d.get("vlm_output", "?")
        spa = d.get("spatial_action", "")
        final_label = spa if spa and spa != "Unknown" else vlm
        zn  = d.get("zone_label", "Unknown_Zone")
        tag = "✅" if final_label == gt else "❌"
        print(f"    {tag} GT={gt:12} | VLM_Hint={vlm:12} → Output={final_label:12} zone={zn}")

# ── 5. Service Proposals ──────────────────────────────────────────────
def check_proposals(db):
    section("ServiceProposalEngine")

    total = db.service_proposals.count_documents({})
    ok(f"Total proactive proposals sent: {total}")
    if total == 0: return

    intents = Counter(d.get("intent", "?") for d in db.service_proposals.find({}))
    print("  Proactive Intent Distributions:")
    for intent, n in intents.most_common():
        print(f"    {n:3}x | {intent}")

# ── 6. 整體健康摘要 ───────────────────────────────────────────────────
def summary(db):
    section("Global System Health Summary")

    checks = [
        ("scene_snapshots",        db.scene_snapshots.count_documents({}),        1),
        ("affinity_matrix",        db.affinity_matrix.count_documents({}),        1),
        ("observation_logs",       db.observation_logs.count_documents({}),       1),
        ("eval_logs",              db.eval_logs.count_documents({}),              1),
        ("manifold_training_data", db.manifold_training_data.count_documents({"$or": [{"user_id": {"$exists": True}}, {"user": {"$exists": True}}]}), 20),
        ("habit_snapshots",        db.habit_snapshots.count_documents({}),        1),
        ("service_proposals",      db.service_proposals.count_documents({}),      0),
    ]

    for name, count, threshold in checks:
        if count >= threshold:
            ok(f"{name:25}: {count}")
        else:
            warn(f"{name:25}: {count} (expected >= {threshold})")

    # High-level data purity checks
    total_obs = db.observation_logs.count_documents({})
    if total_obs > 0:
        semantic = db.observation_logs.count_documents({"zone_name": {"$regex": "_Zone"}})
        pct = semantic / total_obs
        if pct >= 0.95:
            ok(f"Architectural Decoupling Purity: {pct:.1%} semantic grounding (Excellent)")
        elif pct >= 0.70:
            warn(f"Architectural Decoupling Purity: {pct:.1%} semantic grounding (Warning: Dirty leaks present)")
        else:
            err(f"Architectural Decoupling Purity: {pct:.1%} semantic grounding (Critical: System layout failure)")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Only show summary")
    args = parser.parse_args()

    db = connect()
    print(f"\nChecking System Telemetry Database: {DB_NAME}")
    print(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

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