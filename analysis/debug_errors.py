"""
analysis/debug_errors.py
從 eval_logs 抓出所有辨識錯誤的 reason 和 log
直接執行：python3 analysis/debug_errors.py
"""

import os
from collections import Counter, defaultdict
from pymongo import MongoClient

MONGO_URI = "mongodb://127.0.0.1:27017/"
DB_NAME   = "robot_rag_db"

NO_WEIGHT = {"PickingUp", "PuttingDown", "Walking", "Standing", "StandUp"}

NORMALIZE = {
    "drinking":"Drinking","sittingdrink":"SittingDrink","sitting":"Sitting",
    "eating":"Eating","cooking":"Cooking","opening":"Opening","laying":"Laying",
    "watching":"Watching","reading":"Reading","cleaning":"Cleaning",
    "phoneuse":"PhoneUse","typing":"Typing","standing":"Standing",
    "walking":"Walking","unknown":"Unknown",
}

def norm(s):
    if not s: return "Unknown"
    return NORMALIZE.get(s.lower().strip().replace(" ","").replace("_",""), s.strip())

def connect():
    return MongoClient(MONGO_URI)[DB_NAME]

def main():
    db   = connect()
    docs = list(db.eval_logs.find(
        {"ground_truth": {"$exists": True, "$ne": ""}},
        {"ground_truth": 1, "spatial_action": 1, "vlm_output": 1,
         "upgrade_reason": 1, "user": 1, "zone_label": 1,
         "body_position": 1, "held_object": 1, "vlm_confidence": 1,
         "timestamp": 1}
    ).sort("timestamp", -1))

    wrong = [d for d in docs
             if norm(d.get("ground_truth","")) != norm(d.get("spatial_action",""))
             and norm(d.get("ground_truth","")) not in NO_WEIGHT]

    total   = len([d for d in docs
                   if norm(d.get("ground_truth","")) not in NO_WEIGHT])
    correct = total - len(wrong)

    print(f"\n{'='*65}")
    print(f"  Error Analysis — eval_logs")
    print(f"{'='*65}")
    print(f"  Total:   {total}")
    print(f"  Correct: {correct} ({correct/total*100:.1f}%)")
    print(f"  Wrong:   {len(wrong)} ({len(wrong)/total*100:.1f}%)")

    # ── 按 GT 分類的錯誤統計 ───────────────────────────────────
    print(f"\n{'─'*65}")
    print("  Wrong predictions by Ground Truth:")
    print(f"{'─'*65}")

    by_gt = defaultdict(list)
    for d in wrong:
        by_gt[norm(d.get("ground_truth",""))].append(d)

    for gt, errs in sorted(by_gt.items(), key=lambda x: -len(x[1])):
        pred_counts = Counter(norm(d.get("spatial_action","")) for d in errs)
        preds_str   = ", ".join(f"{p}({n})" for p, n in pred_counts.most_common(3))
        print(f"  GT={gt:<15} {len(errs):>3} errors → {preds_str}")

    # ── 按 reason 分類的錯誤統計 ──────────────────────────────
    print(f"\n{'─'*65}")
    print("  Top error reasons (upgrade_reason prefix):")
    print(f"{'─'*65}")

    reason_counts = Counter()
    for d in wrong:
        reason = d.get("upgrade_reason", "")[:40].strip()
        if reason:
            reason_counts[reason] += 1

    for reason, cnt in reason_counts.most_common(15):
        print(f"  {cnt:>3}x  {reason}")

    # ── 詳細錯誤 log ──────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  Detailed error log (most recent 30):")
    print(f"{'─'*65}")
    print(f"  {'GT':<15} {'Pred':<15} {'VLM':<15} {'Conf':>5}  Reason")
    print(f"  {'-'*13} {'-'*13} {'-'*13} {'-'*5}  {'-'*30}")

    for d in wrong[:30]:
        gt     = norm(d.get("ground_truth",""))
        pred   = norm(d.get("spatial_action",""))
        vlm    = norm(d.get("vlm_output",""))
        conf   = d.get("vlm_confidence", 0)
        reason = (d.get("upgrade_reason","") or "")[:45]
        user   = d.get("user","?").replace("User_","")
        print(f"  {gt:<15} {pred:<15} {vlm:<15} {conf:>5.2f}  {reason}")

    # ── 按用戶分類 ────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  Error rate by user:")
    print(f"{'─'*65}")

    by_user = defaultdict(lambda: {"total": 0, "wrong": 0})
    for d in docs:
        if norm(d.get("ground_truth","")) in NO_WEIGHT:
            continue
        user = d.get("user","Unknown")
        by_user[user]["total"] += 1
        if norm(d.get("ground_truth","")) != norm(d.get("spatial_action","")):
            by_user[user]["wrong"] += 1

    for user, stats in sorted(by_user.items()):
        t = stats["total"]
        w = stats["wrong"]
        acc = (t - w) / t * 100 if t > 0 else 0
        print(f"  {user:<15} total={t:>3}  wrong={w:>3}  acc={acc:.1f}%")

    # ── Watching 專項分析 ─────────────────────────────────────
    watching_wrong = [d for d in wrong
                      if norm(d.get("ground_truth","")) == "Watching"]
    if watching_wrong:
        print(f"\n{'─'*65}")
        print(f"  Watching errors ({len(watching_wrong)} cases):")
        print(f"{'─'*65}")
        for d in watching_wrong[:10]:
            pred   = norm(d.get("spatial_action",""))
            reason = (d.get("upgrade_reason","") or "")[:60]
            print(f"  → {pred:<15}  {reason}")

    # ── SittingDrink 專項分析 ─────────────────────────────────
    sd_wrong = [d for d in wrong
                if norm(d.get("ground_truth","")) == "SittingDrink"]
    if sd_wrong:
        print(f"\n{'─'*65}")
        print(f"  SittingDrink errors ({len(sd_wrong)} cases):")
        print(f"{'─'*65}")
        for d in sd_wrong[:10]:
            pred   = norm(d.get("spatial_action",""))
            reason = (d.get("upgrade_reason","") or "")[:60]
            held   = d.get("held_object","?")
            print(f"  → {pred:<15}  held={held:<12}  {reason}")

    # ── Eating 專項分析 ───────────────────────────────────────
    eat_wrong = [d for d in wrong
                 if norm(d.get("ground_truth","")) == "Eating"]
    if eat_wrong:
        print(f"\n{'─'*65}")
        print(f"  Eating errors ({len(eat_wrong)} cases):")
        print(f"{'─'*65}")
        for d in eat_wrong[:10]:
            pred   = norm(d.get("spatial_action",""))
            reason = (d.get("upgrade_reason","") or "")[:60]
            held   = d.get("held_object","?")
            print(f"  → {pred:<15}  held={held:<12}  {reason}")

    print(f"\n{'='*65}")
    print("  Done.")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()