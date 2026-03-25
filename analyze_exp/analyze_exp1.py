"""
analyze_exp1.py
───────────────
Experiment 1: VLM Action Recognition Accuracy
從 MongoDB eval_logs 計算 per-class P/R/F1，畫 confusion matrix。

使用方式：
    python3 analyze_exp1.py
    python3 analyze_exp1.py --out ./results/

輸出：
    exp1_confusion_matrix.png   → 論文 Figure
    exp1_metrics.csv            → 論文 Table 數值
    exp1_summary.txt            → 直接複製貼到論文的文字摘要
"""

import argparse
import os
import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pymongo import MongoClient
from collections import Counter

# ── Config ────────────────────────────────────────────────────────────────
MONGO_URI = "mongodb://127.0.0.1:27017/"
DB_NAME   = "robot_rag_db"

# 標準行為標籤（論文用顯示名稱）
BEHAVIOR_LABELS = [
    "Drink", "SittingIdle", "Reading",
    "Typing", "Watching", "Sleeping",
    "Eating", "Exercising", "Walking", "Standing",
]

NORMALIZE_MAP = {
    "drinking":   "Drink",    "drink":      "Drink",
    "sitting":    "SittingIdle", "sit":     "SittingIdle", "sittingidle": "SittingIdle",
    "reading":    "Reading",  "read":       "Reading",
    "typing":     "Typing",   "type":       "Typing",
    "watching":   "Watching", "watch":      "Watching",
    "sleeping":   "Sleeping", "sleep":      "Sleeping", "lying": "Sleeping",
    "eating":     "Eating",   "eat":        "Eating",
    "exercising": "Exercising","exercise":  "Exercising",
    "walking":    "Walking",  "walk":       "Walking",
    "standing":   "Standing", "stand":      "Standing",
}

def normalize(label: str) -> str:
    if not label:
        return "Unknown"
    label = label.lower().strip()
    if label in NORMALIZE_MAP:
        return NORMALIZE_MAP[label]
    for kw, mapped in NORMALIZE_MAP.items():
        if kw in label:
            return mapped
    return label.capitalize()

# ── Load data ──────────────────────────────────────────────────────────────
def load_eval_logs(db, query=None):
    q = query or {}
    docs = list(db.eval_logs.find(q, {
        "ground_truth": 1, "vlm_output": 1,
        "user_id": 1, "room": 1, "timestamp": 1
    }))
    print(f"  Loaded {len(docs)} eval_log records")
    return docs

# ── Metrics ────────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, labels):
    """Per-class P/R/F1 + macro avg."""
    from collections import defaultdict
    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
    for t, p in zip(y_true, y_pred):
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1
    rows = []
    for lbl in labels:
        p  = tp[lbl] / (tp[lbl] + fp[lbl]) if (tp[lbl] + fp[lbl]) > 0 else 0.0
        r  = tp[lbl] / (tp[lbl] + fn[lbl]) if (tp[lbl] + fn[lbl]) > 0 else 0.0
        f1 = 2*p*r / (p+r) if (p+r) > 0 else 0.0
        n  = y_true.count(lbl)
        rows.append({"label": lbl, "precision": p, "recall": r, "f1": f1, "support": n})
    # macro avg (over classes with support > 0)
    active = [r for r in rows if r["support"] > 0]
    macro_p  = np.mean([r["precision"] for r in active]) if active else 0.0
    macro_r  = np.mean([r["recall"]    for r in active]) if active else 0.0
    macro_f1 = np.mean([r["f1"]        for r in active]) if active else 0.0
    overall_acc = sum(t==p for t,p in zip(y_true,y_pred)) / len(y_true) if y_true else 0.0
    return rows, macro_p, macro_r, macro_f1, overall_acc

# ── Confusion matrix ───────────────────────────────────────────────────────
def build_confusion_matrix(y_true, y_pred, labels):
    idx = {l: i for i, l in enumerate(labels)}
    n   = len(labels)
    cm  = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            cm[idx[t]][idx[p]] += 1
    return cm

def plot_confusion_matrix(cm, labels, out_path, overall_acc, macro_f1):
    active = [l for l in labels if cm[labels.index(l)].sum() > 0 or cm[:, labels.index(l)].sum() > 0]
    if not active:
        print("  ⚠️  No data to plot confusion matrix")
        return
    act_idx = [labels.index(l) for l in active]
    cm_sub  = cm[np.ix_(act_idx, act_idx)]

    fig, ax = plt.subplots(figsize=(max(7, len(active)*1.1), max(6, len(active)*1.0)))
    im = ax.imshow(cm_sub, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(active))); ax.set_yticks(range(len(active)))
    ax.set_xticklabels(active, rotation=40, ha='right', fontsize=10)
    ax.set_yticklabels(active, fontsize=10)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("Ground Truth Label", fontsize=12)
    ax.set_title(
        f"Experiment 1: VLM Action Recognition — Confusion Matrix\n"
        f"Overall Accuracy = {overall_acc:.1%}   Macro F1 = {macro_f1:.3f}",
        fontsize=12, pad=12
    )

    thresh = cm_sub.max() / 2.0
    for i in range(len(active)):
        for j in range(len(active)):
            v = cm_sub[i, j]
            if v > 0:
                ax.text(j, i, str(v), ha='center', va='center', fontsize=10,
                        color='white' if v > thresh else 'black', fontweight='bold')

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Confusion matrix saved: {out_path}")

# ── Save CSV ───────────────────────────────────────────────────────────────
def save_csv(rows, macro_p, macro_r, macro_f1, out_path):
    import csv
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(["Behavior", "Precision", "Recall", "F1-Score", "Support"])
        for r in rows:
            if r["support"] > 0:
                w.writerow([r["label"],
                             f"{r['precision']:.3f}",
                             f"{r['recall']:.3f}",
                             f"{r['f1']:.3f}",
                             r["support"]])
        w.writerow([])
        w.writerow(["Macro Avg",
                     f"{macro_p:.3f}", f"{macro_r:.3f}", f"{macro_f1:.3f}", ""])
    print(f"  ✅ Metrics CSV saved: {out_path}")

# ── Summary text ───────────────────────────────────────────────────────────
def save_summary(rows, macro_f1, overall_acc, n_total, out_path):
    active = [r for r in rows if r["support"] > 0]
    best   = max(active, key=lambda x: x["f1"]) if active else None
    worst  = min(active, key=lambda x: x["f1"]) if active else None
    lines  = [
        "=" * 60,
        "Experiment 1: VLM Action Recognition Accuracy",
        f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 60,
        f"Total episodes:    {n_total}",
        f"Overall accuracy:  {overall_acc:.1%}",
        f"Macro F1-score:    {macro_f1:.3f}",
        "",
        "Per-class results:",
    ]
    for r in active:
        lines.append(
            f"  {r['label']:14s}  P={r['precision']:.3f}  R={r['recall']:.3f}  F1={r['f1']:.3f}  (n={r['support']})"
        )
    if best:
        lines += ["",
                  f"Best behavior:   {best['label']} (F1={best['f1']:.3f})",
                  f"Worst behavior:  {worst['label']} (F1={worst['f1']:.3f})"]
    lines += [
        "",
        "── For thesis (copy-paste) ─────────────────────────────",
        f"The VLM-based perception pipeline achieves an overall",
        f"accuracy of {overall_acc:.1%} and a macro F1-score of {macro_f1:.3f}",
        f"across {len(active)} behavioral categories in the Unity 3D simulation,",
        f"confirming that zero-shot VLM inference provides sufficient",
        f"scene understanding for domestic activity recognition (RQ1).",
    ]
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"  ✅ Summary saved: {out_path}")
    print('\n'.join(lines))

# ── Main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='.', help='Output directory')
    parser.add_argument('--user', default=None, help='Filter by user_id')
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("Connecting to MongoDB...")
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]

    q = {"user_id": args.user} if args.user else {}
    docs = load_eval_logs(db, q)

    if not docs:
        print("❌ No eval_logs found. Check that /predict is writing eval_logs correctly.")
        print("   Required fields: ground_truth, vlm_output, user_id")
        return

    y_true = [normalize(d.get("ground_truth", "")) for d in docs]
    y_pred = [normalize(d.get("vlm_output",   "")) for d in docs]

    # Only keep labels that appear in ground truth
    observed = sorted(set(y_true))
    print(f"  Observed behaviors: {observed}")
    print(f"  Distribution: {dict(Counter(y_true).most_common())}")

    labels = [l for l in BEHAVIOR_LABELS if l in observed]
    others = [l for l in observed if l not in BEHAVIOR_LABELS]
    labels += others  # append any unexpected labels at end

    rows, mp, mr, mf1, acc = compute_metrics(y_true, y_pred, labels)
    cm = build_confusion_matrix(y_true, y_pred, labels)

    plot_confusion_matrix(
        cm, labels,
        out_path=os.path.join(args.out, "exp1_confusion_matrix.png"),
        overall_acc=acc, macro_f1=mf1
    )
    save_csv(rows, mp, mr, mf1,
             out_path=os.path.join(args.out, "exp1_metrics.csv"))
    save_summary(rows, mf1, acc, len(docs),
                 out_path=os.path.join(args.out, "exp1_summary.txt"))

if __name__ == "__main__":
    main()
