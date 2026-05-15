import argparse
import csv
import datetime
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

MONGO_URI = "mongodb://127.0.0.1:27017/"
DB_NAME   = "robot_rag_db"

THRESHOLDS = [2, 3, 5, 8, 10]
USERS      = ["User_Mom", "User_Dad"]
DEDUP_SIM  = 0.78
GT_MIN_WEIGHT = 3


def get_grouped_obs(db, user_id):
    pipeline = [
        {"$match": {"user": user_id}},
        {"$group": {
            "_id": {"action": "$action", "instance": "$instance"},
            "total_weight": {"$sum": "$weight"},
        }},
        {"$sort": {"total_weight": -1}},
    ]
    return [
        {"action":   r["_id"]["action"],
         "instance": r["_id"]["instance"],
         "weight":   r["total_weight"]}
        for r in db.observation_logs.aggregate(pipeline)
    ]


def get_habits_at(grouped, threshold):
    return [o for o in grouped if o["weight"] >= threshold]


def calc_redundancy(habits, model):
    if len(habits) < 2:
        return 0.0
    texts = [f"{h['action']} near {h['instance']}" for h in habits]
    vecs  = model.encode(texts, normalize_embeddings=True)
    pairs = redundant = 0
    for i in range(len(vecs)):
        for j in range(i+1, len(vecs)):
            pairs += 1
            if float(np.dot(vecs[i], vecs[j])) >= DEDUP_SIM:
                redundant += 1
    return redundant / pairs if pairs > 0 else 0.0


def calc_recall(habits, ground_truth):
    if not ground_truth:
        return 0.0
    gt_keys = {f"{h['action']}@{h['instance']}" for h in ground_truth}
    lk      = {f"{h['action']}@{h['instance']}" for h in habits}
    return len(gt_keys & lk) / len(gt_keys)


def plot_fat_curve(all_results, out_path):
    users = list(all_results.keys())
    n     = len(users)
    fig, axes = plt.subplots(1, n, figsize=(6*n, 5.5), sharey=True)
    if n == 1:
        axes = [axes]
    fig.suptitle(
        "FAT Sensitivity Analysis: Precision / Recall / F1",
        fontsize=13, fontweight="bold", y=1.02)

    for ax, user_id in zip(axes, users):
        user_results = all_results[user_id]
        fats       = [r["threshold"] for r in user_results]
        precisions = [r["precision"] for r in user_results]
        recalls    = [r["recall"]    for r in user_results]
        f1s        = [r["f1"]        for r in user_results]
        x          = list(range(len(fats)))

        if 5 in fats:
            fi = fats.index(5)
            ax.axvline(x=fi, color="#E53935", linewidth=1.8,
                       linestyle="--", alpha=0.7, label="FAT=5 (selected)")

        ax.plot(x, recalls,    "o-", color="#2196F3", linewidth=2.2,
                markersize=8, markerfacecolor="white",
                markeredgewidth=2, label="Recall")
        ax.plot(x, precisions, "s-", color="#FF9800", linewidth=2.2,
                markersize=8, markerfacecolor="white",
                markeredgewidth=2, label="Precision")
        ax.plot(x, f1s,        "^-", color="#4CAF50", linewidth=2.5,
                markersize=9, markerfacecolor="white",
                markeredgewidth=2.5, label="F1")

        for i, (r, p, f) in enumerate(zip(recalls, precisions, f1s)):
            ax.text(i, r+0.02, f"{r:.2f}", ha="center",
                    fontsize=8, color="#1565C0")
            ax.text(i, p-0.05, f"{p:.2f}", ha="center",
                    fontsize=8, color="#E65100")

        ax.set_xticks(x)
        ax.set_xticklabels([f"FAT={v}" for v in fats], fontsize=10)
        ax.set_ylim(0, 1.2)
        ax.set_xlabel("Fast Adaptation Threshold", fontsize=11)
        if ax == axes[0]:
            ax.set_ylabel("Score", fontsize=11)
        ax.set_title(user_id.replace("_", " "),
                     fontsize=12, fontweight="bold")
        ax.legend(loc="lower left", fontsize=9)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def save_csv(all_results, out_path):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["user_id", "fat", "count", "redundancy",
                    "precision", "recall", "f1"])
        for user_id, results in all_results.items():
            for r in results:
                w.writerow([user_id, r["threshold"], r["count"],
                            f"{r['redundancy']:.4f}",
                            f"{r['precision']:.4f}",
                            f"{r['recall']:.4f}",
                            f"{r['f1']:.4f}"])
    print(f"  Saved: {out_path}")


def save_summary(all_results, out_path):
    lines = ["="*65,
             "FAT Threshold Analysis (Supplementary)",
             f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
             "="*65, ""]
    for user_id, results in all_results.items():
        lines.append(f"[{user_id}]")
        lines.append(f"  {'FAT':<5} {'n':<4} {'redund':<8}"
                     f" {'recall':<8} {'F1':<8}")
        lines.append("  " + "-"*40)
        for r in results:
            marker = "  <- SELECTED" if r["threshold"] == 5 else ""
            lines.append(f"  FAT={r['threshold']:<2} "
                         f"n={r['count']:<3} "
                         f"redund={r['redundancy']:.2f}  "
                         f"recall={r['recall']:.2f}  "
                         f"F1={r['f1']:.2f}"
                         f"{marker}")
        best = max(results, key=lambda x: x["f1"])
        lines += ["",
                  f"  Best F1 = FAT={best['threshold']}"
                  f" (F1={best['f1']:.2f})", ""]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results")
    parser.add_argument("--gt",  type=int, default=GT_MIN_WEIGHT)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    client = MongoClient(MONGO_URI)
    db     = client[DB_NAME]

    n = db.observation_logs.count_documents({})
    if n == 0:
        print("No observation_logs. Run Experiment3 first.")
        return
    print(f"observation_logs: {n}")

    print("Loading SBERT (CPU)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    all_results = {}
    for user_id in USERS:
        grouped      = get_grouped_obs(db, user_id)
        ground_truth = [o for o in grouped
                        if o["weight"] >= args.gt]
        print(f"\n[{user_id}] GT habits (weight>={args.gt}): "
              f"{len(ground_truth)}")

        user_results = []
        for threshold in THRESHOLDS:
            habits     = get_habits_at(grouped, threshold)
            redundancy = calc_redundancy(habits, model)
            precision  = 1.0 - redundancy
            recall     = calc_recall(habits, ground_truth)
            f1         = (2*precision*recall/(precision+recall)
                          if precision+recall > 0 else 0.0)
            user_results.append({
                "threshold":  threshold,
                "count":      len(habits),
                "redundancy": redundancy,
                "precision":  precision,
                "recall":     recall,
                "f1":         f1,
            })
            marker = " <- FAT=5" if threshold == 5 else ""
            print(f"  FAT={threshold}  n={len(habits)}  "
                  f"redund={redundancy:.2f}  "
                  f"recall={recall:.2f}  F1={f1:.2f}{marker}")
        all_results[user_id] = user_results

    plot_fat_curve(all_results,
                   os.path.join(args.out, "fat_curve.png"))
    save_csv(all_results,
             os.path.join(args.out, "fat_data.csv"))
    save_summary(all_results,
                 os.path.join(args.out, "fat_summary.txt"))


if __name__ == "__main__":
    main()