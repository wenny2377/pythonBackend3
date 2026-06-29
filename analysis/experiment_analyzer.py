import datetime
from collections import defaultdict

try:
    import numpy as np
    _NP_OK = True
except ImportError:
    _NP_OK = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _PLT_OK = True
except ImportError:
    _PLT_OK = False

BEHAVIOR_LABELS = [
    "Drinking", "SeatedDrinking", "Sitting", "Eating", "Cooking", "Opening",
    "Laying", "Watching", "Reading", "Cleaning", "UsingPhone",
    "Typing", "StandUp", "PickingUp", "PuttingDown", "Standing", "Walking",
]

GT_NORMALIZE = {
    "seateddrinking": "Drinking",
    "dadreading":     "Reading",
    "dadcleaning":    "Cleaning",
    "dadphone":       "UsingPhone",
}


def _norm(label: str) -> str:
    if not label:
        return label
    key = label.lower().strip().replace(" ", "").replace("_", "")
    return GT_NORMALIZE.get(key, label)


def _fetch_logs(db, collection_suffix: str = "", query: dict = None) -> list:
    col_name = f"experiment_logs{collection_suffix}" if collection_suffix else "experiment_logs"
    q = query or {}
    return list(db[col_name].find(q, {"_id": 0}))


def confusion_matrix(db, experiment_mode: str = "baseline",
                     collection_suffix: str = "") -> dict:
    logs = _fetch_logs(db, collection_suffix,
                       {"experiment_mode": experiment_mode,
                        "ground_truth": {"$exists": True, "$ne": ""}})
    classes = sorted(set(BEHAVIOR_LABELS))
    idx     = {c: i for i, c in enumerate(classes)}
    n       = len(classes)
    mat     = [[0] * n for _ in range(n)]

    for log in logs:
        gt   = _norm(log.get("ground_truth", ""))
        pred = _norm(log.get("predicted", "") or log.get("spatial_action", ""))
        if gt in idx and pred in idx:
            mat[idx[gt]][idx[pred]] += 1

    return {"classes": classes, "matrix": mat, "total": len(logs)}


def per_class_accuracy(db, experiment_mode: str = "baseline",
                        collection_suffix: str = "") -> dict:
    cm   = confusion_matrix(db, experiment_mode, collection_suffix)
    mat  = cm["matrix"]
    clss = cm["classes"]
    result = {}
    for i, c in enumerate(clss):
        row_sum = sum(mat[i])
        correct = mat[i][i]
        result[c] = {
            "correct": correct,
            "total":   row_sum,
            "acc":     round(correct / row_sum, 4) if row_sum > 0 else 0.0,
        }
    return result


def overall_accuracy(db, experiment_mode: str = "baseline",
                      collection_suffix: str = "") -> float:
    logs = _fetch_logs(db, collection_suffix,
                       {"experiment_mode": experiment_mode,
                        "ground_truth": {"$exists": True, "$ne": ""}})
    if not logs:
        return 0.0
    correct = sum(
        1 for log in logs
        if _norm(log.get("ground_truth", "")) == _norm(
            log.get("predicted", "") or log.get("spatial_action", ""))
    )
    return round(correct / len(logs), 4)


def delta_accuracy(db, baseline_mode: str = "baseline",
                    corruption_mode: str = "corruption_light",
                    collection_suffix: str = "") -> dict:
    base_acc = overall_accuracy(db, baseline_mode, collection_suffix)
    corr_acc = overall_accuracy(db, corruption_mode, collection_suffix)
    return {
        "baseline":        base_acc,
        "corruption":      corr_acc,
        "delta":           round(corr_acc - base_acc, 4),
        "relative_drop":   round((base_acc - corr_acc) / base_acc, 4) if base_acc > 0 else 0.0,
    }


def modality_ablation_table(db, collection_suffix: str = "") -> list:
    ablation_modes = ["full", "no_skeleton", "no_object", "no_vlm"]
    rows = []
    for mode in ablation_modes:
        logs = _fetch_logs(db, collection_suffix,
                           {"ablation_mode": mode,
                            "ground_truth": {"$exists": True, "$ne": ""}})
        if not logs:
            rows.append({"ablation_mode": mode, "n": 0, "accuracy": 0.0})
            continue
        correct = sum(
            1 for log in logs
            if _norm(log.get("ground_truth", "")) == _norm(
                log.get("predicted", "") or log.get("spatial_action", ""))
        )
        rows.append({
            "ablation_mode": mode,
            "n":             len(logs),
            "accuracy":      round(correct / len(logs), 4),
        })
    return rows


def behavior_pattern_comparison(db, user_id: str = None,
                                  collection_suffix: str = "") -> dict:
    query = {}
    if user_id:
        query["user"] = user_id
    logs = _fetch_logs(db, collection_suffix, query)
    gt_counts   = defaultdict(int)
    pred_counts = defaultdict(int)
    for log in logs:
        gt   = _norm(log.get("ground_truth", ""))
        pred = _norm(log.get("predicted", "") or log.get("spatial_action", ""))
        if gt:   gt_counts[gt]   += 1
        if pred: pred_counts[pred] += 1
    return {
        "ground_truth_distribution": dict(gt_counts),
        "predicted_distribution":    dict(pred_counts),
    }


def run_ablation_from_eval_logs(db, ablation_mode: str,
                                 collection_suffix: str = "") -> dict:
    logs = list(db.eval_logs.find(
        {"experiment_mode": "baseline"},
        {"ground_truth": 1, "spatial_action": 1, "user": 1,
         "virtual_hour": 1, "time_slot": 1}))
    if not logs:
        return {"error": "no baseline logs found"}

    for log in logs:
        gt   = _norm(log.get("ground_truth", ""))
        pred = _norm(log.get("spatial_action", ""))
        db[f"experiment_logs{collection_suffix}"].update_one(
            {"episode_id": log.get("episode_id", ""),
             "ablation_mode": ablation_mode},
            {"$setOnInsert": {
                "ground_truth":   gt,
                "spatial_action": pred,
                "predicted":      pred,
                "ablation_mode":  ablation_mode,
                "experiment_mode": "baseline",
                "timestamp":      datetime.datetime.utcnow(),
                "source":         "rerun_from_eval_logs",
            }},
            upsert=True,
        )
    return {"status": "ok", "reprocessed": len(logs)}


def plot_corruption_bar_chart(db, out_path: str = "corruption_accuracy.png",
                               collection_suffix: str = ""):
    if not _PLT_OK:
        print("[Analyzer] matplotlib not available")
        return

    modes  = ["baseline", "corruption_light", "corruption_medium", "corruption_heavy"]
    labels = ["Baseline", "Light", "Medium", "Heavy"]
    accs   = [overall_accuracy(db, m, collection_suffix) for m in modes]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, [a * 100 for a in accs], color=["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy under Sensor Corruption Levels")
    ax.set_ylim(0, 100)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{acc:.1%}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[Analyzer] saved {out_path}")


def plot_behavior_pattern_bar(db, user_id: str = None,
                               out_path: str = "behavior_distribution.png",
                               collection_suffix: str = ""):
    if not _PLT_OK or not _NP_OK:
        print("[Analyzer] matplotlib/numpy not available")
        return

    dist = behavior_pattern_comparison(db, user_id, collection_suffix)
    gt   = dist["ground_truth_distribution"]
    pred = dist["predicted_distribution"]
    keys = sorted(set(list(gt.keys()) + list(pred.keys())))

    gt_vals   = [gt.get(k, 0)   for k in keys]
    pred_vals = [pred.get(k, 0) for k in keys]

    x   = np.arange(len(keys))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w/2, gt_vals,   w, label="Ground Truth", color="#4e79a7")
    ax.bar(x + w/2, pred_vals, w, label="Predicted",    color="#f28e2b")
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Count")
    ax.set_title(f"Behavior Distribution — {user_id or 'All Users'}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[Analyzer] saved {out_path}")


def print_summary(db, collection_suffix: str = ""):
    print("\n=== Experiment Summary ===")
    for mode in ["baseline", "corruption_light", "corruption_medium", "corruption_heavy"]:
        acc = overall_accuracy(db, mode, collection_suffix)
        print(f"  {mode:25s}: {acc:.2%}")

    print("\n--- Ablation Table ---")
    for row in modality_ablation_table(db, collection_suffix):
        print(f"  {row['ablation_mode']:15s}: {row['accuracy']:.2%} (n={row['n']})")

    print("\n--- Delta (baseline vs corruption) ---")
    for cm in ["corruption_light", "corruption_medium", "corruption_heavy"]:
        d = delta_accuracy(db, "baseline", cm, collection_suffix)
        print(f"  {cm}: delta={d['delta']:+.2%} relative_drop={d['relative_drop']:.2%}")
    print()