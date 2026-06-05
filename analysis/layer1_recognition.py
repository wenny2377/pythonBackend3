import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from pymongo import MongoClient

MONGO_URI = "mongodb://127.0.0.1:27017/"
DB_NAME   = "robot_rag_db"
OUT       = os.path.join(os.path.dirname(__file__), "results")

BEHAVIORS = [
    "Drinking", "SittingDrink", "Sitting", "Eating", "Cooking",
    "Opening", "Laying", "Watching", "Reading", "Cleaning",
    "PhoneUse", "Typing",
]
HIGH   = {"Cooking", "Opening", "Laying", "Cleaning", "PhoneUse", "Typing"}
MEDIUM = {"Eating", "Drinking"}
LOW    = {"SittingDrink", "Sitting", "Reading", "Watching"}

NORMALIZE = {
    "drinking":     "Drinking",
    "sittingdrink": "SittingDrink",
    "sitting":      "Sitting",
    "eating":       "Eating",
    "cooking":      "Cooking",
    "opening":      "Opening",
    "laying":       "Laying",
    "watching":     "Watching",
    "reading":      "Reading",
    "cleaning":     "Cleaning",
    "phoneuse":     "PhoneUse",
    "typing":       "Typing",
    "standing":     "Standing",
    "walking":      "Walking",
    "unknown":      "Unknown",
}

NO_WEIGHT = {"PickingUp", "PuttingDown", "Walking", "Standing", "StandUp"}

LAYER_GROUPS = {
    "skeleton":   ["skeleton"],
    "held":       ["held", "strong"],
    "nearby":     ["nearby"],
    "affordance": ["affordance"],
    "geometry":   ["proximity", "raycast", "zone"],
    "vlm":        ["vlm"],
    "temporal":   ["time", "temporal"],
}


def norm(s):
    if not s:
        return "Unknown"
    return NORMALIZE.get(
        s.lower().strip().replace(" ", "").replace("_", ""),
        s.strip()
    )


def group_color(b):
    if b in HIGH:   return "#F44336"
    if b in MEDIUM: return "#FF9800"
    return "#2196F3"


def connect():
    return MongoClient(MONGO_URI)[DB_NAME]


def check_data_version(docs):
    has_layer_scores = sum(1 for d in docs if d.get("layer_scores"))
    has_head_pitch   = sum(1 for d in docs if d.get("head_pitch", -999) != -999)
    total = len(docs)
    print(f"  Data version check:")
    print(f"    layer_scores present : {has_layer_scores}/{total} ({has_layer_scores/total:.0%})")
    print(f"    head_pitch present   : {has_head_pitch}/{total} ({has_head_pitch/total:.0%})")
    if has_layer_scores < total * 0.5:
        print("  WARNING: Less than 50% of docs have layer_scores.")
        print("           Run experiment with updated perception_engine.py first.")
        return False
    return True


def ablation_one_layer_removed(docs, removed_group):
    correct = 0
    removed_keys = LAYER_GROUPS.get(removed_group, [])

    for d in docs:
        gt            = norm(d.get("ground_truth", ""))
        layer_scores  = d.get("layer_scores", {})
        contrib_best  = d.get("layer_contributions_best", {})

        if not layer_scores:
            pred = norm(d.get("vlm_output", ""))
            if pred == gt:
                correct += 1
            continue

        best_action_orig = norm(d.get("spatial_action", ""))
        total_best = layer_scores.get(best_action_orig, 0.0)

        removed_from_best = sum(
            contrib_best.get(k, 0.0) for k in removed_keys
        )

        if total_best <= 0:
            removed_ratio = 0.0
        else:
            removed_ratio = removed_from_best / total_best

        new_scores = {}
        for b, score in layer_scores.items():
            if norm(b) in {norm(x) for x in NO_WEIGHT}:
                continue
            deduct = score * removed_ratio
            new_scores[b] = max(0.0, score - deduct)

        if not new_scores:
            pred = norm(d.get("vlm_output", ""))
        else:
            pred = norm(max(new_scores, key=new_scores.get))

        if pred == gt:
            correct += 1

    return correct


def ablation_vlm_only(docs):
    return sum(
        1 for d in docs
        if norm(d.get("vlm_output", "")) == norm(d.get("ground_truth", ""))
    )


def ablation_full(docs):
    return sum(
        1 for d in docs
        if norm(d.get("spatial_action", "")) == norm(d.get("ground_truth", ""))
    )


def plot_fig1_confusion(db):
    print("Fig1: Confusion Matrix...")
    docs = list(db.eval_logs.find(
        {"ground_truth":   {"$exists": True, "$ne": ""},
         "spatial_action": {"$exists": True}},
        {"ground_truth": 1, "spatial_action": 1}
    ))
    if not docs:
        print("  No eval_logs found.")
        return

    labels = [b for b in BEHAVIORS
              if any(norm(d["ground_truth"]) == b for d in docs)]
    n = len(labels)
    if n == 0:
        print("  No valid labels.")
        return

    # matrix[gt_idx][pred_idx]
    matrix = np.zeros((n, n), dtype=int)
    for d in docs:
        gt   = norm(d.get("ground_truth", ""))
        pred = norm(d.get("spatial_action", ""))
        if gt in labels and pred in labels:
            matrix[labels.index(gt)][labels.index(pred)] += 1

    total   = int(matrix.sum())
    correct = int(np.trace(matrix))
    overall = correct / total if total > 0 else 0

    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    matrix_norm = matrix / row_sums

    def group_acc(group):
        idxs = [i for i, b in enumerate(labels) if b in group]
        if not idxs:
            return 0
        sub = matrix[np.ix_(idxs, idxs)]
        c, t = int(np.trace(sub)), int(sub.sum())
        return c / t if t > 0 else 0

    high_acc   = group_acc(HIGH)
    medium_acc = group_acc(MEDIUM)
    low_acc    = group_acc(LOW)

    # ── Ground Truth on X axis, Predicted on Y axis ──
    # matrix_norm[gt][pred] → imshow needs [row=Y=pred, col=X=gt]
    # so we transpose: matrix_norm.T[pred][gt]
    fig, ax = plt.subplots(figsize=(max(10, n), max(8, n)))
    im = ax.imshow(matrix_norm.T, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Recall Rate")

    for gt_i in range(n):
        for pred_j in range(n):
            v = matrix_norm[gt_i][pred_j]
            if matrix[gt_i][pred_j] > 0:
                ax.text(gt_i, pred_j,
                        f"{v:.2f}\n({matrix[gt_i][pred_j]})",
                        ha="center", va="center", fontsize=7,
                        color="white" if v > 0.55 else "black",
                        fontweight="bold" if gt_i == pred_j else "normal")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    for tick, b in zip(ax.get_xticklabels(), labels):
        tick.set_color(group_color(b))
    for tick, b in zip(ax.get_yticklabels(), labels):
        tick.set_color(group_color(b))

    ax.set_xlabel("Ground Truth", fontsize=11)
    ax.set_ylabel("Predicted", fontsize=11)

    ax.set_title(
        f"Fig1  Behaviour Recognition Confusion Matrix\n"
        f"Overall = {overall:.1%} ({correct}/{total})  |  "
        f"High: {high_acc:.1%}  Medium: {medium_acc:.1%}  Low: {low_acc:.1%}\n"
        f"[Red=High-specificity  Orange=Medium  Blue=Low-specificity]",
        fontsize=11, fontweight="bold", pad=12)

    plt.tight_layout()
    path = os.path.join(OUT, "Fig1_confusion.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    print(f"  Overall: {overall:.1%}  High: {high_acc:.1%}  "
          f"Medium: {medium_acc:.1%}  Low: {low_acc:.1%}")


def plot_fig2_ablation(db):
    print("Fig2: Ablation Study (layer removal)...")
    docs = list(db.eval_logs.find(
        {"ground_truth":   {"$exists": True, "$ne": ""},
         "spatial_action": {"$exists": True},
         "vlm_output":     {"$exists": True}},
        {"ground_truth": 1, "spatial_action": 1, "vlm_output": 1,
         "upgrade_reason": 1, "layer_scores": 1,
         "layer_contributions_best": 1, "head_pitch": 1}
    ))
    if not docs:
        print("  No eval_logs found.")
        return

    total = len(docs)
    if not check_data_version(docs):
        print("  Falling back to upgrade_reason-based analysis...")
        _plot_fig2_fallback(docs, total)
        return

    c_full    = ablation_full(docs)
    c_vlm     = ablation_vlm_only(docs)
    c_no_skel = ablation_one_layer_removed(docs, "skeleton")
    c_no_held = ablation_one_layer_removed(docs, "held")
    c_no_near = ablation_one_layer_removed(docs, "nearby")
    c_no_geom = ablation_one_layer_removed(docs, "geometry")
    c_no_vlm  = ablation_one_layer_removed(docs, "vlm")
    c_no_temp = ablation_one_layer_removed(docs, "temporal")

    configs = [
        ("Full System",                 c_full,    "#F44336"),
        ("- Skeleton\n(hip+head)",      c_no_skel, "#2196F3"),
        ("- Held Object",               c_no_held, "#9C27B0"),
        ("- Nearby Objects",            c_no_near, "#FF9800"),
        ("- Geometry\n(prox+ray+zone)", c_no_geom, "#4CAF50"),
        ("- VLM",                       c_no_vlm,  "#607D8B"),
        ("- Temporal",                  c_no_temp, "#795548"),
        ("VLM Only\n(Baseline)",        c_vlm,     "#BDBDBD"),
    ]

    accs   = [c / total for _, c, _ in configs]
    labels = [l for l, _, _ in configs]
    colors = [col for _, _, col in configs]
    counts = [c for _, c, _ in configs]

    print(f"  Full system:      {accs[0]:.1%}")
    for i, (label, c, _) in enumerate(configs[1:], 1):
        delta = accs[i] - accs[0]
        print(f"  {label.replace(chr(10),' '):30s}: {accs[i]:.1%}  Δ={delta:+.1%}")

    fig, ax = plt.subplots(figsize=(13, 5.5))
    bars = ax.bar(range(len(configs)),
                  [a * 100 for a in accs],
                  color=colors, alpha=0.85,
                  edgecolor="white", width=0.65)

    full_acc = accs[0] * 100
    ax.axhline(y=full_acc, color="#F44336", linestyle="--",
               linewidth=1.2, alpha=0.6, label=f"Full system {accs[0]:.1%}")

    for i, (bar, acc, cnt) in enumerate(zip(bars, accs, counts)):
        delta = acc - accs[0]
        sign  = "+" if delta >= 0 else ""
        delta_str = "" if i == 0 else f"\n({sign}{delta:.1%})"
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                f"{acc:.1%}{delta_str}",
                ha="center", fontsize=9, fontweight="bold")

    ax.set_xticks(range(len(configs)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    y_max = max(accs) * 100
    ax.set_ylim(0, y_max + 20)
    ax.set_title(
        f"Fig2  Ablation Study — Each Layer Removed Independently\n"
        f"Total = {total} episodes  |  "
        f"VLM-only = {accs[-1]:.1%}  →  Full system = {accs[0]:.1%}  "
        f"(+{accs[0]-accs[-1]:.1%})",
        fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(OUT, "Fig2_ablation.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def _plot_fig2_fallback(docs, total):
    def dominant_layer(reason):
        r = (reason or "").lower()
        if "strong:skeleton_lying" in r:
            return "skeleton"
        if any(k in r for k in ("strong:held:", "strong:head(")):
            return "held"
        if any(k in r for k in ("skeleton_lying", "head(", "skeleton")):
            return "skeleton"
        if "held:" in r:
            return "held"
        if "affordance:" in r:
            return "affordance"
        if "nearby:" in r:
            return "nearby"
        if any(k in r for k in ("prox:", "ray:", "zone:")):
            return "geometry"
        if "vlm(" in r:
            return "vlm"
        if "inertia(" in r:
            return "temporal"
        return "vlm_only"

    for d in docs:
        d["_layer"] = dominant_layer(d.get("upgrade_reason", ""))

    layer_counts = Counter(d["_layer"] for d in docs)
    print(f"  Layer distribution:")
    for layer, cnt in sorted(layer_counts.items(), key=lambda x: -x[1]):
        print(f"    {layer:12s}: {cnt:4d} ({cnt/total:.1%})")

    groups = defaultdict(list)
    for d in docs:
        groups[d["_layer"]].append(d)

    vlm_only_acc = sum(
        1 for d in docs
        if norm(d.get("vlm_output","")) == norm(d.get("ground_truth",""))
    ) / total

    rows = []
    for layer in ["skeleton", "held", "nearby", "geometry", "vlm", "temporal", "vlm_only"]:
        ds = groups[layer]
        if not ds:
            continue
        n = len(ds)
        spatial_correct = sum(
            1 for d in ds
            if norm(d.get("spatial_action","")) == norm(d.get("ground_truth",""))
        )
        vlm_correct = sum(
            1 for d in ds
            if norm(d.get("vlm_output","")) == norm(d.get("ground_truth",""))
        )
        rows.append((layer, n, spatial_correct/n, vlm_correct/n))

    fig, ax = plt.subplots(figsize=(12, 5.5))
    x     = range(len(rows))
    width = 0.35
    spatial_accs = [r[2] * 100 for r in rows]
    vlm_accs     = [r[3] * 100 for r in rows]
    labels_x     = [f"{r[0]}\n(n={r[1]})" for r in rows]

    b1 = ax.bar([i - width/2 for i in x], spatial_accs,
                width, label="With spatial reasoning", color="#2196F3", alpha=0.85)
    b2 = ax.bar([i + width/2 for i in x], vlm_accs,
                width, label="VLM only", color="#BDBDBD", alpha=0.85)

    for bar, acc in zip(list(b1) + list(b2),
                        spatial_accs + vlm_accs):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 1,
                f"{acc:.0f}%",
                ha="center", fontsize=8)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels_x, fontsize=9)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_ylim(0, 115)
    ax.set_title(
        f"Fig2  Layer Override Analysis — Per-Layer Case Accuracy\n"
        f"Total = {total} episodes  |  Overall VLM-only = {vlm_only_acc:.1%}\n"
        f"(Note: re-run with updated perception_engine.py for true ablation)",
        fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    path = os.path.join(OUT, "Fig2_ablation.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved (fallback mode): {path}")


def plot_fig3_layer_scores(db):
    print("Fig3: Layer Score Distribution...")
    docs = list(db.eval_logs.find(
        {"layer_contributions_best": {"$exists": True},
         "ground_truth": {"$exists": True, "$ne": ""}},
        {"ground_truth": 1, "spatial_action": 1,
         "layer_contributions_best": 1}
    ))
    if not docs:
        print("  No layer_contributions_best data found. Skip Fig3.")
        return

    layer_keys = ["skeleton", "held", "nearby", "proximity", "raycast",
                  "zone", "vlm", "time", "temporal"]

    correct_docs = [d for d in docs
                    if norm(d.get("spatial_action","")) == norm(d.get("ground_truth",""))]
    wrong_docs   = [d for d in docs
                    if norm(d.get("spatial_action","")) != norm(d.get("ground_truth",""))]

    def avg_contrib(ds, key):
        vals = [d.get("layer_contributions_best", {}).get(key, 0.0) for d in ds]
        return np.mean(vals) if vals else 0.0

    correct_avgs = [avg_contrib(correct_docs, k) for k in layer_keys]
    wrong_avgs   = [avg_contrib(wrong_docs, k)   for k in layer_keys]

    x     = np.arange(len(layer_keys))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width/2, correct_avgs, width,
           label=f"Correct (n={len(correct_docs)})",
           color="#4CAF50", alpha=0.85)
    ax.bar(x + width/2, wrong_avgs, width,
           label=f"Wrong   (n={len(wrong_docs)})",
           color="#F44336", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(layer_keys, fontsize=10)
    ax.set_ylabel("Average score contribution", fontsize=11)
    ax.set_title(
        f"Fig3  Layer Contribution: Correct vs Wrong Predictions\n"
        f"Total = {len(docs)} episodes with layer_contributions_best",
        fontsize=11, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    path = os.path.join(OUT, "Fig3_layer_scores.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    db = connect()
    print(f"Connected → {DB_NAME}")
    plot_fig1_confusion(db)
    plot_fig2_ablation(db)
    plot_fig3_layer_scores(db)
    print("Done.")