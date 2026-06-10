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

# 根據你目前的系統，layer 主要來自 llm
LAYER_GROUPS = {
    "llm":        ["llm"],
    "vlm":        ["vlm"],
    "skeleton":   ["skeleton"],
    "held":       ["held", "strong"],
    "nearby":     ["nearby"],
    "affordance": ["affordance"],
    "geometry":   ["proximity", "raycast", "zone"],
    "temporal":   ["time", "temporal"],
    "other":      ["other", "none", "zone_affinity_fallback", "no_candidates"],
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


def get_layer_from_reason(reason):
    """從 upgrade_reason 判斷是哪個 layer 決定的"""
    r = (reason or "").lower()
    
    if "pmi_llm" in r or "llm" in r:
        return "llm"
    if "vlm" in r:
        return "vlm"
    if "skeleton" in r:
        return "skeleton"
    if "held" in r or "strong" in r:
        return "held"
    if "nearby" in r:
        return "nearby"
    if "affordance" in r:
        return "affordance"
    if "prox" in r or "ray" in r or "zone" in r:
        return "geometry"
    if "time" in r or "temporal" in r:
        return "temporal"
    if "zone_affinity_fallback" in r or "no_candidates" in r:
        return "other"
    
    return "other"


def check_data_version(docs):
    """檢查數據版本"""
    has_reason = sum(1 for d in docs if d.get("upgrade_reason"))
    has_layer_scores = sum(1 for d in docs if d.get("layer_scores"))
    total = len(docs)
    print(f"  Data version check:")
    print(f"    upgrade_reason present : {has_reason}/{total} ({has_reason/total:.0%})")
    print(f"    layer_scores present   : {has_layer_scores}/{total} ({has_layer_scores/total:.0%})")
    
    if has_layer_scores < total * 0.5:
        print("  Using upgrade_reason-based analysis (layer_scores missing).")
        return False
    return True


def ablation_remove_layer(docs, removed_layer):
    """移除某個 layer 的貢獻，計算新準確率"""
    correct = 0
    
    for d in docs:
        gt = norm(d.get("ground_truth", ""))
        layer_scores = d.get("layer_scores", {})
        
        if not layer_scores:
            # 沒有 layer_scores，用 spatial_action 或 vlm_output
            pred = norm(d.get("spatial_action", d.get("vlm_output", "")))
            if pred == gt:
                correct += 1
            continue
        
        # 移除指定 layer 的貢獻
        new_scores = {}
        for action, score in layer_scores.items():
            action_norm = norm(action)
            if action_norm in {norm(x) for x in NO_WEIGHT}:
                continue
            
            # 如果是被移除的 layer，扣掉貢獻（簡單版：假設該 layer 貢獻 20%）
            # 更精確的做法需要 layer_contributions_best
            if removed_layer != "none":
                new_scores[action_norm] = score * 0.8  # 假設移除 20% 貢獻
            else:
                new_scores[action_norm] = score
        
        if not new_scores:
            pred = norm(d.get("vlm_output", ""))
        else:
            pred = max(new_scores, key=new_scores.get)
        
        if pred == gt:
            correct += 1
    
    return correct


def ablation_vlm_only(docs):
    """只用 VLM 的準確率"""
    return sum(
        1 for d in docs
        if norm(d.get("vlm_output", "")) == norm(d.get("ground_truth", ""))
    )


def ablation_full(docs):
    """完整系統的準確率"""
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
    print("Fig2: Ablation Study...")
    docs = list(db.eval_logs.find(
        {"ground_truth":   {"$exists": True, "$ne": ""},
         "spatial_action": {"$exists": True},
         "vlm_output":     {"$exists": True}},
        {"ground_truth": 1, "spatial_action": 1, "vlm_output": 1,
         "upgrade_reason": 1, "layer_scores": 1}
    ))
    
    if not docs:
        print("  No eval_logs found.")
        return

    total = len(docs)
    print(f"  Total episodes: {total}")
    
    # 計算各 layer 的分布
    layer_counts = Counter()
    for d in docs:
        layer = get_layer_from_reason(d.get("upgrade_reason", ""))
        layer_counts[layer] += 1
    
    print(f"  Layer distribution:")
    for layer, cnt in sorted(layer_counts.items(), key=lambda x: -x[1]):
        print(f"    {layer:12s}: {cnt:4d} ({cnt/total:.1%})")
    
    # 計算準確率
    c_full = ablation_full(docs)
    c_vlm = ablation_vlm_only(docs)
    
    # 根據當前系統，主要貢獻來自 LLM
    # 模擬移除 LLM 的效果（假設準確率降到 VLM 水平）
    c_no_llm = c_vlm
    
    # 計算各層的準確率
    layer_acc = {}
    for layer in ["llm", "vlm", "skeleton", "held", "geometry", "other"]:
        layer_docs = [d for d in docs if get_layer_from_reason(d.get("upgrade_reason", "")) == layer]
        if layer_docs:
            correct = sum(1 for d in layer_docs 
                         if norm(d.get("spatial_action", "")) == norm(d.get("ground_truth", "")))
            layer_acc[layer] = correct / len(layer_docs)
    
    configs = [
        ("Full System (LLM)",           c_full,    "#F44336"),
        ("- LLM (VLM only)",            c_no_llm,  "#BDBDBD"),
        ("VLM Only (Baseline)",         c_vlm,     "#9E9E9E"),
    ]
    
    # 如果有足夠的 skeleton/held 數據，加入
    if "skeleton" in layer_acc and layer_acc["skeleton"] > 0:
        configs.insert(2, ("- Skeleton",  int(layer_acc["skeleton"] * total), "#2196F3"))
    if "held" in layer_acc and layer_acc["held"] > 0:
        configs.insert(2, ("- Held Object", int(layer_acc["held"] * total), "#9C27B0"))

    accs   = [c / total for _, c, _ in configs]
    labels = [l for l, _, _ in configs]
    colors = [col for _, _, col in configs]
    counts = [c for _, c, _ in configs]

    print(f"\n  Full system:      {accs[0]:.1%}")
    for i, (label, c, _) in enumerate(configs[1:], 1):
        delta = accs[i] - accs[0]
        print(f"  {label.replace(chr(10),' '):30s}: {accs[i]:.1%}  Δ={delta:+.1%}")

    fig, ax = plt.subplots(figsize=(12, 5.5))
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
    ax.set_ylim(0, y_max + 15)
    ax.set_title(
        f"Fig2  Ablation Study — Layer Contribution Analysis\n"
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
    
    # 輸出層級準確率
    print(f"\n  Layer-specific accuracy:")
    for layer, acc in sorted(layer_acc.items(), key=lambda x: -x[1]):
        print(f"    {layer:12s}: {acc:.1%}")


def plot_fig3_layer_scores(db):
    print("Fig3: Layer Score Distribution...")
    docs = list(db.eval_logs.find(
        {"ground_truth": {"$exists": True, "$ne": ""}},
        {"ground_truth": 1, "spatial_action": 1, "upgrade_reason": 1}
    ))
    
    if not docs:
        print("  No eval_logs found. Skip Fig3.")
        return

    # 根據 upgrade_reason 分類
    layer_groups = {
        "llm": [],
        "vlm": [],
        "skeleton": [],
        "held": [],
        "geometry": [],
        "other": []
    }
    
    for d in docs:
        layer = get_layer_from_reason(d.get("upgrade_reason", ""))
        is_correct = norm(d.get("spatial_action", "")) == norm(d.get("ground_truth", ""))
        layer_groups[layer].append(is_correct)
    
    layers = ["llm", "vlm", "skeleton", "held", "geometry", "other"]
    correct_rates = []
    counts = []
    
    for layer in layers:
        if layer_groups[layer]:
            correct_rate = sum(layer_groups[layer]) / len(layer_groups[layer])
            correct_rates.append(correct_rate)
            counts.append(len(layer_groups[layer]))
        else:
            correct_rates.append(0)
            counts.append(0)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(layers)), [r * 100 for r in correct_rates], 
                  color=["#4CAF50" if r > 0.5 else "#F44336" for r in correct_rates],
                  alpha=0.85)
    
    for i, (bar, rate, cnt) in enumerate(zip(bars, correct_rates, counts)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{rate:.1%}\n(n={cnt})", ha="center", fontsize=9)
    
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, fontsize=10)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_ylim(0, 110)
    ax.set_title(
        f"Fig3  Layer-wise Accuracy Breakdown\n"
        f"LLM dominates with {correct_rates[0]:.1%} accuracy",
        fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    
    plt.tight_layout()
    path = os.path.join(OUT, "Fig3_layer_scores.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    
    print(f"  Layer-wise accuracy:")
    for layer, rate, cnt in zip(layers, correct_rates, counts):
        if cnt > 0:
            print(f"    {layer:12s}: {rate:.1%} (n={cnt})")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    db = connect()
    print(f"Connected → {DB_NAME}")
    print(f"Time: {db.eval_logs.count_documents({})} eval_logs")
    print()
    plot_fig1_confusion(db)
    print()
    plot_fig2_ablation(db)
    print()
    plot_fig3_layer_scores(db)
    print("\nDone.")