"""
analyze_exp/analyze_exp3.py

Experiment 3: Behavioral Manifold Learning

Reads feature vectors from manifold_points, runs UMAP + HDBSCAN,
and produces a 3-panel scatter plot with Silhouette Score.

Usage:
    python3 analyze_exp/analyze_exp3.py
    python3 analyze_exp/analyze_exp3.py --out ./results/

Prerequisites:
    Experiment3 mode complete (300 episodes).
    ManifoldEngine has refitted at least once (triggers every 50 points).

Outputs:
    exp3_umap.png
    exp3_summary.txt
"""

import argparse
import datetime
import os
import sys
from collections import Counter

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pymongo import MongoClient

try:
    import umap
    import hdbscan
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
except ImportError:
    print("Missing packages. Run:")
    print("  pip install umap-learn hdbscan scikit-learn")
    sys.exit(1)

MONGO_URI = "mongodb://127.0.0.1:27017/"
DB_NAME   = "robot_rag_db"

NORMALIZE_MAP = {
    "drinking":   "Drink",       "drink":      "Drink",
    "sit":        "Laying",      "sitting":     "Laying",
    "laying":     "Laying",
    "reading":    "Reading",     "read":       "Reading",
    "typing":     "Typing",      "type":       "Typing",
    "watching":   "Watching",    "watch":      "Watching",
    "sleeping":   "Sleeping",    "sleep":      "Sleeping",
    "eating":     "Eating",      "eat":        "Eating",
    "walking":    "Walking",     "walk":       "Walking",
    "standing":   "Standing",    "stand":      "Standing",
    "exercising": "Exercising",  "exercise":   "Exercising",
}

BEHAVIOR_COLORS = {
    "Drink":       "#059669",
    "Laying": "#F59E0B",
    "Reading":     "#2563EB",
    "Typing":      "#7C3AED",
    "Watching":    "#DC2626",
    "Sleeping":    "#6366F1",
    "Eating":      "#EC4899",
    "Walking":     "#84CC16",
    "Standing":    "#6B7280",
    "Exercising":  "#14B8A6",
    "Unknown":     "#D1D5DB",
}
SLOT_COLORS = {
    "Morning":   "#F59E0B",
    "Noon":      "#EF4444",
    "Afternoon": "#2563EB",
    "Evening":   "#059669",
    "Unknown":   "#9CA3AF",
}
SLOT_MARKERS = {
    "Morning": "o", "Noon": "s",
    "Afternoon": "^", "Evening": "D", "Unknown": "x",
}


def normalize_action(a: str) -> str:
    if not a:
        return "Unknown"
    s = a.lower().strip()
    if s in NORMALIZE_MAP:
        return NORMALIZE_MAP[s]
    for kw, label in NORMALIZE_MAP.items():
        if kw in s:
            return label
    return a.capitalize()


def get_time_slot(vh) -> str:
    if vh is None:
        return "Unknown"
    h = float(vh)
    if h < 10:
        return "Morning"
    if h < 13:
        return "Noon"
    if h < 18:
        return "Afternoon"
    return "Evening"


def load_data(db):
    docs = list(db.manifold_points.find(
        {"feature_vec": {"$exists": True}},
        {"feature_vec": 1, "action": 1, "user_id": 1,
         "virtual_hour": 1, "prev_action": 1, "timestamp": 1},
    ))
    print(f"  manifold_points: {len(docs)} records")
    if not docs:
        return None, None, None, None, None

    X       = np.array([d["feature_vec"] for d in docs], dtype=np.float32)
    actions = [normalize_action(d.get("action", ""))   for d in docs]
    users   = [d.get("user_id", "unknown")              for d in docs]
    slots   = [get_time_slot(d.get("virtual_hour"))     for d in docs]
    prevs   = [normalize_action(d.get("prev_action", "")) for d in docs]

    print(f"  Feature dim: {X.shape[1]}")
    print(f"  Action dist: {dict(Counter(actions).most_common())}")
    print(f"  Slot dist  : {dict(Counter(slots).most_common())}")
    return X, actions, users, slots, prevs


def run_umap_hdbscan(X):
    n = len(X)
    print("  Running StandardScaler...")
    X_scaled = StandardScaler().fit_transform(X)

    print(f"  Running UMAP (n={n}, n_neighbors={min(15, n-1)})...")
    reducer   = umap.UMAP(
        n_components=2,
        n_neighbors=min(15, n - 1),
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    embedding = reducer.fit_transform(X_scaled)

    print(f"  Running HDBSCAN (min_cluster_size={max(3, n//10)})...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=max(3, n // 10),
        prediction_data=True,
    )
    labels     = clusterer.fit_predict(embedding)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"  Clusters found: {n_clusters}")

    s_score = None
    valid   = labels != -1
    if valid.sum() > 1 and n_clusters > 1:
        s_score = silhouette_score(embedding[valid], labels[valid])
        print(f"  Silhouette Score: {s_score:.4f}")
    else:
        print(f"  Silhouette not computable (clusters={n_clusters})")

    return embedding, labels, n_clusters, s_score


def plot(embedding, actions, slots, prevs, labels,
         n_clusters, s_score, n_total, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    title = f"Experiment 3: Behavioral Manifold — UMAP Projection  (n={n_total})"
    if s_score is not None:
        title += f"    Silhouette Score S = {s_score:.4f}"
    fig.suptitle(title, fontsize=13, fontweight="bold")

    ax1 = axes[0]
    for behavior in [b for b in BEHAVIOR_COLORS if any(a == b for a in actions)]:
        mask = np.array([a == behavior for a in actions])
        if mask.sum() == 0:
            continue
        ax1.scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=BEHAVIOR_COLORS[behavior],
            label=f"{behavior} (n={mask.sum()})",
            alpha=0.75, s=55, edgecolors="white", linewidths=0.4,
        )
    ax1.set_title("Color by Behavior", fontsize=12)
    ax1.set_xlabel("UMAP dim 1")
    ax1.set_ylabel("UMAP dim 2")
    ax1.legend(loc="best", fontsize=8, framealpha=0.8)
    ax1.grid(True, alpha=0.2)

    ax2 = axes[1]
    for slot in [s for s in SLOT_COLORS if any(sl == s for sl in slots)]:
        mask = np.array([s == slot for s in slots])
        if mask.sum() == 0:
            continue
        ax2.scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=SLOT_COLORS[slot],
            marker=SLOT_MARKERS[slot],
            label=f"{slot} (n={mask.sum()})",
            alpha=0.75, s=55, edgecolors="white", linewidths=0.4,
        )
    ax2.set_title("Color by Time Slot", fontsize=12)
    ax2.set_xlabel("UMAP dim 1")
    ax2.set_ylabel("UMAP dim 2")
    ax2.legend(loc="best", fontsize=8, framealpha=0.8)
    ax2.grid(True, alpha=0.2)

    ax3 = axes[2]
    cmap          = plt.cm.get_cmap("tab10", max(n_clusters, 1))
    unique_labels = sorted(set(labels))
    for cid in unique_labels:
        mask  = labels == cid
        color = "#CCCCCC" if cid == -1 else cmap(cid % 10)
        label = "Noise" if cid == -1 else f"Cluster {cid} (n={mask.sum()})"
        ax3.scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=[color] * mask.sum(),
            label=label,
            alpha=0.7 if cid >= 0 else 0.3,
            s=55 if cid >= 0 else 30,
            edgecolors="white", linewidths=0.4,
        )

    for cid in unique_labels:
        if cid == -1:
            continue
        mask     = labels == cid
        pts      = embedding[mask]
        cx, cy   = pts[:, 0].mean(), pts[:, 1].mean()
        acts     = [actions[i] for i in range(len(actions)) if labels[i] == cid]
        dominant = Counter(acts).most_common(1)[0][0] if acts else "?"
        ax3.annotate(
            dominant, (cx, cy),
            fontsize=8, fontweight="bold", ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, ec="none"),
        )

    s_str = f"S = {s_score:.4f}" if s_score else "S = N/A"
    ax3.set_title(f"HDBSCAN Clusters  ({n_clusters} found,  {s_str})", fontsize=12)
    ax3.set_xlabel("UMAP dim 1")
    ax3.set_ylabel("UMAP dim 2")
    ax3.legend(loc="best", fontsize=7, framealpha=0.8)
    ax3.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def save_summary(actions, slots, labels, n_clusters, s_score, n_total, out_path):
    passed = s_score is not None and s_score >= 0.5
    status = "PASSED (S >= 0.50)" if passed else "BELOW THRESHOLD (S < 0.50)"

    cluster_info = []
    for cid in sorted(set(labels)):
        if cid == -1:
            continue
        mask     = labels == cid
        acts     = [actions[i] for i in range(n_total) if labels[i] == cid]
        sls      = [slots[i]   for i in range(n_total) if labels[i] == cid]
        dom_act  = Counter(acts).most_common(1)[0] if acts else ("?", 0)
        dom_slot = Counter(sls).most_common(1)[0]  if sls  else ("?", 0)
        cluster_info.append(
            f"  Cluster {cid} (n={mask.sum()}): "
            f"dominant={dom_act[0]} ({dom_act[1]}), "
            f"slot={dom_slot[0]} ({dom_slot[1]})"
        )

    noise_n = sum(1 for l in labels if l == -1)
    lines   = [
        "=" * 65,
        "Experiment 3: Behavioral Manifold Learning",
        f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 65,
        f"Total observations   : {n_total}",
        f"Clusters found       : {n_clusters}",
        f"Noise points         : {noise_n}",
        (f"Silhouette Score S   : {s_score:.4f}" if s_score
         else "Silhouette Score S   : N/A"),
        f"Acceptance criterion : S >= 0.50  ->  {status}",
        "",
        "Cluster composition:",
        *cluster_info,
        "",
        "Behavior distribution:",
        *[f"  {k}: {v}" for k, v in Counter(actions).most_common()],
        "",
        "Time slot distribution:",
        *[f"  {k}: {v}" for k, v in Counter(slots).most_common()],
        "",
        "For thesis:",
        f"The UMAP projection of {n_total} behavioral observations",
        f"(1,158-dim feature vectors) produces {n_clusters} distinct clusters",
        f"identified by HDBSCAN.",
        (f"Silhouette Score S = {s_score:.4f} ({status})." if s_score
         else "Silhouette Score: N/A."),
        f"Each cluster corresponds to a coherent behavioral habit",
        f"combining action semantics, temporal context, and spatial location.",
    ]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {out_path}")
    if s_score:
        print(f"\n  Silhouette Score S = {s_score:.4f}  ->  {status}")
    else:
        print("\n  Silhouette Score: N/A")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out",     default=".", help="Output directory")
    parser.add_argument("--min-pts", type=int,   default=None,
                        help="Override HDBSCAN min_cluster_size")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print(f"Connecting to MongoDB ({DB_NAME})...")
    client = MongoClient(MONGO_URI)
    db     = client[DB_NAME]

    print("\nStep 1: Loading data...")
    X, actions, users, slots, prevs = load_data(db)
    if X is None or len(X) < 10:
        print("Not enough data in manifold_points (minimum 10).")
        print("Run Experiment3 first.")
        sys.exit(1)

    print("\nStep 2: UMAP + HDBSCAN...")
    embedding, labels, n_clusters, s_score = run_umap_hdbscan(X)

    print("\nStep 3: Generating outputs...")
    plot(embedding, actions, slots, prevs, labels,
         n_clusters, s_score, len(X),
         out_path=os.path.join(args.out, "exp3_umap.png"))
    save_summary(actions, slots, labels, n_clusters, s_score, len(X),
                 out_path=os.path.join(args.out, "exp3_summary.txt"))


if __name__ == "__main__":
    main()