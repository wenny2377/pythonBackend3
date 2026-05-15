"""
analyze_exp4.py
Experiment 4: UMAP Behavioral Manifold Visualization
Uses manifold_points from Experiment3

Outputs:
    results/exp4_umap_behavior.png
    results/exp4_umap_timeslot.png
    results/exp4_summary.txt
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
    print("Missing: pip install umap-learn hdbscan scikit-learn")
    sys.exit(1)

MONGO_URI = "mongodb://127.0.0.1:27017/"
DB_NAME   = "robot_rag_db"

NORMALIZE_MAP = {
    "drinking": "Drinking", "drink": "Drinking",
    "sittingdrink": "SittingDrink",
    "eating": "Eating", "eat": "Eating",
    "cooking": "Cooking", "cook": "Cooking",
    "opening": "Opening", "open": "Opening",
    "laying": "Laying", "lay": "Laying",
    "watching": "Watching", "watch": "Watching",
    "reading": "Reading", "read": "Reading",
    "cleaning": "Cleaning", "clean": "Cleaning",
    "phoneuse": "PhoneUse", "phone": "PhoneUse",
    "typing": "Typing", "type": "Typing",
    "standing": "Standing", "walking": "Walking",
}

BEHAVIOR_COLORS = {
    "Drinking":    "#059669", "SittingDrink": "#34D399",
    "Eating":      "#EC4899", "Cooking":      "#F59E0B",
    "Opening":     "#84CC16", "Laying":       "#6366F1",
    "Watching":    "#DC2626", "Reading":      "#2563EB",
    "Cleaning":    "#7C3AED", "PhoneUse":     "#14B8A6",
    "Typing":      "#9333EA", "Standing":     "#6B7280",
    "Walking":     "#94A3B8", "Unknown":      "#D1D5DB",
}

SLOT_COLORS = {
    "Morning":   "#F59E0B",
    "Noon":      "#EF4444",
    "Afternoon": "#2563EB",
    "Evening":   "#059669",
    "Night":     "#7C3AED",
    "Unknown":   "#9CA3AF",
}


def normalize_action(a):
    if not a:
        return "Unknown"
    s = a.lower().strip().replace(" ", "").replace("_", "")
    return NORMALIZE_MAP.get(s, a.capitalize())


def get_time_slot(vh):
    if vh is None:
        return "Unknown"
    h = float(vh)
    if h < 10:   return "Morning"
    if h < 13:   return "Noon"
    if h < 18:   return "Afternoon"
    if h < 22:   return "Evening"
    return "Night"


def load_data(db):
    docs = list(db.manifold_points.find(
        {"feature_vec": {"$exists": True},
         "is_shadow":   {"$ne": True}},
        {"feature_vec": 1, "action": 1,
         "virtual_hour": 1, "user_id": 1},
    ))
    print(f"  manifold_points (VLM only): {len(docs)}")
    if not docs:
        return None, None, None

    X       = np.array([d["feature_vec"] for d in docs], dtype=np.float32)
    actions = [normalize_action(d.get("action", "")) for d in docs]
    slots   = [get_time_slot(d.get("virtual_hour")) for d in docs]
    return X, actions, slots


def run_umap_hdbscan(X):
    n = len(X)
    if n < 10:
        print(f"  Not enough points ({n})")
        return None, None, None, None

    X_scaled  = StandardScaler().fit_transform(X)
    reducer   = umap.UMAP(
        n_components=2,
        n_neighbors=min(15, n - 1),
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    embedding = reducer.fit_transform(X_scaled)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=max(3, n // 10),
        prediction_data=True,
    )
    labels    = clusterer.fit_predict(embedding)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    s_score = None
    valid   = labels != -1
    if valid.sum() > 1 and n_clusters > 1:
        s_score = silhouette_score(embedding[valid], labels[valid])
        print(f"  Silhouette Score S = {s_score:.4f}")
    print(f"  Clusters: {n_clusters}")
    return embedding, labels, n_clusters, s_score


def plot_by_behavior(X, embedding, actions, s_score, out_path):
    fig, ax = plt.subplots(figsize=(10, 7))
    unique_actions = [a for a in BEHAVIOR_COLORS
                      if any(x == a for x in actions)]
    for act in unique_actions:
        mask = np.array([a == act for a in actions])
        if mask.sum() == 0:
            continue
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=BEHAVIOR_COLORS.get(act, "#999"),
            label=f"{act} (n={mask.sum()})",
            alpha=0.75, s=45,
            edgecolors="white", linewidths=0.3,
        )
    s_str = f"S={s_score:.4f}" if s_score else "S=N/A"
    ax.set_title(
        f"Experiment 4: UMAP Projection — Color by Behavior\n"
        f"n={len(actions)}  {s_str}",
        fontsize=12)
    ax.set_xlabel("UMAP dim 1")
    ax.set_ylabel("UMAP dim 2")
    ax.legend(fontsize=8, framealpha=0.8,
              loc="upper right", ncol=2)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def plot_by_timeslot(embedding, slots, s_score, out_path):
    fig, ax = plt.subplots(figsize=(10, 7))
    markers = {"Morning": "o", "Noon": "s",
               "Afternoon": "^", "Evening": "D", "Night": "P",
               "Unknown": "x"}
    for slot in SLOT_COLORS:
        mask = np.array([s == slot for s in slots])
        if mask.sum() == 0:
            continue
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=SLOT_COLORS[slot],
            marker=markers.get(slot, "o"),
            label=f"{slot} (n={mask.sum()})",
            alpha=0.75, s=45,
            edgecolors="white", linewidths=0.3,
        )
    s_str = f"S={s_score:.4f}" if s_score else "S=N/A"
    ax.set_title(
        f"Experiment 4: UMAP Projection — Color by Time Slot\n"
        f"n={len(slots)}  {s_str}",
        fontsize=12)
    ax.set_xlabel("UMAP dim 1")
    ax.set_ylabel("UMAP dim 2")
    ax.legend(fontsize=9, framealpha=0.8)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def save_summary(actions, slots, n_clusters, s_score, out_path):
    act_counts  = Counter(actions)
    slot_counts = Counter(slots)
    passed      = s_score is not None and s_score >= 0.60
    lines = [
        "=" * 65,
        "Experiment 4: UMAP Behavioral Manifold Visualization",
        f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 65,
        "",
        f"Total points : {len(actions)}",
        f"Clusters     : {n_clusters}",
        f"Silhouette S : {s_score:.4f if s_score else 'N/A'}  "
        f"[{'PASSED (>=0.60)' if passed else 'BELOW 0.60'}]",
        "",
        "Behavior distribution:",
        *[f"  {l:16s}: {c}" for l, c in act_counts.most_common()],
        "",
        "Time slot distribution:",
        *[f"  {l:12s}: {c}" for l, c in slot_counts.most_common()],
        "",
        "For thesis:",
        f"UMAP projection of {len(actions)} behavioral feature vectors",
        f"yielded {n_clusters} clusters with Silhouette Score"
        f" S={s_score:.4f if s_score else 'N/A'}.",
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("Connecting to MongoDB...")
    client = MongoClient(MONGO_URI)
    db     = client[DB_NAME]

    print("\nStep 1: Loading data...")
    X, actions, slots = load_data(db)
    if X is None or len(X) < 10:
        print("Not enough data. Run Experiment3 first.")
        return

    print("\nStep 2: UMAP + HDBSCAN...")
    embedding, labels, n_clusters, s_score = run_umap_hdbscan(X)
    if embedding is None:
        return

    print("\nStep 3: Generating plots...")
    plot_by_behavior(X, embedding, actions, s_score,
        os.path.join(args.out, "exp4_umap_behavior.png"))
    plot_by_timeslot(embedding, slots, s_score,
        os.path.join(args.out, "exp4_umap_timeslot.png"))
    save_summary(actions, slots, n_clusters, s_score,
        os.path.join(args.out, "exp4_summary.txt"))


if __name__ == "__main__":
    main()