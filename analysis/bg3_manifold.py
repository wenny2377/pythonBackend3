import os, sys, math
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from config import Config

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
from pymongo import MongoClient

MONGO_URI = "mongodb://127.0.0.1:27017/"
DB_NAME   = Config.DB_NAME
OUT       = os.path.join(_ROOT, "analysis", "results")

BEHAVIOR_LABELS = [
    "Drinking", "SittingDrink", "Sitting", "Eating", "Cooking", "Opening",
    "Laying", "Watching", "Reading", "Cleaning", "PhoneUse",
    "Typing", "StandUp", "PickingUp", "PuttingDown", "Standing", "Walking",
]

def connect():
    return MongoClient(MONGO_URI)[DB_NAME]

def load_data(db):
    docs = list(db.manifold_training_data.find(
        {}, {"X": 1, "y": 1, "user_id": 1, "action": 1}
    ))
    if not docs:
        print("[BG3] No manifold_training_data found.")
        return None, None, None
    X     = np.array([d["X"] for d in docs], dtype=np.float32)
    y     = np.array([d["y"] for d in docs], dtype=np.int64)
    users = [d.get("user_id", "?") for d in docs]
    return X, y, users

def compute_silhouette(X, y):
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    unique, counts = np.unique(y, return_counts=True)
    valid  = unique[counts >= 2]
    mask   = np.isin(y, valid)
    X_f, y_f = X[mask], y[mask]
    if len(np.unique(y_f)) < 2:
        print("[BG3] Not enough classes for silhouette.")
        return None
    X_scaled = StandardScaler().fit_transform(X_f)
    return float(silhouette_score(X_scaled, y_f, metric="cosine"))

def plot_umap(X, y, save_path):
    try:
        import umap
        from sklearn.preprocessing import StandardScaler
        X_scaled  = StandardScaler().fit_transform(X)
        reducer   = umap.UMAP(n_components=2, metric="cosine",
                              random_state=42, n_neighbors=15, min_dist=0.1)
        embedding = reducer.fit_transform(X_scaled)
        unique_labels = np.unique(y)
        colors        = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
        fig, ax = plt.subplots(figsize=(12, 8))
        for label, color in zip(unique_labels, colors):
            mask   = y == label
            action = BEHAVIOR_LABELS[label] if label < len(BEHAVIOR_LABELS) else str(label)
            ax.scatter(embedding[mask, 0], embedding[mask, 1],
                       c=[color], label=action, alpha=0.6, s=20)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        ax.set_title("Behavior Manifold — UMAP Projection", fontsize=14)
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[BG3] UMAP saved → {save_path}")
    except ImportError as e:
        print(f"[BG3] UMAP skipped: {e}")

def plot_silhouette_bar(scores_by_user, overall, save_path):
    users  = list(scores_by_user.keys()) + ["Overall"]
    scores = [scores_by_user[u] for u in scores_by_user] + [overall]
    colors = ["#1976D2", "#E91E63", "#5C6BC0"]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(users, scores, color=colors[:len(users)], width=0.5)
    ax.axhline(y=0.50, color="red", linestyle="--", linewidth=1.2, label="Threshold (0.50)")
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01, f"{score:.4f}",
                ha="center", va="bottom", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Silhouette Score")
    ax.set_title("BG3 — Behavioral Manifold Cluster Quality")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[BG3] Silhouette bar saved → {save_path}")

def main():
    os.makedirs(OUT, exist_ok=True)
    print("=" * 60)
    print("BG3 Manifold Analysis — Silhouette Score")
    print("=" * 60)

    db       = connect()
    X, y, users = load_data(db)
    if X is None:
        return

    print(f"\n[Data] total samples : {len(X)}")
    print(f"[Data] feature dim   : {X.shape[1]}")

    user_counts   = Counter(users)
    action_counts = Counter(
        BEHAVIOR_LABELS[yi] for yi in y if yi < len(BEHAVIOR_LABELS))

    print(f"\n[Data] per-user samples:")
    for u, c in sorted(user_counts.items()):
        print(f"  {u}: {c}")

    print(f"\n[Data] per-action samples (top 10):")
    for action, c in action_counts.most_common(10):
        print(f"  {action:<16} {c}")

    print(f"\n[Computing] Overall Silhouette Score...")
    overall = compute_silhouette(X, y)
    if overall is not None:
        print(f"\n[Result] Overall Silhouette Score = {overall:.4f}")
        status = "PASSED" if overall >= 0.50 else ("MODERATE" if overall >= 0.25 else "LOW")
        print(f"[Result] {status}")

    scores_by_user = {}
    print(f"\n[Per-user Silhouette]")
    for uid in sorted(set(users)):
        mask   = np.array([u == uid for u in users])
        s      = compute_silhouette(X[mask], y[mask])
        if s is not None:
            scores_by_user[uid] = s
            print(f"  {uid}: {s:.4f}")

    plot_umap(X, y, os.path.join(OUT, "bg3_umap.png"))
    if overall is not None:
        plot_silhouette_bar(scores_by_user, overall,
                            os.path.join(OUT, "bg3_silhouette.png"))

    summary_path = os.path.join(OUT, "bg3_manifold_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"BG3 Manifold Analysis\n")
        f.write(f"DB: {DB_NAME}\n")
        f.write(f"Total samples: {len(X)}\n")
        f.write(f"Feature dim:   {X.shape[1]}\n")
        f.write(f"Overall Silhouette Score: {overall:.4f}\n" if overall else "N/A\n")
        for u, s in scores_by_user.items():
            f.write(f"  {u}: {s:.4f}\n")
    print(f"[BG3] Summary saved → {summary_path}")

if __name__ == "__main__":
    main()