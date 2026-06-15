import os, sys
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
from pymongo import MongoClient

from exp_config import (
    MONGO_URI, DB_BASELINE, USERS,
    C, FONT_TITLE, FONT_AXIS, FONT_ANNOT, FONT_TICK,
    FIG_DPI, RESULTS_DIR, apply_style
)

apply_style()

BEHAVIOR_LABELS = [
    "Drinking", "SittingDrink", "Sitting", "Eating", "Cooking", "Opening",
    "Laying", "Watching", "Reading", "Cleaning", "PhoneUse",
    "Typing", "StandUp", "PickingUp", "PuttingDown", "Standing", "Walking",
]

THRESHOLD = 0.50


def load_data(db):
    docs = list(db.manifold_training_data.find(
        {}, {"X": 1, "y": 1, "user_id": 1}))
    if not docs:
        return None, None, None
    X     = np.array([d["X"] for d in docs], dtype=np.float32)
    y     = np.array([d["y"] for d in docs], dtype=np.int64)
    users = [d.get("user_id", "?") for d in docs]
    return X, y, users


def silhouette(X, y):
    try:
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import StandardScaler
        unique, counts = np.unique(y, return_counts=True)
        valid  = unique[counts >= 2]
        mask   = np.isin(y, valid)
        X_f, y_f = X[mask], y[mask]
        if len(np.unique(y_f)) < 2:
            return None
        X_s = StandardScaler().fit_transform(X_f)
        return float(silhouette_score(X_s, y_f, metric="cosine"))
    except ImportError:
        print("[exp7] sklearn not installed — pip install scikit-learn")
        return None


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_silhouette(scores, save_path):
    labels = list(scores.keys())
    values = list(scores.values())
    colors = [C["pass"] if v >= THRESHOLD else C["warn"] for v in values]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color=colors, width=0.45, alpha=0.88)

    ax.axhline(THRESHOLD, color=C["threshold"], linestyle="--",
               lw=1.5, label=f"Threshold ({THRESHOLD})")

    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{v:.3f}",
                ha="center", fontsize=FONT_TICK + 1, fontweight="bold",
                color=C["pass"] if v >= THRESHOLD else C["warn"])

    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Silhouette Score", fontsize=FONT_AXIS)
    ax.set_title(
        "MLP Feature Space Cluster Quality\n"
        "(Higher = behaviours more separable by time + location + prev action)",
        fontsize=FONT_TITLE, fontweight="bold", pad=10)
    ax.legend(fontsize=FONT_TICK)

    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"[exp7] Saved: {save_path}")


# ── Summary ───────────────────────────────────────────────────────────────────

def save_summary(scores, n_total, save_path):
    lines = [
        "Experiment 7: MLP Prediction Quality (Appendix)",
        f"DB: {DB_BASELINE}  |  Total samples: {n_total}",
        f"Threshold: Silhouette ≥ {THRESHOLD}",
        "",
    ]
    for name, score in scores.items():
        status = "PASS" if score >= THRESHOLD else "FAIL"
        lines.append(f"  {name:<12} score={score:.4f}  [{status}]")

    with open(save_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[exp7] Saved: {save_path}")
    print("\n".join(lines))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    db = MongoClient(MONGO_URI)[DB_BASELINE]
    X, y, users = load_data(db)

    if X is None:
        print("[exp7] No manifold_training_data — run experiment first"); return

    print(f"[exp7] {len(X)} samples  |  feature dim: {X.shape[1]}")

    scores = {}

    # Overall
    s = silhouette(X, y)
    if s is not None:
        scores["Overall"] = s
        print(f"  Overall: {s:.4f}")

    # Per user
    for uid in USERS:
        mask   = np.array([u == uid for u in users])
        if mask.sum() < 10:
            continue
        s = silhouette(X[mask], y[mask])
        if s is not None:
            scores[uid.replace("_", " ")] = s
            print(f"  {uid}: {s:.4f}")

    if not scores:
        print("[exp7] Could not compute silhouette scores"); return

    plot_silhouette(scores, os.path.join(RESULTS_DIR, "exp7_mlp_silhouette.png"))
    save_summary(scores, len(X), os.path.join(RESULTS_DIR, "exp7_summary.txt"))


if __name__ == "__main__":
    main()