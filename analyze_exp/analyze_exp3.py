"""
analyze_exp/analyze_exp3.py  (shadow-tracking aware version)

Experiment 3: Behavioral Manifold Learning

Two analysis passes:
  Pass A — VLM points only   (is_shadow=False, confidence=high/medium)
  Pass B — All points        (VLM + shadow tracking)

Outputs:
  exp3_umap_vlm.png       Pass A scatter
  exp3_umap_all.png       Pass B scatter (3-panel)
  exp3_summary.txt        Silhouette scores + thesis text

Usage:
    python3 analyze_exp/analyze_exp3.py
    python3 analyze_exp/analyze_exp3.py --out ./results/
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
    "drinking": "Drink",    "drink":    "Drink",
    "sit":      "Laying",   "sitting":  "Laying",
    "laying":   "Laying",
    "reading":  "Reading",  "read":     "Reading",
    "typing":   "Typing",   "type":     "Typing",
    "watching": "Watching", "watch":    "Watching",
    "sleeping": "Sleeping", "sleep":    "Sleeping",
    "eating":   "Eating",   "eat":      "Eating",
    "walking":  "Walking",  "walk":     "Walking",
    "standing": "Standing", "stand":    "Standing",
    "phoneuse": "PhoneUse", "phone":    "PhoneUse",
}

BEHAVIOR_COLORS = {
    "Drink":    "#059669", "Laying":   "#F59E0B",
    "Reading":  "#2563EB", "Typing":   "#7C3AED",
    "Watching": "#DC2626", "Sleeping": "#6366F1",
    "Eating":   "#EC4899", "Walking":  "#84CC16",
    "Standing": "#6B7280", "PhoneUse": "#14B8A6",
    "Unknown":  "#D1D5DB",
}
SLOT_COLORS   = {
    "Morning":   "#F59E0B", "Noon":      "#EF4444",
    "Afternoon": "#2563EB", "Evening":   "#059669",
    "Unknown":   "#9CA3AF",
}
SLOT_MARKERS  = {
    "Morning": "o", "Noon": "s",
    "Afternoon": "^", "Evening": "D", "Unknown": "x",
}


def normalize_action(a):
    if not a:
        return "Unknown"
    s = a.lower().strip()
    if s in NORMALIZE_MAP:
        return NORMALIZE_MAP[s]
    for kw, label in NORMALIZE_MAP.items():
        if kw in s:
            return label
    return a.capitalize()


def get_time_slot(vh):
    if vh is None:
        return "Unknown"
    h = float(vh)
    if h < 10:  return "Morning"
    if h < 13:  return "Noon"
    if h < 18:  return "Afternoon"
    return "Evening"


def load_data(db):
    docs = list(db.manifold_points.find(
        {"feature_vec": {"$exists": True}},
        {
            "feature_vec":  1,
            "action":       1,
            "user_id":      1,
            "virtual_hour": 1,
            "prev_action":  1,
            "confidence":   1,
            "is_shadow":    1,
            "timestamp":    1,
        },
    ))
    print(f"  manifold_points total: {len(docs)}")
    if not docs:
        return None

    records = []
    for d in docs:
        records.append({
            "feature_vec":  d["feature_vec"],
            "action":       normalize_action(d.get("action", "")),
            "user_id":      d.get("user_id", "unknown"),
            "slot":         get_time_slot(d.get("virtual_hour")),
            "prev_action":  normalize_action(d.get("prev_action", "")),
            "confidence":   d.get("confidence", "unknown"),
            "is_shadow":    d.get("is_shadow", False),
        })

    shadow_n = sum(1 for r in records if r["is_shadow"])
    vlm_n    = len(records) - shadow_n
    print(f"  VLM points   : {vlm_n}")
    print(f"  Shadow points: {shadow_n}")
    return records


def split_records(records):
    vlm_recs = [r for r in records if not r["is_shadow"]]
    all_recs = records
    return vlm_recs, all_recs


def run_umap_hdbscan(records, label=""):
    n = len(records)
    if n < 10:
        print(f"  [{label}] Not enough points ({n}), skipping.")
        return None, None, None, None

    X = np.array([r["feature_vec"] for r in records],
                 dtype=np.float32)
    print(f"  [{label}] StandardScaler on {n} points...")
    X_scaled = StandardScaler().fit_transform(X)

    print(f"  [{label}] UMAP (n_neighbors={min(15, n-1)})...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=min(15, n - 1),
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    embedding = reducer.fit_transform(X_scaled)

    min_cs = max(3, n // 10)
    print(f"  [{label}] HDBSCAN (min_cluster_size={min_cs})...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cs,
        prediction_data=True,
    )
    labels     = clusterer.fit_predict(embedding)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"  [{label}] Clusters: {n_clusters}")

    s_score = None
    valid   = labels != -1
    if valid.sum() > 1 and n_clusters > 1:
        s_score = silhouette_score(
            embedding[valid], labels[valid])
        print(f"  [{label}] Silhouette S = {s_score:.4f}")
    else:
        print(f"  [{label}] Silhouette N/A")

    return embedding, labels, n_clusters, s_score


def plot_pass(records, embedding, labels, n_clusters,
              s_score, out_path, title_prefix):
    actions = [r["action"] for r in records]
    slots   = [r["slot"]   for r in records]
    n_total = len(records)

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    suf = (f"  S={s_score:.4f}" if s_score else "  S=N/A")
    fig.suptitle(
        f"{title_prefix} — UMAP Projection  "
        f"(n={n_total}){suf}",
        fontsize=13, fontweight="bold",
    )

    # Panel 1: color by behavior
    ax = axes[0]
    for b in [b for b in BEHAVIOR_COLORS
              if any(a == b for a in actions)]:
        mask = np.array([a == b for a in actions])
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=BEHAVIOR_COLORS[b],
            label=f"{b} (n={mask.sum()})",
            alpha=0.75, s=40,
            edgecolors="white", linewidths=0.3,
        )
    ax.set_title("Color by behavior", fontsize=11)
    ax.set_xlabel("UMAP dim 1")
    ax.set_ylabel("UMAP dim 2")
    ax.legend(fontsize=7, framealpha=0.8)
    ax.grid(True, alpha=0.2)

    # Panel 2: color by time slot
    ax = axes[1]
    for s in [s for s in SLOT_COLORS
              if any(sl == s for sl in slots)]:
        mask = np.array([sl == s for sl in slots])
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=SLOT_COLORS[s],
            marker=SLOT_MARKERS[s],
            label=f"{s} (n={mask.sum()})",
            alpha=0.75, s=40,
            edgecolors="white", linewidths=0.3,
        )
    ax.set_title("Color by time slot", fontsize=11)
    ax.set_xlabel("UMAP dim 1")
    ax.set_ylabel("UMAP dim 2")
    ax.legend(fontsize=7, framealpha=0.8)
    ax.grid(True, alpha=0.2)

    # Panel 3: HDBSCAN clusters
    ax = axes[2]
    cmap          = plt.cm.get_cmap("tab10", max(n_clusters, 1))
    unique_labels = sorted(set(labels))
    for cid in unique_labels:
        mask  = labels == cid
        color = "#CCCCCC" if cid == -1 else cmap(cid % 10)
        lbl   = ("Noise" if cid == -1
                 else f"Cluster {cid} (n={mask.sum()})")
        ax.scatter(
            embedding[mask, 0], embedding[mask, 1],
            c=[color] * mask.sum(), label=lbl,
            alpha=0.7 if cid >= 0 else 0.3,
            s=40 if cid >= 0 else 20,
            edgecolors="white", linewidths=0.3,
        )
    for cid in unique_labels:
        if cid == -1:
            continue
        mask     = labels == cid
        pts      = embedding[mask]
        cx, cy   = pts[:, 0].mean(), pts[:, 1].mean()
        acts     = [actions[i] for i in range(n_total)
                    if labels[i] == cid]
        dominant = (Counter(acts).most_common(1)[0][0]
                    if acts else "?")
        ax.annotate(
            dominant, (cx, cy),
            fontsize=8, fontweight="bold",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.2",
                      fc="white", alpha=0.7, ec="none"),
        )
    s_str = (f"S={s_score:.4f}" if s_score else "S=N/A")
    ax.set_title(
        f"HDBSCAN clusters ({n_clusters} found, {s_str})",
        fontsize=11)
    ax.set_xlabel("UMAP dim 1")
    ax.set_ylabel("UMAP dim 2")
    ax.legend(fontsize=7, framealpha=0.8)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def save_summary(vlm_res, all_res, n_vlm, n_all, out_path):
    def fmt(res):
        if res[3] is not None:
            passed = res[3] >= 0.5
            status = "PASSED (S>=0.50)" if passed \
                     else "BELOW THRESHOLD"
            return (f"n={res[0]}  clusters={res[1]}  "
                    f"S={res[3]:.4f}  {status}")
        return f"n={res[0]}  clusters={res[1]}  S=N/A"

    lines = [
        "=" * 65,
        "Experiment 3: Behavioral Manifold Learning",
        f"Generated: "
        f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 65,
        "",
        "Pass A — VLM points only (is_shadow=False):",
        f"  {fmt((n_vlm,) + vlm_res[2:])}",
        "",
        "Pass B — All points (VLM + shadow tracking):",
        f"  {fmt((n_all,) + all_res[2:])}",
        "",
        "Interpretation:",
        "  Pass A: baseline clustering from behaviour-at-spot.",
        "  Pass B: adds trajectory points; band structure shows",
        "          system can see intent forming during movement.",
        "",
        "For thesis:",
        f"  VLM-only S = "
        f"{vlm_res[3]:.4f if vlm_res[3] else 'N/A'}",
        f"  Full S     = "
        f"{all_res[3]:.4f if all_res[3] else 'N/A'}",
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Summary: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=".", help="Output dir")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print(f"Connecting to MongoDB ({DB_NAME})...")
    client  = MongoClient(MONGO_URI)
    db      = client[DB_NAME]

    print("\nStep 1: Loading data...")
    records = load_data(db)
    if not records or len(records) < 10:
        print("Not enough data. Run Experiment3 first.")
        sys.exit(1)

    vlm_recs, all_recs = split_records(records)

    print("\nStep 2a: Pass A — VLM points only...")
    vlm_emb, vlm_labels, vlm_nc, vlm_s = \
        run_umap_hdbscan(vlm_recs, "VLM-only")

    print("\nStep 2b: Pass B — All points...")
    all_emb, all_labels, all_nc, all_s = \
        run_umap_hdbscan(all_recs, "All")

    print("\nStep 3: Generating plots...")

    if vlm_emb is not None:
        plot_pass(
            vlm_recs, vlm_emb, vlm_labels, vlm_nc, vlm_s,
            out_path=os.path.join(args.out, "exp3_umap_vlm.png"),
            title_prefix="Exp3 Pass A — VLM points only",
        )

    if all_emb is not None:
        plot_pass(
            all_recs, all_emb, all_labels, all_nc, all_s,
            out_path=os.path.join(args.out, "exp3_umap_all.png"),
            title_prefix="Exp3 Pass B — All points (VLM + shadow)",
        )

    save_summary(
        (vlm_emb, vlm_labels, vlm_nc, vlm_s),
        (all_emb, all_labels, all_nc, all_s),
        len(vlm_recs), len(all_recs),
        out_path=os.path.join(args.out, "exp3_summary.txt"),
    )


if __name__ == "__main__":
    main()