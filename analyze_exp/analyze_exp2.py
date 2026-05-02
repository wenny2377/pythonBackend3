"""
analyze_exp/analyze_exp2.py

Experiment 2: Behavioral Scene Graph Construction & FAISS Semantic Retrieval

Verifies the system correctly builds a Behavioral Scene Graph capturing
user-behavior-furniture-object relationships, and that FAISS semantic
search can navigate the graph using natural language queries.

Graph node types:
    User      (User_Mom, User_Dad)
    Behavior  (Drink, Laying, Reading, Typing, ...)
    Furniture (table, sofa, desk, ...)
    Object    (cup, book, laptop, ...)

Graph edge types:
    performs   User -> Behavior       (from semantic_memories)
    near       Behavior -> Furniture  (from semantic_memories.spatial)
    has_item   Furniture -> Object    (from scene_snapshots)
    in_hand_of Object -> User         (from dynamic_objects, optional)

Queries are generated dynamically from actual MongoDB data.
No hardcoded expected values.

Usage:
    python3 analyze_exp/analyze_exp2.py
    python3 analyze_exp/analyze_exp2.py --out ./results/

Prerequisites:
    Experiment 1 complete; semantic_memories / scene_snapshots populated.

Outputs:
    exp2_graph.png
    exp2_summary.txt
"""

import argparse
import datetime
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pymongo import MongoClient

try:
    import networkx as nx
except ImportError:
    print("Missing: pip install networkx")
    sys.exit(1)

try:
    import faiss
    from sentence_transformers import SentenceTransformer
    HAS_FAISS = True
except ImportError:
    print("faiss / sentence_transformers not available. Query validation skipped.")
    HAS_FAISS = False

MONGO_URI = "mongodb://127.0.0.1:27017/"
DB_NAME   = "robot_rag_db"

NODE_COLORS = {
    "user":      "#7F77DD",
    "behavior":  "#1D9E75",
    "furniture": "#BA7517",
    "object":    "#D85A30",
}
EDGE_COLORS = {
    "performs":   "#534AB7",
    "near":       "#0F6E56",
    "has_item":   "#854F0B",
    "in_hand_of": "#993C1D",
}

LEGACY_MAP = {
    "drink":      "Drink",       "drinking":    "Drink",
    "sit":        "Laying",      "sitting":     "Laying",
    "laying":     "Laying",
    "read":       "Reading",     "reading":     "Reading",
    "type":       "Typing",      "typing":      "Typing",
    "watch":      "Watching",    "watching":    "Watching",
    "sleep":      "Sleeping",    "sleeping":    "Sleeping",
    "walk":       "Walking",     "walking":     "Walking",
    "stand":      "Standing",    "standing":    "Standing",
}


def normalize_action(a: str) -> str:
    if not a:
        return "unknown"
    if a in LEGACY_MAP.values():
        return a
    first = a.lower().strip().split()[0]
    return LEGACY_MAP.get(first, a)


def build_queries_from_db(db) -> list:
    queries = []

    seen_performs = set()
    for m in db.semantic_memories.find(
        {"bound_to": {"$exists": True, "$nin": ["Unknown_Area", "unknown", ""]}},
        {"user": 1, "action": 1, "bound_to": 1},
    ):
        user     = m.get("user", "")
        raw_act  = m.get("action", "")
        bound_to = m.get("bound_to", "")
        norm_act = normalize_action(raw_act)
        if not user or not bound_to:
            continue
        key = (user, norm_act, bound_to)
        if key in seen_performs:
            continue
        seen_performs.add(key)
        short_act = norm_act.lower() if norm_act != "unknown" else raw_act[:20].lower()
        queries.append((
            f"{user} {short_act}",
            bound_to,
            "performs",
            f"{user} -> {norm_act} -> {bound_to}",
        ))
        if sum(1 for q in queries if q[2] == "performs") >= 4:
            break

    seen_near = set()
    for m in db.semantic_memories.find(
        {"bound_to": {"$exists": True, "$nin": ["Unknown_Area", "unknown", ""]}},
        {"action": 1, "bound_to": 1},
    ):
        norm_act = normalize_action(m.get("action", ""))
        bound_to = m.get("bound_to", "")
        key = (norm_act, bound_to)
        if key in seen_near or norm_act == "unknown" or not bound_to:
            continue
        seen_near.add(key)
        queries.append((
            f"{norm_act.lower()} near {bound_to}",
            bound_to,
            "near",
            f"{norm_act} -> {bound_to}",
        ))
        if sum(1 for q in queries if q[2] == "near") >= 4:
            break

    KNOWN_FURNITURE = {
        "table", "table2", "sofa", "desk", "sink", "shelf", "shelf2",
        "tv", "refrigerator", "toilet", "dad's bed", "mom's bed",
        "bed", "chair", "couch", "cabinet", "wardrobe",
    }
    has_item_count = 0
    for snap in db.scene_snapshots.find(
        {"items": {"$exists": True, "$ne": []}},
        {"label": 1, "items": 1},
    ):
        furniture = snap.get("label", "")
        items     = [i for i in snap.get("items", [])
                     if i and i.lower() not in KNOWN_FURNITURE]
        if not furniture or not items:
            continue
        for item in items[:2]:
            queries.append((
                f"{item} on {furniture}",
                item,
                "has_item",
                f"{furniture} -> {item}",
            ))
            has_item_count += 1
            if has_item_count >= 4:
                break
        if has_item_count >= 4:
            break

    print(f"  [QueryGen] {len(queries)} queries from DB")
    print(f"    performs : {sum(1 for q in queries if q[2]=='performs')}")
    print(f"    near     : {sum(1 for q in queries if q[2]=='near')}")
    print(f"    has_item : {sum(1 for q in queries if q[2]=='has_item')}")
    for q in queries:
        print(f"    {q[2]:10s}  '{q[0][:40]}' -> '{q[1]}'")
    return queries


def build_graph(db) -> "nx.DiGraph":
    G = nx.DiGraph()

    memories = list(db.semantic_memories.find(
        {},
        {"user": 1, "action": 1, "bound_to": 1,
         "spatial_relations": 1, "interacting_items": 1},
    ))
    print(f"  semantic_memories: {len(memories)}")

    for m in memories:
        user     = m.get("user", "")
        action   = normalize_action(m.get("action", ""))
        bound_to = m.get("bound_to", "")
        if not user or not action:
            continue
        G.add_node(user,   node_type="user")
        G.add_node(action, node_type="behavior")
        if G.has_edge(user, action):
            G[user][action]["weight"] += 1
        else:
            G.add_edge(user, action, rel="performs", weight=1)
        if bound_to and "unknown" not in bound_to.lower():
            G.add_node(bound_to, node_type="furniture")
            if G.has_edge(action, bound_to):
                G[action][bound_to]["weight"] += 1
            else:
                G.add_edge(action, bound_to, rel="near", weight=1)
        for item in m.get("interacting_items", []):
            item = item.lower().strip()
            if not item:
                continue
            G.add_node(item, node_type="object")
            if bound_to and "unknown" not in bound_to.lower():
                if not G.has_edge(bound_to, item):
                    G.add_edge(bound_to, item, rel="has_item", weight=1)
                else:
                    G[bound_to][item]["weight"] += 1

    snapshots = list(db.scene_snapshots.find({}, {"label": 1, "items": 1}))
    print(f"  scene_snapshots: {len(snapshots)}")
    for snap in snapshots:
        furniture = snap.get("label", "").lower().strip()
        if not furniture:
            continue
        G.add_node(furniture, node_type="furniture")
        for item in snap.get("items", []):
            item = item.lower().strip()
            if not item:
                continue
            G.add_node(item, node_type="object")
            if not G.has_edge(furniture, item):
                G.add_edge(furniture, item, rel="has_item", weight=1)

    dyn_objs = list(db.dynamic_objects.find(
        {"interacted_by": {"$exists": True, "$ne": []}},
        {"label": 1, "interacted_by": 1},
    ))
    if dyn_objs:
        print(f"  dynamic_objects (in_hand_of): {len(dyn_objs)}")
        for obj in dyn_objs:
            item = obj.get("label", "").lower().strip()
            if not item:
                continue
            G.add_node(item, node_type="object")
            for uid in obj.get("interacted_by", []):
                G.add_node(uid, node_type="user")
                if not G.has_edge(item, uid):
                    G.add_edge(item, uid, rel="in_hand_of", weight=1)
    else:
        print("  dynamic_objects (in_hand_of): 0 records")

    obs_logs = list(db.observation_logs.find(
        {}, {"user": 1, "action": 1, "weight": 1}))
    print(f"  observation_logs: {len(obs_logs)}")
    for obs in obs_logs:
        user   = obs.get("user", "")
        action = normalize_action(obs.get("action", ""))
        w      = obs.get("weight", 1)
        if G.has_edge(user, action):
            G[user][action]["weight"] = max(G[user][action]["weight"], w)

    print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def run_query_validation(db) -> list:
    if not HAS_FAISS:
        return []

    print("  Loading SBERT model...")
    model  = SentenceTransformer("paraphrase-MiniLM-L6-v2")
    texts  = []
    labels = []

    for m in db.semantic_memories.find(
        {},
        {"user": 1, "action": 1, "bound_to": 1, "interacting_items": 1},
    ):
        user   = m.get("user", "")
        action = normalize_action(m.get("action", ""))
        bound  = m.get("bound_to", "")
        items  = m.get("interacting_items", [])
        text   = (f"{user} {action.lower()} near {bound} "
                  f"with {', '.join(items) or 'nothing'}.")
        texts.append(text)
        labels.append((bound or action, "furniture" if bound else "behavior"))

    for obj in db.dynamic_objects.find(
        {}, {"label": 1, "room": 1, "last_seen_on": 1}
    ):
        item = obj.get("label", "")
        room = obj.get("room", "")
        on   = obj.get("last_seen_on", "")
        texts.append(f"{item} located in {room} on {on}.")
        labels.append((item, "object"))

    if not texts:
        print("  No data for FAISS index")
        return []

    print(f"  Building FAISS index ({len(texts)} documents)...")
    vecs  = model.encode(texts, normalize_embeddings=True).astype("float32")
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)

    queries = build_queries_from_db(db)
    if not queries:
        print("  No queries generated")
        return []

    results = []
    for query_text, expected, edge_type, desc in queries:
        q_vec        = model.encode(
            [query_text], normalize_embeddings=True).astype("float32")
        scores, idxs = index.search(q_vec, 5)

        top_hits = [
            (labels[i][0], labels[i][1], float(scores[0][k]))
            for k, i in enumerate(idxs[0])
            if i < len(labels)
        ]

        hit_rank = None
        hit_sim  = None
        for rank, (node, ntype, sim) in enumerate(top_hits):
            if (expected.lower() in node.lower() or
                    node.lower() in expected.lower()):
                hit_rank = rank + 1
                hit_sim  = sim
                break

        results.append({
            "query":       query_text,
            "expected":    expected,
            "edge_type":   edge_type,
            "description": desc,
            "top_hits":    top_hits[:3],
            "hit_rank":    hit_rank,
            "hit_sim":     hit_sim,
            "correct":     hit_rank is not None and hit_rank <= 3,
        })

        status = (f"rank={hit_rank} sim={hit_sim:.3f}"
                  if hit_rank else "MISS")
        ok = "v" if hit_rank else "x"
        print(f"    {ok}  {query_text:40s} -> {expected:15s} [{status}]")

    return results


def plot(G: "nx.DiGraph", query_results: list, out_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(22, 9))
    fig.suptitle(
        "Experiment 2: Behavioral Scene Graph & FAISS Semantic Retrieval",
        fontsize=13, fontweight="bold",
    )

    ax1 = axes[0]
    ax1.set_title("Behavioral Scene Graph", fontsize=12)

    if G.number_of_nodes() > 0:
        pos        = {}
        layers     = {"user": 0, "behavior": 1, "furniture": 2, "object": 3}
        layer_nodes = defaultdict(list)
        for n, d in G.nodes(data=True):
            layer_nodes[d.get("node_type", "object")].append(n)
        for nt, y in layers.items():
            nodes = layer_nodes[nt]
            for i, n in enumerate(nodes):
                x = (i - (len(nodes) - 1) / 2) * 1.8
                pos[n] = (x, -y * 2.5)
        for rel, color in EDGE_COLORS.items():
            edges = [(u, v) for u, v, d in G.edges(data=True)
                     if d.get("rel") == rel]
            if edges:
                nx.draw_networkx_edges(
                    G, pos, edgelist=edges, ax=ax1,
                    edge_color=color, arrows=True,
                    arrowsize=15, width=1.2, alpha=0.7,
                    connectionstyle="arc3,rad=0.1", node_size=800,
                )
        for nt, color in NODE_COLORS.items():
            nodes = [n for n, d in G.nodes(data=True)
                     if d.get("node_type") == nt]
            if nodes:
                nx.draw_networkx_nodes(
                    G, pos, nodelist=nodes, ax=ax1,
                    node_color=color, node_size=800, alpha=0.9,
                )
        nx.draw_networkx_labels(G, pos, ax=ax1,
                                font_size=8, font_color="white",
                                font_weight="bold")
    else:
        ax1.text(0.5, 0.5, "No graph data\n(Run Experiment 1 first)",
                 ha="center", va="center",
                 transform=ax1.transAxes, fontsize=11)

    patches    = [mpatches.Patch(color=c, label=t.capitalize())
                  for t, c in NODE_COLORS.items()]
    edge_lines = [plt.Line2D([0], [0], color=c, linewidth=2, label=r)
                  for r, c in EDGE_COLORS.items()]
    ax1.legend(handles=patches + edge_lines,
               loc="lower left", fontsize=8, framealpha=0.8)
    ax1.axis("off")

    ax2 = axes[1]
    ax2.set_title(
        "Query Validation Results\n(queries generated from actual DB data)",
        fontsize=11,
    )
    ax2.axis("off")

    if query_results:
        col_labels = ["Query", "Expected", "Type", "Rank", "Sim", "OK"]
        table_data = []
        for r in query_results:
            sim_str  = f"{r['hit_sim']:.3f}" if r["hit_sim"] else "-"
            rank_str = str(r["hit_rank"]) if r["hit_rank"] else "-"
            ok_str   = "v" if r["correct"] else "x"
            table_data.append([
                r["query"][:28], r["expected"][:14], r["edge_type"],
                rank_str, sim_str, ok_str,
            ])
        tbl = ax2.table(
            cellText=table_data, colLabels=col_labels,
            loc="center", cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8.5)
        tbl.scale(1, 1.5)
        for i, r in enumerate(query_results):
            color = "#E1F5EE" if r["correct"] else "#FCEBEB"
            for j in range(len(col_labels)):
                tbl[i + 1, j].set_facecolor(color)
    else:
        ax2.text(0.5, 0.5, "FAISS not available or no data",
                 ha="center", va="center",
                 transform=ax2.transAxes, fontsize=11)

    ax3 = axes[2]
    ax3.set_title("Retrieval Accuracy by Relation Type", fontsize=12)

    eval_types = ["performs", "near", "has_item"]
    if query_results:
        accs   = []
        ns     = []
        colors = []
        for et in eval_types:
            subset = [r for r in query_results if r["edge_type"] == et]
            n      = len(subset)
            acc    = sum(1 for r in subset if r["correct"]) / n if n > 0 else 0
            accs.append(acc)
            ns.append(n)
            colors.append(EDGE_COLORS[et])
        x    = np.arange(len(eval_types))
        bars = ax3.bar(x, accs, color=colors, alpha=0.85,
                       edgecolor="white", linewidth=0.8)
        for bar, acc, n in zip(bars, accs, ns):
            ax3.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.02,
                     f"{acc:.0%}\n(n={n})",
                     ha="center", va="bottom",
                     fontsize=9, fontweight="bold")
        total       = len(query_results)
        overall_acc = sum(1 for r in query_results if r["correct"]) / total
        ax3.axhline(y=overall_acc, color="#2C2C2A", linewidth=1.2,
                    linestyle="--", alpha=0.6,
                    label=f"Overall = {overall_acc:.0%}")
        ax3.set_xticks(x)
        ax3.set_xticklabels(eval_types, fontsize=10)
        ax3.set_ylabel("Top-3 Recall", fontsize=12)
        ax3.set_ylim(0, 1.25)
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.2, axis="y")
    else:
        ax3.text(0.5, 0.5, "No query results",
                 ha="center", va="center",
                 transform=ax3.transAxes, fontsize=11)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out_path}")


def save_summary(G: "nx.DiGraph", query_results: list, out_path: str):
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    n_users = sum(1 for _, d in G.nodes(data=True) if d.get("node_type") == "user")
    n_behav = sum(1 for _, d in G.nodes(data=True) if d.get("node_type") == "behavior")
    n_furn  = sum(1 for _, d in G.nodes(data=True) if d.get("node_type") == "furniture")
    n_obj   = sum(1 for _, d in G.nodes(data=True) if d.get("node_type") == "object")

    edge_counts = defaultdict(int)
    for _, _, d in G.edges(data=True):
        edge_counts[d.get("rel", "unknown")] += 1

    overall_acc = 0.0
    type_accs   = {}
    if query_results:
        overall_acc = (sum(1 for r in query_results if r["correct"])
                       / len(query_results))
        for et in ["performs", "near", "has_item"]:
            subset = [r for r in query_results if r["edge_type"] == et]
            if subset:
                type_accs[et] = (
                    sum(1 for r in subset if r["correct"]) / len(subset))

    lines = [
        "=" * 65,
        "Experiment 2: Behavioral Scene Graph Construction",
        f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 65,
        "",
        "Graph Statistics:",
        f"  Total nodes : {n_nodes}",
        f"  Users       : {n_users}",
        f"  Behaviors   : {n_behav}",
        f"  Furniture   : {n_furn}",
        f"  Objects     : {n_obj}",
        f"  Total edges : {n_edges}",
        *[f"  {rel:12s}: {cnt}" for rel, cnt in edge_counts.items()],
        "",
        "Query Validation (dynamic from actual DB data):",
        f"  Total queries       : {len(query_results)}",
        f"  Overall Top-3 Recall: {overall_acc:.0%}",
        *[f"  {et:12s}: {acc:.0%}" for et, acc in type_accs.items()],
        "",
        "Per-query results:",
        *[
            f"  {'v' if r['correct'] else 'x'}  "
            f"{r['query']:40s} -> {r['expected']:15s} "
            f"[rank={r['hit_rank'] or '-'}, "
            f"sim={r['hit_sim'] if r['hit_sim'] is not None else 0.0:.3f}]"
            for r in query_results
        ],
        "",
        "For thesis:",
        f"The system constructed a Behavioral Scene Graph comprising {n_nodes} nodes",
        f"({n_users} users, {n_behav} behaviors, {n_furn} furniture, {n_obj} objects)",
        f"and {n_edges} typed relational edges.",
        f"FAISS semantic retrieval evaluated on {len(query_results)} queries",
        f"achieved an overall Top-3 Recall of {overall_acc:.0%}.",
    ]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  Saved: {out_path}")
    print(f"\n  Graph  : {n_nodes} nodes, {n_edges} edges")
    print(f"  Recall : {overall_acc:.0%}  "
          f"({sum(1 for r in query_results if r['correct'])}/{len(query_results)})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=".", help="Output directory")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print(f"Connecting to MongoDB ({DB_NAME})...")
    client = MongoClient(MONGO_URI)
    db     = client[DB_NAME]

    print("\nStep 1: Building Behavioral Scene Graph...")
    G = build_graph(db)
    if G.number_of_nodes() < 3:
        print("  Graph has fewer than 3 nodes. Run Experiment 1 first.")

    print("\nStep 2: Running FAISS query validation...")
    query_results = run_query_validation(db)

    print("\nStep 3: Generating outputs...")
    plot(G, query_results,
         out_path=os.path.join(args.out, "exp2_graph.png"))
    save_summary(G, query_results,
                 out_path=os.path.join(args.out, "exp2_summary.txt"))


if __name__ == "__main__":
    main()