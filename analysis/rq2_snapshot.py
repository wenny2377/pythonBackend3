import os, sys, json, requests
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pymongo import MongoClient

BACKEND   = "http://127.0.0.1:5000"
MONGO_URI = "mongodb://127.0.0.1:27017/"
OUT       = os.path.join(_ROOT, "analysis", "results")

TEST_QUERIES = [
    {"query": "Where is the remote",         "user": "User_Mom", "target_object": "remote"},
    {"query": "Do we have any cola",         "user": "User_Dad", "target_object": "cola"},
    {"query": "Where is the water bottle",   "user": "User_Mom", "target_object": "water"},
    {"query": "Is there any food",           "user": "User_Mom", "target_object": "food"},
    {"query": "Where is the phone",          "user": "User_Dad", "target_object": "phone"},
    {"query": "Do we have juice",            "user": "User_Mom", "target_object": "juice"},
    {"query": "Where is the book",           "user": "User_Mom", "target_object": "book"},
    {"query": "What is on the table",        "user": "User_Dad", "target_object": None},
    {"query": "Is the broom available",      "user": "User_Mom", "target_object": "broom"},
    {"query": "Where can I find something to eat", "user": "User_Dad", "target_object": "food"},
]

HALLUCINATION_KEYWORDS = [
    "i don't know", "i'm not sure", "not available", "cannot find",
    "no information", "unable to", "not in the snapshot",
    "i don't have", "not aware", "not listed",
]

GROUNDED_KEYWORDS = [
    "at", "in the", "on the", "near", "located", "found",
    "kitchen", "living room", "dad room", "table", "sofa",
]

def query_backend(query, user_id):
    try:
        resp = requests.post(
            f"{BACKEND}/interact/stream",
            json={"query": query, "userID": user_id, "room": ""},
            stream=True, timeout=60,
        )
        resp.raise_for_status()
        full_answer = ""
        nav_label   = None
        for line in resp.iter_lines():
            if not line:
                continue
            line_str = line.decode("utf-8") if isinstance(line, bytes) else line
            if not line_str.startswith("data: "):
                continue
            try:
                event = json.loads(line_str[6:])
                if event.get("type") == "token":
                    full_answer += event.get("content", "")
                elif event.get("type") == "done":
                    nav_label = event.get("nav_label")
            except Exception:
                continue
        return full_answer.strip(), nav_label
    except Exception as e:
        print(f"  [error] {e}")
        return "", None

def is_hallucination(answer):
    lower = answer.lower()
    return any(kw in lower for kw in HALLUCINATION_KEYWORDS)

def is_grounded(answer, nav_label):
    lower = answer.lower()
    if nav_label:
        return True
    return any(kw in lower for kw in GROUNDED_KEYWORDS)

def backup_dynamic_objects(db):
    docs = list(db.dynamic_objects.find({}))
    db.dynamic_objects_backup.drop()
    if docs:
        db.dynamic_objects_backup.insert_many(docs)
    print(f"  [backup] {len(docs)} objects backed up")
    return docs

def clear_dynamic_objects(db):
    db.dynamic_objects.delete_many({})
    print("  [clear] dynamic_objects cleared")

def restore_dynamic_objects(db):
    docs = list(db.dynamic_objects_backup.find({}))
    db.dynamic_objects.delete_many({})
    if docs:
        for d in docs:
            d.pop("_id", None)
        db.dynamic_objects.insert_many(docs)
    print(f"  [restore] {len(docs)} objects restored")

def plot_comparison(results_with, results_without, save_path):
    n          = len(results_with)
    labels     = [f"Q{i+1}" for i in range(n)]
    grounded_w = [1 if r["grounded"] else 0 for r in results_with]
    grounded_wo= [1 if r["grounded"] else 0 for r in results_without]
    halluc_w   = [1 if r["hallucination"] else 0 for r in results_with]
    halluc_wo  = [1 if r["hallucination"] else 0 for r in results_without]

    x     = range(n)
    width = 0.35
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.bar([i - width/2 for i in x], grounded_w,  width, label="With Snapshot",    color="#4CAF50", alpha=0.8)
    ax1.bar([i + width/2 for i in x], grounded_wo, width, label="Without Snapshot",  color="#FF9800", alpha=0.8)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Grounded (1=Yes)")
    ax1.set_title("RQ2 — Grounded Responses")
    ax1.legend()
    ax1.set_ylim(0, 1.3)

    ax2.bar([i - width/2 for i in x], halluc_w,  width, label="With Snapshot",   color="#2196F3", alpha=0.8)
    ax2.bar([i + width/2 for i in x], halluc_wo, width, label="Without Snapshot", color="#F44336", alpha=0.8)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Hallucination (1=Yes)")
    ax2.set_title("RQ2 — Hallucination Rate")
    ax2.legend()
    ax2.set_ylim(0, 1.3)

    plt.suptitle("RQ2 — Snapshot Injection Effect", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[RQ2] Comparison plot saved → {save_path}")

def plot_summary_bar(grounded_with, grounded_without,
                     halluc_with, halluc_without, save_path):
    categories = ["Grounded Rate\n(With Snapshot)",
                  "Grounded Rate\n(Without Snapshot)",
                  "Hallucination Rate\n(With Snapshot)",
                  "Hallucination Rate\n(Without Snapshot)"]
    values  = [grounded_with * 100, grounded_without * 100,
               halluc_with * 100,   halluc_without * 100]
    colors  = ["#4CAF50", "#FF9800", "#2196F3", "#F44336"]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars    = ax.bar(categories, values, color=colors, width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1, f"{val:.1f}%",
                ha="center", va="bottom", fontsize=11)
    ax.set_ylim(0, 120)
    ax.set_ylabel("Rate (%)")
    ax.set_title("RQ2 — Snapshot Injection Summary")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[RQ2] Summary bar saved → {save_path}")

def run_queries(label):
    results = []
    print(f"\n[Running] {label}")
    for i, case in enumerate(TEST_QUERIES):
        query   = case["query"]
        user_id = case["user"]
        answer, nav_label = query_backend(query, user_id)
        halluc  = is_hallucination(answer)
        ground  = is_grounded(answer, nav_label)
        results.append({
            "query":         query,
            "user":          user_id,
            "answer":        answer,
            "nav_label":     nav_label,
            "hallucination": halluc,
            "grounded":      ground,
        })
        h_mark = "HAL" if halluc  else "   "
        g_mark = "GRD" if ground  else "   "
        print(f"  Q{i+1:02d} [{h_mark}][{g_mark}] [{user_id}] \"{query}\"")
        if answer:
            print(f"       → {answer[:80]}")
    return results

def main():
    os.makedirs(OUT, exist_ok=True)
    print("=" * 60)
    print("RQ2 Snapshot Injection — Hallucination Prevention")
    print("=" * 60)

    try:
        r = requests.get(f"{BACKEND}/ready", timeout=3)
        if "ready" not in r.text:
            raise Exception
        print("[OK] Backend connected.\n")
    except Exception:
        print("[ERROR] Backend not responding. Start app.py first.")
        return

    from config import Config
    db = MongoClient(MONGO_URI)[Config.DB_NAME]

    results_with = run_queries("WITH snapshot (normal)")

    print("\n[Clearing dynamic_objects for WITHOUT snapshot test...]")
    backup_dynamic_objects(db)
    clear_dynamic_objects(db)

    results_without = run_queries("WITHOUT snapshot (no dynamic_objects)")

    print("\n[Restoring dynamic_objects...]")
    restore_dynamic_objects(db)

    n = len(TEST_QUERIES)
    grounded_with    = sum(1 for r in results_with    if r["grounded"])     / n
    grounded_without = sum(1 for r in results_without if r["grounded"])     / n
    halluc_with      = sum(1 for r in results_with    if r["hallucination"])/ n
    halluc_without   = sum(1 for r in results_without if r["hallucination"])/ n

    print(f"\n{'='*60}")
    print(f"{'Metric':<30} {'With Snapshot':>15} {'Without Snapshot':>18}")
    print(f"{'-'*60}")
    print(f"{'Grounded Rate':<30} {grounded_with*100:>14.1f}% {grounded_without*100:>17.1f}%")
    print(f"{'Hallucination Rate':<30} {halluc_with*100:>14.1f}% {halluc_without*100:>17.1f}%")
    print(f"{'='*60}")

    delta_halluc  = halluc_without  - halluc_with
    delta_ground  = grounded_with   - grounded_without
    status = "PASSED" if delta_halluc > 0.20 else "MARGINAL"
    print(f"\nHallucination reduction: {delta_halluc*100:.1f}%")
    print(f"Grounding improvement:   {delta_ground*100:.1f}%")
    print(f"[Result] {status}")

    plot_comparison(results_with, results_without,
                    os.path.join(OUT, "rq2_comparison.png"))
    plot_summary_bar(grounded_with, grounded_without,
                     halluc_with, halluc_without,
                     os.path.join(OUT, "rq2_summary.png"))

    summary_path = os.path.join(OUT, "rq2_snapshot_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"RQ2 Snapshot Injection\n")
        f.write(f"Total queries: {n}\n\n")
        f.write(f"With Snapshot:\n")
        f.write(f"  Grounded:      {grounded_with*100:.1f}%\n")
        f.write(f"  Hallucination: {halluc_with*100:.1f}%\n\n")
        f.write(f"Without Snapshot:\n")
        f.write(f"  Grounded:      {grounded_without*100:.1f}%\n")
        f.write(f"  Hallucination: {halluc_without*100:.1f}%\n\n")
        f.write(f"Delta Hallucination: {delta_halluc*100:.1f}%\n")
        f.write(f"Status: {status}\n\n")
        f.write("Details (With Snapshot):\n")
        for r in results_with:
            h = "HAL" if r["hallucination"] else "   "
            g = "GRD" if r["grounded"]      else "   "
            f.write(f"  [{h}][{g}] {r['query']}\n")
            f.write(f"         → {r['answer'][:100]}\n")
        f.write("\nDetails (Without Snapshot):\n")
        for r in results_without:
            h = "HAL" if r["hallucination"] else "   "
            g = "GRD" if r["grounded"]      else "   "
            f.write(f"  [{h}][{g}] {r['query']}\n")
            f.write(f"         → {r['answer'][:100]}\n")
    print(f"[RQ2] Summary saved → {summary_path}")

if __name__ == "__main__":
    main()