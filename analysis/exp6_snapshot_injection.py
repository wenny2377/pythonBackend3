import os, sys, json, requests
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pymongo import MongoClient

from exp_config import (
    BACKEND_URL, MONGO_URI,
    C, FONT_TITLE, FONT_AXIS, FONT_ANNOT, FONT_TICK,
    FIG_DPI, RESULTS_DIR, apply_style
)

apply_style()

# Test queries: mix of present and absent objects
TEST_QUERIES = [
    # Objects that should NOT be in the home → expect "not available" response
    {"query": "Is there any cheese",          "user": "User_Mom", "expect_absent": True},
    {"query": "Where is the cheese",          "user": "User_Mom", "expect_absent": True},
    {"query": "Do we have any pizza",         "user": "User_Dad", "expect_absent": True},
    {"query": "Is there wine",                "user": "User_Mom", "expect_absent": True},
    {"query": "Where is the newspaper",       "user": "User_Dad", "expect_absent": True},
    # Objects that should be present → expect grounded location response
    {"query": "Do we have any cola",          "user": "User_Mom", "expect_absent": False},
    {"query": "Where is the remote",          "user": "User_Dad", "expect_absent": False},
    {"query": "Is there any water",           "user": "User_Mom", "expect_absent": False},
    {"query": "What food do we have",         "user": "User_Dad", "expect_absent": False},
    {"query": "Where can I find a drink",     "user": "User_Mom", "expect_absent": False},
]

HALLUCINATION_KEYWORDS = [
    "usually", "typically", "generally", "refrigerator", "fridge",
    "pantry", "cupboard", "cabinet", "i would suggest", "you can find",
    "most likely", "probably", "should be",
]

ABSENT_KEYWORDS = [
    "don't see", "not available", "not in", "cannot find", "no ",
    "i don't have", "not listed", "not aware", "isn't", "aren't",
    "i'm not sure", "don't know",
]


def query_backend(query, user_id):
    try:
        resp = requests.post(
            f"{BACKEND_URL}/interact/stream",
            json={"query": query, "userID": user_id, "room": ""},
            stream=True, timeout=60,
        )
        resp.raise_for_status()
        answer = ""
        for line in resp.iter_lines():
            if not line:
                continue
            s = line.decode("utf-8") if isinstance(line, bytes) else line
            if not s.startswith("data: "):
                continue
            try:
                ev = json.loads(s[6:])
                if ev.get("type") == "token":
                    answer += ev.get("content", "")
                elif ev.get("type") == "done":
                    break
            except Exception:
                continue
        return answer.strip()
    except Exception as e:
        print(f"  [error] {e}")
        return ""


def is_hallucination(answer, expect_absent):
    lower = answer.lower()
    if expect_absent:
        # Hallucination = saying it exists when it shouldn't
        halluc = any(kw in lower for kw in HALLUCINATION_KEYWORDS)
        # Correct = saying it's not there
        correct = any(kw in lower for kw in ABSENT_KEYWORDS)
        return halluc and not correct
    else:
        # For present objects: hallucination = making up a location
        # Hard to detect without ground truth; mark as non-hallucination
        return False


def backup_and_clear(db):
    docs = list(db.dynamic_objects.find({}))
    db.dynamic_objects_backup.drop()
    if docs:
        db.dynamic_objects_backup.insert_many(docs)
    db.dynamic_objects.delete_many({})
    print(f"  [backup] {len(docs)} objects backed up and cleared")
    return docs


def restore(db):
    docs = list(db.dynamic_objects_backup.find({}))
    db.dynamic_objects.delete_many({})
    if docs:
        for d in docs:
            d.pop("_id", None)
        db.dynamic_objects.insert_many(docs)
    print(f"  [restore] {len(docs)} objects restored")


def run_queries(label):
    results = []
    print(f"\n── {label} ──")
    for i, case in enumerate(TEST_QUERIES):
        answer = query_backend(case["query"], case["user"])
        halluc = is_hallucination(answer, case["expect_absent"])
        results.append({
            **case,
            "answer":        answer,
            "hallucination": halluc,
        })
        h = "HAL" if halluc else "   "
        absent_tag = "(absent)" if case["expect_absent"] else "(present)"
        print(f"  Q{i+1:02d} [{h}] {absent_tag} \"{case['query']}\"")
        if answer:
            print(f"       → {answer[:90]}")
    return results


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_comparison(r_with, r_without, save_path):
    # Only compare absent-object queries (where hallucination is detectable)
    absent_with    = [r for r in r_with    if r["expect_absent"]]
    absent_without = [r for r in r_without if r["expect_absent"]]

    n = len(absent_with)
    halluc_with    = sum(1 for r in absent_with    if r["hallucination"]) / n
    halluc_without = sum(1 for r in absent_without if r["hallucination"]) / n

    labels = ["With Snapshot\n(Injection ON)", "Without Snapshot\n(Injection OFF)"]
    values = [halluc_with * 100, halluc_without * 100]
    colors = [C["pass"], C["corruption"]]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(range(2), values, color=colors, width=0.45,
                  alpha=0.88, edgecolor="white")

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f"{val:.0f}%",
                ha="center", fontsize=FONT_TICK + 2, fontweight="bold")

    # Delta annotation
    delta = values[1] - values[0]
    ax.annotate("", xy=(1, values[1] + 4), xytext=(0, values[0] + 4),
                arrowprops=dict(arrowstyle="<->", color="#555", lw=1.5))
    ax.text(0.5, max(values) + 7, f"Δ = +{delta:.0f}% hallucinations\nwithout snapshot",
            ha="center", fontsize=FONT_ANNOT,
            color=C["corruption"], fontweight="bold")

    ax.set_xticks(range(2))
    ax.set_xticklabels(labels, fontsize=FONT_TICK)
    ax.set_ylabel("Hallucination Rate (%)", fontsize=FONT_AXIS)
    ax.set_ylim(0, 120)
    ax.set_title(
        "Snapshot Injection — Hallucination Prevention\n"
        "(Tested on absent objects, e.g. \"Is there any cheese?\")",
        fontsize=FONT_TITLE, fontweight="bold", pad=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"[exp6] Saved: {save_path}")


# ── Summary ───────────────────────────────────────────────────────────────────

def save_summary(r_with, r_without, save_path):
    absent_with    = [r for r in r_with    if r["expect_absent"]]
    absent_without = [r for r in r_without if r["expect_absent"]]
    n = len(absent_with)

    hw = sum(1 for r in absent_with    if r["hallucination"])
    ho = sum(1 for r in absent_without if r["hallucination"])

    lines = [
        "Experiment 6: Snapshot Injection — Hallucination Prevention",
        f"Absent-object queries: {n}",
        "",
        f"With Snapshot:    {hw}/{n} hallucinations = {hw/n*100:.0f}%",
        f"Without Snapshot: {ho}/{n} hallucinations = {ho/n*100:.0f}%",
        "",
        "Details (absent objects only):",
        f"{'Query':<40} {'With':>6} {'Without':>8}",
        "-" * 58,
    ]
    for rw, ro in zip(absent_with, absent_without):
        w = "HAL" if rw["hallucination"] else "OK "
        o = "HAL" if ro["hallucination"] else "OK "
        lines.append(f"  {rw['query']:<38} {w:>6} {o:>8}")

    lines += ["", "Example (cheese query):"]
    cheese_w = next((r for r in r_with    if "cheese" in r["query"].lower()), None)
    cheese_o = next((r for r in r_without if "cheese" in r["query"].lower()), None)
    if cheese_w:
        lines.append(f"  With:    {cheese_w['answer'][:100]}")
    if cheese_o:
        lines.append(f"  Without: {cheese_o['answer'][:100]}")

    with open(save_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[exp6] Saved: {save_path}")
    print("\n".join(lines))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    try:
        r = requests.get(f"{BACKEND_URL}/ready", timeout=3)
        assert "ready" in r.text
        print("[exp6] Backend connected.")
    except Exception:
        print("[exp6] ERROR: Backend not responding. Start app.py first.")
        return

    from config import Config
    db = MongoClient(MONGO_URI)[Config.DB_NAME]

    # With snapshot (normal operation)
    r_with = run_queries("WITH snapshot injection")

    # Without snapshot (clear dynamic_objects)
    print("\n[Clearing dynamic_objects...]")
    backup_and_clear(db)
    r_without = run_queries("WITHOUT snapshot injection")

    print("\n[Restoring dynamic_objects...]")
    restore(db)

    plot_comparison(r_with, r_without,
                    os.path.join(RESULTS_DIR, "exp6_snapshot_injection.png"))
    save_summary(r_with, r_without,
                 os.path.join(RESULTS_DIR, "exp6_summary.txt"))


if __name__ == "__main__":
    main()