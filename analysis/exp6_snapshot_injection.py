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

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:8b"

PRESENT_QUERIES = [
    {"query": "Where is the remote control",  "user": "User_Dad"},
    {"query": "Where is my cola",             "user": "User_Mom"},
    {"query": "Where can I find water",       "user": "User_Mom"},
    {"query": "Where is the phone",           "user": "User_Dad"},
    {"query": "Where is the book",            "user": "User_Mom"},
]

ABSENT_QUERIES = [
    {"query": "Where did I put the cheese",       "user": "User_Mom"},
    {"query": "Where is the wine",                "user": "User_Mom"},
    {"query": "Where is the newspaper",           "user": "User_Dad"},
    {"query": "Where do we keep the pizza",       "user": "User_Dad"},
    {"query": "Where is my medicine bottle",      "user": "User_Mom"},
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


def is_hallucination_llm(query, answer):
    prompt = f"""You are evaluating whether an AI assistant hallucinated.

The user asked: "{query}"
IMPORTANT: The object being asked about does NOT exist in this home.
The assistant replied: "{answer}"

Did the assistant hallucinate by claiming the object exists or giving a specific location for it?

Answer with only one word: YES or NO

YES = assistant claimed the object exists, gave a location, or made assumptions about where it might be
NO = assistant correctly said it does not have information, or that the object is not available"""

    try:
        resp = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0},
            },
            timeout=30,
        )
        result = resp.json().get("response", "").strip().upper()
        return "YES" in result
    except Exception as e:
        print(f"  [llm-judge error] {e}")
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


def run_present_queries(label):
    results = []
    print(f"\n── {label} (present objects) ──")
    for case in PRESENT_QUERIES:
        answer = query_backend(case["query"], case["user"])
        results.append({"query": case["query"], "user": case["user"], "answer": answer})
        print(f"  Q: {case['query']}")
        print(f"  A: {answer[:120]}")
        print()
    return results


def run_absent_queries(label):
    results = []
    print(f"\n── {label} (absent objects) ──")
    for case in ABSENT_QUERIES:
        answer = query_backend(case["query"], case["user"])
        halluc = is_hallucination_llm(case["query"], answer)
        results.append({
            "query":         case["query"],
            "user":          case["user"],
            "answer":        answer,
            "hallucination": halluc,
        })
        tag = "HAL" if halluc else "OK "
        print(f"  [{tag}] Q: {case['query']}")
        print(f"        A: {answer[:120]}")
        print()
    return results


def plot_hallucination(r_with, r_without, save_path):
    n = len(r_with)
    hw = sum(1 for r in r_with    if r["hallucination"])
    ho = sum(1 for r in r_without if r["hallucination"])

    labels = ["With Snapshot\n(Injection ON)", "Without Snapshot\n(Injection OFF)"]
    values = [hw / n * 100, ho / n * 100]
    colors = [C["pass"], C["corruption"]]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(range(2), values, color=colors, width=0.45, alpha=0.88)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f"{val:.0f}%",
                ha="center", fontsize=FONT_TICK + 2, fontweight="bold")

    delta = values[1] - values[0]
    if delta != 0:
        ax.annotate("", xy=(1, values[1] + 4), xytext=(0, values[0] + 4),
                    arrowprops=dict(arrowstyle="<->", color="#555", lw=1.5))
        ax.text(0.5, max(values) + 7,
                f"Δ = +{delta:.0f}% hallucinations\nwithout snapshot",
                ha="center", fontsize=FONT_ANNOT,
                color=C["corruption"], fontweight="bold")

    ax.set_xticks(range(2))
    ax.set_xticklabels(labels, fontsize=FONT_TICK)
    ax.set_ylabel("Hallucination Rate (%)", fontsize=FONT_AXIS)
    ax.set_ylim(0, 120)
    ax.set_title(
        "Snapshot Injection — Hallucination Rate on Absent Objects\n"
        "(Judged by LLM-as-a-Judge, not keyword matching)",
        fontsize=FONT_TITLE, fontweight="bold", pad=12)

    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"[exp6] Saved: {save_path}")


def save_summary(present_with, present_without, absent_with, absent_without, save_path):
    n_absent = len(absent_with)
    hw = sum(1 for r in absent_with    if r["hallucination"])
    ho = sum(1 for r in absent_without if r["hallucination"])

    lines = [
        "Experiment 6: Snapshot Injection",
        "=" * 70,
        "",
        "── Part A: Present Object Location Queries ──",
        f"{'Query':<40} {'With':>20} {'Without':>20}",
        "-" * 82,
    ]
    for rw, ro in zip(present_with, present_without):
        w = rw["answer"][:40].replace("\n", " ")
        o = ro["answer"][:40].replace("\n", " ")
        lines.append(f"  {rw['query']:<38} {w:>20} {o:>20}")

    lines += [
        "",
        "── Part B: Absent Object Hallucination Test (LLM-as-a-Judge) ──",
        f"  With Snapshot:    {hw}/{n_absent} hallucinations = {hw/n_absent*100:.0f}%",
        f"  Without Snapshot: {ho}/{n_absent} hallucinations = {ho/n_absent*100:.0f}%",
        "",
        f"{'Query':<40} {'With':>8} {'Without':>8}",
        "-" * 58,
    ]
    for rw, ro in zip(absent_with, absent_without):
        w = "HAL" if rw["hallucination"] else "OK "
        o = "HAL" if ro["hallucination"] else "OK "
        lines.append(f"  {rw['query']:<38} {w:>8} {o:>8}")

    lines += ["", "── Example Responses (absent objects) ──"]
    for rw, ro in zip(absent_with, absent_without):
        lines += [
            f"\nQ: {rw['query']}",
            f"  With:    {rw['answer'][:150]}",
            f"  Without: {ro['answer'][:150]}",
        ]

    with open(save_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[exp6] Saved: {save_path}")
    print("\n".join(lines[:30]))


def save_markdown_table(present_with, present_without, save_path):
    lines = [
        "# Experiment 6: Snapshot Injection — Present Object Location Queries",
        "",
        "| Query | Without Snapshot | With Snapshot |",
        "|-------|-----------------|---------------|",
    ]
    for rw, ro in zip(present_without, present_with):
        q = rw["query"]
        without = rw["answer"].replace("\n", " ").strip()
        with_ans = ro["answer"].replace("\n", " ").strip()
        lines.append(f"| {q} | {without} | {with_ans} |")

    with open(save_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[exp6] Saved: {save_path}")


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

    present_with = run_present_queries("WITH snapshot")
    absent_with  = run_absent_queries("WITH snapshot")

    print("\n[Clearing dynamic_objects...]")
    backup_and_clear(db)

    present_without = run_present_queries("WITHOUT snapshot")
    absent_without  = run_absent_queries("WITHOUT snapshot")

    print("\n[Restoring dynamic_objects...]")
    restore(db)

    plot_hallucination(
        absent_with, absent_without,
        os.path.join(RESULTS_DIR, "exp6_hallucination.png")
    )
    save_markdown_table(
        present_with, present_without,
        os.path.join(RESULTS_DIR, "exp6_present_table.md")
    )
    save_summary(
        present_with, present_without,
        absent_with,  absent_without,
        os.path.join(RESULTS_DIR, "exp6_summary.txt")
    )


if __name__ == "__main__":
    main()