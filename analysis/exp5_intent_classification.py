import os, sys, json, requests
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

from exp_config import (
    BACKEND_URL, USERS,
    C, FONT_TITLE, FONT_AXIS, FONT_ANNOT, FONT_TICK,
    FIG_DPI, RESULTS_DIR, apply_style
)

apply_style()

TEST_CASES = [
    # need + drink
    {"query": "I am thirsty",                     "expected": "need", "user": "User_Mom"},
    {"query": "Could I have something to drink",   "expected": "need", "user": "User_Mom"},
    {"query": "I could use a drink",               "expected": "need", "user": "User_Dad"},
    {"query": "Feeling parched",                   "expected": "need", "user": "User_Mom"},
    # need + food
    {"query": "I am hungry",                       "expected": "need", "user": "User_Mom"},
    {"query": "I want something to eat",           "expected": "need", "user": "User_Dad"},
    {"query": "Get me a snack",                    "expected": "need", "user": "User_Dad"},
    {"query": "Feeling peckish",                   "expected": "need", "user": "User_Mom"},
    # need + any
    {"query": "Get me something",                  "expected": "need", "user": "User_Dad"},
    # query
    {"query": "Do we have any cola",               "expected": "query", "user": "User_Mom"},
    {"query": "Is there any cheese",               "expected": "query", "user": "User_Mom"},
    {"query": "What food do we have",              "expected": "query", "user": "User_Dad"},
    {"query": "Is the TV on",                      "expected": "query", "user": "User_Dad"},
    {"query": "Where is my phone",                 "expected": "query", "user": "User_Dad"},
    {"query": "What is in the kitchen",            "expected": "query", "user": "User_Mom"},
    {"query": "Where can I find something to eat", "expected": "query", "user": "User_Dad"},
    # chat
    {"query": "Hello",                             "expected": "chat", "user": "User_Mom"},
    {"query": "I am tired",                        "expected": "chat", "user": "User_Mom"},
    {"query": "Good morning",                      "expected": "chat", "user": "User_Dad"},
    {"query": "I feel sad",                        "expected": "chat", "user": "User_Dad"},
    {"query": "Thank you",                         "expected": "chat", "user": "User_Mom"},
]

INTENT_MAP = {
    "need": "need", "need_confirm": "need",
    "execute": "need", "need_unavailable": "need",
    "query": "query",
    "chat": "chat", "interrupt": "chat", "feedback": "chat",
}


def classify(query, user_id):
    try:
        resp = requests.post(
            f"{BACKEND_URL}/interact/stream",
            json={"query": query, "userID": user_id, "room": ""},
            stream=True, timeout=60,
        )
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            s = line.decode("utf-8") if isinstance(line, bytes) else line
            if not s.startswith("data: "):
                continue
            try:
                ev = json.loads(s[6:])
                if ev.get("type") == "done":
                    return INTENT_MAP.get(ev.get("intent_type", ""), "unknown")
            except Exception:
                continue
    except Exception as e:
        print(f"  [error] {e}")
    return "error"


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_accuracy(per_class, overall, save_path):
    intents = ["need", "query", "chat"]
    accs    = [per_class[i] * 100 for i in intents]
    colors  = [C["baseline"], C["ablation"], C["pass"]]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(intents, accs, color=colors, width=0.45, alpha=0.88)

    ax.axhline(85, color=C["threshold"], linestyle="--",
               lw=1.5, label="Target (85%)")
    ax.axhline(overall * 100, color=C["highlight"], linestyle=":",
               lw=1.5, label=f"Overall ({overall*100:.1f}%)")

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1, f"{acc:.1f}%",
                ha="center", fontsize=FONT_TICK + 1, fontweight="bold")

    ax.set_ylim(0, 120)
    ax.set_ylabel("Accuracy (%)", fontsize=FONT_AXIS)
    ax.set_xlabel("Intent Class", fontsize=FONT_AXIS)
    ax.set_title("Intent Classification Accuracy",
                 fontsize=FONT_TITLE, fontweight="bold", pad=10)
    ax.legend(fontsize=FONT_TICK)

    plt.tight_layout()
    plt.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"[exp5] Saved: {save_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    try:
        r = requests.get(f"{BACKEND_URL}/ready", timeout=3)
        assert "ready" in r.text
        print("[exp5] Backend connected.")
    except Exception:
        print("[exp5] ERROR: Backend not responding. Start app.py first.")
        return

    results   = []
    by_class  = defaultdict(lambda: {"correct": 0, "total": 0})

    for case in TEST_CASES:
        predicted = classify(case["query"], case["user"])
        correct   = predicted == case["expected"]
        by_class[case["expected"]]["total"]   += 1
        by_class[case["expected"]]["correct"] += int(correct)
        results.append({**case, "predicted": predicted, "correct": correct})
        mark = "OK" if correct else "XX"
        print(f"  [{mark}] [{case['user']}] \"{case['query']}\"  "
              f"expected={case['expected']} predicted={predicted}")

    total   = len(results)
    correct = sum(1 for r in results if r["correct"])
    overall = correct / total if total else 0

    per_class = {
        intent: (by_class[intent]["correct"] / by_class[intent]["total"]
                 if by_class[intent]["total"] > 0 else 0)
        for intent in ["need", "query", "chat"]
    }

    print(f"\nOverall: {correct}/{total} = {overall*100:.1f}%")
    for intent, acc in per_class.items():
        info = by_class[intent]
        print(f"  {intent:<8} {info['correct']}/{info['total']} = {acc*100:.1f}%")

    status = "PASSED" if overall >= 0.85 else "FAILED"
    print(f"[Result] {status} (target ≥ 85%)")

    plot_accuracy(per_class, overall,
                  os.path.join(RESULTS_DIR, "exp5_intent_accuracy.png"))

    # Summary
    path = os.path.join(RESULTS_DIR, "exp5_summary.txt")
    with open(path, "w") as f:
        f.write(f"Experiment 5: Intent Classification\n")
        f.write(f"Total: {total}  Correct: {correct}  "
                f"Overall: {overall*100:.1f}%  Status: {status}\n\n")
        for intent, acc in per_class.items():
            info = by_class[intent]
            f.write(f"{intent}: {info['correct']}/{info['total']} = {acc*100:.1f}%\n")
        f.write("\nDetails:\n")
        for r in results:
            mark = "OK" if r["correct"] else "XX"
            f.write(f"  [{mark}] {r['expected']:<8} → {r['predicted']:<8} "
                    f"| {r['query']}\n")
    print(f"[exp5] Saved: {path}")


if __name__ == "__main__":
    main()