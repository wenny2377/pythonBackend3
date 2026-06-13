import os, sys, json, requests
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

BACKEND = "http://127.0.0.1:5000"
OUT     = os.path.join(_ROOT, "analysis", "results")

TEST_CASES = [
    {"query": "I am thirsty",                    "expected": "need",  "user": "User_Mom"},
    {"query": "Could I have something to drink",  "expected": "need",  "user": "User_Mom"},
    {"query": "I am hungry",                      "expected": "need",  "user": "User_Mom"},
    {"query": "I want something to eat",          "expected": "need",  "user": "User_Mom"},
    {"query": "Get me a snack",                   "expected": "need",  "user": "User_Dad"},
    {"query": "I could use a drink",              "expected": "need",  "user": "User_Dad"},
    {"query": "Feeling peckish",                  "expected": "need",  "user": "User_Dad"},
    {"query": "Where is the remote",              "expected": "query", "user": "User_Mom"},
    {"query": "What food do we have",             "expected": "query", "user": "User_Mom"},
    {"query": "Do we have any cola",              "expected": "query", "user": "User_Mom"},
    {"query": "Is the TV on",                     "expected": "query", "user": "User_Dad"},
    {"query": "Where is my phone",                "expected": "query", "user": "User_Dad"},
    {"query": "What is in the kitchen",           "expected": "query", "user": "User_Dad"},
    {"query": "Where is the water bottle",        "expected": "query", "user": "User_Mom"},
    {"query": "Hello",                            "expected": "chat",  "user": "User_Mom"},
    {"query": "I am tired",                       "expected": "chat",  "user": "User_Mom"},
    {"query": "How are you",                      "expected": "chat",  "user": "User_Mom"},
    {"query": "I am bored",                       "expected": "chat",  "user": "User_Dad"},
    {"query": "Good morning",                     "expected": "chat",  "user": "User_Dad"},
    {"query": "I feel sad",                       "expected": "chat",  "user": "User_Dad"},
    {"query": "Thank you",                        "expected": "chat",  "user": "User_Mom"},
]

INTENT_MAP = {
    "need":         "need",
    "need_confirm": "need",
    "execute":      "need",
    "need_unavailable": "need",
    "query":        "query",
    "chat":         "chat",
    "interrupt":    "chat",
    "feedback":     "chat",
}

def classify(query, user_id):
    try:
        resp = requests.post(
            f"{BACKEND}/interact/stream",
            json={"query": query, "userID": user_id, "room": ""},
            stream=True, timeout=60,
        )
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            line_str = line.decode("utf-8") if isinstance(line, bytes) else line
            if not line_str.startswith("data: "):
                continue
            try:
                event = json.loads(line_str[6:])
                if event.get("type") == "done":
                    raw = event.get("intent_type", "")
                    return INTENT_MAP.get(raw, raw)
            except Exception:
                continue
    except Exception as e:
        print(f"  [error] {e}")
    return "error"

def plot_confusion(results, save_path):
    intents = ["need", "query", "chat"]
    matrix  = defaultdict(lambda: defaultdict(int))
    for r in results:
        matrix[r["expected"]][r["predicted"]] += 1

    fig, ax = plt.subplots(figsize=(6, 5))
    data    = [[matrix[i][j] for j in intents] for i in intents]
    im      = ax.imshow(data, cmap="Blues")
    ax.set_xticks(range(len(intents)))
    ax.set_yticks(range(len(intents)))
    ax.set_xticklabels(intents)
    ax.set_yticklabels(intents)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_title("RQ1 — Intent Classification Confusion Matrix")
    for i in range(len(intents)):
        for j in range(len(intents)):
            ax.text(j, i, str(data[i][j]),
                    ha="center", va="center",
                    color="white" if data[i][j] > 3 else "black", fontsize=12)
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[RQ1] Confusion matrix saved → {save_path}")

def plot_accuracy_bar(per_class_acc, overall_acc, save_path):
    intents = list(per_class_acc.keys())
    accs    = [per_class_acc[i] * 100 for i in intents]
    colors  = ["#2196F3", "#4CAF50", "#FF9800"]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars    = ax.bar(intents, accs, color=colors, width=0.5)
    ax.axhline(y=85, color="red", linestyle="--", linewidth=1.2, label="Target (85%)")
    ax.axhline(y=overall_acc * 100, color="gray", linestyle=":",
               linewidth=1.2, label=f"Overall ({overall_acc*100:.1f}%)")
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1, f"{acc:.1f}%",
                ha="center", va="bottom", fontsize=11)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("RQ1 — Intent Classification Accuracy per Class")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[RQ1] Accuracy bar saved → {save_path}")

def main():
    os.makedirs(OUT, exist_ok=True)
    print("=" * 60)
    print("RQ1 Intent Classification Accuracy")
    print("=" * 60)

    try:
        r = requests.get(f"{BACKEND}/ready", timeout=3)
        if "ready" not in r.text:
            raise Exception
        print("[OK] Backend connected.\n")
    except Exception:
        print("[ERROR] Backend not responding. Start app.py first.")
        return

    results = []
    by_class = defaultdict(lambda: {"correct": 0, "total": 0})

    for i, case in enumerate(TEST_CASES):
        query    = case["query"]
        expected = case["expected"]
        user_id  = case["user"]
        predicted = classify(query, user_id)
        correct   = predicted == expected
        by_class[expected]["total"]   += 1
        by_class[expected]["correct"] += int(correct)
        results.append({
            "query":     query,
            "expected":  expected,
            "predicted": predicted,
            "correct":   correct,
            "user":      user_id,
        })
        mark = "OK" if correct else "XX"
        print(f"  [{mark}] [{user_id}] \"{query}\"")
        print(f"       expected={expected} predicted={predicted}")

    total   = len(results)
    correct = sum(1 for r in results if r["correct"])
    overall = correct / total if total > 0 else 0

    print(f"\n{'='*60}")
    print(f"Overall Accuracy: {correct}/{total} = {overall*100:.1f}%")
    print(f"\nPer-class:")
    per_class_acc = {}
    for intent in ["need", "query", "chat"]:
        info = by_class[intent]
        acc  = info["correct"] / info["total"] if info["total"] > 0 else 0
        per_class_acc[intent] = acc
        print(f"  {intent:<8} {info['correct']}/{info['total']} = {acc*100:.1f}%")

    status = "PASSED" if overall >= 0.85 else "FAILED"
    print(f"\n[Result] {status} (target ≥ 85%)")

    plot_confusion(results, os.path.join(OUT, "rq1_confusion.png"))
    plot_accuracy_bar(per_class_acc, overall,
                      os.path.join(OUT, "rq1_accuracy.png"))

    summary_path = os.path.join(OUT, "rq1_intent_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"RQ1 Intent Classification\n")
        f.write(f"Total: {total}\n")
        f.write(f"Correct: {correct}\n")
        f.write(f"Overall Accuracy: {overall*100:.1f}%\n")
        f.write(f"Status: {status}\n\n")
        for intent, acc in per_class_acc.items():
            info = by_class[intent]
            f.write(f"{intent}: {info['correct']}/{info['total']} = {acc*100:.1f}%\n")
        f.write("\nDetails:\n")
        for r in results:
            mark = "OK" if r["correct"] else "XX"
            f.write(f"  [{mark}] {r['expected']:<8} → {r['predicted']:<8} | {r['query']}\n")
    print(f"[RQ1] Summary saved → {summary_path}")

if __name__ == "__main__":
    main()