"""
analyze_exp5.py
───────────────
Experiment 5: Fuzzy Need Retrieval via RAG Memory
驗證 InteractionEngine 能否正確回應用戶的模糊需求描述。

設計說明：
    用戶不一定會說出精確的行為名稱（如「喝水」、「閱讀」），
    而是用模糊的自然語言描述需求（如「我口渴了」、「我想找個安靜的地方」）。
    FAISS 語意搜尋的核心價值在於：把模糊描述映射到用戶過去的具體習慣記憶，
    找出最相關的位置和物件，讓機器人提供精確的導航建議。

固定查詢清單（可重現）：
    每個查詢代表一種模糊需求類型，涵蓋不同用戶、不同語意距離。

前提：
    Experiment 3 或 Experiment 4 跑完後執行（記憶庫需要有足夠資料）。
    app.py 必須在執行期間持續運行。

使用方式：
    python3 analyze_exp/analyze_exp5.py
    python3 analyze_exp/analyze_exp5.py --url http://localhost:5000
    python3 analyze_exp/analyze_exp5.py --out ./results/

輸出：
    exp5_results.json   → 完整回答記錄
    exp5_summary.txt    → 論文文字摘要
    exp5_table.csv      → 論文 Table 數值
"""

import argparse
import csv
import datetime
import json
import os
import sys
import time

import requests

BACKEND_URL = "http://localhost:5000"

# ── 固定查詢清單 ───────────────────────────────────────────────────────────
# 格式：(query, user_id, expected_behavior, description)
# expected_behavior：預期系統找到的行為類型（用於評估準確性）
QUERIES = [
    # ── 媽媽的模糊需求 ──────────────────────────────────────────────────
    ("我口渴了",              "User_Mom", "Drink",       "直接需求（口渴→喝水）"),
    ("我想喝點什麼",          "User_Mom", "Drink",       "間接需求（想喝→喝水）"),
    ("我想休息一下",          "User_Mom", "SittingIdle", "模糊需求（休息→坐著）"),
    ("我想坐下來",            "User_Mom", "SittingIdle", "動作需求（坐下→坐著）"),
    ("我想找個地方看書",      "User_Mom", "Reading",     "活動需求（看書→閱讀）"),
    ("我想安靜一下",          "User_Mom", "Reading",     "情緒需求（安靜→閱讀）"),

    # ── 爸爸的模糊需求 ──────────────────────────────────────────────────
    ("我要工作了",            "User_Dad", "Typing",      "工作需求（工作→打字）"),
    ("我想用電腦",            "User_Dad", "Typing",      "工具需求（電腦→打字）"),
    ("我渴了",                "User_Dad", "Drink",       "直接需求（渴→喝水）"),
    ("我想喝水",              "User_Dad", "Drink",       "明確需求（喝水→喝水）"),

    # ── 跨語意距離查詢（較難）──────────────────────────────────────────
    ("我有點累",              "User_Mom", "SittingIdle", "狀態需求（累→坐著）"),
    ("幫我找杯子",            "User_Mom", "Drink",       "物件需求（杯子→喝水位置）"),
]

# ── 呼叫 /interact ─────────────────────────────────────────────────────────
def call_interact(url: str, query: str, user_id: str) -> dict:
    try:
        resp = requests.post(
            f"{url}/interact",
            json={"query": query, "userID": user_id},
            timeout=60,
        )
        if resp.status_code == 200:
            return resp.json()
        return {"error": f"HTTP {resp.status_code}", "detail": resp.text}
    except requests.exceptions.ConnectionError:
        return {"error": "ConnectionError", "detail": "Backend not running"}
    except requests.exceptions.Timeout:
        return {"error": "Timeout", "detail": "Request timed out (60s)"}
    except Exception as e:
        return {"error": str(e)}

# ── Evaluate response ──────────────────────────────────────────────────────
def evaluate(response: dict, expected_behavior: str) -> dict:
    """
    判斷回答是否正確：
    1. 有沒有回傳 nav_target（導航座標）
    2. 回答文字裡有沒有包含預期行為的關鍵字
    3. top_habit 的 action 有沒有對應到預期行為
    """
    if "error" in response:
        return {"has_nav": False, "text_match": False,
                "habit_match": False, "overall": False}

    has_nav = response.get("nav_target") is not None

    answer = response.get("answer", "").lower()
    beh_kw = {
        "Drink":       ["drink", "水", "喝", "cup", "杯"],
        "SittingIdle": ["sit", "couch", "sofa", "沙發", "坐", "休息"],
        "Reading":     ["read", "book", "書", "閱讀"],
        "Typing":      ["type", "desk", "computer", "電腦", "打字", "工作"],
    }
    keywords = beh_kw.get(expected_behavior, [])
    text_match = any(kw in answer for kw in keywords)

    top_habit   = response.get("top_habit") or {}
    habit_actions = top_habit.get("actions", [])
    habit_match = any(
        expected_behavior.lower() in a.lower()
        for a in habit_actions
    ) if habit_actions else False

    overall = has_nav and (text_match or habit_match)

    return {
        "has_nav":     has_nav,
        "text_match":  text_match,
        "habit_match": habit_match,
        "overall":     overall,
    }

# ── Run all queries ────────────────────────────────────────────────────────
def run_experiment(url: str) -> list:
    results = []
    print(f"\nRunning {len(QUERIES)} queries against {url}/interact\n")
    print(f"{'#':>3}  {'User':10}  {'Query':22}  {'Expected':12}  {'Nav':5}  {'Match':6}  {'OK':4}")
    print("─" * 75)

    for i, (query, user_id, expected, desc) in enumerate(QUERIES):
        response = call_interact(url, query, user_id)
        ev       = evaluate(response, expected)

        answer   = response.get("answer", response.get("error", "—"))
        nav      = response.get("nav_target")

        results.append({
            "index":             i + 1,
            "query":             query,
            "user_id":           user_id,
            "expected_behavior": expected,
            "description":       desc,
            "answer":            answer,
            "nav_target":        nav,
            "has_nav":           ev["has_nav"],
            "text_match":        ev["text_match"],
            "habit_match":       ev["habit_match"],
            "overall_correct":   ev["overall"],
            "top_habit":         response.get("top_habit"),
            "semantic_results":  response.get("semantic_results", []),
            "timestamp":         datetime.datetime.now().isoformat(),
        })

        nav_str   = "✓" if ev["has_nav"]   else "✗"
        match_str = "✓" if ev["text_match"] or ev["habit_match"] else "✗"
        ok_str    = "✓" if ev["overall"]    else "✗"
        print(f"{i+1:>3}  {user_id:10}  {query:22}  {expected:12}  "
              f"{nav_str:5}  {match_str:6}  {ok_str:4}")

        time.sleep(0.5)   # 避免 Ollama 過載

    return results

# ── Save outputs ───────────────────────────────────────────────────────────
def save_json(results, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"  ✅ JSON saved: {out_path}")

def save_csv(results, out_path):
    fields = ["index", "query", "user_id", "expected_behavior",
              "description", "has_nav", "text_match", "habit_match",
              "overall_correct", "answer"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(f"  ✅ CSV saved: {out_path}")

def save_summary(results, out_path):
    n        = len(results)
    n_nav    = sum(1 for r in results if r["has_nav"])
    n_match  = sum(1 for r in results if r["text_match"] or r["habit_match"])
    n_ok     = sum(1 for r in results if r["overall_correct"])
    acc      = n_ok / n if n > 0 else 0

    # Per-user breakdown
    users = sorted(set(r["user_id"] for r in results))
    user_stats = {}
    for u in users:
        ur = [r for r in results if r["user_id"] == u]
        user_stats[u] = {
            "n":   len(ur),
            "ok":  sum(1 for r in ur if r["overall_correct"]),
        }

    lines = [
        "=" * 65,
        "Experiment 5: Fuzzy Need Retrieval via RAG Memory",
        f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 65,
        "",
        "── Evaluation Design ───────────────────────────────────────",
        "12 fixed queries across 3 semantic distance levels:",
        "  Direct:   '我口渴了' → Drink",
        "  Indirect: '我想休息一下' → SittingIdle",
        "  Abstract: '我有點累' → SittingIdle",
        "",
        "── Results ──────────────────────────────────────────────────",
        f"Total queries:       {n}",
        f"Nav target returned: {n_nav}/{n} ({n_nav/n:.0%})",
        f"Semantic match:      {n_match}/{n} ({n_match/n:.0%})",
        f"Overall correct:     {n_ok}/{n} ({acc:.0%})",
        "",
        "Per-user:",
        *[f"  {u}: {s['ok']}/{s['n']} ({s['ok']/s['n']:.0%})"
          for u, s in user_stats.items()],
        "",
        "Per-query:",
        *[f"  {'✓' if r['overall_correct'] else '✗'}  "
          f"{r['user_id']:10}  {r['query']:22}  → {r['expected_behavior']}"
          for r in results],
        "",
        "── For thesis (copy-paste) ──────────────────────────────────",
        f"To evaluate the RAG memory system's ability to resolve fuzzy",
        f"user requests, {n} fixed natural language queries were issued",
        f"to the InteractionEngine after Experiment 3 had populated",
        f"the behavioral memory. Queries ranged from direct expressions",
        f"('我口渴了') to abstract emotional states ('我有點累'),",
        f"spanning three semantic distance levels from the stored",
        f"behavioral memory.",
        f"The system returned a valid navigation target in {n_nav} of",
        f"{n} queries ({n_nav/n:.0%}), and produced a semantically",
        f"correct response in {n_ok} of {n} cases",
        f"(overall accuracy = {acc:.0%}), confirming that FAISS-based",
        f"semantic retrieval can bridge the vocabulary gap between",
        f"user expressions and stored behavioral observations.",
    ]

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  ✅ Summary saved: {out_path}")
    print(f"\n  Overall accuracy: {n_ok}/{n} = {acc:.0%}")
    print(f"  Nav returned:     {n_nav}/{n} = {n_nav/n:.0%}")

# ── Main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=BACKEND_URL,
                        help=f"Backend URL (default: {BACKEND_URL})")
    parser.add_argument("--out", default=".",
                        help="Output directory")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # Check backend connectivity
    try:
        requests.get(f"{args.url}/service_history", timeout=5)
    except Exception:
        print(f"❌ Cannot connect to backend at {args.url}")
        print("   Make sure app.py is running before executing this script.")
        sys.exit(1)

    results = run_experiment(args.url)

    print(f"\nSaving outputs...")
    save_json(results,  os.path.join(args.out, "exp5_results.json"))
    save_csv(results,   os.path.join(args.out, "exp5_table.csv"))
    save_summary(results, os.path.join(args.out, "exp5_summary.txt"))

if __name__ == "__main__":
    main()
