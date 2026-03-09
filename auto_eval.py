"""
auto_eval.py
────────────────────────────────────────────────
自動跑完實驗四（25題）和實驗六（30題）的所有問題，
把系統回答存成 CSV，方便你和評審直接打分。

執行方式：
    python auto_eval.py --exp 4          # 只跑實驗四
    python auto_eval.py --exp 6          # 只跑實驗六
    python auto_eval.py                  # 全部跑

產出：
    auto_eval4_responses.csv   ← 實驗四，填 score_r1/r2/r3
    auto_eval6_responses.csv   ← 實驗六，填 intent_correct/nav_correct/personalized
"""

import requests
import csv
import time
import argparse
from datetime import datetime

FLASK_URL = "http://127.0.0.1:5000"

# ══════════════════════════════════════════════
# 實驗四：25 題問答
# 每題測三個條件：無記憶 / 有記憶無個人化 / 本系統
# ══════════════════════════════════════════════
EXP4_QUESTIONS = [
    # (題號, 問題, 用戶)
    ("Q01", "媽媽的藥放在哪裡？",         "User_Mom"),
    ("Q02", "媽媽早上通常在哪裡？",       "User_Mom"),
    ("Q03", "廚房桌上現在有什麼？",       "User_Mom"),
    ("Q04", "爸爸睡覺在哪個房間？",       "User_Dad"),
    ("Q05", "媽媽喝水用什麼杯子？",       "User_Mom"),
    ("Q06", "爸爸最常做什麼活動？",       "User_Dad"),
    ("Q07", "昨天媽媽做了什麼？",         "User_Mom"),
    ("Q08", "遙控器在哪裡？",             "User_Dad"),
    ("Q09", "媽媽通常幾點吃早餐？",       "User_Mom"),
    ("Q10", "爸爸的書桌上有什麼？",       "User_Dad"),
    ("Q11", "媽媽喜歡在哪裡休息？",       "User_Mom"),
    ("Q12", "廚房裡有水果嗎？",           "User_Mom"),
    ("Q13", "爸爸昨晚在做什麼？",         "User_Dad"),
    ("Q14", "媽媽的手機通常在哪？",       "User_Mom"),
    ("Q15", "爸爸最近有沒有在廚房？",     "User_Dad"),
    ("Q16", "媽媽喝什麼飲料？",           "User_Mom"),
    ("Q17", "客廳現在有人嗎？",           "User_Mom"),
    ("Q18", "爸爸的眼鏡在哪？",           "User_Dad"),
    ("Q19", "媽媽上次在廚房是幾點？",     "User_Mom"),
    ("Q20", "誰比較常待在客廳？",         "User_Mom"),
    ("Q21", "媽媽睡前習慣做什麼？",       "User_Mom"),
    ("Q22", "爸爸的杯子在哪？",           "User_Dad"),
    ("Q23", "媽媽今天有做飯嗎？",         "User_Mom"),
    ("Q24", "床頭燈在哪個房間？",         "User_Mom"),
    ("Q25", "爸爸通常幾點睡覺？",         "User_Dad"),
]

# 三個測試條件
EXP4_CONDITIONS = [
    ("no_memory",             None),           # 條件A：不帶 userID
    ("memory_no_personalize", "Unknown_User"), # 條件B：帶假 userID
    ("memory_personalized",   None),           # 條件C：帶真實 userID（從題目拿）
]

# ══════════════════════════════════════════════
# 實驗六：30 題模糊需求
# ══════════════════════════════════════════════
EXP6_QUESTIONS = [
    # (題號, 問題, 用戶)
    ("F01", "我累了",               "User_Mom"),
    ("F02", "我累了",               "User_Dad"),
    ("F03", "我渴了",               "User_Mom"),
    ("F04", "我渴了",               "User_Dad"),
    ("F05", "我餓了",               "User_Mom"),
    ("F06", "我餓了",               "User_Dad"),
    ("F07", "我無聊",               "User_Mom"),
    ("F08", "我無聊",               "User_Dad"),
    ("F09", "我不舒服",             "User_Mom"),
    ("F10", "我不舒服",             "User_Dad"),
    ("F11", "幫我找我的眼鏡",       "User_Dad"),
    ("F12", "幫我找遙控器",         "User_Mom"),
    ("F13", "我想休息",             "User_Mom"),
    ("F14", "我想喝點東西",         "User_Dad"),
    ("F15", "肚子餓了",             "User_Mom"),
    ("F16", "想看電視",             "User_Dad"),
    ("F17", "我想睡覺",             "User_Mom"),
    ("F18", "找一下我的藥",         "User_Mom"),
    ("F19", "有點渴",               "User_Dad"),
    ("F20", "我餓了，有什麼吃的？", "User_Mom"),
    ("F21", "I'm tired",            "User_Mom"),
    ("F22", "I'm hungry",           "User_Dad"),
    ("F23", "I'm thirsty",          "User_Mom"),
    ("F24", "Where's the remote?",  "User_Dad"),
    ("F25", "I need to rest",       "User_Dad"),
    ("F26", "好想吃東西",           "User_Mom"),
    ("F27", "我冷",                 "User_Mom"),
    ("F28", "找我的水杯",           "User_Dad"),
    ("F29", "我想吃點心",           "User_Mom"),
    ("F30", "幫我找媽媽的杯子",     "User_Dad"),
]


def ask(query, user_id):
    """送出一個問題到 /interact，回傳答案字串"""
    try:
        res = requests.post(
            f"{FLASK_URL}/interact",
            json={"query": query, "userID": user_id},
            timeout=60
        )
        data = res.json()
        answer    = data.get("answer", "")
        nav_label = data.get("nav_label", "")
        nav_target= data.get("nav_target", "")
        intent    = data.get("intent_type", "")
        return answer, nav_label, str(nav_target), intent
    except Exception as e:
        return f"[ERROR] {e}", "", "", ""


def run_exp4():
    print("\n══ 實驗四：對話品質評分（25 題 × 3 條件）══")
    print("每題會自動送出三次（無記憶 / 有記憶無個人化 / 本系統）")
    print("完成後打開 auto_eval4_responses.csv 填分數\n")

    rows = []
    total = len(EXP4_QUESTIONS) * 3
    count = 0

    for qid, question, real_user in EXP4_QUESTIONS:
        print(f"[{qid}] {question}")
        row = {
            "question_id": qid,
            "question":    question,
            "user":        real_user,
        }

        for cond_name, cond_user in EXP4_CONDITIONS:
            count += 1
            # 條件C 用真實 userID
            user_to_use = real_user if cond_name == "memory_personalized" else cond_user
            print(f"  條件 {cond_name}...", end=" ", flush=True)

            answer, nav_label, nav_target, intent = ask(question, user_to_use)
            print(f"✓")

            row[f"{cond_name}_answer"]     = answer
            row[f"{cond_name}_nav_label"]  = nav_label
            row[f"{cond_name}_nav_target"] = nav_target
            row[f"score_r1_{cond_name}"]   = ""   # 評審 1 填
            row[f"score_r2_{cond_name}"]   = ""   # 評審 2 填
            row[f"score_r3_{cond_name}"]   = ""   # 評審 3 填

            time.sleep(1)  # 避免 Ollama 過載

        rows.append(row)
        print()

    # 存 CSV
    fname = "auto_eval4_responses.csv"
    fieldnames = ["question_id", "question", "user"]
    for cond_name, _ in EXP4_CONDITIONS:
        fieldnames += [
            f"{cond_name}_answer",
            f"{cond_name}_nav_label",
            f"{cond_name}_nav_target",
            f"score_r1_{cond_name}",
            f"score_r2_{cond_name}",
            f"score_r3_{cond_name}",
        ]

    with open(fname, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ 已存到 {fname}")
    print(f"   打開 Excel → 填 score_r1/r2/r3 欄位（每格填 1-5）")
    print(f"   填完後執行：python eval_all.py --exp 4")


def run_exp6():
    print("\n══ 實驗六：個人化模糊需求（30 題）══")
    print("完成後打開 auto_eval6_responses.csv 填 1/0\n")

    rows = []

    for qid, question, user_id in EXP6_QUESTIONS:
        print(f"[{qid}] {question} ({user_id})...", end=" ", flush=True)

        answer, nav_label, nav_target, intent = ask(question, user_id)
        print("✓")

        rows.append({
            "qid":              qid,
            "input":            question,
            "user":             user_id,
            "system_answer":    answer,
            "nav_label":        nav_label,
            "nav_target":       nav_target,
            "intent_type":      intent,
            # ↓ 這三欄你來填
            "intent_correct":   "",  # 意圖辨識正確？1=對 0=錯
            "nav_correct":      "",  # 導航位置正確？1=對 0=錯
            "personalized":     "",  # 有個人化差異？1=有 0=沒有
            "note":             "",  # 備註（選填）
        })

        time.sleep(1)

    fname = "auto_eval6_responses.csv"
    with open(fname, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅ 已存到 {fname}")
    print(f"   打開 Excel → 填 intent_correct / nav_correct / personalized（填 1 或 0）")
    print(f"   填完後執行：python eval_all.py --exp 6")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default="all", help="4 / 6 / all")
    args = parser.parse_args()

    print(f"[auto_eval] 開始時間：{datetime.now().strftime('%H:%M:%S')}")
    print(f"[auto_eval] Flask URL：{FLASK_URL}")

    if args.exp in ("all", "4"): run_exp4()
    if args.exp in ("all", "6"): run_exp6()

    print(f"\n[auto_eval] 完成時間：{datetime.now().strftime('%H:%M:%S')}")
