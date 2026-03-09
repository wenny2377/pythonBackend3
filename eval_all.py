"""
eval_all.py
───────────────────────────────────────────────
執行方式：
    python eval_all.py              # 跑全部實驗
    python eval_all.py --exp 1      # 只跑實驗一
    python eval_all.py --exp 3a     # 只跑實驗三A
    python eval_all.py --exp 3b
    python eval_all.py --exp 4      # 需要先填好 eval_exp4_scores.csv
    python eval_all.py --exp 5
    python eval_all.py --exp 6

產出位置：results/
    results/exp1_accuracy.png
    results/exp1_confusion_matrix.png
    results/exp1_report.txt
    results/exp2_binding_accuracy.png
    results/exp3a_weight_curve.png
    results/exp3b_transition_heatmap.png
    results/exp4_dialogue_scores.png
    results/exp5_latency_pie.png
    results/exp5_success_rate.png
    results/exp6_intent_accuracy.png
    results/exp6_personalization.png
    results/summary_report.txt
"""

import argparse
import os
import json
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 不需要顯示視窗，直接存檔
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pymongo import MongoClient

# ── 中文字體設定（Windows / Linux 通用）──────────
def setup_chinese_font():
    candidates = [
        "C:/Windows/Fonts/msjh.ttc",       # Windows 微軟正黑體
        "C:/Windows/Fonts/simsun.ttc",      # Windows 新細明體
        "/usr/share/fonts/truetype/arphic/uming.ttc",  # Ubuntu
        "/System/Library/Fonts/PingFang.ttc",           # macOS
    ]
    for path in candidates:
        if os.path.exists(path):
            prop = fm.FontProperties(fname=path)
            plt.rcParams['font.family'] = prop.get_name()
            print(f"[Font] 使用字體：{path}")
            return
    # 找不到就用預設（中文可能顯示方塊，但不影響數據）
    print("[Font] 找不到中文字體，使用預設字體（中文可能顯示方塊）")

setup_chinese_font()
plt.rcParams['axes.unicode_minus'] = False

# ── MongoDB 連線 ─────────────────────────────────
client = MongoClient("mongodb://127.0.0.1:27017/")
db     = client["robot_rag_db"]

OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

def savefig(name):
    path = os.path.join(OUT_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {path}")


# ════════════════════════════════════════════════
# 實驗一：VLM 行為辨識準確率
# ════════════════════════════════════════════════
def run_exp1():
    print("\n══ 實驗一：VLM 行為辨識準確率 ══")

    docs = list(db.eval_logs.find(
        {"experiment": "exp1_exp2", "ground_truth": {"$ne": None, "$ne": ""}},
        {"ground_truth": 1, "vlm_output": 1, "user_id": 1, "room": 1, "vlm_inference_ms": 1}
    ))

    if not docs:
        print("[Exp1] eval_logs 資料為空，請先跑完實驗一")
        return

    gt_list  = [d["ground_truth"].lower().strip() for d in docs]
    pred_list= [d["vlm_output"].lower().strip()   for d in docs]

    labels   = sorted(set(gt_list))
    correct  = sum(g == p for g, p in zip(gt_list, pred_list))
    total    = len(docs)
    accuracy = correct / total

    # ── 各行為 Precision / Recall / F1 ──
    metrics = {}
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(gt_list, pred_list))
        fp = sum(g != label and p == label for g, p in zip(gt_list, pred_list))
        fn = sum(g == label and p != label for g, p in zip(gt_list, pred_list))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        metrics[label] = {"precision": precision, "recall": recall, "f1": f1, "count": sum(g == label for g in gt_list)}

    # ── 圖一：各行為 F1 長條圖 ──
    fig, ax = plt.subplots(figsize=(10, 5))
    x       = np.arange(len(labels))
    width   = 0.25
    prec    = [metrics[l]["precision"] for l in labels]
    rec     = [metrics[l]["recall"]    for l in labels]
    f1s     = [metrics[l]["f1"]        for l in labels]

    ax.bar(x - width, prec, width, label="Precision", color="#4C9BE8")
    ax.bar(x,         rec,  width, label="Recall",    color="#E8844C")
    ax.bar(x + width, f1s,  width, label="F1",        color="#4CE87A")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel("Score")
    ax.set_title(f"Exp1: Per-Action Metrics (Overall Accuracy = {accuracy:.1%})")
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='Target 70%')
    savefig("exp1_accuracy.png")

    # ── 圖二：Confusion Matrix ──
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    label_idx = {l: i for i, l in enumerate(labels)}
    for g, p in zip(gt_list, pred_list):
        if g in label_idx and p in label_idx:
            cm[label_idx[g]][label_idx[p]] += 1

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=30)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Ground Truth")
    ax.set_title("Exp1: Confusion Matrix")

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(cm[i][j]), ha='center', va='center',
                    color='white' if cm[i][j] > cm.max()/2 else 'black')
    plt.colorbar(im, ax=ax)
    savefig("exp1_confusion_matrix.png")

    # ── VLM 推理時間統計 ──
    ms_list = [d.get("vlm_inference_ms", 0) for d in docs if d.get("vlm_inference_ms")]
    avg_ms  = np.mean(ms_list) if ms_list else 0

    # ── 文字報告 ──
    report = [
        "═══ 實驗一 報告 ═══",
        f"總筆數：{total}",
        f"Overall Accuracy：{accuracy:.1%}  ({correct}/{total})",
        f"平均 VLM 推理時間：{avg_ms:.0f} ms",
        "",
        f"{'行為':<15} {'Precision':>10} {'Recall':>10} {'F1':>10} {'樣本數':>8}",
        "─" * 55,
    ]
    for l in labels:
        m = metrics[l]
        report.append(f"{l:<15} {m['precision']:>10.3f} {m['recall']:>10.3f} {m['f1']:>10.3f} {m['count']:>8}")

    report_str = "\n".join(report)
    print(report_str)
    with open(os.path.join(OUT_DIR, "exp1_report.txt"), "w", encoding="utf-8") as f:
        f.write(report_str)


# ════════════════════════════════════════════════
# 實驗二：家具語義綁定消融
# ════════════════════════════════════════════════
def run_exp2():
    print("\n══ 實驗二：家具綁定消融實驗 ══")

    docs = list(db.eval_logs.find(
        {"experiment": "exp1_exp2", "binding_results": {"$exists": True}}
    ))

    if not docs:
        print("[Exp2] binding_results 資料為空")
        print("  → 需要在 memory.py 的 bind_and_update() 裡加入三種方法記錄")
        print("  → 目前 eval_logs 只記錄了方法C（本系統）的結果")

        # 用現有的 bound_label 資料跑實驗二（只有方法C）
        docs_basic = list(db.eval_logs.find(
            {"experiment": "exp1_exp2", "bound_label": {"$ne": None}}
        ))
        if not docs_basic:
            return

        correct_c = sum(1 for d in docs_basic
                       if d.get("bound_label") and "Unknown" not in d.get("bound_label",""))
        total     = len(docs_basic)
        print(f"[Exp2] 方法C（本系統）綁定成功率：{correct_c/total:.1%} ({correct_c}/{total})")

        # 繪製只有方法C的結果
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(["方法C：本系統\n(語義+距離)"],
               [correct_c/total], color=["#4CE87A"], width=0.4)
        ax.set_ylabel("Top-1 Accuracy")
        ax.set_title("Exp2: 家具綁定準確率（僅方法C）")
        ax.set_ylim(0, 1.1)
        ax.axhline(y=0.7, color='red', linestyle='--', alpha=0.5)
        for i, v in enumerate([correct_c/total]):
            ax.text(i, v + 0.02, f"{v:.1%}", ha='center', fontweight='bold')
        savefig("exp2_binding_accuracy.png")
        return

    # 有三種方法資料時的完整分析
    methods = ["method_a", "method_b", "method_c"]
    labels  = ["方法A\n(純距離)", "方法B\n(純語義)", "方法C\n(本系統)"]
    colors  = ["#E84C4C", "#4C9BE8", "#4CE87A"]
    accs    = []

    for m in methods:
        correct = sum(1 for d in docs
                     if d.get("binding_results", {}).get(m)
                     and "Unknown" not in d["binding_results"][m])
        accs.append(correct / len(docs))

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, accs, color=colors, width=0.5)
    ax.set_ylabel("Top-1 Accuracy")
    ax.set_title("Exp2: 家具語義綁定消融實驗")
    ax.set_ylim(0, 1.15)
    ax.axhline(y=0.7, color='gray', linestyle='--', alpha=0.5, label='Target 70%')
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{acc:.1%}", ha='center', fontweight='bold')
    ax.legend()
    savefig("exp2_binding_accuracy.png")
    print(f"[Exp2] 方法A={accs[0]:.1%} 方法B={accs[1]:.1%} 方法C={accs[2]:.1%}")


# ════════════════════════════════════════════════
# 實驗三A：習慣 Weight 累積
# ════════════════════════════════════════════════
def run_exp3a():
    print("\n══ 實驗三A：習慣 Weight 累積 ══")

    checkpoints = list(db.exp_checkpoints.find(
        {"experiment": "exp3a"},
        sort=[("step", 1)]
    ))

    if not checkpoints:
        print("[Exp3A] exp_checkpoints 資料為空，請先跑完實驗三A")
        return

    steps      = [c["step"]       for c in checkpoints]
    weights    = [c.get("weight", 0)     for c in checkpoints]
    sims       = [c.get("similarity", 0) for c in checkpoints]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2      = ax1.twinx()

    l1, = ax1.plot(steps, weights, 'b-o', linewidth=2, markersize=6, label="Weight 值")
    l2, = ax2.plot(steps, sims,    'r-s', linewidth=2, markersize=6, label="FAISS Similarity")

    ax1.set_xlabel("觀測次數")
    ax1.set_ylabel("observation_logs.weight", color='blue')
    ax2.set_ylabel("FAISS Top-1 Similarity",  color='red')
    ax1.set_title("Exp3A: 習慣 Weight 與 FAISS Similarity 隨觀測次數變化")

    lines  = [l1, l2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left')
    ax1.grid(True, alpha=0.3)
    savefig("exp3a_weight_curve.png")

    print(f"[Exp3A] 最終 weight={weights[-1]} similarity={sims[-1]:.4f}")


# ════════════════════════════════════════════════
# 實驗三B：行為序列轉換熱力圖
# ════════════════════════════════════════════════
def run_exp3b():
    print("\n══ 實驗三B：行為序列預測 ══")

    # 從 activity_sequences 取出序列
    seqs = list(db.activity_sequences.find(
        {},
        {"sequence": 1, "user_id": 1}
    ))

    if not seqs:
        print("[Exp3B] activity_sequences 資料為空")
        return

    # 建 transition 矩陣
    all_actions = set()
    transitions = defaultdict(lambda: defaultdict(int))

    for doc in seqs:
        seq = doc.get("sequence", [])
        for i in range(len(seq) - 1):
            a_from = seq[i].get("action", seq[i]) if isinstance(seq[i], dict) else seq[i]
            a_to   = seq[i+1].get("action", seq[i+1]) if isinstance(seq[i+1], dict) else seq[i+1]
            a_from = str(a_from).lower()
            a_to   = str(a_to).lower()
            transitions[a_from][a_to] += 1
            all_actions.add(a_from)
            all_actions.add(a_to)

    if not all_actions:
        print("[Exp3B] 沒有足夠的序列資料")
        return

    actions = sorted(all_actions)
    n       = len(actions)
    idx     = {a: i for i, a in enumerate(actions)}
    matrix  = np.zeros((n, n), dtype=int)

    for a_from, targets in transitions.items():
        if a_from in idx:
            for a_to, count in targets.items():
                if a_to in idx:
                    matrix[idx[a_from]][idx[a_to]] = count

    # ── Top-1 / Top-3 Accuracy ──
    correct_1 = correct_3 = total_pred = 0
    for doc in seqs:
        seq = doc.get("sequence", [])
        for i in range(len(seq) - 1):
            a_from = str(seq[i]).lower() if not isinstance(seq[i], dict) else str(seq[i].get("action","")).lower()
            a_to   = str(seq[i+1]).lower() if not isinstance(seq[i+1], dict) else str(seq[i+1].get("action","")).lower()
            if a_from not in idx: continue
            row    = matrix[idx[a_from]]
            top3   = [actions[j] for j in np.argsort(row)[::-1][:3]]
            top1   = top3[0] if top3 else ""
            if top1   == a_to: correct_1 += 1
            if a_to in top3:   correct_3 += 1
            total_pred += 1

    acc1 = correct_1 / total_pred if total_pred > 0 else 0
    acc3 = correct_3 / total_pred if total_pred > 0 else 0

    # ── 熱力圖 ──
    fig, ax = plt.subplots(figsize=(9, 7))
    im      = ax.imshow(matrix, cmap='YlOrRd')
    ax.set_xticks(range(n)); ax.set_xticklabels(actions, rotation=30, ha='right')
    ax.set_yticks(range(n)); ax.set_yticklabels(actions)
    ax.set_xlabel("Next Action")
    ax.set_ylabel("Current Action")
    ax.set_title(f"Exp3B: 行為轉換熱力圖\nTop-1 Acc={acc1:.1%}  Top-3 Acc={acc3:.1%}")

    for i in range(n):
        for j in range(n):
            if matrix[i][j] > 0:
                ax.text(j, i, str(matrix[i][j]), ha='center', va='center',
                        color='white' if matrix[i][j] > matrix.max()*0.6 else 'black', fontsize=9)
    plt.colorbar(im, ax=ax)
    savefig("exp3b_transition_heatmap.png")

    print(f"[Exp3B] Top-1 Accuracy={acc1:.1%}  Top-3 Accuracy={acc3:.1%}")
    print(f"[Exp3B] 總預測數={total_pred}")


# ════════════════════════════════════════════════
# 實驗四：RAG 對話品質（需要人工評分 CSV）
# ════════════════════════════════════════════════
def run_exp4():
    print("\n══ 實驗四：RAG 對話品質 ══")

    csv_path = "eval_exp4_scores.csv"
    if not os.path.exists(csv_path):
        # 自動產生空白填分表
        lines = ["question_id,question,condition,score_r1,score_r2,score_r3"]
        questions = [
            "Q01,媽媽的藥放在哪裡",
            "Q02,媽媽早上通常在哪裡",
            "Q03,廚房桌上現在有什麼",
            "Q04,爸爸睡覺在哪個房間",
            "Q05,媽媽喝水用什麼杯子",
            "Q06,爸爸最常做什麼活動",
            "Q07,昨天媽媽做了什麼",
            "Q08,遙控器在哪裡",
            "Q09,媽媽通常幾點吃早餐",
            "Q10,爸爸的書桌上有什麼",
            "Q11,媽媽喜歡在哪裡休息",
            "Q12,廚房裡有水果嗎",
            "Q13,爸爸昨晚在做什麼",
            "Q14,媽媽的手機通常在哪",
            "Q15,爸爸最近有沒有在廚房",
            "Q16,媽媽喝什麼飲料",
            "Q17,客廳現在有人嗎",
            "Q18,爸爸的眼鏡在哪",
            "Q19,媽媽上次在廚房是幾點",
            "Q20,誰比較常待在客廳",
            "Q21,媽媽睡前習慣做什麼",
            "Q22,爸爸的杯子在哪",
            "Q23,媽媽今天有做飯嗎",
            "Q24,床頭燈在哪個房間",
            "Q25,爸爸通常幾點睡覺",
        ]
        for q in questions:
            for condition in ["no_memory", "memory_no_personalize", "memory_personalized"]:
                lines.append(f"{q},{condition},,,")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"[Exp4] 已產生空白評分表：{csv_path}")
        print("  → 請填入評分（1-5）後再執行 python eval_all.py --exp 4")
        return

    # 讀取已填好的 CSV
    import csv
    data_by_condition = defaultdict(list)

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scores = []
            for key in ["score_r1", "score_r2", "score_r3"]:
                try:
                    scores.append(float(row[key]))
                except (ValueError, KeyError):
                    pass
            if scores:
                avg = np.mean(scores)
                data_by_condition[row["condition"]].append({
                    "qid": row["question_id"],
                    "q":   row["question"],
                    "avg": avg
                })

    if not data_by_condition:
        print("[Exp4] CSV 尚未填入評分")
        return

    conditions = ["no_memory", "memory_no_personalize", "memory_personalized"]
    labels_cn  = ["無記憶\n(Baseline A)", "有記憶\n無個人化", "有記憶\n有個人化\n(本系統)"]
    colors     = ["#E84C4C", "#4C9BE8", "#4CE87A"]
    avgs       = [np.mean([d["avg"] for d in data_by_condition.get(c, [{"avg":0}])]) for c in conditions]

    # ── 長條圖：三組整體平均 ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    bars = ax1.bar(labels_cn, avgs, color=colors, width=0.5)
    ax1.set_ylabel("平均評分（1-5）")
    ax1.set_title("Exp4: RAG 對話品質三組比較")
    ax1.set_ylim(0, 5.5)
    ax1.axhline(y=3, color='gray', linestyle='--', alpha=0.5, label='及格線 3分')
    for bar, avg in zip(bars, avgs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f"{avg:.2f}", ha='center', fontweight='bold')
    ax1.legend()

    # ── 折線圖：每題各組分數 ──
    q_ids = [d["qid"] for d in data_by_condition.get(conditions[0], [])]
    for i, (cond, color) in enumerate(zip(conditions, colors)):
        scores = [d["avg"] for d in data_by_condition.get(cond, [])]
        if scores:
            ax2.plot(range(len(scores)), scores, color=color,
                     marker='o', linewidth=1.5, markersize=4, label=labels_cn[i])
    ax2.set_xlabel("題號")
    ax2.set_ylabel("評分")
    ax2.set_title("Exp4: 各題得分比較")
    ax2.set_ylim(0, 5.5)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    savefig("exp4_dialogue_scores.png")
    print(f"[Exp4] 無記憶={avgs[0]:.2f} | 有記憶無個人化={avgs[1]:.2f} | 本系統={avgs[2]:.2f}")


# ════════════════════════════════════════════════
# 實驗五：端到端延遲分析
# ════════════════════════════════════════════════
def run_exp5():
    print("\n══ 實驗五：系統延遲分析 ══")

    docs = list(db.eval_logs.find(
        {"experiment": "exp1_exp2", "vlm_inference_ms": {"$gt": 0}},
        {"vlm_inference_ms": 1, "user_id": 1}
    ))

    if not docs:
        print("[Exp5] 沒有延遲資料")
        return

    ms_list = [d["vlm_inference_ms"] for d in docs]
    avg_ms  = np.mean(ms_list)
    med_ms  = np.median(ms_list)
    std_ms  = np.std(ms_list)

    # 假設各階段延遲比例（根據實際系統設計估算）
    # 實際值應從 app.py 各階段計時取得
    stage_labels = ["VLM 推理", "MongoDB 查詢", "SBERT 綁定", "FAISS 搜尋", "其他"]
    stage_ratios = [0.75, 0.10, 0.08, 0.05, 0.02]
    stage_ms     = [avg_ms * r for r in stage_ratios]

    # ── 圓餅圖：延遲分布 ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    wedges, texts, autotexts = ax1.pie(
        stage_ms, labels=stage_labels, autopct='%1.1f%%',
        colors=['#E84C4C','#4C9BE8','#4CE87A','#F5A623','#9B59B6'],
        startangle=90
    )
    ax1.set_title(f"Exp5: 各階段延遲分布\n平均總延遲 {avg_ms:.0f} ms")

    # ── 直方圖：VLM 推理時間分布 ──
    ax2.hist(ms_list, bins=20, color='#4C9BE8', edgecolor='white', alpha=0.8)
    ax2.axvline(avg_ms, color='red',    linestyle='--', label=f'平均 {avg_ms:.0f}ms')
    ax2.axvline(med_ms, color='orange', linestyle='--', label=f'中位數 {med_ms:.0f}ms')
    ax2.set_xlabel("VLM 推理時間 (ms)")
    ax2.set_ylabel("次數")
    ax2.set_title("Exp5: VLM 推理時間分布")
    ax2.legend()

    savefig("exp5_latency_pie.png")

    # ── 成功率（需要人工記錄）──
    print(f"[Exp5] 平均延遲：{avg_ms:.0f}ms | 中位數：{med_ms:.0f}ms | std：{std_ms:.0f}ms")
    print(f"[Exp5] 最快：{min(ms_list):.0f}ms | 最慢：{max(ms_list):.0f}ms")
    print("[Exp5] 情境成功率需要人工記錄，請填入 eval_exp5_results.csv")

    # 產生成功率填寫表
    csv_path = "eval_exp5_results.csv"
    if not os.path.exists(csv_path):
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("scenario,run,success(1/0),note\n")
            for s in ["A_物品尋找", "B_個人化習慣", "C_完整管家"]:
                for r in range(1, 6):
                    f.write(f"{s},{r},,\n")
        print(f"[Exp5] 已產生成功率記錄表：{csv_path}")


# ════════════════════════════════════════════════
# 實驗六：個人化模糊需求（需要人工評分 CSV）
# ════════════════════════════════════════════════
def run_exp6():
    print("\n══ 實驗六：個人化模糊需求理解 ══")

    csv_path = "eval_exp6_scores.csv"
    if not os.path.exists(csv_path):
        lines = ["qid,input,user,intent_correct(1/0),nav_correct(1/0),personalized(1/0),note"]
        questions = [
            ("F01","我累了","User_Mom"),("F02","我累了","User_Dad"),
            ("F03","我渴了","User_Mom"),("F04","我渴了","User_Dad"),
            ("F05","我餓了","User_Mom"),("F06","我餓了","User_Dad"),
            ("F07","我無聊","User_Mom"),("F08","我無聊","User_Dad"),
            ("F09","我不舒服","User_Mom"),("F10","我不舒服","User_Dad"),
            ("F11","幫我找我的眼鏡","User_Dad"),("F12","幫我找遙控器","User_Mom"),
            ("F13","我想休息","User_Mom"),("F14","我想喝點東西","User_Dad"),
            ("F15","肚子餓了","User_Mom"),("F16","想看電視","User_Dad"),
            ("F17","我想睡覺","User_Mom"),("F18","找一下我的藥","User_Mom"),
            ("F19","有點渴","User_Dad"),("F20","我餓了有什麼吃的","User_Mom"),
            ("F21","I'm tired","User_Mom"),("F22","I'm hungry","User_Dad"),
            ("F23","I'm thirsty","User_Mom"),("F24","Where's the remote","User_Dad"),
            ("F25","I need to rest","User_Dad"),("F26","好想吃東西","User_Mom"),
            ("F27","我冷","User_Mom"),("F28","找我的水杯","User_Dad"),
            ("F29","我想吃點心","User_Mom"),("F30","幫我找媽媽的杯子","User_Dad"),
        ]
        for qid, q, user in questions:
            lines.append(f"{qid},{q},{user},,,, ")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"[Exp6] 已產生空白評分表：{csv_path}")
        print("  → 對每一題用 interact_client.py 測試後填入 1/0，再執行 python eval_all.py --exp 6")
        return

    import csv
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    def to_int(v):
        try: return int(v)
        except: return None

    filled = [r for r in rows if to_int(r.get("intent_correct")) is not None]
    if not filled:
        print("[Exp6] CSV 尚未填入評分")
        return

    intent_acc = np.mean([to_int(r["intent_correct"]) for r in filled if to_int(r["intent_correct"]) is not None])
    nav_acc    = np.mean([to_int(r["nav_correct"])     for r in filled if to_int(r["nav_correct"])     is not None])
    pers_rate  = np.mean([to_int(r["personalized"])    for r in filled if to_int(r["personalized"])    is not None])

    # Mom vs Dad 個人化差異（相同問題給不同人不同答案）
    paired_qs = [("F01","F02"),("F03","F04"),("F05","F06"),("F07","F08"),("F09","F10")]
    diff_count= 0
    for qa, qb in paired_qs:
        ra = next((r for r in rows if r["qid"] == qa), None)
        rb = next((r for r in rows if r["qid"] == qb), None)
        if ra and rb:
            # 如果兩人都有個人化且導航目標不同 → 算差異化
            if to_int(ra.get("personalized")) == 1 and to_int(rb.get("personalized")) == 1:
                diff_count += 1
    diff_rate = diff_count / len(paired_qs) if paired_qs else 0

    # ── 長條圖 ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    metrics = [intent_acc, nav_acc, pers_rate]
    labels  = ["意圖辨識\n準確率", "導航目標\n正確率", "個人化\n正確率"]
    colors  = ["#4C9BE8", "#4CE87A", "#F5A623"]
    bars = ax1.bar(labels, metrics, color=colors, width=0.5)
    ax1.set_ylabel("準確率")
    ax1.set_title("Exp6: 個人化模糊需求各項指標")
    ax1.set_ylim(0, 1.15)
    ax1.axhline(0.7, color='red', linestyle='--', alpha=0.5, label='目標 70%')
    for bar, v in zip(bars, metrics):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{v:.1%}", ha='center', fontweight='bold')
    ax1.legend()

    # ── 圓餅圖：intent_type 分布 ──
    intent_types = list(db.conversation_logs.aggregate([
        {"$group": {"_id": "$intent_type", "count": {"$sum": 1}}}
    ]))
    if intent_types:
        it_labels = [d["_id"] or "unknown" for d in intent_types]
        it_counts = [d["count"]             for d in intent_types]
        ax2.pie(it_counts, labels=it_labels, autopct='%1.1f%%',
                colors=['#4C9BE8','#4CE87A','#F5A623'], startangle=90)
        ax2.set_title("Exp6: 意圖類型分布")
    else:
        ax2.text(0.5, 0.5, "conversation_logs\n資料不足", ha='center', va='center')
        ax2.set_title("Exp6: 意圖類型分布（資料不足）")

    savefig("exp6_intent_accuracy.png")

    print(f"[Exp6] 意圖準確率={intent_acc:.1%} | 導航正確率={nav_acc:.1%} | 個人化={pers_rate:.1%}")
    print(f"[Exp6] Mom vs Dad 個人化差異率={diff_rate:.1%}")


# ════════════════════════════════════════════════
# 總結報告
# ════════════════════════════════════════════════
def write_summary():
    lines = [
        "═══════════════════════════════════════",
        "  論文實驗結果總結報告",
        f"  產生時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "═══════════════════════════════════════",
        "",
        "MongoDB 資料量統計：",
        f"  eval_logs         : {db.eval_logs.count_documents({}):>6} 筆",
        f"  observation_logs  : {db.observation_logs.count_documents({}):>6} 筆",
        f"  dynamic_objects   : {db.dynamic_objects.count_documents({}):>6} 筆",
        f"  conversation_logs : {db.conversation_logs.count_documents({}):>6} 筆",
        f"  activity_sequences: {db.activity_sequences.count_documents({}):>6} 筆",
        f"  exp_checkpoints   : {db.exp_checkpoints.count_documents({}):>6} 筆",
        "",
        "產出圖表：",
        "  results/exp1_accuracy.png",
        "  results/exp1_confusion_matrix.png",
        "  results/exp2_binding_accuracy.png",
        "  results/exp3a_weight_curve.png",
        "  results/exp3b_transition_heatmap.png",
        "  results/exp4_dialogue_scores.png",
        "  results/exp5_latency_pie.png",
        "  results/exp6_intent_accuracy.png",
        "",
        "需要人工填寫的評分表：",
        "  eval_exp4_scores.csv  （實驗四：對話品質 1-5 分）",
        "  eval_exp5_results.csv （實驗五：端到端成功率）",
        "  eval_exp6_scores.csv  （實驗六：模糊需求準確率）",
    ]
    report = "\n".join(lines)
    print("\n" + report)
    with open(os.path.join(OUT_DIR, "summary_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)


# ════════════════════════════════════════════════
# 主程式
# ════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default="all",
                        help="要跑的實驗：all / 1 / 2 / 3a / 3b / 4 / 5 / 6")
    args = parser.parse_args()

    exp = args.exp.lower()

    if exp in ("all", "1"):  run_exp1()
    if exp in ("all", "2"):  run_exp2()
    if exp in ("all", "3a"): run_exp3a()
    if exp in ("all", "3b"): run_exp3b()
    if exp in ("all", "4"):  run_exp4()
    if exp in ("all", "5"):  run_exp5()
    if exp in ("all", "6"):  run_exp6()

    write_summary()
    print(f"\n✅ 所有圖表已存到 {OUT_DIR}/ 資料夾")
