import json
import os
import datetime
from pymongo import MongoClient
from config import Config


class TrainingExporter:
    """
    統一訓練資料匯出器
    從 MongoDB 各 collection 匯出 JSONL 格式訓練資料

    輸出類型：
    A. perception_data.jsonl    → VLM fine-tuning（影像 + 標注）
    B. dialogue_data.jsonl      → 對話模型訓練（意圖 + 回應對）
    C. navigation_data.jsonl    → 路徑規劃訓練（起點 + 終點 + 完整路徑）
    D. scene_graph_data.jsonl   → 場景理解（空間關係圖）
    E. habit_sequence_data.jsonl→ 用戶行為序列（習慣預測）
    """

    def __init__(self, mongo_client, output_dir="training_data"):
        self.db         = mongo_client[Config.DB_NAME]
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ─────────────────────────────────────────────
    # 全量匯出
    # ─────────────────────────────────────────────
    def export_all(self, user_id_filter=None):
        stats = {}
        stats["perception"]     = self.export_perception(user_id_filter)
        stats["dialogue"]       = self.export_dialogue(user_id_filter)
        stats["navigation"]     = self.export_navigation(user_id_filter)
        stats["scene_graph"]    = self.export_scene_graph()
        stats["habit_sequence"] = self.export_habit_sequence(user_id_filter)
        return stats

    # ─────────────────────────────────────────────
    # A. 感知資料（semantic_memories → VLM fine-tuning）
    # ─────────────────────────────────────────────
    def export_perception(self, user_id_filter=None):
        """
        格式：LLaVA instruction fine-tuning 標準格式
        {
          "id": "...",
          "conversations": [
            {"from": "human", "value": "<image>\nWhat is the person doing?"},
            {"from": "gpt",   "value": "The person is drinking water near the kitchen counter..."}
          ],
          "metadata": { "user", "action", "bound_to", "spatial_relations" }
        }
        """
        collection = self.db["semantic_memories"]
        query      = {"user": user_id_filter} if user_id_filter else {}
        docs       = list(collection.find(query))

        path  = os.path.join(self.output_dir, "perception_data.jsonl")
        count = 0

        with open(path, 'w', encoding='utf-8') as f:
            for doc in docs:
                details  = doc.get("details", {})
                action   = doc.get("action", "")
                bound_to = doc.get("bound_to", "")
                interact = details.get("interacting_items", [])
                spatial  = details.get("spatial_relations", [])
                context  = details.get("context", "")

                # 組合完整標注答案
                spatial_desc = "; ".join(
                    [f"{r['subject']} is {r['relation']} {r['object']}" for r in spatial]
                ) if spatial else ""

                answer = context
                if interact:
                    answer += f" The person is using: {', '.join(interact)}."
                if spatial_desc:
                    answer += f" Spatial context: {spatial_desc}."

                entry = {
                    "id": str(doc["_id"]),
                    "conversations": [
                        {
                            "from":  "human",
                            "value": "<image>\nDescribe what the person is doing and the spatial arrangement of objects in the scene."
                        },
                        {
                            "from":  "gpt",
                            "value": answer
                        }
                    ],
                    "metadata": {
                        "user":              doc.get("user"),
                        "action":            action,
                        "bound_to":          bound_to,
                        "interacting_items": interact,
                        "spatial_relations": spatial,
                        "source_nodes":      doc.get("source_nodes", []),
                        "timestamp":         str(doc.get("timestamp"))
                    }
                }

                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                count += 1

        print(f"✅ [Export] perception_data.jsonl → {count} 筆")
        return count

    # ─────────────────────────────────────────────
    # B. 對話資料（interaction_logs → 對話模型訓練）
    # ─────────────────────────────────────────────
    def export_dialogue(self, user_id_filter=None):
        """
        格式：OpenAI chat fine-tuning 格式
        {
          "messages": [
            {"role": "system",    "content": "You are a home robot assistant..."},
            {"role": "user",      "content": "我累了"},
            {"role": "assistant", "content": "媽媽，您通常喜歡在沙發休息，我帶您過去。"}
          ],
          "metadata": { intent, need_type, nav_target, habit_used }
        }
        """
        collection = self.db["interaction_logs"]
        query      = {"user_id": user_id_filter} if user_id_filter else {}
        docs       = list(collection.find(query))

        path  = os.path.join(self.output_dir, "dialogue_data.jsonl")
        count = 0

        system_prompt = (
            "You are a personalized home robot assistant. "
            "You know the habits and preferences of each household member. "
            "Respond in a warm, helpful manner and provide navigation guidance when needed."
        )

        with open(path, 'w', encoding='utf-8') as f:
            for doc in docs:
                entry = {
                    "messages": [
                        {"role": "system",    "content": system_prompt},
                        {"role": "user",      "content": doc.get("query", "")},
                        {"role": "assistant", "content": doc.get("answer", "")}
                    ],
                    "metadata": {
                        "user_id":        doc.get("user_id"),
                        "intent":         doc.get("intent"),
                        "need_type":      doc.get("need_type"),
                        "nav_target":     doc.get("nav_target"),
                        "target_label":   doc.get("target_label"),
                        "habit_used":     doc.get("habit_used"),
                        "habit_instance": doc.get("habit_instance"),
                        "room":           doc.get("room"),
                        "timestamp":      str(doc.get("timestamp"))
                    }
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                count += 1

        print(f"✅ [Export] dialogue_data.jsonl → {count} 筆")
        return count

    # ─────────────────────────────────────────────
    # C. 導航資料（navigation_logs → 路徑規劃訓練）
    # ─────────────────────────────────────────────
    def export_navigation(self, user_id_filter=None):
        """
        格式：
        {
          "user_id":        "User_Mom",
          "intent":         "tired",
          "start":          [0.0, 0.0],
          "goal":           [2.5, 3.1],
          "waypoints":      [[x,z,t], [x,z,t], ...],
          "waypoint_count": 24,
          "success":        true,
          "total_time":     8.3,
          "total_distance": 4.2,
          "timestamp":      "..."
        }
        """
        collection = self.db["navigation_logs"]
        query      = {"user_id": user_id_filter} if user_id_filter else {}
        docs       = list(collection.find(query))

        path  = os.path.join(self.output_dir, "navigation_data.jsonl")
        count = 0

        with open(path, 'w', encoding='utf-8') as f:
            for doc in docs:
                entry = {
                    "user_id":        doc.get("user_id"),
                    "intent":         doc.get("intent"),
                    "start":          doc.get("start_pos"),
                    "goal":           doc.get("goal_pos"),
                    "waypoints":      doc.get("waypoints", []),
                    "waypoint_count": doc.get("waypoint_count", 0),
                    "success":        doc.get("success", False),
                    "fail_reason":    doc.get("fail_reason", ""),
                    "total_time":     doc.get("total_time", 0),
                    "total_distance": doc.get("total_distance", 0),
                    "timestamp":      str(doc.get("timestamp"))
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                count += 1

        print(f"✅ [Export] navigation_data.jsonl → {count} 筆")
        return count

    # ─────────────────────────────────────────────
    # D. 場景圖（scene_snapshots → 空間理解訓練）
    # ─────────────────────────────────────────────
    def export_scene_graph(self):
        """
        格式：場景圖 QA 對
        {
          "question": "What items are on the kitchen counter?",
          "answer":   "apple, cup, cutting board",
          "graph": [
            {"subject": "apple", "relation": "on", "object": "kitchen_counter"},
            ...
          ],
          "furniture": "kitchen_counter",
          "pos": [3.2, 1.5]
        }
        """
        docs  = list(self.db["scene_snapshots"].find({}))
        path  = os.path.join(self.output_dir, "scene_graph_data.jsonl")
        count = 0

        with open(path, 'w', encoding='utf-8') as f:
            for doc in docs:
                label    = doc.get("label", "")
                items    = doc.get("items", [])
                spatial  = doc.get("spatial_relations", [])
                pos      = doc.get("pos")

                if not label:
                    continue

                # 物品清單 QA
                if items:
                    entry = {
                        "question":  f"What items are associated with the {label}?",
                        "answer":    ", ".join(items),
                        "graph":     spatial,
                        "furniture": label,
                        "room":      doc.get("room", ""),
                        "pos":       pos,
                        "type":      "item_query"
                    }
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    count += 1

                # 空間關係 QA
                for rel in spatial:
                    entry = {
                        "question":  f"Where is the {rel.get('subject')}?",
                        "answer":    f"The {rel.get('subject')} is {rel.get('relation')} the {rel.get('object')}.",
                        "graph":     [rel],
                        "furniture": label,
                        "pos":       pos,
                        "type":      "spatial_query"
                    }
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    count += 1

        print(f"✅ [Export] scene_graph_data.jsonl → {count} 筆")
        return count

    # ─────────────────────────────────────────────
    # E. 習慣序列（activity_sequences → 行為預測）
    # 滑動視窗 window=3：前 3 個行為預測下一個
    # ─────────────────────────────────────────────
    def export_habit_sequence(self, user_id_filter=None, window_size=3):
        """
        從 activity_sequences 用滑動視窗產生預測訓練資料。

        輸出兩種格式：

        1. 滑動視窗預測格式（給序列模型 / LSTM / Transformer）
        {
          "user_id":       "User_Mom",
          "date":          "2025-01-01",
          "context": [
            {"action": "cooking",  "instance": "stove",           "items": ["knife"],  "time": "12:00"},
            {"action": "drinking", "instance": "kitchen_counter", "items": ["cup"],    "time": "12:30"},
            {"action": "sitting",  "instance": "dining_table",    "items": [],         "time": "12:35"}
          ],
          "next_action":    "sleeping",
          "next_instance":  "bed",
          "time_gap":       45.0,
          "context_str":    "cooking@stove → drinking@kitchen_counter → sitting@dining_table",
          "type":           "sequence_prediction"
        }

        2. 轉換頻率摘要（給習慣推薦）
        {
          "user_id":     "User_Mom",
          "from_action": "cooking",
          "to_action":   "drinking",
          "count":       8,
          "avg_gap_min": 15.3,
          "type":        "transition_summary"
        }
        """
        collection = self.db["activity_sequences"]
        query      = {"user_id": user_id_filter} if user_id_filter else {}
        docs       = list(collection.find(query).sort("date", 1))

        path  = os.path.join(self.output_dir, "habit_sequence_data.jsonl")
        count = 0

        # 轉換頻率統計
        transition_counter = {}

        with open(path, 'w', encoding='utf-8') as f:
            for doc in docs:
                user_id   = doc.get("user_id")
                date      = doc.get("date")
                sequence  = doc.get("sequence", [])
                transitions = doc.get("transitions", [])

                # ── 滑動視窗預測 ──
                # 需要至少 window_size + 1 個行為
                if len(sequence) >= window_size + 1:
                    for i in range(len(sequence) - window_size):
                        context  = sequence[i : i + window_size]
                        next_act = sequence[i + window_size]

                        # 計算最後一個 context 到 next 的時間差
                        try:
                            last_time = datetime.datetime.strptime(
                                f"{date} {context[-1]['time']}", "%Y-%m-%d %H:%M:%S"
                            )
                            next_time = datetime.datetime.strptime(
                                f"{date} {next_act['time']}", "%Y-%m-%d %H:%M:%S"
                            )
                            gap = round((next_time - last_time).total_seconds() / 60, 1)
                        except Exception:
                            gap = 0.0

                        context_str = " → ".join(
                            [f"{s['action']}@{s['instance']}" for s in context]
                        )

                        entry = {
                            "user_id":      user_id,
                            "date":         date,
                            "context":      [
                                {
                                    "action":   s.get("action"),
                                    "instance": s.get("instance"),
                                    "items":    s.get("items", []),
                                    "time":     s.get("time")
                                }
                                for s in context
                            ],
                            "next_action":   next_act.get("action"),
                            "next_instance": next_act.get("instance"),
                            "next_items":    next_act.get("items", []),
                            "time_gap":      gap,
                            "context_str":   context_str,
                            "type":          "sequence_prediction"
                        }
                        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        count += 1

                # ── 轉換頻率統計 ──
                for t in transitions:
                    key = f"{user_id}|{t.get('from')}|{t.get('to')}"
                    if key not in transition_counter:
                        transition_counter[key] = {
                            "user_id":    user_id,
                            "from_action":t.get("from"),
                            "from_instance": t.get("from_instance"),
                            "to_action":  t.get("to"),
                            "to_instance":t.get("to_instance"),
                            "count":      0,
                            "gap_sum":    0.0
                        }
                    transition_counter[key]["count"]   += 1
                    transition_counter[key]["gap_sum"] += t.get("gap_minutes", 0)

            # 寫入轉換頻率摘要
            for key, tc in transition_counter.items():
                avg_gap = round(tc["gap_sum"] / tc["count"], 1) if tc["count"] > 0 else 0
                entry = {
                    "user_id":        tc["user_id"],
                    "from_action":    tc["from_action"],
                    "from_instance":  tc["from_instance"],
                    "to_action":      tc["to_action"],
                    "to_instance":    tc["to_instance"],
                    "count":          tc["count"],
                    "avg_gap_min":    avg_gap,
                    "type":           "transition_summary"
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                count += 1

        print(f"✅ [Export] habit_sequence_data.jsonl → {count} 筆（window={window_size}）")
        return count