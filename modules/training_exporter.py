import json
import os
import datetime
from bson import ObjectId
from config import Config

class TrainingExporter:
    def __init__(self, mongo_client, output_dir="training_data"):
        self.db         = mongo_client[Config.DB_NAME]
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def _format_time(self, dt):
        """統一時間格式處理"""
        if isinstance(dt, datetime.datetime):
            return dt.isoformat()
        return str(dt)

    # ─────────────────────────────────────────────
    # A. 感知資料：優化 VLM 描述邏輯
    # ─────────────────────────────────────────────
    def export_perception(self, user_id_filter=None):
        collection = self.db["semantic_memories"]
        query      = {"user": user_id_filter} if user_id_filter else {}
        docs       = list(collection.find(query))

        path = os.path.join(self.output_dir, "perception_data.jsonl")
        count = 0

        with open(path, 'w', encoding='utf-8') as f:
            for doc in docs:
                details  = doc.get("details", {})
                action   = doc.get("action", "performing an activity")
                bound_to = doc.get("bound_to", "an area")
                interact = details.get("interacting_items", [])
                spatial  = details.get("spatial_relations", [])
                
                # 構建更自然的描述
                answer = f"The person is {action} near the {bound_to}."
                if interact:
                    answer += f" They are currently interacting with {', '.join(interact)}."
                
                if spatial:
                    rel_desc = ", ".join([f"a {r['subject']} is {r['relation']} the {r['object']}" for r in spatial])
                    answer += f" In the surroundings, {rel_desc}."

                entry = {
                    "id": str(doc["_id"]),
                    "image": doc.get("image_path", "placeholder.jpg"), # 確保有影像路徑對應
                    "conversations": [
                        {"from": "human", "value": "<image>\nWhat is happening in this scene?"},
                        {"from": "gpt", "value": answer}
                    ],
                    "metadata": {
                        "user": doc.get("user"),
                        "action": action,
                        "location": bound_to,
                        "timestamp": self._format_time(doc.get("timestamp"))
                    }
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                count += 1
        return count

    # ─────────────────────────────────────────────
    # B. 對話資料：強化個人化 Prompt
    # ─────────────────────────────────────────────
    def export_dialogue(self, user_id_filter=None):
        collection = self.db["interaction_logs"]
        query      = {"user_id": user_id_filter} if user_id_filter else {}
        docs       = list(collection.find(query))

        path = os.path.join(self.output_dir, "dialogue_data.jsonl")
        count = 0

        with open(path, 'w', encoding='utf-8') as f:
            for doc in docs:
                # 這裡可以根據 intent_type 加入不同的 System Prompt
                intent = doc.get("intent_type", "general")
                system_msg = "You are a personalized home robot. You remember user habits to provide better help."
                
                entry = {
                    "messages": [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": doc.get("query", "")},
                        {"role": "assistant", "content": doc.get("answer", "")}
                    ],
                    "context": {
                        "intent": intent,
                        "recommended": doc.get("nav_label"),
                        "is_personalized": doc.get("is_personalized", False)
                    }
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                count += 1
        return count

    # ─────────────────────────────────────────────
    # C. 導航資料：標準化座標系
    # ─────────────────────────────────────────────
    def export_navigation(self, user_id_filter=None):
        collection = self.db["navigation_logs"]
        query      = {"user_id": user_id_filter} if user_id_filter else {}
        docs       = list(collection.find(query))

        path = os.path.join(self.output_dir, "navigation_data.jsonl")
        count = 0

        with open(path, 'w', encoding='utf-8') as f:
            for doc in docs:
                # 確保座標為 float 列表
                def clean_pos(p): return [float(p[0]), float(p[1])] if p else None

                entry = {
                    "user": doc.get("user_id"),
                    "task": doc.get("intent"),
                    "path_geometry": {
                        "start": clean_pos(doc.get("start_pos")),
                        "goal": clean_pos(doc.get("goal_pos")),
                        "trajectory": doc.get("waypoints", [])
                    },
                    "performance": {
                        "success": doc.get("success", False),
                        "distance": round(doc.get("total_distance", 0), 2),
                        "time": round(doc.get("total_time", 0), 2)
                    },
                    "timestamp": self._format_time(doc.get("timestamp"))
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                count += 1
        return count

    # ─────────────────────────────────────────────
    # E. 習慣序列：滑動視窗預測 (LSTM/Transformer 格式)
    # ─────────────────────────────────────────────
    def export_habit_sequence(self, user_id_filter=None, window_size=3):
        collection = self.db["activity_sequences"]
        query      = {"user_id": user_id_filter} if user_id_filter else {}
        docs       = list(collection.find(query).sort("date", 1))

        path = os.path.join(self.output_dir, "habit_sequence_data.jsonl")
        count = 0

        

        with open(path, 'w', encoding='utf-8') as f:
            for doc in docs:
                user_id = doc.get("user_id")
                date = doc.get("date")
                seq = doc.get("sequence", [])

                if len(seq) < window_size + 1:
                    continue

                for i in range(len(seq) - window_size):
                    window = seq[i : i + window_size]
                    target = seq[i + window_size]

                    # 格式化訓練用字串
                    # 範例: "sitting@sofa -> drinking@table"
                    history_str = " -> ".join([f"{s['action']}@{s['instance']}" for s in window])
                    
                    entry = {
                        "user": user_id,
                        "history": history_str,
                        "predict_action": target['action'],
                        "predict_location": target['instance'],
                        "time_of_day": target['time'],
                        "metadata": {
                            "date": date,
                            "window_items": [s.get("items", []) for s in window]
                        }
                    }
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    count += 1
        return count

    # ─────────────────────────────────────────────
    # 全量執行
    # ─────────────────────────────────────────────
    def export_all(self, user_id_filter=None):
        return {
            "perception": self.export_perception(user_id_filter),
            "dialogue":   self.export_dialogue(user_id_filter),
            "navigation": self.export_navigation(user_id_filter),
            "habit":      self.export_habit_sequence(user_id_filter)
        }