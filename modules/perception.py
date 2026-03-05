import requests
import re
import cv2
import numpy as np
import base64
import json
from pymongo import MongoClient
import datetime


class PerceptionEngine:
    def __init__(self, ollama_url, model_name, face_analyzer=None, face_bank=None,
                 mongo_uri="mongodb://127.0.0.1:27017/", db_name="robot_rag_db",
                 spatial_module=None):
        self.url        = ollama_url
        self.model      = model_name
        self.face_app   = face_analyzer   # None = 定點相機模式，直接用 userID
        self.face_bank  = face_bank
        self.spatial    = spatial_module  # None = 定點相機模式，直接用 user_pos

        self.client = MongoClient(mongo_uri)
        self.db     = self.client[db_name]
        self.semantic_memories_collection = self.db["semantic_memories"]

    # ─────────────────────────────────────────────
    # 人臉辨識（face_app 為 None 時直接 return hint）
    # ─────────────────────────────────────────────
    def _get_user_id(self, img_b64, hint_user_id="Unknown_User"):
        if not self.face_app or not self.face_bank:
            return hint_user_id  # 定點相機模式：直接用 Unity 傳來的 userID

        try:
            encoded_data = img_b64.split(',')[1] if ',' in img_b64 else img_b64
            nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
            img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            faces = self.face_app.get(img)
            if not faces:
                return hint_user_id

            face = sorted(
                faces,
                key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]),
                reverse=True
            )[0]
            emb = face.normed_embedding
            best_name, max_sim = hint_user_id, 0
            for name, known_emb in self.face_bank.items():
                sim = np.dot(emb, known_emb)
                if sim > max_sim:
                    max_sim, best_name = sim, name
            return best_name if max_sim > 0.40 else hint_user_id
        except Exception as e:
            print(f"⚠️ Face ReID Error: {e}")
            return hint_user_id

    # ─────────────────────────────────────────────
    # 主要分析入口：接收完整 MultiImagePayload dict
    # ─────────────────────────────────────────────
    def analyze_action_burst(self, payload: dict):
        """
        接收來自 Unity MultiImagePayload 的完整 dict：
        {
            "image_list":    [...],   # Base64 JPEG 陣列
            "userID":        "...",   # Unity 已知用戶 ID
            "activity":      "...",   # 行為 hint
            "user_pos":      {"x":..., "y":..., "z":...},
            "source_nodes":  [...],   # 各影像對應相機節點
            "node_scores":   [...],   # 各節點評分
            "image_count":   N,
            "timestamp":     "...",
            "robot_pos":     null,    # 定點模式不填
            "robot_rotation_y": 0,
            "camera_fov":    0
        }
        """
        image_list    = payload.get("image_list", [])
        hint_user_id  = payload.get("userID", "Unknown_User")
        activity_hint = payload.get("activity", "")
        source_nodes  = payload.get("source_nodes", [])
        node_scores   = payload.get("node_scores", [])

        if not image_list:
            return {
                "user": hint_user_id, "action": "none",
                "result": {}, "items": [], "bound_instance": "Unknown_Area"
            }

        # 依 node_scores 選取最多 3 張最優影像
        sample_indices = self._select_sample_indices(image_list, node_scores, max_samples=3)

        prompt = """
Analyze the image and respond ONLY in valid JSON format:
{
  "action": "Describe the verb",
  "main_object": "Major furniture/area",
  "small_items": ["item1", "item2"],
  "description": "Short natural sentence"
}
Rules:
1. If no small items are seen, return [].
2. Do not use 'item1' or 'item2' in your list.
3. Be specific about item names (e.g., apple, cup, laptop).
"""

        user_votes, action_votes, object_votes = [], [], []
        item_pool, descriptions = [], []

        for idx in sample_indices:
            try:
                img_b64 = image_list[idx]

                # 人臉辨識或直接用 hint
                uid = self._get_user_id(img_b64, hint_user_id)
                user_votes.append(uid)

                img_clean = img_b64.split(',')[1] if ',' in img_b64 else img_b64
                api_payload = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt, "images": [img_clean]}],
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 128}
                }

                response   = requests.post(f"{self.url}/api/chat", json=api_payload, timeout=120)
                raw_result = response.json().get("message", {}).get("content", "").strip()

                node_name = source_nodes[idx] if idx < len(source_nodes) else f"node_{idx}"
                print(f"✅ [Frame {idx} | {node_name}] VLM: {raw_result}")

                clean_json = re.sub(r'```json\n?|```', '', raw_result)

                try:
                    data      = json.loads(clean_json)
                    act       = data.get("action", "none").lower()
                    obj       = data.get("main_object", "unknown").lower()
                    items_raw = data.get("small_items", [])
                    desc      = data.get("description", "")

                    if act not in ["none", "describe the verb", "unknown"]:
                        action_votes.append(act)
                        object_votes.append(obj)
                        descriptions.append(desc)
                        if isinstance(items_raw, list):
                            filtered = [
                                i.lower() for i in items_raw
                                if i.lower() not in ["item1", "item2", "none", "small_items"]
                            ]
                            item_pool.extend(filtered)
                except Exception as je:
                    print(f"   ⚠️ JSON Parse Error Frame {idx}: {je}")
                    continue

            except Exception as e:
                print(f"❌ [Frame {idx}] Perception Failed: {e}")

        # ─────────────────────────────────────────────
        # 投票決定最終結果
        # ─────────────────────────────────────────────
        final_user  = max(set(user_votes), key=user_votes.count) if user_votes else hint_user_id
        final_items = list(set(item_pool))

        if not action_votes:
            return {
                "user":           final_user,
                "action":         "none",
                "result":         {"location": "unknown", "detected_items": final_items, "context": "No action detected"},
                "items":          final_items,
                "bound_instance": "Unknown_Area"
            }

        final_action = max(set(action_votes), key=action_votes.count)
        final_object = max(set(object_votes), key=object_votes.count)

        try:
            base_desc = descriptions[action_votes.index(final_action)]
        except:
            base_desc = "Observed behavior."

        final_result = {
            "location":      final_object,
            "detected_items": final_items,
            "context":       base_desc
        }

        # 寫入語義記憶
        self.semantic_memories_collection.insert_one({
            "user":         final_user,
            "action":       final_action,
            "bound_to":     "Unknown_Area",  # bind_and_update 會再更新
            "details":      final_result,
            "source_nodes": source_nodes,
            "timestamp":    datetime.datetime.utcnow()
        })

        return {
            "user":           final_user,
            "action":         final_action,
            "result":         final_result,
            "items":          final_items,
            "bound_instance": final_object   # 用 VLM 辨識的家具名稱做初始綁定
        }

    # ─────────────────────────────────────────────
    # 依 node_scores 加權選取 sample frames
    # ─────────────────────────────────────────────
    def _select_sample_indices(self, image_list, node_scores, max_samples=3):
        n = len(image_list)
        if n == 0:
            return []
        if n <= max_samples:
            return list(range(n))

        if node_scores and len(node_scores) == n:
            # 依分數由高到低排序，取前 max_samples 個
            sorted_idx = sorted(range(n), key=lambda i: node_scores[i], reverse=True)
            return sorted_idx[:max_samples]

        # 沒有分數時均勻取樣
        step = n / max_samples
        return [int(i * step) for i in range(max_samples)]