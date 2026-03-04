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
        self.url = ollama_url
        self.model = model_name
        self.face_app = face_analyzer
        self.face_bank = face_bank
        self.spatial = spatial_module

        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.scene_collection = self.db["scene_snapshots"]
        self.semantic_memories_collection = self.db["semantic_memories"]
        self.observation_logs = self.db["observation_logs"]

    def _get_user_id(self, img_b64):
        if not self.face_app or not self.face_bank:
            return "Unknown_User"
        try:
            encoded_data = img_b64.split(',')[1] if ',' in img_b64 else img_b64
            nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            faces = self.face_app.get(img)
            if not faces: return "Unknown_User"
            face = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)[0]
            emb = face.normed_embedding
            best_name, max_sim = "Unknown_User", 0
            for name, known_emb in self.face_bank.items():
                sim = np.dot(emb, known_emb)
                if sim > max_sim:
                    max_sim, best_name = sim, name
            return best_name if max_sim > 0.40 else "Unknown_User"
        except Exception as e:
            print(f"⚠️ Face ReID Error: {e}")
            return "Unknown_User"

    def analyze_action_burst(self, image_list, robot_pos=None, robot_yaw=None, camera_fov=None):
        if not image_list:
            return {"user": "Unknown_User", "action": "none", "result": {}, "items": [], "bound_instance": "Unknown_Area"}

        user_votes, action_votes, object_votes, item_pool, descriptions = [], [], [], [], []
        candidate_votes = {}
        sample_indices = [0, len(image_list)//2, len(image_list)-1]

        # 修改 Prompt：讓範例標籤更具通用性，避免與真實物品衝突
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
3. Be specific about fruit names if seen (e.g., apple, banana).
"""
        for idx in sample_indices:
            try:
                img_b64 = image_list[idx]
                uid = self._get_user_id(img_b64)
                user_votes.append(uid)

                img_clean = img_b64.split(',')[1] if ',' in img_b64 else img_b64
                payload = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt, "images": [img_clean]}],
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 128}
                }

                response = requests.post(f"{self.url}/api/chat", json=payload, timeout=30)
                raw_result = response.json().get("message", {}).get("content", "").strip()
                
                # 🔍 DEBUG: 查看 VLM 到底說了什麼
                print(f"   [Frame {idx}] VLM Raw: {raw_result}")

                # 清洗 JSON (去除 Markdown 標籤)
                clean_json = re.sub(r'```json\n?|```', '', raw_result)
                
                try:
                    data = json.loads(clean_json)
                    act = data.get("action", "none").lower()
                    obj = data.get("main_object", "unknown").lower()
                    items_list = data.get("small_items", [])
                    desc = data.get("description", "")
                    
                    # 修正過濾邏輯：只過濾掉明顯的 placeholder 字眼，不要過濾 apple
                    if act not in ["none", "describe the verb", "unknown"]:
                        action_votes.append(act)
                        object_votes.append(obj)
                        descriptions.append(desc)
                        if isinstance(items_list, list):
                            # 過濾掉範例中的佔位符
                            filtered_items = [i.lower() for i in items_list if i.lower() not in ["item1", "item2", "none", "small_items"]]
                            item_pool.extend(filtered_items)
                except Exception as je:
                    print(f"   ⚠️ JSON Parse Error at Frame {idx}: {je}")
                    continue

                # 空間座標投票 (保持不變)
                if self.spatial and robot_pos is not None:
                    nparr = np.frombuffer(base64.b64decode(img_clean), np.uint8)
                    frame_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    est_pos = self.spatial.estimate_coordinate(frame_cv, robot_pos, robot_yaw, camera_fov)
                    
                    nearby = list(self.scene_collection.find({
                        "pos": {"$near": est_pos, "$maxDistance": 5.0}
                    }).limit(3))
                    for c in nearby:
                        candidate_votes[c['label']] = candidate_votes.get(c['label'], 0) + 1

            except Exception as e:
                print(f"[Frame {idx}] Perception Failed: {e}")

        # -----------------------------
        # 🗳️ 投票與結果處理
        # -----------------------------
        # 即使沒有 action，也嘗試回傳 user 和 items
        final_user = max(set(user_votes), key=user_votes.count) if user_votes else "Unknown_User"
        final_items = list(set(item_pool)) # apple 和 banana 應該會出現在這
        
        if not action_votes:
            return {
                "user": final_user, 
                "action": "none", 
                "result": {"location": "unknown", "detected_items": final_items, "context": "No action detected"}, 
                "items": final_items, 
                "bound_instance": "Unknown_Area"
            }

        final_action = max(set(action_votes), key=action_votes.count)
        final_object = max(set(object_votes), key=object_votes.count)
        
        try:
            winner_idx = action_votes.index(final_action)
            base_desc = descriptions[winner_idx]
        except:
            base_desc = "Observed behavior."

        bound_instance_id = max(candidate_votes, key=candidate_votes.get) if candidate_votes else "Unknown_Area"

        # -----------------------------
        # 💾 儲存與回傳
        # -----------------------------
        final_result = {"location": final_object, "detected_items": final_items, "context": base_desc}
        
        self.semantic_memories_collection.insert_one({
            "user": final_user,
            "action": final_action,
            "bound_to": bound_instance_id,
            "details": final_result,
            "timestamp": datetime.datetime.utcnow()
        })

        return {
            "user": final_user,
            "action": final_action,
            "result": final_result,
            "items": final_items,
            "bound_instance": bound_instance_id
        }