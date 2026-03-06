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
        self.url       = ollama_url
        self.model     = model_name
        self.face_app  = face_analyzer
        self.face_bank = face_bank
        self.spatial   = spatial_module

        self.client = MongoClient(mongo_uri)
        self.db     = self.client[db_name]
        self.semantic_memories_collection = self.db["semantic_memories"]

    # ─────────────────────────────────────────────
    # 人臉辨識（None = 定點相機模式）
    # ─────────────────────────────────────────────
    def _get_user_id(self, img_b64, hint_user_id="Unknown_User"):
        if not self.face_app or not self.face_bank:
            return hint_user_id
        try:
            encoded_data = img_b64.split(',')[1] if ',' in img_b64 else img_b64
            nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
            img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            faces = self.face_app.get(img)
            if not faces:
                return hint_user_id
            face = sorted(faces,
                          key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]),
                          reverse=True)[0]
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
    # 主入口
    # ─────────────────────────────────────────────
    def analyze_action_burst(self, payload: dict):
        image_list    = payload.get("image_list", [])
        hint_user_id  = payload.get("userID", "Unknown_User")
        source_nodes  = payload.get("source_nodes", [])
        node_scores   = payload.get("node_scores", [])

        if not image_list:
            return self._empty_result(hint_user_id)

        sample_indices = self._select_sample_indices(image_list, node_scores, max_samples=3)

        # ── 強化版 Prompt：要求輸出空間關係 + 物品互動分層 ──
        prompt = """Analyze this home scene image carefully. Respond ONLY in valid JSON, no extra text.

{
  "action": "single verb describing what the person is doing (e.g. drinking, cooking, typing)",
  "main_object": "the primary furniture or area the person is at (e.g. kitchen_counter, sofa, desk)",
  "interacting_items": ["items the person is directly touching or using"],
  "scene_items": ["other visible items NOT being directly used by the person"],
  "spatial_relations": [
    {"subject": "item_name", "relation": "on/in/next_to/above/below/in_hand_of/on_top_of", "object": "furniture_or_person_name"}
  ],
  "description": "one natural sentence summarizing the scene"
}

Rules:
1. interacting_items: ONLY items the person is physically holding or directly using.
2. scene_items: everything else visible in the scene (on surfaces, shelves, background).
3. spatial_relations: describe WHERE each item is relative to furniture or the person.
   - Use "in_hand_of" when person is holding an item.
   - Use "on" for items resting on surfaces.
   - Use "in" for items inside containers/drawers.
   - Use "next_to" for items beside furniture.
4. Do NOT include placeholder names like "item1". Be specific.
5. If no items are visible, return empty lists [].
"""

        user_votes, action_votes, object_votes = [], [], []
        interacting_pool, scene_pool = [], []
        spatial_pool, descriptions   = [], []

        for idx in sample_indices:
            try:
                img_b64   = image_list[idx]
                uid       = self._get_user_id(img_b64, hint_user_id)
                user_votes.append(uid)

                img_clean   = img_b64.split(',')[1] if ',' in img_b64 else img_b64
                api_payload = {
                    "model":   self.model,
                    "messages": [{"role": "user", "content": prompt, "images": [img_clean]}],
                    "stream":  False,
                    "options": {"temperature": 0.1, "num_predict": 256}
                }

                response   = requests.post(f"{self.url}/api/chat", json=api_payload, timeout=120)
                raw_result = response.json().get("message", {}).get("content", "").strip()
                node_name  = source_nodes[idx] if idx < len(source_nodes) else f"node_{idx}"
                print(f"✅ [Frame {idx} | {node_name}] VLM raw: {raw_result}")

                clean_json = re.sub(r'```json\n?|```', '', raw_result).strip()

                try:
                    data = json.loads(clean_json)

                    act        = data.get("action", "none").lower()
                    obj        = data.get("main_object", "unknown").lower()
                    interact   = data.get("interacting_items", [])
                    scene      = data.get("scene_items", [])
                    spatial    = data.get("spatial_relations", [])
                    desc       = data.get("description", "")

                    if act in ["none", "describe the verb", "unknown"]:
                        continue

                    action_votes.append(act)
                    object_votes.append(obj)
                    descriptions.append(desc)

                    # 清洗物品清單
                    blacklist = {"item1", "item2", "none", "small_items", "unknown"}

                    if isinstance(interact, list):
                        interacting_pool.extend([
                            i.lower() for i in interact
                            if i.lower() not in blacklist
                        ])

                    if isinstance(scene, list):
                        scene_pool.extend([
                            i.lower() for i in scene
                            if i.lower() not in blacklist
                        ])

                    # 清洗空間關係
                    if isinstance(spatial, list):
                        for rel in spatial:
                            if (isinstance(rel, dict)
                                    and rel.get("subject")
                                    and rel.get("relation")
                                    and rel.get("object")):
                                spatial_pool.append({
                                    "subject":  rel["subject"].lower(),
                                    "relation": rel["relation"].lower(),
                                    "object":   rel["object"].lower()
                                })

                except Exception as je:
                    print(f"   ⚠️ JSON Parse Error Frame {idx}: {je}")
                    continue

            except Exception as e:
                print(f"❌ [Frame {idx}] Perception Failed: {e}")

        # ── 投票 ──
        final_user   = max(set(user_votes), key=user_votes.count) if user_votes else hint_user_id
        final_items  = list(set(interacting_pool))   # 用戶直接互動的物品
        all_items    = list(set(interacting_pool + scene_pool))  # 畫面中所有物品

        if not action_votes:
            return self._empty_result(final_user)

        final_action = max(set(action_votes), key=action_votes.count)
        final_object = max(set(object_votes), key=object_votes.count)

        try:
            base_desc = descriptions[action_votes.index(final_action)]
        except:
            base_desc = "Observed behavior."

        # 空間關係去重聚合（同一對 subject-relation-object 合併）
        spatial_merged = self._merge_spatial_relations(spatial_pool)

        final_result = {
            "location":          final_object,
            "interacting_items": final_items,     # 人正在用的
            "scene_items":       list(set(scene_pool)),  # 畫面中其他物品
            "all_items":         all_items,       # 全部
            "spatial_relations": spatial_merged,  # 空間介係詞關係
            "context":           base_desc
        }

        # 寫入語義記憶
        self.semantic_memories_collection.insert_one({
            "user":              final_user,
            "action":            final_action,
            "bound_to":          "Unknown_Area",
            "details":           final_result,
            "source_nodes":      source_nodes,
            "timestamp":         datetime.datetime.utcnow()
        })

        return {
            "user":           final_user,
            "action":         final_action,
            "result":         final_result,
            "items":          final_items,        # 互動物品（主要）
            "all_items":      all_items,          # 全部物品
            "spatial":        spatial_merged,     # 空間關係
            "bound_instance": final_object
        }

    # ─────────────────────────────────────────────
    # 空間關係去重：相同 subject-relation-object 只保留一筆
    # ─────────────────────────────────────────────
    def _merge_spatial_relations(self, spatial_pool):
        seen   = {}
        merged = []
        for rel in spatial_pool:
            key = f"{rel['subject']}|{rel['relation']}|{rel['object']}"
            if key not in seen:
                seen[key] = True
                merged.append(rel)
        return merged

    # ─────────────────────────────────────────────
    # 依 node_scores 選取 sample frames
    # ─────────────────────────────────────────────
    def _select_sample_indices(self, image_list, node_scores, max_samples=3):
        n = len(image_list)
        if n == 0:
            return []
        if n <= max_samples:
            return list(range(n))
        if node_scores and len(node_scores) == n:
            sorted_idx = sorted(range(n), key=lambda i: node_scores[i], reverse=True)
            return sorted_idx[:max_samples]
        step = n / max_samples
        return [int(i * step) for i in range(max_samples)]

    def _empty_result(self, user_id):
        return {
            "user": user_id, "action": "none",
            "result": {}, "items": [], "all_items": [],
            "spatial": [], "bound_instance": "Unknown_Area"
        }