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
        self.scene_col = self.db["scene_snapshots"]   # ← 新增，用來查同房間家具

    # ─────────────────────────────────────────────
    # 人臉辨識
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
        image_list   = payload.get("image_list", [])
        hint_user_id = payload.get("userID", "Unknown_User")
        source_nodes = payload.get("source_nodes", [])
        node_scores  = payload.get("node_scores", [])
        room_name    = payload.get("room_name", "a room")   # Unity 傳入的房間名

        if not image_list:
            return self._empty_result(hint_user_id)

        sample_indices = self._select_sample_indices(image_list, node_scores, max_samples=3)

        # ── 從 scene_snapshots 查同房間家具清單，給 VLM 當 context ──
        try:
            room_docs = list(self.scene_col.find(
                {"room": {"$regex": room_name, "$options": "i"}} if room_name else {}
            ))
            room_furniture = [d.get("label", "") for d in room_docs if d.get("label")]
        except Exception:
            room_furniture = []

        furniture_ctx = ""
        if room_furniture:
            furniture_ctx = (
                f"\nFURNITURE in this room: {', '.join(room_furniture)}.\n"
                "Use these names for main_object. Do NOT use furniture from other rooms.\n"
            )

        # ── FIX: 改用 f-string，room_name 正確注入 ──
        prompt = f"""You are analyzing a home camera image inside "{room_name}".
{furniture_ctx}
Respond ONLY in valid JSON. No markdown, no extra text.

{{
  "action": "most specific single verb (sleeping/eating/cooking/typing/sitting/standing/watching/...)",
  "main_object": "primary furniture the person is at",
  "interacting_items": ["items person physically holds or uses — [] if none"],
  "scene_items": ["other visible background items — [] if none"],
  "spatial_relations": [
    {{"subject": "item_or_person", "relation": "on/in/next_to/above/below/in_hand_of/lying_on", "object": "furniture_or_person"}}
  ],
  "description": "one natural sentence"
}}

RULES:
1. action: be specific. "sleeping" not "lying". "eating" not "sitting". "watching" not "sitting".
2. main_object: must come from FURNITURE list above if provided.
3. interacting_items: only items physically held or operated. Empty [] if none.
4. scene_items: background items on surfaces/shelves. Do NOT repeat interacting_items.
5. Use "lying_on" when person is horizontal on bed or sofa.
6. Do NOT invent furniture not in the room list.
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
                    "options": {"temperature": 0.05, "num_predict": 512}
                }

                response   = requests.post(f"{self.url}/api/chat", json=api_payload, timeout=120)
                raw_result = response.json().get("message", {}).get("content", "").strip()
                node_name  = source_nodes[idx] if idx < len(source_nodes) else f"node_{idx}"
                print(f"✅ [Frame {idx} | {node_name}] VLM raw: {raw_result}")

                clean_json = self._extract_json(raw_result)

                try:
                    data = json.loads(clean_json)

                    act      = data.get("action", "none").lower().strip()
                    obj      = data.get("main_object", "unknown").lower().strip()
                    interact = data.get("interacting_items", [])
                    scene    = data.get("scene_items", [])
                    spatial  = data.get("spatial_relations", [])
                    desc     = data.get("description", "")

                    invalid_actions = {
                        "none", "describe the verb", "unknown", "n/a",
                        "not visible", "cannot determine", ""
                    }
                    if act in invalid_actions:
                        print(f"   ⚠️ Skipping invalid action: '{act}'")
                        continue

                    action_votes.append(act)
                    object_votes.append(obj)
                    descriptions.append(desc)

                    blacklist = {"item1", "item2", "none", "small_items", "unknown", "n/a", ""}

                    if isinstance(interact, list):
                        interacting_pool.extend([
                            i.lower().strip() for i in interact
                            if isinstance(i, str) and i.lower().strip() not in blacklist
                        ])
                    if isinstance(scene, list):
                        scene_pool.extend([
                            i.lower().strip() for i in scene
                            if isinstance(i, str) and i.lower().strip() not in blacklist
                        ])
                    if isinstance(spatial, list):
                        for rel in spatial:
                            if (isinstance(rel, dict)
                                    and rel.get("subject")
                                    and rel.get("relation")
                                    and rel.get("object")):
                                spatial_pool.append({
                                    "subject":  rel["subject"].lower().strip(),
                                    "relation": rel["relation"].lower().strip(),
                                    "object":   rel["object"].lower().strip()
                                })

                except Exception as je:
                    print(f"   ⚠️ JSON Parse Error Frame {idx}: {je}")
                    print(f"   Cleaned JSON: {clean_json[:200]}")
                    continue

            except Exception as e:
                print(f"❌ [Frame {idx}] Perception Failed: {e}")

        # ── 投票 ──
        final_user   = max(set(user_votes), key=user_votes.count) if user_votes else hint_user_id
        final_items  = list(set(interacting_pool))
        all_items    = list(set(interacting_pool + scene_pool))

        if not action_votes:
            return self._empty_result(final_user)

        final_action = max(set(action_votes), key=action_votes.count)
        final_object = max(set(object_votes), key=object_votes.count)

        try:
            base_desc = descriptions[action_votes.index(final_action)]
        except Exception:
            base_desc = "Observed behavior."

        spatial_merged = self._merge_spatial_relations(spatial_pool)

        final_result = {
            "location":          final_object,
            "interacting_items": final_items,
            "scene_items":       list(set(scene_pool)),
            "all_items":         all_items,
            "spatial_relations": spatial_merged,
            "context":           base_desc
        }

        self.semantic_memories_collection.insert_one({
            "user":         final_user,
            "action":       final_action,
            "bound_to":     "Unknown_Area",   # 由 MemoryManager.bind_and_update 覆蓋
            "details":      final_result,
            "source_nodes": source_nodes,
            "timestamp":    datetime.datetime.utcnow()
        })

        return {
            "user":           final_user,
            "action":         final_action,
            "result":         final_result,
            "items":          final_items,
            "all_items":      all_items,
            "spatial":        spatial_merged,
            "bound_instance": final_object    # VLM 的原始輸出，MemoryManager 會校正
        }

    # ─────────────────────────────────────────────
    # 工具方法
    # ─────────────────────────────────────────────
    def _extract_json(self, raw: str) -> str:
        cleaned = re.sub(r'```(?:json)?\s*', '', raw).strip()
        match   = re.search(r'\{.*\}', cleaned, re.DOTALL)
        return match.group(0).strip() if match else cleaned

    def _merge_spatial_relations(self, spatial_pool):
        seen, merged = {}, []
        for rel in spatial_pool:
            key = f"{rel['subject']}|{rel['relation']}|{rel['object']}"
            if key not in seen:
                seen[key] = True
                merged.append(rel)
        return merged

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