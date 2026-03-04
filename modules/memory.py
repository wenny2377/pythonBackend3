import datetime
import numpy as np
import requests
from config import Config

class MemoryManager:
    def __init__(self, client, embedding_model=None, ollama_url=None):
        self.db = client[Config.DB_NAME]
        self.scene = self.db["scene_snapshots"]
        self.logs = self.db["observation_logs"]
        self.model = embedding_model
        self.ollama_url = ollama_url
        self.expansion_cache = {}

    def sync_scene(self, objects):
        """Synchronize the furniture coordinate list sent from Unity"""
        if not objects:
            return 0
        count = 0
        for obj in objects:
            try:
                location_2d = [float(obj.get('x', 0)), float(obj.get('z', 0))]
                self.scene.update_one(
                    {"id": obj['id']},
                    {"$set": {
                        "label": obj.get('label', 'unknown').lower(),
                        "pos": location_2d,
                        "y_height": obj.get('y', 0),
                        "room": obj.get('room', "Unknown"),
                        "last_updated": datetime.datetime.now()
                    }},
                    upsert=True
                )
                count += 1
            except Exception as e:
                print(f"⚠️ [Sync Item Error] Failed to sync object {obj.get('id')}: {e}")
        print(f"✅ [Memory] Successfully synchronized {count} scene objects.")
        return count

    def _expand_label_with_llm(self, label):
        """Expand label semantics using LLM"""
        if not self.ollama_url:
            return label
        if label in self.expansion_cache:
            return self.expansion_cache[label]

        prompt = f"List 5 descriptors for '{label}'. Output words only."
        try:
            response = requests.post(self.ollama_url, json={
                "model": "gemma3",
                "prompt": prompt,
                "stream": False
            }, timeout=5)
            expanded = response.json().get("response", "").strip().lower()
            full_label = f"{label} {expanded}"
            self.expansion_cache[label] = full_label
            return full_label
        except:
            return label

    def bind_and_update(self, user_id, action, est_pos, vlm_description="", detected_items=None, max_distance=4.0, target_label=None):
        """
        Update memory and align with object
        """
        detected_items = detected_items or []
        if isinstance(detected_items, str):
            detected_items = [detected_items]

        instance_id = "Unknown_ID"
        instance_label = target_label if target_label else "Unknown_Area"

        try:
            # 1. Determine the object ID to bind
            if not target_label or target_label == "Unknown_Area":
                target_pt = [float(est_pos.get('x', 0)), float(est_pos.get('z', 0))]
                candidates = list(self.scene.find({}))
                nearby_candidates = []
                for c in candidates:
                    c_pos = c.get('pos', [0, 0])
                    dist = np.linalg.norm(np.array(c_pos) - np.array(target_pt))
                    if dist <= max_distance:
                        nearby_candidates.append((c, dist))
                
                if nearby_candidates:
                    nearby_candidates.sort(key=lambda x: x[1])
                    instance_id = nearby_candidates[0][0]['id']
                    instance_label = nearby_candidates[0][0]['label']
                else:
                    # Fallback to closest object
                    all_c = sorted(
                        [(c, np.linalg.norm(np.array(c.get('pos', [0,0])) - np.array(target_pt))) 
                         for c in candidates],
                        key=lambda x: x[1]
                    )
                    if all_c:
                        instance_id = all_c[0][0]['id']
                        instance_label = all_c[0][0]['label']
            else:
                matched = self.scene.find_one({"label": target_label})
                if matched:
                    instance_id = matched['id']
                    instance_label = matched['label']

            # 2. --- Core fix: Update furniture contents ---
            if detected_items and instance_id != "Unknown_ID":
                # Correct MongoDB update syntax: specify the field name 'items'
                update_payload = {
                    "$set": {
                        "current_contents": detected_items,
                        "last_observation": datetime.datetime.now()
                    },
                    "$addToSet": {
                        "items": {"$each": detected_items}  # Must explicitly specify the 'items' field here
                    }
                }
                self.scene.update_one({"id": instance_id}, update_payload)
                print(f"[Inventory] Successfully updated item list of {instance_label} (ID: {instance_id}): {detected_items}")

        except Exception as e:
            print(f"❌ [Error] Binding update failed: {e}")

        # 3. --- Update action logs ---
        try:
            self.logs.update_one(
                {"user": user_id, "instance": instance_label, "action": action},
                {
                    "$inc": {"weight": 1},
                    "$set": {
                        "last_seen": datetime.datetime.now(),
                        "raw_vlm_desc": vlm_description
                    }
                },
                upsert=True
            )
        except Exception as e:
            print(f"❌ [Error] Log update failed: {e}")

        return instance_label