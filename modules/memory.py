import datetime
import numpy as np
from config import Config


class MemoryManager:
    WINDOW_SIZE = 3

    def __init__(self, client, embedding_model=None):
        self.db        = client[Config.DB_NAME]
        self.scene     = self.db["scene_snapshots"]
        self.logs      = self.db["observation_logs"]
        self.sequences = self.db["activity_sequences"]
        self.dynamics  = self.db["dynamic_objects"]     # ← 新增
        self.model     = embedding_model

    # ─────────────────────────────────────────────
    # 場景同步
    # ─────────────────────────────────────────────
    def sync_scene(self, objects):
        if not objects:
            return 0
        count = 0
        for obj in objects:
            try:
                location_2d = [float(obj.get('x', 0)), float(obj.get('z', 0))]
                self.scene.update_one(
                    {"id": obj['id']},
                    {"$set": {
                        "label":        obj.get('label', 'unknown').lower(),
                        "pos":          location_2d,
                        "y_height":     obj.get('y', 0),
                        "room":         obj.get('room', "Unknown"),
                        "last_updated": datetime.datetime.now()
                    }},
                    upsert=True
                )
                count += 1
            except Exception as e:
                print(f"⚠️ [Sync Error] {obj.get('id')}: {e}")
        print(f"✅ [Memory] Synced {count} scene objects.")
        return count

    # ─────────────────────────────────────────────
    # 核心：語義綁定 + 物品雙軌 + 空間關係記錄
    # ─────────────────────────────────────────────
    def bind_and_update(self, user_id, action, est_pos,
                        vlm_description="",
                        detected_items=None,
                        all_items=None,
                        spatial_relations=None,
                        max_distance=4.0,
                        target_label=None,
                        room_name=""):

        detected_items    = detected_items    or []
        all_items         = all_items         or []
        spatial_relations = spatial_relations or []

        # scene_items = 畫面中有但人沒有直接用的
        scene_items = [i for i in all_items if i not in detected_items]

        instance_id    = "Unknown_ID"
        instance_label = target_label if target_label else "Unknown_Area"
        instance_pos   = None

        try:
            # ── A. 家具綁定 ──
            if not target_label or target_label in ["Unknown_Area", "unknown"]:
                instance_id, instance_label, instance_pos = \
                    self._bind_by_position_and_semantics(
                        est_pos, target_label, max_distance,
                        room_name=room_name
                    )
            else:
                matched = self.scene.find_one({
                    "label": {"$regex": target_label, "$options": "i"},
                    **({"room": {"$regex": room_name, "$options": "i"}} if room_name else {})
                })
                if not matched and room_name:
                    matched = self.scene.find_one(
                        {"label": {"$regex": target_label, "$options": "i"}}
                    )
                if matched:
                    instance_id    = matched['id']
                    instance_label = matched['label']
                    instance_pos   = matched.get('pos')
                else:
                    instance_id, instance_label, instance_pos = \
                        self._bind_by_semantics_only(target_label, room_name=room_name)

            # ── B-1. 物品跟著家具（scene_snapshots）──
            if instance_id != "Unknown_ID":
                update_ops = {
                    "$set": {
                        "last_observation": datetime.datetime.now(),
                        "current_contents": all_items
                    }
                }
                if all_items:
                    update_ops["$addToSet"] = {"items": {"$each": all_items}}

                if spatial_relations:
                    for rel in spatial_relations:
                        rel_key = f"{rel['subject']}|{rel['relation']}|{rel['object']}"
                        self.scene.update_one(
                            {"id": instance_id},
                            {
                                "$inc":      {f"spatial_counts.{rel_key}": 1},
                                "$addToSet": {"spatial_relations": rel}
                            }
                        )

                self.scene.update_one({"id": instance_id}, update_ops)
                print(f"[Inventory] {instance_label} ← all:{all_items} | interacting:{detected_items}")

        except Exception as e:
            print(f"❌ [Bind Error] {e}")

        try:
            # ── B-2. 物品跟著用戶（observation_logs）──
            self.logs.update_one(
                {"user": user_id, "instance": instance_label, "action": action},
                {
                    "$inc": {"weight": 1},
                    "$set": {
                        "last_seen":    datetime.datetime.now(),
                        "raw_vlm_desc": vlm_description,
                        "pos":          instance_pos
                    },
                    "$addToSet": {
                        "interacting_items": {"$each": detected_items}
                    }
                },
                upsert=True
            )
            if spatial_relations:
                self.logs.update_one(
                    {"user": user_id, "instance": instance_label, "action": action},
                    {"$addToSet": {"observed_relations": {"$each": spatial_relations}}}
                )
        except Exception as e:
            print(f"❌ [Log Error] {e}")

        # ── B-3. 動態物件更新（dynamic_objects）──
        try:
            self._update_dynamic_objects(
                user_id          = user_id,
                interacting_items= detected_items,
                scene_items      = scene_items,
                spatial_relations= spatial_relations,
                bound_label      = instance_label,
                furniture_pos    = instance_pos,
                room_name        = room_name
            )
        except Exception as e:
            print(f"❌ [Dynamic Error] {e}")

        # ── D. 行為時間序列 ──
        try:
            self._update_activity_sequence(
                user_id=user_id,
                action=action,
                instance=instance_label,
                items=detected_items
            )
        except Exception as e:
            print(f"❌ [Sequence Error] {e}")

        return instance_label

    # ─────────────────────────────────────────────
    # B-3. dynamic_objects upsert
    # ─────────────────────────────────────────────
    def _update_dynamic_objects(self, user_id, interacting_items, scene_items,
                                 spatial_relations, bound_label, furniture_pos, room_name):
        """
        - interacting_items: seen_count+1, interact_count+1, interacted_by addToSet
        - scene_items:       seen_count+1 only
        唯一鍵：label（last seen 覆蓋位置）
        """
        now = datetime.datetime.now()

        # spatial_rel lookup：item → relation（從 spatial_relations 反查）
        item_rel_map = {}
        for rel in spatial_relations:
            subj = rel.get("subject", "").lower().strip()
            if subj:
                item_rel_map[subj] = rel.get("relation", "on")

        def _upsert_item(label: str, is_interacting: bool):
            label = label.lower().strip()
            if not label:
                return

            spatial_rel = item_rel_map.get(label, "near")

            base_set = {
                "last_seen_on":  bound_label,
                "spatial_rel":   spatial_rel,
                "room":          room_name,
                "last_seen":     now,
            }
            if furniture_pos:
                base_set["furniture_pos"] = furniture_pos

            inc_ops = {"seen_count": 1}
            if is_interacting:
                inc_ops["interact_count"] = 1

            update = {
                "$inc":        inc_ops,
                "$set":        base_set,
                "$setOnInsert":{"first_seen": now},
            }
            if is_interacting:
                update["$addToSet"] = {"interacted_by": user_id}

            self.dynamics.update_one(
                {"label": label},
                update,
                upsert=True
            )
            print(f"   🧩 [Dynamic] '{label}' "
                  f"{'interacting' if is_interacting else 'scene'} "
                  f"@ {bound_label} ({spatial_rel})")

        for item in interacting_items:
            _upsert_item(item, is_interacting=True)

        for item in scene_items:
            _upsert_item(item, is_interacting=False)

    # ─────────────────────────────────────────────
    # D. activity_sequences
    # ─────────────────────────────────────────────
    def _update_activity_sequence(self, user_id, action, instance, items):
        now       = datetime.datetime.now()
        today_str = now.strftime("%Y-%m-%d")

        new_entry = {
            "action":    action,
            "instance":  instance,
            "items":     items,
            "time":      now.strftime("%H:%M:%S"),
            "timestamp": now
        }

        doc = self.sequences.find_one({"user_id": user_id, "date": today_str})

        if doc:
            seq = doc.get("sequence", [])
            if seq:
                last      = seq[-1]
                last_time = datetime.datetime.strptime(
                    f"{today_str} {last['time']}", "%Y-%m-%d %H:%M:%S"
                )
                gap_minutes = round((now - last_time).total_seconds() / 60, 1)
                transition  = {
                    "from":          last["action"],
                    "from_instance": last["instance"],
                    "to":            action,
                    "to_instance":   instance,
                    "gap_minutes":   gap_minutes
                }
                self.sequences.update_one(
                    {"user_id": user_id, "date": today_str},
                    {
                        "$push": {"sequence": new_entry, "transitions": transition},
                        "$set":  {"last_updated": now}
                    }
                )
            else:
                self.sequences.update_one(
                    {"user_id": user_id, "date": today_str},
                    {"$push": {"sequence": new_entry}, "$set": {"last_updated": now}}
                )
        else:
            self.sequences.insert_one({
                "user_id":      user_id,
                "date":         today_str,
                "sequence":     [new_entry],
                "transitions":  [],
                "last_updated": now
            })

        print(f"[Sequence] {user_id} → {action}@{instance} ({today_str})")

    # ─────────────────────────────────────────────
    # A. 距離搜尋 + SBERT 語義驗證
    # ─────────────────────────────────────────────
    def _bind_by_position_and_semantics(self, est_pos, vlm_label, max_distance,
                                         room_name=""):
        if not est_pos:
            return "Unknown_ID", "Unknown_Area", None

        target_pt = [float(est_pos.get('x', 0)), float(est_pos.get('z', 0))]

        if room_name:
            candidates = list(self.scene.find(
                {"room": {"$regex": room_name, "$options": "i"}}
            ))
            if not candidates:
                print(f"   ⚠️ [Bind] room '{room_name}' 無家具，fallback 全部")
                candidates = list(self.scene.find({}))
        else:
            candidates = list(self.scene.find({}))

        nearby = []
        for c in candidates:
            c_pos = c.get('pos', [0, 0])
            dist  = np.linalg.norm(np.array(c_pos) - np.array(target_pt))
            if dist <= max_distance:
                nearby.append((c, dist))

        if not nearby:
            all_sorted = sorted(
                [(c, np.linalg.norm(np.array(c.get('pos', [0,0])) - np.array(target_pt)))
                 for c in candidates],
                key=lambda x: x[1]
            )
            if all_sorted:
                best = all_sorted[0][0]
                return best['id'], best['label'], best.get('pos')
            return "Unknown_ID", "Unknown_Area", None

        if len(nearby) == 1:
            best = nearby[0][0]
            return best['id'], best['label'], best.get('pos')

        if self.model and vlm_label and vlm_label not in ["Unknown_Area", "unknown", None]:
            best_item = self._semantic_rerank(vlm_label, nearby)
        else:
            nearby.sort(key=lambda x: x[1])
            best_item = nearby[0][0]

        return best_item['id'], best_item['label'], best_item.get('pos')

    def _bind_by_semantics_only(self, target_label, room_name=""):
        if not self.model:
            return "Unknown_ID", "Unknown_Area", None

        if room_name:
            candidates = list(self.scene.find(
                {"room": {"$regex": room_name, "$options": "i"}}
            ))
            if not candidates:
                candidates = list(self.scene.find({}))
        else:
            candidates = list(self.scene.find({}))

        if not candidates:
            return "Unknown_ID", "Unknown_Area", None

        query_vec  = self.model.encode(target_label)
        best_score = -1
        best_item  = None

        for c in candidates:
            label_vec = self.model.encode(c['label'])
            score     = float(np.dot(query_vec, label_vec) /
                              (np.linalg.norm(query_vec) * np.linalg.norm(label_vec) + 1e-8))
            if score > best_score:
                best_score = score
                best_item  = c

        if best_item and best_score > 0.4:
            print(f"[SBERT] '{target_label}' → '{best_item['label']}' ({best_score:.2f})")
            return best_item['id'], best_item['label'], best_item.get('pos')

        return "Unknown_ID", "Unknown_Area", None

    def _semantic_rerank(self, vlm_label, nearby_list):
        """語義相似度 × 0.6 + 距離分數 × 0.4"""
        max_dist   = max(d for _, d in nearby_list) + 1e-8
        query_vec  = self.model.encode(vlm_label)
        best_score = -1
        best_item  = nearby_list[0][0]

        for item, dist in nearby_list:
            label_vec    = self.model.encode(item['label'])
            semantic_sim = float(np.dot(query_vec, label_vec) /
                                 (np.linalg.norm(query_vec) * np.linalg.norm(label_vec) + 1e-8))
            dist_score   = 1.0 - (dist / max_dist)
            final_score  = semantic_sim * 0.6 + dist_score * 0.4

            print(f"[Rerank] {item['label']}: sem={semantic_sim:.2f} "
                  f"dist={dist:.1f}m → {final_score:.2f}")

            if final_score > best_score:
                best_score = final_score
                best_item  = item

        return best_item