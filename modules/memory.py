import datetime
import numpy as np
from config import Config


class MemoryManager:
    WINDOW_SIZE = 3  # 滑動視窗大小（用於行為預測訓練資料）

    def __init__(self, client, embedding_model=None):
        self.db        = client[Config.DB_NAME]
        self.scene     = self.db["scene_snapshots"]
        self.logs      = self.db["observation_logs"]
        self.sequences = self.db["activity_sequences"]  # 行為時間序列
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
                        detected_items=None,    # 互動物品（人直接使用的）
                        all_items=None,         # 畫面中所有物品
                        spatial_relations=None, # 空間介係詞關係
                        max_distance=4.0,
                        target_label=None):

        detected_items    = detected_items    or []
        all_items         = all_items         or []
        spatial_relations = spatial_relations or []

        instance_id    = "Unknown_ID"
        instance_label = target_label if target_label else "Unknown_Area"
        instance_pos   = None

        try:
            # ── A. 家具綁定：距離搜尋 + SBERT 語義驗證 ──
            if not target_label or target_label in ["Unknown_Area", "unknown"]:
                instance_id, instance_label, instance_pos = \
                    self._bind_by_position_and_semantics(est_pos, target_label, max_distance)
            else:
                matched = self.scene.find_one({"label": target_label})
                if matched:
                    instance_id    = matched['id']
                    instance_label = matched['label']
                    instance_pos   = matched.get('pos')
                else:
                    instance_id, instance_label, instance_pos = \
                        self._bind_by_semantics_only(target_label)

            # ── B-1. 物品跟著家具 ──
            if instance_id != "Unknown_ID":
                update_ops = {
                    "$set": {
                        "last_observation": datetime.datetime.now(),
                        # current_contents：當前鏡頭看到的物品（覆蓋）
                        # 定點相機沒看到 → 空清單 → 代表東西不在了
                        "current_contents": all_items
                    }
                }
                # items：歷史累積清單（只增不減，用於習慣學習）
                if all_items:
                    update_ops["$addToSet"] = {
                        "items": {"$each": all_items}
                    }

                # 空間關係：累積記錄 subject 在這個家具附近的位置關係
                if spatial_relations:
                    for rel in spatial_relations:
                        # 每個關係累計出現次數
                        rel_key = f"{rel['subject']}|{rel['relation']}|{rel['object']}"
                        self.scene.update_one(
                            {"id": instance_id},
                            {
                                "$inc": {f"spatial_counts.{rel_key}": 1},
                                "$addToSet": {"spatial_relations": rel}
                            }
                        )

                self.scene.update_one({"id": instance_id}, update_ops)
                print(f"[Inventory] {instance_label} ← all:{all_items} | interacting:{detected_items}")

        except Exception as e:
            print(f"❌ [Bind Error] {e}")

        try:
            # ── B-2. 物品跟著用戶（行為日誌）──
            self.logs.update_one(
                {"user": user_id, "instance": instance_label, "action": action},
                {
                    "$inc": {"weight": 1},
                    "$set": {
                        "last_seen":    datetime.datetime.now(),
                        "raw_vlm_desc": vlm_description,
                        "pos":          instance_pos
                    },
                    # 互動物品：這個人在這個動作中直接使用的
                    "$addToSet": {
                        "interacting_items": {"$each": detected_items}
                    }
                },
                upsert=True
            )

            # 空間關係也記錄到行為日誌（用於查詢「媽媽喝水時杯子在哪」）
            if spatial_relations:
                self.logs.update_one(
                    {"user": user_id, "instance": instance_label, "action": action},
                    {"$addToSet": {"observed_relations": {"$each": spatial_relations}}}
                )

        except Exception as e:
            print(f"❌ [Log Error] {e}")

        # ── D. 即時更新行為時間序列 ──
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
    # D. 即時更新 activity_sequences
    # ─────────────────────────────────────────────
    def _update_activity_sequence(self, user_id, action, instance, items):
        """
        每次感知後即時 append 到當天的序列。
        同時記錄與上一筆的 transition（行為轉換）。
        """
        now       = datetime.datetime.now()
        today_str = now.strftime("%Y-%m-%d")

        new_entry = {
            "action":    action,
            "instance":  instance,
            "items":     items,
            "time":      now.strftime("%H:%M:%S"),
            "timestamp": now
        }

        # 取當天序列（若不存在則自動建立）
        doc = self.sequences.find_one({"user_id": user_id, "date": today_str})

        if doc:
            seq = doc.get("sequence", [])

            # 計算 transition
            if seq:
                last        = seq[-1]
                last_time   = datetime.datetime.strptime(
                    f"{today_str} {last['time']}", "%Y-%m-%d %H:%M:%S"
                )
                gap_minutes = round((now - last_time).total_seconds() / 60, 1)
                transition  = {
                    "from":         last["action"],
                    "from_instance":last["instance"],
                    "to":           action,
                    "to_instance":  instance,
                    "gap_minutes":  gap_minutes
                }
                self.sequences.update_one(
                    {"user_id": user_id, "date": today_str},
                    {
                        "$push":  {"sequence": new_entry, "transitions": transition},
                        "$set":   {"last_updated": now}
                    }
                )
            else:
                self.sequences.update_one(
                    {"user_id": user_id, "date": today_str},
                    {"$push": {"sequence": new_entry}, "$set": {"last_updated": now}}
                )
        else:
            # 建立新的當天序列
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
    def _bind_by_position_and_semantics(self, est_pos, vlm_label, max_distance):
        if not est_pos:
            return "Unknown_ID", "Unknown_Area", None

        target_pt  = [float(est_pos.get('x', 0)), float(est_pos.get('z', 0))]
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

        # 多個候選：SBERT 重排序
        if self.model and vlm_label and vlm_label not in ["Unknown_Area", "unknown", None]:
            best_item = self._semantic_rerank(vlm_label, nearby)
        else:
            nearby.sort(key=lambda x: x[1])
            best_item = nearby[0][0]

        return best_item['id'], best_item['label'], best_item.get('pos')

    def _bind_by_semantics_only(self, target_label):
        if not self.model:
            return "Unknown_ID", "Unknown_Area", None

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

            print(f"[Rerank] {item['label']}: sem={semantic_sim:.2f} dist={dist:.1f}m → {final_score:.2f}")

            if final_score > best_score:
                best_score = final_score
                best_item  = item

        return best_item