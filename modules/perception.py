"""
PerceptionEngine v5
四大效能優化：
  1. Room Embedding Cache    → 事件驅動初始化，切房間才重建 SBERT 向量
  2. Change Streams Sync     → MongoDB watch() 被動訂閱，平時只讀記憶體
  3. Top-K Semantic Filter   → 語意 Top-3 + 距離決策，提高綁定容錯率
  4. Diff & Async Bulk Write → 狀態比對攔截無效更新，累積 20 筆或 30 秒批量寫入

動態物件綁定優先順序：
  1. relation == in_hand → 跟人走（bound_label）
  2. VLM 說物件在某家具上 → Top-K 語意過濾 + 距離決策
  3. fallback → bound_label
"""

import re
import json
import math
import time
import datetime
import threading
import requests
import base64

import cv2
import numpy as np
import faiss

from pymongo import MongoClient, ReturnDocument, UpdateOne
from sentence_transformers import SentenceTransformer


# ══════════════════════════════════════════════
#  常數
# ══════════════════════════════════════════════
STRUCTURAL_BLACKLIST = {
    "wall", "floor", "ceiling", "wooden floor", "white wall",
    "white ceiling", "window", "door", "ground", "white box",
    "concrete floor", "tile floor", "carpet", "baseboard"
}

LABEL_NORMALIZE_MAP = {
    "remote control": "remote",
    "tv remote":      "remote",
    "television":     "tv",
    "laptop":         "computer",
    "notebook":       "computer",
    "cell phone":     "phone",
    "mobile phone":   "phone",
    "smartphone":     "phone",
    "drinking glass": "cup",
    "water glass":    "cup",
    "mug":            "cup",
}

BULK_WRITE_THRESHOLD = 20    # 累積幾筆才批量寫入
BULK_WRITE_INTERVAL  = 30.0  # 或超過幾秒強制寫入


# ══════════════════════════════════════════════
#  1. Room Embedding Cache
#  事件驅動：切房間才重建 SBERT 向量，避免每次都 encode
# ══════════════════════════════════════════════
class RoomEmbeddingCache:
    """
    每個房間維護一個向量快取。
    切房間時呼叫 switch_room()，之後 bind() 只做純矩陣乘法（微秒級）。
    """
    def __init__(self, sbert_model: SentenceTransformer):
        self.model        = sbert_model
        self._room        = None
        self._labels      = []
        self._docs        = []
        self._embeddings  = None   # np.ndarray (N, D)，已 normalize

    def switch_room(self, room_name: str, scene_col):
        """切換房間時呼叫，重建該房間的向量索引"""
        if room_name == self._room and self._embeddings is not None:
            return  # 同一個房間不重建

        q = {"$or": [
            {"room":      {"$regex": room_name, "$options": "i"}},
            {"room_name": {"$regex": room_name, "$options": "i"}},
        ]} if room_name else {}

        docs = list(scene_col.find(q))
        if not docs:
            docs = list(scene_col.find({}))

        self._room   = room_name
        self._docs   = docs
        self._labels = [
            f"{d.get('label', '')} in {d.get('room', d.get('room_name', ''))}"
            for d in docs
        ]

        if self._labels:
            embs = self.model.encode(
                self._labels, normalize_embeddings=True, show_progress_bar=False
            )
            self._embeddings = embs.astype("float32")
        else:
            self._embeddings = None

        print(f"   🏠 [RoomCache] Room '{room_name}' → {len(self._labels)} furniture cached")

    def bind_topk(self, vlm_label: str, k: int = 3, threshold: float = 0.40):
        """
        回傳語意最接近的 Top-K 家具 [(doc, score), ...]
        純矩陣乘法，微秒級
        """
        if self._embeddings is None or not self._labels:
            return []

        q_emb = self.model.encode(
            [vlm_label], normalize_embeddings=True
        )[0].astype("float32")

        sims     = self._embeddings @ q_emb
        top_idx  = np.argsort(sims)[::-1][:k]
        results  = []
        for i in top_idx:
            score = float(sims[i])
            if score >= threshold:
                results.append((self._docs[i], score))
        return results

    @property
    def all_docs(self):
        return self._docs

    @property
    def current_room(self):
        return self._room


# ══════════════════════════════════════════════
#  2. Change Streams Sync
#  MongoDB watch() 被動訂閱，平時只讀記憶體 local_scene_map
# ══════════════════════════════════════════════
class ChangeStreamSync:
    """
    背景執行緒監聽 scene_snapshots 的變更。
    平時 PerceptionEngine 只讀 local_scene_map（記憶體）。
    同事更新 DB 時，MongoDB 主動通知，才同步更新記憶體。
    """
    def __init__(self, scene_col, room_cache: RoomEmbeddingCache):
        self.scene_col  = scene_col
        self.room_cache = room_cache
        self._map       = {}      # label → doc
        self._lock      = threading.Lock()
        self._thread    = None
        self._running   = False
        self._load_all()

    def _load_all(self):
        """啟動時全量載入"""
        docs = list(self.scene_col.find({}))
        with self._lock:
            self._map = {d.get("label", ""): d for d in docs}
        print(f"   📦 [ChangeSync] Loaded {len(self._map)} scene objects into memory")

    def start(self):
        """
        啟動背景同步執行緒。
        優先嘗試 Change Stream（需 replica set）。
        開發環境自動 fallback 到 Polling（每 10 秒輪詢）。
        """
        self._running = True
        self._thread  = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def _watch_loop(self):
        # 先嘗試 Change Stream
        try:
            with self.scene_col.watch(full_document="updateLookup") as stream:
                print("   ✅ [ChangeSync] Change Stream 模式")
                for change in stream:
                    if not self._running:
                        break
                    op  = change.get("operationType")
                    doc = change.get("fullDocument")
                    if doc and op in ("insert", "update", "replace"):
                        label = doc.get("label", "")
                        with self._lock:
                            self._map[label] = doc
                        room = doc.get("room", doc.get("room_name", ""))
                        if self.room_cache.current_room and \
                           self.room_cache.current_room.lower() in room.lower():
                            self.room_cache.switch_room(
                                self.room_cache.current_room, self.scene_col
                            )
                        print(f"   🔄 [ChangeSync] Updated '{label}'")
                    elif op == "delete":
                        key = change.get("documentKey", {}).get("label", "")
                        with self._lock:
                            self._map.pop(key, None)
        except Exception:
            # Fallback：Polling 模式（開發環境 MongoDB 無 replica set）
            print("   ℹ️  [ChangeSync] Polling 模式（每 10 秒）")
            self._poll_loop()

    def _poll_loop(self):
        """每 10 秒重新載入 scene_snapshots 到記憶體"""
        import time
        while self._running:
            try:
                docs = list(self.scene_col.find({}))
                with self._lock:
                    new_map = {d.get("label", ""): d for d in docs}
                    changed = set(new_map) - set(self._map) | \
                              {k for k in new_map if new_map[k] != self._map.get(k)}
                    self._map = new_map
                if changed:
                    print(f"   🔄 [ChangeSync] Polled {len(docs)} docs, changed={changed}")
                    # 觸發 Room Cache 重建
                    if self.room_cache.current_room:
                        self.room_cache.switch_room(
                            self.room_cache.current_room, self.scene_col
                        )
            except Exception as e:
                print(f"   ⚠️ [ChangeSync] Poll error: {e}")
            time.sleep(10)
    def get(self, label: str):
        with self._lock:
            return self._map.get(label)

    def find_by_room(self, room_name: str):
        with self._lock:
            return [
                d for d in self._map.values()
                if room_name.lower() in (
                    d.get("room", "") + d.get("room_name", "")
                ).lower()
            ]

    def all_docs(self):
        with self._lock:
            return list(self._map.values())


# ══════════════════════════════════════════════
#  4. Diff & Async Bulk Write
#  狀態比對 + 批量寫入，減少 DB I/O
# ══════════════════════════════════════════════
class BulkWriteBuffer:
    """
    維護 last_state 做 Diff，無變化就攔截。
    累積 BULK_WRITE_THRESHOLD 筆或超過 BULK_WRITE_INTERVAL 秒才批量寫入。
    """
    def __init__(self, dynamics_col):
        self.col        = dynamics_col
        self._last      = {}      # label → (last_seen_on, spatial_rel, room)
        self._pending   = []      # List[UpdateOne]
        self._last_flush= time.time()
        self._lock      = threading.Lock()

    def upsert(self, label: str, update_op: dict, now: datetime.datetime):
        """
        先比對狀態，無變化就只累加計數（不寫 DB）。
        有變化才加入 pending queue。
        """
        key     = label
        new_state = (
            update_op.get("$set", {}).get("last_seen_on", ""),
            update_op.get("$set", {}).get("spatial_rel", ""),
            update_op.get("$set", {}).get("room", ""),
        )

        with self._lock:
            old_state = self._last.get(key)
            if old_state == new_state:
                # 無變化：只累加計數，不寫 DB
                return False

            # 有變化：記錄新狀態，加入 pending
            self._last[key] = new_state
            self._pending.append(UpdateOne({"label": label}, update_op, upsert=True))

            # 判斷是否觸發批量寫入
            elapsed = time.time() - self._last_flush
            should_flush = (
                len(self._pending) >= BULK_WRITE_THRESHOLD or
                elapsed >= BULK_WRITE_INTERVAL
            )

        if should_flush:
            self._flush()
        return True

    def _flush(self):
        with self._lock:
            if not self._pending:
                return
            ops  = self._pending.copy()
            self._pending.clear()
            self._last_flush = time.time()

        try:
            result = self.col.bulk_write(ops, ordered=False)
            print(f"   💾 [BulkWrite] Flushed {len(ops)} ops "
                  f"(upserted={result.upserted_count}, modified={result.modified_count})")
        except Exception as e:
            print(f"   ❌ [BulkWrite] Failed: {e}")

    def force_flush(self):
        """程式結束或實驗結束時強制寫入"""
        self._flush()

    @property
    def pending_count(self):
        with self._lock:
            return len(self._pending)


# ══════════════════════════════════════════════
#  FAISS Memory Store
# ══════════════════════════════════════════════
class FAISSMemoryStore:
    def __init__(self, sbert_model: SentenceTransformer, dim: int = 384):
        self.model    = sbert_model
        self.dim      = dim
        self.index    = faiss.IndexFlatIP(dim)
        self.metadata = []

    def build_memory_text(self, user, action, instance,
                          interacting_items, all_items, spatial_relations) -> str:
        parts = [f"{user} {action} near {instance}"]
        if interacting_items:
            parts[0] += f" with {', '.join(interacting_items)}"
        parts[0] += "."
        for rel in spatial_relations:
            s = rel.get("subject", ""); r = rel.get("relation", ""); o = rel.get("object", "")
            if s and r and o:
                parts.append(f"{s} {r} {o}.")
        bg = [i for i in all_items if i not in interacting_items]
        if bg:
            parts.append(f"Visible: {', '.join(bg)}.")
        return " ".join(parts)

    def add(self, memory_text: str, metadata: dict):
        emb = self.model.encode(
            [memory_text], normalize_embeddings=True
        )[0].astype("float32")
        self.index.add(np.array([emb]))
        self.metadata.append({**metadata, "memory_text": memory_text})

    def search(self, query: str, k: int = 5):
        if self.index.ntotal == 0:
            return []
        q_emb = self.model.encode(
            [query], normalize_embeddings=True
        )[0].astype("float32")
        scores, indices = self.index.search(np.array([q_emb]), k)
        return [
            {"score": float(s), **self.metadata[i]}
            for s, i in zip(scores[0], indices[0]) if i >= 0
        ]


# ══════════════════════════════════════════════
#  PerceptionEngine v5
# ══════════════════════════════════════════════
class PerceptionEngine:

    def __init__(self, ollama_url: str, model_name: str,
                 face_analyzer=None, face_bank=None,
                 mongo_uri: str = "mongodb://127.0.0.1:27017/",
                 db_name:   str = "robot_rag_db",
                 sbert_model_name: str = "all-MiniLM-L6-v2"):

        self.url       = ollama_url
        self.model     = model_name
        self.face_app  = face_analyzer
        self.face_bank = face_bank

        # MongoDB
        self.client       = MongoClient(mongo_uri)
        self.db           = self.client[db_name]
        self.col_scene    = self.db["scene_snapshots"]
        self.col_obs      = self.db["observation_logs"]
        self.col_memory   = self.db["semantic_memories"]
        self.col_activity = self.db["activity_sequences"]
        self.col_dynamics = self.db["dynamic_objects"]

        # SBERT
        self.sbert = SentenceTransformer(sbert_model_name)

        # 1. Room Embedding Cache
        self.room_cache = RoomEmbeddingCache(self.sbert)

        # 2. Change Streams Sync
        self.scene_sync = ChangeStreamSync(self.col_scene, self.room_cache)
        self.scene_sync.start()

        # 4. Bulk Write Buffer
        self.bulk_buffer = BulkWriteBuffer(self.col_dynamics)

        # FAISS
        self.faiss_store = FAISSMemoryStore(self.sbert)

    # ─────────────────────────────────────────────
    # 人臉辨識
    # ─────────────────────────────────────────────
    def _get_user_id(self, img_b64: str, hint: str = "Unknown_User") -> str:
        if not self.face_app or not self.face_bank:
            return hint
        try:
            raw   = img_b64.split(',')[1] if ',' in img_b64 else img_b64
            nparr = np.frombuffer(base64.b64decode(raw), np.uint8)
            img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            faces = self.face_app.get(img)
            if not faces:
                return hint
            face = sorted(faces,
                          key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]),
                          reverse=True)[0]
            emb = face.normed_embedding
            best, max_sim = hint, 0.0
            for name, known in self.face_bank.items():
                sim = float(np.dot(emb, known))
                if sim > max_sim:
                    max_sim, best = sim, name
            return best if max_sim > 0.40 else hint
        except Exception as e:
            print(f"⚠️ Face ReID: {e}")
            return hint

    # ─────────────────────────────────────────────
    # 座標距離計算（只用記憶體 local_scene_map）
    # ─────────────────────────────────────────────
    def _nearest_by_coord(self, user_pos: dict, room_name: str, max_dist: float = 3.0):
        if not user_pos:
            return None, float('inf')
        ux, uz = user_pos.get("x", 0), user_pos.get("z", 0)

        # 從記憶體讀（Change Streams 保持同步，不查 DB）
        docs = self.scene_sync.find_by_room(room_name) if room_name \
               else self.scene_sync.all_docs()
        if not docs:
            docs = self.scene_sync.all_docs()

        best_doc, best_dist = None, float('inf')
        for doc in docs:
            pos = doc.get("pos")
            if isinstance(pos, list) and len(pos) >= 2:
                fx, fz = pos[0], pos[1]
            elif doc.get("x") is not None:
                fx, fz = doc.get("x", 0), doc.get("z", 0)
            else:
                continue
            dist = math.sqrt((ux - fx) ** 2 + (uz - fz) ** 2)
            if dist < best_dist:
                best_dist, best_doc = dist, doc

        if best_doc and best_dist <= max_dist:
            return best_doc, best_dist
        return None, best_dist

    # ─────────────────────────────────────────────
    # 3. Top-K Semantic Filter + 距離決策
    # 語意 Top-3 → 距離最近的 → 最終綁定
    # ─────────────────────────────────────────────
    def _bind_furniture(self, vlm_label: str, user_pos: dict, room_name: str):
        """
        回傳 (bound_doc, confidence_str)
        決策流程：
          1. 語意 Top-3（Room Cache 矩陣乘法，微秒級）
          2. 在 Top-3 中找距離最近的
          3. 同時做座標最近查詢做交叉驗證
        """
        # 語意 Top-3（利用 Room Cache，不重新 encode 整個 DB）
        topk = self.room_cache.bind_topk(vlm_label, k=3, threshold=0.40)

        # 座標最近
        coord_doc, coord_dist = self._nearest_by_coord(user_pos, room_name)

        if not topk and not coord_doc:
            return None, "unknown"

        # 如果座標很近（< 1.5m）且在 Top-K 裡 → high confidence
        if coord_doc and coord_dist < 1.5:
            coord_label = coord_doc.get("label", "").lower()
            for doc, score in topk:
                if doc.get("label", "").lower() == coord_label:
                    return doc, "high"
            # 座標近但語意不在 Top-K → 信任座標
            return coord_doc, "coord_priority"

        # 在 Top-K 中找距離最近的
        if topk and user_pos:
            ux, uz = user_pos.get("x", 0), user_pos.get("z", 0)
            best_doc, best_dist_topk, best_score = None, float('inf'), 0.0
            for doc, score in topk:
                pos = doc.get("pos")
                if isinstance(pos, list) and len(pos) >= 2:
                    dist = math.sqrt((ux - pos[0]) ** 2 + (uz - pos[1]) ** 2)
                else:
                    dist = float('inf')
                if dist < best_dist_topk:
                    best_dist_topk = dist
                    best_doc       = doc
                    best_score     = score

            if best_doc:
                conf = "medium" if best_dist_topk < 3.0 else "sbert_priority"
                return best_doc, conf

        # Fallback：語意最高分
        if topk:
            return topk[0][0], "sbert_low"

        if coord_doc:
            return coord_doc, "coord_only"

        return None, "unknown"

    # ─────────────────────────────────────────────
    # 更新 scene_snapshots（家具層）
    # 從記憶體讀 doc，只寫 DB（Change Stream 會自動同步回記憶體）
    # ─────────────────────────────────────────────
    def _update_scene_snapshot(self, bound_doc, interacting_items,
                                scene_items, spatial_relations):
        if not bound_doc:
            return
        doc_id    = bound_doc.get("_id")
        all_items = list(set(interacting_items + scene_items))

        counts_inc = {}
        for rel in spatial_relations:
            s = rel.get("subject", ""); r = rel.get("relation", ""); o = rel.get("object", "")
            if s and r and o:
                counts_inc[f"spatial_counts.{s}|{r}|{o}"] = 1

        update_op = {
            "$addToSet": {"items": {"$each": all_items}},
            "$set": {
                "current_contents":  interacting_items,
                "spatial_relations": spatial_relations,
                "last_observation":  datetime.datetime.utcnow(),
            },
        }
        if counts_inc:
            update_op["$inc"] = counts_inc

        self.col_scene.update_one({"_id": doc_id}, update_op)

    # ─────────────────────────────────────────────
    # 更新 observation_logs
    # ─────────────────────────────────────────────
    def _update_observation_log(self, user, action, bound_doc,
                                 interacting_items, spatial_relations, raw_desc):
        if not bound_doc:
            return
        instance = bound_doc.get("label", "Unknown")
        pos_raw  = bound_doc.get("pos")
        pos_xy   = pos_raw if isinstance(pos_raw, list) else [
            bound_doc.get("x", 0), bound_doc.get("z", 0)
        ]

        self.col_obs.find_one_and_update(
            {"user": user, "instance": instance, "action": action},
            {
                "$inc":      {"weight": 1},
                "$addToSet": {"interacting_items": {"$each": interacting_items}},
                "$set": {
                    "observed_relations": spatial_relations,
                    "pos":               pos_xy,
                    "last_seen":         datetime.datetime.utcnow(),
                    "raw_vlm_desc":      raw_desc,
                },
                "$setOnInsert": {"user": user, "instance": instance, "action": action},
            },
            upsert=True,
            return_document=ReturnDocument.AFTER
        )

    # ─────────────────────────────────────────────
    # 4. 動態物件更新（Diff + Bulk Write）
    # ─────────────────────────────────────────────
    def _update_dynamic_objects(self, user_id, interacting_items, scene_items,
                                 spatial_relations, bound_doc, room_name):
        now         = datetime.datetime.utcnow()
        bound_label = bound_doc.get("label", "Unknown_Area") if bound_doc else "Unknown_Area"
        bound_pos   = bound_doc.get("pos") if bound_doc else None

        item_rel_map       = {}
        item_furniture_map = {}
        for rel in spatial_relations:
            subj     = rel.get("subject", "").lower().strip()
            obj      = rel.get("object",  "").lower().strip()
            relation = rel.get("relation","on").lower().strip()
            if subj:
                item_rel_map[subj]       = relation
                item_furniture_map[subj] = obj

        def _resolve_furniture(label: str, relation: str):
            # in_hand → 跟人走
            if relation in ("in_hand", "in_hand_of", "held_by", "carrying"):
                return bound_label, bound_pos

            # 3. Top-K Semantic Filter 應用於物件綁定
            vlm_furn = item_furniture_map.get(label)
            if vlm_furn and vlm_furn not in ("unknown", "", "none"):
                # 用 Room Cache 做 Top-1 語意匹配（微秒級）
                topk = self.room_cache.bind_topk(vlm_furn, k=1, threshold=0.40)
                if topk:
                    matched_doc = topk[0][0]
                    return matched_doc["label"], matched_doc.get("pos")

            return bound_label, bound_pos

        def _upsert(label: str, is_interacting: bool):
            label = LABEL_NORMALIZE_MAP.get(label.lower().strip(), label.lower().strip())
            if not label or label in STRUCTURAL_BLACKLIST:
                return

            # 靜態家具過濾：已在 scene_snapshots 的 label 不存入 dynamic_objects
            # 從 ChangeStreamSync 記憶體查（不查 DB，微秒級）
            if self.scene_sync.get(label):
                print(f"   🚫 [Dynamic] Skip furniture: '{label}'")
                return

            relation                     = item_rel_map.get(label, "near")
            resolved_label, resolved_pos = _resolve_furniture(label, relation)

            base_set = {
                "last_seen_on": resolved_label,
                "spatial_rel":  relation,
                "room":         room_name,
                "last_seen":    now,
                "source":       "vlm",
            }
            if resolved_pos:
                base_set["furniture_pos"] = resolved_pos

            inc_ops = {"seen_count": 1}
            if is_interacting:
                inc_ops["interact_count"] = 1

            update_op = {
                "$inc":         inc_ops,
                "$set":         base_set,
                "$setOnInsert": {"first_seen": now},
            }
            if is_interacting:
                update_op["$addToSet"] = {"interacted_by": user_id}

            # 4. Diff + Bulk Write（有變化才加入 queue）
            changed = self.bulk_buffer.upsert(label, update_op, now)
            status  = "changed" if changed else "no-change"
            print(f"   🧩 [Dynamic] '{label}' @ {resolved_label} ({relation}) [{status}]")

        for item in interacting_items:
            _upsert(item, is_interacting=True)
        for item in scene_items:
            _upsert(item, is_interacting=False)

    # ─────────────────────────────────────────────
    # 寫入 semantic_memories
    # ─────────────────────────────────────────────
    def _write_semantic_memory(self, user, action, bound_doc,
                                confidence, result, source_nodes):
        instance = bound_doc.get("label", "Unknown_Area") if bound_doc else "Unknown_Area"
        room     = (bound_doc.get("room") or bound_doc.get("room_name", "")
                    if bound_doc else "")
        self.col_memory.insert_one({
            "user":         user,
            "action":       action,
            "bound_to":     instance,
            "bound_room":   room,
            "confidence":   confidence,
            "details":      result,
            "source_nodes": source_nodes,
            "timestamp":    datetime.datetime.utcnow(),
        })

    # ─────────────────────────────────────────────
    # 更新 activity_sequences
    # ─────────────────────────────────────────────
    def _update_activity_sequence(self, user, action, instance):
        today = datetime.datetime.utcnow().strftime("%Y-%m-%d")
        self.col_activity.update_one(
            {"user": user, "date": today},
            {
                "$push":        {"sequence": {
                    "action":    action,
                    "instance":  instance,
                    "timestamp": datetime.datetime.utcnow().isoformat()
                }},
                "$setOnInsert": {"user": user, "date": today},
            },
            upsert=True
        )

    # ─────────────────────────────────────────────
    # 向量化存入 FAISS
    # ─────────────────────────────────────────────
    def _index_to_faiss(self, user, action, bound_doc, result, mongo_id):
        instance = bound_doc.get("label", "Unknown") if bound_doc else "Unknown"
        pos_raw  = bound_doc.get("pos") if bound_doc else None
        pos_xy   = pos_raw if isinstance(pos_raw, list) else [
            bound_doc.get("x", 0), bound_doc.get("z", 0)
        ] if bound_doc else [0, 0]

        memory_text = self.faiss_store.build_memory_text(
            user              = user,
            action            = action,
            instance          = instance,
            interacting_items = result.get("interacting_items", []),
            all_items         = result.get("all_items", []),
            spatial_relations = result.get("spatial_relations", []),
        )
        self.faiss_store.add(memory_text, {
            "user":              user,
            "action":            action,
            "instance":          instance,
            "interacting_items": result.get("interacting_items", []),
            "all_items":         result.get("all_items", []),
            "spatial_relations": result.get("spatial_relations", []),
            "furniture_pos":     pos_xy,
            "mongo_id":          mongo_id,
            "timestamp":         datetime.datetime.utcnow().isoformat(),
        })

    # ─────────────────────────────────────────────
    # VLM Prompt
    # ─────────────────────────────────────────────
    def _build_prompt(self, room_name, room_furniture, coord_label, coord_dist) -> str:
        furniture_ctx = ""
        if room_furniture:
            furniture_ctx = (
                f"\nFURNITURE in this room: {', '.join(room_furniture)}.\n"
                "Use these exact names for main_object.\n"
            )
        coord_ctx = ""
        if coord_label and coord_dist < 3.0:
            coord_ctx = (
                f"\nSPATIAL FACT: Person is {coord_dist:.1f}m from '{coord_label}'.\n"
            )

        return f"""Analyze this home camera image. Room: "{room_name}".
{furniture_ctx}{coord_ctx}
Reply ONLY in valid JSON, no markdown, no extra text.

{{
  "action": "most specific single verb (sleeping/eating/cooking/typing/sitting/standing/watching/drinking/swinging/...)",
  "main_object": "furniture the person is at",
  "interacting_items": ["items person physically holds or uses — [] if none"],
  "scene_items": ["background items visible on surfaces — [] if none"],
  "spatial_relations": [
    {{"subject": "item_or_person", "relation": "on/in/next_to/above/below/in_hand_of/lying_on", "object": "furniture_or_person"}}
  ],
  "description": "one natural sentence"
}}

RULES:
- action: be specific. "sleeping" not "lying". "drinking" not "standing".
- main_object: must come from FURNITURE list if provided.
- interacting_items: only items physically held or operated. [] if none.
- scene_items: items on surfaces. Do NOT repeat interacting_items.
- Use "lying_on" for horizontal person. "in_hand_of" for held items.
- Do NOT include wall/floor/ceiling in any list.
"""

    # ─────────────────────────────────────────────
    # 主入口
    # ─────────────────────────────────────────────
    def analyze_action_burst(self, payload: dict) -> dict:
        image_list   = payload.get("image_list", [])
        hint_user_id = payload.get("userID", "Unknown_User")
        source_nodes = payload.get("source_nodes", [])
        node_scores  = payload.get("node_scores", [])
        user_pos     = payload.get("user_pos", None)
        room_name    = payload.get("room_name", "")

        if not image_list:
            return self._empty_result(hint_user_id)

        # 1. 事件驅動：切房間才重建 Room Cache
        self.room_cache.switch_room(room_name, self.col_scene)

        # 座標預查（給 prompt 用，從記憶體讀）
        coord_doc, coord_dist = self._nearest_by_coord(user_pos, room_name)
        coord_label = coord_doc.get("label", "") if coord_doc else ""

        # 房間家具清單（從 Room Cache 取，不查 DB）
        room_furniture = [d.get("label", "") for d in self.room_cache.all_docs if d.get("label")]

        prompt         = self._build_prompt(room_name, room_furniture, coord_label, coord_dist)
        sample_indices = self._select_sample_indices(image_list, node_scores)

        user_votes, action_votes, object_votes = [], [], []
        interacting_pool, scene_pool, spatial_pool, descriptions = [], [], [], []

        for idx in sample_indices:
            try:
                img_b64   = image_list[idx]
                uid       = self._get_user_id(img_b64, hint_user_id)
                user_votes.append(uid)

                img_clean = img_b64.split(',')[1] if ',' in img_b64 else img_b64
                api_body  = {
                    "model":    self.model,
                    "messages": [{"role": "user", "content": prompt, "images": [img_clean]}],
                    "stream":   False,
                    "options":  {"temperature": 0.05, "num_predict": 512},
                }
                resp      = requests.post(f"{self.url}/api/chat", json=api_body, timeout=120)
                raw       = resp.json().get("message", {}).get("content", "").strip()
                node_name = source_nodes[idx] if idx < len(source_nodes) else f"node_{idx}"
                print(f"✅ [Frame {idx}|{node_name}] {raw[:120]}")

                data    = json.loads(self._extract_json(raw))
                act     = data.get("action",     "none").lower().strip()
                obj     = data.get("main_object","unknown").lower().strip()
                interact= data.get("interacting_items", [])
                scene   = data.get("scene_items", [])
                spatial = data.get("spatial_relations", [])
                desc    = data.get("description", "")

                if act in {"none", "unknown", "n/a", "not visible", "cannot determine", ""}:
                    continue

                action_votes.append(act)
                object_votes.append(obj)
                descriptions.append(desc)

                bl = {"item1", "item2", "none", "unknown", "n/a", ""}
                interacting_pool.extend([
                    i.lower().strip() for i in interact
                    if isinstance(i, str) and i.lower().strip() not in bl
                ])
                scene_pool.extend([
                    i.lower().strip() for i in scene
                    if isinstance(i, str) and i.lower().strip() not in bl
                ])
                for rel in spatial:
                    if (isinstance(rel, dict)
                            and rel.get("subject") and rel.get("relation") and rel.get("object")):
                        spatial_pool.append({k: v.lower().strip() for k, v in rel.items()})

            except Exception as e:
                print(f"❌ [Frame {idx}] {e}")

        if not action_votes:
            return self._empty_result(
                max(set(user_votes), key=user_votes.count) if user_votes else hint_user_id
            )

        final_user    = max(set(user_votes),   key=user_votes.count)
        final_action  = max(set(action_votes), key=action_votes.count)
        final_object  = max(set(object_votes), key=object_votes.count)
        final_items   = list(set(interacting_pool))
        all_items     = list(set(interacting_pool + scene_pool))
        spatial_merged= self._merge_spatial(spatial_pool)
        base_desc     = descriptions[action_votes.index(final_action)] if descriptions else ""
        scene_items   = [i for i in list(set(scene_pool)) if i not in final_items]

        # 3. Top-K Semantic Filter + 距離決策
        bound_doc, confidence = self._bind_furniture(final_object, user_pos, room_name)
        bound_label = bound_doc.get("label", "Unknown_Area") if bound_doc else "Unknown_Area"
        bound_room  = (bound_doc.get("room") or bound_doc.get("room_name", room_name)
                       if bound_doc else room_name)

        result = {
            "location":          bound_label,
            "room":              bound_room,
            "interacting_items": final_items,
            "scene_items":       scene_items,
            "all_items":         all_items,
            "spatial_relations": spatial_merged,
            "context":           base_desc,
            "_vlm_raw_object":   final_object,
            "_coord_label":      coord_label,
            "_coord_dist":       round(coord_dist, 2) if coord_dist != float('inf') else None,
            "_confidence":       confidence,
        }

        # Pipeline 寫入
        self._update_scene_snapshot(bound_doc, final_items, scene_items, spatial_merged)
        self._update_observation_log(final_user, final_action, bound_doc,
                                      final_items, spatial_merged, base_desc)
        self._update_dynamic_objects(
            user_id           = final_user,
            interacting_items = final_items,
            scene_items       = scene_items,
            spatial_relations = spatial_merged,
            bound_doc         = bound_doc,
            room_name         = bound_room,
        )
        self._write_semantic_memory(final_user, final_action, bound_doc,
                                     confidence, result, source_nodes)
        self._update_activity_sequence(final_user, final_action, bound_label)

        mem_doc  = self.col_memory.find_one(
            {"user": final_user, "action": final_action}, sort=[("timestamp", -1)]
        )
        mongo_id = str(mem_doc["_id"]) if mem_doc else ""
        self._index_to_faiss(final_user, final_action, bound_doc, result, mongo_id)

        print(f"\n✅ [Done] {final_user} → {final_action} @ {bound_label} "
              f"(room={bound_room}, conf={confidence}, "
              f"pending_writes={self.bulk_buffer.pending_count})\n")

        return {
            "user":           final_user,
            "action":         final_action,
            "result":         result,
            "items":          final_items,
            "all_items":      all_items,
            "spatial":        spatial_merged,
            "bound_instance": bound_label,
            "bound_room":     bound_room,
            "confidence":     confidence,
        }

    # ─────────────────────────────────────────────
    # 工具方法
    # ─────────────────────────────────────────────
    def _extract_json(self, raw: str) -> str:
        cleaned = re.sub(r'```(?:json)?\s*', '', raw).strip()
        m = re.search(r'\{.*\}', cleaned, re.DOTALL)
        return m.group(0) if m else cleaned

    def _merge_spatial(self, pool: list) -> list:
        seen, out = set(), []
        for rel in pool:
            key = f"{rel['subject']}|{rel['relation']}|{rel['object']}"
            if key not in seen:
                seen.add(key)
                out.append(rel)
        return out

    def _select_sample_indices(self, image_list, node_scores, max_samples=3):
        n = len(image_list)
        if n <= max_samples:
            return list(range(n))
        if node_scores and len(node_scores) == n:
            return sorted(range(n), key=lambda i: node_scores[i], reverse=True)[:max_samples]
        step = n / max_samples
        return [int(i * step) for i in range(max_samples)]

    def _empty_result(self, user_id):
        return {
            "user": user_id, "action": "none", "result": {},
            "items": [], "all_items": [], "spatial": [],
            "bound_instance": "Unknown_Area", "bound_room": ""
        }

    def shutdown(self):
        """程式結束時呼叫，確保 bulk buffer 全部寫入"""
        self.bulk_buffer.force_flush()
        self.scene_sync.stop()
        print("✅ [PerceptionEngine] Shutdown complete")