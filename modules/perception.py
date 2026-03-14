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


STRUCTURAL_BLACKLIST = {
    "wall", "floor", "ceiling", "wooden floor", "white wall",
    "white ceiling", "window", "door", "ground", "white box",
    "concrete floor", "tile floor", "carpet", "baseboard"
}

# VLM 自由描述時可能用的別名 → DB label
LABEL_NORMALIZE_MAP = {
    "remote control":  "remote",
    "tv remote":       "remote",
    "television":      "tv",
    "laptop":          "computer",
    "notebook":        "computer",
    "cell phone":      "phone",
    "mobile phone":    "phone",
    "smartphone":      "phone",
    "drinking glass":  "cup",
    "water glass":     "cup",
    "mug":             "cup",
    "juice bottle":    "juicebottle",
    "juice":           "juicebottle",
    "soda can":        "soda",
    "soda bottle":     "soda",
    "kitchen table":   "table",
    "dining table":    "table",
    "coffee table":    "table",
    "computer desk":   "desk",
    "working desk":    "desk",
    "mother's bed":    "mom's bed",
    "father's bed":    "dad's bed",
    "parents bed":     "dad's bed",
}

BULK_WRITE_THRESHOLD = 20
BULK_WRITE_INTERVAL  = 30.0

# 驗證閾值
SEMANTIC_THRESHOLD   = 0.35   # SBERT 語意相似度下限
COORD_VERIFY_DIST    = 2.0    # 座標驗證：物件到家具的合理距離上限 (m)
COORD_MATCH_DIST     = 1.5    # 座標+語意都符合 → high confidence



class RoomEmbeddingCache:
    def __init__(self, sbert_model: SentenceTransformer):
        self.model       = sbert_model
        self._room       = None
        self._labels     = []
        self._docs       = []
        self._embeddings = None

    def switch_room(self, room_name: str, scene_col):
        if room_name == self._room and self._embeddings is not None:
            return
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
            f"{d.get('label','')} in {d.get('room', d.get('room_name',''))}"
            for d in docs
        ]
        if self._labels:
            embs = self.model.encode(
                self._labels, normalize_embeddings=True, show_progress_bar=False
            )
            self._embeddings = embs.astype("float32")
        else:
            self._embeddings = None
        print(f"[RoomCache] Room '{room_name}' → {len(self._labels)} furniture cached")

    def bind_topk(self, vlm_label: str, k: int = 3, threshold: float = 0.35):
        if self._embeddings is None or not self._labels:
            return []
        q_emb = self.model.encode(
            [vlm_label], normalize_embeddings=True
        )[0].astype("float32")
        sims    = self._embeddings @ q_emb
        top_idx = np.argsort(sims)[::-1][:k]
        return [
            (self._docs[i], float(sims[i]))
            for i in top_idx if float(sims[i]) >= threshold
        ]

    @property
    def all_docs(self):
        return self._docs

    @property
    def current_room(self):
        return self._room


# ══════════════════════════════════════════════
#  Change Streams Sync
# ══════════════════════════════════════════════
class ChangeStreamSync:
    def __init__(self, scene_col, room_cache: RoomEmbeddingCache):
        self.scene_col  = scene_col
        self.room_cache = room_cache
        self._map       = {}
        self._lock      = threading.Lock()
        self._thread    = None
        self._running   = False
        self._load_all()

    def _load_all(self):
        docs = list(self.scene_col.find({}))
        with self._lock:
            self._map = {d.get("label", ""): d for d in docs}
        print(f"    [ChangeSync] Loaded {len(self._map)} scene objects into memory")

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def _watch_loop(self):
        try:
            with self.scene_col.watch(full_document="updateLookup") as stream:
                print("    [ChangeSync] Change Stream mode")
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
                    elif op == "delete":
                        key = change.get("documentKey", {}).get("label", "")
                        with self._lock:
                            self._map.pop(key, None)
        except Exception:
            print("    [ChangeSync] Polling mode (every 10secs）")
            self._poll_loop()

    def _poll_loop(self):
        while self._running:
            try:
                docs = list(self.scene_col.find({}))
                with self._lock:
                    new_map = {d.get("label", ""): d for d in docs}
                    changed = set(new_map) - set(self._map) | \
                              {k for k in new_map if new_map[k] != self._map.get(k)}
                    self._map = new_map
                if changed:
                    print(f"    [ChangeSync] Polled {len(docs)} docs, changed={changed}")
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


class BulkWriteBuffer:
    def __init__(self, dynamics_col):
        self.col        = dynamics_col
        self._last      = {}
        self._pending   = []
        self._last_flush= time.time()
        self._lock      = threading.Lock()

    def upsert(self, label: str, update_op: dict, now: datetime.datetime):
        key       = label
        new_state = (
            update_op.get("$set", {}).get("last_seen_on", ""),
            update_op.get("$set", {}).get("spatial_rel", ""),
            update_op.get("$set", {}).get("room", ""),
        )
        with self._lock:
            old_state = self._last.get(key)
            if old_state == new_state:
                return False
            self._last[key] = new_state
            self._pending.append(UpdateOne({"label": label}, update_op, upsert=True))
            elapsed      = time.time() - self._last_flush
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
            print(f"    [BulkWrite] Flushed {len(ops)} ops "
                  f"(upserted={result.upserted_count}, modified={result.modified_count})")
        except Exception as e:
            print(f"    [BulkWrite] Failed: {e}")

    def force_flush(self):
        self._flush()

    @property
    def pending_count(self):
        with self._lock:
            return len(self._pending)

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
            s = rel.get("subject",""); r = rel.get("relation",""); o = rel.get("object","")
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

        self.client       = MongoClient(mongo_uri)
        self.db           = self.client[db_name]
        self.col_scene    = self.db["scene_snapshots"]
        self.col_obs      = self.db["observation_logs"]
        self.col_memory   = self.db["semantic_memories"]
        self.col_activity = self.db["activity_sequences"]
        self.col_dynamics = self.db["dynamic_objects"]

        self.sbert      = SentenceTransformer(sbert_model_name)
        self.room_cache = RoomEmbeddingCache(self.sbert)
        self.scene_sync = ChangeStreamSync(self.col_scene, self.room_cache)
        self.scene_sync.start()
        self.bulk_buffer = BulkWriteBuffer(self.col_dynamics)
        self.faiss_store = FAISSMemoryStore(self.sbert)

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

    def _nearest_by_coord(self, user_pos: dict, room_name: str, max_dist: float = 3.0):
        if not user_pos:
            return None, float('inf')
        ux, uz = user_pos.get("x", 0), user_pos.get("z", 0)
        docs   = self.scene_sync.find_by_room(room_name) if room_name \
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
            dist = math.sqrt((ux - fx)**2 + (uz - fz)**2)
            if dist < best_dist:
                best_dist, best_doc = dist, doc
        return (best_doc, best_dist) if best_doc and best_dist <= max_dist else (None, best_dist)

    def _normalize_label(self, raw: str) -> str:
        """VLM 自由描述的名稱 → 標準化"""
        raw = raw.lower().strip()
        return LABEL_NORMALIZE_MAP.get(raw, raw)

    def _semantic_match_furniture(self, vlm_name: str, k: int = 3):

        norm = self._normalize_label(vlm_name)
        # 先查完全匹配（有就直接用，不需要 SBERT）
        exact = self.scene_sync.get(norm)
        if exact:
            return [(exact, 1.0)]
        # SBERT Top-K
        return self.room_cache.bind_topk(norm, k=k, threshold=SEMANTIC_THRESHOLD)

    def _semantic_match_dynamic(self, vlm_name: str):

        norm = self._normalize_label(vlm_name)
        # 先查完全匹配
        exact = self.col_dynamics.find_one({"label": norm})
        if exact:
            return norm, 1.0
        # SBERT 對現有動態物件做語意匹配
        all_dyn = list(self.col_dynamics.find({}, {"label": 1}))
        if not all_dyn:
            return norm, 0.5   # DB 沒資料，相信 VLM
        labels = [d["label"] for d in all_dyn]
        embs   = self.sbert.encode(labels, normalize_embeddings=True).astype("float32")
        q_emb  = self.sbert.encode([norm], normalize_embeddings=True)[0].astype("float32")
        sims   = embs @ q_emb
        best_i = int(np.argmax(sims))
        best_s = float(sims[best_i])
        if best_s >= SEMANTIC_THRESHOLD:
            return labels[best_i], best_s
        return norm, 0.0   # 語意太低 → 用原始名稱（可能是新物件）

    def _verify_with_coord(self, topk_furniture, obj_sensor_pos):

        if not obj_sensor_pos or not topk_furniture:
            return topk_furniture[0] if topk_furniture else (None, "unknown")

        ox, oz = obj_sensor_pos[0], obj_sensor_pos[1]

        # 找在 Top-K 裡距離最近的
        best_doc, best_dist, best_score = None, float('inf'), 0.0
        for doc, score in topk_furniture:
            pos = doc.get("pos")
            if isinstance(pos, list) and len(pos) >= 2:
                dist = math.sqrt((ox - pos[0])**2 + (oz - pos[1])**2)
                if dist < best_dist:
                    best_dist, best_doc, best_score = dist, doc, score

        if best_doc is None:
            return topk_furniture[0][0], "sbert_only"

        if best_dist <= COORD_MATCH_DIST:
            conf = "high" if best_score >= 0.7 else "medium"
            print(f"    [Verify] '{best_doc.get('label')}' dist={best_dist:.2f}m score={best_score:.2f} → {conf}")
            return best_doc, conf
        elif best_dist <= COORD_VERIFY_DIST:
            print(f"    [Verify] '{best_doc.get('label')}' dist={best_dist:.2f}m → coord_ok")
            return best_doc, "coord_ok"
        else:
            # 語意說這個家具，但座標太遠 → 改用座標最近的
            print(f"    [Verify] '{best_doc.get('label')}' dist={best_dist:.2f}m too far → coord_fallback")
            return None, "coord_fallback"

    def _bind_furniture(self, vlm_label: str, user_pos: dict, room_name: str):
        """
        人的位置綁定：
        Step 1: 語意匹配 VLM 說的家具名稱
        Step 2: 座標驗證
        Step 3: fallback 到座標最近
        """
        topk       = self._semantic_match_furniture(vlm_label, k=3)
        coord_doc, coord_dist = self._nearest_by_coord(user_pos, room_name)

        if not topk and not coord_doc:
            return None, "unknown"

        if topk:
            # 用人的座標驗證語意結果
            if user_pos:
                user_sensor = [user_pos.get("x", 0), user_pos.get("z", 0)]
                doc, conf = self._verify_with_coord(topk, user_sensor)
                if doc:
                    return doc, conf

        if coord_doc and coord_dist < 1.5:
            return coord_doc, "coord_priority"
        if coord_doc:
            return coord_doc, "coord_only"
        if topk:
            return topk[0][0], "sbert_low"

        return None, "unknown"


    def _build_prompt(self, room_name, room_furniture, coord_label, coord_dist) -> str:
        furniture_hint = ""
        if room_furniture:
            furniture_hint = f"\nKNOWN FURNITURE in this room: {', '.join(room_furniture)}.\n"

        coord_hint = ""
        if coord_label and coord_dist < 3.0:
            coord_hint = (
                f"\nSPATIAL HINT: The person is approximately {coord_dist:.1f}m "
                f"from '{coord_label}'.\n"
            )

        return f"""You are a scene analysis system for a home robot.
Your job is to describe EXACTLY what you observe in this image.
Room: "{room_name}".
{furniture_hint}{coord_hint}
Reply ONLY in valid JSON. No markdown, no explanation.

{{
  "action": "what the person is doing — be specific (sleeping/eating/drinking/typing/watching/cooking/exercising/standing/...)",
  "person_holding": ["objects the person is visibly holding in their hands — [] if none"],
  "person_near": "name of the furniture or surface the person is closest to",
  "objects_on_furniture": [
    {{"object": "item name", "on": "furniture name", "relation": "on/in/next_to/above"}}
  ],
  "description": "one sentence describing the scene"
}}

OBSERVATION RULES:
- action: describe the ACTIVITY, not the posture.
  PRIORITY: what is the person DOING with their body or objects?
  Holding bottle/cup → "drinking". Holding food → "eating".
  At keyboard/computer → "typing". Looking at screen/tv → "watching".
  Lying down/eyes closed → "sleeping". Swinging arm → "exercising".
  Use "standing" ONLY if the person is idle with nothing in hand and no clear activity.
- person_holding: ONLY objects physically in the person's hands. If holding bottle → ["bottle"].
- person_near: the closest furniture or surface the person is at. Any name is fine.
- objects_on_furniture: ALL portable objects visible on surfaces (NOT in hands).
  Each item needs: what it is, what surface it is on, and the spatial relation.
- description: one natural sentence summarizing the scene.
- NEVER include wall/floor/ceiling/window/door in any field.
"""

    def _parse_vlm_output(self, raw: str) -> dict:

        try:
            data = json.loads(self._extract_json(raw))
        except Exception:
            return {}

        action         = data.get("action", "none").lower().strip()
        person_holding = data.get("person_holding", [])
        person_near    = data.get("person_near", "unknown").lower().strip()
        objs_on_furn   = data.get("objects_on_furniture", [])
        description    = data.get("description", "")

        # 轉換 objects_on_furniture → spatial_relations
        spatial_relations = []
        scene_items       = []
        for entry in objs_on_furn:
            if not isinstance(entry, dict):
                continue
            obj  = entry.get("object", "").lower().strip()
            on   = entry.get("on",     "").lower().strip()
            rel  = entry.get("relation", "on").lower().strip()
            if obj and on:
                scene_items.append(obj)
                spatial_relations.append({
                    "subject":  obj,
                    "relation": rel,
                    "object":   on,
                })

        interacting_items = [
            i.lower().strip() for i in person_holding
            if isinstance(i, str) and i.lower().strip()
        ]
        for item in interacting_items:
            spatial_relations.append({
                "subject":  item,
                "relation": "in_hand_of",
                "object":   "person",   # 後續由 hint_user_id 替換
            })

        return {
            "action":            action,
            "main_object":       person_near,         # VLM 原始名稱
            "interacting_items": interacting_items,   # VLM 原始名稱
            "scene_items":       scene_items,         # VLM 原始名稱
            "spatial_relations": spatial_relations,   # VLM 原始名稱
            "description":       description,
        }

    def _validate_scene_graph(self, parsed: dict, user_pos: dict, room_name: str, user_id: str):

        print(f"    [Validate] Starting scene graph validation...")

        # 1. 驗證人的位置（main_object）
        vlm_furn    = parsed.get("main_object", "unknown")
        bound_doc, confidence = self._bind_furniture(vlm_furn, user_pos, room_name)
        bound_label = bound_doc.get("label", "Unknown_Area") if bound_doc else "Unknown_Area"
        bound_room  = (bound_doc.get("room") or bound_doc.get("room_name", room_name)
                       if bound_doc else room_name)
        print(f"    [Validate] person_near: '{vlm_furn}' → '{bound_label}' (conf={confidence})")

        # 2. 驗證每條 spatial_relation
        validated_spatial = []
        for rel in parsed.get("spatial_relations", []):
            subj = rel.get("subject", "").lower().strip()
            obj  = rel.get("object",  "").lower().strip()
            r    = rel.get("relation","on").lower().strip()

            # person → user_id
            if obj == "person":
                validated_spatial.append({
                    "subject":  self._normalize_label(subj),
                    "relation": r,
                    "object":   user_id,
                })
                continue


            topk_furn = self._semantic_match_furniture(obj, k=3)

            # 如果物件有 sensor_pos，做座標驗證
            obj_norm    = self._normalize_label(subj)
            dyn_doc     = self.col_dynamics.find_one({"label": obj_norm})
            sensor_pos  = dyn_doc.get("sensor_pos") if dyn_doc else None

            if topk_furn:
                furn_doc, furn_conf = self._verify_with_coord(topk_furn, sensor_pos)
                furn_label = furn_doc.get("label", obj) if furn_doc else obj
            else:
                furn_label = self._normalize_label(obj)
                furn_conf  = "no_match"

            # 物件名稱標準化
            matched_subj, subj_score = self._semantic_match_dynamic(subj)
            print(f"    [Validate] '{subj}'→'{matched_subj}'(score={subj_score:.2f}) "
                  f"on '{obj}'→'{furn_label}'(conf={furn_conf})")

            validated_spatial.append({
                "subject":  matched_subj,
                "relation": r,
                "object":   furn_label,
                "_confidence": furn_conf,
            })

        # 3. 驗證 interacting_items（只標準化名稱，不做位置驗證）
        validated_interact = []
        for item in parsed.get("interacting_items", []):
            matched, score = self._semantic_match_dynamic(item)
            print(f"    [Validate] holding '{item}' → '{matched}' (score={score:.2f})")
            validated_interact.append(matched)

        # 4. scene_items 從 validated_spatial 裡取（已驗證過）
        validated_scene = [
            r["subject"] for r in validated_spatial
            if r.get("relation") not in ("in_hand_of", "held_by")
               and r.get("object") != user_id
        ]

        return {
            "bound_doc":         bound_doc,
            "bound_label":       bound_label,
            "bound_room":        bound_room,
            "confidence":        confidence,
            "action":            parsed.get("action", "none"),
            "interacting_items": validated_interact,
            "scene_items":       validated_scene,
            "all_items":         list(set(validated_interact + validated_scene)),
            "spatial_relations": validated_spatial,
            "description":       parsed.get("description", ""),
        }


    def analyze_action_burst(self, payload: dict) -> dict:
        image_list   = payload.get("image_list", [])
        hint_user_id = payload.get("userID", "Unknown_User")
        source_nodes = payload.get("source_nodes", [])
        node_scores  = payload.get("node_scores", [])
        user_pos     = payload.get("user_pos", None)
        room_name    = payload.get("room_name", "")

        if not image_list:
            return self._empty_result(hint_user_id)

        # 切房間（事件驅動，同房間不重建）
        self.room_cache.switch_room(room_name, self.col_scene)
        if not self.room_cache.all_docs:
            print(f"   ⚠️  [RoomCache] Empty for room={room_name}, fallback to full DB load")
            self.room_cache._room = None
            self.room_cache.switch_room("", self.col_scene)

        # 座標預查（給 prompt 的 hint 用）
        coord_doc, coord_dist = self._nearest_by_coord(user_pos, room_name)
        coord_label = coord_doc.get("label", "") if coord_doc else ""

        # 房間家具清單（給 prompt 的 hint 用）
        room_furniture = [d.get("label", "") for d in self.room_cache.all_docs if d.get("label")]

        prompt         = self._build_prompt(room_name, room_furniture, coord_label, coord_dist)
        sample_indices = self._select_sample_indices(image_list, node_scores)

        user_votes   = []
        parsed_list  = []   # 每幀的 parsed 結果

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
                    "options":  {"temperature": 0.05, "num_predict": 900},
                }
                resp      = requests.post(f"{self.url}/api/chat", json=api_body, timeout=120)
                raw       = resp.json().get("message", {}).get("content", "").strip()
                node_name = source_nodes[idx] if idx < len(source_nodes) else f"node_{idx}"
                print(f" [Frame {idx}|{node_name}] {raw[:150]}")

                parsed = self._parse_vlm_output(raw)
                if not parsed:
                    continue

                act = parsed.get("action", "none")
                if act in {"none", "unknown", "n/a", "not visible", "cannot determine", ""}:
                    continue

                parsed_list.append(parsed)

            except Exception as e:
                print(f"❌ [Frame {idx}] {e}")

        if not parsed_list:
            return self._empty_result(
                max(set(user_votes), key=user_votes.count) if user_votes else hint_user_id
            )

        # 多幀投票（action + main_object）
        final_user   = max(set(user_votes), key=user_votes.count) if user_votes else hint_user_id
        action_votes = [p["action"] for p in parsed_list]
        object_votes = [p["main_object"] for p in parsed_list]
        final_action = max(set(action_votes), key=action_votes.count)
        final_object = max(set(object_votes), key=object_votes.count)

        best_parsed = parsed_list[action_votes.index(final_action)]
        best_parsed["action"]      = final_action
        best_parsed["main_object"] = final_object

        validated = self._validate_scene_graph(
            parsed    = best_parsed,
            user_pos  = user_pos,
            room_name = room_name,
            user_id   = final_user,
        )

        bound_doc   = validated["bound_doc"]
        bound_label = validated["bound_label"]
        bound_room  = validated["bound_room"]
        confidence  = validated["confidence"]

        result = {
            "location":          bound_label,
            "room":              bound_room,
            "interacting_items": validated["interacting_items"],
            "scene_items":       validated["scene_items"],
            "all_items":         validated["all_items"],
            "spatial_relations": validated["spatial_relations"],
            "context":           validated["description"],
            "_vlm_raw_object":   final_object,
            "_coord_label":      coord_label,
            "_coord_dist":       round(coord_dist, 2) if coord_dist != float('inf') else None,
            "_confidence":       confidence,
        }

        # Pipeline 寫入
        self._update_scene_snapshot(bound_doc, validated["interacting_items"],
                                    validated["scene_items"], validated["spatial_relations"])
        self._update_observation_log(final_user, final_action, bound_doc,
                                     validated["interacting_items"],
                                     validated["spatial_relations"],
                                     validated["description"])
        self._update_dynamic_objects(
            user_id           = final_user,
            interacting_items = validated["interacting_items"],
            scene_items       = validated["scene_items"],
            spatial_relations = validated["spatial_relations"],
            bound_doc         = bound_doc,
            room_name         = bound_room,
            user_pos          = user_pos,
        )
        self._write_semantic_memory(final_user, final_action, bound_doc,
                                    confidence, result, source_nodes)
        self._update_activity_sequence(final_user, final_action, bound_label)

        mem_doc  = self.col_memory.find_one(
            {"user": final_user, "action": final_action}, sort=[("timestamp", -1)]
        )
        mongo_id = str(mem_doc["_id"]) if mem_doc else ""
        self._index_to_faiss(final_user, final_action, bound_doc, result, mongo_id)

        print(f"\n [Done] {final_user} → {final_action} @ {bound_label} "
              f"(room={bound_room}, conf={confidence}, "
              f"pending_writes={self.bulk_buffer.pending_count})\n")

        return {
            "user":           final_user,
            "action":         final_action,
            "result":         result,
            "items":          validated["interacting_items"],
            "all_items":      validated["all_items"],
            "spatial":        validated["spatial_relations"],
            "bound_instance": bound_label,
            "bound_room":     bound_room,
            "confidence":     confidence,
        }


    def _extract_json(self, raw: str) -> str:
        cleaned = re.sub(r'```(?:json)?\s*', '', raw).strip()
        m = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if not m:
            return cleaned
        text = m.group(0)
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            for i in range(len(text) - 1, 0, -1):
                if text[i] in (',', '{'):
                    candidate = text[:i].rstrip(',') + '}}'
                    try:
                        json.loads(candidate)
                        print(f"   🔧 [JSON Repair] Truncated JSON fixed")
                        return candidate
                    except Exception:
                        continue
            return text

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
            "bound_instance": "Unknown_Area", "bound_room": "",
            "confidence": "unknown",
        }

    def _update_scene_snapshot(self, bound_doc, interacting_items,
                                scene_items, spatial_relations):
        if not bound_doc:
            return
        doc_id    = bound_doc.get("_id")
        all_items = list(set(interacting_items + scene_items))
        counts_inc = {}
        for rel in spatial_relations:
            s = rel.get("subject",""); r = rel.get("relation",""); o = rel.get("object","")
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
        print(f"[ObsLog] {user} → {action} @ {instance} (+1 weight)")

    def _update_dynamic_objects(self, user_id, interacting_items, scene_items,
                                 spatial_relations, bound_doc, room_name, user_pos=None):
        now         = datetime.datetime.utcnow()
        bound_label = bound_doc.get("label", "Unknown_Area") if bound_doc else "Unknown_Area"
        bound_pos   = bound_doc.get("pos") if bound_doc else None

        # spatial_relations 裡已驗證過，直接用
        # ⚠️ key 要 normalize，因為 _upsert 裡 label 已被 normalize
        item_rel_map       = {}
        item_furniture_map = {}
        for rel in spatial_relations:
            subj = rel.get("subject","").lower().strip()
            obj  = rel.get("object", "").lower().strip()
            r    = rel.get("relation","on").lower().strip()
            if subj:
                norm_subj = LABEL_NORMALIZE_MAP.get(subj, subj)
                item_rel_map[norm_subj]       = r
                item_furniture_map[norm_subj] = obj
                # 原始名稱也保留，避免已是標準名的情況漏掉
                item_rel_map[subj]       = r
                item_furniture_map[subj] = obj

        def _resolve_furniture(label: str, relation: str):
            if relation in ("in_hand_of", "held_by", "carrying", "in_hand"):
                # 物件在人手上 → last_seen_on = user_id，不是家具
                return user_id, None
            vlm_furn = item_furniture_map.get(label)
            if vlm_furn and vlm_furn not in ("unknown", "", "none", user_id):
                # 已經驗證過的家具名稱，直接從 scene_sync 取座標
                furn_doc = self.scene_sync.get(vlm_furn)
                if furn_doc:
                    return furn_doc["label"], furn_doc.get("pos")
            return bound_label, bound_pos

        def _upsert(label: str, is_interacting: bool):
            label = LABEL_NORMALIZE_MAP.get(label.lower().strip(), label.lower().strip())
            if not label or label in STRUCTURAL_BLACKLIST:
                return
            if self.scene_sync.get(label):
                return  # 靜態家具不存入 dynamic_objects

            relation                     = item_rel_map.get(label, "near")
            resolved_label, resolved_pos = _resolve_furniture(label, relation)

            is_held = relation in ("in_hand_of", "held_by", "carrying", "in_hand")
            base_set = {
                "last_seen_on": resolved_label,   # user_id 或 家具 label
                "spatial_rel":  "held_by" if is_held else relation,
                "room":         room_name,
                "last_seen":    now,
                "source":       "vlm",
            }
            if is_held and user_pos:
                # 物件跟著人走，記錄人的當前座標
                base_set["furniture_pos"] = [
                    user_pos.get("x", 0), user_pos.get("z", 0)
                ]
            elif resolved_pos:
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

            changed = self.bulk_buffer.upsert(label, update_op, now)
            status  = "changed" if changed else "no-change"
            print(f"   🧩 [Dynamic] '{label}' @ {resolved_label} ({relation}) [{status}]")

        for item in interacting_items:
            _upsert(item, is_interacting=True)
        for item in scene_items:
            _upsert(item, is_interacting=False)

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
        print(f"[Sequence] {user} → {action}@{instance} ({today})")

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
        print(f"[FAISS] memory：{instance} | {action}")

    def shutdown(self):
        self.bulk_buffer.force_flush()
        self.scene_sync.stop()
        print("[PerceptionEngine] Shutdown complete")