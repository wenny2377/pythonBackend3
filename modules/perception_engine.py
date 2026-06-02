import re
import json
import math
import time
import base64
import datetime
import threading
import requests
import os

import cv2
import numpy as np
import faiss

from dataclasses import dataclass, field
from collections import defaultdict, Counter
from pymongo import MongoClient, UpdateOne, ReturnDocument
from sentence_transformers import SentenceTransformer

try:
    import yaml as _yaml
    _YAML_OK = True
except ImportError:
    _YAML_OK = False


def _load_config(path: str) -> dict:
    if _YAML_OK and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return _yaml.safe_load(f) or {}
    return {}


_ontology = _load_config("config/robot_ontology.yaml")
_beh_cfg  = _load_config("config/behavior_config.yaml")
_sys_cfg  = _load_config("config/system_config.yaml")
_coco_cfg = _load_config("config/coco_objects.yaml")

_hp = _sys_cfg.get("hyperparameters", {})

STRUCTURAL_BLACKLIST = set(_ontology.get("structural_blacklist", [
    "wall", "floor", "ceiling", "wooden floor", "white wall", "window",
    "door", "ground", "concrete floor", "tile floor", "carpet", "baseboard",
]))

YOUR_OBJECTS = (
    set(_coco_cfg.get("coco_objects", [])) |
    set(_ontology.get("scene_objects", [])) |
    set(str(k).lower() for k in _ontology.get("item_to_action", {}).keys()) |
    {"remote", "remote control", "tv remote", "juice", "cola", "pan",
     "broom", "mop", "spatula", "bowl", "saladbowl", "cup", "bottle", "phone",
     "book", "laptop", "keyboard", "fork", "spoon", "plate", "food"}
)

LABEL_NORMALIZE_MAP = {
    str(k).lower(): str(v).strip('"').strip()
    for k, v in _ontology.get("label_normalize_map", {}).items()
}

ITEM_TO_ACTION = {
    str(k).lower(): str(v).strip('"').strip()
    for k, v in _ontology.get("item_to_action", {}).items()
}

OBJECT_VOCAB = list(dict.fromkeys(
    list(ITEM_TO_ACTION.keys()) +
    (list(_ontology.get("object_vocab", []))) +
    ["none", "remote", "book", "phone", "laptop", "broom", "mop",
     "cup", "glass", "bottle", "bowl", "fork", "spoon", "pan",
     "spatula", "keyboard", "magazine", "chopsticks", "smartphone"]
))

_BODY_CONSTRAINTS_RAW = _ontology.get("body_constraints", {}).get("impossible", {})
BODY_IMPOSSIBLE = {
    (pos.lower(), beh)
    for pos, behs in _BODY_CONSTRAINTS_RAW.items()
    for beh in (behs or [])
}

BEHAVIOR_LABELS = _beh_cfg.get("behavior_labels", [
    "Drinking", "SittingDrink", "Sitting", "Eating", "Cooking", "Opening",
    "Laying", "Watching", "Reading", "Cleaning", "PhoneUse",
    "Typing", "StandUp", "PickingUp", "PuttingDown", "Standing", "Walking",
])

NO_WEIGHT_ACTIONS = set(_beh_cfg.get("no_weight_actions",
    ["PickingUp", "PuttingDown", "Walking", "Standing", "StandUp"]))

VISION_PROTOTYPES = {
    k: v.strip()
    for k, v in _beh_cfg.get("vision_prototypes", {}).items()
}

NEARBY_OBJECT_RADIUS  = float(_hp.get("nearby_object_radius", 2.0))
HEADING_THRESHOLD     = float(_hp.get("heading_threshold", 0.55))
NORMALIZE_THRESHOLD   = float(_hp.get("normalize_threshold", 0.38))
SEMANTIC_THRESHOLD    = float(_hp.get("semantic_threshold", 0.35))
COORD_MATCH_DIST      = float(_hp.get("coord_match_dist", 1.5))
COORD_VERIFY_DIST     = float(_hp.get("coord_verify_dist", 2.0))
BULK_WRITE_THRESHOLD  = int(_hp.get("bulk_write_threshold", 20))
BULK_WRITE_INTERVAL   = float(_hp.get("bulk_write_interval", 30.0))
L3_STANDARD_THRESHOLD = float(_hp.get("l3_standard_threshold", 0.40))
HIGH_AFFORDANCE_L3_THRESHOLD = float(_hp.get("high_affordance_l3_threshold", 0.30))

VLM_CONFIDENCE_THRESHOLD = float(_hp.get("vlm_confidence_threshold", 0.50))
VLM_HINT_CONFIDENCE_GATE = float(_hp.get("vlm_hint_confidence_gate", 0.60))

HIGH_AFFORDANCE_FURNITURE = set(_ontology.get("high_affordance_furniture", [
    "tv", "television", "stove", "oven", "refrigerator", "fridge",
    "keyboard", "monitor", "cabinet"
]))

HELD_WEIGHT          = 0.7
NEARBY_WEIGHT        = 0.3
SAYCAN_MIN_SCORE     = 0.15
MIN_WRITE_CONFIDENCE = 0.20

ACTION_TO_RELATION = {
    "Drinking":     "holding",
    "SittingDrink": "sitting_on",
    "Sitting":      "sitting_on",
    "Eating":       "eating_at",
    "Cooking":      "using",
    "Opening":      "interacting_with",
    "Laying":       "lying_on",
    "Watching":     "watching",
    "Reading":      "holding",
    "Cleaning":     "using",
    "PhoneUse":     "holding",
    "Typing":       "using",
    "StandUp":      "near",
    "Standing":     "near",
    "Walking":      "near",
}


def _get_time_slot(virtual_hour) -> str:
    if virtual_hour is None:
        return "Unknown"
    try:
        h = float(virtual_hour)
        if h < 10:  return "Morning"
        if h < 13:  return "Noon"
        if h < 18:  return "Afternoon"
        if h < 22:  return "Evening"
        return "Night"
    except Exception:
        return "Unknown"


def _virtual_day_to_date(virtual_day) -> str:
    if virtual_day is None:
        return datetime.datetime.utcnow().strftime("%Y-%m-%d")
    if isinstance(virtual_day, str) and len(virtual_day) == 10:
        try:
            datetime.datetime.strptime(virtual_day, "%Y-%m-%d")
            return virtual_day
        except ValueError:
            pass
    return datetime.datetime.utcnow().strftime("%Y-%m-%d")


def normalize_label(label, sbert_model=None):
    if not label:
        return ""
    key = label.lower().strip().replace(" ", "").replace("_", "")
    if key in LABEL_NORMALIZE_MAP:
        return LABEL_NORMALIZE_MAP[key]
    clean = label.lower().strip()
    if clean in LABEL_NORMALIZE_MAP:
        return LABEL_NORMALIZE_MAP[clean]
    return clean


@dataclass
class PerceptionResult:
    vlm_output:        str   = ""
    spatial_action:    str   = ""
    zone_name:         str   = ""
    sbert_sim:         float = 0.0
    upgrade_reason:    str   = ""
    interacting_items: list  = field(default_factory=list)
    user_pos:          dict  = field(default_factory=dict)
    room:              str   = ""
    instance:          str   = ""
    spatial_relations: dict  = field(default_factory=dict)
    raw_desc:          str   = ""
    ground_truth:      str   = ""
    experiment_mode:   str   = "habit"
    virtual_hour:      float = 12.0
    user_id:           str   = ""


class RoomEmbeddingCache:
    def __init__(self, sbert_model):
        self.model       = sbert_model
        self._room       = None
        self._labels     = []
        self._docs       = []
        self._embeddings = None

    def switch_room(self, room_name, scene_col):
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
            self._embeddings = self.model.encode(
                self._labels, normalize_embeddings=True,
                show_progress_bar=False).astype("float32")
        else:
            self._embeddings = None

    def bind_topk(self, label, k=3, threshold=0.35):
        if self._embeddings is None or not self._labels:
            return []
        q_emb = self.model.encode([label], normalize_embeddings=True)[0].astype("float32")
        sims    = self._embeddings @ q_emb
        top_idx = np.argsort(sims)[::-1][:k]
        return [(self._docs[i], float(sims[i])) for i in top_idx if float(sims[i]) >= threshold]

    @property
    def all_docs(self):     return self._docs
    @property
    def current_room(self): return self._room


class ChangeStreamSync:
    def __init__(self, scene_col, room_cache):
        self.scene_col  = scene_col
        self.room_cache = room_cache
        self._map       = {}
        self._lock      = threading.Lock()
        self._running   = False
        self._load_all()

    def _load_all(self):
        docs = list(self.scene_col.find({}))
        with self._lock:
            self._map = {d.get("label", ""): d for d in docs}

    def start(self):
        self._running = True
        threading.Thread(target=self._watch_loop, daemon=True).start()

    def stop(self):
        self._running = False

    def _watch_loop(self):
        try:
            with self.scene_col.watch(full_document="updateLookup") as stream:
                for change in stream:
                    if not self._running: break
                    op  = change.get("operationType")
                    doc = change.get("fullDocument")
                    if doc and op in ("insert", "update", "replace"):
                        with self._lock:
                            self._map[doc.get("label", "")] = doc
                    elif op == "delete":
                        with self._lock:
                            self._map.pop(
                                change.get("documentKey", {}).get("label", ""), None)
        except Exception:
            while self._running:
                try:
                    docs = list(self.scene_col.find({}))
                    with self._lock:
                        self._map = {d.get("label", ""): d for d in docs}
                except Exception:
                    pass
                time.sleep(10)

    def get(self, label):
        with self._lock: return self._map.get(label)

    def find_by_room(self, room_name):
        with self._lock:
            return [d for d in self._map.values()
                    if room_name.lower() in
                    (d.get("room", "") + d.get("room_name", "")).lower()]

    def all_docs(self):
        with self._lock: return list(self._map.values())


class BulkWriteBuffer:
    def __init__(self, dynamics_col):
        self.col         = dynamics_col
        self._last       = {}
        self._pending    = []
        self._last_flush = time.time()
        self._lock       = threading.Lock()

    def upsert(self, label, update_op, now):
        new_state = (
            update_op.get("$set", {}).get("last_seen_on", ""),
            update_op.get("$set", {}).get("spatial_rel", ""),
            update_op.get("$set", {}).get("room", ""),
        )
        with self._lock:
            if self._last.get(label) == new_state:
                return False
            self._last[label] = new_state
            self._pending.append(UpdateOne({"label": label}, update_op, upsert=True))
            should_flush = (len(self._pending) >= BULK_WRITE_THRESHOLD or
                            time.time() - self._last_flush >= BULK_WRITE_INTERVAL)
        if should_flush:
            self._flush()
        return True

    def _flush(self):
        with self._lock:
            if not self._pending: return
            ops = self._pending.copy()
            self._pending.clear()
            self._last_flush = time.time()
        try:
            self.col.bulk_write(ops, ordered=False)
        except Exception as e:
            print(f"[BulkWrite] Failed: {e}")

    def force_flush(self): self._flush()

    @property
    def pending_count(self):
        with self._lock: return len(self._pending)


class FAISSMemoryStore:
    def __init__(self, sbert_model, dim=384):
        self.model    = sbert_model
        self.index    = faiss.IndexFlatIP(dim)
        self.metadata = []

    def build_memory_text(self, user, action, instance,
                          interacting_items, all_items, spatial_relations):
        parts = [f"{user} {action} near {instance}"]
        if interacting_items:
            parts[0] += f" with {', '.join(interacting_items)}"
        parts[0] += "."
        for rel in spatial_relations:
            s = rel.get("subject", "")
            r = rel.get("relation", "")
            o = rel.get("object", "")
            if s and r and o:
                parts.append(f"{s} {r} {o}.")
        bg = [i for i in all_items if i not in interacting_items]
        if bg:
            parts.append(f"Visible: {', '.join(bg)}.")
        return " ".join(parts)

    def add(self, text, metadata):
        emb = self.model.encode(
            [text], normalize_embeddings=True)[0].astype("float32")
        self.index.add(np.array([emb]))
        self.metadata.append({**metadata, "memory_text": text})

    def search(self, query, k=5):
        if self.index.ntotal == 0: return []
        q = self.model.encode(
            [query], normalize_embeddings=True)[0].astype("float32")
        scores, indices = self.index.search(np.array([q]), k)
        return [{"score": float(s), **self.metadata[i]}
                for s, i in zip(scores[0], indices[0]) if i >= 0]


class PerceptionEngine:

    def __init__(self, db, ollama_url: str, vlm_model: str,
                 sbert_model, scene_engine,
                 face_analyzer=None, face_bank=None):
        self.db           = db
        self.url          = ollama_url
        self.model        = vlm_model
        self.sbert        = sbert_model
        self.scene_engine = scene_engine
        self.face_app     = face_analyzer
        self.face_bank    = face_bank

        self.col_scene       = db.scene_snapshots
        self.col_dynamic     = db.dynamic_objects
        self.col_raw         = db.raw_objects
        self.col_eval        = db.eval_logs
        self.col_sem         = db.semantic_memories
        self.col_obs         = db.observation_logs
        self.col_habit_snap  = db.habit_snapshots
        self.col_activity    = db.activity_sequences
        self.col_user_aff    = db.user_spatial_affinity
        self.col_aff_history = db.affinity_history
        self.col_memory      = db.robot_memory

        self._proto_labels = self.scene_engine._proto_labels
        self._proto_vecs   = self.scene_engine._proto_vecs

        self._proto_vecs_sbert = None
        self._build_prototype_vecs()

        self._room_cache = RoomEmbeddingCache(self.sbert)
        self._bulk_buf   = BulkWriteBuffer(self.col_dynamic)

        dim = self.sbert.get_sentence_embedding_dimension()
        self._faiss_dyn  = FAISSMemoryStore(sbert_model=self.sbert, dim=dim)
        self.faiss_store = FAISSMemoryStore(sbert_model=self.sbert, dim=dim)

        self._lock           = threading.Lock()
        self.room_cache      = RoomEmbeddingCache(self.sbert)
        self.scene_sync      = None
        self.manifold_engine = None

    def _build_prototype_vecs(self):
        labels = list(VISION_PROTOTYPES.keys())
        texts  = [VISION_PROTOTYPES[l] for l in labels]
        if texts:
            self._proto_vecs_sbert = self.sbert.encode(
                texts, normalize_embeddings=True).astype("float32")
            self._proto_behavior_labels = labels

    def _build_prompt(self, room_name, room_furniture, coord_label, coord_dist):
        furniture_str = ", ".join(room_furniture[:5]) if room_furniture else "unknown"
        nearby_str    = coord_label if coord_label else "unknown"
        return (
            f"Scene: {room_name} room. "
            f"Person is near {nearby_str}. "
            f"Furniture visible: {furniture_str}.\n\n"
            "Output ONLY valid JSON with these exact fields:\n"
            '{"activity":"...","body_position":"...","body_orientation":"...","held_object":"...","confidence":0.0}\n\n'
            "Rules:\n"
            "- activity: ONE of Eating/Drinking/SittingDrink/Sitting/Cooking/Opening/"
            "Laying/Watching/Reading/Cleaning/PhoneUse/Typing/StandUp/PickingUp/PuttingDown/Standing/Walking\n"
            "- Sitting = sitting idle with empty hands. SittingDrink = sitting while holding beverage.\n"
            "- StandUp = person actively rising from seated/lying position.\n"
            "- body_position: standing/sitting/lying\n"
            "- body_orientation: facing_toward/facing_away/side\n"
            "- held_object: name of object in hand, "
            "or 'none' if hands clearly empty, "
            "or 'unknown' if hands not visible\n"
            "- confidence: 0.0=unsure, 1.0=certain\n"
            "Output ONLY the JSON. No markdown."
        )

    def _parse_vlm_output(self, raw: str) -> dict:
        try:
            cleaned = re.sub(r'```(?:json)?\s*', '', raw).strip().rstrip('`')
            m = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if not m:
                return {}
            data = json.loads(m.group(0))
        except Exception:
            return {}

        activity         = data.get("activity", "").strip()
        body_position    = data.get("body_position", "standing").strip().lower()
        body_orientation = data.get("body_orientation", "facing_toward").strip().lower()
        held_object      = data.get("held_object", "none").strip().lower()
        confidence       = float(data.get("confidence", 0.5))

        if held_object in ("", "null", "nothing", "empty", "n/a"):
            held_object = "none"

        if "away" in body_orientation:
            body_orientation = "facing_away"
        elif "side" in body_orientation or "lateral" in body_orientation:
            body_orientation = "side"
        else:
            body_orientation = "facing_toward"

        CANONICAL_ACTIVITY = {v.lower(): v for v in BEHAVIOR_LABELS}
        activity = CANONICAL_ACTIVITY.get(activity.lower(), activity)

        return {
            "activity":         activity,
            "body_position":    body_position,
            "body_orientation": body_orientation,
            "held_object":      held_object,
            "confidence":       min(max(confidence, 0.0), 1.0),
        }

    def _normalize_object(self, raw_obj: str) -> str:
        if not raw_obj or raw_obj.lower() in ("none", "empty", "", "unknown"):
            return "none"
        raw_lower = raw_obj.lower().strip()
        for vocab_word in OBJECT_VOCAB:
            if vocab_word in raw_lower:
                return vocab_word
        try:
            vocab_vecs = self.sbert.encode(OBJECT_VOCAB, normalize_embeddings=True)
            raw_vec    = self.sbert.encode([raw_lower], normalize_embeddings=True)[0]
            sims       = vocab_vecs @ raw_vec
            best_idx   = int(sims.argmax())
            if float(sims[best_idx]) > 0.45:
                return OBJECT_VOCAB[best_idx]
        except Exception:
            pass
        return "none"

    def _get_nearby_objects(self, user_pos: dict, room_name: str) -> list:
        if not user_pos:
            return []
        ux = float(user_pos.get("x", 0))
        uz = float(user_pos.get("z", 0))
        nearby = []
        try:
            docs = list(self.col_dynamic.find(
                {"room": {"$regex": room_name, "$options": "i"}} if room_name else {},
                {"label": 1, "sensor_pos": 1, "furniture_pos": 1}
            ))
            for doc in docs:
                pos = doc.get("sensor_pos") or doc.get("furniture_pos")
                if not isinstance(pos, list) or len(pos) < 2:
                    continue
                dist = math.sqrt((ux - pos[0]) ** 2 + (uz - pos[1]) ** 2)
                if dist <= NEARBY_OBJECT_RADIUS:
                    label = doc.get("label", "").lower().strip()
                    if label and label not in STRUCTURAL_BLACKLIST:
                        nearby.append(label)
        except Exception:
            pass
        return nearby

    def _infer_held_from_disappeared_objects(
            self, user_pos: dict, room_name: str,
            disappeared_threshold_sec: float = 60.0,
            user_id: str = "") -> str:
        if not user_pos:
            return "none"

        ux  = float(user_pos.get("x", 0))
        uz  = float(user_pos.get("z", 0))
        now = datetime.datetime.utcnow()

        try:
            held_doc = self.col_dynamic.find_one(
                {"held_by": user_id} if user_id else {},
                sort=[("interact_count", -1)]
            ) if user_id else None
            if held_doc:
                label = held_doc.get("label", "")
                if label and label not in STRUCTURAL_BLACKLIST:
                    print(f"[Layer0] held_by: {label}")
                    return label
        except Exception:
            pass

        try:
            cutoff = now - datetime.timedelta(seconds=disappeared_threshold_sec)
            query = {
                "last_seen":    {"$lt": cutoff},
                "last_seen_on": {"$exists": True, "$nin": ["", "Unknown_Area"]},
            }
            if room_name:
                query["room"] = {"$regex": room_name, "$options": "i"}

            candidates = list(self.col_dynamic.find(
                query,
                {"label": 1, "last_seen_on": 1, "interact_count": 1, "last_seen": 1}
            ).sort("interact_count", -1).limit(10))

            for obj in candidates:
                furniture_label = obj.get("last_seen_on", "")
                if not furniture_label:
                    continue
                furniture_doc = self.col_scene.find_one(
                    {"label": furniture_label}, {"pos": 1})
                if not furniture_doc:
                    continue
                pos = furniture_doc.get("pos")
                if not isinstance(pos, list) or len(pos) < 2:
                    continue
                dist = math.sqrt((ux - pos[0])**2 + (uz - pos[1])**2)
                if dist <= 2.0:
                    label = obj.get("label", "")
                    if label and label not in STRUCTURAL_BLACKLIST:
                        secs = (now - obj["last_seen"]).total_seconds()
                        print(f"[Layer0] disappeared: {label} "
                              f"({secs:.0f}s ago on {furniture_label},"
                              f" person dist={dist:.2f}m)")
                        return label
        except Exception as e:
            print(f"[Layer0] error: {e}")

        return "none"

    def _infer_held_object(self, nearest_furniture: str,
                            zone_name: str,
                            room_name: str,
                            user_pos: dict = None,
                            user_id: str = "") -> str:
        if user_pos:
            _disappeared = self._infer_held_from_disappeared_objects(
                user_pos, room_name, user_id=user_id)
            if _disappeared and _disappeared != "none":
                return _disappeared

        if nearest_furniture and nearest_furniture != "Unknown_Area":
            try:
                docs = list(self.col_dynamic.find(
                    {"last_seen_on": nearest_furniture},
                    {"label": 1, "interact_count": 1}
                ).sort("interact_count", -1).limit(3))
                if docs:
                    label = docs[0].get("label", "")
                    if label and label not in STRUCTURAL_BLACKLIST:
                        print(f"[Layer1] dynamic: {nearest_furniture} -> {label}")
                        return label
            except Exception as e:
                print(f"[Layer1] error: {e}")

        if zone_name:
            try:
                docs = list(self.col_obs.find(
                    {"zone_name": zone_name,
                     "interacting_items": {"$exists": True, "$ne": []}},
                    {"interacting_items": 1, "weight": 1}
                ).sort("weight", -1).limit(10))
                if docs:
                    item_counts = Counter()
                    for doc in docs:
                        for item in doc.get("interacting_items", []):
                            item_counts[item] += doc.get("weight", 1)
                    if item_counts:
                        top1, top1_count = item_counts.most_common(1)[0]
                        if top1_count >= 1:
                            print(f"[Layer2] obs_log: {zone_name} -> {top1}")
                            return top1
            except Exception as e:
                print(f"[Layer2] error: {e}")

        if nearest_furniture and nearest_furniture != "Unknown_Area":
            try:
                furn_key = nearest_furniture.lower().strip()
                action_scores = self.scene_engine._affinity_matrix.get(furn_key, {})
                if action_scores:
                    top_action = max(action_scores, key=action_scores.get)
                    matched_items = [
                        item for item, action in ITEM_TO_ACTION.items()
                        if action == top_action
                    ]
                    if matched_items:
                        result = matched_items[0]
                        print(f"[Layer3] ontology: {furn_key} -> {top_action} -> {result}")
                        return result
            except Exception as e:
                print(f"[Layer3] error: {e}")

        return "none"

    def _compute_p_l(self, obj_label: str, behavior: str) -> float:
        if not obj_label or obj_label == "none":
            return 0.0
        try:
            prototype = VISION_PROTOTYPES.get(behavior, behavior)
            obj_vec   = self.sbert.encode([obj_label], normalize_embeddings=True)[0]
            beh_vec   = self.sbert.encode([prototype], normalize_embeddings=True)[0]
            return float(max(0.0, obj_vec @ beh_vec))
        except Exception:
            return 0.0

    def _compute_p_v(self, body_position: str, behavior: str, zone_affinity: float = 0.0) -> float:
        pos = body_position.lower().strip() if body_position else "standing"
        if (pos, behavior) in BODY_IMPOSSIBLE:
            return 0.0
        ergonomic = 0.50
        return 0.6 * ergonomic + 0.4 * float(zone_affinity)

    def _saycan_score(self, held_obj: str, nearby_objs: list,
                      body_position: str, behavior: str,
                      zone_affinity: float = 0.0) -> float:
        if (body_position.lower(), behavior) in BODY_IMPOSSIBLE:
            return 0.0
        p_l_held   = self._compute_p_l(held_obj, behavior)
        p_l_nearby = 0.0
        if nearby_objs:
            p_l_nearby = max(self._compute_p_l(obj, behavior) for obj in nearby_objs)
        if held_obj and held_obj != "none":
            p_l = HELD_WEIGHT * p_l_held + NEARBY_WEIGHT * p_l_nearby
        else:
            p_l = p_l_nearby
        p_v = self._compute_p_v(body_position, behavior, zone_affinity)
        return p_l * p_v

    def _layer3a_saycan(self, held_obj: str, nearby_objs: list,
                        body_position: str, nearest_zone: dict) -> tuple:
        scores = {}
        for behavior in BEHAVIOR_LABELS:
            if behavior in NO_WEIGHT_ACTIONS:
                continue
            z_aff = self.scene_engine.get_zone_affinity(nearest_zone, behavior) \
                    if nearest_zone else 0.0
            scores[behavior] = self._saycan_score(
                held_obj, nearby_objs, body_position, behavior, z_aff)

        if not scores:
            return None, "", 0.0

        best_behavior = max(scores, key=scores.get)
        best_score    = scores[best_behavior]

        if best_score < SAYCAN_MIN_SCORE:
            return None, "", best_score

        held_info = held_obj if held_obj != "none" else f"nearby:{nearby_objs[:1]}"
        reason = f"L3a_saycan:{held_info}->{best_behavior}(score={best_score:.2f})"
        return best_behavior, reason, best_score

    def _layer3b_heading(self, user_pos: dict, user_forward: dict,
                          nearest_zone: dict, body_position: str) -> tuple:
        if not user_pos or not user_forward or not nearest_zone:
            return None, ""

        ux = float(user_pos.get("x", 0))
        uz = float(user_pos.get("z", 0))
        cx = nearest_zone["center"][0]
        cz = nearest_zone["center"][1]

        dx, dz = cx - ux, cz - uz
        dist = math.sqrt(dx * dx + dz * dz)
        if dist < 0.01:
            return None, ""

        dx /= dist
        dz /= dist

        fwd_x = float(user_forward.get("x", 0))
        fwd_z = float(user_forward.get("z", 0))
        fwd_len = math.sqrt(fwd_x * fwd_x + fwd_z * fwd_z)
        if fwd_len < 0.01:
            return None, ""

        fwd_x /= fwd_len
        fwd_z /= fwd_len

        heading = float(fwd_x * dx + fwd_z * dz)
        if heading < HEADING_THRESHOLD:
            return None, ""

        zone_furniture = set(f.lower() for f in nearest_zone.get("furniture", []))
        is_high_aff    = bool(zone_furniture & HIGH_AFFORDANCE_FURNITURE)

        if not is_high_aff and heading < 0.65:
            return None, ""

        action_label = nearest_zone.get("action_label", "")
        if not action_label or action_label in NO_WEIGHT_ACTIONS:
            return None, ""

        if (body_position.lower(), action_label) in BODY_IMPOSSIBLE:
            return None, ""

        reason = (
            f"L3b_heading:{nearest_zone['zone_name']}"
            f"->{action_label}(cos={heading:.2f})"
        )
        return action_label, reason

    def _layer3b5_proximity(self, user_pos: dict,
                             room_name: str,
                             body_position: str,
                             nearest_zone: dict = None) -> tuple:
        if not user_pos:
            return None, ""

        ux = float(user_pos.get("x", 0))
        uz = float(user_pos.get("z", 0))

        INTERACTION_RADIUS = 1.5
        MIN_AFFINITY       = 0.40

        query = {"room": {"$regex": room_name, "$options": "i"}} if room_name else {}
        docs = list(self.col_scene.find(query, {"label": 1, "pos": 1}))
        if not docs:
            docs = list(self.col_scene.find({}, {"label": 1, "pos": 1}))

        best_behavior = None
        best_score    = MIN_AFFINITY
        best_label    = ""
        best_dist     = float("inf")

        for doc in docs:
            pos = doc.get("pos")
            if not isinstance(pos, list) or len(pos) < 2:
                continue
            dist = math.sqrt((ux - pos[0])**2 + (uz - pos[1])**2)
            if dist > INTERACTION_RADIUS:
                continue

            furn_label    = doc.get("label", "").lower().strip()
            action_scores = self.scene_engine._affinity_matrix.get(furn_label, {})

            for behavior, score in action_scores.items():
                if behavior in NO_WEIGHT_ACTIONS:
                    continue
                if (body_position.lower(), behavior) in BODY_IMPOSSIBLE:
                    continue
                if score > best_score or (score == best_score and dist < best_dist):
                    best_score    = score
                    best_behavior = behavior
                    best_label    = furn_label
                    best_dist     = dist

        # Zone action_label 覆蓋：
        # 當 proximity 給出 Sitting，但當前 zone 有明確的非 Sitting 行為
        # 讓 zone 的 action_label 覆蓋（解決 Typing_Zone 被 chair→Sitting 蓋掉的問題）
        if best_behavior in ("Sitting", "Eating") and nearest_zone:
            zone_action = nearest_zone.get("action_label", "")
            if (zone_action
                    and zone_action not in ("Sitting", "Standing", "Walking", "")
                    and zone_action in BEHAVIOR_LABELS
                    and zone_action not in NO_WEIGHT_ACTIONS
                    and (body_position.lower(), zone_action) not in BODY_IMPOSSIBLE):
                best_behavior = zone_action
                reason = (f"L3b5_proximity:{best_label}"
                          f"->{best_behavior}"
                          f"(zone_override,dist={best_dist:.2f}m)")
                return best_behavior, reason

        if best_behavior:
            reason = (f"L3b5_proximity:{best_label}"
                      f"->{best_behavior}"
                      f"(aff={best_score:.2f},dist={best_dist:.2f}m)")
            return best_behavior, reason

        return None, ""

    def _layer3c_zone_affinity(self, nearest_zone: dict,
                                body_position: str) -> tuple:
        if not nearest_zone:
            return None, ""

        zone_furniture = set(f.lower() for f in nearest_zone.get("furniture", []))
        is_high_aff    = bool(zone_furniture & HIGH_AFFORDANCE_FURNITURE)
        threshold      = HIGH_AFFORDANCE_L3_THRESHOLD if is_high_aff else L3_STANDARD_THRESHOLD

        best_behavior = None
        best_aff      = threshold

        for behavior in BEHAVIOR_LABELS:
            if behavior in NO_WEIGHT_ACTIONS:
                continue
            if (body_position.lower(), behavior) in BODY_IMPOSSIBLE:
                continue
            aff = self.scene_engine.get_zone_affinity(nearest_zone, behavior)
            if aff > best_aff:
                best_aff      = aff
                best_behavior = behavior

        if not best_behavior:
            return None, ""

        reason = (
            f"L3c_zone:{nearest_zone['zone_name']}"
            f"->{best_behavior}(aff={best_aff:.2f})"
        )
        return best_behavior, reason

    def _temporal_smooth(self, new_action: str,
                          evidence_score: float,
                          user_id: str,
                          hip_height: float = -1) -> str:
        prev_doc = self.col_activity.find_one(
            {"user": user_id}, sort=[("date", -1)])
        if not prev_doc or not prev_doc.get("sequence"):
            return new_action
        seq = prev_doc["sequence"]
        if not seq:
            return new_action

        prev_action = seq[-1].get("action", "Standing")
        if prev_action == new_action:
            return new_action

        # 計算連續相同行為的幀數（行為慣性）
        n_consecutive = 0
        for entry in reversed(seq):
            if entry.get("action") == prev_action:
                n_consecutive += 1
            else:
                break

        # 骨架 hip 劇烈變化 → 重置慣性，允許立即轉換
        # 例如：從坐到站，hip 變化 > 0.15m，允許行為立即轉換
        if hip_height > 0:
            prev_hip_entries = [
                e for e in reversed(seq[-3:])
                if e.get("hip_height") is not None
            ]
            if prev_hip_entries:
                prev_hip = float(prev_hip_entries[0].get("hip_height", hip_height))
                hip_delta = abs(hip_height - prev_hip)
                if hip_delta > 0.15:
                    n_consecutive = 0  # 重置慣性

        # 動態門檻：N 幀越多，轉換需要越高的信心
        if n_consecutive < 3:
            conf_threshold = 0.55   # 自由轉換
        elif n_consecutive < 5:
            conf_threshold = 0.65   # 需要較高信心
        else:
            conf_threshold = 0.75   # 需要很高信心

        trans_matrix = getattr(self.scene_engine, "_transition_matrix", {})
        prob = trans_matrix.get(prev_action, {}).get(new_action, 0.0)

        if prob >= 0.05:
            if evidence_score >= conf_threshold:
                return new_action
            return prev_action
        else:
            # 轉移矩陣不支持此轉換，需要更高信心
            if evidence_score >= max(conf_threshold + 0.10, 0.80):
                return new_action
            return prev_action

    def _layer3b_heading_extended(self, user_pos: dict,
                                   user_forward: dict,
                                   body_position: str) -> tuple:
        if not user_pos or not user_forward:
            return None, ""

        ux    = float(user_pos.get("x", 0))
        uz    = float(user_pos.get("z", 0))
        fwd_x = float(user_forward.get("x", 0))
        fwd_z = float(user_forward.get("z", 0))
        fwd_len = math.sqrt(fwd_x**2 + fwd_z**2)
        if fwd_len < 0.01:
            return None, ""
        fwd_x /= fwd_len
        fwd_z /= fwd_len

        target_map = {
            "tv":           "Watching",
            "television":   "Watching",
            "monitor":      "Watching",
            "stove":        "Cooking",
            "refrigerator": "Opening",
            "fridge":       "Opening",
        }

        best_action = None
        best_reason = ""
        best_angle  = 30.0

        for label, action in target_map.items():
            if (body_position.lower(), action) in BODY_IMPOSSIBLE:
                continue
            doc = self.col_scene.find_one({"label": label}, {"pos": 1})
            if not doc:
                continue
            pos = doc.get("pos")
            if not isinstance(pos, list) or len(pos) < 2:
                continue
            dx   = pos[0] - ux
            dz   = pos[1] - uz
            dist = math.sqrt(dx**2 + dz**2)
            if dist < 0.01 or dist > 5.0:
                continue
            dx /= dist
            dz /= dist
            cos_angle = fwd_x * dx + fwd_z * dz
            angle_deg = math.degrees(math.acos(max(-1.0, min(1.0, cos_angle))))
            if angle_deg < best_angle:
                best_angle  = angle_deg
                best_action = action
                best_reason = (f"ray_cast:{label}"
                               f"->{action}"
                               f"(angle={angle_deg:.1f}°,dist={dist:.1f}m)")

        return best_action, best_reason

    def _minimum_prior_defense(self, body_position: str, held_obj: str,
                                head_pitch: float = -999,
                                virtual_hour: float = None) -> tuple:
        pos = body_position.lower().strip() if body_position else "standing"

        if pos == "lying":
            return "Laying", "MinPrior:lying→Laying"

        if pos == "sitting":
            # 1. head_pitch 優先：仰頭喝飲料
            if head_pitch != -999 and head_pitch < -15:
                return "SittingDrink", f"MinPrior:sitting+head{head_pitch:.0f}°→SittingDrink"

            # 2. held_object 次之
            if held_obj and held_obj != "none":
                action_hint = ITEM_TO_ACTION.get(held_obj.lower(), "")
                if action_hint == "Drinking":
                    return "SittingDrink", "MinPrior:sitting+drink→SittingDrink"
                if action_hint in ("Eating", "Cooking"):
                    return "Eating", "MinPrior:sitting+food→Eating"
                if action_hint == "Reading":
                    return "Reading", "MinPrior:sitting+book→Reading"
                if action_hint == "PhoneUse":
                    return "PhoneUse", "MinPrior:sitting+phone→PhoneUse"

            # 3. head_pitch Eating 候選
            if head_pitch != -999 and 15 <= head_pitch <= 28:
                return "Eating", f"MinPrior:sitting+head{head_pitch:.0f}°→Eating"

            # 4. 時間先驗（空間幾何無法決定時）
            if virtual_hour is not None:
                h = float(virtual_hour)
                if 7 <= h < 10 or 12 <= h < 13:
                    return "Eating", f"MinPrior:sitting+hour{h:.0f}→Eating"
                if 19 <= h < 22:
                    return "Watching", f"MinPrior:sitting+hour{h:.0f}→Watching"
                if h >= 23 or h < 6:
                    return "Laying", f"MinPrior:sitting+hour{h:.0f}→Laying"

            return "Sitting", "MinPrior:sitting→Sitting"

        if pos == "standing":
            return "Standing", "MinPrior:standing→Standing"

        return "Standing", "MinPrior:default→Standing"

    def _skeleton_body_position(self, payload: dict) -> tuple:
        hip_h      = float(payload.get("hip_height",    -1))
        head_pitch = float(payload.get("head_pitch",  -999))

        # 實測數據：
        #   Laying  hip=0.789-0.846  head=-82 到 -83°（躺下頭部水平）
        #   坐姿群  hip=0.690-0.742  head=0 到 38°
        #   站姿群  hip=0.895-0.993  head=-7 到 38°
        #
        # 關鍵發現：Sitting(0.690-0.742) < Laying(0.789-0.846)
        # 不能用 hip 單獨區分坐/躺，必須用 head_pitch 輔助
        # Laying head=-83°，Sitting head=0-5°，差距極大，可完全區分
        if hip_h < 0:
            skel_body = None
        elif hip_h >= 0.900:
            skel_body = "standing"
        elif hip_h < 0.900:
            if head_pitch < -45:
                skel_body = "lying"
            elif hip_h >= 0.880 and head_pitch > 45:
                skel_body = "standing"
            else:
                skel_body = "sitting"

        # head_pitch 實測數值：
        #   SittingDrink : -33 到 -15°（仰頭喝飲料，負值）
        #   Sitting      :   0 到  5°（平視）
        #   Watching     :   0°（平視電視）
        #   Eating       :  21°（略低頭）
        #   PhoneUse     :  34-38°（低頭看手機）
        #   Cleaning     :  60-66°（大幅低頭）
        #   Standing     :  -1°
        head_hint = None
        if head_pitch > -999:  # 有效值
            if skel_body == "sitting":
                if head_pitch < -15:
                    head_hint = "SittingDrink"   # 仰頭喝飲料（-33 到 -15°）
                elif 15 <= head_pitch <= 28:
                    head_hint = "Eating"          # 略低頭看食物（21°）
                elif head_pitch > 55:
                    head_hint = "Reading"         # 大幅低頭看書（實測 56-87°）
                # Cleaning head=60-66° 是站姿，骨架已確認 standing，不會進此分支
                # 0-15° 平視：Watching / Sitting，交給幾何推理
                # Cleaning 是站姿（hip 0.895-0.940 邊界），head_pitch 60-66°
                # 若被誤判為 sitting，head>28° 會觸發 Reading，需要在幾何層否決
            elif skel_body == "standing":
                pass

        return skel_body, head_hint

    def _spatial_reasoning(self, activity_hint: str, body_position: str,
                            held_obj: str, user_pos: dict, user_forward: dict,
                            room_name: str, user_id: str,
                            vlm_confidence: float = 0.0,
                            payload: dict = None) -> tuple:

        if not self.scene_engine.is_ready():
            return activity_hint or "Unknown", "zone_not_ready", ""

        nearest_zone = self.scene_engine.find_nearest_zone(user_pos, room_name)
        zone_label   = nearest_zone["zone_name"] if nearest_zone else ""
        norm_held    = self._normalize_object(held_obj)

        # ── 骨架解析：hip_height 主導 body_position ────────────────
        # hip_height simulates MediaPipe normalized hip ratio (hip_y/body_height)
        # Decision authority: skeleton > VLM for body_position
        skel_body  = None
        head_pitch = float(payload.get("head_pitch", -999)) if payload else -999
        if payload:
            skel_body, _ = self._skeleton_body_position(payload)
            if skel_body in ("sitting", "standing", "lying"):
                body_position = skel_body
            elif skel_body is None:
                # hip_height unavailable: fall back to VLM body_position
                # In real deployment this means MediaPipe failed
                pass

        # ── 評分初始化 ──────────────────────────────────────────────
        scores  = {b: 0.0 for b in BEHAVIOR_LABELS}
        reasons = {b: ""  for b in BEHAVIOR_LABELS}

        # ── 層 0：BODY_IMPOSSIBLE 硬過濾 ────────────────────────────
        for b in BEHAVIOR_LABELS:
            if (body_position.lower(), b) in BODY_IMPOSSIBLE:
                scores[b] = -999.0

        # ── 層 1：骨架 head_pitch 匹配分 ────────────────────────────
        # 每個行為有理想 head_pitch 和容忍範圍，統一公式計算
        # (理想角度, 容忍範圍, 適用 body_position)
        HEAD_PITCH_PROFILE = {
            "SittingDrink": (-25,  8, "sitting"),
            "Eating":       ( 21,  8, "sitting"),
            "Reading":      ( 70, 18, "sitting"),
            "Sitting":      (  2,  8, "sitting"),
            "Watching":     (  0,  8, "sitting"),
            "Typing":       (  2, 12, "sitting"),
            "Laying":       (-83, 10, "lying"),
            "Cleaning":     ( 63,  8, "standing"),
            "Cooking":      ( 25,  8, "standing"),
            "Opening":      ( 32,  8, "standing"),
            "PhoneUse":     ( 36, 10, "standing"),
            "Drinking":     ( 24, 10, "standing"),
        }

        W_HEAD = 0.45
        if head_pitch > -999 and skel_body:
            for b, (ideal, tol, req_body) in HEAD_PITCH_PROFILE.items():
                if scores[b] <= -999:
                    continue
                if skel_body != req_body:
                    continue
                diff = abs(head_pitch - ideal)
                if diff <= tol:
                    gain = W_HEAD * (1.0 - diff / tol)
                    scores[b]  += gain
                    reasons[b] += f"head({head_pitch:.0f}°≈{ideal}°)+{gain:.2f} "
                else:
                    scores[b] -= W_HEAD * 0.3

        # ── 層 2：Laying 直接強命中 ─────────────────────────────────
        # lying 是唯一完全可靠的骨架判斷，給予最高分
        if skel_body == "lying" and scores.get("Laying", -999) > -999:
            scores["Laying"] += 0.90
            reasons["Laying"] += "skeleton_lying+0.90 "

        # ── 層 X：held_object 加分（最直接的行為依據）────────────────────
        # 來源：Layer 0 held_by / Layer 2 消失推斷 / Layer 3 VLM
        # SittingDrink 特殊：cup/bottle 在坐姿時對應 SittingDrink 而非 Drinking
        W_HELD = 0.55
        if norm_held and norm_held != "none":
            _item_action = ITEM_TO_ACTION.get(norm_held.lower(), "")
            if _item_action == "Drinking" and body_position == "sitting":
                _item_action = "SittingDrink"
            if (_item_action
                    and _item_action in scores
                    and scores.get(_item_action, -999) > -999
                    and _item_action not in NO_WEIGHT_ACTIONS):
                scores[_item_action]  += W_HELD
                reasons[_item_action] += f"held:{norm_held}→{_item_action}+{W_HELD} "

        # ── 層 X2：Nearby Dynamic Objects 加分 ──────────────────────────
        # 物件在用戶 1.5m 內（不需要消失）→ 推斷可能在使用
        # 對應真實場景：同事物件偵測系統看到鍵盤在用戶旁邊 → Typing 可能
        # 比 held_object 弱（W=0.40）因為在附近不等於在使用
        W_NEARBY = 0.25
        # user_pos fallback from user_positions if not in payload
        if not user_pos:
            _upos_doc = self.db.user_positions.find_one(
                {"$or": [
                    {"user_id": user_id},
                    {"user_id": user_id.lower()},
                ]},
                {"x": 1, "z": 1}
            )
            if _upos_doc:
                user_pos = {"x": float(_upos_doc.get("x", 0)),
                            "z": float(_upos_doc.get("z", 0))}
        if user_pos:
            _ux = float(user_pos.get("x", 0))
            _uz = float(user_pos.get("z", 0))
            _nearby_docs = list(self.col_dynamic.find(
                {"label": {"$exists": True},
                 "source": {"$ne": "unity_user"}},
                {"label": 1, "pos": 1, "position": 1, "sensor_pos": 1, "held_by": 1}
            ))
            for _ndoc in _nearby_docs:
                _npos = (_ndoc.get("pos")
                         or _ndoc.get("position")
                         or _ndoc.get("sensor_pos"))
                if not isinstance(_npos, list) or len(_npos) < 2:
                    continue
                _ndist = math.sqrt(
                    (_ux - _npos[0])**2 + (_uz - _npos[1])**2)
                if _ndist > 1.5:
                    continue
                _nlabel = self._normalize_object(
                    _ndoc.get("label", "").lower())
                _naction = ITEM_TO_ACTION.get(_nlabel, "")
                if (_naction and _naction in scores
                        and scores.get(_naction, -999) > -999
                        and _naction not in NO_WEIGHT_ACTIONS):
                    _proximity_factor = max(0.0, 1.0 - _ndist / 1.5)
                    _gain = W_NEARBY * _proximity_factor
                    # If held_object already scored this action, skip nearby
                    # to avoid double-counting the same evidence
                    _held_already = (
                        norm_held and norm_held != "none" and
                        ITEM_TO_ACTION.get(norm_held.lower(), "") == _naction
                    )
                    if not _held_already:
                        scores[_naction]  += _gain
                        reasons[_naction] += (
                            f"nearby:{_nlabel}→{_naction}+{_gain:.2f} ")

        # ── 層 3：幾何 Proximity 分 ──────────────────────────────────
        W_PROXIMITY = 0.30
        if user_pos:
            ux = float(user_pos.get("x", 0))
            uz = float(user_pos.get("z", 0))
            INTERACTION_RADIUS = 1.5
            query = {"room": {"$regex": room_name, "$options": "i"}} if room_name else {}
            docs  = list(self.col_scene.find(query, {"label": 1, "pos": 1}))
            if not docs:
                docs = list(self.col_scene.find({}, {"label": 1, "pos": 1}))

            for doc in docs:
                pos = doc.get("pos") or doc.get("sensor_pos")
                if not isinstance(pos, list) or len(pos) < 2:
                    continue
                dist = math.sqrt((ux - pos[0])**2 + (uz - pos[1])**2)
                if dist > INTERACTION_RADIUS:
                    continue
                furn_label    = doc.get("label", "").lower().strip()
                action_scores = self.scene_engine._affinity_matrix.get(furn_label, {})
                proximity_factor = max(0.0, 1.0 - dist / INTERACTION_RADIUS)
                for b, aff in action_scores.items():
                    if b in scores and scores[b] > -999 and b not in NO_WEIGHT_ACTIONS:
                        gain = W_PROXIMITY * aff * proximity_factor
                        scores[b]  += gain
                        reasons[b] += f"prox:{furn_label}({aff:.2f})+{gain:.2f} "

        # ── 層 4：Ray-cast 朝向分 ────────────────────────────────────
        W_RAYCAST = 0.25
        if user_pos and user_forward:
            ux    = float(user_pos.get("x", 0))
            uz    = float(user_pos.get("z", 0))
            fwd_x = float(user_forward.get("x", 0))
            fwd_z = float(user_forward.get("z", 0))
            fwd_len = math.sqrt(fwd_x**2 + fwd_z**2)
            if fwd_len > 0.01:
                fwd_x /= fwd_len
                fwd_z /= fwd_len
                # Check TV state: only boost Watching if TV is on
                _tv_doc   = self.db.device_states.find_one({"label": "tv"})
                _tv_on    = _tv_doc and _tv_doc.get("state") == "on"
                _tv_boost = 1.0 if _tv_on else 0.4

                target_map = {
                    "tv":           ("Watching", _tv_boost),
                    "television":   ("Watching", _tv_boost),
                    "stove":        ("Cooking",  1.0),
                    "refrigerator": ("Opening",  1.0),
                    "fridge":       ("Opening",  1.0),
                }
                for label, (action, tv_factor) in target_map.items():
                    if scores.get(action, -999) <= -999:
                        continue
                    doc = self.col_scene.find_one({"label": label}, {"pos": 1})
                    if not doc:
                        continue
                    pos = doc.get("pos")
                    if not isinstance(pos, list) or len(pos) < 2:
                        continue
                    dx   = pos[0] - ux
                    dz   = pos[1] - uz
                    dist = math.sqrt(dx**2 + dz**2)
                    if dist < 0.01 or dist > 5.0:
                        continue
                    dx /= dist
                    dz /= dist
                    cos_a = fwd_x * dx + fwd_z * dz
                    if cos_a > 0.50:
                        gain = W_RAYCAST * cos_a * tv_factor
                        scores[action]  += gain
                        reasons[action] += f"ray:{label}(cos={cos_a:.2f})+{gain:.2f} "

        # ── 層 5：Zone affinity 分 ───────────────────────────────────
        W_ZONE = 0.25
        if nearest_zone:
            for b in BEHAVIOR_LABELS:
                if scores.get(b, -999) <= -999 or b in NO_WEIGHT_ACTIONS:
                    continue
                aff = self.scene_engine.get_zone_affinity(nearest_zone, b)
                if aff > 0:
                    gain = W_ZONE * aff
                    scores[b]  += gain
                    reasons[b] += f"zone:{nearest_zone['zone_name']}({aff:.2f})+{gain:.2f} "

        # ── 層 6：VLM 加分（不跳過任何行為）────────────────────────
        W_VLM = 0.20
        if activity_hint and activity_hint in BEHAVIOR_LABELS:
            if (scores.get(activity_hint, -999) > -999
                    and activity_hint not in NO_WEIGHT_ACTIONS):
                gain = W_VLM * vlm_confidence
                scores[activity_hint]  += gain
                reasons[activity_hint] += f"vlm({vlm_confidence:.2f})+{gain:.2f} "

        # ── 層 7：時間先驗分 ─────────────────────────────────────────
        W_TIME = 0.10
        virtual_hour = float(payload.get("virtual_hour", -1)) if payload else -1
        if virtual_hour >= 0:
            h = virtual_hour
            time_boosts = {}
            if skel_body == "sitting":
                if 7 <= h < 10:
                    time_boosts["Eating"]   = 0.25
                if 12 <= h < 13:
                    time_boosts["Eating"]   = 0.20
                    time_boosts["Laying"]   = 0.15
                if 19 <= h < 22:
                    time_boosts["Watching"] = 0.30
                    time_boosts["Eating"]   = 0.15
                if h >= 23 or h < 6:
                    time_boosts["Laying"]   = 0.25
            if skel_body == "lying":
                if 12 <= h < 14:
                    time_boosts["Laying"]   = 0.20
                if h >= 23 or h < 6:
                    time_boosts["Laying"]   = 0.30
            for b, boost in time_boosts.items():
                if scores.get(b, -999) > -999:
                    gain = W_TIME * boost
                    scores[b]  += gain
                    reasons[b] += f"time({h:.0f}h)+{gain:.2f} "

        # ── 層 8：行為慣性分 ─────────────────────────────────────────
        W_TEMPORAL = 0.10
        prev_doc = self.col_activity.find_one(
            {"user": user_id}, sort=[("date", -1)])
        if prev_doc and prev_doc.get("sequence"):
            seq = prev_doc["sequence"]
            if seq:
                prev_action = seq[-1].get("action", "")
                n_consec = sum(
                    1 for e in reversed(seq)
                    if e.get("action") == prev_action
                )
                inertia = min(W_TEMPORAL * n_consec * 0.3, W_TEMPORAL)
                if prev_action in scores and scores[prev_action] > -999:
                    scores[prev_action]  += inertia
                    reasons[prev_action] += f"inertia(n={n_consec})+{inertia:.2f} "

                trans_matrix = getattr(self.scene_engine, "_transition_matrix", {})
                for b in BEHAVIOR_LABELS:
                    if scores.get(b, -999) <= -999:
                        continue
                    prob = trans_matrix.get(prev_action, {}).get(b, 0.0)
                    if prob > 0:
                        gain = W_TEMPORAL * prob * 0.5
                        scores[b] += gain

        # ── 最終決策：選最高分 ───────────────────────────────────────
        valid = {b: s for b, s in scores.items()
                 if s > -999 and b not in NO_WEIGHT_ACTIONS}
        if not valid:
            return "Standing", "MinPrior:no_valid", zone_label

        best_action = max(valid, key=valid.get)
        best_score  = valid[best_action]

        reason_str = reasons.get(best_action, "").strip()[:60]
        if not reason_str:
            reason_str = f"score:{best_score:.2f}"

        return best_action, reason_str, zone_label


    def _emit_edge(self, user_id: str, action: str,
                   bound_label: str, confidence: float,
                   user_pos: dict):
        if confidence < 0.55:
            return
        relation = ACTION_TO_RELATION.get(action, "near")
        now      = int(datetime.datetime.utcnow().timestamp())

        agent_node = {
            "id":             user_id,
            "type":           "agent",
            "label":          user_id,
            "pos":            [user_pos.get("x", 0), user_pos.get("z", 0)],
            "current_action": action,
            "status":         "active",
            "timestamp":      now,
        }

        furniture_doc = self.col_scene.find_one(
            {"label": bound_label}, {"pos": 1, "room": 1})
        furniture_node = {
            "id":     bound_label,
            "type":   "furniture",
            "label":  bound_label,
            "pos":    furniture_doc.get("pos", [0, 0]) if furniture_doc else [0, 0],
            "room":   furniture_doc.get("room", "") if furniture_doc else "",
            "status": "active",
        }

        edge = {
            "from":       user_id,
            "relation":   relation,
            "to":         bound_label,
            "confidence": round(confidence, 3),
            "source":     "spatial_reasoning",
            "timestamp":  now,
        }

        try:
            self.db.scene_graph.update_one(
                {"agent_id": user_id},
                {"$set": {
                    "agent_id":       user_id,
                    "agent_node":     agent_node,
                    "furniture_node": furniture_node,
                    "edges":          [edge],
                    "updated_at":     datetime.datetime.utcnow(),
                }},
                upsert=True
            )
        except Exception as e:
            print(f"[EmitEdge] {e}")

    def _get_user_id(self, img_b64, hint="Unknown_User"):
        if not self.face_app or not self.face_bank:
            return hint
        try:
            raw   = img_b64.split(',')[1] if ',' in img_b64 else img_b64
            nparr = np.frombuffer(base64.b64decode(raw), np.uint8)
            img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            faces = self.face_app.get(img)
            if not faces: return hint
            face = sorted(
                faces,
                key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]),
                reverse=True)[0]
            best, max_sim = hint, 0.0
            for name, known in self.face_bank.items():
                sim = float(np.dot(face.normed_embedding, known))
                if sim > max_sim: max_sim, best = sim, name
            return best if max_sim > 0.40 else hint
        except Exception:
            return hint

    def _nearest_by_coord(self, user_pos, room_name, max_dist=3.0):
        if not user_pos: return None, float('inf')
        ux, uz = user_pos.get("x", 0), user_pos.get("z", 0)
        query = {"room": {"$regex": room_name, "$options": "i"}} if room_name else {}
        docs = list(self.col_scene.find(query, {"label":1,"pos":1,"room":1,"x":1,"z":1}))
        if not docs:
            docs = list(self.col_scene.find({}, {"label":1,"pos":1,"room":1,"x":1,"z":1}))
        best_doc, best_dist = None, float('inf')
        for doc in docs:
            pos = doc.get("pos")
            if isinstance(pos, list) and len(pos) >= 2:
                fx, fz = pos[0], pos[1]
            elif doc.get("x") is not None:
                fx, fz = doc.get("x", 0), doc.get("z", 0)
            else:
                continue
            dist = math.sqrt((ux-fx)**2 + (uz-fz)**2)
            if dist < best_dist: best_dist, best_doc = dist, doc
        return (best_doc, best_dist) if best_doc and best_dist <= max_dist \
               else (None, best_dist)

    def _sem_match_furniture(self, label, k=3):
        norm  = normalize_label(label)
        exact = self.col_scene.find_one({"label": norm})
        if exact: return [(exact, 1.0)]
        return self.room_cache.bind_topk(norm, k=k, threshold=SEMANTIC_THRESHOLD)

    def _verify_with_coord(self, topk, sensor_pos):
        if not sensor_pos or not topk:
            return topk[0][0] if topk else None, "sbert_only"
        ox, oz = sensor_pos[0], sensor_pos[1]
        best_doc, best_dist, best_score = None, float('inf'), 0.0
        for doc, score in topk:
            pos = doc.get("pos")
            if isinstance(pos, list) and len(pos) >= 2:
                dist = math.sqrt((ox-pos[0])**2 + (oz-pos[1])**2)
                if dist < best_dist:
                    best_dist, best_doc, best_score = dist, doc, score
        if best_doc is None: return topk[0][0], "sbert_only"
        if best_dist <= COORD_MATCH_DIST:
            return best_doc, "high" if best_score >= 0.7 else "medium"
        if best_dist <= COORD_VERIFY_DIST:
            return best_doc, "coord_ok"
        return None, "coord_fallback"

    def _bind_furniture(self, vlm_label, user_pos, room_name):
        topk                  = self._sem_match_furniture(vlm_label, k=3)
        coord_doc, coord_dist = self._nearest_by_coord(user_pos, room_name)
        if not topk and not coord_doc: return None, "unknown"
        if topk and user_pos:
            doc, conf = self._verify_with_coord(
                topk, [user_pos.get("x", 0), user_pos.get("z", 0)])
            if doc: return doc, conf
        if coord_doc and coord_dist < 1.5: return coord_doc, "coord_priority"
        if coord_doc: return coord_doc, "coord_only"
        if topk: return topk[0][0], "sbert_low"
        return None, "unknown"

    def _select_sample_indices(self, image_list, node_scores, max_samples=3):
        n = len(image_list)
        if n <= max_samples: return list(range(n))
        if node_scores and len(node_scores) == n:
            return sorted(range(n),
                          key=lambda i: node_scores[i], reverse=True)[:max_samples]
        step = n / max_samples
        return [int(i * step) for i in range(max_samples)]

    def _empty_result(self, user_id):
        return {
            "user": user_id, "action": "none", "result": {},
            "items": [], "all_items": [], "spatial": [],
            "bound_instance": "Unknown_Area", "bound_room": "", "confidence": "unknown",
        }

    def _extract_json(self, raw):
        cleaned = re.sub(r'```(?:json)?\s*', '', raw).strip()
        m = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if not m: return cleaned
        text = m.group(0)
        try:
            json.loads(text)
            return text
        except json.JSONDecodeError:
            for i in range(len(text)-1, 0, -1):
                if text[i] in (',', '{'):
                    candidate = text[:i].rstrip(',') + '}}'
                    try:
                        json.loads(candidate)
                        return candidate
                    except Exception:
                        continue
            return text

    def analyze_action_burst(self, payload: dict) -> dict:
        image_list   = payload.get("image_list", [])
        hint_user_id = payload.get("userID", "Unknown_User")
        source_nodes = payload.get("source_nodes", [])
        node_scores  = payload.get("node_scores", [])
        user_pos     = payload.get("user_pos", None)
        user_forward = payload.get("user_forward", None)
        # Fallback: read user_forward from user_positions (DynamicSyncManager)
        # unity_user records are stored in user_positions, not dynamic_objects
        if not user_forward or (
            float(user_forward.get("x", 0)) == 0 and
            float(user_forward.get("z", 0)) == 0
        ):
            _col_upos = self.db.user_positions
            _udoc = _col_upos.find_one(
                {"$or": [
                    {"user_id": hint_user_id},
                    {"user_id": hint_user_id.lower()},
                    {"user_id": hint_user_id.lower().replace("_", "")},
                ]},
                {"forward": 1, "x": 1, "z": 1}
            )
            if _udoc and _udoc.get("forward"):
                fwd = _udoc["forward"]
                if isinstance(fwd, list) and len(fwd) >= 3:
                    user_forward = {"x": fwd[0], "y": fwd[1], "z": fwd[2]}
                elif isinstance(fwd, dict):
                    user_forward = fwd
            # Also update user_pos if not in payload
            if not user_pos and _udoc:
                user_pos = {"x": float(_udoc.get("x", 0)),
                            "z": float(_udoc.get("z", 0))}
        room_name    = payload.get("room_name", "")
        virtual_hour = payload.get("virtual_hour", None)
        virtual_day  = payload.get("virtual_day", None)

        if not image_list:
            return self._empty_result(hint_user_id)

        self.room_cache.switch_room(room_name, self.col_scene)
        if not self.room_cache.all_docs:
            self.room_cache._room = None
            self.room_cache.switch_room("", self.col_scene)

        coord_doc, coord_dist = self._nearest_by_coord(user_pos, room_name)
        coord_label    = coord_doc.get("label", "") if coord_doc else ""
        room_furniture = [d.get("label", "") for d in self.room_cache.all_docs if d.get("label")]
        prompt         = self._build_prompt(room_name, room_furniture, coord_label, coord_dist)
        sample_indices = self._select_sample_indices(image_list, node_scores)

        user_votes        = []
        parsed_list       = []
        activity_votes    = []
        body_votes        = []
        held_votes        = []
        orientation_votes = []
        confidence_list   = []
        used_scores_list  = []

        for idx in sample_indices:
            try:
                img_b64   = image_list[idx]
                uid       = self._get_user_id(img_b64, hint_user_id)
                user_votes.append(uid)
                img_clean = img_b64.split(',')[1] if ',' in img_b64 else img_b64
                resp = requests.post(
                    f"{self.url}/api/chat",
                    json={
                        "model":    self.model,
                        "messages": [{"role": "user", "content": prompt,
                                      "images": [img_clean]}],
                        "stream":   False,
                        "options":  {"temperature": 0.05, "num_predict": 200},
                    },
                    timeout=120)
                raw    = resp.json().get("message", {}).get("content", "").strip()
                parsed = self._parse_vlm_output(raw)
                if parsed:
                    parsed_list.append(parsed)
                    used_scores_list.append(
                        node_scores[idx] if idx < len(node_scores) else 0.5)
                    if parsed.get("activity"):
                        activity_votes.append(parsed["activity"])
                    if parsed.get("body_position"):
                        body_votes.append(parsed["body_position"])
                    if parsed.get("held_object"):
                        held_votes.append(parsed["held_object"])
                    if parsed.get("body_orientation"):
                        orientation_votes.append(parsed["body_orientation"])
                    confidence_list.append(parsed.get("confidence", 0.5))
            except Exception as e:
                print(f"[Frame {idx}] error: {e}")

        if not parsed_list:
            return self._empty_result(
                max(set(user_votes), key=user_votes.count) if user_votes else hint_user_id)

        def _weighted_vote(votes: list, scores: list, default: str = "") -> str:
            if not votes:
                return default
            weights = defaultdict(float)
            for v, s in zip(votes, scores):
                weights[v] += max(float(s), 0.1)
            return max(weights, key=weights.get)

        final_user       = max(set(user_votes), key=user_votes.count) \
                           if user_votes else hint_user_id
        activity_hint    = _weighted_vote(activity_votes, used_scores_list, default="")
        body_position    = _weighted_vote(body_votes,     used_scores_list, default="standing")
        held_object      = _weighted_vote(held_votes,     used_scores_list, default="none")
        body_orientation = _weighted_vote(orientation_votes, used_scores_list,
                                          default="facing_toward")
        vlm_confidence   = float(sum(confidence_list) / max(len(confidence_list), 1))

        if held_object in ("", "null", "nothing", "empty", "n/a"):
            held_object = "none"

        _nearest_zone_tmp = self.scene_engine.find_nearest_zone(user_pos, room_name)
        _zone_name_tmp    = _nearest_zone_tmp["zone_name"] if _nearest_zone_tmp else ""

        _infer_source = "vlm"

        if body_orientation == "facing_away":
            vlm_confidence = 0.0
            _infer_source  = "low_conf_facing_away"
            if held_object in ("none", "unknown"):
                _inferred = self._infer_held_object(
                    nearest_furniture = coord_label,
                    zone_name         = _zone_name_tmp,
                    room_name         = room_name,
                    user_pos          = user_pos,
                    user_id           = final_user,
                )
                if _inferred and _inferred != "none":
                    held_object   = _inferred
                    _infer_source = "three_layer_facing_away"

        elif held_object in ("none", "unknown") and \
                vlm_confidence < VLM_CONFIDENCE_THRESHOLD:
            _inferred = self._infer_held_object(
                nearest_furniture = coord_label,
                zone_name         = _zone_name_tmp,
                room_name         = room_name,
                user_pos          = user_pos,
                user_id           = final_user,
            )
            if _inferred and _inferred != "none":
                held_object   = _inferred
                _infer_source = "three_layer_low_conf"

        print(f"[VLM] activity={activity_hint} body={body_position} "
              f"orient={body_orientation} held={held_object} "
              f"conf={vlm_confidence:.2f} src={_infer_source}")

        spatial_action, upgrade_reason, zone_label = self._spatial_reasoning(
            activity_hint  = activity_hint,
            body_position  = body_position,
            held_obj       = held_object,
            user_pos       = user_pos,
            user_forward   = user_forward,
            room_name      = room_name,
            user_id        = final_user,
            vlm_confidence = vlm_confidence,
            payload        = payload,
        )

        bound_doc, confidence = self._bind_furniture(coord_label, user_pos, room_name)
        bound_label = bound_doc.get("label", "Unknown_Area") if bound_doc else "Unknown_Area"
        bound_room  = bound_doc.get("room", room_name) if bound_doc else room_name

        interacting_items = []
        norm_held = self._normalize_object(held_object)
        if norm_held and norm_held != "none":
            interacting_items.append(norm_held)

        nearby_objs = self._get_nearby_objects(user_pos, room_name)

        result = {
            "location":          bound_label,
            "room":              bound_room,
            "interacting_items": interacting_items,
            "scene_items":       nearby_objs,
            "all_items":         list(set(interacting_items + nearby_objs)),
            "spatial_relations": [],
            "context":           f"{final_user} {spatial_action} near {bound_label}",
            "_body_position":    body_position,
            "_held_object":      held_object,
            "_activity_hint":    activity_hint,
            "_coord_label":      coord_label,
            "_coord_dist":       round(coord_dist, 2) if coord_dist != float('inf') else None,
            "_confidence":       confidence,
            "_vlm_confidence":   round(vlm_confidence, 3),
            "_body_orientation": body_orientation,
            "_infer_source":     _infer_source,
        }

        self._update_scene_snapshot(bound_doc, interacting_items, nearby_objs, [])
        self._update_dynamic_objects(
            user_id=final_user,
            interacting_items=interacting_items,
            scene_items=nearby_objs,
            spatial_relations=[],
            bound_doc=bound_doc,
            room_name=bound_room,
            user_pos=user_pos)
        self._write_semantic_memory(final_user, spatial_action, bound_doc, confidence, result, source_nodes)
        self._update_activity_sequence(final_user, spatial_action, bound_label)

        ground_truth_activity = payload.get("activity", None)

        log_action = spatial_action if spatial_action not in ("Unknown", "none", "") \
                     else activity_hint or "Unknown"

        if vlm_confidence >= MIN_WRITE_CONFIDENCE:
            self._update_observation_log(
                final_user, log_action, bound_doc,
                interacting_items, [],
                result["context"], virtual_hour, virtual_day,
                ground_truth_activity=ground_truth_activity,
                user_pos=user_pos,
                room_name=room_name)
        else:
            print(f"[Gate] conf={vlm_confidence:.2f} < {MIN_WRITE_CONFIDENCE} skip obs_log")

        mem_doc  = self.col_memory.find_one(
            {"user": final_user, "action": log_action},
            sort=[("timestamp", -1)])
        mongo_id = str(mem_doc["_id"]) if mem_doc else ""
        self._index_to_faiss(final_user, log_action, bound_doc, result, mongo_id)

        if ground_truth_activity:
            import uuid as _uuid
            self.db["eval_logs"].insert_one({
                "episode_id":        str(_uuid.uuid4()),
                "t_capture":         payload.get("t_capture", ""),
                "user":              final_user,
                "user_id":           final_user,
                "ground_truth":      ground_truth_activity,
                "vlm_output":        activity_hint,
                "spatial_action":    spatial_action,
                "upgrade_reason":    upgrade_reason,
                "zone_label":        zone_label,
                "body_position":     body_position,
                "body_orientation":  body_orientation,
                "held_object":       held_object,
                "vlm_confidence":    round(vlm_confidence, 3),
                "infer_source":      _infer_source,
                "room_name":         room_name,
                "user_pos":          user_pos,
                "interacting_items": interacting_items,
                "timestamp":         datetime.datetime.utcnow(),
            })

        if (spatial_action not in ("Unknown", "none", "") and
                bound_label != "Unknown_Area" and
                user_pos):
            self._emit_edge(
                user_id     = final_user,
                action      = spatial_action,
                bound_label = bound_label,
                confidence  = float(result.get("_vlm_confidence", 0.5)),
                user_pos    = user_pos,
            )

        print(f"[Done] {final_user} | hint={activity_hint} | "
              f"spatial={spatial_action} | reason={upgrade_reason} | "
              f"zone={zone_label} | bound={bound_label}")

        return {
            "user":              final_user,
            "action":            activity_hint,
            "spatial_action":    spatial_action,
            "upgrade_reason":    upgrade_reason,
            "zone_label":        zone_label,
            "result":            result,
            "items":             interacting_items,
            "all_items":         result["all_items"],
            "spatial_relations": [],
            "bound_instance":    bound_label,
            "bound_room":        bound_room,
            "confidence":        confidence,
            "sbert_sim":         0.0,
            "user_pos":          user_pos or {},
            "virtual_hour":      virtual_hour or 12.0,
            "room":              room_name or "",
            "experiment_mode":   payload.get("experiment_mode", "habit"),
            "time_slot":         _get_time_slot(virtual_hour),
            "virtual_day":       virtual_day,
        }

    def _update_scene_snapshot(self, bound_doc, interacting_items,
                                scene_items, spatial_relations):
        if not bound_doc: return
        all_items = list(set(interacting_items + scene_items))
        update_op = {
            "$addToSet": {"items": {"$each": all_items}},
            "$set": {
                "current_contents":  interacting_items,
                "spatial_relations": spatial_relations,
                "last_observation":  datetime.datetime.utcnow(),
            },
        }
        self.col_scene.update_one({"_id": bound_doc.get("_id")}, update_op)

    def _update_dynamic_objects(self, user_id, interacting_items, scene_items,
                                 spatial_relations, bound_doc, room_name, user_pos=None):
        now         = datetime.datetime.utcnow()
        bound_label = bound_doc.get("label", "Unknown_Area") if bound_doc else "Unknown_Area"
        bound_pos   = bound_doc.get("pos") if bound_doc else None

        def _upsert(label, is_interacting):
            if not label or label in STRUCTURAL_BLACKLIST:
                return
            if self.col_scene.find_one({"label": label}):
                return
            base_set = {
                "last_seen_on": bound_label,
                "spatial_rel":  "near",
                "room":         room_name,
                "last_seen":    now,
                "source":       "vlm",
                "status":       "active",
                "status_since": now,
            }
            if bound_pos:
                base_set["furniture_pos"] = bound_pos
            inc_ops = {"seen_count": 1}
            if is_interacting: inc_ops["interact_count"] = 1
            update_op = {
                "$inc":         inc_ops,
                "$set":         base_set,
                "$setOnInsert": {
                    "first_seen": now,
                },
            }
            if is_interacting:
                update_op["$addToSet"] = {"interacted_by": user_id}
            self._bulk_buf.upsert(label, update_op, now)

        for item in interacting_items: _upsert(item, True)
        for item in scene_items:       _upsert(item, False)

    def _write_semantic_memory(self, user, action, bound_doc,
                                confidence, result, source_nodes):
        instance = bound_doc.get("label", "Unknown_Area") if bound_doc else "Unknown_Area"
        room     = bound_doc.get("room", "") if bound_doc else ""
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

    def _index_to_faiss(self, user, action, bound_doc, result, mongo_id):
        instance = bound_doc.get("label", "Unknown") if bound_doc else "Unknown"
        pos_raw  = bound_doc.get("pos") if bound_doc else None
        pos_xy   = pos_raw if isinstance(pos_raw, list) else \
                   ([bound_doc.get("x", 0), bound_doc.get("z", 0)] if bound_doc else [0, 0])
        text = self.faiss_store.build_memory_text(
            user=user, action=action, instance=instance,
            interacting_items=result.get("interacting_items", []),
            all_items=result.get("all_items", []),
            spatial_relations=result.get("spatial_relations", []))
        self.faiss_store.add(text, {
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

    def _find_nearest_zone(self, user_pos, room_name=""):
        return self.scene_engine.find_nearest_zone(user_pos, room_name)

    def _compute_zone_affinity(self, zone, behavior):
        return self.scene_engine.get_zone_affinity(zone, behavior)

    def _is_ambiguous_zone(self, zone):
        return self.scene_engine.is_ambiguous(zone)

    def _update_activity_sequence(self, user, action, instance):
        today = datetime.datetime.utcnow().strftime("%Y-%m-%d")
        self.col_activity.update_one(
            {"user": user, "date": today},
            {
                "$push": {"sequence": {
                    "action":    action,
                    "instance":  instance,
                    "timestamp": datetime.datetime.utcnow().isoformat(),
                }},
                "$setOnInsert": {"user": user, "date": today},
            },
            upsert=True)

    def _update_observation_log(self, user, action, bound_doc,
                                 interacting_items, spatial_relations,
                                 raw_desc, virtual_hour=None, virtual_day=None,
                                 ground_truth_activity=None,
                                 user_pos=None, room_name=""):
        if not bound_doc: return
        if action in NO_WEIGHT_ACTIONS: return

        instance  = bound_doc.get("label", "Unknown")
        pos_raw   = bound_doc.get("pos")
        pos_xy    = pos_raw if isinstance(pos_raw, list) else \
                    [bound_doc.get("x", 0), bound_doc.get("z", 0)]
        time_slot = _get_time_slot(virtual_hour)
        today     = _virtual_day_to_date(virtual_day)

        try:
            nz = self._find_nearest_zone(user_pos, room_name)
            zone_name_for_log = nz["zone_name"] if nz else ""
        except Exception:
            zone_name_for_log = ""

        canonical_key = zone_name_for_log if zone_name_for_log else instance

        self.col_obs.find_one_and_update(
            {"user": user, "zone_name": canonical_key,
             "action": action, "time_slot": time_slot},
            {
                "$inc":         {"weight": 1},
                "$addToSet":    {"interacting_items": {"$each": interacting_items}},
                "$set":         {
                    "observed_relations": spatial_relations,
                    "pos":               pos_xy,
                    "room":              bound_doc.get("room", "").strip() if bound_doc else "",
                    "instance":          instance,
                    "last_seen":         datetime.datetime.utcnow(),
                    "last_date":         today,
                    "raw_vlm_desc":      raw_desc,
                },
                "$setOnInsert": {
                    "user":      user,
                    "zone_name": canonical_key,
                    "action":    action,
                    "time_slot": time_slot,
                },
            },
            upsert=True, return_document=ReturnDocument.AFTER,
        )
        self._write_habit_snapshot(user, action, canonical_key, zone_name_for_log, today)
        self._update_user_affinity(user, action, canonical_key, instance)

    def _update_user_affinity(self, user: str, action: str,
                               zone_name: str, instance: str):
        if not action or not user:
            return
        try:
            pipeline = [
                {"$match": {"user": user, "action": action}},
                {"$group": {"_id": "$zone_name", "total_weight": {"$sum": "$weight"}}},
            ]
            results = list(self.col_obs.aggregate(pipeline))
            total   = sum(r["total_weight"] for r in results)
            if total == 0:
                return
            for r in results:
                zone_key = r["_id"] or "Unknown_Zone"
                personal = r["total_weight"] / total
                self.col_user_aff.update_one(
                    {"user_id": user, "action": action, "zone": zone_key},
                    {"$set": {"affinity": round(personal, 4),
                              "updated_at": datetime.datetime.utcnow()}},
                    upsert=True,
                )
                today = datetime.datetime.utcnow().strftime("%Y-%m-%d")
                self.col_aff_history.update_one(
                    {"user_id": user, "action": action, "zone": zone_key, "date": today},
                    {"$set": {"affinity": round(personal, 4),
                              "timestamp": datetime.datetime.utcnow()}},
                    upsert=True,
                )
        except Exception as e:
            print(f"[UserAffinity] {e}")

    def _write_habit_snapshot(self, user: str, action: str,
                               canonical_key: str, zone_name: str, today: str):
        try:
            self.col_habit_snap.update_one(
                {"user": user, "action": action,
                 "canonical_key": canonical_key, "date": today},
                {
                    "$inc": {"daily_count": 1},
                    "$set": {"zone": zone_name or canonical_key},
                },
                upsert=True,
            )
        except Exception as e:
            print(f"[HabitSnap] {e}")

    def shutdown(self):
        try:
            self._bulk_buf.force_flush()
        except Exception:
            pass