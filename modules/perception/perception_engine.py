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

from dataclasses import dataclass, field
from collections import defaultdict, Counter
from pymongo import UpdateOne, ReturnDocument
from sentence_transformers import SentenceTransformer

from config import Config

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


_defs    = _load_config(Config.DEFINITIONS_YAML)
_objects = _load_config(Config.OBJECTS_YAML)

STRUCTURAL_BLACKLIST = set(_objects.get("structural_blacklist", [
    "wall", "floor", "ceiling", "wooden floor", "white wall", "window",
    "door", "ground", "concrete floor", "tile floor", "carpet", "baseboard",
]))

YOUR_OBJECTS = (
    set(_objects.get("coco_objects", [])) |
    set(_objects.get("item_to_action", {}).keys()) |
    {"remote", "remote control", "tv remote", "juice", "cola", "pan",
     "broom", "mop", "spatula", "bowl", "saladbowl", "cup", "bottle", "phone",
     "book", "laptop", "keyboard", "fork", "spoon", "plate", "food"}
)

LABEL_NORMALIZE_MAP = {
    str(k).lower(): str(v).strip('"').strip()
    for k, v in _objects.get("label_normalize_map", {}).items()
}

ITEM_TO_ACTION = {
    str(k).lower(): str(v).strip('"').strip()
    for k, v in _objects.get("item_to_action", {}).items()
}

OBJECT_VOCAB = list(dict.fromkeys(
    list(ITEM_TO_ACTION.keys()) +
    list(_objects.get("object_vocab", [])) +
    ["none", "remote", "book", "phone", "laptop", "broom", "mop",
     "cup", "glass", "bottle", "bowl", "fork", "spoon", "pan",
     "spatula", "keyboard", "magazine", "chopsticks", "smartphone"]
))

STRONG_HELD_ITEMS = {
    str(k).lower(): str(v).strip()
    for k, v in _defs.get("strong_held_items", {
        "broom": "Cleaning", "mop": "Cleaning",
        "pan": "Cooking", "spatula": "Cooking",
    }).items()
}

ROOM_IMPOSSIBLE = {
    str(room): list(behaviors)
    for room, behaviors in _defs.get("room_impossible", {}).items()
}

BEHAVIOR_LABELS = _defs.get("behavior_labels", [
    "Drinking", "SeatedDrinking", "Sitting", "Eating", "Cooking", "Opening",
    "Laying", "Watching", "Reading", "Cleaning", "UsingPhone",
    "Typing", "StandUp", "PickingUp", "PuttingDown", "Standing", "Walking",
])

NO_WEIGHT_ACTIONS = set(_defs.get("no_weight_actions",
    ["PickingUp", "PuttingDown", "Walking", "Standing", "StandUp"]))

VISION_PROTOTYPES = {
    k: v.strip()
    for k, v in _defs.get("vision_prototypes", {}).items()
}

HIGH_AFFORDANCE_FURNITURE = set(_objects.get("high_affordance_furniture", [
    "tv", "television", "stove", "oven", "refrigerator", "fridge",
    "keyboard", "monitor", "cabinet"
]))

NEARBY_OBJECT_RADIUS     = Config.NEARBY_OBJECT_RADIUS
HEADING_THRESHOLD        = Config.HEADING_THRESHOLD
NORMALIZE_THRESHOLD      = Config.NORMALIZE_THRESHOLD
SEMANTIC_THRESHOLD       = Config.SEMANTIC_THRESHOLD
COORD_MATCH_DIST         = Config.COORD_MATCH_DIST
COORD_VERIFY_DIST        = Config.COORD_VERIFY_DIST
BULK_WRITE_THRESHOLD     = Config.BULK_WRITE_THRESHOLD
BULK_WRITE_INTERVAL      = Config.BULK_WRITE_INTERVAL
VLM_CONFIDENCE_THRESHOLD = Config.VLM_CONFIDENCE_THRESHOLD
MIN_WRITE_CONFIDENCE     = Config.MIN_WRITE_CONFIDENCE
VLM_MAX_RETRIES          = Config.VLM_MAX_RETRIES
VLM_RETRY_DELAY          = Config.VLM_RETRY_DELAY


ACTION_TO_RELATION = {
    "Drinking":       "holding",
    "SeatedDrinking": "holding",
    "Sitting":        "sitting_on",
    "Eating":         "eating_at",
    "Cooking":        "using",
    "Opening":        "interacting_with",
    "Laying":         "lying_on",
    "Watching":       "watching",
    "Reading":        "holding",
    "Cleaning":       "using",
    "UsingPhone":     "holding",
    "Typing":         "using",
    "StandUp":        "near",
    "Standing":       "near",
    "Walking":        "near",
}

GT_NORMALIZE_MAP = {
    "seateddrinking": "Drinking",
    "dadreading":     "Reading",
    "dadcleaning":    "Cleaning",
    "dadphone":       "UsingPhone",
}

DEBUG_IMAGE_DIR = "debug_images"


def normalize_ground_truth(label: str) -> str:
    if not label:
        return label
    key = label.lower().strip().replace(" ", "").replace("_", "")
    return GT_NORMALIZE_MAP.get(key, label)


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
    try:
        day_int = int(virtual_day)
        if day_int >= 1:
            base = datetime.date(2025, 1, 1)
            return (base + datetime.timedelta(days=day_int - 1)).strftime("%Y-%m-%d")
    except (ValueError, TypeError):
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
        q_emb   = self.model.encode([label], normalize_embeddings=True)[0].astype("float32")
        sims    = self._embeddings @ q_emb
        top_idx = np.argsort(sims)[::-1][:k]
        return [(self._docs[i], float(sims[i])) for i in top_idx if float(sims[i]) >= threshold]

    @property
    def all_docs(self):
        return self._docs

    @property
    def current_room(self):
        return self._room


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
            if not self._pending:
                return
            ops = self._pending.copy()
            self._pending.clear()
            self._last_flush = time.time()
        try:
            self.col.bulk_write(ops, ordered=False)
        except Exception as e:
            print(f"[BulkWrite] Failed: {e}")

    def force_flush(self):
        self._flush()

    @property
    def pending_count(self):
        with self._lock:
            return len(self._pending)


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

        self._vlm_timeout = Config.VLM_TIMEOUT
        self._llm_timeout = Config.LLM_TIMEOUT
        self._llm_url     = ollama_url
        self._llm_model   = Config.LLM_MODEL

        self.col_scene    = db.scene_snapshots
        self.col_dynamic  = db.dynamic_objects
        self.col_raw      = db.raw_objects
        self.col_obs      = db.observation_logs
        self.col_activity = db.activity_sequences
        self.col_user_aff = db.user_spatial_affinity
        self.col_aff_hist = db.affinity_history

        self._proto_labels = self.scene_engine._proto_labels
        self._proto_vecs   = self.scene_engine._proto_vecs

        self._proto_vecs_sbert = None
        self._build_prototype_vecs()

        self._room_cache = RoomEmbeddingCache(self.sbert)
        self.room_cache  = RoomEmbeddingCache(self.sbert)
        self._bulk_buf   = BulkWriteBuffer(self.col_dynamic)
        self._lock       = threading.Lock()

    def _build_prototype_vecs(self):
        labels = list(VISION_PROTOTYPES.keys())
        texts  = [VISION_PROTOTYPES[l] for l in labels]
        if texts:
            self._proto_vecs_sbert      = self.sbert.encode(
                texts, normalize_embeddings=True).astype("float32")
            self._proto_behavior_labels = labels

    def _build_prompt(self, room_name: str, room_furniture: list,
                      coord_label: str, coord_dist: float,
                      som_text: str = "") -> str:
        nearby_str = coord_label if coord_label else "unknown"
        som_block  = f"\nLabeled objects:\n{som_text}" if som_text else ""
        return (
            f"Scene: {room_name} room. Person is near {nearby_str}.{som_block}\n\n"
            "Output ONLY valid JSON with these exact fields:\n"
            '{"visual_state":"...","key_object":"...","confidence":0.0}\n\n'
            "Rules:\n"
            "- visual_state: brief description of what the person is doing (max 10 words)\n"
            "- key_object: the most relevant object the person is interacting with, "
            "or 'none' if nothing\n"
            "- confidence: 0.0=unsure, 1.0=certain\n"
            "Focus ONLY on the person's activity and the object they are using.\n"
            "Output ONLY the JSON. No markdown."
        )

    def _parse_vlm_output(self, raw: str) -> dict:
        try:
            cleaned = re.sub(r'```(?:json)?\s*', '', raw).strip().rstrip('`')
            m       = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if not m:
                return {}
            data = json.loads(m.group(0))
        except Exception:
            return {}

        visual_state = data.get("visual_state", "").strip()
        key_object   = data.get("key_object",   "none").strip().lower()
        confidence   = float(data.get("confidence", 0.5))
        if key_object in ("", "unknown"):
            key_object = "none"
        return {
            "visual_state": visual_state,
            "key_object":   key_object,
            "confidence":   min(max(confidence, 0.0), 1.0),
        }

    def _call_vlm_with_retry(self, prompt: str, img_clean: str) -> dict:
        for attempt in range(VLM_MAX_RETRIES + 1):
            try:
                resp = requests.post(
                    f"{self.url}/api/chat",
                    json={
                        "model":    self.model,
                        "messages": [{"role": "user", "content": prompt,
                                      "images": [img_clean]}],
                        "stream":   False,
                        "options":  {"temperature": 0.05, "num_predict": 200},
                    },
                    timeout=self._vlm_timeout)
                raw    = resp.json().get("message", {}).get("content", "").strip()
                parsed = self._parse_vlm_output(raw)
                if parsed:
                    return parsed
                if attempt < VLM_MAX_RETRIES:
                    print(f"[VLM] empty parse attempt {attempt+1}, retrying...")
                    time.sleep(VLM_RETRY_DELAY)
            except requests.exceptions.Timeout:
                print(f"[VLM] timeout attempt {attempt+1}/{VLM_MAX_RETRIES+1}")
                if attempt < VLM_MAX_RETRIES:
                    time.sleep(VLM_RETRY_DELAY)
            except Exception as e:
                print(f"[VLM] error attempt {attempt+1}: {e}")
                if attempt < VLM_MAX_RETRIES:
                    time.sleep(VLM_RETRY_DELAY)
        print("[VLM] all retries exhausted")
        return {}

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
        ux     = float(user_pos.get("x", 0))
        uz     = float(user_pos.get("z", 0))
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
                dist  = math.sqrt((ux - pos[0]) ** 2 + (uz - pos[1]) ** 2)
                if dist <= NEARBY_OBJECT_RADIUS:
                    label = doc.get("label", "").lower().strip()
                    if label and label not in STRUCTURAL_BLACKLIST:
                        nearby.append(label)
        except Exception:
            pass
        return nearby

    def _get_som_objects(self, user_pos: dict, room_name: str,
                         user_id: str, held_event: str,
                         radius: float = 3.0) -> list:
        try:
            from modules.perception.som_marker import get_som_objects_from_db
            return get_som_objects_from_db(
                db=self.db, user_pos=user_pos, room_name=room_name,
                user_id=user_id, held_event=held_event, radius=radius)
        except Exception as e:
            print(f"[SoM] {e}")
            return []

    def _som_objects_from_2d(self, objects_2d: list) -> list:
        if not objects_2d:
            return []
        result = []
        for obj in objects_2d:
            label = obj.get("label", "").strip()
            if not label:
                continue
            result.append({
                "label":  label,
                "status": "held" if obj.get("held") else "nearby",
            })
        return result[:8]

    def _save_som_debug_image(self, marked_b64: str, episode_id: str,
                               ground_truth: str, user_id: str,
                               frame_idx: int) -> None:
        try:
            save_dir = os.path.join(DEBUG_IMAGE_DIR, "som")
            os.makedirs(save_dir, exist_ok=True)
            raw = marked_b64.split(',')[1] if ',' in marked_b64 else marked_b64
            img_bytes = base64.b64decode(raw)
            gt_clean   = (ground_truth or "Unknown").replace(" ", "")
            user_clean = (user_id or "Unknown").replace(" ", "")
            fname = f"{episode_id}_{gt_clean}_{user_clean}_f{frame_idx}.jpg"
            with open(os.path.join(save_dir, fname), "wb") as f:
                f.write(img_bytes)
        except Exception as e:
            print(f"[SoM Debug] save failed: {e}")

    def _get_held_object_from_scene(self, user_id: str,
                                    user_pos: dict = None,
                                    t_capture: str = None) -> tuple:
        if not t_capture:
            return "none", 0
        try:
            if isinstance(t_capture, str):
                capture_time = datetime.datetime.fromisoformat(
                    t_capture.replace('Z', '+00:00')).replace(tzinfo=None)
            else:
                capture_time = t_capture
            doc = self.db.object_events.find_one({
                "user":        user_id,
                "pickup_time": {
                    "$lte": capture_time,
                    "$gte": capture_time - datetime.timedelta(seconds=30),
                },
                "$or": [
                    {"putdown_time": None},
                    {"putdown_time": {"$gt": capture_time}},
                ],
            })
            if not doc:
                return "none", 0
            label       = doc.get("object")
            pickup_time = doc.get("pickup_time")
            age         = (capture_time - pickup_time).total_seconds()
            if age < 3:
                event_desc = f"just picked up {label}"
            elif age < 10:
                event_desc = f"holding {label} for {int(age)} seconds"
            else:
                event_desc = f"has been holding {label} for a while"
            print(f"[HeldObj] {user_id}: {event_desc}")
            return event_desc, age
        except Exception as e:
            print(f"[HeldObj] error: {e}")
            return "none", 0

    def _skeleton_args_from_payload(self, payload: dict) -> dict:
        if not payload:
            return {k: -1.0 for k in (
                "body_axis_angle", "head_pitch", "hand_to_head",
                "left_hand_to_head", "knee_hip_ratio",
                "arm_elevation", "left_arm_elevation")}
        return {
            "body_axis_angle":    float(payload.get("body_axis_angle",    -1)),
            "head_pitch":         float(payload.get("head_pitch",         -1)),
            "hand_to_head":       float(payload.get("hand_to_head",       -1)),
            "left_hand_to_head":  float(payload.get("left_hand_to_head",  -1)),
            "knee_hip_ratio":     float(payload.get("knee_hip_ratio",     -1)),
            "arm_elevation":      float(payload.get("arm_elevation",      -1)),
            "left_arm_elevation": float(payload.get("left_arm_elevation", -1)),
        }

    def _apply_ablation(self, payload: dict, ablation_mode: str) -> dict:
        p = dict(payload)
        if ablation_mode == "no_skeleton":
            for k in ("body_axis_angle", "head_pitch", "hand_to_head",
                      "left_hand_to_head", "knee_hip_ratio",
                      "arm_elevation", "left_arm_elevation"):
                p[k] = -1.0
        elif ablation_mode == "no_object":
            p["_ablation_held_event"]  = "none"
            p["_ablation_som_objects"] = []
        elif ablation_mode == "no_vlm":
            p["_ablation_vlm_desc"]    = ""
            p["_ablation_vlm_key_obj"] = "none"
        return p

    def _spatial_reasoning(self, activity_hint: str, vlm_scene_desc: str,
                            vlm_key_object: str, held_event: str,
                            user_pos: dict, user_forward: dict,
                            room_name: str, user_id: str,
                            vlm_confidence: float = 0.0,
                            payload: dict = None,
                            vlm_weight_override: float = None,
                            coord_label: str = "",
                            held_age: float = 0.0,
                            som_objects: list = None) -> tuple:

        if not self.scene_engine.is_ready():
            return activity_hint or "Unknown", "zone_not_ready", ""

        nearest_zone = self.scene_engine.find_nearest_zone(user_pos, room_name)
        zone_label   = nearest_zone["zone_name"] if nearest_zone else ""

        candidates = set(b for b in BEHAVIOR_LABELS if b not in NO_WEIGHT_ACTIONS)
        if room_name:
            for room_key, forbidden in ROOM_IMPOSSIBLE.items():
                if room_key.lower() in room_name.lower():
                    for b in forbidden:
                        candidates.discard(b)
        if not candidates:
            return "Standing", "no_candidates", zone_label

        try:
            from modules.perception.scene_graph import build_scene_text
            skel_args   = self._skeleton_args_from_payload(payload)
            _tv_on      = bool(payload.get("tv_on", False)) if payload else None
            _scene_text = build_scene_text(
                user_pos=user_pos, user_forward=user_forward,
                room_name=room_name, held_event=held_event,
                held_age=held_age, db=self.db, user_id=user_id,
                tv_on=_tv_on,
                virtual_hour=float(payload.get("virtual_hour", -1)) if payload else -1,
                **skel_args,
            )
            nearby_objs = self._get_nearby_objects(user_pos, room_name)
            pmi = {
                "near":        coord_label or "",
                "nearby":      nearby_objs,
                "vlm_desc":    vlm_scene_desc,
                "key_object":  vlm_key_object,
                "som_objects": som_objects or [],
            }
            llm_action, llm_reason, llm_conf = self._llm_reason(
                _scene_text, pmi, candidates, self._llm_url, self._llm_model)
            if llm_action:
                return llm_action, llm_reason, zone_label
        except Exception as e:
            print(f"[LLM Reason] skipped: {e}")

        try:
            zone_ranked    = sorted(
                candidates,
                key=lambda a: self.scene_engine.get_zone_affinity(zone_label, a)
                if zone_label else 0,
                reverse=True)
            fallback_action = zone_ranked[0]
        except Exception:
            fallback_action = sorted(candidates)[0]
        print(f"[ZoneFallback] {fallback_action}")
        return fallback_action, "zone_affinity_fallback", zone_label

    def _llm_reason(self, scene_text: str, pmi: dict,
                    candidates: set, llm_url: str, llm_model: str) -> tuple:
        if not candidates:
            return None, "", 0.0

        near        = pmi.get("near",       "")
        nearby_objs = pmi.get("nearby",     [])
        vlm_desc    = pmi.get("vlm_desc",   "")
        key_object  = pmi.get("key_object", "none")
        som_objects = pmi.get("som_objects", [])

        _room_str    = ""
        _time_str    = ""
        _posture_str = ""
        _event_line  = ""
        _facing_str  = ""

        for _line in scene_text.split("\n"):
            _l = _line.strip()
            if _l.startswith("Room:"):         _room_str    = _l.replace("Room: ", "").strip()
            if _l.startswith("Time:"):         _time_str    = _l
            if _l.startswith("Posture"):       _posture_str = _l.replace("Posture cues:", "").strip()
            if _l.startswith("Object event:"): _event_line  = _l
            if _l.startswith("Facing:"):       _facing_str  = _l

        tv_on = any(
            _l.strip().startswith("TV:") and "ON" in _l
            for _l in scene_text.split("\n"))

        nl_parts = [f"A person is in the {_room_str or 'home'}."]
        if _time_str:                           nl_parts.append(_time_str + ".")
        if _posture_str:                        nl_parts.append(f"Posture: {_posture_str}.")
        if _event_line:                         nl_parts.append(f"{_event_line}.")
        if _facing_str:                         nl_parts.append(f"{_facing_str}.")
        if vlm_desc:                            nl_parts.append(f"VLM observation: {vlm_desc}.")
        if key_object and key_object != "none": nl_parts.append(f"Key object: {key_object}.")
        if som_objects:
            labels_str = ", ".join(
                f"{o['label']} ({o.get('status','')})" for o in som_objects)
            nl_parts.append(f"Scene objects: {labels_str}.")
        nl_parts.append(f"Nearest area: {near or 'unknown'}.")
        nl_parts.append("TV: ON" if tv_on else "TV: off")

        prompt = (
            f"{scene_text}\n\n"
            f"Summary: {' '.join(nl_parts)}\n\n"
            f"What is this person most likely doing?\n"
            f"Choose exactly one from: {', '.join(sorted(candidates))}\n\n"
            "Reply with JSON only, no markdown:\n"
            '{"action": "<one from the list>", "confidence": 0.0, "reason": "<max 40 chars>"}'
        )
        print(f"[LLM PROMPT] {prompt[:400]}...")

        try:
            resp = requests.post(
                f"{llm_url}/api/chat",
                json={
                    "model":    llm_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream":   False,
                    "options":  {"temperature": 0.0, "num_predict": 256},
                },
                timeout=self._llm_timeout,
            )
            raw   = resp.json().get("message", {}).get("content", "").strip()
            print(f"[LLM RAW] {raw[:200]}")
            clean = re.sub(r'```(?:json)?\s*', '', raw).strip().rstrip('`')
            m     = re.search(r'\{.*\}', clean, re.DOTALL)
            if not m:
                return None, "", 0.0
            data   = json.loads(m.group(0))
            action = data.get("action", "").strip()
            reason = data.get("reason", "llm").strip()[:60]
            conf   = float(data.get("confidence", 0.7))

            if action in candidates:
                print(f"[LLM] {action} ({conf:.2f}) | {reason}")
                return action, f"pmi_llm:{reason}", conf
            for c_action in candidates:
                if c_action.lower() == action.lower():
                    return c_action, f"pmi_llm:{reason}", conf
            if candidates:
                return sorted(candidates)[0], f"pmi_llm_fallback:{reason}", 0.3
        except requests.exceptions.Timeout:
            print(f"[LLM] timeout after {self._llm_timeout}s")
        except Exception as e:
            print(f"[LLM] error: {e}")

        if candidates:
            return sorted(candidates)[0], "pmi_llm_error_fallback", 0.1
        return None, "", 0.0

    def _emit_edge(self, user_id: str, action: str, bound_label: str,
                   confidence: float, user_pos: dict):
        if confidence < 0.55:
            return
        relation   = ACTION_TO_RELATION.get(action, "near")
        now        = int(datetime.datetime.utcnow().timestamp())
        furn_doc   = self.col_scene.find_one({"label": bound_label}, {"pos": 1, "room": 1})
        try:
            self.db.scene_graph.update_one(
                {"agent_id": user_id},
                {"$set": {
                    "agent_id": user_id,
                    "agent_node": {
                        "id": user_id, "type": "agent",
                        "pos": [user_pos.get("x", 0), user_pos.get("z", 0)],
                        "current_action": action, "timestamp": now,
                    },
                    "furniture_node": {
                        "id":   bound_label,
                        "pos":  furn_doc.get("pos",  [0, 0]) if furn_doc else [0, 0],
                        "room": furn_doc.get("room", "")     if furn_doc else "",
                    },
                    "edges": [{"from": user_id, "relation": relation,
                               "to": bound_label, "confidence": round(confidence, 3),
                               "timestamp": now}],
                    "updated_at": datetime.datetime.utcnow(),
                }},
                upsert=True)
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
            if not faces:
                return hint
            face = sorted(faces,
                key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]),
                reverse=True)[0]
            best, max_sim = hint, 0.0
            for name, known in self.face_bank.items():
                sim = float(np.dot(face.normed_embedding, known))
                if sim > max_sim:
                    max_sim, best = sim, name
            return best if max_sim > 0.40 else hint
        except Exception:
            return hint

    def _nearest_by_coord(self, user_pos, room_name, max_dist=3.0):
        if not user_pos:
            return None, float('inf')
        ux, uz = user_pos.get("x", 0), user_pos.get("z", 0)
        query  = {"room": {"$regex": room_name, "$options": "i"}} if room_name else {}
        docs   = list(self.col_scene.find(
            query, {"label": 1, "pos": 1, "room": 1, "x": 1, "z": 1}))
        if not docs:
            docs = list(self.col_scene.find(
                {}, {"label": 1, "pos": 1, "room": 1, "x": 1, "z": 1}))
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

    def _sem_match_furniture(self, label, k=3):
        norm  = normalize_label(label)
        exact = self.col_scene.find_one({"label": norm})
        if exact:
            return [(exact, 1.0)]
        return self.room_cache.bind_topk(norm, k=k, threshold=SEMANTIC_THRESHOLD)

    def _verify_with_coord(self, topk, sensor_pos):
        if not sensor_pos or not topk:
            return (topk[0][0] if topk else None), "sbert_only"
        ox, oz = sensor_pos[0], sensor_pos[1]
        best_doc, best_dist, best_score = None, float('inf'), 0.0
        for doc, score in topk:
            pos = doc.get("pos")
            if isinstance(pos, list) and len(pos) >= 2:
                dist = math.sqrt((ox - pos[0])**2 + (oz - pos[1])**2)
                if dist < best_dist:
                    best_dist, best_doc, best_score = dist, doc, score
        if best_doc is None:
            return topk[0][0], "sbert_only"
        if best_dist <= COORD_MATCH_DIST:
            return best_doc, "high" if best_score >= 0.7 else "medium"
        if best_dist <= COORD_VERIFY_DIST:
            return best_doc, "coord_ok"
        return None, "coord_fallback"

    def _bind_furniture(self, vlm_label, user_pos, room_name):
        topk                  = self._sem_match_furniture(vlm_label, k=3)
        coord_doc, coord_dist = self._nearest_by_coord(user_pos, room_name)
        if not topk and not coord_doc:
            return None, "unknown"
        if topk and user_pos:
            doc, conf = self._verify_with_coord(
                topk, [user_pos.get("x", 0), user_pos.get("z", 0)])
            if doc:
                return doc, conf
        if coord_doc and coord_dist < 1.5:
            return coord_doc, "coord_priority"
        if coord_doc:
            return coord_doc, "coord_only"
        if topk:
            return topk[0][0], "sbert_low"
        return None, "unknown"

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
            "bound_instance": "Unknown_Area", "bound_room": "", "confidence": "unknown",
        }

    def _write_experiment_log(self, payload: dict, final_user: str,
                               spatial_action: str, activity_hint: str,
                               upgrade_reason: str, zone_label: str,
                               vlm_confidence: float, held_event: str,
                               user_pos: dict, interacting_items: list,
                               ground_truth_raw: str,
                               ablation_mode: str, experiment_mode: str,
                               collection_suffix: str,
                               vlm_timed_out: bool = False):
        ground_truth = normalize_ground_truth(ground_truth_raw)
        predicted    = normalize_ground_truth(spatial_action)
        correct      = (ground_truth == predicted) if ground_truth else None
        col_name     = f"experiment_logs{collection_suffix}" if collection_suffix else "experiment_logs"
        try:
            self.db[col_name].insert_one({
                "episode_id":         payload.get("episode_id", ""),
                "t_capture":          payload.get("t_capture", ""),
                "user":               final_user,
                "user_id":            final_user,
                "ground_truth":       ground_truth,
                "ground_truth_raw":   ground_truth_raw,
                "vlm_output":         activity_hint,
                "vlm_scene_desc":     activity_hint,
                "vlm_key_object":     payload.get("_vlm_key_object", "none"),
                "spatial_action":     spatial_action,
                "predicted":          predicted,
                "correct":            correct,
                "upgrade_reason":     upgrade_reason,
                "zone_label":         zone_label,
                "vlm_confidence":     round(vlm_confidence, 3),
                "held_event":         held_event,
                "vlm_timed_out":      vlm_timed_out,
                "room_name":          payload.get("room_name", ""),
                "user_pos":           user_pos,
                "interacting_items":  interacting_items,
                "ablation_mode":      ablation_mode,
                "experiment_mode":    experiment_mode,
                "system_mode":        payload.get("system_mode", "semantic"),
                "collection_suffix":  collection_suffix,
                "body_axis_angle":    float(payload.get("body_axis_angle",    -1)),
                "head_pitch":         float(payload.get("head_pitch",         -1)),
                "hand_to_head":       float(payload.get("hand_to_head",       -1)),
                "left_hand_to_head":  float(payload.get("left_hand_to_head",  -1)),
                "knee_hip_ratio":     float(payload.get("knee_hip_ratio",     -1)),
                "arm_elevation":      float(payload.get("arm_elevation",      -1)),
                "left_arm_elevation": float(payload.get("left_arm_elevation", -1)),
                "virtual_day":        payload.get("virtual_day"),
                "virtual_hour":       payload.get("virtual_hour"),
                "time_slot":          _get_time_slot(payload.get("virtual_hour")),
                "virtual_date":       _virtual_day_to_date(payload.get("virtual_day")),
                "timestamp":          datetime.datetime.utcnow(),
            })
        except Exception as e:
            print(f"[ExpLog] write error: {e}")

    def analyze_action_burst(self, payload: dict) -> dict:
        image_list   = payload.get("image_list", [])
        hint_user_id = payload.get("userID", "Unknown_User")
        source_nodes = payload.get("source_nodes", [])
        node_scores  = payload.get("node_scores", [])
        user_pos     = payload.get("user_pos", None)
        user_forward = payload.get("user_forward", None)

        experiment_mode   = payload.get("experiment_mode",   "baseline")
        ablation_mode     = payload.get("ablation_mode",     "full")
        collection_suffix = payload.get("collection_suffix", "")

        ablated_payload = self._apply_ablation(payload, ablation_mode)

        if not user_forward or (
            float(user_forward.get("x", 0)) == 0 and
            float(user_forward.get("z", 0)) == 0
        ):
            _udoc = self.db.user_positions.find_one({"$or": [
                {"user_id": hint_user_id},
                {"user_id": hint_user_id.lower()},
            ]}, {"forward": 1, "x": 1, "z": 1})
            if _udoc and _udoc.get("forward"):
                fwd = _udoc["forward"]
                if isinstance(fwd, list) and len(fwd) >= 3:
                    user_forward = {"x": fwd[0], "y": fwd[1], "z": fwd[2]}
                elif isinstance(fwd, dict):
                    user_forward = fwd
            if not user_pos and _udoc:
                user_pos = {"x": float(_udoc.get("x", 0)), "z": float(_udoc.get("z", 0))}

        room_name             = payload.get("room_name", "")
        virtual_hour          = payload.get("virtual_hour", None)
        virtual_day           = payload.get("virtual_day",  None)
        ground_truth_activity = payload.get("activity", None)
        episode_id             = payload.get("episode_id", "unknown")

        if not image_list:
            return self._empty_result(hint_user_id)

        self.room_cache.switch_room(room_name, self.col_scene)
        if not self.room_cache.all_docs:
            self.room_cache._room = None
            self.room_cache.switch_room("", self.col_scene)

        coord_doc, coord_dist = self._nearest_by_coord(user_pos, room_name)
        coord_label           = coord_doc.get("label", "") if coord_doc else ""
        room_furniture        = [d.get("label", "") for d in self.room_cache.all_docs if d.get("label")]

        t_capture                  = payload.get("t_capture", None)
        held_event_raw, held_age   = self._get_held_object_from_scene(
            hint_user_id, user_pos, t_capture)
        held_event = ablated_payload.get("_ablation_held_event", held_event_raw)

        _objects_2d = payload.get("objects_2d", [])
        _system_mode = payload.get("system_mode", Config.SYSTEM_MODE)
        if _system_mode == "vlm_som":
            if _objects_2d:
                som_objects = self._som_objects_from_2d(_objects_2d)
            else:
                som_objects = self._get_som_objects(
                    user_pos=user_pos, room_name=room_name,
                    user_id=hint_user_id, held_event=held_event)
        else:
            som_objects = []
        if ablated_payload.get("_ablation_som_objects") is not None:
            som_objects = ablated_payload["_ablation_som_objects"]

        try:
            from modules.perception.som_marker import build_som_text, mark_objects_on_image
            som_text = build_som_text(som_objects)
        except Exception:
            som_text = ""

        prompt         = self._build_prompt(room_name, room_furniture, coord_label, coord_dist, som_text)
        sample_indices = self._select_sample_indices(image_list, node_scores)

        user_votes         = []
        visual_state_votes = []
        key_object_votes   = []
        confidence_list    = []
        used_scores_list   = []
        vlm_timed_out      = False

        system_mode = payload.get("system_mode", Config.SYSTEM_MODE)

        for idx in sample_indices:
            try:
                img_b64   = image_list[idx]
                uid       = self._get_user_id(img_b64, hint_user_id)
                user_votes.append(uid)

                if ablation_mode == "no_vlm" or system_mode == "semantic":
                    visual_state_votes.append("")
                    key_object_votes.append("none")
                    confidence_list.append(0.0)
                    used_scores_list.append(node_scores[idx] if idx < len(node_scores) else 0.5)
                    continue

                try:
                    from modules.perception.som_marker import mark_objects_on_image as _mark
                    _objects_2d = payload.get("objects_2d", [])
                    marked_b64  = _mark(img_b64, som_objects,
                                        objects_2d=_objects_2d if _objects_2d else None)
                except Exception:
                    marked_b64 = img_b64

                self._save_som_debug_image(
                    marked_b64,
                    episode_id=episode_id,
                    ground_truth=ground_truth_activity,
                    user_id=hint_user_id,
                    frame_idx=idx)

                img_clean = marked_b64.split(',')[1] if ',' in marked_b64 else marked_b64
                parsed    = self._call_vlm_with_retry(prompt, img_clean)

                if parsed:
                    used_scores_list.append(node_scores[idx] if idx < len(node_scores) else 0.5)
                    visual_state_votes.append(parsed.get("visual_state", ""))
                    key_object_votes.append(parsed.get("key_object", "none"))
                    confidence_list.append(parsed.get("confidence", 0.5))
                else:
                    vlm_timed_out = True
                    print(f"[VLM] frame {idx} failed all retries")

            except Exception as e:
                print(f"[Frame {idx}] error: {e}")
                vlm_timed_out = True

        def _weighted_vote(votes: list, scores: list, default: str = "") -> str:
            if not votes:
                return default
            weights = defaultdict(float)
            for v, s in zip(votes, scores):
                weights[v] += max(float(s), 0.1)
            return max(weights, key=weights.get)

        final_user     = max(set(user_votes), key=user_votes.count) if user_votes else hint_user_id
        vlm_scene_desc = _weighted_vote(visual_state_votes, used_scores_list, default="")
        vlm_key_object = _weighted_vote(key_object_votes,   used_scores_list, default="none")
        vlm_confidence = (sum(confidence_list) / len(confidence_list)
                          if confidence_list else 0.0)

        if ablation_mode == "no_vlm":
            vlm_scene_desc = ablated_payload.get("_ablation_vlm_desc",    "")
            vlm_key_object = ablated_payload.get("_ablation_vlm_key_obj", "none")
            vlm_confidence = 0.0

        activity_hint = vlm_scene_desc

        from collections import Counter as _Counter
        vlm_scene_desc = (
            _Counter(visual_state_votes).most_common(1)[0][0]
            if visual_state_votes else ""
        )
        vlm_key_object = (
            _Counter(key_object_votes).most_common(1)[0][0]
            if key_object_votes else "none"
        )

        print(f"[VLM] desc='{vlm_scene_desc[:30]}' key={vlm_key_object} "
              f"held={held_event} conf={vlm_confidence:.2f} "
              f"frames={len(visual_state_votes)}")

        spatial_action, upgrade_reason, zone_label = self._spatial_reasoning(
            activity_hint=activity_hint, vlm_scene_desc=vlm_scene_desc,
            vlm_key_object=vlm_key_object, held_event=held_event,
            user_pos=user_pos, user_forward=user_forward,
            room_name=room_name, user_id=final_user,
            vlm_confidence=vlm_confidence, payload=ablated_payload,
            vlm_weight_override=0.0,
            coord_label=coord_label, held_age=held_age,
            som_objects=som_objects,
        )

        bound_doc, confidence = self._bind_furniture(coord_label, user_pos, room_name)
        bound_label = bound_doc.get("label", "Unknown_Area") if bound_doc else "Unknown_Area"
        bound_room  = bound_doc.get("room",  room_name)      if bound_doc else room_name
        nearby_objs = self._get_nearby_objects(user_pos, room_name)

        result = {
            "location":         bound_label,
            "room":             bound_room,
            "interacting_items":[],
            "scene_items":      nearby_objs,
            "all_items":        nearby_objs,
            "spatial_relations":[],
            "context":          f"{final_user} {spatial_action} near {bound_label}",
            "_vlm_scene_desc":  vlm_scene_desc,
            "_vlm_key_object":  vlm_key_object,
            "_held_event":      held_event,
            "_activity_hint":   activity_hint,
            "_coord_label":     coord_label,
            "_coord_dist":      round(coord_dist, 2) if coord_dist != float('inf') else None,
            "_confidence":      confidence,
            "_vlm_confidence":  round(vlm_confidence, 3),
            "_som_objects":     som_objects,
            "_vlm_timed_out":   vlm_timed_out,
            "_entropy":         0.0,
        }

        self._update_scene_snapshot(bound_doc, [], nearby_objs, [])
        self._update_dynamic_objects(
            user_id=final_user, interacting_items=[],
            scene_items=nearby_objs, bound_doc=bound_doc,
            room_name=bound_room, user_pos=user_pos)

        if ground_truth_activity:
            payload["_vlm_key_object"] = vlm_key_object
            self._write_experiment_log(
                payload=payload, final_user=final_user,
                spatial_action=spatial_action, activity_hint=activity_hint,
                upgrade_reason=upgrade_reason, zone_label=zone_label,
                vlm_confidence=vlm_confidence, held_event=held_event,
                user_pos=user_pos, interacting_items=[],
                ground_truth_raw=ground_truth_activity,
                ablation_mode=ablation_mode, experiment_mode=experiment_mode,
                collection_suffix=collection_suffix, vlm_timed_out=vlm_timed_out,
            )

        if (spatial_action not in ("Unknown", "none", "") and
                bound_label != "Unknown_Area" and user_pos):
            self._emit_edge(
                user_id=final_user, action=spatial_action,
                bound_label=bound_label,
                confidence=float(result["_vlm_confidence"]),
                user_pos=user_pos)

        print(f"[Done] {final_user} | spatial={spatial_action} | "
              f"reason={upgrade_reason} | zone={zone_label} | "
              f"ablation={ablation_mode} | vlm_ok={not vlm_timed_out}")

        return {
            "user":              final_user,
            "action":            activity_hint,
            "spatial_action":    spatial_action,
            "upgrade_reason":    upgrade_reason,
            "zone_label":        zone_label,
            "result":            result,
            "items":             [],
            "all_items":         nearby_objs,
            "spatial_relations": [],
            "bound_instance":    bound_label,
            "bound_room":        bound_room,
            "confidence":        confidence,
            "sbert_sim":         0.0,
            "user_pos":          user_pos or {},
            "virtual_hour":      virtual_hour or 12.0,
            "room":              room_name or "",
            "experiment_mode":   experiment_mode,
            "ablation_mode":     ablation_mode,
            "time_slot":         _get_time_slot(virtual_hour),
            "virtual_day":       virtual_day,
        }

    def _update_scene_snapshot(self, bound_doc, interacting_items,
                                scene_items, spatial_relations):
        if not bound_doc:
            return
        all_items = list(set(interacting_items + scene_items))
        self.col_scene.update_one(
            {"_id": bound_doc.get("_id")},
            {"$addToSet": {"items": {"$each": all_items}},
             "$set": {
                 "current_contents":  interacting_items,
                 "spatial_relations": spatial_relations,
                 "last_observation":  datetime.datetime.utcnow(),
             }})

    def _update_dynamic_objects(self, user_id, interacting_items, scene_items,
                                 bound_doc, room_name, user_pos=None):
        now         = datetime.datetime.utcnow()
        bound_label = bound_doc.get("label", "Unknown_Area") if bound_doc else "Unknown_Area"
        bound_pos   = bound_doc.get("pos")                   if bound_doc else None

        def _upsert(label, is_interacting):
            if not label or label in STRUCTURAL_BLACKLIST:
                return
            if self.col_scene.find_one({"label": label}):
                return
            base_set  = {
                "last_seen_on": bound_label, "spatial_rel": "near",
                "room":         room_name,   "last_seen":   now,
                "source":       "vlm",       "status":      "active",
                "status_since": now,
            }
            if bound_pos:
                base_set["furniture_pos"] = bound_pos
            inc_ops   = {"seen_count": 1}
            if is_interacting:
                inc_ops["interact_count"] = 1
            update_op = {
                "$inc":         inc_ops,
                "$set":         base_set,
                "$setOnInsert": {"first_seen": now},
            }
            if is_interacting:
                update_op["$addToSet"] = {"interacted_by": user_id}
            self._bulk_buf.upsert(label, update_op, now)

        for item in interacting_items:
            _upsert(item, True)
        for item in scene_items:
            _upsert(item, False)

    def _update_activity_sequence(self, user, action, instance):
        today = datetime.datetime.utcnow().strftime("%Y-%m-%d")
        self.col_activity.update_one(
            {"user": user, "date": today},
            {"$push": {"sequence": {
                "action":    action,
                "instance":  instance,
                "timestamp": datetime.datetime.utcnow().isoformat(),
            }},
             "$setOnInsert": {"user": user, "date": today}},
            upsert=True)

    def shutdown(self):
        try:
            self._bulk_buf.force_flush()
        except Exception:
            pass
