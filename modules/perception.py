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

from collections import defaultdict
from pymongo import MongoClient, ReturnDocument, UpdateOne
from sentence_transformers import SentenceTransformer


STRUCTURAL_BLACKLIST = {
    "wall", "floor", "ceiling", "wooden floor", "white wall",
    "white ceiling", "window", "door", "ground", "white box",
    "concrete floor", "tile floor", "carpet", "baseboard",
    "room", "background", "surface", "area",
}

YOUR_OBJECTS = {
    "juice", "cola",
    "bottle", "cup", "wine glass",
    "bowl", "fork", "knife", "spoon",
    "banana", "apple", "orange", "sandwich",
    "pizza", "donut", "cake", "hot dog",
    "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone",
    "microwave", "oven", "toaster",
    "sink", "refrigerator",
    "chair", "couch", "bed", "dining table", "toilet",
    "book", "clock", "vase", "scissors",
    "teddy bear", "toothbrush",
    "backpack", "handbag",
    "potted plant",
    "pan", "frying pan", "pot",
    "broom", "mop", "dustpan",
    "food", "plate", "dish",
    "person",
}

LABEL_NORMALIZE_MAP = {
    "juicebottle":        "juice",
    "juice bottle":       "juice",
    "orange juice":       "juice",
    "juice box":          "juice",
    "fruit juice":        "juice",
    "cola can":           "cola",
    "coca cola":          "cola",
    "coke":               "cola",
    "soda can":           "cola",
    "cola bottle":        "cola",
    "phone":              "cell phone",
    "mobile phone":       "cell phone",
    "smartphone":         "cell phone",
    "iphone":             "cell phone",
    "android":            "cell phone",
    "television":         "tv",
    "screen":             "tv",
    "notebook":           "laptop",
    "computer":           "laptop",
    "tablet":             "laptop",
    "ipad":               "laptop",
    "macbook":            "laptop",
    "remote control":     "remote",
    "tv remote":          "remote",
    "mug":                "cup",
    "coffee cup":         "cup",
    "tea cup":            "cup",
    "drinking glass":     "cup",
    "water glass":        "cup",
    "water bottle":       "bottle",
    "milk":               "bottle",
    "beer":               "bottle",
    "soda bottle":        "bottle",
    "sofa":               "couch",
    "settee":             "couch",
    "table":              "dining table",
    "kitchen table":      "dining table",
    "coffee table":       "dining table",
    "desk":               "dining table",
    "mother's bed":       "bed",
    "father's bed":       "bed",
    "parents bed":        "bed",
    "frying pan":         "pan",
    "cooking pan":        "pan",
    "skillet":            "pan",
    "sweep":              "broom",
    "sweeping brush":     "broom",
    "mop stick":          "mop",
}

BULK_WRITE_THRESHOLD = 20
BULK_WRITE_INTERVAL  = 30.0
SEMANTIC_THRESHOLD   = 0.35
COORD_VERIFY_DIST    = 2.0
COORD_MATCH_DIST     = 1.5

# Actions that are captured by camera (Choice B: all actions)
# but should NOT accumulate weight in observation_logs.
# These are transitional/utility actions — not habitual behaviors.
NO_WEIGHT_ACTIONS = {
    "PickingUp", "PuttingDown",
    "Walking", "Standing",
    # Opening removed: it is an Intent Pivot action whose weight
    # accumulation is needed for Manifold Engine prev_action context.
}

BEHAVIOR_LABELS = [
    "Drinking", "SittingDrink", "Eating", "Cooking", "Opening",
    "Laying", "Watching", "Reading", "Cleaning", "PhoneUse",
    "Typing", "PickingUp", "PuttingDown", "Standing", "Walking",
]

VISION_PROTOTYPES = {
    "Drinking": (
        "standing upright vertical posture one hand raised wrist elevated close to face. "
        "gripping beverage container bottle glass cup can metal_can fluid_holder bringing to lips. "
        "drinking sipping liquid swallowing fluid refreshment hydration cola juice water while standing"
    ),
    "SittingDrink": (
        "seated stationary resting posture single hand or both hands holding cup vessel mug glass can box. "
        "lifting container toward lips swallowing hot cold beverage tea coffee juice soda cola refreshment. "
        "sipping drinking liquid intake without chewing jaw stationary relaxed sitting position"
    ),
    "Eating": (
        "seated posture head tilted slightly forward hands holding dining tools chopsticks fork spoon utensils bowl plate. "
        "solid food substance nourishment meal chunk raised toward open mouth. "
        "jaw actively moving rhythmic chewing mastication grinding swallowing ingesting foodstuff"
    ),
    "Cooking": (
        "standing stationary fixed position facing forward heat source stove flame. "
        "both hands gripping cookware pan pot wok spatula ladle utensil. "
        "active wrist rotation forearm stirring circular mixing motion. "
        "head slightly bowed gaze directed downward into vessel container. "
        "thermal culinary processing food preparation repetitive arm cycle"
    ),
    "Opening": (
        "standing upright position facing large vertical surface volumetric structure refrigerator fridge unit. "
        "one arm fully extended forward hand gripping long door handle knob profile. "
        "pulling door assembly outward executing swinging motion opening large interior storage appliance cabinet. "
        "cool cold air release from interior"
    ),
    "Laying": (
        "body horizontal reclined flat spine fully extended parallel to ground floor. "
        "head supported resting total body weight distributed passive stillness repose. "
        "eyes closed or ceiling directed gaze non focused facial muscles relaxed sleeping napping"
    ),
    "Watching": (
        "seated upright vertical spine orientation head level. "
        "eyes directed straight forward sustained visual focus fixed forward gaze alignment. "
        "hands empty resting relaxed on lap passive reception. "
        "stationary still body minimal facial skeletal movement ongoing attention"
    ),
    "Reading": (
        "seated or resting posture both hands holding open reading material book magazine paper document text pages. "
        "head bowed downward chin tucked eyes focused scanning words reading turning sheets"
    ),
    "Cleaning": (
        "standing active locomotion lateral weight shift stepping displacement. "
        "both hands firmly gripping elongated shaft pole handle broom mop tool. "
        "wide sweeping arc arm motion bilateral rhythmic push pull stroke. "
        "tool tip contact floor ground surface low trajectory. "
        "repetitive cyclical whole body coordination house chores maintenance"
    ),
    "PhoneUse": (
        "unilateral movement single hand holding thin small portable electronic device smartphone phone digital screen. "
        "face tilted downward eyes focused on glowing display thumb finger scrolling tapping touching swiping single-handed"
    ),
    "Typing": (
        "seated upright forward workstation posture. "
        "both hands dual hands flat palms down fingers actively pressing tapping discrete keys input interface. "
        "wrists low forearms level synchronized bilateral hand activity typing working coding"
    ),
    "PickingUp": (
        "bending forward at waist spine flexing hand reaching downward close to ground floor surface. "
        "grasping small object item entity lifting upward elevation from ground counter"
    ),
    "PuttingDown": (
        "bending forward or lowering arm hand lowering object item asset placing settling. "
        "releasing grip item contact surface table floor setting down terminating carry"
    ),
    "Standing": (
        "standing upright vertical body alignment both arms hanging loose at sides no object hands empty. "
        "not moving idle stationary waiting neutral posture baseline status"
    ),
    "Walking": (
        "locomotion legs stepping alternating feet forward displacement moving room path hallway. "
        "arms swinging naturally body coordination walking gait translation motion velocity"
    ),
}


NORMALIZE_THRESHOLD = 0.42

# Imported by manifold recording inside analyze_action_burst
def build_x_for_record(virtual_hour, user_pos, prev_action):
    import math as _math
    h = float(virtual_hour) if virtual_hour is not None else 12.0
    rad = 2 * _math.pi * h / 24.0
    sin_t, cos_t = _math.sin(rad), _math.cos(rad)
    x = float(user_pos.get("x", 0)) / 10.0 if user_pos else 0.0
    z = float(user_pos.get("z", 0)) / 10.0 if user_pos else 0.0
    prev_vec = [0.0] * len(BEHAVIOR_LABELS)
    if prev_action in BEHAVIOR_LABELS:
        prev_vec[BEHAVIOR_LABELS.index(prev_action)] = 1.0
    return [sin_t, cos_t, x, z] + prev_vec


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


# Pre-encoded YOUR_OBJECTS vectors (populated by PerceptionEngine.__init__)
_YOUR_OBJECTS_LIST: list = []
_YOUR_OBJECTS_VECS = None


def normalize_label(raw: str, sbert_model=None) -> str:
    """
    L2a Symbolic Normalization:
    1. Fast path: exact string match (LABEL_NORMALIZE_MAP / YOUR_OBJECTS)
    2. Slow path: SBERT semantic clustering (fallback for free-text VLM output)
    """
    global _YOUR_OBJECTS_LIST, _YOUR_OBJECTS_VECS
    if not raw:
        return raw
    raw_lower = raw.lower().strip()

    # Fast path 1: direct map
    if raw_lower in LABEL_NORMALIZE_MAP:
        return LABEL_NORMALIZE_MAP[raw_lower]
    if raw_lower in YOUR_OBJECTS:
        return raw_lower

    # Fast path 2: word-level match
    for word in raw_lower.split():
        if word in YOUR_OBJECTS:
            return word
        if word in LABEL_NORMALIZE_MAP:
            return LABEL_NORMALIZE_MAP[word]

    # Slow path: SBERT semantic clustering
    if sbert_model is not None and _YOUR_OBJECTS_LIST:
        try:
            if _YOUR_OBJECTS_VECS is None:
                return raw_lower
            raw_vec = sbert_model.encode(
                raw, normalize_embeddings=True).astype("float32")
            sims    = _YOUR_OBJECTS_VECS @ raw_vec
            best_i  = int(np.argmax(sims))
            best_s  = float(sims[best_i])
            if best_s >= 0.75:
                matched = _YOUR_OBJECTS_LIST[best_i]
                print(f"[L2a] SBERT normalize: '{raw}' -> '{matched}' "
                      f"(sim={best_s:.2f})")
                return matched
        except Exception as e:
            print(f"[L2a] SBERT normalize failed: {e}")

    return raw_lower


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
        print(f"[RoomCache] Room '{room_name}' -> {len(self._labels)} furniture cached")

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
        print(f"    [ChangeSync] Loaded {len(self._map)} scene objects")

    def start(self):
        self._running = True
        threading.Thread(target=self._watch_loop, daemon=True).start()

    def stop(self):
        self._running = False

    def _watch_loop(self):
        try:
            with self.scene_col.watch(full_document="updateLookup") as stream:
                print("    [ChangeSync] Change Stream mode")
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
            print("    [ChangeSync] Polling mode")
            while self._running:
                try:
                    docs = list(self.scene_col.find({}))
                    with self._lock:
                        self._map = {d.get("label", ""): d for d in docs}
                except Exception as e:
                    print(f"    [ChangeSync] Poll error: {e}")
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
            r = self.col.bulk_write(ops, ordered=False)
            print(f"    [BulkWrite] Flushed {len(ops)} ops "
                  f"(upserted={r.upserted_count}, modified={r.modified_count})")
        except Exception as e:
            print(f"    [BulkWrite] Failed: {e}")

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

    def __init__(self, ollama_url, model_name,
                 face_analyzer=None, face_bank=None,
                 mongo_uri="mongodb://127.0.0.1:27017/",
                 db_name="robot_rag_db",
                 sbert_model_name="all-MiniLM-L6-v2"):

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

        self.sbert       = SentenceTransformer(sbert_model_name)
        self.room_cache  = RoomEmbeddingCache(self.sbert)
        self.scene_sync  = ChangeStreamSync(self.col_scene, self.room_cache)
        self.scene_sync.start()
        self.bulk_buffer = BulkWriteBuffer(self.col_dynamics)
        self.faiss_store = FAISSMemoryStore(self.sbert)

        self._proto_vecs   = None
        self._proto_labels = BEHAVIOR_LABELS
        self._your_objects_list = sorted(YOUR_OBJECTS)
        self._your_objects_vecs = self.sbert.encode(
            self._your_objects_list,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")
        # populate module-level cache for normalize_label()
        global _YOUR_OBJECTS_LIST, _YOUR_OBJECTS_VECS
        _YOUR_OBJECTS_LIST = self._your_objects_list
        _YOUR_OBJECTS_VECS = self._your_objects_vecs
        self.zone_graph    = []
        self._affinity_matrix = {}
        self.col_affinity      = self.db["affinity_matrix"]
        self.col_user_aff      = self.db["user_spatial_affinity"]
        self.col_habit_snap    = self.db["habit_snapshots"]
        self.col_aff_history   = self.db["affinity_history"]
        self.col_manifold_data = self.db["manifold_training_data"]

        self._load_affinity_matrix()
        self._discover_zones()

        if not self.zone_graph:
            import threading
            threading.Thread(
                target=self._discover_zones_with_retry,
                daemon=True
            ).start()

        print(f"[PerceptionEngine] Ready | model={model_name}")


    ITEM_TO_ACTION = {
        "bowl": "Eating", "fork": "Eating", "spoon": "Eating",
        "plate": "Eating", "food": "Eating",
        "banana": "Eating", "apple": "Eating",
        "cell phone": "PhoneUse", "phone": "PhoneUse",
        "book": "Reading", "magazine": "Reading",
        "laptop": "Typing", "keyboard": "Typing",
        "cup": "Drinking", "bottle": "Drinking",
        "mug": "Drinking", "juice": "Drinking",
        "cola": "Drinking",
        "broom": "Cleaning", "mop": "Cleaning",
        "pan": "Cooking", "spatula": "Cooking",
        "remote": "Watching",
    }




    def _load_affinity_matrix(self):
        docs = list(self.col_affinity.find({}))
        if docs:
            for doc in docs:
                furn    = doc["furniture"]
                action  = doc["action"]
                affinity= doc["affinity"]
                if furn not in self._affinity_matrix:
                    self._affinity_matrix[furn] = {}
                self._affinity_matrix[furn][action] = affinity
            print(f"[Affinity] Loaded {len(docs)} entries from MongoDB")
        else:
            print("[Affinity] No affinity_matrix found, will distill after zones ready")

    def _distill_affinity_matrix(self):
        furnitures = [
            d["label"] for d in self.col_scene.find(
                {}, {"label": 1}) if d.get("label")
        ]
        if not furnitures:
            print("[Affinity] No furniture in scene_snapshots, skip distillation")
            return

        behaviors = [
            l for l in self._proto_labels
            if l not in ("Standing", "Walking", "PickingUp", "PuttingDown")
        ]

        furniture_list = ", ".join(furnitures)
        behavior_list  = ", ".join(behaviors)

        prompt = (
            "You are a spatial behavior expert for home robots. "
            "For each furniture item, rate its affinity (0.00 to 1.00) "
            "with each behavior. "
            "High affinity means the behavior commonly happens near that furniture. "
            "Key rules: "
            "tv: very high for Watching only (0.90+), near 0 for others. "
            "monitor: very high for Typing only (0.90+), near 0 for others. "
            "stove: very high for Cooking only (0.90+), near 0 for others. "
            "refrigerator: very high for Opening only (0.85+), near 0 for others. "
            "sofa: distribute across Watching(0.35), Laying(0.35), "
            "Reading(0.15), SittingDrink(0.1), PhoneUse(0.05). "
            "bed: Laying(0.6), Reading(0.3), PhoneUse(0.1). "
            "desk: Typing(0.8), Reading(0.15), PhoneUse(0.05). "
            "dining table or table: Eating(0.4), SittingDrink(0.4), "
            "Cooking(0.1), Drinking(0.1). "
            "sink: Drinking(0.5), Cooking(0.3), Cleaning(0.2). "
            "chair: Eating(0.35), SittingDrink(0.35), Typing(0.2), Reading(0.1). "
            f"Behaviors: {behavior_list}. "
            f"Furniture items: {furniture_list}. "
            "Output ONLY valid JSON, no markdown, no explanation. "
            "Format: furniture as key, behavior scores as value dict."
        )

        print(f"[Affinity] Distilling via gemma3:4b "
              f"({len(furnitures)} furniture x {len(behaviors)} behaviors)...")
        try:
            resp = requests.post(
                f"{self.url}/api/chat",
                json={
                    "model":    "gemma3:4b",
                    "messages": [{"role": "user", "content": prompt}],
                    "stream":   False,
                    "options":  {"temperature": 0.1, "num_predict": 2000},
                },
                timeout=120,
            )
            raw = resp.json().get("message", {}).get("content", "").strip()

            import re as _re
            m = _re.search(r'\{.*\}', raw, _re.DOTALL)
            if not m:
                print(f"[Affinity] JSON not found in response")
                return

            matrix = json.loads(m.group(0))
            bulk   = []
            for furn, action_scores in matrix.items():
                for action, score in action_scores.items():
                    if action in self._proto_labels:
                        bulk.append({
                            "furniture": furn.lower().strip(),
                            "action":    action,
                            "affinity":  float(score),
                            "source":    "gemma3:4b",
                        })
                        fkey = furn.lower().strip()
                        if fkey not in self._affinity_matrix:
                            self._affinity_matrix[fkey] = {}
                        self._affinity_matrix[fkey][action] = float(score)

            if bulk:
                self.col_affinity.delete_many({})
                self.col_affinity.insert_many(bulk)
                print(f"[Affinity] Distilled {len(bulk)} entries, stored in MongoDB")

        except Exception as e:
            import traceback
            print(f"[Affinity] Distillation failed: {e}")
            print(traceback.format_exc())

    def _get_furniture_affinity(self, furniture_label: str, action: str) -> float:
        label = furniture_label.lower().strip()
        return self._affinity_matrix.get(label, {}).get(action, 0.0)

    def _compute_furniture_weight(self, label: str) -> float:
        lbl = label.lower().strip()
        if self._affinity_matrix and lbl in self._affinity_matrix:
            scores = list(self._affinity_matrix[lbl].values())
            if scores:
                sorted_s = sorted(scores, reverse=True)
                top1     = sorted_s[0]
                top2     = sorted_s[1] if len(sorted_s) > 1 else 0.0
                uniqueness = top1 - top2
                weight     = 1.0 + uniqueness * 10.0
                return max(1.0, round(weight, 2))
        try:
            furn_vec    = self.sbert.encode(
                label, normalize_embeddings=True
            ).astype("float32")
            proto_vecs  = self._get_proto_vecs()
            sims        = proto_vecs @ furn_vec
            sorted_sims = np.sort(sims)[::-1]
            top1        = float(sorted_sims[0])
            top2        = float(sorted_sims[1]) if len(sorted_sims) > 1 else 0.0
            uniqueness  = top1 - top2
            weight      = 1.0 + uniqueness * 10.0
            return max(1.0, round(weight, 2))
        except Exception:
            return 1.0

    def _discover_zones(self):
        print("[Zones] Discovering functional zones from scene_snapshots...")
        try:
            all_docs = list(self.col_scene.find(
                {}, {"label": 1, "pos": 1, "room": 1}
            ))
            if not all_docs:
                print("[Zones] No furniture found in scene_snapshots")
                self.zone_graph = []
                return

            proto_vecs    = self._get_proto_vecs()
            proto_labels  = self._proto_labels

            by_room = {}
            for doc in all_docs:
                room = (doc.get("room") or "Unknown").strip()
                by_room.setdefault(room, []).append(doc)

            zones = []
            for room_name, furniture_list in by_room.items():
                furn_with_label = []
                for f in furniture_list:
                    label = f.get("label", "")
                    pos   = f.get("pos")
                    if not label or not isinstance(pos, list) or len(pos) < 2:
                        continue
                    vec     = self.sbert.encode(
                        [label], normalize_embeddings=True)[0].astype("float32")
                    sims    = proto_vecs @ vec
                    best_i  = int(np.argmax(sims))
                    best_s  = float(sims[best_i])
                    act_lbl = proto_labels[best_i] if best_s > 0.25 else "Unknown"
                    furn_with_label.append({
                        "label":        label,
                        "pos":          pos,
                        "action_label": act_lbl,
                        "vec":          vec,
                        "sim":          best_s,
                    })

                if not furn_with_label:
                    continue

                grouped = {}
                for f in furn_with_label:
                    placed = False
                    for key, group in grouped.items():
                        lbl, idx = key
                        if lbl != f["action_label"]:
                            continue
                        rep_pos = group[0]["pos"]
                        fx, fz  = f["pos"][0], f["pos"][1]
                        rx, rz  = rep_pos[0], rep_pos[1]
                        dist    = math.sqrt((fx - rx)**2 + (fz - rz)**2)
                        if dist <= 2.5:
                            group.append(f)
                            placed = True
                            break
                    if not placed:
                        new_key = (f["action_label"], len(grouped))
                        grouped[new_key] = [f]

                for (act_lbl, zone_idx), group in grouped.items():
                    labels    = [g["label"] for g in group]
                    positions = [g["pos"] for g in group]
                    cx = float(np.mean([p[0] for p in positions]))
                    cz = float(np.mean([p[1] for p in positions]))

                    weights = np.array([
                        self._compute_furniture_weight(g["label"])
                        for g in group
                    ])
                    vecs    = np.stack([g["vec"] for g in group])
                    v_space = (weights[:, None] * vecs).sum(axis=0)
                    norm    = np.linalg.norm(v_space)
                    if norm > 1e-8:
                        v_space /= norm

                    zone_name = f"{act_lbl}_Zone"
                    if zone_idx > 0:
                        zone_name = f"{act_lbl}_Zone_{zone_idx + 1}"

                    zones.append({
                        "room":       room_name,
                        "zone_name":  zone_name,
                        "action_label": act_lbl,
                        "center":     [cx, cz],
                        "v_space":    v_space.tolist(),
                        "furniture":  labels,
                    })
                    print(f"  [Zone] {room_name} | {zone_name} | "
                          f"furniture={labels} center=({cx:.1f},{cz:.1f})")

            self.zone_graph = zones
            print(f"[Zones] Built {len(zones)} zones across "
                  f"{len(by_room)} rooms")

        except Exception as e:
            import traceback
            print(f"[Zones] Error: {e}\n{traceback.format_exc()}")
            self.zone_graph = []


    def _discover_zones_with_retry(self):
        import time as _time
        print("[Zones] Starting background retry loop...")
        for attempt in range(20):
            _time.sleep(15)
            count = self.col_scene.count_documents({})
            if count > 0:
                print(f"[Zones] Retry {attempt+1}: "
                      f"scene_snapshots has {count} docs, rebuilding...")
                if not self._affinity_matrix:
                    print("[Zones] Affinity matrix empty, distilling now...")
                    self._distill_affinity_matrix()
                self._discover_zones()
                if self.zone_graph:
                    print(f"[Zones] Zone Graph ready: "
                          f"{len(self.zone_graph)} zones")
                    return
        print("[Zones] Retry exhausted, Zone Graph still empty")

    def _compute_zone_affinity(self, behavior, zone):
        if not zone:
            return 0.0
        furnitures = zone.get("furniture", [])
        if self._affinity_matrix and furnitures:
            scores = [
                self._affinity_matrix.get(f.lower().strip(), {}).get(behavior, 0.0)
                for f in furnitures
            ]
            weights = [
                self._compute_furniture_weight(f) for f in furnitures
            ]
            total_w = sum(weights)
            if total_w > 0:
                weighted_aff = sum(s * w for s, w in zip(scores, weights)) / total_w
                return round(weighted_aff, 3)
        if not zone or "v_space" not in zone:
            return 0.0
        if behavior not in self._proto_labels:
            return 0.0
        try:
            idx       = self._proto_labels.index(behavior)
            proto_vec = self._get_proto_vecs()[idx]
            zone_vec  = np.array(zone["v_space"], dtype="float32")
            affinity  = float(proto_vec @ zone_vec)
            return max(0.0, min(1.0, affinity))
        except Exception:
            return 0.0

    def _find_nearest_zone(self, user_pos, room_name=""):
        if not self.zone_graph or not user_pos:
            return None
        ux = float(user_pos.get("x", 0))
        uz = float(user_pos.get("z", 0))

        candidates = [
            z for z in self.zone_graph
            if not room_name or
            room_name.lower() in z["room"].lower() or
            z["room"].lower() in room_name.lower()
        ]
        if not candidates:
            candidates = self.zone_graph

        best_zone = None
        best_dist = float("inf")
        for zone in candidates:
            cx, cz = zone["center"][0], zone["center"][1]
            dist   = math.sqrt((ux - cx)**2 + (uz - cz)**2)
            if dist < best_dist:
                best_dist = dist
                best_zone = zone
        return best_zone


    def _is_ambiguous_zone(self, zone) -> bool:
        if not zone or not self._affinity_matrix:
            return False
        scores = []
        for behavior in BEHAVIOR_LABELS:
            s = self._compute_zone_affinity(behavior, zone)
            scores.append(s)
        if not scores or max(scores) < 0.05:
            return True
        sorted_s = sorted(scores, reverse=True)
        top1 = sorted_s[0]
        top2 = sorted_s[1] if len(sorted_s) > 1 else 0.0
        return (top1 - top2) < 0.25

    def _spatial_reasoning(self, vlm_action, sbert_sim,
                            user_pos, user_forward,
                            interacting_items, room_name,
                            user_id=""):
        upgraded_action = vlm_action
        upgrade_reason  = ""
        zone_label      = ""

        nearest_zone = self._find_nearest_zone(user_pos, room_name)
        if nearest_zone:
            zone_label = nearest_zone["zone_name"]

        items = interacting_items or []
        if isinstance(items, str):
            items = [items] if items else []

        # L2A：持握物判斷（最高優先級）
        for item in items:
            item_lower = item.lower().strip()
            for obj_key, action in self.ITEM_TO_ACTION.items():
                if obj_key in item_lower:
                    if upgraded_action != action:
                        upgraded_action = action
                        upgrade_reason  = f"L2A_held:{item}->{action}"
                    return upgraded_action, upgrade_reason, zone_label

        should_upgrade  = (vlm_action == "Unknown") or (sbert_sim < 0.50)
        high_confidence = sbert_sim >= 0.80

        if high_confidence:
            return upgraded_action, "", zone_label

        if not should_upgrade or not nearest_zone:
            return upgraded_action, upgrade_reason, zone_label

        # 判斷是否為多義模糊區
        is_ambiguous = self._is_ambiguous_zone(nearest_zone)

        proto_vecs = self._get_proto_vecs()
        zone_v     = np.array(nearest_zone["v_space"], dtype="float32")

        if is_ambiguous:
            # 多義模糊區：攔截 L3，只靠 L2B 方位
            if user_forward and user_pos:
                best_score  = 0.65
                best_action = vlm_action
                for i, label in enumerate(self._proto_labels):
                    if label in ("Standing", "Walking",
                                 "PickingUp", "PuttingDown"):
                        continue
                    sim = float(proto_vecs[i] @ zone_v)
                    if sim > best_score:
                        best_score  = sim
                        best_action = label

                if best_action != vlm_action:
                    ux  = float(user_pos.get("x", 0))
                    uz  = float(user_pos.get("z", 0))
                    cx  = nearest_zone["center"][0]
                    cz  = nearest_zone["center"][1]
                    dx, dz = cx - ux, cz - uz
                    dist   = math.sqrt(dx*dx + dz*dz)
                    if dist > 0.01:
                        dx /= dist
                        dz /= dist
                        fwd_x   = float(user_forward.get("x", 0))
                        fwd_z   = float(user_forward.get("z", 0))
                        fwd_len = math.sqrt(fwd_x*fwd_x + fwd_z*fwd_z)
                        if fwd_len > 0.01:
                            fwd_x   /= fwd_len
                            fwd_z   /= fwd_len
                            heading  = max(0.0, fwd_x*dx + fwd_z*dz)
                            combined = best_score * 0.55 + heading * 0.45
                            if combined > 0.65:
                                upgraded_action = best_action
                                upgrade_reason  = (
                                    f"L2B_ambiguous_heading:"
                                    f"{nearest_zone['zone_name']}"
                                    f"->{best_action}"
                                    f" vsim={best_score:.2f}"
                                    f" heading={heading:.2f}"
                                )
            return upgraded_action, upgrade_reason, zone_label

        # 明確功能區：L3 強制補全流程
        best_score  = 0.55
        best_action = vlm_action
        for i, label in enumerate(self._proto_labels):
            if label in ("Standing", "Walking",
                         "PickingUp", "PuttingDown"):
                continue
            sim = float(proto_vecs[i] @ zone_v)
            if sim > best_score:
                best_score  = sim
                best_action = label

        # L2B：方位對齊（明確區）
        if user_forward and user_pos and best_action != vlm_action:
            ux  = float(user_pos.get("x", 0))
            uz  = float(user_pos.get("z", 0))
            cx  = nearest_zone["center"][0]
            cz  = nearest_zone["center"][1]
            dx, dz = cx - ux, cz - uz
            dist   = math.sqrt(dx*dx + dz*dz)
            if dist > 0.01:
                dx /= dist
                dz /= dist
                fwd_x   = float(user_forward.get("x", 0))
                fwd_z   = float(user_forward.get("z", 0))
                fwd_len = math.sqrt(fwd_x*fwd_x + fwd_z*fwd_z)
                if fwd_len > 0.01:
                    fwd_x   /= fwd_len
                    fwd_z   /= fwd_len
                    heading  = max(0.0, fwd_x*dx + fwd_z*dz)
                    combined = best_score * 0.6 + heading * 0.4
                    if combined > 0.55:
                        upgraded_action = best_action
                        upgrade_reason  = (
                            f"L2B_heading+zone:"
                            f"{nearest_zone['zone_name']}"
                            f"->{best_action}"
                            f" vsim={best_score:.2f}"
                            f" heading={heading:.2f}"
                        )
                        return upgraded_action, upgrade_reason, zone_label

        # L3：Zone Affinity 補全
        if best_action != vlm_action and best_score > 0.55:
            zone_affinity = self._compute_zone_affinity(
                best_action, nearest_zone)

            personal_aff = 0.0
            if user_id and hasattr(self, "col_user_aff"):
                doc = self.col_user_aff.find_one({
                    "user_id": user_id,
                    "action":  best_action,
                    "zone":    nearest_zone.get("zone_name", ""),
                })
                if doc:
                    personal_aff = doc.get("affinity", 0.0)

            effective_aff = max(zone_affinity, personal_aff)
            if effective_aff >= 0.40:
                upgraded_action = best_action
                aff_src        = ("personal" if personal_aff > zone_affinity
                                  else "static")
                upgrade_reason  = (
                    f"L3_zone:{nearest_zone['zone_name']}"
                    f"->{best_action}"
                    f" vsim={best_score:.2f}"
                    f" aff={effective_aff:.2f}({aff_src})"
                )

        return upgraded_action, upgrade_reason, zone_label

    def _get_proto_vecs(self):
        if self._proto_vecs is None:
            self._proto_vecs = self.sbert.encode(
                list(VISION_PROTOTYPES.values()),
                normalize_embeddings=True).astype("float32")
            print(f"[SBERT] prototype vectors built ({len(self._proto_labels)} classes)")
        return self._proto_vecs

    def _build_sbert_input(self, parsed: dict) -> str:
        parts = [
            parsed.get("body_posture", ""),
            parsed.get("gaze_target",  ""),
            parsed.get("hand_state",   ""),
            parsed.get("summary",      ""),
            parsed.get("summary",      ""),
        ]
        return " ".join(p for p in parts if p).strip()

    def _normalize_action(self, sbert_input: str) -> str:
        if not sbert_input or sbert_input.strip() in ("", "none", "unknown"):
            return "Unknown"
        try:
            vecs   = self._get_proto_vecs()
            vec    = self.sbert.encode(
                [sbert_input], normalize_embeddings=True)[0].astype("float32")
            sims   = vecs @ vec
            best_i = int(np.argmax(sims))
            best_s = float(sims[best_i])
            best_l = self._proto_labels[best_i]

            if best_s >= NORMALIZE_THRESHOLD:
                print(f"[Normalize] '{sbert_input[:60]}' -> '{best_l}' "
                      f"(sim={best_s:.2f})")
                return best_l

            # Below threshold: return Unknown instead of raw string
            # This keeps eval_logs clean for analysis
            print(f"[Normalize] low sim (best={best_l} sim={best_s:.2f} "
                  f"< {NORMALIZE_THRESHOLD}) -> Unknown")
            return "Unknown"

        except Exception as e:
            print(f"[Normalize] failed: {e}")
            return "Unknown"

    def _normalize_action_with_score(self, sbert_input: str):
        if not sbert_input or sbert_input.strip() in ("", "none", "unknown"):
            return "Unknown", 0.0
        try:
            vecs   = self._get_proto_vecs()
            vec    = self.sbert.encode(
                [sbert_input], normalize_embeddings=True)[0].astype("float32")
            sims   = vecs @ vec
            best_i = int(np.argmax(sims))
            best_s = float(sims[best_i])
            best_l = self._proto_labels[best_i]
            if best_s >= NORMALIZE_THRESHOLD:
                return best_l, best_s
            return "Unknown", best_s
        except Exception as e:
            print(f"[Normalize] failed: {e}")
            return "Unknown", 0.0

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
        except Exception as e:
            print(f"Face ReID: {e}")
            return hint

    def _nearest_by_coord(self, user_pos, room_name, max_dist=3.0):
        if not user_pos: return None, float('inf')
        ux, uz = user_pos.get("x", 0), user_pos.get("z", 0)
        docs   = (self.scene_sync.find_by_room(room_name)
                  if room_name else self.scene_sync.all_docs())
        if not docs: docs = self.scene_sync.all_docs()
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
        exact = self.scene_sync.get(norm)
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

    def _build_prompt(self, room_name, room_furniture, coord_label, coord_dist):
        furn_hint = (f"\nKNOWN FURNITURE: {', '.join(room_furniture)}.\n"
                     if room_furniture else "")
        coord_hint = (f"\nSPATIAL HINT: person is ~{coord_dist:.1f}m from '{coord_label}'.\n"
                      if coord_label and coord_dist < 3.0 else "")
        return f"""You are a visual observation system for a home robot.
Analyze the person in the image. Do NOT name the activity.
Room: "{room_name}".{furn_hint}{coord_hint}
Output ONLY valid JSON:

{{
  "body_posture": "describe spine and leg position. Use terms like: upright, bent forward, horizontal, walking, or similar.",
  "gaze_target": "where are the eyes looking. Use terms like: downward at hands, at screen, forward, closed eyes, or similar.",
  "hand_state": "what are the hands doing. Use terms like: holding small object, empty, on surface, raising to face, holding device, reaching downward, or similar.",
  "summary": "one short sentence of what you actually see",
  "person_near": "name of the closest furniture or surface",
  "objects_on_furniture": [
    {{"object": "object name", "on": "furniture name", "relation": "on/in/next_to"}}
  ]
}}

RULES:
- body_posture: is the person flat, upright, bent, or moving?
- gaze_target: follow head and eye direction carefully.
- hand_state: look carefully at hands — what do they hold or touch?
- objects_on_furniture: use standard object names (cup, bottle, juice, cola, book, laptop, remote, cell phone, pan, broom, food, bowl, etc).
- NEVER include wall, floor, ceiling, window, door in objects_on_furniture.
- Do NOT guess the activity name.
"""

    def _parse_vlm_output(self, raw: str) -> dict:
        try:
            data = json.loads(self._extract_json(raw))
        except Exception:
            return {}

        body_posture = data.get("body_posture", "").strip()
        gaze_target  = data.get("gaze_target",  "").strip()
        hand_state   = data.get("hand_state",   "").strip()
        summary      = data.get("summary",      "").strip()
        person_near  = data.get("person_near",  "unknown").lower().strip()
        objs         = data.get("objects_on_furniture", [])

        spatial_relations = []
        scene_items       = []
        interacting_items = []

        for entry in objs:
            if not isinstance(entry, dict): continue
            obj = entry.get("object", "").lower().strip()
            on  = entry.get("on",     "").lower().strip()
            rel = entry.get("relation", "on").lower().strip()
            if not obj or not on: continue
            norm = normalize_label(obj, self.sbert)
            if norm in STRUCTURAL_BLACKLIST: continue
            scene_items.append(norm)
            spatial_relations.append({"subject": norm, "relation": rel, "object": on})

        hand_lower = hand_state.lower()
        if any(kw in hand_lower for kw in
               ["holding", "raising", "carrying", "gripping", "reaching"]):
            words = re.findall(r'\b\w+(?:\s+\w+)?\b', hand_lower)
            for w in words:
                w = w.strip()
                if len(w) < 3: continue
                if w in {"the", "and", "with", "its", "his", "her",
                         "holding", "raising", "carrying", "small", "large",
                         "object", "something", "device", "reaching",
                         "downward", "forward"}: continue
                norm = normalize_label(w, self.sbert)
                if norm not in STRUCTURAL_BLACKLIST and norm not in YOUR_OBJECTS:
                    continue
                interacting_items.append(norm)
                spatial_relations.append({
                    "subject": norm, "relation": "in_hand_of", "object": "person"})
                break

        return {
            "body_posture":      body_posture,
            "gaze_target":       gaze_target,
            "hand_state":        hand_state,
            "summary":           summary,
            "main_object":       person_near,
            "interacting_items": interacting_items,
            "scene_items":       scene_items,
            "spatial_relations": spatial_relations,
        }

    def _validate_scene_graph(self, parsed, user_pos, room_name, user_id):
        vlm_furn              = parsed.get("main_object", "unknown")
        bound_doc, confidence = self._bind_furniture(vlm_furn, user_pos, room_name)
        bound_label = bound_doc.get("label", "Unknown_Area") if bound_doc else "Unknown_Area"
        bound_room  = (bound_doc.get("room") or bound_doc.get("room_name", room_name)
                       if bound_doc else room_name)
        print(f"    [Validate] person_near: '{vlm_furn}' -> '{bound_label}' (conf={confidence})")

        validated_spatial = []
        for rel in parsed.get("spatial_relations", []):
            subj = rel.get("subject", "")
            obj  = rel.get("object",  "")
            r    = rel.get("relation", "on")
            if obj == "person":
                validated_spatial.append(
                    {"subject": subj, "relation": r, "object": user_id})
                continue
            topk = self._sem_match_furniture(obj, k=3)
            furn_label = topk[0][0].get("label", obj) if topk else normalize_label(obj)
            validated_spatial.append(
                {"subject": subj, "relation": r, "object": furn_label})

        return {
            "bound_doc":         bound_doc,
            "bound_label":       bound_label,
            "bound_room":        bound_room,
            "confidence":        confidence,
            "body_posture":      parsed.get("body_posture", ""),
            "gaze_target":       parsed.get("gaze_target",  ""),
            "hand_state":        parsed.get("hand_state",   ""),
            "summary":           parsed.get("summary",      ""),
            "interacting_items": parsed.get("interacting_items", []),
            "scene_items":       parsed.get("scene_items", []),
            "all_items":         list(set(
                parsed.get("interacting_items", []) +
                parsed.get("scene_items", []))),
            "spatial_relations": validated_spatial,
        }

    def _aggregate_frames(self, parsed_list, used_scores):
        parts = []
        for parsed, score in zip(parsed_list, used_scores):
            frame_str = self._build_sbert_input(parsed)
            if frame_str:
                repeat = max(1, round(max(float(score), 0.1) * 3))
                parts.extend([frame_str] * repeat)
        combined = " ".join(parts)
        print(f"[AtomicAgg] {len(parts)} weighted frames -> '{combined[:100]}'")
        return combined

    def _vote_main_object(self, parsed_list, used_scores):
        weights = defaultdict(float)
        for parsed, score in zip(parsed_list, used_scores):
            weights[parsed.get("main_object", "unknown")] += max(float(score), 0.1)
        return max(weights, key=weights.get) if weights else "unknown"

    def analyze_action_burst(self, payload: dict) -> dict:
        image_list   = payload.get("image_list", [])
        hint_user_id = payload.get("userID", "Unknown_User")
        source_nodes = payload.get("source_nodes", [])
        node_scores  = payload.get("node_scores", [])
        user_pos     = payload.get("user_pos", None)
        user_forward = payload.get("user_forward", None)
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
        room_furniture = [d.get("label", "") for d in self.room_cache.all_docs
                          if d.get("label")]
        prompt         = self._build_prompt(
            room_name, room_furniture, coord_label, coord_dist)
        sample_indices = self._select_sample_indices(image_list, node_scores)

        user_votes, parsed_list, used_scores_list = [], [], []

        for idx in sample_indices:
            try:
                img_b64    = image_list[idx]
                uid        = self._get_user_id(img_b64, hint_user_id)
                user_votes.append(uid)
                img_clean  = img_b64.split(',')[1] if ',' in img_b64 else img_b64
                node_score = node_scores[idx] if idx < len(node_scores) else 0.5
                resp = requests.post(
                    f"{self.url}/api/chat",
                    json={
                        "model":    self.model,
                        "messages": [{"role": "user", "content": prompt,
                                      "images": [img_clean]}],
                        "stream":   False,
                        "options":  {"temperature": 0.05, "num_predict": 600},
                    },
                    timeout=120)
                raw       = resp.json().get("message", {}).get("content", "").strip()
                node_name = source_nodes[idx] if idx < len(source_nodes) else f"node_{idx}"
                print(f" [Frame {idx}|{node_name}|score={node_score:.2f}] {raw[:200]}")
                parsed = self._parse_vlm_output(raw)
                if parsed and any([parsed.get("body_posture"), parsed.get("summary")]):
                    parsed_list.append(parsed)
                    used_scores_list.append(node_score)
            except Exception as e:
                print(f"[Frame {idx}] error: {e}")

        if not parsed_list:
            return self._empty_result(
                max(set(user_votes), key=user_votes.count) if user_votes else hint_user_id)

        final_user              = max(set(user_votes), key=user_votes.count) if user_votes else hint_user_id
        combined_str            = self._aggregate_frames(parsed_list, used_scores_list)
        final_action, sbert_sim = self._normalize_action_with_score(combined_str)
        final_object            = self._vote_main_object(parsed_list, used_scores_list)

        best_parsed = dict(parsed_list[0])
        best_parsed["main_object"] = final_object

        validated = self._validate_scene_graph(
            best_parsed, user_pos, room_name, final_user)

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
            "context":           validated.get("summary", ""),
            "_body_posture":     validated.get("body_posture", ""),
            "_gaze_target":      validated.get("gaze_target",  ""),
            "_hand_state":       validated.get("hand_state",   ""),
            "_sbert_input":      combined_str,
            "_coord_label":      coord_label,
            "_coord_dist":       round(coord_dist, 2) if coord_dist != float('inf') else None,
            "_confidence":       confidence,
        }

        self._update_scene_snapshot(
            bound_doc, validated["interacting_items"],
            validated["scene_items"], validated["spatial_relations"])

        # Pass ground-truth activity label from Unity if available.
        # Used only for eval_logs — perception still runs blind.
        ground_truth_activity = payload.get("activity", None)

        self._update_observation_log(
            final_user, final_action, bound_doc,
            validated["interacting_items"], validated["spatial_relations"],
            validated.get("summary", ""), virtual_hour, virtual_day,
            ground_truth_activity=ground_truth_activity,
            sbert_sim=sbert_sim,
            combined_str=combined_str,
            user_pos=user_pos,
            user_forward=user_forward,
            room_name=room_name)

        self._update_dynamic_objects(
            user_id=final_user,
            interacting_items=validated["interacting_items"],
            scene_items=validated["scene_items"],
            spatial_relations=validated["spatial_relations"],
            bound_doc=bound_doc, room_name=bound_room, user_pos=user_pos)

        self._write_semantic_memory(
            final_user, final_action, bound_doc, confidence, result, source_nodes)

        self._update_activity_sequence(final_user, final_action, bound_label)

        mem_doc  = self.col_memory.find_one(
            {"user": final_user, "action": final_action},
            sort=[("timestamp", -1)])
        mongo_id = str(mem_doc["_id"]) if mem_doc else ""
        self._index_to_faiss(final_user, final_action, bound_doc, result, mongo_id)

        day_key = _virtual_day_to_date(virtual_day)
        print(f"\n [Done] {final_user} -> {final_action} @ {bound_label} "
              f"(conf={confidence}, date={day_key}, "
              f"slot={_get_time_slot(virtual_hour)}, "
              f"pending={self.bulk_buffer.pending_count})\n")

        spatial_action, upgrade_reason, zone_label = self._spatial_reasoning(
            vlm_action        = final_action,
            sbert_sim         = sbert_sim,
            user_pos          = user_pos,
            user_forward      = user_forward,
            interacting_items = validated["interacting_items"],
            room_name         = room_name,
            user_id           = final_user,
        )

        if upgrade_reason:
            print(f"[Spatial] {final_action} -> {spatial_action} | {upgrade_reason}")

        # Record manifold training sample (L4 HabitLearner input)
        # Use spatial_action (post-L3 补全) as the ground-truth label
        # Only record habitual actions — skip transitional ones
        _record_action = spatial_action if spatial_action != "Unknown" else final_action
        if (_record_action not in ("Unknown", "Standing", "Walking",
                                    "PickingUp", "PuttingDown", "none", "")):
            try:
                prev_seq = self.col_activity.find_one(
                    {"user": final_user},
                    sort=[("timestamp", -1)])
                _prev = ""
                if prev_seq and prev_seq.get("sequence"):
                    last_acts = prev_seq["sequence"]
                    if len(last_acts) >= 2:
                        _prev = last_acts[-2].get("action", "")
                self.col_manifold_data.insert_one({
                    "user_id":      final_user,
                    "X":            build_x_for_record(
                                        virtual_hour, user_pos, _prev),
                    "y":            BEHAVIOR_LABELS.index(_record_action)
                                    if _record_action in BEHAVIOR_LABELS else -1,
                    "action":       _record_action,
                    "prev_action":  _prev,
                    "virtual_hour": float(virtual_hour) if virtual_hour else 12.0,
                    "timestamp":    datetime.datetime.utcnow(),
                })
            except Exception as _me:
                pass   # non-critical

        if ground_truth_activity:
            self.db["eval_logs"].insert_one({
                "user_id":           final_user,
                "ground_truth":      ground_truth_activity,
                "vlm_output":        final_action,
                "spatial_action":    spatial_action,
                "upgrade_reason":    upgrade_reason,
                "zone_label":        zone_label,
                "sbert_sim":         sbert_sim,
                "room_name":         room_name,
                "user_pos":          user_pos,
                "user_forward":      user_forward,
                "interacting_items": validated["interacting_items"],
                "timestamp":         datetime.datetime.utcnow(),
            })

        return {
            "user":           final_user,
            "action":         final_action,
            "spatial_action": spatial_action,
            "upgrade_reason": upgrade_reason,
            "zone_label":     zone_label,
            "result":         result,
            "items":          validated["interacting_items"],
            "all_items":      validated["all_items"],
            "spatial":        validated["spatial_relations"],
            "bound_instance": bound_label,
            "bound_room":     bound_room,
            "confidence":     confidence,
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
        counts = {f"spatial_counts.{r['subject']}|{r['relation']}|{r['object']}": 1
                  for r in spatial_relations
                  if r.get("subject") and r.get("object")}
        if counts: update_op["$inc"] = counts
        self.col_scene.update_one({"_id": bound_doc.get("_id")}, update_op)


    def _update_user_affinity(self, user: str, action: str,
                               zone_name: str, instance: str):
        if not action or not user:
            return
        try:
            pipeline = [
                {"$match": {"user": user, "action": action}},
                {"$group": {
                    "_id":          "$zone_name",
                    "total_weight": {"$sum": "$weight"},
                }},
            ]
            results = list(self.col_obs.aggregate(pipeline))
            total   = sum(r["total_weight"] for r in results)
            if total == 0:
                return

            for r in results:
                zone_key = r["_id"] or "Unknown_Zone"
                personal = r["total_weight"] / total
                self.col_user_aff.update_one(
                    {"user_id": user,
                     "action":  action,
                     "zone":    zone_key},
                    {"$set": {
                        "affinity":    round(personal, 4),
                        "updated_at":  datetime.datetime.utcnow(),
                    }},
                    upsert=True,
                )
                # record daily history for convergence curve
                today = datetime.datetime.utcnow().strftime("%Y-%m-%d")
                self.col_aff_history.update_one(
                    {"user_id": user, "action": action,
                     "zone": zone_key, "date": today},
                    {"$set": {
                        "affinity":  round(personal, 4),
                        "timestamp": datetime.datetime.utcnow(),
                    }},
                    upsert=True,
                )
        except Exception as e:
            print(f"[UserAffinity] {e}")

    def _write_habit_snapshot(self, user: str, action: str,
                               instance: str, zone_name: str,
                               today: str):
        try:
            self.col_habit_snap.update_one(
                {
                    "user":     user,
                    "action":   action,
                    "instance": instance,
                    "zone":     zone_name or "Unknown",
                    "date":     today,
                },
                {"$inc": {"daily_count": 1}},
                upsert=True,
            )
        except Exception as e:
            print(f"[HabitSnap] {e}")

    def _update_observation_log(self, user, action, bound_doc,
                                 interacting_items, spatial_relations,
                                 raw_desc, virtual_hour=None, virtual_day=None,
                                 ground_truth_activity=None,
                                 sbert_sim=0.0,
                                 combined_str="",
                                 user_pos=None,
                                 user_forward=None,
                                 room_name=""):
        if not bound_doc: return

        instance  = bound_doc.get("label", "Unknown")
        pos_raw   = bound_doc.get("pos")
        pos_xy    = pos_raw if isinstance(pos_raw, list) else \
                    [bound_doc.get("x", 0), bound_doc.get("z", 0)]
        time_slot = _get_time_slot(virtual_hour)
        today     = _virtual_day_to_date(virtual_day)

        # NO_WEIGHT_ACTIONS: record to semantic_memories and dynamic_objects
        # but do NOT accumulate weight in observation_logs.
        # These transitional actions should not trigger FAT.
        if action in NO_WEIGHT_ACTIONS:
            print(f"[ObsLog] {user} -> {action} @ {instance} "
                  f"[{time_slot}] no-weight action, skip weight update")
            return

        existing = self.col_obs.find_one({
            "user": user, "instance": instance,
            "action": action, "time_slot": time_slot, "last_date": today,
        })
        if existing:
            self.col_obs.update_one(
                {"_id": existing["_id"]},
                {"$set": {"last_seen": datetime.datetime.utcnow(),
                          "raw_vlm_desc": raw_desc}})
            print(f"[ObsLog] {user} -> {action} @ {instance} "
                  f"[{time_slot}] date={today} already counted")
            return

        # derive zone_name from nearest zone
        try:
            _nz = self._find_nearest_zone(user_pos, room_name)
            zone_name_for_log = _nz["zone_name"] if _nz else ""
        except Exception:
            zone_name_for_log = ""

        self.col_obs.find_one_and_update(
            {"user": user, "instance": instance,
             "action": action, "time_slot": time_slot},
            {
                "$inc":         {"weight": 1},
                "$addToSet":    {"interacting_items": {"$each": interacting_items}},
                "$set":         {
                    "observed_relations": spatial_relations,
                    "pos":               pos_xy,
                    "room":              bound_doc.get("room", "").strip()
                                         if bound_doc else "",
                    "zone_name":         zone_name_for_log,
                    "last_seen":         datetime.datetime.utcnow(),
                    "last_date":         today,
                    "raw_vlm_desc":      raw_desc,
                },
                "$setOnInsert": {
                    "user": user, "instance": instance,
                    "action": action, "time_slot": time_slot,
                },
            },
            upsert=True, return_document=ReturnDocument.AFTER,
        )
        self._write_habit_snapshot(user, action, instance,
                                   zone_name_for_log, today)
        self._update_user_affinity(user, action,
                                   zone_name_for_log, instance)
        print(f"[ObsLog] {user} -> {action} @ {instance} "
              f"[{time_slot}] date={today} +1 weight")

    def _update_dynamic_objects(self, user_id, interacting_items, scene_items,
                                 spatial_relations, bound_doc, room_name, user_pos=None):
        now         = datetime.datetime.utcnow()
        bound_label = bound_doc.get("label", "Unknown_Area") if bound_doc else "Unknown_Area"
        bound_pos   = bound_doc.get("pos") if bound_doc else None

        item_rel_map, item_furn_map = {}, {}
        for rel in spatial_relations:
            subj = rel.get("subject", "").lower().strip()
            obj  = rel.get("object",  "").lower().strip()
            r    = rel.get("relation", "on").lower().strip()
            if subj:
                item_rel_map[subj]  = r
                item_furn_map[subj] = obj

        def _upsert(label, is_interacting):
            if not label or label in STRUCTURAL_BLACKLIST:
                return
            if self.scene_sync.get(label):
                return
            relation = item_rel_map.get(label, "near")
            is_held  = relation in ("in_hand_of", "held_by", "carrying", "in_hand")
            furn     = item_furn_map.get(label)
            if is_held:
                resolved_label, resolved_pos = user_id, None
            elif furn:
                fd = self.scene_sync.get(furn)
                resolved_label = fd["label"] if fd else bound_label
                resolved_pos   = fd.get("pos") if fd else bound_pos
            else:
                resolved_label, resolved_pos = bound_label, bound_pos

            base_set = {
                "last_seen_on": resolved_label,
                "spatial_rel":  "held_by" if is_held else relation,
                "room":         room_name,
                "last_seen":    now,
                "source":       "vlm",
            }
            if is_held and user_pos:
                base_set["furniture_pos"] = [user_pos.get("x", 0), user_pos.get("z", 0)]
            elif resolved_pos:
                base_set["furniture_pos"] = resolved_pos

            inc_ops   = {"seen_count": 1}
            if is_interacting: inc_ops["interact_count"] = 1
            update_op = {
                "$inc":         inc_ops,
                "$set":         base_set,
                "$setOnInsert": {"first_seen": now},
            }
            if is_interacting:
                update_op["$addToSet"] = {"interacted_by": user_id}
            self.bulk_buffer.upsert(label, update_op, now)

        for item in interacting_items: _upsert(item, True)
        for item in scene_items:       _upsert(item, False)

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
                "$push": {"sequence": {
                    "action":    action,
                    "instance":  instance,
                    "timestamp": datetime.datetime.utcnow().isoformat(),
                }},
                "$setOnInsert": {"user": user, "date": today},
            },
            upsert=True)
        print(f"[Sequence] {user} -> {action}@{instance} ({today})")

    def _index_to_faiss(self, user, action, bound_doc, result, mongo_id):
        instance = bound_doc.get("label", "Unknown") if bound_doc else "Unknown"
        pos_raw  = bound_doc.get("pos") if bound_doc else None
        pos_xy   = pos_raw if isinstance(pos_raw, list) else \
                   ([bound_doc.get("x", 0), bound_doc.get("z", 0)]
                    if bound_doc else [0, 0])
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
        print(f"[FAISS] memory: {instance} | {action}")

    def shutdown(self):
        self.bulk_buffer.force_flush()
        self.scene_sync.stop()
        print("[PerceptionEngine] Shutdown complete")