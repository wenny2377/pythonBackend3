import threading
import datetime
import time
import math
from pymongo import UpdateOne
from config import Config

BASE_FURNITURE_KEYWORDS = {
    "sofa", "couch", "bed", "table", "desk", "chair", "tv", "television",
    "refrigerator", "fridge", "sink", "toilet", "shelf", "wardrobe",
    "cabinet", "nightstand", "bookshelf", "dresser", "stove", "bathtub",
    "dining table", "kitchen_table", "dining_table",
    "monitor",
}

OBJECT_CATEGORIES = {
    "food": {
        "banana", "apple", "sandwich", "orange", "broccoli",
        "carrot", "hot dog", "pizza", "donut", "cake",
        "bowl","saladbowl", "plate",
    },
    "drink": {
        "bottle", "wine glass", "cup", "juice", "cola",
    },
    "device": {
        "tv", "laptop", "mouse", "remote", "keyboard",
        "cell phone", "microwave", "oven", "toaster",
        "refrigerator", "sink",
    },
    "personal": {
        "backpack", "umbrella", "handbag", "tie",
        "suitcase", "book", "scissors", "toothbrush", "hair drier",
    },
    "kitchen": {
        "pan", "frying pan", "pot", "spoon", "fork", "knife",
    },
    "cleaning": {
        "broom", "mop", "dustpan",
    },
}

POLL_INTERVAL       = 5
BATCH_SIZE          = 50
DISTANCE_THRESHOLD  = 0.8
INERTIA_THRESHOLD   = 1.1
HAND_THRESHOLD      = 0.9
HAND_STICKY_LIMIT   = 1.3
ACTIVE_FURNITURE_H  = 24
CLEANUP_DAYS        = 7


class ObjectClassifier:

    def __init__(self, db):
        self.db           = db
        self.col_raw      = db["raw_objects"]
        self.col_scene    = db["scene_snapshots"]
        self.col_dynamic  = db["dynamic_objects"]
        self._running     = False
        self._thread      = None
        self._furniture_cache = {}
        self._user_cache  = []

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print("[Classifier] Background classification thread started")

    def stop(self):
        self._running = False
        print("[Classifier] Stopped")

    def _is_furniture(self, label: str) -> bool:
        label_norm = label.lower().strip()
        return any(kw in label_norm for kw in BASE_FURNITURE_KEYWORDS)

    def _get_category(self, label: str) -> str:
        label_l = label.lower().strip()
        for category, keywords in OBJECT_CATEGORIES.items():
            if label_l in keywords:
                return category
            if any(kw in label_l for kw in keywords):
                return category
        return "other"

    @staticmethod
    def _make_instance_key(label: str, room: str, anchor: str) -> str:
        return f"{label.lower()}|{room.lower()}|{anchor.lower()}"

    def _loop(self):
        while self._running:
            try:
                self._refresh_furniture_cache()
                self._process_batch()
                self._cleanup_old_data()
            except Exception as e:
                print(f"[Classifier Error] {e}")
            time.sleep(POLL_INTERVAL)

    def _refresh_furniture_cache(self):
        cutoff   = datetime.datetime.utcnow() - datetime.timedelta(hours=ACTIVE_FURNITURE_H)
        active_f = list(self.col_scene.find(
            {"last_updated": {"$gt": cutoff}},
            {"label": 1, "pos": 1, "room": 1}
        ))
        self._furniture_cache = {f["label"]: f for f in active_f}
        self._user_cache = list(self.db["user_positions"].find(
            {}, {"user_id": 1, "x": 1, "z": 1, "updated_at": 1}
        ))

    def _find_closest_user(self, x, z, stale_seconds=60):
        now = datetime.datetime.utcnow()
        best_uid, best_dist = None, float("inf")
        for u in self._user_cache:
            updated = u.get("updated_at")
            if updated and (now - updated).total_seconds() > stale_seconds:
                continue
            dist = math.sqrt((x - u.get("x", 0)) ** 2 + (z - u.get("z", 0)) ** 2)
            if dist < best_dist:
                best_dist, best_uid = dist, u.get("user_id", "")
        return best_uid, best_dist

    def _find_closest_furniture(self, x, z, room, current_anchor=None):
        def _room_match(f_room, obj_room):
            if not f_room or not obj_room:
                return False
            f_n = f_room.lower().replace(" ", "").replace("_", "")
            o_n = obj_room.lower().replace(" ", "").replace("_", "")
            return o_n in f_n or f_n in o_n

        if current_anchor and current_anchor in self._furniture_cache:
            f     = self._furniture_cache[current_anchor]
            f_pos = f.get("pos")
            if f_pos and math.sqrt(
                (x - f_pos[0]) ** 2 + (z - f_pos[1]) ** 2
            ) <= INERTIA_THRESHOLD:
                return current_anchor

        best_label, best_dist = None, float("inf")
        for label, f in self._furniture_cache.items():
            f_pos = f.get("pos")
            if not f_pos:
                continue
            dist = math.sqrt((x - f_pos[0]) ** 2 + (z - f_pos[1]) ** 2)
            if _room_match(f.get("room", ""), room) and dist <= DISTANCE_THRESHOLD:
                return label
            if dist < best_dist:
                best_dist, best_label = dist, label
        return best_label if best_dist <= DISTANCE_THRESHOLD else "floor"

    def _process_batch(self):
        raw_docs = list(self.col_raw.find({"processed": False}).limit(BATCH_SIZE))
        if not raw_docs:
            return

        furniture_items, dynamic_items = [], []
        for doc in raw_docs:
            label = doc.get("label", "").lower().strip()
            if not label:
                continue
            if self._is_furniture(label):
                furniture_items.append(doc)
            else:
                dynamic_items.append(doc)

        if furniture_items:
            self.col_scene.bulk_write(
                [self._make_furniture_op(d) for d in furniture_items],
                ordered=False
            )
            self._refresh_furniture_cache()

        if dynamic_items:
            self.col_dynamic.bulk_write(
                [self._make_dynamic_op(d) for d in dynamic_items],
                ordered=False
            )

        self.col_raw.update_many(
            {"_id": {"$in": [doc["_id"] for doc in raw_docs]}},
            {"$set": {
                "processed":    True,
                "processed_at": datetime.datetime.utcnow(),
            }}
        )

    def _make_furniture_op(self, doc):
        label = doc.get("label", "").lower().strip()
        now   = datetime.datetime.utcnow()
        return UpdateOne(
            {"label": label},
            {
                "$set": {
                    "label":        label,
                    "pos":          [doc.get("x", 0), doc.get("z", 0)],
                    "x":            doc.get("x", 0),
                    "y":            doc.get("y", 0),
                    "z":            doc.get("z", 0),
                    "room":         doc.get("room", ""),
                    "source":       doc.get("source", "sensor"),
                    "last_updated": now,
                    "is_static":    True,
                },
                "$setOnInsert": {"first_seen": now},
            },
            upsert=True
        )

    def _make_dynamic_op(self, doc):
        label = doc.get("label", "").lower().strip()
        now   = datetime.datetime.utcnow()
        x     = doc.get("x", 0)
        z     = doc.get("z", 0)
        room  = doc.get("room", "")

        old        = self.col_dynamic.find_one(
            {"label": label, "room": room},
            {"last_seen_on": 1, "spatial_rel": 1}
        )
        old_anchor = old.get("last_seen_on") if old else None
        old_rel    = old.get("spatial_rel")  if old else None

        u_id, u_dist = self._find_closest_user(x, z)
        is_held = False
        if u_id:
            if u_dist <= HAND_THRESHOLD:
                is_held = True
            elif old_rel == "held_by" and u_id == old_anchor and u_dist <= HAND_STICKY_LIMIT:
                is_held = True

        if is_held:
            anchor = u_id
            rel    = "held_by"
        else:
            anchor = self._find_closest_furniture(x, z, room, current_anchor=old_anchor)
            rel    = "at"

        if old_anchor and anchor != old_anchor:
            print(f"[Classifier] Anchor change: {label} "
                  f"({old_anchor} → {anchor}) | dist_to_user: {u_dist:.2f}m")

        category     = self._get_category(label)
        instance_key = self._make_instance_key(label, room, anchor)

        return UpdateOne(
            {"instance_key": instance_key},
            {
                "$set": {
                    "instance_key":  instance_key,
                    "label":         label,
                    "category":      category,
                    "room":          room,
                    "sensor_pos":    [x, z],
                    "last_seen":     now,
                    "last_seen_on":  anchor,
                    "spatial_rel":   rel,
                    "source":        doc.get("source", "sensor"),
                },
                "$inc":        {"seen_count": 1},
                "$setOnInsert": {
                    "first_seen": now,
                    "is_movable": True,
                },
            },
            upsert=True
        )

    def _cleanup_old_data(self):
        cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=CLEANUP_DAYS)
        self.col_raw.delete_many({
            "processed":    True,
            "processed_at": {"$lt": cutoff},
        })