import threading
import datetime
import time
import math
from pymongo import UpdateOne
from config import Config

# ─────────────────────────────────────────────
# 靜態家具清單
# ─────────────────────────────────────────────
FURNITURE_LABELS = {
    "sofa", "couch", "bed", "table", "desk", "chair", "tv", "television","desk2","chair2",
    "refrigerator", "fridge", "sink", "toilet", "shelf", "shelf2",
    "wardrobe", "cabinet", "nightstand", "bookshelf", "dresser",
    "stove", "bathtub", "dining_table", "kitchen_table",
    "mom's bed", "dad's bed", "mom_bed", "dad_bed",
}

POLL_INTERVAL      = 5      # 秒
BATCH_SIZE         = 50
DISTANCE_THRESHOLD = 0.8    # 公尺：判定物件在家具上的距離閾值
CLEANUP_DAYS       = 7      # 刪除多久前的 raw 資料

class ObjectClassifier:

    def __init__(self, db):
        self.db         = db
        self.col_raw    = db["raw_objects"]
        self.col_scene  = db["scene_snapshots"]
        self.col_dynamic= db["dynamic_objects"]
        self._running   = False
        self._thread    = None
        self._furniture_cache = []
        self._user_cache      = []   # 從 user_positions 讀取

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print("[Classifier] ✅ 背景分類執行緒啟動 (含空間語意對齊)")

    def stop(self):
        self._running = False
        print("[Classifier] 已停止")

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
        """從 scene_snapshots 取得目前地圖上的家具位置"""
        self._furniture_cache = list(self.col_scene.find({}, {"label": 1, "pos": 1, "room": 1}))
        # 同時更新用戶位置快取（從 /predict 寫入的 user_positions）
        self._user_cache = list(self.db["user_positions"].find(
            {}, {"user_id": 1, "x": 1, "z": 1, "updated_at": 1}
        ))

    def _find_closest_user(self, x, z, stale_seconds=60):
        """
        找距離物件最近的用戶。
        stale_seconds：超過此秒數未更新的位置視為過期（用戶已離開）。
        回傳 (user_id, dist) 或 (None, inf)
        """
        now     = datetime.datetime.utcnow()
        best_uid  = None
        best_dist = float('inf')

        for u in self._user_cache:
            # 過期檢查：user_pos 超過 stale_seconds 秒沒更新 → 忽略
            updated = u.get("updated_at")
            if updated:
                age = (now - updated).total_seconds()
                if age > stale_seconds:
                    continue

            ux = u.get("x", 0)
            uz = u.get("z", 0)
            dist = math.sqrt((x - ux)**2 + (z - uz)**2)
            if dist < best_dist:
                best_dist = dist
                best_uid  = u.get("user_id", "")

        return best_uid, best_dist

    def _find_closest_furniture(self, x, z, room):
        """
        幾何運算：尋找距離物件最近的家具作為語意錨點
        策略：
          1. 優先找同房間（模糊比對）且距離 <= DISTANCE_THRESHOLD 的家具
          2. 同房間找不到 → 全場景找最近的（跨房間 fallback）
          3. 全場景最近距離還是 > DISTANCE_THRESHOLD → 回傳 "floor"
        """
        def _room_match(f_room, obj_room):
            """模糊房間比對：大小寫、空格不敏感"""
            if not f_room or not obj_room:
                return False
            f_norm   = f_room.lower().replace(" ", "").replace("_", "")
            obj_norm = obj_room.lower().replace(" ", "").replace("_", "")
            return obj_norm in f_norm or f_norm in obj_norm

        # Pass 1：同房間
        same_room_label = None
        same_room_dist  = float('inf')
        for f in self._furniture_cache:
            if not _room_match(f.get("room", ""), room):
                continue
            f_pos = f.get("pos")
            if f_pos and len(f_pos) >= 2:
                dist = math.sqrt((x - f_pos[0])**2 + (z - f_pos[1])**2)
                if dist < same_room_dist:
                    same_room_dist  = dist
                    same_room_label = f.get("label")

        if same_room_label and same_room_dist <= DISTANCE_THRESHOLD:
            return same_room_label

        # Pass 2：全場景 fallback（房間名對不上時仍能找到最近家具）
        global_label = None
        global_dist  = float('inf')
        for f in self._furniture_cache:
            f_pos = f.get("pos")
            if f_pos and len(f_pos) >= 2:
                dist = math.sqrt((x - f_pos[0])**2 + (z - f_pos[1])**2)
                if dist < global_dist:
                    global_dist  = dist
                    global_label = f.get("label")

        if global_label and global_dist <= DISTANCE_THRESHOLD:
            return global_label

        return "floor"

    def _process_batch(self):
        raw_docs = list(self.col_raw.find({"processed": False}).limit(BATCH_SIZE))
        if not raw_docs:
            return

        furniture_ops = []
        dynamic_ops   = []

        for doc in raw_docs:
            label = doc.get("label", "").lower().strip()
            if not label:
                continue

            if label in FURNITURE_LABELS:
                furniture_ops.append(self._make_furniture_op(doc))
            else:
                dynamic_ops.append(self._make_dynamic_op(doc))

        if furniture_ops:
            self.col_scene.bulk_write(furniture_ops, ordered=False)
            print(f"[Classifier] 🔄 靜態家具同步: {len(furniture_ops)} 筆")

        if dynamic_ops:
            self.col_dynamic.bulk_write(dynamic_ops, ordered=False)
            print(f"[Classifier] 📦 動態物件追蹤: {len(dynamic_ops)} 筆")

        ids = [doc["_id"] for doc in raw_docs]
        self.col_raw.update_many(
            {"_id": {"$in": ids}},
            {"$set": {"processed": True, "processed_at": datetime.datetime.utcnow()}}
        )

    def _make_furniture_op(self, doc):
        label = doc.get("label", "").lower().strip()
        now   = datetime.datetime.utcnow()
        return UpdateOne(
            {"label": label},
            {
                "$set": {
                    "label":  label,
                    "pos":    [doc.get("x", 0), doc.get("z", 0)],
                    "x":      doc.get("x", 0),
                    "y":      doc.get("y", 0),
                    "z":      doc.get("z", 0),
                    "room":   doc.get("room", ""),
                    "image":  doc.get("image", ""),
                    "source": doc.get("source", "sensor"),
                    "last_updated": now,
                    "is_static": True
                },
                "$setOnInsert": {"first_seen": now},
            },
            upsert=True
        )

    def _make_dynamic_op(self, doc):
        label = doc.get("label", "").lower().strip()
        now   = datetime.datetime.utcnow()
        x, z  = doc.get("x", 0), doc.get("z", 0)
        room  = doc.get("room", "")

        # 優先判斷：物件是否在人手邊（距離 < 0.5m）
        USER_DIST_THRESHOLD = 0.5
        user_id, user_dist  = self._find_closest_user(x, z)

        if user_id and user_dist <= USER_DIST_THRESHOLD:
            anchor      = user_id   # e.g. "User_Dad"
            spatial_rel = "held_by"
            print(f"[Classifier] 🤲 In-hand: {label} near {user_id} (dist={user_dist:.2f}m)")
        else:
            anchor      = self._find_closest_furniture(x, z, room)
            spatial_rel = "at"
            print(f"[Classifier] 🔍 Binding Check: {label} (at {x}, {z}) -> Selected Anchor: {anchor}")

        return UpdateOne(
            {"label": label},
            {
                "$set": {
                    "label":      label,
                    "room":       room,
                    "sensor_pos": [x, z],
                    "last_seen":  now,
                    "source":     doc.get("source", "sensor"),
                },
                "$inc": {"seen_count": 1},
                "$setOnInsert": {
                    "first_seen":     now,
                    "last_seen_on":   anchor,
                    "spatial_rel":    spatial_rel,
                    "interact_count": 0,
                    "is_movable":     True
                },
            },
            upsert=True
        )

    def _cleanup_old_data(self):
        """定期清除 raw_objects，維持資料庫效能"""
        cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=CLEANUP_DAYS)
        result = self.col_raw.delete_many({
            "processed": True,
            "processed_at": {"$lt": cutoff}
        })
        if result.deleted_count > 0:
            print(f"[Classifier] 🧹 已清理 {result.deleted_count} 筆過期原始資料")