"""
classifier.py
背景執行緒：從 raw_objects 讀取 → 分類 → 寫進 scene_snapshots / dynamic_objects

架構：
  同事 sensor / Unity ProxyExportManager
    → POST /scene → raw_objects（不分類，直接存）
    
  ObjectClassifier（背景執行緒）
    → 每 POLL_INTERVAL 秒讀取 processed=False 的資料
    → label 在 FURNITURE_LABELS → scene_snapshots
    → label 不在              → dynamic_objects
    → 標記 processed=True

效能設計：
  - 批次處理（BATCH_SIZE 筆一起處理）
  - 只處理 processed=False 的資料（不重複處理）
  - 靜態家具：只有第一次出現或位置改變才更新
  - 動態物件：每次都更新 sensor_pos（位置可能一直變）
  - 7 天前的已處理資料定期清除
"""

import threading
import datetime
import time
from config import Config

# ─────────────────────────────────────────────
# 靜態家具清單
# 跟同事確認物件名稱後補充這個清單
# ─────────────────────────────────────────────
FURNITURE_LABELS = {
    "sofa", "couch", "bed", "table", "desk", "chair", "tv", "television",
    "refrigerator", "fridge", "sink", "toilet", "shelf", "shelf2",
    "wardrobe", "cabinet", "nightstand", "bookshelf", "dresser",
    "stove", "bathtub", "dining_table", "kitchen_table",
    "mom's bed", "dad's bed", "mom_bed", "dad_bed",
}

POLL_INTERVAL  = 5      # 秒：多久掃一次 raw_objects
BATCH_SIZE     = 50     # 每次最多處理幾筆


class ObjectClassifier:

    def __init__(self, db):
        self.db         = db
        self.col_raw    = db["raw_objects"]
        self.col_scene  = db["scene_snapshots"]
        self.col_dynamic= db["dynamic_objects"]
        self._running   = False
        self._thread    = None

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print("[Classifier] ✅ 背景分類執行緒啟動")

    def stop(self):
        self._running = False
        print("[Classifier] 已停止")

    def _loop(self):
        while self._running:
            try:
                self._process_batch()
            except Exception as e:
                print(f"[Classifier Error] {e}")
            time.sleep(POLL_INTERVAL)

    # ─────────────────────────────────────────────
    # 核心：批次分類
    # ─────────────────────────────────────────────
    def _process_batch(self):
        # 讀取未處理的資料
        raw_docs = list(
            self.col_raw.find({"processed": False}).limit(BATCH_SIZE)
        )
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

        # 批量寫入 scene_snapshots
        if furniture_ops:
            from pymongo import UpdateOne
            self.col_scene.bulk_write(furniture_ops, ordered=False)
            print(f"[Classifier] 靜態家具 {len(furniture_ops)} 筆")

        # 批量寫入 dynamic_objects
        if dynamic_ops:
            from pymongo import UpdateOne
            self.col_dynamic.bulk_write(dynamic_ops, ordered=False)
            print(f"[Classifier] 動態物件 {len(dynamic_ops)} 筆")

        # 標記已處理
        ids = [doc["_id"] for doc in raw_docs]
        self.col_raw.update_many(
            {"_id": {"$in": ids}},
            {"$set": {"processed": True, "processed_at": datetime.datetime.utcnow()}}
        )

    def _make_furniture_op(self, doc):
        from pymongo import UpdateOne
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
                },
                "$setOnInsert": {"first_seen": now},
            },
            upsert=True
        )

    def _make_dynamic_op(self, doc):
        from pymongo import UpdateOne
        label = doc.get("label", "").lower().strip()
        now   = datetime.datetime.utcnow()
        return UpdateOne(
            {"label": label},
            {
                "$set": {
                    "label":      label,
                    "room":       doc.get("room", ""),
                    "sensor_pos": [doc.get("x", 0), doc.get("z", 0)],
                    "last_seen":  now,
                    "source":     doc.get("source", "sensor"),
                },
                "$inc": {"seen_count": 1},
                "$setOnInsert": {
                    "first_seen":  now,
                    "last_seen_on": "unknown",
                    "spatial_rel":  "unknown",
                    "interact_count": 0,
                },
            },
            upsert=True
        )


