import os
import datetime
import threading
import json
import faiss
from pymongo import MongoClient
from config import Config


class CleanupManager:
    """
    定期清理 MongoDB + FAISS。

    清理策略：
    ┌─────────────────────┬──────────────────────────────────┐
    │ Collection          │ 策略                              │
    ├─────────────────────┼──────────────────────────────────┤
    │ semantic_memories   │ 保留最近 RETAIN_DAYS 天            │
    │ activity_sequences  │ 保留最近 RETAIN_DAYS 天            │
    │ interaction_logs    │ 直接刪（訓練資料由 /export 另存）   │
    │ navigation_logs     │ 直接刪（同上）                     │
    │ observation_logs    │ 不清除（長期 weight 累加記憶）       │
    │ scene_snapshots     │ 不清除（家具座標）                  │
    │ FAISS index         │ 超過 MAX_FAISS_VECTORS 時重建      │
    │ debug_images/       │ 保留最近 7 天                      │
    └─────────────────────┴──────────────────────────────────┘
    """

    def __init__(self, mongo_client):
        self.db          = mongo_client[Config.DB_NAME]
        self.retain_days = getattr(Config, 'CLEANUP_RETAIN_DAYS', 90)
        self.max_vectors = getattr(Config, 'MAX_FAISS_VECTORS', 5000)
        self.index_path  = getattr(Config, 'FAISS_INDEX_PATH', 'robot_memory.index')
        self.meta_path   = getattr(Config, 'FAISS_META_PATH',  'robot_memory_meta.json')
        self._timer      = None

    # ─────────────────────────────────────────────
    # 定時排程（Flask 啟動時呼叫一次）
    # ─────────────────────────────────────────────
    def start_scheduler(self, interval_hours=24):
        print(f"[Cleanup] 排程啟動，每 {interval_hours}h 執行，保留 {self.retain_days} 天")
        self._schedule(interval_hours)

    def _schedule(self, interval_hours):
        self.run_all(auto=True)
        self._timer = threading.Timer(
            interval_hours * 3600,
            self._schedule,
            args=[interval_hours]
        )
        self._timer.daemon = True
        self._timer.start()

    def stop_scheduler(self):
        if self._timer:
            self._timer.cancel()

    # ─────────────────────────────────────────────
    # 全量清理（自動 + 手動共用）
    # ─────────────────────────────────────────────
    def run_all(self, auto=False):
        tag    = "[AutoCleanup]" if auto else "[ManualCleanup]"
        cutoff = datetime.datetime.utcnow() - datetime.timedelta(days=self.retain_days)
        print(f"\n{tag} cutoff={cutoff.strftime('%Y-%m-%d')} | retain={self.retain_days}d")

        stats = {
            "semantic_memories":  self._clean_by_timestamp("semantic_memories",  cutoff),
            "activity_sequences": self._clean_activity_sequences(cutoff),
            "interaction_logs":   self._clean_by_timestamp("interaction_logs",   cutoff),
            "navigation_logs":    self._clean_by_timestamp("navigation_logs",    cutoff),
            "observation_logs":   self._decay_and_prune_habits(),
            "faiss":              self._rebuild_faiss_if_needed(),
            "debug_images":       self._clean_debug_images(days=7),
        }

        total = sum(v for v in stats.values() if isinstance(v, int))
        print(f"{tag} 完成，共清理 {total} 筆 | {stats}\n")
        return stats

    # ─────────────────────────────────────────────
    # observation_logs：權重衰減 + 閾值刪除
    # ─────────────────────────────────────────────
    def _decay_and_prune_habits(self):
        """
        每次清理執行：
        1. 所有記錄 weight × decay_factor
        2. weight < min_threshold → 刪除（習慣已淡忘）
        3. 每次 /predict 觀察到行為時 weight += 1 強化

        衰減速度參考（decay=0.95，threshold=1.0）：
          weight=5  → ~31 天後低於閾值
          weight=12 → ~48 天後低於閾值
          weight=30 → ~65 天後低於閾值
        """
        try:
            decay     = getattr(Config, 'HABIT_DECAY_FACTOR',    0.95)
            threshold = getattr(Config, 'HABIT_MIN_WEIGHT',       1.0)
            col       = self.db["observation_logs"]

            # Step 1：對所有記錄執行乘法衰減
            # MongoDB 沒有原生 multiply update，用 pipeline update
            col.update_many(
                {},
                [{"$set": {"weight": {"$multiply": ["$weight", decay]}}}]
            )

            # Step 2：刪除 weight 低於閾值的記錄
            result  = col.delete_many({"weight": {"$lt": threshold}})
            pruned  = result.deleted_count

            # 統計還剩多少
            remaining = col.count_documents({})

            if pruned:
                print(f"  [observation_logs] 衰減 ×{decay}，刪除 {pruned} 筆（weight < {threshold}），剩餘 {remaining} 筆")
            else:
                print(f"  [observation_logs] 衰減 ×{decay}，無刪除，剩餘 {remaining} 筆")

            return pruned

        except Exception as e:
            print(f"  ❌ [observation_logs decay] {e}")
            return 0

    # ─────────────────────────────────────────────
    # 按 timestamp 欄位直接刪
    # ─────────────────────────────────────────────
    def _clean_by_timestamp(self, name, cutoff):
        try:
            n = self.db[name].delete_many({"timestamp": {"$lt": cutoff}}).deleted_count
            if n:
                print(f"  [{name}] 刪除 {n} 筆")
            return n
        except Exception as e:
            print(f"  ❌ [{name}] {e}")
            return 0

    # ─────────────────────────────────────────────
    # activity_sequences 用 date 字串比對（YYYY-MM-DD）
    # ─────────────────────────────────────────────
    def _clean_activity_sequences(self, cutoff):
        try:
            cutoff_str = cutoff.strftime("%Y-%m-%d")
            n = self.db["activity_sequences"].delete_many(
                {"date": {"$lt": cutoff_str}}
            ).deleted_count
            if n:
                print(f"  [activity_sequences] 刪除 {n} 筆（{cutoff_str} 之前）")
            return n
        except Exception as e:
            print(f"  ❌ [activity_sequences] {e}")
            return 0

    # ─────────────────────────────────────────────
    # FAISS：超過上限才重建，保留最新的 max_vectors 筆
    # ─────────────────────────────────────────────
    def _rebuild_faiss_if_needed(self):
        try:
            if not os.path.exists(self.meta_path):
                return 0

            with open(self.meta_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            total = len(metadata)
            if total <= self.max_vectors:
                print(f"  [FAISS] {total} 筆，未超過上限 {self.max_vectors}")
                return 0

            # 保留最新的 max_vectors 筆
            keep    = metadata[-self.max_vectors:]
            removed = total - len(keep)

            # 取舊索引的維度
            dim = 384
            if os.path.exists(self.index_path):
                old = faiss.read_index(self.index_path)
                dim = old.d

            # 重新 encode + 重建索引
            from sentence_transformers import SentenceTransformer
            model     = SentenceTransformer('paraphrase-MiniLM-L6-v2', device='cpu')
            texts     = [m.get('memory_text', '') for m in keep]
            vecs      = model.encode(texts).astype('float32')
            new_index = faiss.IndexFlatL2(dim)
            new_index.add(vecs)

            for i, m in enumerate(keep):
                m['faiss_idx'] = i

            faiss.write_index(new_index, self.index_path)
            with open(self.meta_path, 'w', encoding='utf-8') as f:
                json.dump(keep, f, ensure_ascii=False, indent=2)

            print(f"  [FAISS] 重建完成：{total} → {len(keep)} 筆（移除 {removed} 筆）")
            return removed

        except Exception as e:
            print(f"  ❌ [FAISS] {e}")
            return 0

    # ─────────────────────────────────────────────
    # debug_images：刪除超過 N 天的影像
    # ─────────────────────────────────────────────
    def _clean_debug_images(self, days=7):
        try:
            d = "debug_images"
            if not os.path.exists(d):
                return 0
            cutoff = datetime.datetime.now() - datetime.timedelta(days=days)
            count  = 0
            for f in os.listdir(d):
                fp = os.path.join(d, f)
                if os.path.isfile(fp) and \
                   datetime.datetime.fromtimestamp(os.path.getmtime(fp)) < cutoff:
                    os.remove(fp)
                    count += 1
            if count:
                print(f"  [debug_images] 刪除 {count} 個舊影像")
            return count
        except Exception as e:
            print(f"  ❌ [debug_images] {e}")
            return 0

    # ─────────────────────────────────────────────
    # 健康狀態查詢
    # ─────────────────────────────────────────────
    def status(self):
        cols = [
            "scene_snapshots", "observation_logs", "semantic_memories",
            "activity_sequences", "interaction_logs", "navigation_logs"
        ]
        result = {c: self.db[c].count_documents({}) for c in cols}

        result["faiss_vectors"] = 0
        if os.path.exists(self.meta_path):
            with open(self.meta_path, 'r') as f:
                result["faiss_vectors"] = len(json.load(f))

        result["debug_images"] = len(os.listdir("debug_images")) \
            if os.path.exists("debug_images") else 0

        result["retain_days"]  = self.retain_days
        result["max_faiss"]    = self.max_vectors
        return result