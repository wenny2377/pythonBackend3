import os
import datetime
import threading
import json
import faiss
from config import Config


class CleanupManager:
    def __init__(self, mongo_client):
        self.db               = mongo_client[Config.DB_NAME]
        self.retain_days      = getattr(Config, 'CLEANUP_RETAIN_DAYS',   90)
        self.max_vectors      = getattr(Config, 'MAX_FAISS_VECTORS',    5000)
        self.index_path       = getattr(Config, 'FAISS_INDEX_PATH',      'robot_memory.index')
        self.meta_path        = getattr(Config, 'FAISS_META_PATH',       'robot_memory_meta.json')
        self.dyn_index_path   = getattr(Config, 'DYNAMIC_INDEX_PATH',    'dynamic_memory.index')
        self.dyn_meta_path    = getattr(Config, 'DYNAMIC_META_PATH',     'dynamic_memory_meta.json')
        self._timer           = None

    # ─────────────────────────────────────────────
    # 排程
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
    # 全量清理
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
            "conversation_logs":  self._clean_by_timestamp("conversation_logs",  cutoff),
            "observation_logs":   self._decay_and_prune_habits(),
            "faiss_habit":        self._rebuild_faiss_if_needed(
                                      self.index_path, self.meta_path, "習慣記憶"),
            "faiss_dynamic":      self._rebuild_faiss_if_needed(
                                      self.dyn_index_path, self.dyn_meta_path, "動態物件"),
            "debug_images":       self._clean_debug_images(days=7),
        }

        total = sum(v for v in stats.values() if isinstance(v, int))
        print(f"{tag} 完成，共清理 {total} 筆 | {stats}\n")
        return stats

    # ─────────────────────────────────────────────
    # observation_logs：weight 衰減 + 閾值刪除
    # ─────────────────────────────────────────────
    def _decay_and_prune_habits(self):
        try:
            decay     = getattr(Config, 'HABIT_DECAY_FACTOR', 0.95)
            threshold = getattr(Config, 'HABIT_MIN_WEIGHT',    1.0)
            col       = self.db["observation_logs"]

            col.update_many(
                {},
                [{"$set": {"weight": {"$multiply": ["$weight", decay]}}}]
            )

            result    = col.delete_many({"weight": {"$lt": threshold}})
            pruned    = result.deleted_count
            remaining = col.count_documents({})

            if pruned:
                print(f"  [observation_logs] 衰減 ×{decay}，刪除 {pruned} 筆，剩餘 {remaining} 筆")
            else:
                print(f"  [observation_logs] 衰減 ×{decay}，無刪除，剩餘 {remaining} 筆")

            return pruned

        except Exception as e:
            print(f"  ❌ [observation_logs decay] {e}")
            return 0

    # ─────────────────────────────────────────────
    # 按 timestamp 刪
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
    # activity_sequences 用 date 字串比對
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
    # FAISS 重建（習慣記憶 + 動態物件共用）
    # 注意：使用 IndexFlatIP（cosine），與 memory_vector.py 一致
    # ─────────────────────────────────────────────
    def _rebuild_faiss_if_needed(self, index_path, meta_path, label="FAISS"):
        try:
            if not os.path.exists(meta_path):
                return 0

            with open(meta_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            total = len(metadata)
            if total <= self.max_vectors:
                print(f"  [FAISS {label}] {total} 筆，未超過上限 {self.max_vectors}")
                return 0

            keep    = metadata[-self.max_vectors:]
            removed = total - len(keep)

            from sentence_transformers import SentenceTransformer
            import numpy as np
            model = SentenceTransformer('paraphrase-MiniLM-L6-v2', device='cpu')
            texts = [m.get('memory_text', '') for m in keep]
            vecs  = model.encode(texts).astype('float32')

            # ← FIX: IndexFlatIP（cosine），與 memory_vector.py 一致
            faiss.normalize_L2(vecs)
            new_index = faiss.IndexFlatIP(384)
            new_index.add(vecs)

            for i, m in enumerate(keep):
                m['faiss_idx'] = i

            faiss.write_index(new_index, index_path)
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(keep, f, ensure_ascii=False, indent=2)

            print(f"  [FAISS {label}] 重建：{total} → {len(keep)} 筆（移除 {removed} 筆）")
            return removed

        except Exception as e:
            print(f"  ❌ [FAISS {label}] {e}")
            return 0

    # ─────────────────────────────────────────────
    # debug_images 清理
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
            "activity_sequences", "interaction_logs", "navigation_logs",
            "conversation_logs", "dynamic_objects"
        ]
        result = {c: self.db[c].count_documents({}) for c in cols}

        for label, path in [("faiss_habit", self.meta_path),
                             ("faiss_dynamic", self.dyn_meta_path)]:
            result[label] = 0
            if os.path.exists(path):
                with open(path, 'r') as f:
                    result[label] = len(json.load(f))

        result["debug_images"] = len(os.listdir("debug_images")) \
            if os.path.exists("debug_images") else 0
        result["retain_days"]  = self.retain_days
        result["max_faiss"]    = self.max_vectors
        return result