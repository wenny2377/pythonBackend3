import numpy as np
import faiss
import json
import os
import datetime 
from sentence_transformers import SentenceTransformer


class VectorMemory:
    def __init__(self, index_path="robot_memory.index", meta_path="robot_memory_meta.json"):
        self.index_path = index_path
        self.meta_path  = meta_path
        self.model      = SentenceTransformer('paraphrase-MiniLM-L6-v2', device='cpu')
        self.dim        = 384

        # 載入或建立 FAISS index
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            print(f"✅ [FAISS] 載入現有索引，共 {self.index.ntotal} 筆")
        else:
            self.index = faiss.IndexFlatL2(self.dim)
            print("✅ [FAISS] 建立新索引")

        # 載入 metadata（每筆向量對應的完整資料）
        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = []

# ─────────────────────────────────────────────
    # C. 加強儲存：文字向量 + 完整 metadata（含家具座標與空間關係）
    # ─────────────────────────────────────────────
    def add_memory(self, user_id, action, furniture_label, vlm_description,
                   detected_items=None, all_items=None, spatial_relations=None, 
                   furniture_pos=None, mongo_id=None):
        """
        修正版：新增支援 all_items 與 spatial_relations
        """
        detected_items = detected_items or []
        all_items = all_items or []
        spatial_relations = spatial_relations or []

        # 1. 組合語意豐富的文字：加入空間關係描述，讓向量搜尋能搜到「香蕉在桌上」
        items_str = ", ".join(detected_items) if detected_items else "nothing"
        
        # 將空間關係轉為文字描述，例如: "banana on desk. blue line next to table."
        spatial_text = " ".join([f"{r['subject']} {r['relation']} {r['object']}." for r in spatial_relations])
        
        # 最終要向量化的長字串
        memory_text = f"{user_id} {action} near {furniture_label} with {items_str}. {vlm_description} {spatial_text}".strip()

        vec = self.model.encode([memory_text]).astype('float32')
        self.index.add(vec)

        # 2. metadata 同時儲存所有豐富資訊
        entry = {
            "faiss_idx":         self.index.ntotal - 1,
            "user":              user_id,
            "action":            action,
            "instance":          furniture_label,
            "interacting_items": detected_items,   # 修改 key 名稱與 app.py 對齊
            "all_items":         all_items,         # 新增：畫面所有物品
            "spatial_relations": spatial_relations, # 新增：空間關係
            "furniture_pos":     furniture_pos,
            "mongo_id":          str(mongo_id) if mongo_id else None,
            "description":       vlm_description,
            "memory_text":       memory_text,
            "timestamp":         datetime.datetime.now().isoformat()
        }
        self.metadata.append(entry)
        self._save()

        print(f"✅ [FAISS] 新增記憶（含空間語意）：{furniture_label} | {action}")
        return entry

    # ─────────────────────────────────────────────
    # 模糊搜尋：回傳含家具座標的完整結果
    # ─────────────────────────────────────────────
    def search_habit(self, query, user_id=None, top_k=3):
        """
        自然語言查詢 → FAISS 向量搜尋 → 回傳含導航座標的結果

        範例查詢：
        - "媽媽通常在哪裡喝水"
        - "Where does mom usually sit?"
        - "找媽媽常用的東西"
        """
        if self.index.ntotal == 0:
            return []

        query_vec = self.model.encode([query]).astype('float32')
        k         = min(top_k * 3, self.index.ntotal)  # 多搜幾筆再過濾
        distances, indices = self.index.search(query_vec, k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue

            entry = self.metadata[idx]

            # 可選：過濾特定用戶
            if user_id and entry.get('user') != user_id:
                continue

            results.append({
                "user":          entry.get('user'),
                "action":        entry.get('action'),
                "instance":      entry.get('instance'),
                "items":         entry.get('items', []),
                "furniture_pos": entry.get('furniture_pos'),   # 直接可用於導航
                "mongo_id":      entry.get('mongo_id'),
                "description":   entry.get('description'),
                "similarity":    float(1 / (1 + dist))         # 距離轉換為相似度分數 0~1
            })

            if len(results) >= top_k:
                break

        return results

    # ─────────────────────────────────────────────
    # 習慣聚合：同一用戶同一動作，哪個家具最高頻
    # ─────────────────────────────────────────────
    def get_top_habit(self, query, user_id=None, top_k=1):
        """
        查詢最常發生的行為地點
        例如：「媽媽最常在哪坐著」→ 回傳頻率最高的家具 + 座標
        """
        results = self.search_habit(query, user_id=user_id, top_k=20)

        # 按 instance 聚合，計算出現次數
        habit_count = {}
        for r in results:
            key = r['instance']
            if key not in habit_count:
                habit_count[key] = {
                    "instance":      r['instance'],
                    "furniture_pos": r['furniture_pos'],
                    "count":         0,
                    "actions":       [],
                    "items":         []
                }
            habit_count[key]['count']   += 1
            habit_count[key]['actions'].append(r['action'])
            habit_count[key]['items'].extend(r.get('items', []))

        sorted_habits = sorted(habit_count.values(), key=lambda x: x['count'], reverse=True)

        if not sorted_habits:
            return None

        top = sorted_habits[:top_k]
        for h in top:
            h['items'] = list(set(h['items']))  # 去重

        return top[0] if top_k == 1 else top

    # ─────────────────────────────────────────────
    # 持久化
    # ─────────────────────────────────────────────
    def _save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    # ─────────────────────────────────────────────
    # 語意擴散：把模糊查詢展開成具體物品標籤
    # ─────────────────────────────────────────────
    def expand_query(self, query: str, candidate_items: list, top_k=10, threshold=0.35) -> list:
        """
        把模糊需求展開成最相關的具體物品標籤。

        例如：
          query="甜的"     candidates=[chocolate, cookie, apple, knife, cup]
          → ["chocolate", "cookie"]

          query="喝的東西"  candidates=[cup, water_bottle, milk, knife, plate]
          → ["cup", "water_bottle", "milk"]

        candidate_items：從 scene_snapshots 收集到的所有出現過的物品標籤。
        threshold：語意相似度門檻，低於此值不列入（預設 0.35）。
        """
        if not candidate_items:
            return []

        try:
            query_vec = self.model.encode(query)
            scored    = []

            for item in candidate_items:
                item_vec = self.model.encode(item)
                sim      = float(
                    np.dot(query_vec, item_vec) /
                    (np.linalg.norm(query_vec) * np.linalg.norm(item_vec) + 1e-8)
                )
                if sim >= threshold:
                    scored.append((item, sim))

            scored.sort(key=lambda x: x[1], reverse=True)
            result = [item for item, _ in scored[:top_k]]

            print(f"[SemanticExpand] '{query}' → {result}")
            return result

        except Exception as e:
            print(f"⚠️ [SemanticExpand] {e}")
            return []

    def get_all_known_items(self) -> list:
        """FAISS metadata 中所有出現過的物品標籤（去重）"""
        items = set()
        for m in self.metadata:
            for item in m.get('interacting_items', []):
                items.add(item.lower())
            for item in m.get('all_items', []):
                items.add(item.lower())
        return list(items)