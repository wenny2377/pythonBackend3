"""
manifold_engine.py
------------------
ManifoldEngine：行為特徵向量化 → UMAP 2D 投影 → HDBSCAN 聚類 → 軌跡意圖預判

流程：
  每次 /predict 觀察 → build_feature_vector() → record_point()
  每 50 筆           → refit_manifold()  (async thread)
  每次觀察           → predict_intent()  → 回傳 trigger / confidence
"""

import threading
import datetime
import numpy as np
from bson import ObjectId

# ── 延遲 import，讓沒裝的環境不會在 import 時爆炸 ──
try:
    import umap
    import hdbscan
    from sklearn.preprocessing import StandardScaler
    _MANIFOLD_OK = True
except ImportError:
    _MANIFOLD_OK = False
    print("⚠️  [Manifold] umap-learn / hdbscan 未安裝，Manifold 功能停用")
    print("   pip install umap-learn hdbscan scikit-learn")


TRIGGER_THRESHOLD = 2.0   # UMAP 2D 距離閾值
MIN_CONFIDENCE    = 0.60  # 低於此值不觸發提案
MIN_POINTS_REFIT  = 30    # 至少幾筆才 refit
REFIT_EVERY       = 50    # 每幾筆新資料觸發一次 refit
TRAJECTORY_WINDOW = 5     # 用最近幾個點算軌跡慣性


class ManifoldEngine:

    def __init__(self, db, sbert_model):
        self.db          = db
        self.sbert       = sbert_model
        self._lock       = threading.Lock()

        # UMAP / HDBSCAN 模型（refit 後才有）
        self._umap_model    = None
        self._hdbscan_model = None
        self._scaler        = None
        self._fitted        = False
        self._point_count   = 0   # 累積筆數計數器

        print("✅ [ManifoldEngine] 初始化完成")
        if not _MANIFOLD_OK:
            print("   ⚠️  Manifold 功能停用（缺少套件）")

    # ──────────────────────────────────────────────────
    # 公開 API 1：build_feature_vector
    # 在 /predict 取得 perception 結果後呼叫
    # ──────────────────────────────────────────────────
    def build_feature_vector(self, user_id: str, action: str,
                              user_pos: dict, room_name: str,
                              detected_items: list, confidence: str) -> np.ndarray:
        """
        回傳 1155-dim numpy 向量：
          384  動作語意 (SBERT)
          2    用戶座標 (x, z 正規化)
          384  物件語意 (SBERT)
          384  房間語意 (SBERT)
          1    綁定信心度
        """
        try:
            # 動作語意
            action_vec = self.sbert.encode(
                action or "unknown", normalize_embeddings=True
            ).astype(np.float32)

            # 用戶座標
            x = float(user_pos.get("x", 0)) / 10.0 if user_pos else 0.0
            z = float(user_pos.get("z", 0)) / 10.0 if user_pos else 0.0
            pos_vec = np.array([x, z], dtype=np.float32)

            # 物件語意（多個物件平均池化）
            if detected_items:
                items_text = " ".join(detected_items[:5])
            else:
                items_text = "nothing"
            items_vec = self.sbert.encode(
                items_text, normalize_embeddings=True
            ).astype(np.float32)

            # 房間語意
            room_vec = self.sbert.encode(
                room_name or "unknown room", normalize_embeddings=True
            ).astype(np.float32)

            # 信心度
            conf_map  = {"high": 1.0, "medium": 0.6, "low": 0.3, "unknown": 0.1}
            conf_val  = conf_map.get(str(confidence).lower(), 0.1)
            conf_vec  = np.array([conf_val], dtype=np.float32)

            feature = np.concatenate([action_vec, pos_vec, items_vec, room_vec, conf_vec])
            return feature  # shape: (1155,)

        except Exception as e:
            print(f"[Manifold] build_feature_vector error: {e}")
            return np.zeros(1155, dtype=np.float32)

    # ──────────────────────────────────────────────────
    # 公開 API 2：record_point
    # 把特徵向量存進 MongoDB manifold_points
    # ──────────────────────────────────────────────────
    def record_point(self, user_id: str, feature_vec: np.ndarray,
                     action: str, bound_label: str) -> str:
        """回傳 inserted _id (str)"""
        try:
            doc = {
                "user_id":        user_id,
                "action":         action,
                "bound_label":    bound_label,
                "feature_vec":    feature_vec.tolist(),
                "manifold_xy":    None,   # refit 後填入
                "cluster_id":     -1,
                "service_result": "pending",
                "voice_trigger":  False,
                "timestamp":      datetime.datetime.utcnow(),
            }
            result = self.db.manifold_points.insert_one(doc)
            self._point_count += 1
            print(f"   📍 [Manifold] point recorded #{self._point_count} | {user_id} {action}")
            return str(result.inserted_id)

        except Exception as e:
            print(f"[Manifold] record_point error: {e}")
            return ""

    # ──────────────────────────────────────────────────
    # 公開 API 3：maybe_refit
    # 在 /predict 末尾呼叫，每 REFIT_EVERY 筆觸發一次
    # ──────────────────────────────────────────────────
    def maybe_refit(self, user_id: str):
        """每 REFIT_EVERY 筆非同步 refit（不阻塞主執行緒）"""
        if self._point_count % REFIT_EVERY == 0 and self._point_count > 0:
            t = threading.Thread(
                target=self._refit_manifold_all,
                daemon=True
            )
            t.start()
            print(f"   🔄 [Manifold] async refit triggered (total={self._point_count})")

    # ──────────────────────────────────────────────────
    # 公開 API 4：predict_intent
    # 每次 /predict 呼叫，回傳是否觸發提案
    # ──────────────────────────────────────────────────
    def predict_intent(self, user_id: str, current_feature: np.ndarray) -> dict:
        """
        回傳：
        {
            "trigger":        bool,
            "intent":         str,
            "cluster_id":     int,
            "confidence":     float,
            "current_xy":     [x, y],
            "predicted_xy":   [x, y],
        }
        """
        empty = {
            "trigger": False, "intent": "unknown",
            "cluster_id": -1, "confidence": 0.0,
            "current_xy": [0, 0], "predicted_xy": [0, 0],
        }

        if not _MANIFOLD_OK or not self._fitted:
            return empty

        try:
            with self._lock:
                scaled  = self._scaler.transform([current_feature])
                xy      = self._umap_model.transform(scaled)[0]  # (2,)

            current_xy = xy.tolist()

            # 取最近 TRAJECTORY_WINDOW 筆的 manifold_xy
            recent = list(
                self.db.manifold_points.find(
                    {"user_id": user_id, "manifold_xy": {"$ne": None}},
                    {"manifold_xy": 1}
                ).sort("timestamp", -1).limit(TRAJECTORY_WINDOW)
            )

            if len(recent) >= 2:
                pts      = np.array([r["manifold_xy"] for r in reversed(recent)])
                velocity = pts[-1] - pts[0]
                predicted_xy = (xy + velocity).tolist()
            else:
                predicted_xy = current_xy

            # 找最近且 success_rate > 0.3 的簇
            clusters = list(self.db.behavior_clusters.find(
                {"success_rate": {"$gte": 0.3}}
            ))

            if not clusters:
                return {**empty, "current_xy": current_xy, "predicted_xy": predicted_xy}

            pred_arr = np.array(predicted_xy)
            best_cluster  = None
            best_dist     = float("inf")

            for c in clusters:
                center = np.array(c["center_xy"])
                dist   = float(np.linalg.norm(pred_arr - center))
                if dist < best_dist:
                    best_dist    = dist
                    best_cluster = c

            confidence = max(0.0, 1.0 - best_dist / TRIGGER_THRESHOLD)
            trigger    = confidence >= MIN_CONFIDENCE

            print(f"   🧭 [Manifold] intent={best_cluster['dominant_action']} "
                  f"conf={confidence:.2f} trigger={trigger}")

            return {
                "trigger":      trigger,
                "intent":       best_cluster["dominant_action"],
                "cluster_id":   best_cluster.get("cluster_id", -1),
                "confidence":   round(confidence, 3),
                "current_xy":   current_xy,
                "predicted_xy": predicted_xy,
            }

        except Exception as e:
            print(f"[Manifold] predict_intent error: {e}")
            return empty

    # ──────────────────────────────────────────────────
    # 公開 API 5：update_service_result
    # /service_response 呼叫，把人的回應寫回流形點
    # ──────────────────────────────────────────────────
    def update_service_result(self, point_id: str, result: str):
        """result: 'accepted' / 'rejected' / 'ignored'"""
        try:
            self.db.manifold_points.update_one(
                {"_id": ObjectId(point_id)},
                {"$set": {"service_result": result}}
            )
            # 更新 intent_stats（給 ServiceProposalEngine 用）
            point = self.db.manifold_points.find_one({"_id": ObjectId(point_id)})
            if point:
                intent = point.get("action", "unknown")
                user   = point.get("user_id", "unknown")
                self.db.intent_stats.update_one(
                    {"user_id": user, "intent": intent},
                    {"$inc": {result: 1}},
                    upsert=True
                )
            print(f"   ✅ [Manifold] point {point_id} → {result}")
        except Exception as e:
            print(f"[Manifold] update_service_result error: {e}")

    # ──────────────────────────────────────────────────
    # 內部：refit（在 async thread 裡跑）
    # ──────────────────────────────────────────────────
    def _refit_manifold_all(self):
        if not _MANIFOLD_OK:
            return
        try:
            # 取所有 manifold_points
            docs = list(self.db.manifold_points.find(
                {}, {"_id": 1, "feature_vec": 1, "action": 1, "user_id": 1}
            ))
            if len(docs) < MIN_POINTS_REFIT:
                print(f"   ⚠️  [Manifold] refit skipped: only {len(docs)} pts < {MIN_POINTS_REFIT}")
                return

            print(f"   🔄 [Manifold] refitting {len(docs)} points...")
            t0 = datetime.datetime.now()

            X       = np.array([d["feature_vec"] for d in docs], dtype=np.float32)
            ids     = [d["_id"] for d in docs]

            # StandardScaler
            scaler  = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # UMAP
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=15,
                min_dist=0.1,
                metric="cosine",
                random_state=42,
            )
            xy_all  = reducer.fit_transform(X_scaled)  # (N, 2)

            # HDBSCAN
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=5,
                prediction_data=True,
            )
            labels = clusterer.fit_predict(xy_all)     # (N,)

            # 更新 MongoDB manifold_points
            bulk = []
            from pymongo import UpdateOne
            for i, doc_id in enumerate(ids):
                bulk.append(UpdateOne(
                    {"_id": doc_id},
                    {"$set": {
                        "manifold_xy": xy_all[i].tolist(),
                        "cluster_id":  int(labels[i]),
                    }}
                ))
            if bulk:
                self.db.manifold_points.bulk_write(bulk)

            # 更新 behavior_clusters
            unique_labels = set(labels) - {-1}
            self.db.behavior_clusters.delete_many({})  # 清空重建

            for cid in unique_labels:
                mask    = labels == cid
                pts     = xy_all[mask]
                actions = [docs[i]["action"] for i in range(len(docs)) if labels[i] == cid]

                # 主導行為（眾數）
                dominant = max(set(actions), key=actions.count)

                # 成功率（從 manifold_points service_result 算）
                point_ids = [ids[i] for i in range(len(docs)) if labels[i] == cid]
                results   = list(self.db.manifold_points.find(
                    {"_id": {"$in": point_ids}},
                    {"service_result": 1}
                ))
                total    = len([r for r in results if r["service_result"] != "pending"])
                accepted = len([r for r in results if r["service_result"] == "accepted"])
                rate     = accepted / total if total > 0 else 0.5  # 無資料時預設 0.5

                self.db.behavior_clusters.insert_one({
                    "cluster_id":       int(cid),
                    "dominant_action":  dominant,
                    "center_xy":        pts.mean(axis=0).tolist(),
                    "size":             int(mask.sum()),
                    "success_rate":     round(rate, 3),
                    "updated_at":       datetime.datetime.utcnow(),
                })

            # 更新 self（加鎖）
            with self._lock:
                self._scaler        = scaler
                self._umap_model    = reducer
                self._hdbscan_model = clusterer
                self._fitted        = True

            elapsed = (datetime.datetime.now() - t0).total_seconds()
            print(f"   ✅ [Manifold] refit done: {len(unique_labels)} clusters | {elapsed:.1f}s")

        except Exception as e:
            import traceback
            print(f"[Manifold] refit error: {e}\n{traceback.format_exc()}")