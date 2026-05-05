import math
import threading
import datetime
import numpy as np
from bson import ObjectId

try:
    import umap
    import hdbscan
    from sklearn.preprocessing import StandardScaler
    _MANIFOLD_OK = True
except ImportError:
    _MANIFOLD_OK = False
    print("⚠️  [Manifold] umap-learn / hdbscan 未安裝，Manifold 功能停用")

TRIGGER_THRESHOLD = 2.0
MIN_CONFIDENCE    = 0.60
MIN_POINTS_REFIT  = 30
REFIT_EVERY       = 20
TRAJECTORY_WINDOW = 5


class ManifoldEngine:

    def __init__(self, db, sbert_model):
        self.db             = db
        self.sbert          = sbert_model
        self._lock          = threading.Lock()
        self._umap_model    = None
        self._hdbscan_model = None
        self._scaler        = None
        self._fitted        = False
        self._point_count   = 0
        print("✅ [ManifoldEngine] 初始化完成（1158-dim）")

    def build_feature_vector(self, user_id, action, user_pos,
                              room_name, detected_items, confidence,
                              virtual_hour=None, prev_action=None):
        try:
            action_vec = self.sbert.encode(
                action or "unknown",
                normalize_embeddings=True).astype(np.float32)

            h = float(virtual_hour) if virtual_hour is not None \
                else float(datetime.datetime.now().hour)
            time_vec = np.array([
                math.sin(2 * math.pi * h / 24),
                math.cos(2 * math.pi * h / 24),
            ], dtype=np.float32)

            prev_text = prev_action if prev_action else "unknown"
            prev_vec  = self.sbert.encode(
                prev_text,
                normalize_embeddings=True).astype(np.float32)

            x = float(user_pos.get("x", 0)) / 10.0 if user_pos else 0.0
            z = float(user_pos.get("z", 0)) / 10.0 if user_pos else 0.0
            pos_vec = np.array([x, z], dtype=np.float32)

            items_text = " ".join(detected_items[:5]) \
                         if detected_items else "nothing"
            items_vec = self.sbert.encode(
                items_text,
                normalize_embeddings=True).astype(np.float32)

            conf_map = {
                "high": 1.0, "medium": 0.6,
                "low":  0.3, "unknown": 0.1,
            }
            conf_val = conf_map.get(str(confidence).lower(), 0.1)
            conf_vec = np.array([conf_val, 0.0], dtype=np.float32)

            feature = np.concatenate([
                action_vec, time_vec, prev_vec,
                pos_vec, items_vec, conf_vec,
            ])
            return feature

        except Exception as e:
            print(f"[Manifold] build_feature_vector error: {e}")
            return np.zeros(1158, dtype=np.float32)

    def record_point(self, user_id, feature_vec, action,
                     bound_label, virtual_hour=None,
                     prev_action=None, confidence="unknown",
                     is_shadow=False):
        try:
            doc = {
                "user_id":        user_id,
                "action":         action,
                "prev_action":    prev_action,
                "bound_label":    bound_label,
                "feature_vec":    feature_vec.tolist(),
                "virtual_hour":   virtual_hour,
                "confidence":     confidence,
                "is_shadow":      is_shadow,
                "manifold_xy":    None,
                "cluster_id":     -1,
                "service_result": "pending",
                "voice_trigger":  False,
                "timestamp":      datetime.datetime.utcnow(),
            }
            result = self.db.manifold_points.insert_one(doc)
            self._point_count += 1
            tag = "shadow" if is_shadow else "vlm"
            print(f"   📍 [Manifold] {tag} point #{self._point_count} | "
                  f"{user_id} {action} h={virtual_hour}")
            return str(result.inserted_id)

        except Exception as e:
            print(f"[Manifold] record_point error: {e}")
            return ""

    def maybe_refit(self, user_id):
        if self._point_count % REFIT_EVERY == 0 \
                and self._point_count > 0:
            t = threading.Thread(
                target=self._refit_manifold_all, daemon=True)
            t.start()
            print(f"   🔄 [Manifold] async refit triggered "
                  f"(total={self._point_count})")

    def predict_intent(self, user_id, current_feature):
        empty = {
            "trigger": False, "intent": "unknown",
            "cluster_id": -1, "confidence": 0.0,
            "current_xy": [0, 0], "predicted_xy": [0, 0],
        }

        if not _MANIFOLD_OK or not self._fitted:
            return empty

        try:
            with self._lock:
                scaled = self._scaler.transform([current_feature])
                xy     = self._umap_model.transform(scaled)[0]

            current_xy = xy.tolist()

            recent = list(self.db.manifold_points.find(
                {
                    "user_id":    user_id,
                    "manifold_xy": {"$ne": None},
                },
                {"manifold_xy": 1}
            ).sort("timestamp", -1).limit(TRAJECTORY_WINDOW))

            if len(recent) >= 2:
                pts          = np.array(
                    [r["manifold_xy"] for r in reversed(recent)])
                velocity     = pts[-1] - pts[0]
                predicted_xy = (xy + velocity).tolist()
            else:
                predicted_xy = current_xy

            clusters = list(self.db.behavior_clusters.find(
                {"success_rate": {"$gte": 0.3}}))

            if not clusters:
                return {
                    **empty,
                    "current_xy":   current_xy,
                    "predicted_xy": predicted_xy,
                }

            pred_arr     = np.array(predicted_xy)
            best_cluster = None
            best_dist    = float("inf")

            for c in clusters:
                center = np.array(c["center_xy"])
                dist   = float(np.linalg.norm(pred_arr - center))
                if dist < best_dist:
                    best_dist    = dist
                    best_cluster = c

            confidence_score = max(
                0.0, 1.0 - best_dist / TRIGGER_THRESHOLD)
            trigger = confidence_score >= MIN_CONFIDENCE

            print(f"   🧭 [Manifold] intent={best_cluster['dominant_action']} "
                  f"conf={confidence_score:.2f} trigger={trigger}")

            return {
                "trigger":      trigger,
                "intent":       best_cluster["dominant_action"],
                "cluster_id":   best_cluster.get("cluster_id", -1),
                "confidence":   round(confidence_score, 3),
                "current_xy":   current_xy,
                "predicted_xy": predicted_xy,
            }

        except Exception as e:
            print(f"[Manifold] predict_intent error: {e}")
            return empty

    def update_service_result(self, point_id, result):
        try:
            self.db.manifold_points.update_one(
                {"_id": ObjectId(point_id)},
                {"$set": {"service_result": result}}
            )
            point = self.db.manifold_points.find_one(
                {"_id": ObjectId(point_id)})
            if point:
                intent = point.get("action", "unknown")
                user   = point.get("user_id", "unknown")
                self.db.intent_stats.update_one(
                    {"user_id": user, "intent": intent},
                    {"$inc": {result: 1}},
                    upsert=True,
                )
            print(f"   ✅ [Manifold] point {point_id} → {result}")
        except Exception as e:
            print(f"[Manifold] update_service_result error: {e}")

    def _refit_manifold_all(self):
        if not _MANIFOLD_OK:
            return
        try:
            docs = list(self.db.manifold_points.find(
                {},
                {"_id": 1, "feature_vec": 1,
                 "action": 1, "user_id": 1,
                 "confidence": 1, "is_shadow": 1},
            ))
            if len(docs) < MIN_POINTS_REFIT:
                print(f"   ⚠️  [Manifold] refit skipped: "
                      f"only {len(docs)} pts < {MIN_POINTS_REFIT}")
                return

            print(f"   🔄 [Manifold] refitting {len(docs)} points...")
            t0 = datetime.datetime.now()

            X        = np.array(
                [d["feature_vec"] for d in docs], dtype=np.float32)
            ids      = [d["_id"] for d in docs]
            scaler   = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=min(15, len(docs) - 1),
                min_dist=0.1,
                metric="cosine",
                random_state=42,
            )
            xy_all = reducer.fit_transform(X_scaled)

            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=max(5, len(docs) // 15),
                prediction_data=True,
            )
            labels = clusterer.fit_predict(xy_all)

            from pymongo import UpdateOne
            bulk = []
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

            unique_labels = set(labels) - {-1}
            self.db.behavior_clusters.delete_many({})

            for cid in unique_labels:
                mask     = labels == cid
                pts      = xy_all[mask]
                actions  = [
                    docs[i]["action"]
                    for i in range(len(docs)) if labels[i] == cid
                ]
                dominant = max(set(actions), key=actions.count)

                point_ids = [
                    ids[i] for i in range(len(docs))
                    if labels[i] == cid
                ]
                results = list(self.db.manifold_points.find(
                    {"_id": {"$in": point_ids}},
                    {"service_result": 1},
                ))
                total    = len([
                    r for r in results
                    if r["service_result"] != "pending"
                ])
                accepted = len([
                    r for r in results
                    if r["service_result"] == "accepted"
                ])
                rate = accepted / total if total > 0 else 0.5

                self.db.behavior_clusters.insert_one({
                    "cluster_id":      int(cid),
                    "dominant_action": dominant,
                    "center_xy":       pts.mean(axis=0).tolist(),
                    "size":            int(mask.sum()),
                    "success_rate":    round(rate, 3),
                    "updated_at":      datetime.datetime.utcnow(),
                })

            with self._lock:
                self._scaler        = scaler
                self._umap_model    = reducer
                self._hdbscan_model = clusterer
                self._fitted        = True

            elapsed = (datetime.datetime.now() - t0).total_seconds()
            print(f"   ✅ [Manifold] refit done: "
                  f"{len(unique_labels)} clusters | {elapsed:.1f}s")

        except Exception as e:
            import traceback
            print(f"[Manifold] refit error: {e}\n"
                  f"{traceback.format_exc()}")