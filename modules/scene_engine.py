import json
import re
import math
import time
import threading
import datetime
import requests

import numpy as np
from pymongo import MongoClient


AFFORDANCE_DESCRIPTIONS = {
    "Eating":       "dining table or chair where person eats food with fork spoon or chopsticks",
    "Drinking":     "sink or table where person drinks water juice or beverage from cup or bottle",
    "SittingDrink": "sofa or couch where person sits relaxing and drinks coffee tea or beverage",
    "Cooking":      "stove or oven where person cooks food using pan pot or spatula",
    "Opening":      "refrigerator fridge or cabinet door that person opens to get items",
    "Laying":       "bed or sofa where person lies down to sleep or rest horizontally",
    "Watching":     "television sofa or couch where person watches screen or tv",
    "Reading":      "bed desk or sofa where person reads book magazine or document",
    "Cleaning":     "floor sink or surface where person cleans using broom mop or cloth",
    "PhoneUse":     "sofa bed or chair where person uses smartphone or makes phone call",
    "Typing":       "desk keyboard or monitor where person types on computer or laptop",
    "PickingUp":    "floor or table where person bends down to pick up an object",
    "PuttingDown":  "table shelf or surface where person sets down or places an object",
    "Standing":     "open area hallway or room where person stands still doing nothing",
    "Walking":      "hallway corridor or open floor where person walks or moves around",
}


class SceneEngine:

    def __init__(self, db, ollama_url: str, sbert_model,
                 ontology: dict, system_cfg: dict, behavior_labels: list):
        self.db           = db
        self.ollama_url   = ollama_url
        self.sbert        = sbert_model
        self.col_scene    = db.scene_snapshots
        self.col_affinity = db.affinity_matrix
        self.col_hist     = db.affinity_history
        self.col_user_aff = db.user_spatial_affinity

        self.behavior_labels = behavior_labels

        self._exclusive_behaviours = set(
            ontology.get("exclusive_behaviours",
                ["Laying", "Cooking", "Opening", "Typing", "Cleaning", "Watching"]))
        self._static_fixtures = set(
            ontology.get("static_fixtures",
                ["sofa", "couch", "bed", "refrigerator", "fridge", "toilet",
                 "tv", "television", "monitor"]))

        hp = system_cfg.get("hyperparameters", {})
        self._delta_threshold  = hp.get("delta_threshold",  0.30)
        self._cross_room_gamma = hp.get("cross_room_gamma", 10.0)
        self._base_mass_ch12   = hp.get("base_mass_ch12",   1.0)
        self._base_mass_ch3    = hp.get("base_mass_ch3",    1.2)
        self._base_mass_weak   = hp.get("base_mass_weak",   0.5)
        self._max_zone_search  = hp.get("max_zone_search",  5.0)
        self._retry_interval   = hp.get("scene_retry_interval", 5.0)
        self._retry_max        = hp.get("scene_retry_max",   60)

        self.zone_graph         = []
        self._affinity_matrix   = {}
        self._ready             = False
        self._lock              = threading.Lock()

        self._proto_vecs        = None
        self._proto_labels      = None
        self._affordance_vecs   = None
        self._affordance_labels = None
        self._build_proto_vecs()

        self._load_affinity_matrix()
        self._discover_zones()

        if not self._ready:
            threading.Thread(
                target=self._retry_loop, daemon=True).start()

    def is_ready(self) -> bool:
        return self._ready

    def build(self):
        self._distill_affinity_matrix()
        self._discover_zones()

    def find_nearest_zone(self, user_pos: dict, room_name: str = "") -> dict | None:
        return self._find_nearest_zone(user_pos, room_name)

    def get_zone_affinity(self, zone: dict, behavior: str) -> float:
        return self._compute_zone_affinity(zone, behavior)

    def is_ambiguous(self, zone: dict) -> bool:
        return self._is_ambiguous_zone(zone)

    def update_user_affinity(self, user_id: str, zone_name: str,
                              action: str, room: str,
                              virtual_day: str = ""):
        self._update_user_affinity(user_id, action, zone_name, room,
                                   virtual_day=virtual_day)

    def status(self) -> dict:
        return {
            "ready":          self._ready,
            "zone_count":     len(self.zone_graph),
            "affinity_count": len(self._affinity_matrix),
            "zones":          [z["zone_name"] for z in self.zone_graph],
        }

    def _build_proto_vecs(self):
        self._proto_labels = self.behavior_labels
        self._proto_vecs   = self.sbert.encode(
            self._proto_labels,
            normalize_embeddings=True).astype("float32")

        aff_labels = []
        aff_texts  = []
        for b in self.behavior_labels:
            desc = AFFORDANCE_DESCRIPTIONS.get(b, b)
            aff_labels.append(b)
            aff_texts.append(desc)

        self._affordance_labels = aff_labels
        self._affordance_vecs   = self.sbert.encode(
            aff_texts,
            normalize_embeddings=True).astype("float32")

        print(f"[SceneEngine] Built affordance vecs for {len(aff_labels)} behaviors")

    def _get_proto_vecs(self):
        return self._proto_vecs

    def _get_affordance_vecs(self):
        return self._affordance_vecs

    def _retry_loop(self):
        print(f"[SceneEngine] Retry loop started ({self._retry_interval}s interval)")
        for attempt in range(self._retry_max):
            time.sleep(self._retry_interval)
            count = self.col_scene.count_documents({})
            if count > 0:
                print(f"[SceneEngine] Retry {attempt+1}: "
                      f"{count} docs found, building...")
                self._distill_affinity_matrix()
                self._discover_zones()
                if self._ready:
                    return
        print("[SceneEngine] Retry exhausted — Zone Graph still empty")

    def _load_affinity_matrix(self):
        docs = list(self.col_affinity.find({}))
        if docs:
            for doc in docs:
                furn   = doc.get("furniture", "").lower().strip()
                action = doc.get("behavior") or doc.get("action", "")
                score  = doc.get("score") or doc.get("affinity", 0.0)
                if not furn or not action:
                    continue
                if furn not in self._affinity_matrix:
                    self._affinity_matrix[furn] = {}
                self._affinity_matrix[furn][action] = float(score)
            print(f"[Affinity] Loaded {len(docs)} entries from MongoDB")
        else:
            print("[Affinity] No affinity_matrix found, building from scratch")
            self._distill_affinity_matrix()

    def _builtin_affinity_fallback(self, furniture_list: list) -> dict:
        BASE = {
            "tv":           {"Watching":0.90, "Laying":0.05, "PhoneUse":0.03, "Reading":0.02},
            "television":   {"Watching":0.90, "Laying":0.05, "PhoneUse":0.03, "Reading":0.02},
            "monitor":      {"Typing":0.90,   "Reading":0.07, "PhoneUse":0.03},
            "keyboard":     {"Typing":0.95,   "Reading":0.05},
            "desk":         {"Typing":0.70,   "Reading":0.20, "PhoneUse":0.10},
            "stove":        {"Cooking":0.95,  "Cleaning":0.03, "Eating":0.02},
            "refrigerator": {"Opening":0.85,  "Cooking":0.10, "Drinking":0.05},
            "fridge":       {"Opening":0.85,  "Cooking":0.10, "Drinking":0.05},
            "cabinet":      {"Opening":0.60,  "Cleaning":0.20, "Cooking":0.10, "Eating":0.10},
            "cabinet2":     {"Opening":0.60,  "Cleaning":0.20, "Cooking":0.10, "Eating":0.10},
            "sofa":         {"Watching":0.90, "Laying":0.30, "Reading":0.15,
                             "SittingDrink":0.10, "PhoneUse":0.05},
            "couch":        {"Watching":0.90, "Laying":0.30, "Reading":0.15,
                             "SittingDrink":0.10, "PhoneUse":0.05},
            "sofa side":    {"Watching":0.70, "Laying":0.30,
                             "SittingDrink":0.20, "PhoneUse":0.10},
            "sofa side 2":  {"Watching":0.70, "Laying":0.30,
                             "SittingDrink":0.20, "PhoneUse":0.10},
            "bed":          {"Laying":0.60,   "Reading":0.30, "PhoneUse":0.10},
            "dad's bed":    {"Laying":0.60,   "Reading":0.30, "PhoneUse":0.10},
            "dining table": {"Eating":0.70,   "SittingDrink":0.20,
                             "Cooking":0.05,  "Drinking":0.05},
            "table":        {"Eating":0.70,   "SittingDrink":0.20,
                             "Cooking":0.05,  "Drinking":0.05},
            "table2":       {"Eating":0.70,   "SittingDrink":0.20,
                             "Cooking":0.05,  "Drinking":0.05},
            "sink":         {"Drinking":0.40, "Cooking":0.35, "Cleaning":0.25},
            "toilet":       {"Cleaning":0.60, "Standing":0.40},
            "chair":        {"Eating":0.55,   "SittingDrink":0.35,
                             "Typing":0.20,   "Reading":0.10},
            "chair1":       {"Eating":0.55,   "SittingDrink":0.35,
                             "Typing":0.20,   "Reading":0.10},
            "chair2":       {"Eating":0.55,   "SittingDrink":0.35,
                             "Typing":0.20,   "Reading":0.10},
            "chair3":       {"Eating":0.55,   "SittingDrink":0.35,
                             "Typing":0.20,   "Reading":0.10},
        }
        result = {}
        for furn in furniture_list:
            key = furn.lower().strip()
            if key in BASE:
                result[key] = BASE[key]
            else:
                result[key] = {
                    "Eating":0.15, "Watching":0.15, "Laying":0.15,
                    "Typing":0.10, "Reading":0.10,  "Cleaning":0.10,
                    "Drinking":0.10, "Cooking":0.10, "SittingDrink":0.05,
                }
        print(f"[Affinity] Cold-start prior loaded: {len(result)} furniture entries")
        return result

    def _distill_affinity_matrix(self):
        all_docs = list(self.col_scene.find({}, {"label": 1}))
        furniture_list = list({
            doc["label"] for doc in all_docs if doc.get("label")
        })
        if not furniture_list:
            print("[Affinity] No furniture in scene_snapshots, using empty matrix")
            return

        behavior_list    = self.behavior_labels
        affordance_vecs  = self._get_affordance_vecs()

        print(f"[Affinity] Computing via SBERT affordance descriptions "
              f"({len(furniture_list)} furniture x {len(behavior_list)} behaviors)...")

        all_matrix = {}

        for furn in furniture_list:
            key      = furn.lower().strip()
            furn_vec = self.sbert.encode(
                [furn], normalize_embeddings=True)[0].astype("float32")
            sims   = affordance_vecs @ furn_vec
            scores = {}
            for i, beh in enumerate(behavior_list):
                scores[beh] = round(float(max(0.0, sims[i])), 3)
            all_matrix[key] = scores

        fallback = self._builtin_affinity_fallback(furniture_list)
        for key, beh_scores in fallback.items():
            if key not in all_matrix:
                all_matrix[key] = beh_scores
            else:
                for beh, score in beh_scores.items():
                    all_matrix[key][beh] = score
        print(f"[Affinity] Merged SBERT affordance + cold-start prior")

        bulk = []
        for furn_key, action_scores in all_matrix.items():
            for action, score in action_scores.items():
                if action in self.behavior_labels:
                    bulk.append({
                        "furniture": furn_key,
                        "behavior":  action,
                        "score":     float(score),
                    })

        if bulk:
            self.col_affinity.delete_many({})
            self.col_affinity.insert_many(bulk)
            aff = {}
            for d in bulk:
                furn = d["furniture"]
                if furn not in aff:
                    aff[furn] = {}
                aff[furn][d["behavior"]] = d["score"]
            self._affinity_matrix = aff
            furn_count = len(set(d["furniture"] for d in bulk))
            print(f"[Affinity] Distilled {len(bulk)} entries "
                  f"({furn_count} furniture x {len(behavior_list)} behaviors)")
            print(f"[Affinity] Sample keys: {list(aff.keys())[:5]}")
        else:
            print("[Affinity] No valid entries")

    def _get_furniture_affinity(self, furniture_label: str, action: str) -> float:
        label = furniture_label.lower().strip()
        return self._affinity_matrix.get(label, {}).get(action, 0.0)

    def _compute_furniture_weight(self, label: str) -> float:
        lbl = label.lower().strip()
        if self._affinity_matrix and lbl in self._affinity_matrix:
            scores = list(self._affinity_matrix[lbl].values())
            if scores:
                sorted_s   = sorted(scores, reverse=True)
                top1       = sorted_s[0]
                top2       = sorted_s[1] if len(sorted_s) > 1 else 0.0
                uniqueness = top1 - top2
                weight     = 1.0 + uniqueness * 10.0
                return max(1.0, round(weight, 2))
        try:
            furn_vec    = self.sbert.encode(
                label, normalize_embeddings=True).astype("float32")
            aff_vecs    = self._get_affordance_vecs()
            sims        = aff_vecs @ furn_vec
            sorted_sims = np.sort(sims)[::-1]
            top1        = float(sorted_sims[0])
            top2        = float(sorted_sims[1]) if len(sorted_sims) > 1 else 0.0
            uniqueness  = top1 - top2
            weight      = 1.0 + uniqueness * 10.0
            return max(1.0, round(weight, 2))
        except Exception:
            return 1.0

    def _discover_zones(self):
        EXCLUSIVE_BEHAVIOURS = self._exclusive_behaviours
        STATIC_FIXTURES      = self._static_fixtures
        DELTA_THRESHOLD      = self._delta_threshold
        CROSS_ROOM_GAMMA     = self._cross_room_gamma
        BASE_MASS_CH12       = self._base_mass_ch12
        BASE_MASS_CH3        = self._base_mass_ch3
        BASE_MASS_WEAK       = self._base_mass_weak

        print("[Zones] Discovering functional zones from scene_snapshots...")
        try:
            all_docs = list(self.col_scene.find(
                {}, {"label": 1, "pos": 1, "room": 1}))
            if not all_docs:
                print("[Zones] No furniture found in scene_snapshots")
                self.zone_graph = []
                return

            affordance_vecs   = self._get_affordance_vecs()
            affordance_labels = self._affordance_labels

            furniture_all = []
            for doc in all_docs:
                label = (doc.get("label") or "").strip()
                pos   = doc.get("pos")
                room  = (doc.get("room") or "Unknown").strip()
                if not label or not isinstance(pos, list) or len(pos) < 2:
                    continue

                vec  = self.sbert.encode(
                    [label], normalize_embeddings=True)[0].astype("float32")
                sims = affordance_vecs @ vec

                sorted_idx = np.argsort(sims)[::-1]
                top1_i     = int(sorted_idx[0])
                top2_i     = int(sorted_idx[1])
                top1_score = float(sims[top1_i])
                top2_score = float(sims[top2_i])
                delta      = top1_score - top2_score
                top1_label = affordance_labels[top1_i]

                lbl_lower = label.lower().strip()
                if lbl_lower in self._affinity_matrix:
                    aff_scores = self._affinity_matrix[lbl_lower]
                    best_beh   = max(aff_scores, key=aff_scores.get)
                    best_score = aff_scores[best_beh]
                    if best_score > top1_score:
                        top1_label = best_beh
                        top1_score = best_score

                furniture_all.append({
                    "label":      label,
                    "pos":        pos,
                    "room":       room,
                    "vec":        vec,
                    "top1_label": top1_label,
                    "top1_score": top1_score,
                    "delta":      delta,
                })

            if not furniture_all:
                self.zone_graph = []
                return

            anchors    = []
            dependents = []

            for f in furniture_all:
                lbl_lower  = f["label"].lower().strip()
                top1_label = f["top1_label"]
                top1_score = f["top1_score"]
                delta      = f["delta"]

                ch1 = delta >= DELTA_THRESHOLD
                ch2 = top1_label in EXCLUSIVE_BEHAVIOURS
                ch3 = any(fix in lbl_lower for fix in STATIC_FIXTURES)

                is_anchor = ch1 or ch2 or ch3

                base = 0.0
                if ch1: base = max(base, BASE_MASS_CH12)
                if ch2: base = max(base, BASE_MASS_CH12)
                if ch3: base = max(base, BASE_MASS_CH3)

                mass = base + top1_score * (1.0 + delta)

                channel_str = "".join([
                    "1" if ch1 else "_",
                    "2" if ch2 else "_",
                    "3" if ch3 else "_",
                ])
                f["is_anchor"] = is_anchor
                f["mass"]      = mass
                f["channel"]   = channel_str

                if is_anchor:
                    anchors.append(f)
                    print(f"  [Anchor|{channel_str}] {f['label']:20} "
                          f"-> {top1_label:15} "
                          f"delta={delta:.2f} mass={mass:.2f}")
                else:
                    dependents.append(f)

            rooms_with_anchor_pre = {a["room"] for a in anchors}
            by_room_dep = {}
            for f in furniture_all:
                if not f.get("is_anchor", False):
                    by_room_dep.setdefault(f["room"], []).append(f)

            for room, flist in by_room_dep.items():
                if room in rooms_with_anchor_pre:
                    continue
                best = max(flist, key=lambda x: x["top1_score"])
                best["is_anchor"] = True
                best["channel"]   = "W"
                best["mass"]      = (BASE_MASS_WEAK
                                     + best["top1_score"]
                                     * (1.0 + best["delta"]))
                anchors.append(best)
                print(f"  [WeakAnchor] {best['label']:20} "
                      f"promoted in room '{room}' "
                      f"mass={best['mass']:.2f}")

            dependents = [
                f for f in furniture_all
                if not f.get("is_anchor", False)
            ]

            if not anchors:
                print("[Zones] No anchors found — zone graph empty")
                self.zone_graph = []
                return

            zone_members = {i: [a] for i, a in enumerate(anchors)}

            for dep in dependents:
                dx, dz    = dep["pos"][0], dep["pos"][1]
                dep_room  = dep["room"]
                best_zone = None
                best_cost = float("inf")

                for i, anchor in enumerate(anchors):
                    ax, az  = anchor["pos"][0], anchor["pos"][1]
                    dist_sq = (dx - ax)**2 + (dz - az)**2
                    gamma   = (1.0 if dep_room == anchor["room"]
                               else CROSS_ROOM_GAMMA)
                    cost    = dist_sq / anchor["mass"] * gamma

                    if cost < best_cost:
                        best_cost = cost
                        best_zone = i

                if best_zone is not None:
                    zone_members[best_zone].append(dep)

            zones         = []
            zone_name_cnt = {}

            for i, anchor in enumerate(anchors):
                members   = zone_members.get(i, [anchor])
                positions = [m["pos"] for m in members]
                labels    = [m["label"] for m in members]
                act_lbl   = anchor["top1_label"]

                weights = np.array([
                    m["mass"] if m.get("is_anchor") else 0.3
                    for m in members
                ], dtype=np.float32)
                cx = float(np.average(
                    [p[0] for p in positions], weights=weights))
                cz = float(np.average(
                    [p[1] for p in positions], weights=weights))
                vecs    = np.stack([m["vec"] for m in members])
                v_space = (weights[:, None] * vecs).sum(axis=0)
                v_norm  = np.linalg.norm(v_space)
                if v_norm > 1e-8:
                    v_space /= v_norm

                base_name = f"{act_lbl}_Zone"
                cnt       = zone_name_cnt.get(base_name, 0)
                zone_name_cnt[base_name] = cnt + 1
                zone_name = base_name if cnt == 0 else f"{base_name}_{cnt+1}"

                zones.append({
                    "room":         anchor["room"],
                    "zone_name":    zone_name,
                    "action_label": act_lbl,
                    "center":       [cx, cz],
                    "v_space":      v_space.tolist(),
                    "furniture":    labels,
                    "anchor":       anchor["label"],
                    "anchor_mass":  round(anchor["mass"], 3),
                })
                print(f"  [Zone] {anchor['room']} | {zone_name} | "
                      f"anchor={anchor['label']} "
                      f"furniture={labels} "
                      f"center=({cx:.1f},{cz:.1f})")

            self._set_ready(zones)
            print(f"[Zones] Built {len(zones)} zones across "
                  f"{len({z['room'] for z in zones})} rooms")

        except Exception as e:
            import traceback
            print(f"[Zones] Error: {e}\n{traceback.format_exc()}")
            self.zone_graph = []
            self._ready = False

    def _compute_zone_affinity(self, zone, behavior):
        if not zone:
            return 0.0
        furnitures = zone.get("furniture", [])

        static_aff = 0.0
        if self._affinity_matrix and furnitures:
            scores  = [
                self._affinity_matrix.get(f.lower().strip(), {}).get(behavior, 0.0)
                for f in furnitures
            ]
            weights = [self._compute_furniture_weight(f) for f in furnitures]
            total_w = sum(weights)
            if total_w > 0:
                static_aff = sum(s * w for s, w in zip(scores, weights)) / total_w

        zone_name    = zone.get("zone_name", "")
        personal_aff = 0.0
        try:
            docs = list(self.col_user_aff.find(
                {"action": behavior, "zone": zone_name},
                {"affinity": 1}
            ))
            if docs:
                personal_aff = max(d.get("affinity", 0.0) for d in docs)
        except Exception:
            pass

        effective_aff = max(static_aff, personal_aff)

        if personal_aff > static_aff and personal_aff > 0:
            print(f"[Affinity] Bayesian override: {zone_name} {behavior} "
                  f"static={static_aff:.2f} personal={personal_aff:.2f}")

        if not zone or "v_space" not in zone:
            return round(effective_aff, 3)

        if behavior not in self._proto_labels:
            return round(effective_aff, 3)

        try:
            idx       = self._proto_labels.index(behavior)
            proto_vec = self._get_proto_vecs()[idx]
            zone_vec  = np.array(zone["v_space"], dtype="float32")
            sbert_aff = float(proto_vec @ zone_vec)
            sbert_aff = max(0.0, min(1.0, sbert_aff))
            final     = 0.6 * effective_aff + 0.4 * sbert_aff
            return round(max(0.0, min(1.0, final)), 3)
        except Exception:
            return round(effective_aff, 3)

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

        MAX_ZONE_SEARCH = getattr(self, "_max_zone_search", 5.0)
        best_zone = None
        best_dist = float("inf")
        for zone in candidates:
            cx, cz = zone["center"][0], zone["center"][1]
            dist   = math.sqrt((ux - cx)**2 + (uz - cz)**2)
            if dist < best_dist:
                best_dist = dist
                best_zone = zone
        if best_dist > MAX_ZONE_SEARCH:
            return None
        return best_zone

    def _is_ambiguous_zone(self, zone) -> bool:
        if not zone or not self._affinity_matrix:
            return False
        scores = []
        for behavior in self.behavior_labels:
            s = self._compute_zone_affinity(zone, behavior)
            scores.append(s)
        if not scores or max(scores) < 0.05:
            return True
        sorted_s = sorted(scores, reverse=True)
        top1 = sorted_s[0]
        top2 = sorted_s[1] if len(sorted_s) > 1 else 0.0
        return (top1 - top2) < 0.25

    def _update_user_affinity(self, user: str, action: str,
                               zone_name: str, instance: str,
                               virtual_day: str = ""):
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
            results = list(self.db.observation_logs.aggregate(pipeline))
            total   = sum(r["total_weight"] for r in results)
            if total == 0:
                return

            today = virtual_day if virtual_day else \
                    datetime.datetime.utcnow().strftime("%Y-%m-%d")

            for r in results:
                zone_key = r["_id"] or "Unknown_Zone"
                personal = r["total_weight"] / total
                self.col_user_aff.update_one(
                    {"user_id": user, "action": action, "zone": zone_key},
                    {"$set": {
                        "affinity":   round(personal, 4),
                        "updated_at": datetime.datetime.utcnow(),
                    }},
                    upsert=True,
                )
                self.col_hist.update_one(
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

    def _set_ready(self, zones):
        with self._lock:
            self.zone_graph = zones
            self._ready     = len(zones) > 0
        if self._ready:
            print(f"[SceneEngine] READY — {len(zones)} zones built")
        return self._ready