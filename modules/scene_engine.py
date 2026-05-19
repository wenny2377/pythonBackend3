"""
scene_engine.py
SceneEngine — Scene Understanding Module.

Responsibilities:
  - Zone Graph construction (3-stage semantic gravity)
  - Gemma-2 affinity matrix distillation
  - Nearest zone lookup (read-only after build)
  - User spatial affinity tracking

Dependencies: MongoDB, SBERT, Ollama (Gemma)
Injected into: PerceptionEngine, SayCanEngine
"""

import re
import math
import time
import threading
import datetime
import requests

import numpy as np
from pymongo import MongoClient


class SceneEngine:
    """
    Owns and manages the Zone Graph.
    Must call build() (or wait for auto-retry) before is_ready() returns True.
    Thread-safe: zone_graph is rebuilt atomically.
    """

    def __init__(self, db, ollama_url: str, sbert_model,
                 ontology: dict, system_cfg: dict, behavior_labels: list):
        self.db          = db
        self.ollama_url  = ollama_url
        self.sbert       = sbert_model
        self.col_scene   = db.scene_snapshots
        self.col_affinity = db.affinity_matrix
        self.col_hist    = db.affinity_history
        self.col_user_aff = db.user_spatial_affinity

        # Behavior labels for affinity distillation
        self.behavior_labels = behavior_labels

        # Ontology config
        self._exclusive_behaviours = set(
            ontology.get("exclusive_behaviours",
                ["Laying","Cooking","Opening","Typing","Cleaning"]))
        self._static_fixtures = set(
            ontology.get("static_fixtures",
                ["sofa","couch","bed","refrigerator","fridge","toilet"]))

        # System hyperparameters
        hp = system_cfg.get("hyperparameters", {})
        self._delta_threshold  = hp.get("delta_threshold",  0.30)
        self._cross_room_gamma = hp.get("cross_room_gamma", 10.0)
        self._base_mass_ch12   = hp.get("base_mass_ch12",   1.0)
        self._base_mass_ch3    = hp.get("base_mass_ch3",    1.2)
        self._base_mass_weak   = hp.get("base_mass_weak",   0.5)
        self._max_zone_search  = hp.get("max_zone_search",  5.0)
        self._retry_interval   = hp.get("scene_retry_interval", 5.0)
        self._retry_max        = hp.get("scene_retry_max",   60)

        # State
        self.zone_graph       = []
        self._affinity_matrix = {}
        self._ready           = False
        self._lock            = threading.Lock()

        # Proto vecs for SBERT zone labeling
        self._proto_vecs   = None
        self._proto_labels = None
        self._build_proto_vecs()

        # Try to load existing affinity and build zones
        self._load_affinity_matrix()
        self._discover_zones()

        # Start background retry if not ready
        if not self._ready:
            threading.Thread(
                target=self._retry_loop, daemon=True).start()

    # ── Public API ────────────────────────────────────────────────────

    def is_ready(self) -> bool:
        return self._ready

    def build(self):
        """Called by /scene endpoint after new furniture data arrives."""
        if not self._affinity_matrix:
            self._distill_affinity_matrix()
        self._discover_zones()

    def find_nearest_zone(self, user_pos: dict, room_name: str = "") -> dict | None:
        return self._find_nearest_zone(user_pos, room_name)

    def get_zone_affinity(self, zone: dict, behavior: str) -> float:
        return self._compute_zone_affinity(zone, behavior)

    def is_ambiguous(self, zone: dict) -> bool:
        return self._is_ambiguous_zone(zone)

    def update_user_affinity(self, user_id: str, zone_name: str,
                              action: str, room: str):
        self._update_user_affinity(user_id, zone_name, action, room)

    def status(self) -> dict:
        return {
            "ready":          self._ready,
            "zone_count":     len(self.zone_graph),
            "affinity_count": len(self._affinity_matrix),
            "zones":          [z["zone_name"] for z in self.zone_graph],
        }

    # ── Internal: proto vecs ──────────────────────────────────────────

    def _build_proto_vecs(self):
        self._proto_labels = self.behavior_labels
        self._proto_vecs   = self.sbert.encode(
            self._proto_labels,
            normalize_embeddings=True).astype("float32")

    def _get_proto_vecs(self):
        return self._proto_vecs

    # ── Internal: retry loop ──────────────────────────────────────────

    def _retry_loop(self):
        print(f"[SceneEngine] Retry loop started ({self._retry_interval}s interval)")
        for attempt in range(self._retry_max):
            time.sleep(self._retry_interval)
            count = self.col_scene.count_documents({})
            if count > 0:
                print(f"[SceneEngine] Retry {attempt+1}: "
                      f"{count} docs found, building...")
                if not self._affinity_matrix:
                    self._distill_affinity_matrix()
                self._discover_zones()
                if self._ready:
                    return
        print("[SceneEngine] Retry exhausted — Zone Graph still empty")


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

        # ── Dynamic prompt — no hardcoded scores ─────────────────
        # Gemma's commonsense is sufficient; we only provide the
        # furniture list and behavior list. This makes the system
        # scene-agnostic: a new scene only needs a new furniture list.
        prompt = (
            "You are a spatial behavior expert for home service robots. "
            "Your task: build a semantic affordance matrix. "
            "For each furniture item, rate its affinity (0.00 to 1.00) "
            "with each behavior. "
            "A high score means the behavior commonly occurs near "
            "or inside the spatial boundary of that furniture. "
            "Apply realistic spatial logic based on common sense "
            "(e.g., sleeping happens on beds, cooking near stoves, "
            "watching near televisions). "
            "Each furniture's scores should sum to approximately 1.0. "
            f"Behaviors: {behavior_list}. "
            f"Furniture items: {furniture_list}. "
            "Output ONLY valid JSON, no markdown, no explanation. "
            "Format: { furniture_name: { behavior_name: score } }"
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
        """
        Three-stage semantic gravity zone discovery.

        Stage 1 — Anchor classification (triple-channel filter):
          Ch1: Semantic sharpness  Δ >= 0.30  (data-driven)
          Ch2: Exclusive behaviour whitelist   (prior knowledge)
          Ch3: Static fixture whitelist        (physical prior)
          Mass = Base_Mass + top1_score × (1 + Δ)
          Base_Mass: Ch1/Ch2 = 1.0, Ch3 = 1.2, weak fallback = 0.5

        Stage 2 — Seed zone creation:
          Each Anchor declares its own Zone (semantics locked).
          Rooms with no Anchor → promote highest-scoring furniture
          as weak Anchor (Base_Mass = 0.5).

        Stage 3 — Semantic gravity field propagation:
          Each Dependent finds the Anchor with minimum semantic cost.
          Cost = dist² / Mass × γ
          γ = 1.0 same room, γ = 10.0 cross-room.
        """

        # ── Constants (from YAML config or built-in defaults) ───────
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
                {}, {"label": 1, "pos": 1, "room": 1}
            ))
            if not all_docs:
                print("[Zones] No furniture found in scene_snapshots")
                self.zone_graph = []
                return

            proto_vecs   = self._get_proto_vecs()
            proto_labels = self._proto_labels

            # ── Step 0: compute affinity for every furniture ──────────
            furniture_all = []
            for doc in all_docs:
                label = (doc.get("label") or "").strip()
                pos   = doc.get("pos")
                room  = (doc.get("room") or "Unknown").strip()
                if not label or not isinstance(pos, list) or len(pos) < 2:
                    continue

                vec  = self.sbert.encode(
                    [label], normalize_embeddings=True
                )[0].astype("float32")
                sims = proto_vecs @ vec

                sorted_idx = np.argsort(sims)[::-1]
                top1_i     = int(sorted_idx[0])
                top2_i     = int(sorted_idx[1])
                top1_score = float(sims[top1_i])
                top2_score = float(sims[top2_i])
                delta      = top1_score - top2_score
                top1_label = proto_labels[top1_i]

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

            # ── Stage 1: classify Anchor vs Dependent ─────────────────
            anchors    = []
            dependents = []

            for f in furniture_all:
                lbl_lower  = f["label"].lower().strip()
                top1_label = f["top1_label"]
                top1_score = f["top1_score"]
                delta      = f["delta"]

                # Channel 1: semantic sharpness
                ch1 = delta >= DELTA_THRESHOLD

                # Channel 2: exclusive behaviour whitelist
                ch2 = top1_label in EXCLUSIVE_BEHAVIOURS

                # Channel 3: static fixture whitelist
                ch3 = any(fix in lbl_lower for fix in STATIC_FIXTURES)

                is_anchor = ch1 or ch2 or ch3

                # Base mass: take maximum across triggered channels
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
                f["is_anchor"]  = is_anchor
                f["mass"]       = mass
                f["channel"]    = channel_str

                if is_anchor:
                    anchors.append(f)
                    print(f"  [Anchor|{channel_str}] {f['label']:20} "
                          f"→ {top1_label:15} "
                          f"Δ={delta:.2f} mass={mass:.2f}")
                else:
                    dependents.append(f)

            # ── Stage 2: per-room weak Anchor fallback ────────────────
            rooms_with_anchor = {a["room"] for a in anchors}
            # Stage 2: weak Anchor fallback for rooms with no Anchor.
            # Rebuild dependents AFTER weak Anchor promotion to avoid
            # remove() timing issues and keep the logic clean.
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

            # Final dependents: all non-anchor furniture after promotions
            dependents = [
                f for f in furniture_all
                if not f.get("is_anchor", False)
            ]

            if not anchors:
                print("[Zones] No anchors found — zone graph empty")
                self.zone_graph = []
                return

            # ── Stage 3: gravity field propagation ────────────────────
            # Each Anchor seeds its own Zone
            zone_members = {i: [a] for i, a in enumerate(anchors)}

            for dep in dependents:
                dx, dz    = dep["pos"][0], dep["pos"][1]
                dep_room  = dep["room"]
                best_zone = None
                best_cost = float("inf")

                for i, anchor in enumerate(anchors):
                    ax, az     = anchor["pos"][0], anchor["pos"][1]
                    dist_sq    = (dx - ax)**2 + (dz - az)**2
                    gamma      = (1.0 if dep_room == anchor["room"]
                                  else CROSS_ROOM_GAMMA)
                    cost       = dist_sq / anchor["mass"] * gamma

                    if cost < best_cost:
                        best_cost = cost
                        best_zone = i

                if best_zone is not None:
                    zone_members[best_zone].append(dep)

            # ── Build zone_graph entries ───────────────────────────────
            zones          = []
            zone_name_cnt  = {}

            for i, anchor in enumerate(anchors):
                members   = zone_members.get(i, [anchor])
                positions = [m["pos"] for m in members]
                labels    = [m["label"] for m in members]
                act_lbl   = anchor["top1_label"]

                # Weighted geometric centroid — Anchor mass dominates.
                # Using np.mean() would allow chairs to drag the zone
                # center away from the Anchor (semantic centroid drift).
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

        # Max search radius 5.0m to handle real home layouts
        # (e.g. tv at (6.59,-0.67) and sofa at (2.35,-1.41) are 4.3m apart)
        MAX_ZONE_SEARCH = getattr(self, "_max_zone_search", 5.0)
        best_zone = None
        best_dist = float("inf")
        for zone in candidates:
            cx, cz = zone["center"][0], zone["center"][1]
            dist   = math.sqrt((ux - cx)**2 + (uz - cz)**2)
            if dist < best_dist:
                best_dist = dist
                best_zone = zone
        # Return None if too far (no meaningful zone found)
        if best_dist > MAX_ZONE_SEARCH:
            return None
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

    def _set_ready(self, zones):
        """Called after successful zone build."""
        with self._lock:
            self.zone_graph = zones
            self._ready     = len(zones) > 0
        if self._ready:
            print(f"[SceneEngine] READY — {len(zones)} zones built")
        return self._ready