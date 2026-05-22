import json
import re
import datetime
import threading
import requests
from collections import defaultdict


BEHAVIOR_LABELS = [
    "Eating", "Drinking", "SittingDrink", "Cooking", "Opening",
    "Laying", "Watching", "Reading", "Cleaning", "PhoneUse", "Typing",
]

_FALLBACK_BEHAVIOR_TO_OBJECTS = {
    "Eating":       ["food", "apple", "banana", "bowl", "plate", "fork", "spoon"],
    "Drinking":     ["cup", "bottle", "juice", "cola", "water", "mug", "glass"],
    "SittingDrink": ["cup", "mug", "tea", "coffee", "bottle"],
    "Cooking":      ["pan", "pot", "stove", "spatula", "oven"],
    "Opening":      ["refrigerator", "fridge"],
    "Watching":     ["tv", "television", "remote"],
    "Typing":       ["laptop", "keyboard", "computer", "monitor"],
    "Reading":      ["book", "magazine"],
    "PhoneUse":     ["phone", "cell phone", "smartphone"],
    "Cleaning":     ["broom", "mop", "dustpan"],
    "Laying":       [],
}

WEIGHT_SAY   = 0.40
WEIGHT_HABIT = 0.35
WEIGHT_ENV   = 0.15
WEIGHT_SKILL = 0.10

MIN_GATE_SCORE = 0.05
ENV_FALLBACK   = 0.30


class SayCanEngine:

    def __init__(self, db, manifold_engine,
                 ollama_url: str, llm_model: str,
                 sbert_model=None):
        self.db              = db
        self.manifold_engine = manifold_engine
        self.ollama_url      = ollama_url
        self.llm_model       = llm_model
        self.sbert           = sbert_model
        self._lock           = threading.Lock()

        self._behavior_objects: dict    = {}
        self._behavior_proto_vecs: dict = {}

        self._distill_behavior_objects()

        if self.sbert is not None:
            self._build_behavior_prototypes()

        print("[SayCan] Ready")

    def locate(self, query: str, user_id: str) -> dict:
        all_objs = list(self.db.dynamic_objects.find(
            {}, {"label": 1, "last_seen_on": 1, "room": 1, "furniture_pos": 1}))

        best_obj = None
        best_sim = 0.0

        if self.sbert is not None and all_objs:
            import numpy as np
            labels   = [d.get("label", "") for d in all_objs]
            vecs     = self.sbert.encode(labels, normalize_embeddings=True)
            q_vec    = self.sbert.encode(query, normalize_embeddings=True)
            sims     = vecs @ q_vec
            best_i   = int(sims.argmax())
            best_sim = float(sims[best_i])
            if best_sim >= 0.35:
                best_obj = all_objs[best_i]
        else:
            q_lower = query.lower()
            for d in all_objs:
                if d.get("label", "").lower() in q_lower:
                    best_obj = d
                    break

        if best_obj:
            label    = best_obj.get("label", "object")
            location = best_obj.get("last_seen_on", "unknown area")
            room     = best_obj.get("room", "")
            nav      = best_obj.get("furniture_pos")
            answer   = (
                f"The {label} was last seen near {location}"
                f"{' in the ' + room if room else ''}."
            )
        else:
            answer   = "I haven't observed that item recently."
            nav      = None
            label    = "unknown"
            location = "unknown"

        return {
            "intent":       "LOCATE",
            "best_action":  "LOCATE",
            "explanation":  answer,
            "nav_target":   nav,
            "nav_label":    location,
            "final_scores": {},
            "say_scores":   {},
            "habit_probs":  {},
        }

    def resolve(self, query: str, user_id: str,
                virtual_hour=None, user_pos=None,
                prev_action: str = "Standing") -> dict:
        say_scores   = self._compute_say(query)
        habit_probs  = self._compute_can_habit(user_id, virtual_hour, user_pos, prev_action)
        env_scores   = self._compute_can_env()
        skill_scores = self._compute_can_skill(user_id)

        eps          = 1e-3
        final_scores = {}
        for b in BEHAVIOR_LABELS:
            s = say_scores.get(b,   0.0)
            h = habit_probs.get(b,  0.0)
            e = env_scores.get(b,   ENV_FALLBACK)
            k = skill_scores.get(b, 1.0)
            final_scores[b] = (
                max(eps, s) *
                max(eps, h) *
                max(eps, e) *
                max(eps, k)
            )

        best_action = max(final_scores, key=final_scores.get)
        best_score  = final_scores[best_action]

        nav_target, nav_label = self._resolve_nav(best_action)
        explanation = self._generate_response(
            query, user_id, best_action, nav_label,
            best_score, say_scores, habit_probs)

        self.db.saycan_logs.insert_one({
            "user_id":      user_id,
            "query":        query,
            "best_action":  best_action,
            "best_score":   round(best_score, 4),
            "say_scores":   {k: round(v, 4) for k, v in say_scores.items()},
            "habit_probs":  {k: round(v, 4) for k, v in habit_probs.items()},
            "env_scores":   {k: round(v, 4) for k, v in env_scores.items()},
            "skill_scores": {k: round(v, 4) for k, v in skill_scores.items()},
            "final_scores": {k: round(v, 6) for k, v in final_scores.items()},
            "nav_target":   nav_target,
            "nav_label":    nav_label,
            "virtual_hour": virtual_hour,
            "user_pos":     user_pos,
            "prev_action":  prev_action,
            "timestamp":    datetime.datetime.utcnow(),
        })

        print(f"[SayCan] '{query}' -> {best_action} (score={best_score:.4f})")

        return {
            "intent":       "NEED",
            "best_action":  best_action,
            "best_score":   round(best_score, 4),
            "final_scores": {k: round(v, 6)
                             for k, v in sorted(
                                 final_scores.items(), key=lambda x: -x[1])},
            "say_scores":   say_scores,
            "habit_probs":  habit_probs,
            "env_scores":   env_scores,
            "skill_scores": skill_scores,
            "nav_target":   nav_target,
            "nav_label":    nav_label,
            "explanation":  explanation,
        }

    def _compute_say(self, query: str) -> dict:
        behavior_list = ", ".join(BEHAVIOR_LABELS)
        prompt = (
            f"You are a home robot intent classifier.\n"
            f"User said: \"{query}\"\n\n"
            f"Rate how relevant each behaviour is (0.0 = irrelevant, 1.0 = perfect match).\n"
            f"Behaviours: {behavior_list}\n\n"
            f"Output ONLY valid JSON. No explanation.\n"
            f"Example: {{\"Eating\": 0.9, \"Drinking\": 0.3}}"
        )
        try:
            resp = requests.post(
                f"{self.ollama_url}/api/chat",
                json={
                    "model":    self.llm_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream":   False,
                    "options":  {"temperature": 0.1, "num_predict": 300},
                },
                timeout=30,
            )
            raw = resp.json().get("message", {}).get("content", "")
            m   = re.search(r'\{.*\}', raw, re.DOTALL)
            if m:
                scores = json.loads(m.group(0))
                result = {}
                for b in BEHAVIOR_LABELS:
                    v = float(scores.get(b, 0.0))
                    result[b] = max(0.0, min(1.0, v))
                return result
        except Exception as e:
            print(f"[SayCan-Say] LLM failed: {e}")

        return {b: 1.0 / len(BEHAVIOR_LABELS) for b in BEHAVIOR_LABELS}

    def _compute_can_habit(self, user_id, virtual_hour, user_pos, prev_action) -> dict:
        if self.manifold_engine is None:
            return {b: 1.0 / len(BEHAVIOR_LABELS) for b in BEHAVIOR_LABELS}
        try:
            result = self.manifold_engine.predict_intent(
                user_id      = user_id,
                virtual_hour = virtual_hour,
                user_pos     = user_pos,
                prev_action  = prev_action or "Standing",
            )
            probs = result.get("probs", {})
            out   = {b: float(probs.get(b, 0.0)) for b in BEHAVIOR_LABELS}
            total = sum(out.values()) or 1.0
            return {b: v / total for b, v in out.items()}
        except Exception as e:
            print(f"[SayCan-Habit] failed: {e}")
            return {b: 1.0 / len(BEHAVIOR_LABELS) for b in BEHAVIOR_LABELS}

    def _compute_can_env(self) -> dict:
        import numpy as np

        try:
            dyn_docs   = list(self.db.dynamic_objects.find({}, {"label": 1}))
            dyn_labels = [d["label"].lower() for d in dyn_docs if d.get("label")]
        except Exception:
            dyn_labels = []

        if not dyn_labels:
            return {b: ENV_FALLBACK for b in BEHAVIOR_LABELS}

        if self._behavior_proto_vecs and self.sbert is not None:
            obj_vecs = self.sbert.encode(dyn_labels, normalize_embeddings=True)
            result   = {}
            for b in BEHAVIOR_LABELS:
                proto = self._behavior_proto_vecs.get(b)
                if proto is None:
                    result[b] = ENV_FALLBACK
                    continue
                sims    = obj_vecs @ proto
                max_sim = float(sims.max())
                if max_sim >= 0.60:
                    result[b] = 1.0
                elif max_sim >= 0.40:
                    result[b] = 0.60
                else:
                    result[b] = ENV_FALLBACK
            return result

        dyn_set = set(dyn_labels)
        obj_map = self._get_behavior_objects()
        result  = {}
        for b in BEHAVIOR_LABELS:
            needed = obj_map.get(b, [])
            if not needed:
                result[b] = 0.5
                continue
            found      = any(any(kw in lbl for lbl in dyn_set) for kw in needed)
            result[b]  = 1.0 if found else ENV_FALLBACK
        return result

    def _compute_can_skill(self, user_id: str) -> dict:
        try:
            doc      = self.db.user_skills.find_one({"user_id": user_id})
            skill_md = doc.get("skill_md", "") if doc else ""
        except Exception:
            skill_md = ""

        rejected   = set()
        in_section = False
        for line in skill_md.split("\n"):
            clean = line.strip()
            if not clean:
                continue
            if "what not to do" in clean.lower():
                in_section = True
                continue
            if in_section:
                if clean.startswith("#") and not clean.startswith("###"):
                    break
                if clean.startswith("-"):
                    rejected.add(clean.lower())

        obj_map = self._get_behavior_objects()
        result  = {}
        for b in BEHAVIOR_LABELS:
            needed     = obj_map.get(b, [])
            if not needed:
                result[b] = 1.0
                continue
            penalised  = any(any(item in rej for rej in rejected) for item in needed)
            result[b]  = 0.1 if penalised else 1.0
        return result

    def _resolve_nav(self, action: str):
        obj_map = self._get_behavior_objects()
        needed  = obj_map.get(action, [])

        for kw in needed:
            doc = self.db.dynamic_objects.find_one(
                {"label": {"$regex": kw, "$options": "i"}},
                {"furniture_pos": 1, "last_seen_on": 1},
            )
            if doc and doc.get("furniture_pos"):
                return doc["furniture_pos"], doc.get("last_seen_on", kw)

        zone_doc = self.db.observation_logs.find_one(
            {"action": action}, sort=[("weight", -1)])
        if zone_doc:
            return zone_doc.get("pos"), zone_doc.get("zone_name", action)

        return None, action

    def _generate_response(self, query: str, user_id: str,
                            best_action: str, nav_label: str,
                            best_score: float,
                            say_scores: dict, habit_probs: dict) -> str:
        prompt = (
            f"You are a friendly home robot.\n"
            f"User ({user_id}) said: \"{query}\"\n"
            f"System decided: {best_action} near {nav_label} "
            f"(confidence={best_score:.2f})\n\n"
            f"Generate a SHORT natural English response (1-2 sentences).\n"
            f"Output ONLY the response, no explanation."
        )
        try:
            resp = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model":   self.llm_model,
                    "prompt":  prompt,
                    "stream":  False,
                    "options": {"temperature": 0.3, "num_predict": 80},
                },
                timeout=20,
            )
            msg = resp.json().get("response", "").strip()
            return msg.split("\n")[0].strip() if msg else ""
        except Exception as e:
            print(f"[SayCan-Response] LLM failed: {e}")
            loc = f" near {nav_label}" if nav_label else ""
            return f"I think you want to {best_action.lower()}{loc}. Shall I help?"

    def _build_behavior_prototypes(self):
        import numpy as np
        obj_map = self._get_behavior_objects()
        for behavior, objects in obj_map.items():
            if objects:
                vecs  = self.sbert.encode(objects, normalize_embeddings=True)
                proto = vecs.mean(axis=0)
            else:
                proto = self.sbert.encode(behavior, normalize_embeddings=True)
            norm = float(np.linalg.norm(proto))
            self._behavior_proto_vecs[behavior] = (proto / (norm + 1e-8)).astype("float32")
        print(f"[SayCan] Behavior prototypes built ({len(self._behavior_proto_vecs)} behaviours)")

    def _distill_behavior_objects(self):
        cached = list(self.db.saycan_behavior_objects.find({}))
        if cached:
            for doc in cached:
                b = doc.get("behavior")
                o = doc.get("objects", [])
                if b:
                    self._behavior_objects[b] = o
            print(f"[SayCan] Loaded behavior-object map ({len(self._behavior_objects)} entries)")
            return

        print("[SayCan] Distilling behavior-object map via LLM...")

        scene_objects = list(self.db.dynamic_objects.distinct("label"))
        object_vocab  = list(self.db.scene_snapshots.distinct("label"))
        all_objects   = list(dict.fromkeys(
            [o.lower().strip() for o in scene_objects + object_vocab if o and len(o) > 1]
        ))
        object_list_str = ", ".join(all_objects) if all_objects else \
                          "remote, book, cup, broom, phone, laptop, fork, pan"
        behavior_list   = ", ".join(BEHAVIOR_LABELS)

        prompt = (
            f"You are an AI robotics ontologist.\n"
            f"For each action, list the most relevant physical objects required.\n\n"
            f"Allowed Actions: [{behavior_list}]\n"
            f"STRICT CONSTRAINT: Only use objects from this list: [{object_list_str}]\n\n"
            f"Output ONLY valid JSON. No explanations.\n"
            f"Example: {{\"Eating\": [\"bowl\", \"fork\"], \"Watching\": [\"remote\"]}}"
        )
        try:
            resp = requests.post(
                f"{self.ollama_url}/api/chat",
                json={
                    "model":    self.llm_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream":   False,
                    "options":  {"temperature": 0.1, "num_predict": 600},
                },
                timeout=90,
            )
            raw = resp.json().get("message", {}).get("content", "")
            m   = re.search(r'\{.*\}', raw, re.DOTALL)
            if m:
                mapping = json.loads(m.group(0))
                bulk    = []
                for b in BEHAVIOR_LABELS:
                    objs = [o.lower().strip() for o in mapping.get(b, [])
                            if isinstance(o, str)]
                    self._behavior_objects[b] = objs
                    bulk.append({
                        "behavior":  b,
                        "objects":   objs,
                        "source":    self.llm_model,
                        "timestamp": datetime.datetime.utcnow(),
                    })
                if bulk:
                    self.db.saycan_behavior_objects.delete_many({})
                    self.db.saycan_behavior_objects.insert_many(bulk)
                print(f"[SayCan] Distilled {len(bulk)} behavior-object entries")
                return
        except Exception as e:
            print(f"[SayCan] Distillation failed: {e}")

        self._behavior_objects = dict(_FALLBACK_BEHAVIOR_TO_OBJECTS)
        print("[SayCan] Using fallback behavior-object map")

    def _get_behavior_objects(self) -> dict:
        with self._lock:
            if self._behavior_objects:
                return dict(self._behavior_objects)
        return dict(_FALLBACK_BEHAVIOR_TO_OBJECTS)
