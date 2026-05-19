"""
saycan_engine.py
Value-Mapped SayCan Gate for Home Robot Intent Resolution

Architecture:
  Say  = LLM semantic scoring over BEHAVIOR_LABELS
  Can  = ManifoldEngine habit priors × dynamic_objects feasibility
         × SKILL.md preference filter
  Gate = Say × Can_habit × Can_env × Can_skill (pure product)

References:
  SayCan (Ahn et al., Google DeepMind 2022)
  Inner Monologue (Huang et al., 2022)

Usage:
  engine = SayCanEngine(db, manifold_engine, ollama_url, llm_model)
  result = engine.resolve(query, user_id, virtual_hour, user_pos, prev_action)
"""

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

# ── Intent classification ─────────────────────────────────────────────
# Three classes: CHAT | LOCATE | NEED
# Stage 1: keyword rules (milliseconds)
# Stage 2: SBERT cosine similarity (only for ambiguous queries)

LOCATE_KEYWORDS = [
    "where", "which room", "find", "location", "show me",
    "在哪", "哪裡", "找", "位置", "在哪裡",
]
NEED_KEYWORDS = [
    "hungry", "thirsty", "tired", "want", "need", "feel like",
    "bored", "cold", "hot", "rest", "craving", "fancy",
    "餓", "渴", "累", "想", "需要", "不舒服", "想要",
]

LOCATE_PROTO = "where is the item object location find room"
NEED_PROTO   = "I need want something feel hungry tired thirsty rest"
CHAT_PROTO   = "hello how are you chat talk nice weather fun"


class IntentClassifier:
    """
    Lightweight three-class intent classifier.
    Stage 1: keyword rules → instant.
    Stage 2: SBERT cosine similarity → only for ambiguous queries.
    """

    CLASSES = ("CHAT", "LOCATE", "NEED")

    def __init__(self, sbert_model=None):
        self._sbert    = sbert_model
        self._protos   = {}
        if sbert_model is not None:
            self._build_protos()

    def _build_protos(self):
        import numpy as np
        texts = [CHAT_PROTO, LOCATE_PROTO, NEED_PROTO]
        vecs  = self._sbert.encode(texts, normalize_embeddings=True)
        self._protos = {
            "CHAT":   vecs[0],
            "LOCATE": vecs[1],
            "NEED":   vecs[2],
        }

    def classify(self, query: str) -> str:
        q_lower = query.lower().strip()

        # Stage 1: keyword rules
        if any(kw in q_lower for kw in LOCATE_KEYWORDS):
            return "LOCATE"
        if any(kw in q_lower for kw in NEED_KEYWORDS):
            return "NEED"

        # Stage 2: SBERT (only if protos available)
        if self._protos and self._sbert is not None:
            import numpy as np
            vec  = self._sbert.encode(query, normalize_embeddings=True)
            sims = {cls: float(np.dot(vec, proto))
                    for cls, proto in self._protos.items()}
            best = max(sims, key=sims.get)
            print(f"[IntentClassifier] SBERT sims={sims} → {best}")
            return best

        return "CHAT"

# Fallback mapping if Gemma distillation is unavailable
_FALLBACK_BEHAVIOR_TO_OBJECTS = {
    "Eating":       ["food", "apple", "banana", "bowl", "plate",
                     "fork", "spoon", "sandwich"],
    "Drinking":     ["cup", "bottle", "juice", "cola", "water",
                     "mug", "glass"],
    "SittingDrink": ["cup", "mug", "tea", "coffee", "bottle"],
    "Cooking":      ["pan", "pot", "stove", "spatula", "oven"],
    "Opening":      ["refrigerator", "fridge"],
    "Watching":     ["tv", "television", "remote"],
    "Typing":       ["laptop", "keyboard", "computer", "monitor"],
    "Reading":      ["book", "magazine"],
    "PhoneUse":     ["phone", "cell phone", "smartphone"],
    "Cleaning":     ["broom", "mop", "dustpan"],
    "Laying":       [],   # no object needed
}

# Can weights
WEIGHT_SAY   = 0.40
WEIGHT_HABIT = 0.35
WEIGHT_ENV   = 0.15
WEIGHT_SKILL = 0.10

# Minimum score to trigger a result
MIN_GATE_SCORE = 0.05

# ENV fallback when object not observed (uncertain, not impossible)
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

        # behavior → required objects (distilled or fallback)
        self._behavior_objects: dict  = {}
        # behavior → prototype SBERT vector (open-vocabulary Can_env)
        self._behavior_proto_vecs: dict = {}

        self._distill_behavior_objects()

        # Build SBERT prototype vectors for open-vocabulary Can_env
        if self.sbert is not None:
            self._build_behavior_prototypes()

        # Intent classifier (three-class: CHAT / LOCATE / NEED)
        self.intent_classifier = IntentClassifier(
            sbert_model=self.sbert)

        print("[SayCan] Ready")

    # ── public API ───────────────────────────────────────────────────

    def resolve_with_intent(self, query: str, user_id: str,
                             virtual_hour=None, user_pos=None,
                             prev_action: str = "Standing") -> dict:
        """
        Main entry point with intent classification.
        Dispatches to the correct pipeline:
          LOCATE → direct DB query (zero hallucination)
          NEED   → Say × Can gate
          CHAT   → caller handles with LLM directly
        """
        intent = self.intent_classifier.classify(query)
        print(f"[SayCan] Intent={intent} | query='{query[:40]}'")

        if intent == "LOCATE":
            return self._locate(query, user_id)

        if intent == "NEED":
            return self.resolve(
                query, user_id, virtual_hour, user_pos, prev_action)

        # CHAT: return signal for caller to handle
        return {
            "intent":      "CHAT",
            "best_action": None,
            "explanation": None,
        }

    def _locate(self, query: str, user_id: str) -> dict:
        """
        Direct DB lookup for object location queries.
        No LLM involved — zero hallucination.
        """
        # Search dynamic_objects for relevant labels
        q_lower = query.lower()
        all_objs = list(self.db.dynamic_objects.find(
            {}, {"label": 1, "last_seen_on": 1,
                 "room": 1, "furniture_pos": 1}))

        best_obj  = None
        best_sim  = 0.0

        if self.sbert is not None and all_objs:
            import numpy as np
            labels = [d.get("label", "") for d in all_objs]
            vecs   = self.sbert.encode(
                labels, normalize_embeddings=True)
            q_vec  = self.sbert.encode(
                query, normalize_embeddings=True)
            sims   = vecs @ q_vec
            best_i = int(sims.argmax())
            best_sim = float(sims[best_i])
            if best_sim >= 0.35:
                best_obj = all_objs[best_i]
        else:
            # Fallback: substring match
            for d in all_objs:
                if d.get("label", "").lower() in q_lower:
                    best_obj = d
                    break

        if best_obj:
            label    = best_obj.get("label", "object")
            location = best_obj.get("last_seen_on", "unknown area")
            room     = best_obj.get("room", "")
            nav      = best_obj.get("furniture_pos")
            answer   = (f"The {label} was last seen near "
                        f"{location}"
                        f"{' in the ' + room if room else ''}.")
            print(f"[SayCan-Locate] '{label}' @ '{location}' "
                  f"sim={best_sim:.2f}")
        else:
            answer = ("I haven't observed that item recently. "
                      "It may not be in my field of view.")
            nav    = None
            label  = "unknown"
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
        """
        Main entry point.
        Returns:
          best_action, scores, nav_target, explanation
        """
        say_scores  = self._compute_say(query)
        habit_probs = self._compute_can_habit(
            user_id, virtual_hour, user_pos, prev_action)
        env_scores  = self._compute_can_env()
        skill_scores= self._compute_can_skill(user_id)

        final_scores = {}
        for b in BEHAVIOR_LABELS:
            s = say_scores.get(b,   0.0)
            h = habit_probs.get(b,  0.0)
            e = env_scores.get(b,   ENV_FALLBACK)
            k = skill_scores.get(b, 1.0)
            # Pure product with epsilon floor (SayCan original,
        # hardened against zero-collapse when any single Can = 0).
        # max(eps, x) ensures no single weak signal kills the entire score.
        eps = 1e-3
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

        # Persist to MongoDB for analysis
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

        print(f"[SayCan] '{query}' → {best_action} "
              f"(score={best_score:.4f})")

        return {
            "best_action":  best_action,
            "best_score":   round(best_score, 4),
            "final_scores": {k: round(v, 6)
                             for k, v in sorted(
                                 final_scores.items(),
                                 key=lambda x: -x[1])},
            "say_scores":   say_scores,
            "habit_probs":  habit_probs,
            "env_scores":   env_scores,
            "skill_scores": skill_scores,
            "nav_target":   nav_target,
            "nav_label":    nav_label,
            "explanation":  explanation,
        }

    # ── Say: LLM semantic scoring ────────────────────────────────────

    def _compute_say(self, query: str) -> dict:
        """
        Ask LLM to score each behaviour for relevance to the query.
        LLM does NOT touch the database — semantic scoring only.
        """
        behavior_list = ", ".join(BEHAVIOR_LABELS)
        prompt = (
            f"You are a home robot intent classifier.\n"
            f"User said: \"{query}\"\n\n"
            f"Rate how relevant each behaviour is to this request "
            f"(0.0 = irrelevant, 1.0 = perfectly matches).\n"
            f"Behaviours: {behavior_list}\n\n"
            f"Output ONLY valid JSON. No explanation. Example:\n"
            f"{{\"Eating\": 0.9, \"Drinking\": 0.3, \"Watching\": 0.0}}"
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
                print(f"[SayCan-Say] top: "
                      f"{sorted(result.items(), key=lambda x:-x[1])[:3]}")
                return result
        except Exception as e:
            print(f"[SayCan-Say] LLM failed: {e}")

        # Fallback: uniform
        return {b: 1.0 / len(BEHAVIOR_LABELS) for b in BEHAVIOR_LABELS}

    # ── Can_habit: ManifoldEngine priors ────────────────────────────

    def _compute_can_habit(self, user_id, virtual_hour,
                            user_pos, prev_action) -> dict:
        """
        Use ManifoldEngine MLP to get spatiotemporal habit priors.
        Returns 15-class probability distribution filtered to 11 behaviours.
        """
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
            out   = {}
            for b in BEHAVIOR_LABELS:
                out[b] = float(probs.get(b, 0.0))
            total = sum(out.values()) or 1.0
            return {b: v / total for b, v in out.items()}
        except Exception as e:
            print(f"[SayCan-Habit] failed: {e}")
            return {b: 1.0 / len(BEHAVIOR_LABELS) for b in BEHAVIOR_LABELS}

    # ── Can_env: dynamic_objects feasibility ────────────────────────

    def _compute_can_env(self) -> dict:
        """
        Open-Vocabulary Can_env using SBERT cosine alignment.

        For each dynamic object in the environment, compute cosine
        similarity against each behaviour's prototype vector.
        The best matching object determines the behaviour feasibility.

        Fallback (no SBERT): hardcoded keyword substring match.

        Thresholds:
          sim >= 0.60 → 1.0  (high feasibility)
          sim >= 0.40 → 0.60 (moderate feasibility)
          sim <  0.40 → 0.30 (low, object not relevant)
        """
        import numpy as np

        try:
            dyn_docs = list(self.db.dynamic_objects.find(
                {}, {"label": 1}))
            dyn_labels = [
                d["label"].lower()
                for d in dyn_docs if d.get("label")]
        except Exception:
            dyn_labels = []

        if not dyn_labels:
            return {b: ENV_FALLBACK for b in BEHAVIOR_LABELS}

        # ── SBERT path (open-vocabulary) ─────────────────────────────
        if self._behavior_proto_vecs and self.sbert is not None:
            obj_vecs = self.sbert.encode(
                dyn_labels, normalize_embeddings=True)

            result = {}
            for b in BEHAVIOR_LABELS:
                proto = self._behavior_proto_vecs.get(b)
                if proto is None:
                    result[b] = ENV_FALLBACK
                    continue
                sims    = obj_vecs @ proto          # shape (n_objects,)
                max_sim = float(sims.max())

                if max_sim >= 0.60:
                    result[b] = 1.0
                elif max_sim >= 0.40:
                    result[b] = 0.60
                else:
                    result[b] = ENV_FALLBACK

                # Log top matching object for debugging
                top_i = int(sims.argmax())
                if max_sim >= 0.40:
                    print(f"[SayCan-Env] {b}: "
                          f"'{dyn_labels[top_i]}' sim={max_sim:.2f} "
                          f"→ {result[b]}")

            return result

        # ── Fallback: keyword substring match ────────────────────────
        dyn_set = set(dyn_labels)
        obj_map = self._get_behavior_objects()
        result  = {}
        for b in BEHAVIOR_LABELS:
            needed = obj_map.get(b, [])
            if not needed:
                result[b] = 0.5
                continue
            found = any(
                any(kw in lbl for lbl in dyn_set)
                for kw in needed)
            result[b] = 1.0 if found else ENV_FALLBACK

        feasible = [b for b, v in result.items() if v >= 1.0]
        print(f"[SayCan-Env] feasible (keyword): {feasible}")
        return result

    # ── Can_skill: SKILL.md preference filter ───────────────────────

    def _compute_can_skill(self, user_id: str) -> dict:
        """
        Apply SKILL.md 'What NOT to do' as soft gate.
        If all required objects for a behaviour are rejected → 0.1
        Otherwise → 1.0
        """
        try:
            doc = self.db.user_skills.find_one({"user_id": user_id})
            skill_md = doc.get("skill_md", "") if doc else ""
        except Exception:
            skill_md = ""

        # Extract rejected items from "## What NOT to do" section.
        # Robust parsing: strip whitespace, case-insensitive match,
        # stop at any new top-level section (## but not ###).
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
                # Stop at next top-level markdown section
                if (clean.startswith("#")
                        and not clean.startswith("###")):
                    break
                if clean.startswith("-"):
                    rejected.add(clean.lower())

        obj_map = self._get_behavior_objects()
        result  = {}
        for b in BEHAVIOR_LABELS:
            needed = obj_map.get(b, [])
            if not needed:
                result[b] = 1.0
                continue
            # If any needed object appears in rejection list
            penalised = any(
                any(item in rej for rej in rejected)
                for item in needed
            )
            result[b] = 0.1 if penalised else 1.0

        blocked = [b for b, v in result.items() if v < 1.0]
        if blocked:
            print(f"[SayCan-Skill] penalised: {blocked}")
        return result

    # ── Navigation target resolution ────────────────────────────────

    def _resolve_nav(self, action: str):
        """Find the most relevant object/location for the chosen action."""
        obj_map = self._get_behavior_objects()
        needed  = obj_map.get(action, [])

        for kw in needed:
            doc = self.db.dynamic_objects.find_one(
                {"label": {"$regex": kw, "$options": "i"}},
                {"furniture_pos": 1, "last_seen_on": 1, "room": 1},
            )
            if doc and doc.get("furniture_pos"):
                label = doc.get("last_seen_on", kw)
                return doc["furniture_pos"], label

        # Fallback: find zone from observation_logs
        zone_doc = self.db.observation_logs.find_one(
            {"action": action},
            sort=[("weight", -1)],
        )
        if zone_doc:
            return zone_doc.get("pos"), zone_doc.get("zone_name", action)

        return None, action

    # ── Response generation ──────────────────────────────────────────

    def _generate_response(self, query: str, user_id: str,
                            best_action: str, nav_label: str,
                            best_score: float,
                            say_scores: dict,
                            habit_probs: dict) -> str:
        """
        LLM generates a natural response AFTER the action is decided.
        LLM receives the decided action — no hallucination possible.
        """
        say_top = sorted(say_scores.items(), key=lambda x: -x[1])[:3]
        hab_top = sorted(habit_probs.items(), key=lambda x: -x[1])[:3]

        prompt = (
            f"You are a friendly home robot.\n"
            f"User ({user_id}) said: \"{query}\"\n"
            f"System decided: {best_action} "
            f"near {nav_label} (confidence={best_score:.2f})\n\n"
            f"Generate a SHORT, natural English response (1-2 sentences).\n"
            f"Mention the location if known. End with a question if helpful.\n"
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
            return (f"Based on your request, I think you want to "
                    f"{best_action.lower()}{loc}. Shall I help?")

    # ── SBERT Open-Vocabulary prototype building ────────────────────

    def _build_behavior_prototypes(self):
        """
        Build one prototype SBERT vector per behaviour by averaging
        the embeddings of its associated objects.
        Used by _compute_can_env for open-vocabulary object alignment:
          any new object (e.g. dragon_fruit) is automatically mapped
          to the nearest behaviour without any code change.
        """
        import numpy as np
        obj_map = self._get_behavior_objects()

        for behavior, objects in obj_map.items():
            if objects:
                vecs = self.sbert.encode(
                    objects, normalize_embeddings=True)
                proto = vecs.mean(axis=0)
            else:
                # No objects → use behaviour name as proxy
                proto = self.sbert.encode(
                    behavior, normalize_embeddings=True)
            norm = float(np.linalg.norm(proto))
            self._behavior_proto_vecs[behavior] = (
                proto / (norm + 1e-8)).astype("float32")

        print(f"[SayCan] Behavior prototypes built "
              f"({len(self._behavior_proto_vecs)} behaviours)")

    # ── Gemma distillation ───────────────────────────────────────────

    def _distill_behavior_objects(self):
        """
        Ask Gemma to map each behaviour to required physical objects.
        Result cached in MongoDB (saycan_behavior_objects collection).
        """
        cached = list(self.db.saycan_behavior_objects.find({}))
        if cached:
            for doc in cached:
                b = doc.get("behavior")
                o = doc.get("objects", [])
                if b:
                    self._behavior_objects[b] = o
            print(f"[SayCan] Loaded behavior-object map "
                  f"({len(self._behavior_objects)} entries)")
            return

        print("[SayCan] Distilling behavior-object map via Gemma...")
        behavior_list = ", ".join(BEHAVIOR_LABELS)
        prompt = (
            f"You are a home robot spatial expert.\n"
            f"For each behaviour, list the physical objects "
            f"that are REQUIRED or strongly associated.\n"
            f"Use simple lowercase English nouns only.\n"
            f"Behaviours: {behavior_list}\n\n"
            f"Output ONLY valid JSON. Example:\n"
            f"{{\"Eating\": [\"food\", \"bowl\", \"fork\"], "
            f"\"Watching\": [\"tv\", \"remote\"]}}"
        )
        try:
            resp = requests.post(
                f"{self.ollama_url}/api/chat",
                json={
                    "model":    "llama3.1:8b-instruct-q4_K_M",
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
                    objs = [o.lower().strip()
                            for o in mapping.get(b, [])
                            if isinstance(o, str)]
                    self._behavior_objects[b] = objs
                    bulk.append({
                        "behavior": b,
                        "objects":  objs,
                        "source":   "llama3.1:8b-instruct-q4_K_M",
                        "timestamp": datetime.datetime.utcnow(),
                    })
                if bulk:
                    self.db.saycan_behavior_objects.delete_many({})
                    self.db.saycan_behavior_objects.insert_many(bulk)
                print(f"[SayCan] Distilled {len(bulk)} behavior-object entries")
                return
        except Exception as e:
            print(f"[SayCan] Gemma distillation failed: {e}")

        # Use fallback
        self._behavior_objects = dict(_FALLBACK_BEHAVIOR_TO_OBJECTS)
        print("[SayCan] Using fallback behavior-object map")

    def _get_behavior_objects(self) -> dict:
        with self._lock:
            if self._behavior_objects:
                return dict(self._behavior_objects)
        return dict(_FALLBACK_BEHAVIOR_TO_OBJECTS)