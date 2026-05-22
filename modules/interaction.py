import datetime
import requests
import json
import re
import logging
import threading
from collections import defaultdict

from config import Config

logger = logging.getLogger(__name__)

LLM_TIMEOUT              = Config.LLM_TIMEOUT
LLM_TEMP                 = Config.LLM_TEMPERATURE
LLM_TOKENS               = Config.LLM_MAX_TOKENS
SNAPSHOT_TTL_HOURS       = Config.SNAPSHOT_TTL_HOURS
SNAPSHOT_MAX_ITEMS       = Config.SNAPSHOT_MAX_ITEMS
SBERT_CATEGORY_THRESHOLD = 0.35
CONV_BUFFER_MAX_TURNS    = 5

LOCATE_KEYWORDS = {
    "where", "which room", "find", "location", "show me",
    "在哪", "哪裡", "找", "位置", "在哪裡",
}

NEED_KEYWORDS = {
    "hungry", "thirsty", "tired", "want", "need", "feel like",
    "bored", "cold", "hot", "rest", "craving", "fancy",
    "餓", "渴", "累", "想", "需要", "不舒服", "想要",
}

INTERRUPT_KEYWORDS = {
    "stop", "cancel", "abort", "never mind", "forget it",
    "停下", "停止", "算了", "取消", "不用了",
}

NEGATION_PATTERNS = [
    "don't want", "dont want", "not want",
    "don't like", "dont like", "not like",
    "don't need", "dont need",
    "not today", "something else", "another one",
    "change", "switch", "instead", "other",
    "no juice", "no cola", "no milk", "no water",
]

REFERENCE_PATTERNS = [
    "bring it", "get it", "fetch it",
    "bring that", "get that", "that one",
    "help me get", "help me grab",
]

CONFIRM_PATTERNS = [
    "yes", "ok", "okay", "sure", "please do",
    "go ahead", "that works", "sounds good",
]

ONE_SHOT_SYSTEM = """You are a home service robot assistant.

## Environment snapshot
{env_snapshot}

## User skill profile
{skill_md}

## Hard constraints
- NEVER recommend items marked as disliked in the skill profile.
- If an item the user wants is NOT in the snapshot, say so honestly.
- CATEGORY: if the user is hungry, only recommend [food] items. If thirsty, only [drink] items.
- Always mention SPECIFIC item names from the snapshot.
- If the requested item does not exist in the snapshot, say it is not available.

## Output format
Reply with valid JSON only:
{{
  "answer": "natural sentence for the user",
  "nav_target": "furniture label where the object is, or unknown",
  "nav_label": "same as nav_target",
  "recommended_items": ["item1", "item2"]
}}
"""

CHAT_SYSTEM = """You are a friendly home robot companion.
Reply warmly and briefly in English. 1-2 sentences max.
STRICT RULES:
- Do NOT mention, suggest, fetch, or promise any specific food, drink, or object.
- Do NOT say you will get anything for the user.
- Do NOT ask if the user wants something from the home.
- Only provide emotional support or casual conversation.
- Do NOT wrap your response in quotation marks."""

NEED_DESCRIPTIONS = {
    "food": (
        "The user is hungry and wants something to eat: "
        "food, meal, snack, something to munch on, starving, craving food."
    ),
    "drink": (
        "The user is thirsty and wants something to drink: "
        "water, juice, beverage, something to sip, parched, craving a drink."
    ),
}


def _call_llm(ollama_url, model, system, user, max_tokens=LLM_TOKENS):
    try:
        resp = requests.post(
            f"{ollama_url}/api/chat",
            json={
                "model":    model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                "stream":  False,
                "options": {"temperature": LLM_TEMP, "num_predict": max_tokens},
            },
            timeout=LLM_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return None


def _call_llm_json(ollama_url, model, system, user, max_tokens=300):
    raw = _call_llm(ollama_url, model, system, user, max_tokens=max_tokens)
    if not raw:
        return None
    try:
        clean = re.sub(r'```(?:json)?\s*', '', raw).strip().rstrip('`')
        match = re.search(r'\{.*\}', clean, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse failed: {e}")
    return None


class SessionState:
    def __init__(self):
        self.last_nav_label   = ""
        self.last_nav_target  = None
        self.last_recommended = []
        self.excluded_items   = set()
        self.last_intent      = ""
        self.pending_confirm  = False

    def reset(self):
        self.__init__()


class InteractionEngine:

    def __init__(self, mongo_client, vector_memory, ollama_url, model_name,
                 saycan_engine=None):
        self.db           = mongo_client[Config.DB_NAME]
        self.vector       = vector_memory
        self.ollama_url   = ollama_url
        self.model_name   = model_name
        self.conv_logs    = self.db["conversation_logs"]
        self.saycan       = saycan_engine

        try:
            from modules.skill_manager import SkillManager
            self.skill_manager = SkillManager(
                db_client  = mongo_client,
                ollama_url = ollama_url,
                model_name = model_name,
            )
            self._has_skill_manager = True
        except Exception as e:
            logger.warning(f"[InteractionEngine] SkillManager not available: {e}")
            self._has_skill_manager = False

        self._sbert     = None
        self._need_vecs = None
        self._init_sbert()

        self._conv_buffer = defaultdict(list)
        self._sessions    = {}

    def _init_sbert(self):
        try:
            if hasattr(self.vector, 'model'):
                self._sbert     = self.vector.model
                self._need_vecs = self._build_description_vecs(NEED_DESCRIPTIONS)
        except Exception as e:
            logger.warning(f"[SBERT] init failed: {e}")

    def _build_description_vecs(self, descriptions: dict) -> dict:
        vecs = {}
        for key, text in descriptions.items():
            vecs[key] = self._sbert.encode(text, normalize_embeddings=True)
        return vecs

    def _get_session(self, user_id: str) -> SessionState:
        if user_id not in self._sessions:
            self._sessions[user_id] = SessionState()
        return self._sessions[user_id]

    def _reset_session(self, user_id: str):
        self._sessions[user_id] = SessionState()

    def _classify_intent(self, query: str) -> str:
        q = query.lower().strip()

        if any(kw in q for kw in INTERRUPT_KEYWORDS):
            return "interrupt"

        if any(kw in q for kw in LOCATE_KEYWORDS):
            return "locate"

        if any(kw in q for kw in NEED_KEYWORDS):
            return "need"

        if self._sbert and self._need_vecs:
            import numpy as np
            q_vec  = self._sbert.encode(query, normalize_embeddings=True)
            scores = {cat: float(np.dot(q_vec, vec))
                      for cat, vec in self._need_vecs.items()}
            best_cat   = max(scores, key=scores.get)
            best_score = scores[best_cat]
            if best_score >= SBERT_CATEGORY_THRESHOLD:
                return "need"

        result = _call_llm(
            self.ollama_url, self.model_name,
            "You are an intent classifier. Reply with exactly one word only: service, query, or chat.",
            f"Classify: service=fetch/bring/get/prepare, "
            f"query=where/is there/do we have, chat=everything else.\n"
            f"Message: \"{query}\"\nReply with one word only.",
            max_tokens=5,
        )

        if not result:
            return "chat"

        intent = result.strip().lower().split()[0]
        if intent == "service":
            return "need"
        if intent == "query":
            return "locate"
        return "chat"

    def _is_negation(self, query: str, user_id: str) -> str | None:
        query_lower = query.lower().strip()
        session     = self._get_session(user_id)

        if not any(p in query_lower for p in NEGATION_PATTERNS):
            return None
        if not session.last_recommended:
            return None

        for item in session.last_recommended:
            if item.lower() in query_lower:
                return item
        return session.last_recommended[0]

    def _resolve_reference(self, query: str, session: SessionState) -> str:
        query_lower = query.lower().strip()

        if any(p in query_lower for p in CONFIRM_PATTERNS) and session.last_nav_label:
            return f"get {session.last_nav_label}"

        if any(p in query_lower for p in REFERENCE_PATTERNS) and session.last_nav_label:
            return f"{query} ({session.last_nav_label})"

        return query

    def process(self, query, user_id="Unknown", robot_pos=None,
                user_pos=None, room=""):
        print(f"\n[Interact] user={user_id} | query='{query}' | room={room}")

        session = self._get_session(user_id)

        if any(kw in query.lower() for kw in INTERRUPT_KEYWORDS):
            self._reset_session(user_id)
            return self._interrupt_response()

        negated = self._is_negation(query, user_id)
        if negated:
            session.excluded_items.add(negated.lower())
            result = self._recommend_excluding(session, user_id, room, query)
            self._schedule_skill_update(
                user_id=user_id, query=query,
                answer=result["answer"], env_snapshot="", rec_items=[])
            return result

        query = self._resolve_reference(query, session)
        intent = self._classify_intent(query)
        print(f"[Classify] intent={intent}")

        if intent == "interrupt":
            self._reset_session(user_id)
            return self._interrupt_response()

        if intent == "chat":
            return self._chat_response(query, user_id)

        if intent == "locate":
            if self.saycan:
                result = self.saycan.locate(query, user_id)
                result["intent_type"] = "query"
                result["status"]      = "Success"
                result["options"]     = self._build_options(
                    result.get("nav_target"), result.get("nav_label"), query)
                result["recommendations"]  = []
                result["is_personalized"]  = False
                result["confidence"]       = 0.9
            else:
                result = self._query_response(query, user_id, room)
            session.last_nav_label  = result.get("nav_label", "")
            session.last_nav_target = result.get("nav_target")
            return result

        if intent == "need":
            if self.saycan:
                virtual_hour = self.db.command("ping") and None
                try:
                    vh_doc = self.db.system_config.find_one({"key": "virtual_hour"})
                    virtual_hour = vh_doc.get("value") if vh_doc else None
                except Exception:
                    virtual_hour = None

                prev_doc = self.db.activity_sequences.find_one(
                    {"user": user_id}, sort=[("timestamp", -1)])
                prev_action = "Standing"
                if prev_doc and prev_doc.get("sequence"):
                    seq = prev_doc["sequence"]
                    if len(seq) >= 2:
                        prev_action = seq[-2].get("action", "Standing")

                pos_doc = self.db.user_positions.find_one({"user_id": user_id})
                est_pos = None
                if pos_doc:
                    est_pos = {"x": float(pos_doc.get("x", 0)),
                               "z": float(pos_doc.get("z", 0))}

                sc_result = self.saycan.resolve(
                    query        = query,
                    user_id      = user_id,
                    virtual_hour = virtual_hour,
                    user_pos     = est_pos,
                    prev_action  = prev_action,
                )

                answer     = sc_result.get("explanation", "")
                nav_target = sc_result.get("nav_target")
                nav_label  = sc_result.get("nav_label", "")
                best_action = sc_result.get("best_action", "")

                skill_md = ""
                if self._has_skill_manager:
                    skill_md = self.skill_manager.get_skill(user_id) or ""
                is_personalized = bool(skill_md and "(No skill profile" not in skill_md)

                result = {
                    "status":          "Success",
                    "answer":          answer,
                    "nav_target":      nav_target,
                    "nav_label":       nav_label,
                    "options":         self._build_options(nav_target, nav_label, query),
                    "confidence":      sc_result.get("best_score", 0.85),
                    "intent_type":     "saycan",
                    "recommendations": [{"label": best_action}] if best_action else [],
                    "is_personalized": is_personalized,
                    "saycan_scores":   sc_result.get("final_scores", {}),
                }
            else:
                result = self._oneshot_fallback(query, user_id, room)

            session.last_nav_label   = result.get("nav_label", "")
            session.last_nav_target  = result.get("nav_target")
            session.last_recommended = [
                r.get("label", "") for r in result.get("recommendations", [])
            ]
            session.last_intent     = "need"
            session.pending_confirm = bool(result.get("nav_target"))

            self._log_conversation(
                query=query, user_id=user_id,
                answer=result["answer"],
                nav_target=result["nav_target"],
                nav_label=result["nav_label"],
                room=room,
                intent_type=result["intent_type"],
                recommendations=result.get("recommendations", []),
                is_personalized=result.get("is_personalized", False),
            )
            self._schedule_skill_update(
                user_id=user_id, query=query,
                answer=result["answer"], env_snapshot="", rec_items=[])
            return result

        return self._oneshot_fallback(query, user_id, room)

    def _interrupt_response(self):
        return {
            "status":          "Interrupted",
            "answer":          "Understood, stopping now.",
            "nav_target":      None,
            "nav_label":       None,
            "options":         [{"id": 3, "label": "Close"}],
            "confidence":      1.0,
            "intent_type":     "interrupt",
            "recommendations": [],
            "is_personalized": False,
        }

    def _chat_response(self, query, user_id):
        answer = self._call_llm_with_buffer(user_id, CHAT_SYSTEM, query) \
                 or "I am here for you!"
        answer = answer.strip().strip('"').strip("'")
        self._schedule_skill_update(
            user_id=user_id, query=query,
            answer=answer, env_snapshot="", rec_items=[])
        return {
            "status":          "Success",
            "answer":          answer,
            "nav_target":      None,
            "nav_label":       None,
            "options":         [{"id": 3, "label": "Close"}],
            "confidence":      1.0,
            "intent_type":     "chat",
            "recommendations": [],
            "is_personalized": False,
        }

    def _query_response(self, query, user_id, room):
        from datetime import timedelta

        if self._is_person_query(query):
            users  = list(self.db.user_positions.find(
                {"room": {"$exists": True, "$ne": ""}},
                {"user_id": 1, "room": 1}))
            seen   = {}
            for u in users:
                key = u.get("user_id", "").lower()
                if key not in seen:
                    seen[key] = u
            users  = list(seen.values())
            answer = (
                ", ".join(f"{u.get('user_id','?')} is in {u.get('room','?')}"
                          for u in users)
                if users else "No family members currently tracked."
            )
            return {
                "status": "Success", "answer": answer,
                "nav_target": None, "nav_label": None,
                "options": [{"id": 3, "label": "Close"}],
                "confidence": 0.9, "intent_type": "query",
                "recommendations": [], "is_personalized": False,
            }

        specific_item = self._extract_specific_item(query)
        if specific_item:
            found = self.db.dynamic_objects.find_one(
                {"label": {"$regex": specific_item, "$options": "i"}})
            if not found:
                return {
                    "status":          "Success",
                    "answer":          f"I don't see any {specific_item} at home right now.",
                    "nav_target":      None,
                    "nav_label":       None,
                    "options":         [{"id": 3, "label": "Close"}],
                    "confidence":      0.9,
                    "intent_type":     "query",
                    "recommendations": [],
                    "is_personalized": False,
                }

        cutoff = datetime.datetime.utcnow() - timedelta(hours=2)
        docs   = list(self.db.dynamic_objects.find(
            {"last_seen": {"$gte": cutoff}},
            {"label": 1, "room": 1, "last_seen_on": 1, "interact_count": 1, "category": 1},
        ).sort("interact_count", -1))

        if not docs:
            docs = list(self.db.dynamic_objects.find(
                {},
                {"label": 1, "room": 1, "last_seen_on": 1, "interact_count": 1, "category": 1},
            ).sort("interact_count", -1).limit(15))

        EXCLUDE = {"user_mom", "user_dad", "user", "person", "people"}
        docs    = [d for d in docs if d.get("label", "").lower() not in EXCLUDE]

        if not docs:
            return {
                "status": "Success",
                "answer": "No items currently detected in the home.",
                "nav_target": None, "nav_label": None,
                "options": [{"id": 3, "label": "Close"}],
                "confidence": 0.5, "intent_type": "query",
                "recommendations": [], "is_personalized": False,
            }

        relevant_docs = docs[:3]
        if self._sbert:
            try:
                import numpy as np
                q_vec  = self._sbert.encode(query, normalize_embeddings=True)
                scored = sorted(
                    [(d, float(np.dot(q_vec,
                        self._sbert.encode(d.get("label", ""), normalize_embeddings=True))))
                     for d in docs],
                    key=lambda x: x[1], reverse=True,
                )
                relevant_docs = [d for d, s in scored[:5] if s > 0.20] or docs[:3]
            except Exception:
                pass

        items_str  = ", ".join(
            f"{d['label']} (on {d.get('last_seen_on','?')} in {d.get('room','?')})"
            for d in relevant_docs[:3]
        )
        nav_label  = relevant_docs[0].get("last_seen_on") if relevant_docs else None
        nav_target = self._resolve_pos(nav_label)

        return {
            "status":    "Success",
            "answer":    f"I found: {items_str}.",
            "nav_target": nav_target,
            "nav_label":  nav_label,
            "options":    self._build_options(nav_target, nav_label, query),
            "confidence": 0.85,
            "intent_type": "query",
            "recommendations": [
                {"label": d["label"], "last_seen_on": d.get("last_seen_on"),
                 "room": d.get("room")}
                for d in relevant_docs[:4]
            ],
            "is_personalized": False,
        }

    def _oneshot_fallback(self, query, user_id, room):
        need_category = self._extract_need_category(query)
        env_snapshot  = self._build_env_snapshot(need_category)

        skill_md = ""
        if self._has_skill_manager:
            skill_md = (self.skill_manager.get_skill_chunks(user_id, query)
                        or self.skill_manager.get_skill(user_id) or "")
        if not skill_md:
            skill_md = "(No skill profile yet.)"

        furniture_labels = [
            d["label"] for d in self.db.scene_snapshots.find({}, {"label": 1})
            if "label" in d
        ]

        system = ONE_SHOT_SYSTEM.format(
            env_snapshot=env_snapshot,
            skill_md=skill_md,
        )

        user_prompt = (
            f"User ID: {user_id}\n"
            f"User said: \"{query}\"\n"
            f"Robot current room: {room}\n"
            f"Known furniture: {', '.join(furniture_labels) or 'unknown'}\n"
            f"Time: {datetime.datetime.now().strftime('%H:%M')}\n"
            f"Reply with the JSON format specified."
        )

        result = _call_llm_json(
            self.ollama_url, self.model_name,
            system, user_prompt, max_tokens=LLM_TOKENS)

        if not result:
            return {
                "status": "Success",
                "answer": "Sorry, I cannot process that right now.",
                "nav_target": None, "nav_label": None,
                "options": [{"id": 3, "label": "Close"}],
                "confidence": 0.3, "intent_type": "fallback",
                "recommendations": [], "is_personalized": False,
            }

        answer     = result.get("answer", "").strip().strip('"').strip("'")
        nav_target = result.get("nav_target", "unknown")
        nav_label  = result.get("nav_label", nav_target)
        rec_items  = result.get("recommended_items", [])
        nav_pos    = self._resolve_pos(nav_target) \
                     if nav_target and nav_target != "unknown" else None

        self._schedule_skill_update(
            user_id=user_id, query=query,
            answer=answer, env_snapshot=env_snapshot, rec_items=rec_items)

        if self._has_skill_manager:
            self.db.user_skills.update_one(
                {"user_id": user_id},
                {"$set": {"last_used": datetime.datetime.utcnow()}})
            self.skill_manager.check_stale(user_id)

        is_personalized = bool(skill_md and "(No skill profile" not in skill_md)

        return {
            "status":          "Success",
            "answer":          answer,
            "nav_target":      nav_pos or nav_target,
            "nav_label":       nav_label,
            "options":         self._build_options(nav_pos, nav_label, query),
            "confidence":      0.85,
            "intent_type":     "oneshot",
            "recommendations": [{"label": i} for i in rec_items],
            "is_personalized": is_personalized,
        }

    def _recommend_excluding(self, session, user_id, room, query):
        from datetime import timedelta
        cutoff = datetime.datetime.utcnow() - timedelta(hours=2)

        docs = list(self.db.dynamic_objects.find(
            {"last_seen": {"$gte": cutoff}, "category": "drink",
             "label": {"$nin": list(session.excluded_items)}},
            {"label": 1, "last_seen_on": 1, "room": 1}))

        if not docs:
            docs = list(self.db.dynamic_objects.find(
                {"category": "drink",
                 "label": {"$nin": list(session.excluded_items)}},
                {"label": 1, "last_seen_on": 1, "room": 1}).limit(5))

        if not docs:
            return {
                "status": "Success",
                "answer": "I'm sorry, there are no other drinks available right now.",
                "nav_target": None, "nav_label": None,
                "options": [{"id": 3, "label": "Close"}],
                "confidence": 0.8, "intent_type": "service",
                "recommendations": [], "is_personalized": True,
            }

        session.last_recommended = [d["label"] for d in docs[:3]]
        items_str    = ", ".join(
            f"{d['label']} (on {d.get('last_seen_on', '?')})" for d in docs[:3])
        excluded_str = ", ".join(session.excluded_items)

        answer = _call_llm(
            self.ollama_url, self.model_name,
            "You are a friendly home robot. The user declined the previous recommendation. "
            "Suggest alternatives naturally in one sentence. "
            "Do not mention excluded items. Do not wrap in quotes.",
            f"User said: \"{query}\"\n"
            f"Available alternatives: {items_str}\n"
            f"Do NOT suggest: {excluded_str}\n"
            f"Reply in one natural sentence.",
            max_tokens=80,
        ) or f"There's also {items_str} available."

        nav_label  = docs[0].get("last_seen_on") if docs else None
        nav_target = self._resolve_pos(nav_label)

        session.last_nav_label  = nav_label or ""
        session.last_nav_target = nav_target
        session.pending_confirm = bool(nav_target)

        return {
            "status":          "Success",
            "answer":          answer,
            "nav_target":      nav_target,
            "nav_label":       nav_label,
            "options":         self._build_options(nav_target, nav_label, query),
            "confidence":      0.85,
            "intent_type":     "service",
            "recommendations": [{"label": d["label"]} for d in docs[:3]],
            "is_personalized": True,
        }

    def _extract_need_category(self, query: str) -> str | None:
        if not self._sbert or not self._need_vecs:
            return None
        import numpy as np
        q_vec  = self._sbert.encode(query, normalize_embeddings=True)
        scores = {cat: float(np.dot(q_vec, vec))
                  for cat, vec in self._need_vecs.items()}
        best_cat   = max(scores, key=scores.get)
        best_score = scores[best_cat]
        return best_cat if best_score >= SBERT_CATEGORY_THRESHOLD else None

    def _is_person_query(self, query: str) -> bool:
        known_users = set()
        try:
            for doc in self.db.user_positions.find(
                    {"room": {"$exists": True, "$ne": ""}}, {"user_id": 1}):
                uid = doc.get("user_id", "")
                if uid:
                    known_users.add(uid.lower())
                    known_users.update(uid.lower().replace("_", " ").split())
        except Exception:
            pass
        known_users.update({
            "dad", "mom", "father", "mother", "papa", "mama",
            "grandpa", "grandma", "husband", "wife", "brother", "sister",
        })
        q = query.lower()
        return any(name in q for name in known_users)

    def _extract_specific_item(self, query: str) -> str | None:
        q = query.lower().strip()
        patterns = [
            r"is there any (\w+)",
            r"do we have any (\w+)",
            r"do you have (\w+)",
            r"is there (\w+)",
            r"any (\w+) at home",
            r"where is (?:the |my )?(\w+)",
            r"i want (?:some |a |an )?(\w+)",
            r"i need (?:some |a |an )?(\w+)",
            r"get me (?:some |a |an )?(\w+)",
        ]
        STOP_WORDS = {
            "any", "some", "the", "a", "an", "there",
            "food", "drink", "drinks", "snack", "snacks",
            "fruit", "fruits", "water", "something",
        }
        for pattern in patterns:
            m = re.search(pattern, q)
            if m:
                item = m.group(1).strip()
                if item not in STOP_WORDS and len(item) > 2:
                    return item
        return None

    def _build_env_snapshot(self, need_category: str | None = None) -> str:
        from datetime import timedelta
        EXCLUDE = {
            "user_mom", "user_dad", "user", "person", "people",
            "wall", "floor", "ceiling", "window", "door",
        }
        cutoff = datetime.datetime.utcnow() - timedelta(hours=SNAPSHOT_TTL_HOURS)
        docs   = list(self.db.dynamic_objects.find(
            {"last_seen": {"$gte": cutoff}},
            {"label": 1, "category": 1, "room": 1,
             "last_seen_on": 1, "interact_count": 1},
        ).sort("interact_count", -1).limit(SNAPSHOT_MAX_ITEMS))

        if not docs:
            docs = list(self.db.dynamic_objects.find(
                {},
                {"label": 1, "category": 1, "room": 1,
                 "last_seen_on": 1, "interact_count": 1},
            ).sort("interact_count", -1).limit(SNAPSHOT_MAX_ITEMS))

        docs = [d for d in docs if d.get("label", "").lower() not in EXCLUDE]

        if not docs:
            return "(No objects currently detected in the home.)"

        def _fmt(d):
            cat = f" [{d['category']}]" if d.get("category") else ""
            return (f"- {d.get('label','?')}{cat}: "
                    f"on {d.get('last_seen_on','?')} in {d.get('room','?')}")

        if need_category:
            priority = [d for d in docs if d.get("category") == need_category]
            others   = [d for d in docs if d.get("category") != need_category]
            lines    = (
                [f"=== {need_category.upper()} items (priority) ==="]
                + [_fmt(d) for d in priority]
                + (["=== Other items ==="] + [_fmt(d) for d in others] if others else [])
            )
        else:
            lines = [_fmt(d) for d in docs]

        return "\n".join(lines)

    def _schedule_skill_update(self, user_id, query, answer, env_snapshot, rec_items):
        if not self._has_skill_manager:
            return

        def _bg():
            try:
                sm     = self.skill_manager
                should = sm.should_update(
                    user_id=user_id, query=query, answer=answer, trace=[])
                if should:
                    sm.update(user_id, query, answer, trace=[{
                        "step":   1,
                        "tool":   "interaction",
                        "input":  {"query": query},
                        "result": (
                            f"User: \"{query}\"\nRobot: \"{answer}\"\n"
                            f"Objects:\n{env_snapshot}\nRecommended: {rec_items}"
                        ),
                    }])
            except Exception as e:
                print(f"[BgEvolve] Error: {e}")

        threading.Thread(target=_bg, daemon=True).start()

    def _call_llm_with_buffer(self, user_id: str, system: str,
                               query: str, max_tokens: int = None) -> str:
        history  = self._conv_buffer.get(user_id, [])
        messages = [{"role": "system", "content": system}]
        messages.extend(history)
        messages.append({"role": "user", "content": query})

        try:
            resp = requests.post(
                f"{self.ollama_url}/api/chat",
                json={
                    "model":    self.model_name,
                    "messages": messages,
                    "stream":   False,
                    "options":  {"temperature": LLM_TEMP,
                                 "num_predict": max_tokens or LLM_TOKENS},
                },
                timeout=LLM_TIMEOUT,
            )
            resp.raise_for_status()
            answer = resp.json()["message"]["content"].strip()
            buf    = self._conv_buffer[user_id]
            buf.append({"role": "user",      "content": query})
            buf.append({"role": "assistant", "content": answer})
            if len(buf) > CONV_BUFFER_MAX_TURNS * 2:
                self._conv_buffer[user_id] = buf[-(CONV_BUFFER_MAX_TURNS * 2):]
            return answer
        except Exception as e:
            logger.error(f"[Buffer LLM] failed: {e}")
            return ""

    def confirm(self, choice, nav_target, nav_label, user_id, query):
        try:
            self.conv_logs.find_one_and_update(
                {"user_id": user_id, "query": query},
                {"$set": {"user_choice": choice,
                          "confirmed_at": datetime.datetime.now()}},
                sort=[("timestamp", -1)],
            )
        except Exception as e:
            print(f"[Confirm] skipped: {e}")

        if choice == 1:
            return {
                "status":    "navigate",
                "nav_target": nav_target,
                "nav_label":  nav_label,
                "message":   f"Navigating to {nav_label}.",
            }
        if choice == 2:
            pos_str = (f"[{nav_target[0]:.1f}, {nav_target[1]:.1f}]"
                       if nav_target else "unknown")
            return {"status": "info_only",
                    "message": f"{nav_label} is at {pos_str}."}
        return {"status": "cancelled", "message": "Cancelled."}

    def _resolve_pos(self, nav_label):
        if not nav_label or nav_label == "unknown":
            return None
        doc = self.db.scene_snapshots.find_one({"label": nav_label})
        return doc["pos"] if doc and doc.get("pos") else None

    def _build_options(self, nav_target, nav_label, query=""):
        if nav_target and nav_label:
            return [
                {"id": 1, "label": f"Navigate to '{nav_label}'"},
                {"id": 2, "label": "Just tell me the location"},
                {"id": 3, "label": "Cancel"},
            ]
        return [{"id": 3, "label": "Close"}]

    def _log_conversation(self, query, user_id, answer, nav_target,
                           nav_label, room, intent_type,
                           recommendations, is_personalized):
        self.conv_logs.insert_one({
            "user_id":         user_id,
            "query":           query,
            "intent_type":     intent_type,
            "answer":          answer,
            "nav_label":       nav_label,
            "nav_target":      nav_target,
            "room":            room,
            "recommendations": recommendations,
            "is_personalized": is_personalized,
            "timestamp":       datetime.datetime.now(),
        })
