import datetime
import requests
import json
import re
import logging
import threading

from config import Config

logger = logging.getLogger(__name__)

LLM_TIMEOUT              = Config.LLM_TIMEOUT
LLM_TEMP                 = Config.LLM_TEMPERATURE
LLM_TOKENS               = Config.LLM_MAX_TOKENS
SNAPSHOT_TTL_HOURS       = Config.SNAPSHOT_TTL_HOURS
SNAPSHOT_MAX_ITEMS       = Config.SNAPSHOT_MAX_ITEMS
SBERT_CATEGORY_THRESHOLD = 0.35

ONE_SHOT_SYSTEM = """You are a home service robot assistant.

## Environment snapshot
The following is a COMPLETE list of objects currently detected in the home.
This list is ground truth — do NOT assume objects exist unless listed here.
{env_snapshot}

## User skill profile
{skill_md}

## Hard constraints
- NEVER recommend items marked as disliked in the skill profile.
- If an item the user wants is NOT in the snapshot, say so honestly. Do NOT invent locations.
- CATEGORY: if the user is hungry, only recommend [food] items. If thirsty, only [drink] items.
- Always mention SPECIFIC item names from the snapshot. Never say "some food" or "some drinks".
- If the requested item does not exist in the snapshot, say it is not available. Do NOT pretend it exists.

## Output format
Reply with valid JSON only:
{{
  "answer": "natural sentence for the user",
  "nav_target": "furniture label where the object is, or 'unknown'",
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
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                "stream": False,
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
        logger.warning(f"JSON parse failed: {e} | raw: {raw[:150]}")
    return None


class InteractionEngine:

    def __init__(self, mongo_client, vector_memory, ollama_url, model_name):
        self.db         = mongo_client[Config.DB_NAME]
        self.vector     = vector_memory
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.conv_logs  = self.db["conversation_logs"]

        try:
            from modules.skill_manager import SkillManager
            self.skill_manager      = SkillManager(
                db_client  = mongo_client,
                ollama_url = ollama_url,
                model_name = model_name,
            )
            self._has_skill_manager = True
        except Exception as e:
            logger.warning(f"[InteractionEngine] SkillManager not available: {e}")
            self._has_skill_manager = False

        self._sbert       = None
        self._need_vecs   = None
        self._init_sbert()

    def _init_sbert(self):
        try:
            if hasattr(self.vector, 'model'):
                self._sbert     = self.vector.model
                self._need_vecs = self._build_description_vecs(NEED_DESCRIPTIONS)
                logger.info("[SBERT] need category vectors built")
        except Exception as e:
            logger.warning(f"[SBERT] init failed: {e}")

    def _build_description_vecs(self, descriptions: dict) -> dict:
        vecs = {}
        for key, text in descriptions.items():
            vecs[key] = self._sbert.encode(text, normalize_embeddings=True)
        return vecs

    INTERRUPT_KEYWORDS = {
        "stop", "cancel", "abort", "never mind", "forget it",
        "停下", "停止", "算了", "取消", "不用了",
    }

    CLASSIFY_SYSTEM = "You are an intent classifier. Reply with exactly one word only: service, query, or chat."

    CLASSIFY_PROMPT = """Classify this message into exactly one category.

service - the user wants something done physically: fetch, bring, get, grab, navigate, prepare an object. Also when the user mentions wanting a specific item (food, drink, object) even casually.
query   - the user asks about object locations or availability in the home (where is X, is there any X, do we have X).
chat    - casual conversation, emotions, opinions, complaints, topics unrelated to home objects.

Message: "{query}"

Reply with one word only: service, query, or chat"""

    def _classify_intent(self, query: str) -> str:
        q = query.lower().strip()

        if any(kw in q for kw in self.INTERRUPT_KEYWORDS):
            return "interrupt"

        result = _call_llm(
            self.ollama_url,
            self.model_name,
            self.CLASSIFY_SYSTEM,
            self.CLASSIFY_PROMPT.format(query=query),
            max_tokens=5,
        )

        if not result:
            logger.warning("[Classify] LLM returned None, defaulting to chat")
            return "chat"

        intent = result.strip().lower().split()[0]
        if intent not in ("service", "query", "chat"):
            logger.warning(f"[Classify] unexpected LLM output '{result}', defaulting to chat")
            intent = "chat"

        print(f"[Classify] LLM -> {intent}")
        return intent

    def _extract_need_category(self, query: str,
                               threshold: float = SBERT_CATEGORY_THRESHOLD) -> str | None:
        if not self._sbert or not self._need_vecs:
            return None

        import numpy as np
        q_vec  = self._sbert.encode(query, normalize_embeddings=True)
        scores = {
            cat: float(np.dot(q_vec, vec))
            for cat, vec in self._need_vecs.items()
        }
        logger.debug(f"[SBERT category] scores={scores}")

        best_cat   = max(scores, key=scores.get)
        best_score = scores[best_cat]

        print(f"[Category] {best_cat} ({best_score:.3f}) threshold={threshold}")
        if best_score >= threshold:
            return best_cat
        return None

    def _is_person_query(self, query: str) -> bool:
        known_users = set()
        try:
            for doc in self.db.user_positions.find(
                {"room": {"$exists": True, "$ne": ""}},
                {"user_id": 1},
            ):
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
            return (
                f"- {d.get('label','?')}{cat}: "
                f"on {d.get('last_seen_on','?')} in {d.get('room','?')}"
            )

        if need_category:
            priority = [d for d in docs if d.get("category") == need_category]
            others   = [d for d in docs if d.get("category") != need_category]
            lines = (
                [f"=== {need_category.upper()} items (priority) ==="]
                + [_fmt(d) for d in priority]
                + (["=== Other items ==="] + [_fmt(d) for d in others] if others else [])
            )
        else:
            lines = [_fmt(d) for d in docs]

        return "\n".join(lines)

    def _oneshot_process(self, query: str, user_id: str, room: str,
                         need_category: str | None = None) -> dict | None:
        sm = self.skill_manager if self._has_skill_manager else None

        env_snapshot = self._build_env_snapshot(need_category)
        print(f"[Snapshot] {len(env_snapshot.splitlines())} items in context")

        skill_md = ""
        if sm:
            skill_md = sm.get_skill_chunks(user_id, query) or sm.get_skill(user_id) or ""
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
            f"Time: {datetime.datetime.now().strftime('%H:%M')}\n\n"
            f"Reply with the JSON format specified."
        )

        result = _call_llm_json(
            self.ollama_url, self.model_name,
            system, user_prompt, max_tokens=LLM_TOKENS,
        )

        if not result:
            logger.warning("[Oneshot] LLM returned None or invalid JSON")
            return None

        answer     = result.get("answer", "").strip().strip('"').strip("'")
        nav_target = result.get("nav_target", "unknown")
        nav_label  = result.get("nav_label", nav_target)
        rec_items  = result.get("recommended_items", [])
        nav_pos    = (
            self._resolve_pos(nav_target)
            if nav_target and nav_target != "unknown" else None
        )

        print(f"[Oneshot] answer='{answer[:80]}' | nav={nav_label}")

        self._schedule_skill_update(
            user_id=user_id, query=query,
            answer=answer, env_snapshot=env_snapshot, rec_items=rec_items,
        )

        if sm:
            self.db.user_skills.update_one(
                {"user_id": user_id},
                {"$set": {"last_used": datetime.datetime.utcnow()}},
            )
            sm.check_stale(user_id)

        return {
            "status":          "Success",
            "answer":          answer,
            "nav_target":      nav_pos or nav_target,
            "nav_label":       nav_label,
            "options":         self._build_options(nav_pos, nav_label, query),
            "confidence":      0.85,
            "intent_type":     "oneshot",
            "recommendations": [{"label": i} for i in rec_items],
            "is_personalized": bool(skill_md and "(No skill profile" not in skill_md),
        }

    def _schedule_skill_update(self, user_id, query, answer, env_snapshot, rec_items):
        if not self._has_skill_manager:
            return

        import re as _re, json as _json

        plain_answer = answer
        try:
            m = _re.search(r'\{.*\}', answer, _re.DOTALL)
            if m:
                parsed = _json.loads(m.group(0))
                if "answer" in parsed:
                    plain_answer = parsed["answer"]
        except Exception:
            pass

        def _bg():
            try:
                sm     = self.skill_manager
                should = sm.should_update(
                    user_id=user_id, query=query, answer=plain_answer, trace=[],
                )
                if should:
                    sm.update(user_id, query, plain_answer, trace=[{
                        "step":   1,
                        "tool":   "oneshot",
                        "input":  {"query": query},
                        "result": (
                            f"User: \"{query}\"\nRobot: \"{plain_answer}\"\n"
                            f"Objects:\n{env_snapshot}\nRecommended: {rec_items}"
                        ),
                    }])
                    print(f"[BgEvolve] Skill updated for {user_id}")

                has_gap, missing = sm.detect_gap(user_id, query, plain_answer)
                if has_gap and missing:
                    print(f"[BgEvolve] Gap: {missing} -> fill_gap()")
                    sm.fill_gap(user_id, query, missing)
            except Exception as e:
                logger.warning(f"[BgEvolve] Error: {e}")

        threading.Thread(target=_bg, daemon=True).start()

    def process(self, query, user_id="Unknown", robot_pos=None,
                user_pos=None, room=""):
        print(f"\n[Interact] user={user_id} | query='{query}' | room={room}")

        intent = self._classify_intent(query)
        print(f"[Classify] intent={intent}")

        if intent == "interrupt":
            return self._interrupt_response(query, user_id)
        if intent == "chat":
            return self._chat_response(query, user_id)
        if intent == "query":
            return self._query_response(query, user_id, room)

        need_category = self._extract_need_category(query)
        print(f"[Category] need_category={need_category}")

        result = self._oneshot_process(
            query=query, user_id=user_id, room=room,
            need_category=need_category,
        )

        if result:
            self._log_conversation(
                query=query, expanded_query=query,
                intent_type="oneshot", user_id=user_id,
                answer=result["answer"],
                nav_target=result["nav_target"],
                nav_label=result["nav_label"],
                room=room,
                recommendations=result.get("recommendations", []),
                is_personalized=result.get("is_personalized", False),
            )
            return result

        logger.warning("[Interact] Oneshot failed, using fallback")
        return self._pipeline_fallback(query, user_id, room)

    def _interrupt_response(self, query, user_id):
        return {
            "status": "Interrupted", "answer": "Understood, stopping now.",
            "nav_target": None, "nav_label": None,
            "options": [{"id": 3, "label": "Close"}],
            "confidence": 1.0, "intent_type": "interrupt",
            "recommendations": [], "is_personalized": False,
        }

    def _chat_response(self, query, user_id):
        answer = _call_llm(
            self.ollama_url, self.model_name,
            CHAT_SYSTEM,
            f'{user_id} said: "{query}"',
        ) or "I am here for you!"
        return {
            "status": "Success",
            "answer": answer.strip().strip('"').strip("'"),
            "nav_target": None, "nav_label": None,
            "options": [{"id": 3, "label": "Close"}],
            "confidence": 1.0, "intent_type": "chat",
            "recommendations": [], "is_personalized": False,
        }

    def _detect_query_category(self, query: str) -> str | None:
        q = query.lower()
        FOOD_WORDS  = {
            "fruit", "fruits", "food", "snack", "snacks", "meal",
            "apple", "banana", "pizza", "sandwich", "cake", "donut",
            "carrot", "broccoli", "orange", "eat", "edible",
        }
        DRINK_WORDS = {
            "drink", "drinks", "beverage", "beverages",
            "juice", "water", "cola", "coffee", "tea", "soda",
            "bottle", "cup", "glass", "sip", "thirsty",
        }
        if any(w in q for w in FOOD_WORDS):
            return "food"
        if any(w in q for w in DRINK_WORDS):
            return "drink"
        return None

    def _query_response(self, query, user_id, room):
        from datetime import timedelta

        if self._is_person_query(query):
            users = list(self.db.user_positions.find(
                {"room": {"$exists": True, "$ne": ""}},
                {"user_id": 1, "room": 1},
            ))
            seen = {}
            for u in users:
                key = u.get("user_id", "").lower()
                if key not in seen:
                    seen[key] = u
            users = list(seen.values())

            answer = (
                ", ".join(
                    f"{u.get('user_id','?')} is in {u.get('room','?')}"
                    for u in users
                ) if users else "No family members currently tracked."
            )
            return {
                "status": "Success", "answer": answer,
                "nav_target": None, "nav_label": None,
                "options": [{"id": 3, "label": "Close"}],
                "confidence": 0.9, "intent_type": "query",
                "recommendations": [], "is_personalized": False,
            }

        query_category = self._detect_query_category(query)

        cutoff = datetime.datetime.utcnow() - timedelta(hours=2)
        docs   = list(self.db.dynamic_objects.find(
            {"last_seen": {"$gte": cutoff}},
            {"label": 1, "room": 1, "last_seen_on": 1,
             "interact_count": 1, "category": 1},
        ).sort("interact_count", -1))

        if not docs:
            docs = list(self.db.dynamic_objects.find(
                {},
                {"label": 1, "room": 1, "last_seen_on": 1,
                 "interact_count": 1, "category": 1},
            ).sort("interact_count", -1).limit(15))

        EXCLUDE = {"user_mom", "user_dad", "user", "person", "people"}
        docs    = [d for d in docs if d.get("label", "").lower() not in EXCLUDE]

        if query_category:
            filtered = [d for d in docs if d.get("category") == query_category]
            if filtered:
                docs = filtered
                print(f"[Query] category filter '{query_category}' -> {len(docs)} items")

        if not docs:
            category_msg = f"{query_category} " if query_category else ""
            return {
                "status": "Success",
                "answer": f"No {category_msg}items currently detected in the home.",
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
                    [
                        (d, float(np.dot(
                            q_vec,
                            self._sbert.encode(
                                d.get("label", ""), normalize_embeddings=True
                            ),
                        )))
                        for d in docs
                    ],
                    key=lambda x: x[1],
                    reverse=True,
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
            "status": "Success", "answer": f"I found: {items_str}.",
            "nav_target": nav_target, "nav_label": nav_label,
            "options": self._build_options(nav_target, nav_label, query),
            "confidence": 0.85, "intent_type": "query",
            "recommendations": [
                {"label": d["label"], "last_seen_on": d.get("last_seen_on"),
                 "room": d.get("room"), "score": 0.8}
                for d in relevant_docs[:4]
            ],
            "is_personalized": False,
        }

    def _pipeline_fallback(self, query, user_id, room):
        env_snapshot = self._build_env_snapshot(self._extract_need_category(query))
        answer = _call_llm(
            self.ollama_url, self.model_name,
            "You are a home robot. Reply in ONE sentence in English. Do NOT wrap in quotes.",
            f'User said: "{query}"\n\nObjects in home:\n{env_snapshot}\n\nGive a direct answer.',
            max_tokens=120,
        ) or "Sorry, I cannot process that right now."
        return {
            "status": "Success",
            "answer": answer.strip().strip('"').strip("'"),
            "nav_target": None, "nav_label": None,
            "options": [{"id": 3, "label": "Close"}],
            "confidence": 0.3, "intent_type": "fallback",
            "recommendations": [], "is_personalized": False,
        }

    def confirm(self, choice, nav_target, nav_label, user_id, query):
        try:
            self.conv_logs.find_one_and_update(
                {"user_id": user_id, "query": query},
                {"$set": {
                    "user_choice":  choice,
                    "confirmed_at": datetime.datetime.now(),
                }},
                sort=[("timestamp", -1)],
            )
        except Exception as e:
            print(f"[Confirm] skipped: {e}")

        if choice == 1:
            return {
                "status": "navigate", "nav_target": nav_target,
                "nav_label": nav_label, "message": f"Navigating to {nav_label}.",
            }
        if choice == 2:
            pos_str = (
                f"[{nav_target[0]:.1f}, {nav_target[1]:.1f}]"
                if nav_target else "unknown"
            )
            return {"status": "info_only", "message": f"{nav_label} is at {pos_str}."}
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

    def _log_conversation(self, query, expanded_query, intent_type,
                           user_id, answer, nav_target, nav_label,
                           room, recommendations, is_personalized):
        self.conv_logs.insert_one({
            "user_id":         user_id,
            "query":           query,
            "expanded_query":  expanded_query,
            "intent_type":     intent_type,
            "answer":          answer,
            "nav_label":       nav_label,
            "nav_target":      nav_target,
            "room":            room,
            "recommendations": recommendations,
            "is_personalized": is_personalized,
            "timestamp":       datetime.datetime.now(),
        })