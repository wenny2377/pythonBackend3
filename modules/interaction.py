import datetime
import requests
import json
import re
import logging
import threading
from collections import defaultdict

from config import Config
from modules.perception.scene_graph import build_scene_text

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

EXECUTE_KEYWORDS = {
    "拿", "bring", "取", "fetch", "幫我拿", "給我", "get me"
}

CONFIRM_WORDS = {"好", "ok", "是的", "對", "可以", "麻煩", "請", "yes", "okay", "sure"}
REJECT_WORDS = {"不要", "換一個", "other", "another"}
DISLIKE_WORDS = {"不喜歡", "討厭", "don't like", "hate"}
CANCEL_WORDS = {"取消", "算了", "cancel", "never mind"}

ONE_SHOT_SYSTEM = """You are a home service robot assistant.

## Current Scene Graph
{scene_graph}

## Environment snapshot (all objects at home)
{env_snapshot}

## User skill profile
{skill_md}

## Hard constraints
- NEVER recommend items marked as disliked in the skill profile.
- If an item the user wants is NOT in the snapshot, say so honestly.
- CATEGORY: if the user is hungry, only recommend [food] items. If thirsty, only [drink] items.
- Always mention SPECIFIC item names from the snapshot.
- Use the Scene Graph to understand what the user is currently doing and where they are.
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
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
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
        logger.warning(f"JSON parse failed: {e}")
    return None


class SessionState:
    def __init__(self):
        self.last_nav_label = ""
        self.last_nav_target = None
        self.last_recommended = []
        self.excluded_items = set()
        self.excluded_categories = set()
        self.pending_confirm = False
        self.pending_need_type = None
        self.pending_query = ""
        self.pending_item = ""
        self.available_items = []
        self.conversation_history = []
        self.last_scene_graph = ""

    def reset(self):
        self.__init__()


class InteractionEngine:

    def __init__(self, mongo_client, vector_memory, ollama_url, model_name,
                 saycan_engine=None):
        self.db = mongo_client[Config.DB_NAME]
        self.vector = vector_memory
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.conv_logs = self.db["conversation_logs"]
        self.saycan = saycan_engine

        try:
            from modules.memory.skill_manager import SkillManager
            self.skill_manager = SkillManager(
                db_client=mongo_client,
                ollama_url=ollama_url,
                model_name=model_name,
            )
            self._has_skill_manager = True
        except Exception as e:
            logger.warning(f"[InteractionEngine] SkillManager not available: {e}")
            self._has_skill_manager = False

        self._sbert = None
        self._need_vecs = None
        self._init_sbert()

        self._conv_buffer = defaultdict(list)
        self._sessions = {}

    def _init_sbert(self):
        try:
            if hasattr(self.vector, 'model'):
                self._sbert = self.vector.model
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

    def _build_current_scene_graph(self, user_id: str, room: str,
                                    user_pos=None) -> str:
        try:
            pos_doc = self.db.user_positions.find_one(
                {"$or": [{"user_id": user_id},
                         {"user_id": user_id.lower()}]},
                {"x": 1, "z": 1, "forward": 1}
            )
            if pos_doc and not user_pos:
                user_pos = {"x": float(pos_doc.get("x", 0)),
                            "z": float(pos_doc.get("z", 0))}

            user_forward = None
            if pos_doc and pos_doc.get("forward"):
                fwd = pos_doc["forward"]
                if isinstance(fwd, list) and len(fwd) >= 3:
                    user_forward = {"x": fwd[0], "z": fwd[2]}
                elif isinstance(fwd, dict):
                    user_forward = fwd

            latest_eval = self.db.eval_logs.find_one(
                {"user": user_id},
                sort=[("timestamp", -1)],
                projection={"body_position": 1, "held_object": 1}
            )
            skel_body = latest_eval.get("body_position", "unknown") if latest_eval else "unknown"
            held_object = latest_eval.get("held_object", "none") if latest_eval else "none"

            vh_doc = self.db.system_config.find_one({"key": "virtual_hour"})
            virtual_hour = vh_doc.get("value") if vh_doc else None

            scene_text = build_scene_text(
                user_pos=user_pos,
                user_forward=user_forward,
                room_name=room,
                skel_body=skel_body,
                head_pitch=-999,
                held_object=held_object,
                db=self.db,
                user_id=user_id,
                virtual_hour=virtual_hour,
            )
            return scene_text
        except Exception as e:
            logger.warning(f"[SceneGraph] build failed: {e}")
            return "(Scene graph unavailable)"

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
            q_vec = self._sbert.encode(query, normalize_embeddings=True)
            scores = {cat: float(np.dot(q_vec, vec))
                      for cat, vec in self._need_vecs.items()}
            best_cat = max(scores, key=scores.get)
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

    def _is_execute_command(self, query: str) -> bool:
        q = query.lower().strip()
        return any(kw in q for kw in EXECUTE_KEYWORDS)

    def _is_confirmation(self, query: str) -> bool:
        q = query.lower().strip()
        return any(w in q for w in CONFIRM_WORDS)

    def _is_rejection(self, query: str) -> bool:
        q = query.lower().strip()
        return any(w in q for w in REJECT_WORDS)

    def _is_dislike(self, query: str) -> bool:
        q = query.lower().strip()
        return any(w in q for w in DISLIKE_WORDS)

    def _is_cancel(self, query: str) -> bool:
        q = query.lower().strip()
        return any(w in q for w in CANCEL_WORDS)

    def _extract_need_category(self, query: str) -> str:
        if not self._sbert or not self._need_vecs:
            return "food"

        import numpy as np
        q_vec = self._sbert.encode(query, normalize_embeddings=True)
        scores = {cat: float(np.dot(q_vec, vec))
                  for cat, vec in self._need_vecs.items()}
        best_cat = max(scores, key=scores.get)
        best_score = scores[best_cat]
        return best_cat if best_score >= SBERT_CATEGORY_THRESHOLD else "food"

    def _get_available_items(self, need_type, user_id, session):
        from datetime import datetime, timedelta

        cutoff = datetime.utcnow() - timedelta(hours=SNAPSHOT_TTL_HOURS)

        docs = list(self.db.dynamic_objects.find({
            "last_seen": {"$gte": cutoff},
            "category": need_type,
            "label": {"$nin": list(session.excluded_items)}
        }, {
            "label": 1, "category": 1, "last_seen_on": 1
        }).sort("interact_count", -1).limit(10))

        if not docs:
            docs = list(self.db.dynamic_objects.find({
                "category": need_type,
                "label": {"$nin": list(session.excluded_items)}
            }, {
                "label": 1, "category": 1, "last_seen_on": 1
            }).limit(10))

        return docs

    def _get_user_preference(self, user_id, need_type):
        doc = self.db.user_skills.find_one({"user_id": user_id})
        if not doc:
            return None

        skill_md = doc.get("skill_md", "")
        import re
        match = re.search(r"## Preferences\n(.*?)(?=\n## |$)", skill_md, re.DOTALL)
        if not match:
            return None

        prefs = match.group(1)
        target = "food" if need_type == "food" else "drink"

        for line in prefs.split('\n'):
            if target in line.lower():
                parts = line.split(':')
                if len(parts) > 1:
                    items = [i.strip() for i in parts[1].split(',')]
                    return items[0] if items else None
        return None

    def _add_to_disliked(self, user_id, item):
        doc = self.db.user_skills.find_one({"user_id": user_id})
        skill_md = doc.get("skill_md", "") if doc else ""

        if "## Disliked" not in skill_md:
            skill_md += "\n\n## Disliked\n"

        if f"- {item}" not in skill_md:
            skill_md += f"- {item}\n"

        self.db.user_skills.update_one(
            {"user_id": user_id},
            {"$set": {"skill_md": skill_md, "version": (doc.get("version", 0) + 1) if doc else 1}},
            upsert=True
        )

    def _get_user_skill_md(self, user_id):
        if not self._has_skill_manager:
            return None
        try:
            return self.skill_manager.get_skill(user_id) or ""
        except Exception:
            return None

    def _handle_need_with_rag(self, query, user_id, session):
        all_items = list(self.db.dynamic_objects.find(
            {},
            {"label": 1, "category": 1, "last_seen_on": 1, "interact_count": 1}
        ))

        EXCLUDE = {"user_mom", "user_dad", "user", "person", "people"}
        all_items = [d for d in all_items if d.get("label", "").lower() not in EXCLUDE]

        items_text = ""
        for item in all_items:
            cat = f" [{item.get('category', 'unknown')}]" if item.get("category") else ""
            items_text += f"- {item['label']}{cat}: on {item.get('last_seen_on', 'unknown')}\n"

        if not items_text:
            items_text = "(No objects currently detected in the home.)"

        excluded_text = ""
        if session.excluded_items:
            excluded_text = f"User already rejected: {', '.join(session.excluded_items)}\n"

        skill_md = self._get_user_skill_md(user_id) or ""

        system = f"""You are a home robot assistant. Based ONLY on the information below, respond to the user.

ITEMS AVAILABLE AT HOME:
{items_text}

USER PREFERENCES (from skill profile):
{skill_md}

ITEMS ALREADY REJECTED IN THIS CONVERSATION:
{excluded_text}

RULES:
1. Only recommend items from the AVAILABLE list above
2. NEVER recommend items in REJECTED list
3. NEVER recommend items marked as disliked in USER PREFERENCES
4. If user is hungry, recommend a SPECIFIC food item from the list
5. If user is thirsty, recommend a SPECIFIC drink item from the list
6. ALWAYS output a specific recommended_item name, never null when user wants food/drink
7. If user wants something specific that is not available, say honestly it's not available
8. Always ask for confirmation before proceeding
9. Output ONLY valid JSON

OUTPUT FORMAT:
{{"answer": "your response to the user", "recommended_item": "specific item name"}}

User: {query}"""

        result = _call_llm_json(self.ollama_url, self.model_name, "", system, max_tokens=200)

        if not result:
            return None

        recommended = result.get("recommended_item")
        
        if recommended and recommended != "null":
            for item in all_items:
                if item["label"].lower() == recommended.lower():
                    session.pending_confirm = True
                    session.pending_item = recommended
                    session.available_items = all_items
                    session.pending_query = query
                    return {
                        "status": "Success",
                        "answer": result.get("answer", f"Would you like me to get you {recommended}?"),
                        "nav_target": None,
                        "nav_label": None,
                        "intent_type": "need_confirm",
                        "options": [{"id": 1, "label": "Yes"}, {"id": 2, "label": "No, something else"}, {"id": 3, "label": "Cancel"}],
                        "recommendations": [{"label": recommended}],
                        "available_items": [i["label"] for i in all_items[:5]],
                        "is_personalized": bool(skill_md),
                        "confidence": 0.9
                    }

        return {
            "status": "Success",
            "answer": result.get("answer", "I'm not sure how to help with that."),
            "nav_target": None,
            "nav_label": None,
            "intent_type": "need_response",
            "options": [{"id": 3, "label": "Close"}],
            "recommendations": [],
            "is_personalized": False,
            "confidence": 0.85
        }

    def _need_with_confirm(self, query, user_id, session, scene_graph):
        need_type = self._extract_need_category(query)

        available = self._get_available_items(need_type, user_id, session)

        if not available:
            return {
                "answer": f"Sorry, there is no {need_type} available at home right now.",
                "nav_target": None,
                "nav_label": None,
                "intent_type": "need_unavailable",
                "options": [{"id": 3, "label": "Close"}],
                "recommendations": [],
                "is_personalized": False,
                "confidence": 0.8
            }

        preferred = self._get_user_preference(user_id, need_type)
        recommended_item = None

        if preferred:
            for item in available:
                if preferred.lower() in item["label"].lower():
                    recommended_item = item["label"]
                    break

        if not recommended_item and available:
            recommended_item = available[0]["label"]

        answer = f"Would you like me to get you {recommended_item}?"

        session.pending_confirm = True
        session.pending_need_type = need_type
        session.pending_query = query
        session.pending_item = recommended_item
        session.available_items = available

        return {
            "answer": answer,
            "nav_target": None,
            "nav_label": None,
            "intent_type": "need_confirm",
            "options": [{"id": 1, "label": "Yes"}, {"id": 2, "label": "No, something else"}, {"id": 3, "label": "Cancel"}],
            "recommendations": [{"label": recommended_item}],
            "available_items": [i["label"] for i in available[:5]],
            "is_personalized": preferred is not None,
            "confidence": 0.9
        }

    def _process_feedback(self, query, user_id, session):
        q = query.lower().strip()

        if self._is_rejection(q):
            if hasattr(session, 'pending_item') and session.pending_item:
                session.excluded_items.add(session.pending_item.lower())

            available = [i for i in session.available_items
                         if i["label"].lower() not in session.excluded_items]

            if available:
                new_item = available[0]["label"]
                session.pending_item = new_item
                return {
                    "answer": f"How about {new_item}? Would you like that?",
                    "nav_target": None,
                    "nav_label": None,
                    "intent_type": "need_confirm",
                    "options": [{"id": 1, "label": "Yes"}, {"id": 2, "label": "No, something else"}, {"id": 3, "label": "Cancel"}],
                    "recommendations": [{"label": new_item}],
                    "is_personalized": True,
                    "confidence": 0.9
                }
            else:
                session.pending_confirm = False
                return {
                    "answer": "Sorry, there are no other options available.",
                    "nav_target": None,
                    "nav_label": None,
                    "intent_type": "need_unavailable",
                    "options": [{"id": 3, "label": "Close"}],
                    "recommendations": [],
                    "is_personalized": False,
                    "confidence": 0.8
                }

        if self._is_dislike(q):
            for item in session.excluded_items:
                if item in q:
                    self._add_to_disliked(user_id, item)
                    return {
                        "answer": f"Got it, I will never recommend {item} again.",
                        "nav_target": None,
                        "nav_label": None,
                        "intent_type": "feedback",
                        "options": [{"id": 3, "label": "Close"}],
                        "recommendations": [],
                        "is_personalized": True,
                        "confidence": 0.9
                    }

            if hasattr(session, 'pending_item') and session.pending_item:
                self._add_to_disliked(user_id, session.pending_item)
                session.excluded_items.add(session.pending_item.lower())

                available = [i for i in session.available_items
                             if i["label"].lower() not in session.excluded_items]

                if available:
                    new_item = available[0]["label"]
                    session.pending_item = new_item
                    return {
                        "answer": f"Understood. Would you like {new_item} instead?",
                        "nav_target": None,
                        "nav_label": None,
                        "intent_type": "need_confirm",
                        "options": [{"id": 1, "label": "Yes"}, {"id": 2, "label": "No, something else"}, {"id": 3, "label": "Cancel"}],
                        "recommendations": [{"label": new_item}],
                        "is_personalized": True,
                        "confidence": 0.9
                    }
                else:
                    session.pending_confirm = False
                    return {
                        "answer": "Sorry, there are no other options available.",
                        "nav_target": None,
                        "nav_label": None,
                        "intent_type": "need_unavailable",
                        "options": [{"id": 3, "label": "Close"}],
                        "recommendations": [],
                        "is_personalized": False,
                        "confidence": 0.8
                    }

        if self._is_confirmation(q):
            return self._execute_pending(session, user_id)

        if self._is_cancel(q):
            session.pending_confirm = False
            return {
                "answer": "Okay, let me know if you need anything else.",
                "nav_target": None,
                "nav_label": None,
                "intent_type": "cancelled",
                "options": [{"id": 3, "label": "Close"}],
                "recommendations": [],
                "is_personalized": False,
                "confidence": 0.9
            }

        return None

    def _execute_pending(self, session, user_id):
        session.pending_confirm = False

        recommended = getattr(session, 'pending_item', None)

        if recommended:
            self.db.observation_logs.insert_one({
                "user": user_id,
                "action": "get",
                "instance": recommended,
                "interacting_items": [recommended],
                "weight": 1,
                "timestamp": datetime.datetime.utcnow()
            })
            print(f"[Record] {user_id} accepted {recommended}")
            
            obj = self.db.dynamic_objects.find_one({"label": recommended.lower()})
            if obj:
                nav_label = obj.get("last_seen_on")
                nav_target = self._resolve_pos(nav_label)
                return {
                    "status": "Success",
                    "answer": f"Okay, getting you {recommended} from {nav_label}",
                    "nav_target": nav_target,
                    "nav_label": nav_label,
                    "options": self._build_options(nav_target, nav_label, ""),
                    "confidence": 0.9,
                    "intent_type": "execute",
                    "recommendations": [{"label": recommended}],
                    "is_personalized": True
                }
            
            return {
                "status": "Success",
                "answer": f"Okay, getting you {recommended}",
                "nav_target": None,
                "nav_label": None,
                "options": [{"id": 3, "label": "Close"}],
                "confidence": 0.9,
                "intent_type": "execute",
                "recommendations": [{"label": recommended}],
                "is_personalized": True
            }

        return {
            "answer": "Okay",
            "nav_target": None,
            "nav_label": None,
            "intent_type": "execute",
            "options": [{"id": 3, "label": "Close"}],
            "recommendations": [],
            "is_personalized": True,
            "confidence": 0.9
        }

    def _handle_execute(self, query, user_id):
        if self.saycan:
            sc_result = self.saycan.resolve(query, user_id)
            return {
                "status": "Success",
                "answer": sc_result.get("explanation", ""),
                "nav_target": sc_result.get("nav_target"),
                "nav_label": sc_result.get("nav_label", ""),
                "options": self._build_options(sc_result.get("nav_target"), sc_result.get("nav_label"), query),
                "confidence": sc_result.get("best_score", 0.85),
                "intent_type": "execute",
                "recommendations": [{"label": sc_result.get("best_action", "")}] if sc_result.get("best_action") else [],
                "is_personalized": False
            }

        return self._oneshot_fallback(query, user_id, "", "")

    def process(self, query, user_id="Unknown", robot_pos=None,
                user_pos=None, room=""):
        print(f"\n[Interact] user={user_id} | query='{query}' | room={room}")

        session = self._get_session(user_id)

        if any(kw in query.lower() for kw in INTERRUPT_KEYWORDS):
            self._reset_session(user_id)
            return self._interrupt_response()

        if session.pending_confirm:
            feedback_result = self._process_feedback(query, user_id, session)
            if feedback_result:
                self._log_conversation(
                    query=query, user_id=user_id,
                    answer=feedback_result.get("answer", ""),
                    nav_target=feedback_result.get("nav_target"),
                    nav_label=feedback_result.get("nav_label", ""),
                    room=room,
                    intent_type=feedback_result.get("intent_type", "feedback"),
                    recommendations=feedback_result.get("recommendations", []),
                    is_personalized=feedback_result.get("is_personalized", False),
                )
                return feedback_result

        intent = self._classify_intent(query)
        print(f"[Classify] intent={intent}")

        scene_graph = self._build_current_scene_graph(user_id, room, user_pos)
        session.last_scene_graph = scene_graph

        if intent == "interrupt":
            self._reset_session(user_id)
            return self._interrupt_response()

        if intent == "chat":
            return self._chat_response(query, user_id, scene_graph)

        if intent == "locate":
            result = self._query_response(query, user_id, room)

            session.last_nav_label = result.get("nav_label", "")
            session.last_nav_target = result.get("nav_target")

            self._log_conversation(
                query=query, user_id=user_id,
                answer=result.get("answer", ""),
                nav_target=result.get("nav_target"),
                nav_label=result.get("nav_label", ""),
                room=room,
                intent_type=result.get("intent_type", "query"),
                recommendations=result.get("recommendations", []),
                is_personalized=result.get("is_personalized", False),
            )
            return result

        if intent == "need":
            if self._is_execute_command(query):
                result = self._handle_execute(query, user_id)
            else:
                rag_result = self._handle_need_with_rag(query, user_id, session)
                if rag_result:
                    result = rag_result
                else:
                    result = self._need_with_confirm(query, user_id, session, scene_graph)

            session.last_nav_label = result.get("nav_label", "")
            session.last_nav_target = result.get("nav_target")
            session.last_recommended = [
                r.get("label", "") for r in result.get("recommendations", [])
            ]

            self._log_conversation(
                query=query, user_id=user_id,
                answer=result.get("answer", ""),
                nav_target=result.get("nav_target"),
                nav_label=result.get("nav_label", ""),
                room=room,
                intent_type=result.get("intent_type", "need"),
                recommendations=result.get("recommendations", []),
                is_personalized=result.get("is_personalized", False),
            )
            return result

        return self._oneshot_fallback(query, user_id, room, scene_graph)

    def _interrupt_response(self):
        return {
            "status": "Interrupted",
            "answer": "Understood, stopping now.",
            "nav_target": None,
            "nav_label": None,
            "options": [{"id": 3, "label": "Close"}],
            "confidence": 1.0,
            "intent_type": "interrupt",
            "recommendations": [],
            "is_personalized": False,
        }

    def _chat_response(self, query, user_id, scene_graph=""):
        system = CHAT_SYSTEM
        if scene_graph:
            system = (
                "You are a friendly home robot companion.\n"
                "You are aware of the current scene:\n"
                f"{scene_graph}\n\n"
                "Reply warmly and briefly. 1-2 sentences max.\n"
                "RULES:\n"
                "- Do NOT promise to fetch anything unless asked.\n"
                "- Do NOT wrap your response in quotation marks."
            )

        answer = self._call_llm_with_buffer(user_id, system, query) \
                 or "I am here for you!"
        answer = answer.strip().strip('"').strip("'")
        self._schedule_skill_update(
            user_id=user_id, query=query,
            answer=answer, env_snapshot=scene_graph, rec_items=[])
        return {
            "status": "Success",
            "answer": answer,
            "nav_target": None,
            "nav_label": None,
            "options": [{"id": 3, "label": "Close"}],
            "confidence": 1.0,
            "intent_type": "chat",
            "recommendations": [],
            "is_personalized": False,
        }

    def _query_response(self, query, user_id, room):
        all_items = list(self.db.dynamic_objects.find(
            {},
            {"label": 1, "category": 1, "last_seen_on": 1, "room": 1}
        ))

        EXCLUDE = {"user_mom", "user_dad", "user", "person", "people"}
        all_items = [d for d in all_items if d.get("label", "").lower() not in EXCLUDE]

        if not all_items:
            return {
                "status": "Success",
                "answer": "No items currently detected in the home.",
                "nav_target": None,
                "nav_label": None,
                "options": [{"id": 3, "label": "Close"}],
                "confidence": 0.5,
                "intent_type": "query",
                "recommendations": [],
                "is_personalized": False,
            }

        items_text = ""
        for item in all_items:
            cat = f" [{item.get('category', 'unknown')}]" if item.get("category") else ""
            items_text += f"- {item['label']}{cat}: on {item.get('last_seen_on', 'unknown')} in {item.get('room', 'unknown')}\n"

        system = f"""You are a home robot assistant. Based ONLY on the items below, answer the user's question.

ITEMS IN THE HOME:
{items_text}

RULES:
- If user asks "is there X" or "do we have X", check if X exists in the list above
- If user asks "where is X", provide the location from the list
- If user asks "what do we have", list items from the list
- If the item is NOT in the list, say "I don't have that"
- Do NOT invent items not in the list
- Keep response short

User: {query}

Respond directly to the user:"""

        answer = self._call_llm_with_buffer(user_id, "", system, max_tokens=150)

        if not answer:
            answer = "I'm not sure how to help with that."

        return {
            "status": "Success",
            "answer": answer,
            "nav_target": None,
            "nav_label": None,
            "options": [{"id": 3, "label": "Close"}],
            "confidence": 0.85,
            "intent_type": "query",
            "recommendations": [],
            "is_personalized": False,
        }

    def _oneshot_fallback(self, query, user_id, room, scene_graph=""):
        need_category = self._extract_need_category(query)
        env_snapshot = self._build_env_snapshot(need_category)

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
            scene_graph=scene_graph or "(not available)",
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

        answer = result.get("answer", "").strip().strip('"').strip("'")
        nav_target = result.get("nav_target", "unknown")
        nav_label = result.get("nav_label", nav_target)
        rec_items = result.get("recommended_items", [])
        nav_pos = self._resolve_pos(nav_target) \
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
            "status": "Success",
            "answer": answer,
            "nav_target": nav_pos or nav_target,
            "nav_label": nav_label,
            "options": self._build_options(nav_pos, nav_label, query),
            "confidence": 0.85,
            "intent_type": "oneshot",
            "recommendations": [{"label": i} for i in rec_items],
            "is_personalized": is_personalized,
        }

    def _extract_need_category(self, query: str) -> str | None:
        if not self._sbert or not self._need_vecs:
            return None
        import numpy as np
        q_vec = self._sbert.encode(query, normalize_embeddings=True)
        scores = {cat: float(np.dot(q_vec, vec))
                  for cat, vec in self._need_vecs.items()}
        best_cat = max(scores, key=scores.get)
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
        docs = list(self.db.dynamic_objects.find(
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
            others = [d for d in docs if d.get("category") != need_category]
            lines = (
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
                sm = self.skill_manager
                should = sm.should_update(
                    user_id=user_id, query=query, answer=answer, trace=[])
                if should:
                    sm.update(user_id, query, answer, trace=[{
                        "step": 1,
                        "tool": "interaction",
                        "input": {"query": query},
                        "result": (
                            f"User: \"{query}\"\nRobot: \"{answer}\"\n"
                            f"Scene:\n{env_snapshot}\nRecommended: {rec_items}"
                        ),
                    }])
            except Exception as e:
                print(f"[BgEvolve] Error: {e}")

        threading.Thread(target=_bg, daemon=True).start()

    def _call_llm_with_buffer(self, user_id: str, system: str,
                               query: str, max_tokens: int = None) -> str:
        history = self._conv_buffer.get(user_id, [])
        messages = [{"role": "system", "content": system}]
        messages.extend(history)
        messages.append({"role": "user", "content": query})

        try:
            resp = requests.post(
                f"{self.ollama_url}/api/chat",
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": LLM_TEMP,
                                "num_predict": max_tokens or LLM_TOKENS},
                },
                timeout=LLM_TIMEOUT,
            )
            resp.raise_for_status()
            answer = resp.json()["message"]["content"].strip()
            buf = self._conv_buffer[user_id]
            buf.append({"role": "user", "content": query})
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
                "status": "navigate",
                "nav_target": nav_target,
                "nav_label": nav_label,
                "message": f"Navigating to {nav_label}.",
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
            "user_id": user_id,
            "query": query,
            "intent_type": intent_type,
            "answer": answer,
            "nav_label": nav_label,
            "nav_target": nav_target,
            "room": room,
            "recommendations": recommendations,
            "is_personalized": is_personalized,
            "timestamp": datetime.datetime.now(),
        })