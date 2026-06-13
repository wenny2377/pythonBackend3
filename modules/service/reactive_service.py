import re
import json
import logging
import datetime
import requests
from dataclasses import dataclass, field
from collections import defaultdict

from config import Config

logger = logging.getLogger(__name__)

LLM_TIMEOUT   = Config.LLM_TIMEOUT
LLM_TEMP      = Config.LLM_TEMPERATURE
LLM_TOKENS    = Config.LLM_MAX_TOKENS
TTL_HOURS     = Config.SNAPSHOT_TTL_HOURS
MAX_ITEMS     = Config.SNAPSHOT_MAX_ITEMS

SESSION_TIMEOUT_MINUTES = 30
MAX_TURN_HISTORY        = 5
HIGH_WEIGHT_THRESHOLD   = 10

INTENT_KEYWORDS = {
    "interrupt": {"stop", "cancel", "abort", "never mind", "forget it",
                  "停下", "停止", "算了", "取消", "不用了"},
    "locate":    {"where", "which room", "find", "location", "show me",
                  "在哪", "哪裡", "找", "位置"},
    "need":      {"hungry", "thirsty", "tired", "want", "need", "feel like",
                  "bored", "craving", "餓", "渴", "想", "需要", "想要"},
    "confirm":   {"好", "ok", "是", "對", "可以", "yes", "okay", "sure",
                  "要", "行"},
    "reject":    {"不要", "換", "other", "another", "something else",
                  "不行", "換一個", "no"},
    "dislike":   {"不喜歡", "討厭", "don't like", "hate", "disgusting",
                  "噁心", "最討厭", "never", "stop recommending"},
}

NEED_CATEGORIES = {
    "food":  ["food", "eat", "hungry", "meal", "snack", "餓", "吃"],
    "drink": ["drink", "thirsty", "water", "juice", "beverage", "渴", "喝"],
}

OBJECT_EXCLUDES = {
    "user_mom", "user_dad", "user", "person", "people",
    "wall", "floor", "ceiling", "window", "door",
}

PREF_STOPWORDS = {
    "user", "enjoys", "drink", "food", "likes", "frequently",
    "uses", "during", "in", "the", "a", "an", "some", "often",
    "usually", "mom", "dad",
}


@dataclass
class ConversationSession:
    user_id:         str
    state:           str  = "IDLE"
    pending_item:    str  = ""
    pending_nav:     dict = field(default_factory=dict)
    excluded_items:  set  = field(default_factory=set)
    turn_history:    list = field(default_factory=list)
    need_type:       str  = ""
    available_items: list = field(default_factory=list)
    last_updated:    datetime.datetime = field(
        default_factory=datetime.datetime.utcnow)

    def is_expired(self) -> bool:
        return (datetime.datetime.utcnow() - self.last_updated).seconds \
               > SESSION_TIMEOUT_MINUTES * 60

    def touch(self):
        self.last_updated = datetime.datetime.utcnow()

    def add_turn(self, role: str, content: str):
        self.turn_history.append({"role": role, "content": content})
        if len(self.turn_history) > MAX_TURN_HISTORY * 2:
            self.turn_history = self.turn_history[-(MAX_TURN_HISTORY * 2):]

    def reset(self):
        self.state           = "IDLE"
        self.pending_item    = ""
        self.pending_nav     = {}
        self.excluded_items  = set()
        self.turn_history    = []
        self.need_type       = ""
        self.available_items = []
        self.touch()


class ReactiveService:

    def __init__(self, db, skill_manager, vector_memory,
                 ollama_url: str, llm_model: str):
        self.db            = db
        self.skill_manager = skill_manager
        self.vector        = vector_memory
        self.ollama_url    = ollama_url
        self.llm_model     = llm_model
        self._sessions: dict[str, ConversationSession] = {}
        self._sbert     = None
        self._need_vecs = None
        try:
            if hasattr(vector_memory, 'model'):
                self._sbert     = vector_memory.model
                self._need_vecs = self._build_need_vecs()
        except Exception:
            pass

    def process(self, query: str, user_id: str, room: str = "") -> dict:
        print(f"\n[ReactiveService] user={user_id} | query='{query}'")
        session = self._get_or_create_session(user_id)

        if self._match_intent(query, "interrupt"):
            session.reset()
            return self._response_interrupt()

        if session.state == "CONFIRMING":
            return self._handle_confirmation_turn(query, session)

        intent = self._classify_intent(query)
        print(f"[ReactiveService] intent={intent}")

        session.add_turn("user", query)
        session.touch()

        if intent == "locate":
            result = self._handle_locate(query, user_id, room)
        elif intent == "need":
            result = self._handle_need(query, user_id, session)
        else:
            result = self._handle_chat(query, user_id)

        session.add_turn("assistant", result.get("answer", ""))
        return result

    def _classify_intent(self, query: str) -> str:
        q = query.lower().strip()

        for intent, keywords in INTENT_KEYWORDS.items():
            if any(kw in q for kw in keywords):
                return intent

        if self._sbert and self._need_vecs:
            import numpy as np
            q_vec  = self._sbert.encode(query, normalize_embeddings=True)
            scores = {cat: float(np.dot(q_vec, vec))
                      for cat, vec in self._need_vecs.items()}
            best   = max(scores, key=scores.get)
            if scores[best] >= 0.35:
                return "need"

        result = self._call_llm(
            system=(
                "Classify the user message into exactly one word: "
                "need, locate, or chat.\n"
                "need = wants food/drink/object\n"
                "locate = asks where something is\n"
                "chat = everything else\n"
                "Reply with one word only."
            ),
            user=f'Message: "{query}"',
            max_tokens=5,
        )
        if result:
            word = result.strip().lower().split()[0]
            if word in ("need", "locate", "chat"):
                return word

        return "chat"

    def _handle_locate(self, query: str, user_id: str, room: str) -> dict:
        items = self._get_available_items(category=None, max_items=50)

        if not items:
            return self._response(
                answer="I don't see any objects tracked at home right now.",
                intent_type="locate",
            )

        items_text = "\n".join(
            f"- {i['label']}: on {i.get('last_seen_on','?')} "
            f"in {i.get('room','?')}"
            for i in items
        )

        answer = self._call_llm(
            system=(
                f"You are a home robot. Answer ONLY based on this list:\n"
                f"{items_text}\n\n"
                "RULES:\n"
                "- If item exists: state location clearly\n"
                "- If item does NOT exist: say 'I don't see [item] at home'\n"
                "- Never invent locations\n"
                "- Keep answer to 1-2 sentences"
            ),
            user=query,
            max_tokens=80,
        ) or "I couldn't find that information."

        nav_label, nav_target = self._extract_nav_from_answer(answer, items)

        return self._response(
            answer=answer,
            nav_label=nav_label,
            nav_target=nav_target,
            intent_type="locate",
        )

    def _handle_need(self, query: str, user_id: str,
                     session: ConversationSession) -> dict:
        need_type = self._classify_need_type(query)
        session.need_type = need_type

        skill_exclusions = self._get_skill_exclusions(user_id)
        session.excluded_items.update(skill_exclusions)

        items = self._get_available_items(
            category=need_type,
            excluded=session.excluded_items,
        )

        if not items:
            return self._response(
                answer=f"I don't see any {need_type} available at home right now.",
                intent_type="need_unavailable",
            )

        preferred   = self._get_user_preference(user_id, need_type)
        recommended = None

        if preferred:
            for item in items:
                if preferred.lower() in item["label"].lower():
                    recommended = item
                    break

        if not recommended:
            recommended = items[0]

        obs_weight = self._get_item_obs_weight(user_id, recommended["label"], need_type)

        session.pending_item    = recommended["label"]
        session.available_items = items
        session.state           = "CONFIRMING"

        skill_md    = self._get_skill_md(user_id) or ""
        is_personal = bool(preferred)

        if obs_weight >= HIGH_WEIGHT_THRESHOLD:
            answer = self._call_llm(
                system=(
                    "You are a friendly home robot assistant. "
                    "The user has a strong established habit with this item. "
                    "Say you will get it for them directly. Max 1 sentence."
                ),
                user=(
                    f"User: '{query}'\n"
                    f"Item: {recommended['label']} "
                    f"(observed {obs_weight} times)\n"
                    f"Location: {recommended.get('last_seen_on','nearby')}"
                ),
                max_tokens=50,
            ) or f"I'll get your {recommended['label']} right away!"

            session.state = "EXECUTING"
            obj       = self.db.dynamic_objects.find_one(
                {"label": recommended["label"].lower()})
            nav_label  = obj.get("last_seen_on") if obj else None
            nav_target = self._resolve_nav(nav_label)
            session.reset()

            return self._response(
                answer=answer,
                nav_label=nav_label,
                nav_target=nav_target,
                intent_type="execute",
                recommendations=[{"label": recommended["label"]}],
                is_personalized=True,
                options=self._build_nav_options(nav_label, nav_target),
            )

        answer = self._call_llm(
            system=(
                "You are a friendly home robot assistant.\n"
                f"User skill profile:\n{skill_md[:500] if skill_md else '(none)'}\n\n"
                "Generate a short, natural offer for the recommended item. "
                "Ask for confirmation. Max 1 sentence."
            ),
            user=(
                f"User said: '{query}'\n"
                f"Recommended item: {recommended['label']}\n"
                f"Location: {recommended.get('last_seen_on', 'nearby')}\n"
                f"Generate the offer:"
            ),
            max_tokens=60,
        ) or f"Would you like some {recommended['label']}?"

        return self._response(
            answer=answer,
            intent_type="need_confirm",
            options=[
                {"id": 1, "label": "Yes"},
                {"id": 2, "label": "No, something else"},
                {"id": 3, "label": "Cancel"},
            ],
            recommendations=[{"label": recommended["label"]}],
            is_personalized=is_personal,
        )

    def _handle_confirmation_turn(self, query: str,
                                   session: ConversationSession) -> dict:
        session.add_turn("user", query)
        session.touch()

        if self._match_intent(query, "dislike"):
            return self._handle_dislike(query, session)
        if self._match_intent(query, "reject"):
            return self._handle_reject(session)
        if self._match_intent(query, "confirm"):
            return self._handle_confirm(session)

        return self._handle_reject(session)

    def _handle_confirm(self, session: ConversationSession) -> dict:
        item    = session.pending_item
        user_id = session.user_id

        self._record_preference(
            user_id=user_id, item=item,
            positive=True, time_slot=self._current_time_slot(),
        )

        obj        = self.db.dynamic_objects.find_one({"label": item.lower()})
        nav_label  = obj.get("last_seen_on") if obj else None
        nav_target = self._resolve_nav(nav_label)
        session.reset()

        return self._response(
            answer=f"Great! Getting you {item} from {nav_label}.",
            nav_label=nav_label,
            nav_target=nav_target,
            intent_type="execute",
            recommendations=[{"label": item}],
            is_personalized=True,
            options=self._build_nav_options(nav_label, nav_target),
        )

    def _handle_reject(self, session: ConversationSession) -> dict:
        if session.pending_item:
            session.excluded_items.add(session.pending_item.lower())

        remaining = [
            i for i in session.available_items
            if i["label"].lower() not in session.excluded_items
        ]

        if not remaining:
            session.reset()
            return self._response(
                answer="Sorry, I don't have any other options available.",
                intent_type="need_unavailable",
            )

        next_item               = remaining[0]
        session.pending_item    = next_item["label"]
        session.available_items = remaining

        return self._response(
            answer=f"How about {next_item['label']} instead?",
            intent_type="need_confirm",
            options=[
                {"id": 1, "label": "Yes"},
                {"id": 2, "label": "No, something else"},
                {"id": 3, "label": "Cancel"},
            ],
            recommendations=[{"label": next_item["label"]}],
        )

    def _handle_dislike(self, query: str,
                         session: ConversationSession) -> dict:
        item    = session.pending_item
        user_id = session.user_id

        if item:
            session.excluded_items.add(item.lower())
            self._record_preference(
                user_id=user_id, item=item,
                positive=False, time_slot=self._current_time_slot(),
            )

        remaining = [
            i for i in session.available_items
            if i["label"].lower() not in session.excluded_items
        ]

        if remaining:
            next_item            = remaining[0]
            session.pending_item = next_item["label"]
            return self._response(
                answer=(f"Got it, I won't recommend {item} again. "
                        f"Would you like {next_item['label']} instead?"),
                intent_type="need_confirm",
                options=[
                    {"id": 1, "label": "Yes"},
                    {"id": 2, "label": "No, something else"},
                    {"id": 3, "label": "Cancel"},
                ],
                recommendations=[{"label": next_item["label"]}],
                is_personalized=True,
            )

        session.reset()
        return self._response(
            answer=f"Got it, I won't recommend {item} again.",
            intent_type="feedback",
            is_personalized=True,
        )

    def _handle_chat(self, query: str, user_id: str) -> dict:
        session = self._get_or_create_session(user_id)
        history = session.turn_history[-MAX_TURN_HISTORY * 2:]
        messages = [
            {"role": "system", "content": (
                "You are a friendly home robot companion. "
                "Reply warmly in 1-2 sentences. "
                "Do NOT promise to fetch or prepare anything. "
                "Do NOT wrap response in quotes."
            )}
        ] + history + [{"role": "user", "content": query}]

        answer = self._call_llm_messages(messages, max_tokens=80) \
                 or "I'm here for you!"

        return self._response(answer=answer, intent_type="chat")

    def _record_preference(self, user_id: str, item: str,
                            positive: bool, time_slot: str = ""):
        try:
            if positive:
                bullet  = f"- User enjoys {item}"
                section = "## Preferences"
            else:
                bullet  = f"- Do not recommend {item} to this user"
                section = "## What NOT to do"
            self.skill_manager._insert_if_new(user_id, section, bullet)
            print(f"[ReactiveService] Preference recorded: {bullet}")
        except Exception as e:
            print(f"[ReactiveService] preference record error: {e}")

    def _get_available_items(self, category: str = None,
                              excluded: set = None,
                              max_items: int = MAX_ITEMS) -> list:
        cutoff    = datetime.datetime.utcnow() - datetime.timedelta(hours=TTL_HOURS)
        nin_labels = list(OBJECT_EXCLUDES | (excluded or set()))
        query = {
            "last_seen": {"$gte": cutoff},
            "label":     {"$nin": nin_labels},
        }
        if category:
            query["category"] = category

        docs = list(
            self.db.dynamic_objects.find(
                query,
                {"label":1, "category":1, "last_seen_on":1,
                 "room":1, "interact_count":1},
            ).sort("interact_count", -1).limit(max_items)
        )

        if not docs:
            query.pop("last_seen", None)
            docs = list(
                self.db.dynamic_objects.find(
                    query,
                    {"label":1, "category":1, "last_seen_on":1,
                     "room":1, "interact_count":1},
                ).limit(max_items)
            )

        return docs

    def verify_item_exists(self, item_name: str) -> bool:
        cutoff = datetime.datetime.utcnow() - datetime.timedelta(hours=TTL_HOURS)
        return self.db.dynamic_objects.find_one({
            "label":     item_name.lower(),
            "last_seen": {"$gte": cutoff},
        }) is not None

    def _get_user_preference(self, user_id: str,
                              need_type: str) -> str | None:
        skill_md = self._get_skill_md(user_id)
        if skill_md:
            match = re.search(
                r"## Preferences\n(.*?)(?=\n## |$)", skill_md, re.DOTALL)
            if match:
                target = "food" if need_type == "food" else "drink"
                for line in match.group(1).split("\n"):
                    if target in line.lower() or "enjoys" in line.lower() \
                            or "frequently" in line.lower():
                        parts = re.findall(r"\b\w+\b", line)
                        for p in parts:
                            if p.lower() not in PREF_STOPWORDS and len(p) > 2:
                                return p

        try:
            actions = (["Drinking", "SittingDrink"]
                       if need_type == "drink" else ["Eating", "Cooking"])
            obs = list(self.db.observation_logs.find(
                {"user": user_id, "action": {"$in": actions}},
                {"zone_name":1, "weight":1}
            ).sort("weight", -1).limit(3))

            if obs:
                top_zone = obs[0].get("zone_name", "")
                obj = self.db.dynamic_objects.find_one(
                    {"category": need_type, "last_seen_on": top_zone},
                    sort=[("interact_count", -1)]
                )
                if obj:
                    return obj["label"]

                obj = self.db.dynamic_objects.find_one(
                    {"category": need_type},
                    sort=[("interact_count", -1)]
                )
                if obj:
                    return obj["label"]
        except Exception as e:
            print(f"[ReactiveService] obs fallback error: {e}")

        return None

    def _get_skill_exclusions(self, user_id: str) -> set:
        skill_md = self._get_skill_md(user_id)
        if not skill_md:
            return set()
        excluded = set()
        match = re.search(
            r"## What NOT to do\n(.*?)(?=\n## |$)", skill_md, re.DOTALL)
        if match:
            for line in match.group(1).split("\n"):
                if "recommend" in line.lower():
                    parts = re.findall(r"\b\w+\b", line)
                    for p in parts:
                        if p.lower() not in {
                            "do", "not", "recommend", "to", "this", "user"
                        } and len(p) > 2:
                            excluded.add(p.lower())
        return excluded

    def _get_item_obs_weight(self, user_id: str,
                              item_label: str, need_type: str) -> float:
        try:
            actions = (["Drinking", "SittingDrink"]
                       if need_type == "drink" else ["Eating", "Cooking"])
            obj = self.db.dynamic_objects.find_one(
                {"label": item_label.lower()},
                {"last_seen_on": 1}
            )
            zone = obj.get("last_seen_on", "") if obj else ""

            docs = list(self.db.observation_logs.find(
                {"user": user_id, "action": {"$in": actions},
                 "zone_name": zone} if zone else
                {"user": user_id, "action": {"$in": actions}},
                {"weight": 1}
            ))
            return sum(d.get("weight", 0) for d in docs)
        except Exception:
            return 0.0

    def _get_skill_md(self, user_id: str) -> str | None:
        try:
            chunks = self.skill_manager.get_skill_chunks(user_id, "preferences")
            if chunks:
                return chunks
            return self.skill_manager.get_skill(user_id)
        except Exception:
            return None

    def _call_llm(self, system: str, user: str,
                  max_tokens: int = None) -> str | None:
        try:
            resp = requests.post(
                f"{self.ollama_url}/api/chat",
                json={
                    "model":    self.llm_model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user},
                    ],
                    "stream":  False,
                    "options": {
                        "temperature": LLM_TEMP,
                        "num_predict": max_tokens or LLM_TOKENS,
                    },
                },
                timeout=LLM_TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"].strip()
        except Exception as e:
            logger.error(f"[ReactiveService] LLM error: {e}")
            return None

    def _call_llm_messages(self, messages: list,
                            max_tokens: int = None) -> str | None:
        try:
            resp = requests.post(
                f"{self.ollama_url}/api/chat",
                json={
                    "model":    self.llm_model,
                    "messages": messages,
                    "stream":   False,
                    "options": {
                        "temperature": LLM_TEMP,
                        "num_predict": max_tokens or LLM_TOKENS,
                    },
                },
                timeout=LLM_TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"].strip()
        except Exception as e:
            logger.error(f"[ReactiveService] LLM messages error: {e}")
            return None

    def _get_or_create_session(self, user_id: str) -> ConversationSession:
        session = self._sessions.get(user_id)
        if session is None or session.is_expired():
            self._sessions[user_id] = ConversationSession(user_id=user_id)
        return self._sessions[user_id]

    def _match_intent(self, query: str, intent: str) -> bool:
        q = query.lower().strip()
        return any(kw in q for kw in INTENT_KEYWORDS.get(intent, set()))

    def _classify_need_type(self, query: str) -> str:
        q = query.lower()
        for cat, keywords in NEED_CATEGORIES.items():
            if any(kw in q for kw in keywords):
                return cat
        return "food"

    def _current_time_slot(self) -> str:
        h = datetime.datetime.now().hour
        if h < 10:  return "Morning"
        if h < 13:  return "Noon"
        if h < 18:  return "Afternoon"
        if h < 22:  return "Evening"
        return "Night"

    def _resolve_nav(self, label: str) -> list | None:
        if not label:
            return None
        doc = self.db.scene_snapshots.find_one({"label": label})
        return doc.get("pos") if doc else None

    def _extract_nav_from_answer(self, answer: str,
                                  items: list) -> tuple:
        answer_lower = answer.lower()
        for item in items:
            if item["label"].lower() in answer_lower:
                nav_label  = item.get("last_seen_on")
                nav_target = self._resolve_nav(nav_label)
                return nav_label, nav_target
        return None, None

    def _build_nav_options(self, nav_label, nav_target) -> list:
        if nav_label and nav_target:
            return [
                {"id": 1, "label": f"Navigate to '{nav_label}'"},
                {"id": 2, "label": "Just tell me the location"},
                {"id": 3, "label": "Cancel"},
            ]
        return [{"id": 3, "label": "Close"}]

    def _build_need_vecs(self) -> dict:
        descriptions = {
            "food":  "hungry want food eat meal snack 餓 吃飯",
            "drink": "thirsty want drink water juice beverage 渴 喝水",
        }
        return {
            k: self._sbert.encode(text, normalize_embeddings=True)
            for k, text in descriptions.items()
        }

    def _response(self, answer: str = "",
                  nav_label: str = None,
                  nav_target=None,
                  intent_type: str = "chat",
                  options: list = None,
                  recommendations: list = None,
                  is_personalized: bool = False,
                  confidence: float = 0.85) -> dict:
        return {
            "status":          "Success",
            "answer":          answer,
            "nav_label":       nav_label,
            "nav_target":      nav_target,
            "intent_type":     intent_type,
            "options":         options or [{"id": 3, "label": "Close"}],
            "recommendations": recommendations or [],
            "is_personalized": is_personalized,
            "confidence":      confidence,
        }

    def _response_interrupt(self) -> dict:
        return self._response(
            answer="Understood, stopping now.",
            intent_type="interrupt",
            confidence=1.0,
        )