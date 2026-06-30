import re
import datetime
import logging
import threading
import requests
import numpy as np
from dataclasses import dataclass, field

from config import Config

logger = logging.getLogger(__name__)

LLM_TIMEOUT = Config.LLM_TIMEOUT
LLM_TEMP    = Config.LLM_TEMPERATURE
LLM_TOKENS  = Config.LLM_MAX_TOKENS
TTL_HOURS   = Config.SNAPSHOT_TTL_HOURS
MAX_ITEMS   = Config.SNAPSHOT_MAX_ITEMS

HIGH_WEIGHT     = 10
SESSION_TTL     = 30
MAX_HISTORY     = 5
SBERT_THRESHOLD = 0.40

OBJECT_EXCLUDES = {
    "user_mom", "user_dad", "user", "person", "people",
    "wall", "floor", "ceiling", "window", "door",
}

PREF_STOPWORDS = {
    "user", "enjoys", "drink", "food", "likes", "frequently", "uses",
    "during", "in", "the", "a", "an", "some", "often", "usually",
    "mom", "dad", "recommend", "not", "do", "to", "this",
}

INTERRUPT_WORDS = {"stop", "cancel", "abort", "never mind", "forget it"}
CONFIRM_WORDS   = {
    "yes", "ok", "okay", "sure", "yeah", "yep", "please",
    "go ahead", "alright", "fine", "do it", "get it", "bring it",
    "that works", "sounds good",
}
REJECT_WORDS = {
    "no", "nope", "another", "something else", "other",
    "different", "not that", "change", "not really",
}
DISLIKE_WORDS = {
    "hate", "don't like", "dislike", "disgusting",
    "never", "not again", "awful", "terrible",
}

NO_INTERRUPT_ACTIONS = {
    "Laying",
    "Typing",
    "UsingPhone",
}

TIME_TONE = {
    "Morning":   "cheerful and energetic",
    "Noon":      "calm and helpful",
    "Afternoon": "relaxed and friendly",
    "Evening":   "warm and attentive",
    "Night":     "quiet and gentle",
}

DRINK_KEYWORDS = {"drink", "beverage", "thirst", "water", "juice", "soda", "cola", "bottle"}
FOOD_KEYWORDS  = {"eat", "food", "meal", "snack", "hungry", "fruit", "bowl", "plate"}

_ITEM_SYNONYMS = {
    "cola":     "cola soda coke fizzy carbonated cold drink beverage",
    "water":    "water bottle waterbottle hydration drink beverage H2O",
    "bottle":   "bottle water waterbottle hydration drink beverage",
    "juice":    "juice fruit drink beverage sweet",
    "cup":      "cup mug glass hot cold drink beverage",
    "apple":    "apple fruit food snack healthy",
    "banana":   "banana fruit food snack sweet",
    "bowl":     "bowl dish food meal soup cereal",
    "plate":    "plate dish food meal dinner",
    "food":     "food meal snack eat dinner lunch",
    "remote":   "remote clicker TV controller television channel",
    "phone":    "phone smartphone mobile call",
    "book":     "book reading novel literature",
    "laptop":   "laptop computer notebook typing work",
    "broom":    "broom sweep cleaning floor",
    "mop":      "mop cleaning floor wet",
    "pan":      "pan frying cooking kitchen skillet",
    "spatula":  "spatula cooking kitchen tool",
    "keyboard": "keyboard typing computer input",
}

DRINK_ACTIONS = ["Drinking", "SeatedDrinking"]
FOOD_ACTIONS  = ["Eating", "Cooking"]


@dataclass
class ConversationSession:
    user_id:         str
    state:           str  = "IDLE"
    pending_item:    str  = ""
    need_type:       str  = ""
    excluded_items:  set  = field(default_factory=set)
    available_items: list = field(default_factory=list)
    turn_history:    list = field(default_factory=list)
    last_updated:    datetime.datetime = field(
        default_factory=datetime.datetime.utcnow)

    def expired(self):
        return (datetime.datetime.utcnow() -
                self.last_updated).seconds > SESSION_TTL * 60

    def touch(self):
        self.last_updated = datetime.datetime.utcnow()

    def add_turn(self, role, content):
        self.turn_history.append({"role": role, "content": content})
        if len(self.turn_history) > MAX_HISTORY * 2:
            self.turn_history = self.turn_history[-(MAX_HISTORY * 2):]

    def reset(self):
        self.state           = "IDLE"
        self.pending_item    = ""
        self.need_type       = ""
        self.excluded_items  = set()
        self.available_items = []
        self.turn_history    = []
        self.touch()


class ReactiveService:

    def __init__(self, db, skill_manager, ollama_url, llm_model):
        self.db            = db
        self.skill_manager = skill_manager
        self.ollama_url    = ollama_url
        self.llm_model     = llm_model
        self._sessions     = {}
        self._vec_cache    = {}
        from sentence_transformers import SentenceTransformer
        from config import Config
        self._sbert = SentenceTransformer('paraphrase-MiniLM-L6-v2', device=Config.DEVICE)
        self._warm_cache()

    def process(self, query, user_id, room=""):
        session = self._session(user_id)

        if self._is_interrupt(query):
            session.reset()
            return self._resp("Understood, stopping now.", "interrupt")

        if session.state == "CONFIRMING":
            r = self._confirmation_turn(query, session)
            session.add_turn("assistant", r.get("answer", ""))
            return r

        intent, category = self._classify_intent(query)
        session.add_turn("user", query)
        session.touch()

        if intent == "query":
            r = self._handle_query(query, user_id)
        elif intent == "need":
            r = self._handle_need(query, user_id, session, category)
        else:
            r = self._handle_chat(query, user_id)

        session.add_turn("assistant", r.get("answer", ""))
        return r

    def process_stream(self, query, user_id, room=""):
        session = self._session(user_id)

        if self._is_interrupt(query):
            session.reset()
            yield {"type": "done",
                   **self._resp("Understood, stopping now.", "interrupt")}
            return

        if session.state == "CONFIRMING":
            r = self._confirmation_turn(query, session)
            session.add_turn("assistant", r.get("answer", ""))
            yield from self._wrap_stream(r)
            return

        intent, category = self._classify_intent(query)
        session.add_turn("user", query)
        session.touch()

        if intent == "query":
            yield from self._query_stream(query, user_id)
        elif intent == "need":
            yield from self._need_stream(query, user_id, session, category)
        else:
            yield from self._chat_stream(query, user_id)

    def _is_interrupt(self, query):
        return any(w in query.lower() for w in INTERRUPT_WORDS)

    def _classify_intent(self, query) -> tuple:
        raw = self._llm_call(
            "You are an intent classifier for a home robot.\n"
            "Reply with exactly two comma-separated words.\n"
            "Word 1: need, query, or chat\n"
            "Word 2: drink, food, or any\n\n"
            "need  = user is hungry/thirsty or wants something fetched\n"
            "query = user asks about objects, locations, people, devices\n"
            "chat  = greetings, feelings, small talk, anything else\n"
            "drink = thirsty, beverage, water, juice, soda, cold drink\n"
            "food  = hungry, eating, meal, snack, something to eat\n"
            "any   = not specific\n\n"
            "Examples:\n"
            "I am thirsty        → need, drink\n"
            "could use a drink   → need, drink\n"
            "I am hungry         → need, food\n"
            "feeling peckish     → need, food\n"
            "get me something    → need, any\n"
            "where is the remote → query, any\n"
            "what food do we have→ query, food\n"
            "I am sad            → chat, any\n"
            "hello               → chat, any",
            f'"{query}"',
            max_tokens=10,
        )
        intent   = "chat"
        category = None
        if raw:
            parts = [p.strip().lower() for p in raw.split(",")]
            if parts[0] in ("need", "query", "chat"):
                intent = parts[0]
            if len(parts) > 1 and parts[1] in ("drink", "food"):
                category = parts[1]
        return intent, category

    def _build_snapshot(self, user_id: str) -> str:
        parts = []

        objects = self._available(category=None)
        if objects:
            lines = "\n".join(
                f"  {i['label']} ({i.get('category','?')}): "
                f"at {i.get('last_seen_on','?')} in {i.get('room','?')}"
                for i in objects)
            parts.append(f"Objects at home:\n{lines}")

        users = list(self.db.user_positions.find(
            {}, {"user_id": 1, "room": 1, "activity": 1}))
        if users:
            lines = "\n".join(
                f"  {u['user_id']}: in {u.get('room','unknown')}"
                + (f", doing {u['activity']}" if u.get('activity') else "")
                for u in users)
            parts.append(f"People:\n{lines}")

        devices = list(self.db.device_states.find({}, {"label": 1, "state": 1}))
        if devices:
            lines = "\n".join(
                f"  {d['label']}: {d.get('state','unknown')}"
                for d in devices)
            parts.append(f"Device states:\n{lines}")

        habits = list(self.db.observation_logs.find(
            {"user": user_id, "weight": {"$gte": 5}},
            {"action": 1, "zone_name": 1, "time_slot": 1, "weight": 1},
            sort=[("weight", -1)],
        ).limit(5))
        if habits:
            lines = "\n".join(
                f"  {h['action']} at {h.get('zone_name','')} "
                f"({h.get('time_slot','')}): {int(h.get('weight',0))} times"
                for h in habits)
            parts.append(f"User habits:\n{lines}")

        skill_md = self._skill_md(user_id)
        if skill_md:
            parts.append(f"User preferences:\n{skill_md[:400]}")

        return "\n\n".join(parts)

    def _handle_query(self, query, user_id):
        snapshot = self._build_snapshot(user_id)
        system   = (
            "You are a home robot assistant.\n\n"
            f"Current home state:\n{snapshot}\n\n"
            "Rules:\n"
            "- Answer ONLY from the information above.\n"
            "- Never invent objects, locations, people, or states.\n"
            "- If something is not in the snapshot, say you don't know.\n"
            "- Keep answers concise. 1-3 sentences."
        )
        answer    = self._llm_call(system, query, max_tokens=120) \
                    or "I'm not sure, let me check."
        nav_label, nav_target = self._extract_nav(answer, self._available(None))
        return self._resp(answer, "query",
                          nav_label=nav_label, nav_target=nav_target)

    def _query_stream(self, query, user_id):
        snapshot = self._build_snapshot(user_id)
        system   = (
            "You are a home robot assistant.\n\n"
            f"Current home state:\n{snapshot}\n\n"
            "Rules:\n"
            "- Answer ONLY from the information above.\n"
            "- Never invent objects, locations, people, or states.\n"
            "- If something is not in the snapshot, say you don't know.\n"
            "- Keep answers concise. 1-3 sentences."
        )
        items = self._available(None)
        full  = ""
        for token in self._llm_stream(system, query, max_tokens=120):
            full += token
            yield {"type": "token", "content": token}
        if not full:
            full = "I'm not sure, let me check."
            yield {"type": "token", "content": full}
        nav_label, nav_target = self._extract_nav(full, items)
        yield {"type": "done",
               **self._resp(full, "query",
                            nav_label=nav_label, nav_target=nav_target)}

    def _resolve_need(self, query, user_id, session, category):
        session.need_type = category or ""
        excl      = session.excluded_items | self._skill_exclusions(user_id)
        available = self._available(category=category, excluded=excl)
        if not available and category:
            available = self._available(category=None, excluded=excl)
        item = self._pick_item(query, user_id, available, category)
        return item, available

    def _handle_need(self, query, user_id, session, category=None):
        item, available = self._resolve_need(query, user_id, session, category)

        if not item:
            return self._unavailable(query, user_id)

        session.pending_item    = item["label"]
        session.available_items = available
        obj        = self.db.dynamic_objects.find_one(
            {"label": item["label"].lower()})
        nav_label  = obj.get("last_seen_on") if obj else None
        nav_target = self._resolve_nav(nav_label)
        weight     = self._obs_weight(user_id, item["label"])
        skill_md   = self._skill_md(user_id) or ""

        if weight >= HIGH_WEIGHT:
            session.state = "IDLE"
            answer = self._llm_call(
                "You are a home robot. The user has a strong habit with "
                "this item. Tell them you will get it. One natural sentence.",
                f'User: "{query}"\nItem: {item["label"]}'
                + (f' at {nav_label}' if nav_label else ''),
                max_tokens=50) or f"I'll get your {item['label']} right away!"
            return self._resp(answer, "execute",
                              nav_label=nav_label, nav_target=nav_target,
                              recommendations=[{"label": item["label"]}],
                              is_personalized=True,
                              options=self._nav_options(nav_label, nav_target))

        session.state = "CONFIRMING"
        answer = self._llm_call(
            f"You are a home robot.\n"
            f"User preferences:\n{skill_md[:300]}\n\n"
            f"Offer ONLY '{item['label']}' naturally and ask for confirmation. "
            f"One sentence. Do NOT mention any other items.",
            f'User: "{query}"\nItem to offer: {item["label"]}'
            + (f' at {nav_label}' if nav_label else ''),
            max_tokens=60) or f"Would you like some {item['label']}?"
        return self._resp(answer, "need_confirm",
                          recommendations=[{"label": item["label"]}],
                          is_personalized=bool(
                              self._preference(user_id, available, category)),
                          options=[{"id": 1, "label": "Yes"},
                                   {"id": 2, "label": "No, something else"},
                                   {"id": 3, "label": "Cancel"}])

    def _need_stream(self, query, user_id, session, category=None):
        yield {"type": "token", "content": "Let me check... "}

        item, available = self._resolve_need(query, user_id, session, category)

        if not item:
            tone   = TIME_TONE.get(self._time_slot(), "friendly")
            system = (f"You are a home robot with a {tone} tone. "
                      "Nothing matches. Apologise warmly and suggest "
                      "an alternative (e.g. add to shopping list). "
                      "1-2 sentences.")
            full = ""
            for token in self._llm_stream(system, f'User: "{query}"',
                                          max_tokens=80):
                full += token
                yield {"type": "token", "content": token}
            if not full:
                full = "Sorry, I don't have anything for that right now."
                yield {"type": "token", "content": full}
            yield {"type": "done", **self._resp(full, "need_unavailable")}
            return

        session.pending_item    = item["label"]
        session.available_items = available
        obj        = self.db.dynamic_objects.find_one(
            {"label": item["label"].lower()})
        nav_label  = obj.get("last_seen_on") if obj else None
        nav_target = self._resolve_nav(nav_label)
        weight     = self._obs_weight(user_id, item["label"])
        skill_md   = self._skill_md(user_id) or ""

        if weight >= HIGH_WEIGHT:
            session.state = "IDLE"
            system  = ("You are a home robot. The user has a strong habit "
                       "with this item. Tell them you will get it. "
                       "One natural sentence.")
            context = (f'User: "{query}"\nItem: {item["label"]}'
                       + (f' at {nav_label}' if nav_label else ''))
            full = ""
            for token in self._llm_stream(system, context, max_tokens=50):
                full += token
                yield {"type": "token", "content": token}
            if not full:
                full = f"I'll get your {item['label']} right away!"
                yield {"type": "token", "content": full}
            yield {"type": "done",
                   **self._resp(full, "execute",
                                nav_label=nav_label, nav_target=nav_target,
                                recommendations=[{"label": item["label"]}],
                                is_personalized=True,
                                options=self._nav_options(nav_label, nav_target))}
            return

        session.state = "CONFIRMING"
        system  = (f"You are a home robot.\n"
                   f"User preferences:\n{skill_md[:300]}\n\n"
                   f"Offer ONLY '{item['label']}' naturally and ask for "
                   f"confirmation. One sentence. Do NOT mention other items.")
        context = (f'User: "{query}"\nItem to offer: {item["label"]}'
                   + (f' at {nav_label}' if nav_label else ''))
        full = ""
        for token in self._llm_stream(system, context, max_tokens=60):
            full += token
            yield {"type": "token", "content": token}
        if not full:
            full = f"Would you like some {item['label']}?"
            yield {"type": "token", "content": full}
        yield {"type": "done",
               **self._resp(full, "need_confirm",
                            recommendations=[{"label": item["label"]}],
                            is_personalized=bool(
                                self._preference(user_id, available, category)),
                            options=[{"id": 1, "label": "Yes"},
                                     {"id": 2, "label": "No, something else"},
                                     {"id": 3, "label": "Cancel"}])}

    def _handle_chat(self, query, user_id):
        session  = self._session(user_id)
        messages = (
            [{"role": "system", "content":
              "You are a friendly home robot companion. "
              "Reply warmly in 1-2 sentences. "
              "Do NOT promise to fetch anything unless asked. "
              "Do NOT wrap response in quotes."}]
            + session.turn_history[-MAX_HISTORY * 2:]
            + [{"role": "user", "content": query}]
        )
        answer = self._llm_messages(messages, max_tokens=80) or "I'm here for you!"
        return self._resp(answer, "chat")

    def _chat_stream(self, query, user_id):
        session  = self._session(user_id)
        messages = (
            [{"role": "system", "content":
              "You are a friendly home robot companion. "
              "Reply warmly in 1-2 sentences. "
              "Do NOT promise to fetch anything unless asked. "
              "Do NOT wrap response in quotes."}]
            + session.turn_history[-MAX_HISTORY * 2:]
            + [{"role": "user", "content": query}]
        )
        full = ""
        for token in self._llm_stream_messages(messages, max_tokens=80):
            full += token
            yield {"type": "token", "content": token}
        if not full:
            full = "I'm here for you!"
            yield {"type": "token", "content": full}
        yield {"type": "done", **self._resp(full, "chat")}

    def _confirmation_turn(self, query, session):
        session.add_turn("user", query)
        session.touch()
        q = query.lower().strip()

        if any(w in q for w in DISLIKE_WORDS): return self._dislike(session)
        if any(w in q for w in CONFIRM_WORDS): return self._confirm(session)
        if any(w in q for w in REJECT_WORDS):  return self._reject(session)

        raw = self._llm_call(
            "Reply with exactly one word: confirm, reject, dislike, or new.\n"
            "confirm = user agrees\n"
            "reject  = user wants something else\n"
            "dislike = user strongly dislikes this item\n"
            "new     = user is making a completely different request",
            f'Context: robot just offered an item. User replied: "{query}"',
            max_tokens=5)
        if raw:
            w = raw.strip().lower().split()[0]
            if w == "dislike": return self._dislike(session)
            if w == "confirm": return self._confirm(session)
            if w == "new":
                session.reset()
                intent, category = self._classify_intent(query)
                if intent == "need":
                    return self._handle_need(query, session.user_id, session, category)
                elif intent == "query":
                    return self._handle_query(query, session.user_id)
                else:
                    return self._handle_chat(query, session.user_id)
        return self._reject(session)

    def _confirm(self, session):
        item       = session.pending_item
        user_id    = session.user_id
        obj        = self.db.dynamic_objects.find_one({"label": item.lower()})
        nav_label  = obj.get("last_seen_on") if obj else None
        nav_target = self._resolve_nav(nav_label)
        self._update_skill_async(user_id, item, session.need_type, positive=True)
        session.reset()
        loc_str = f" from {nav_label}" if nav_label else ""
        return self._resp(
            f"Great! Getting you {item}{loc_str}.",
            "execute", nav_label=nav_label, nav_target=nav_target,
            recommendations=[{"label": item}], is_personalized=True,
            options=self._nav_options(nav_label, nav_target))

    def _reject(self, session):
        if session.pending_item:
            session.excluded_items.add(session.pending_item.lower())
        remaining = [i for i in session.available_items
                     if i["label"].lower() not in session.excluded_items]
        if not remaining:
            session.reset()
            return self._resp("Sorry, no other options available right now.",
                              "need_unavailable")
        nxt              = remaining[0]
        session.pending_item = nxt["label"]
        obj        = self.db.dynamic_objects.find_one(
            {"label": nxt["label"].lower()})
        nav_label  = obj.get("last_seen_on") if obj else None
        nav_target = self._resolve_nav(nav_label)
        loc_str    = f" at {nav_label}" if nav_label else ""
        return self._resp(
            f"How about {nxt['label']}{loc_str}?", "need_confirm",
            recommendations=[{"label": nxt["label"]}],
            nav_label=nav_label, nav_target=nav_target,
            options=[{"id": 1, "label": "Yes"},
                     {"id": 2, "label": "No, something else"},
                     {"id": 3, "label": "Cancel"}])

    def _dislike(self, session):
        item    = session.pending_item
        user_id = session.user_id
        if item:
            session.excluded_items.add(item.lower())
            self._update_skill_async(user_id, item, session.need_type, positive=False)
        remaining = [i for i in session.available_items
                     if i["label"].lower() not in session.excluded_items]
        if remaining:
            nxt              = remaining[0]
            session.pending_item = nxt["label"]
            obj        = self.db.dynamic_objects.find_one(
                {"label": nxt["label"].lower()})
            nav_label  = obj.get("last_seen_on") if obj else None
            loc_str    = f" at {nav_label}" if nav_label else ""
            return self._resp(
                f"Got it, I won't recommend {item} again. "
                f"Would you like {nxt['label']}{loc_str}?",
                "need_confirm",
                recommendations=[{"label": nxt["label"]}],
                is_personalized=True,
                options=[{"id": 1, "label": "Yes"},
                         {"id": 2, "label": "No, something else"},
                         {"id": 3, "label": "Cancel"}])
        session.reset()
        return self._resp(f"Got it, I won't recommend {item} again.",
                          "feedback", is_personalized=True)

    def _pick_item(self, query, user_id, available, need_type=None):
        if not available:
            return None
        preferred = self._preference(user_id, available, need_type)
        if preferred:
            for a in available:
                if a["label"].lower() == preferred.lower():
                    return a
        return self._semantic_match(query, available)

    def _semantic_match(self, query: str, items: list):
        if not items:
            return None
        if self._sbert:
            try:
                q_vec     = self._sbert.encode(query, normalize_embeddings=True)
                best_item = None
                best_score = -1.0
                for item in items:
                    ivec = self._get_item_vec(item)
                    if ivec is None:
                        continue
                    score = float(np.dot(q_vec, ivec))
                    if score > best_score:
                        best_score = score
                        best_item  = item
                if best_score >= SBERT_THRESHOLD and best_item:
                    return best_item
            except Exception:
                pass
        labels_text = "\n".join(
            f"- {i['label']} (category: {i.get('category','?')},"
            f" at {i.get('last_seen_on','?')})"
            for i in items)
        raw = self._llm_call(
            "Reply with ONLY the exact label from the list. "
            "Nothing else. No explanation.\n"
            "If nothing matches, reply: none\n\n"
            f"Items:\n{labels_text}",
            f'User wants: "{query}". Pick the best matching label.',
            max_tokens=10)
        if raw:
            raw_clean = raw.strip().lower().split()[0].rstrip(".,!?()")
            for i in items:
                if i["label"].lower() == raw_clean:
                    return i
            for i in items:
                if raw_clean in i["label"].lower():
                    return i
        return None

    def _get_item_vec(self, item):
        label    = item["label"].lower()
        category = item.get("category", "")
        synonyms = _ITEM_SYNONYMS.get(label, "")
        text     = f"{label} {category} {synonyms}".strip()
        cached   = self._vec_cache.get(label)
        if cached and cached[0] == text:
            return cached[1]
        if self._sbert is None:
            return None
        vec = self._sbert.encode(text, normalize_embeddings=True)
        self._vec_cache[label] = (text, vec)
        return vec

    def _warm_cache(self):
        if self._sbert is None:
            return
        def _bg():
            try:
                items = self._available(category=None)
                for item in items:
                    self._get_item_vec(item)
                print(f"[ReactiveService] SBERT cache ready: {len(items)} items")
            except Exception as e:
                print(f"[ReactiveService] warm cache error: {e}")
        threading.Thread(target=_bg, daemon=True).start()

    def _preference(self, user_id, available_items, need_type=None):
        available_labels = {i["label"].lower() for i in available_items}
        skill_md = self._skill_md(user_id)
        if skill_md:
            m = re.search(r"## Preferences\n(.*?)(?=\n## |$)",
                          skill_md, re.DOTALL)
            if m:
                for line in m.group(1).split("\n"):
                    if not line.strip():
                        continue
                    if need_type == "drink" and not any(
                            w in line.lower() for w in DRINK_KEYWORDS):
                        continue
                    if need_type == "food" and not any(
                            w in line.lower() for w in FOOD_KEYWORDS):
                        continue
                    if any(w in line.lower() for w in
                           ["enjoys", "likes", "frequently",
                            "drinks", "eats", "prefers"]):
                        for p in re.findall(r"\b[a-z]+\b", line):
                            if (p not in PREF_STOPWORDS
                                    and len(p) > 2
                                    and p in available_labels):
                                return p
        try:
            actions = (DRINK_ACTIONS if need_type == "drink"
                       else FOOD_ACTIONS if need_type == "food"
                       else DRINK_ACTIONS + FOOD_ACTIONS)
            obs = list(self.db.observation_logs.find(
                {"user": user_id, "action": {"$in": actions}},
                {"zone_name": 1, "weight": 1},
            ).sort("weight", -1).limit(5))
            for doc in obs:
                zone = doc.get("zone_name", "")
                obj  = self.db.dynamic_objects.find_one(
                    {"label":        {"$in": list(available_labels)},
                     "last_seen_on": zone},
                    sort=[("interact_count", -1)])
                if obj:
                    return obj["label"]
        except Exception:
            pass
        return None

    def _skill_exclusions(self, user_id):
        skill_md = self._skill_md(user_id)
        if not skill_md:
            return set()
        excluded = set()
        m = re.search(r"## What NOT to do\n(.*?)(?=\n## |$)",
                      skill_md, re.DOTALL)
        if m:
            for line in m.group(1).split("\n"):
                if "recommend" in line.lower():
                    for p in re.findall(r"\b[a-z]+\b", line):
                        if (p not in {"do", "not", "recommend", "to",
                                      "this", "user"} and len(p) > 2):
                            excluded.add(p.lower())
        return excluded

    def _skill_md(self, user_id) -> str | None:
        try:
            return self.skill_manager.get_skill(user_id)
        except Exception:
            return None

    def _update_skill_async(self, user_id, item, need_type, positive):
        def _bg():
            try:
                section = "## Preferences" if positive else "## What NOT to do"
                if positive:
                    action_word = "drinks" if need_type == "drink" else "eats"
                    bullet = f"- User enjoys {item} during {action_word.capitalize()}"
                else:
                    bullet = f"- Do not recommend {item} to this user"
                self.skill_manager._insert_if_new(user_id, section, bullet)
            except Exception as e:
                print(f"[ReactiveService] skill update error: {e}")
        threading.Thread(target=_bg, daemon=True).start()

    def _available(self, category=None, excluded=None):
        excluded = excluded or set()
        nin      = list(OBJECT_EXCLUDES | excluded)
        cutoff   = datetime.datetime.utcnow() - datetime.timedelta(hours=TTL_HOURS)
        q        = {"last_seen": {"$gte": cutoff}, "label": {"$nin": nin}}
        if category:
            q["category"] = category
        docs = list(self.db.dynamic_objects.find(
            q, {"label": 1, "category": 1, "last_seen_on": 1,
                "room": 1, "interact_count": 1},
        ).sort("interact_count", -1).limit(MAX_ITEMS))
        if not docs:
            q2 = {"label": {"$nin": nin}}
            if category:
                q2["category"] = category
            docs = list(self.db.dynamic_objects.find(
                q2, {"label": 1, "category": 1, "last_seen_on": 1,
                     "room": 1, "interact_count": 1},
            ).limit(MAX_ITEMS))
        return docs

    def _obs_weight(self, user_id, item_label):
        try:
            obj  = self.db.dynamic_objects.find_one(
                {"label": item_label.lower()}, {"last_seen_on": 1})
            zone = obj.get("last_seen_on", "") if obj else ""
            q    = {"user": user_id,
                    "action": {"$in": DRINK_ACTIONS + FOOD_ACTIONS}}
            if zone:
                q["zone_name"] = zone
            return sum(d.get("weight", 0) for d in
                       self.db.observation_logs.find(q, {"weight": 1}))
        except Exception:
            return 0.0

    def _unavailable(self, query, user_id):
        tone   = TIME_TONE.get(self._time_slot(), "friendly")
        system = (f"You are a home robot with a {tone} tone. "
                  "Nothing matches the user's request. "
                  "Apologise warmly and offer an alternative "
                  "(e.g. add to shopping list). 1-2 sentences.")
        answer = self._llm_call(system, f'User: "{query}"', max_tokens=80) \
                 or "I'm sorry, I don't have anything for that right now."
        return self._resp(answer, "need_unavailable")

    def _extract_nav(self, answer, items):
        al = answer.lower()
        for i in items:
            if i["label"].lower() in al:
                nav = i.get("last_seen_on")
                return nav, self._resolve_nav(nav)
        return None, None

    def _resolve_nav(self, label):
        if not label:
            return None
        doc = self.db.scene_snapshots.find_one(
            {"label": {"$regex": f"^{re.escape(label.strip())}$",
                       "$options": "i"}})
        return doc.get("pos") if doc else None

    def _nav_options(self, nav_label, nav_target):
        if nav_label and nav_target:
            return [{"id": 1, "label": f"Navigate to '{nav_label}'"},
                    {"id": 2, "label": "Just tell me the location"},
                    {"id": 3, "label": "Cancel"}]
        return [{"id": 3, "label": "Close"}]

    def _time_slot(self) -> str:
        h = datetime.datetime.now().hour
        if h < 10: return "Morning"
        if h < 13: return "Noon"
        if h < 18: return "Afternoon"
        if h < 22: return "Evening"
        return "Night"

    def _session(self, user_id):
        s = self._sessions.get(user_id)
        if s is None or s.expired():
            self._sessions[user_id] = ConversationSession(user_id=user_id)
        return self._sessions[user_id]

    def _llm_call(self, system, user, max_tokens=None):
        try:
            r = requests.post(
                f"{self.ollama_url}/api/chat",
                json={"model": self.llm_model,
                      "messages": [{"role": "system", "content": system},
                                   {"role": "user",   "content": user}],
                      "stream":  False,
                      "options": {"temperature": LLM_TEMP,
                                  "num_predict": max_tokens or LLM_TOKENS}},
                timeout=LLM_TIMEOUT)
            r.raise_for_status()
            return r.json()["message"]["content"].strip()
        except Exception as e:
            logger.error(f"[ReactiveService] LLM: {e}")
            return None

    def _llm_messages(self, messages, max_tokens=None):
        try:
            r = requests.post(
                f"{self.ollama_url}/api/chat",
                json={"model": self.llm_model, "messages": messages,
                      "stream":  False,
                      "options": {"temperature": LLM_TEMP,
                                  "num_predict": max_tokens or LLM_TOKENS}},
                timeout=LLM_TIMEOUT)
            r.raise_for_status()
            return r.json()["message"]["content"].strip()
        except Exception as e:
            logger.error(f"[ReactiveService] LLM messages: {e}")
            return None

    def _llm_stream(self, system, user, max_tokens=None):
        try:
            import json as _j
            r = requests.post(
                f"{self.ollama_url}/api/chat",
                json={"model": self.llm_model,
                      "messages": [{"role": "system", "content": system},
                                   {"role": "user",   "content": user}],
                      "stream":  True,
                      "options": {"temperature": LLM_TEMP,
                                  "num_predict": max_tokens or LLM_TOKENS}},
                stream=True, timeout=LLM_TIMEOUT)
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    chunk = _j.loads(line)
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if chunk.get("done"):
                        break
                except Exception:
                    continue
        except Exception as e:
            logger.error(f"[ReactiveService] stream: {e}")

    def _llm_stream_messages(self, messages, max_tokens=None):
        try:
            import json as _j
            r = requests.post(
                f"{self.ollama_url}/api/chat",
                json={"model": self.llm_model, "messages": messages,
                      "stream":  True,
                      "options": {"temperature": LLM_TEMP,
                                  "num_predict": max_tokens or LLM_TOKENS}},
                stream=True, timeout=LLM_TIMEOUT)
            r.raise_for_status()
            for line in r.iter_lines():
                if not line:
                    continue
                try:
                    chunk = _j.loads(line)
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        yield token
                    if chunk.get("done"):
                        break
                except Exception:
                    continue
        except Exception as e:
            logger.error(f"[ReactiveService] stream messages: {e}")

    def _wrap_stream(self, result):
        for char in result.get("answer", ""):
            yield {"type": "token", "content": char}
        yield {"type": "done", **result}

    def _resp(self, answer="", intent_type="chat", nav_label=None,
              nav_target=None, options=None, recommendations=None,
              is_personalized=False, confidence=0.85):
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