import re
import json
import logging
import requests
from datetime import datetime, timedelta
from pymongo import MongoClient
from config import Config

logger = logging.getLogger(__name__)

LLM_TIMEOUT = Config.LLM_TIMEOUT
LLM_TEMP    = Config.LLM_TEMPERATURE
STALE_DAYS  = 30

MAX_SKILL_LEN = 2500
MAX_BULLETS   = 8

REQUIRED_SECTIONS = [
    "## Behavior Patterns",
    "## Preferences",
    "## How to Handle Requests",
    "## What NOT to do",
]

FORBIDDEN_KEYWORDS = [
    "weather", "politics", "religion",
    "feelings about", "personal belief", "opinion on",
]

DRINK_KEYWORDS = {"drink", "beverage", "thirst", "water", "juice", "soda", "cola", "bottle"}
FOOD_KEYWORDS  = {"eat", "food", "meal", "snack", "hungry", "fruit", "bowl", "plate"}

PREF_STOPWORDS = {
    "user", "enjoys", "drink", "food", "likes", "frequently", "uses",
    "during", "in", "the", "a", "an", "some", "often", "usually",
    "mom", "dad", "recommend", "not", "do", "to", "this",
}

GENERATE_SYSTEM = """You are a skill profile generator for a home service robot.
RULES:
1. ONLY record: physical needs, location/object preferences, time patterns, service feedback.
2. NEVER include: weather, greetings, small talk, assumptions not backed by observations.
3. Each section: max 8 bullets. Start each bullet with "- " (hyphen space). Do NOT use * or **.
4. If no data for a section, leave the section header only.
5. Output ONLY the filled Markdown. No explanations. No bold text."""

UPDATE_SYSTEM = """You are a skill profile updater for a home service robot.
Update ONLY the parts that have EXPLICIT new information from the conversation.
Keep the exact 4-section structure. Max 8 bullets per section.
Start each bullet with "- " (hyphen space). Do NOT use * or **.
Output ONLY the complete updated Markdown."""

RELEVANCE_SYSTEM = (
    'Relevance judge for home robot. '
    'JSON only: {"should_update":true/false,"reason":"one line"}\n'
    'should_update: true ONLY if the user expresses an explicit preference, '
    'correction, or rejection about FOOD, DRINK, or HOME OBJECTS. '
    'Ignore preferences about people, emotions, or topics unrelated to home objects.'
)

PREF_EXTRACT_SYSTEM = (
    "You are a preference extractor for a home robot. "
    "Extract ONE explicit preference or correction from the conversation. "
    "Output ONE bullet starting with '- '. "
    "Examples: "
    "'- User dislikes cola' "
    "'- User prefers juice over water' "
    "'- Do not recommend cola to this user' "
    "If no explicit preference or correction exists, output exactly: NONE"
)


def preferred_item_from_skill_md(skill_md: str, available_labels: set,
                                  need_type: str = None) -> str:
    if not skill_md:
        return ""

    m = re.search(r"## Preferences\n(.*?)(?=\n## |$)", skill_md, re.DOTALL)
    if not m:
        return ""

    for line in m.group(1).split("\n"):
        if not line.strip():
            continue
        line_lower = line.lower()
        if need_type == "drink" and not any(w in line_lower for w in DRINK_KEYWORDS):
            continue
        if need_type == "food" and not any(w in line_lower for w in FOOD_KEYWORDS):
            continue
        if any(w in line_lower for w in ["enjoys", "likes", "frequently", "drinks", "eats", "prefers"]):
            for p in re.findall(r"\b[a-zA-Z]+\b", line):
                p_lower = p.lower()
                if p_lower not in PREF_STOPWORDS and len(p_lower) > 2 and p_lower in available_labels:
                    return p_lower
    return ""


def _call_llm(ollama_url, model, system, user, max_tokens=600):
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
        logger.error(f"LLM failed: {e}")
        return None


def _call_llm_json(ollama_url, model, system, user):
    raw = _call_llm(ollama_url, model, system, user, max_tokens=150)
    if not raw:
        return None
    try:
        clean = re.sub(r'```(?:json)?\s*', '', raw).strip().rstrip('`')
        m     = re.search(r'\{.*\}', clean, re.DOTALL)
        if m:
            return json.loads(m.group(0))
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse failed: {e}")
    return None


def _normalize_bullets(skill_md: str) -> str:
    lines = []
    for line in skill_md.split('\n'):
        stripped = line.strip()
        if stripped.startswith('* ') or stripped.startswith('*\t'):
            line = line.replace('*', '-', 1)
        if re.match(r'^\s*\*\*.+\*\*\s*$', line):
            line = re.sub(r'\*\*', '', line)
        lines.append(line)
    return '\n'.join(lines)


def validate_skill(skill_md: str) -> tuple:
    if len(skill_md) > MAX_SKILL_LEN:
        return False, f"Too long ({len(skill_md)})"
    for s in REQUIRED_SECTIONS:
        if s not in skill_md:
            return False, f"Missing: {s}"
    lower = skill_md.lower()
    for kw in FORBIDDEN_KEYWORDS:
        if kw in lower:
            return False, f"Forbidden: '{kw}'"
    for s in REQUIRED_SECTIONS:
        start = skill_md.find(s)
        end   = len(skill_md)
        for other in REQUIRED_SECTIONS:
            if other != s:
                idx = skill_md.find(other, start + 1)
                if 0 < idx < end:
                    end = idx
        bullets = [l for l in skill_md[start:end].split('\n') if l.strip().startswith('-')]
        if len(bullets) > MAX_BULLETS:
            return False, f"{s} has {len(bullets)} bullets (max {MAX_BULLETS})"
    return True, "OK"


def _skill_template(user_id: str, version: int) -> str:
    return (
        f"# {user_id} Skill Profile\n"
        f"*Version {version} | Updated: {datetime.now().strftime('%Y-%m-%d')}*\n\n"
        f"## Behavior Patterns\n\n"
        f"## Preferences\n\n"
        f"## How to Handle Requests\n\n"
        f"## What NOT to do\n"
    )


def _section_bounds(skill_md: str, section: str) -> tuple:
    idx   = skill_md.find(section)
    after = skill_md[idx:]
    end   = len(after)
    for s in REQUIRED_SECTIONS:
        if s == section:
            continue
        i = after.find(s, len(section))
        if i != -1 and i < end:
            end = i
    return idx, idx + end


def _insert_bullet(skill_md: str, section: str, bullet: str) -> str:
    if section not in skill_md:
        return skill_md

    start, end = _section_bounds(skill_md, section)
    block      = skill_md[start:end]
    rest       = skill_md[end:]

    bullet = bullet.strip()
    if not bullet.startswith('-'):
        bullet = f"- {bullet}"

    existing = [l.strip() for l in block.split('\n') if l.strip().startswith('-')]
    if len(existing) >= MAX_BULLETS:
        return skill_md

    bullet_lower = bullet.lower()
    if any(bullet_lower == b.lower() for b in existing):
        return skill_md

    updated_block = block.rstrip() + f"\n{bullet}\n"
    return skill_md[:start] + updated_block + rest


def _is_duplicate(new_bullet: str, skill_md: str, section: str) -> bool:
    if section not in skill_md:
        return False
    start, end = _section_bounds(skill_md, section)
    block      = skill_md[start:end]
    bullets    = [l.strip() for l in block.split('\n') if l.strip().startswith('-')]
    new_low    = new_bullet.lower().strip()
    return any(new_low in b.lower() or b.lower() in new_low for b in bullets)


def _pattern_to_bullets(pattern: dict) -> tuple:
    action    = pattern.get("action", "")
    zone_name = pattern.get("zone_name", "")
    time_slot = pattern.get("time_slot", "")
    weight    = int(pattern.get("sample_count", 0))
    items     = pattern.get("common_items", [])

    item_str        = f" with {', '.join(items)}" if items else ""
    slot_str        = f" in {time_slot}" if time_slot and time_slot != "Unknown" else ""
    behavior_bullet = f"- {action} near {zone_name}{item_str}{slot_str} ({weight} times)"

    preference_bullets = [
        f"- User frequently uses {item} during {action}{slot_str}"
        for item in items
    ]
    return behavior_bullet, preference_bullets


class SkillManager:

    def __init__(self, db_client=None, ollama_url="http://localhost:11434",
                 model_name="llama3.1:8b", db_name="robot_rag_db"):
        self.db         = (db_client[db_name] if db_client
                           else MongoClient("mongodb://localhost:27017")[db_name])
        self.ollama_url = ollama_url
        self.model_name = model_name

    def sync_from_patterns(self, user_id: str, patterns: list) -> bool:
        doc = self.db.user_skills.find_one({"user_id": user_id})
        if not doc:
            self.generate(user_id)
            doc = self.db.user_skills.find_one({"user_id": user_id})
            if not doc:
                return False

        current = doc.get("skill_md", "")
        changed = False

        for pattern in patterns:
            behavior_bullet, preference_bullets = _pattern_to_bullets(pattern)

            if not _is_duplicate(behavior_bullet, current, "## Behavior Patterns"):
                candidate = _insert_bullet(current, "## Behavior Patterns", behavior_bullet)
                if candidate != current:
                    current = candidate
                    changed = True

            for pb in preference_bullets:
                if not _is_duplicate(pb, current, "## Preferences"):
                    candidate = _insert_bullet(current, "## Preferences", pb)
                    if candidate != current:
                        current = candidate
                        changed = True

        if not changed:
            return False

        current = _normalize_bullets(current)
        valid, reason = validate_skill(current)
        if not valid:
            logger.warning(f"[SkillManager] Invalid after pattern sync: {reason}")
            return False

        self._save(user_id, current)
        return True

    def _insert_if_new(self, user_id: str, section: str, bullet: str) -> bool:
        doc = self.db.user_skills.find_one({"user_id": user_id})
        if not doc:
            return False
        current = doc.get("skill_md", "")

        if _is_duplicate(bullet, current, section):
            return False

        updated = _insert_bullet(current, section, bullet)
        updated = _normalize_bullets(updated)
        valid, reason = validate_skill(updated)
        if not valid:
            logger.warning(f"[SkillManager] Invalid after insert: {reason}")
            return False

        self._save(user_id, updated)
        print(f"[SkillManager] Written to {section}: {bullet[:60]}")
        return True

    def get_skill(self, user_id: str) -> str | None:
        doc = self.db.user_skills.find_one({"user_id": user_id})
        if not doc:
            return None
        skill = doc["skill_md"]
        if doc.get("is_stale", False):
            skill = f"> Warning: skill profile not updated for over {STALE_DAYS} days.\n\n" + skill
        return skill

    def get_version(self, user_id: str) -> int:
        doc = self.db.user_skills.find_one({"user_id": user_id})
        return doc.get("version", 0) if doc else 0

    def generate(self, user_id: str) -> str:
        habits = list(self.db.observation_logs.find(
            {"user": user_id},
            {"action": 1, "instance": 1, "weight": 1, "interacting_items": 1},
        ).sort("weight", -1).limit(10))

        habit_text = "\n".join(
            f"- {h['action']} near {h['instance']} "
            f"({h['weight']} times, items: {h.get('interacting_items', [])})"
            for h in habits
        ) or "None recorded yet."

        user_prompt = (
            f"Fill in this SKILL.md for user: {user_id}\n\n"
            f"Observed habits:\n{habit_text}\n\n"
            f"Template:\n{_skill_template(user_id, 1)}\n\n"
            f"Rules: only fill from observations above, "
            f"max {MAX_BULLETS} bullets per section, no new sections."
        )

        skill_md = _call_llm(self.ollama_url, self.model_name, GENERATE_SYSTEM, user_prompt)
        if skill_md:
            skill_md = _normalize_bullets(skill_md)
            valid, _ = validate_skill(skill_md)
            if valid:
                self._save(user_id, skill_md)
                return skill_md

        fallback = self._fallback(user_id, habits)
        self._save(user_id, fallback)
        return fallback

    def should_update(self, user_id: str, query: str, answer: str, trace) -> bool:
        if len(query.strip()) < 3:
            return False
        result = _call_llm_json(
            self.ollama_url, self.model_name, RELEVANCE_SYSTEM,
            f'User said: "{query}"\nRobot answered: "{answer}"\n\nUpdate skill profile?',
        )
        if result is None:
            return False
        should = result.get("should_update", False)
        print(f"[RelevanceGate] {should} — {result.get('reason', '')}")
        return should

    def update(self, user_id: str, query: str, answer: str, trace) -> str:
        doc = self.db.user_skills.find_one({"user_id": user_id})
        if not doc:
            return self.generate(user_id)

        current = doc["skill_md"]

        new_bullet = _call_llm(
            self.ollama_url, self.model_name,
            PREF_EXTRACT_SYSTEM,
            f'User said: "{query}"\nRobot answered: "{answer}"',
            max_tokens=60,
        )

        if not new_bullet or new_bullet.strip().upper() == "NONE":
            return current

        new_bullet = new_bullet.strip()
        if not new_bullet.startswith('-'):
            new_bullet = f"- {new_bullet}"

        q_lower     = query.lower()
        has_dislike = any(w in q_lower for w in (
            "not like", "dislike", "don't like", "hate", "never", "stop", "no more"
        ))
        has_prefer  = any(w in q_lower for w in (
            "prefer", "love", "enjoy", "want", "like"
        ))

        updated = current
        if has_dislike and has_prefer:
            updated = _insert_bullet(updated, "## What NOT to do", new_bullet)
            updated = _insert_bullet(updated, "## Preferences",    new_bullet)
        elif has_dislike:
            updated = _insert_bullet(updated, "## What NOT to do", new_bullet)
        else:
            updated = _insert_bullet(updated, "## Preferences", new_bullet)

        updated = _normalize_bullets(updated)
        valid, _ = validate_skill(updated)
        if valid:
            self._save(user_id, updated)
            return updated
        return current

    def check_stale(self, user_id: str):
        doc = self.db.user_skills.find_one({"user_id": user_id})
        if not doc:
            return
        last_used = doc.get("last_used")
        is_stale  = bool(
            last_used
            and isinstance(last_used, datetime)
            and (datetime.utcnow() - last_used.replace(tzinfo=None)).days > STALE_DAYS
        )
        self.db.user_skills.update_one(
            {"user_id": user_id},
            {"$set": {"is_stale": is_stale}},
        )

    def nightly_refactor(self, user_id: str) -> str:
        doc = self.db.user_skills.find_one({"user_id": user_id})
        if not doc:
            return ""
        current    = doc["skill_md"]
        refactored = _call_llm(
            self.ollama_url, self.model_name, UPDATE_SYSTEM,
            f"Refactor this SKILL.md:\n{current}\n\n"
            f"Merge duplicate or similar rules. Remove contradictions (keep newer).\n"
            f"Max {MAX_BULLETS} bullets per section. Keep exact 4-section structure. "
            f"Output Markdown only.",
        )
        if refactored:
            refactored = _normalize_bullets(refactored)
            valid, _   = validate_skill(refactored)
            if valid:
                self._save(user_id, refactored)
                return refactored
        return current

    def _save(self, user_id: str, skill_md: str):
        skill_md = _normalize_bullets(skill_md)
        version  = self.get_version(user_id) + 1
        self.db.user_skills.update_one(
            {"user_id": user_id},
            {"$set": {
                "skill_md":   skill_md,
                "version":    version,
                "updated_at": datetime.utcnow(),
                "last_used":  datetime.utcnow(),
                "is_stale":   False,
            }},
            upsert=True,
        )

    def _fallback(self, user_id: str, habits: list) -> str:
        lines = "\n".join(
            f"- {h['action']} near {h['instance']} ({h['weight']} times)"
            for h in habits[:5]
        ) or "- No habits recorded yet"
        return (
            f"# {user_id} Skill Profile\n"
            f"*Version 1 | Updated: {datetime.now().strftime('%Y-%m-%d')}*\n\n"
            f"## Behavior Patterns\n{lines}\n\n"
            f"## Preferences\n\n"
            f"## How to Handle Requests\n"
            f"- Check object availability before recommending\n"
            f"- If requested item is unavailable, suggest nearest alternative\n\n"
            f"## What NOT to do\n"
            f"- Do not invent object locations\n"
            f"- Do not recommend items not in the environment snapshot\n"
        )