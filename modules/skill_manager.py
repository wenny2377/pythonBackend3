"""
SkillManager — Dynamic User Skill Profile
三層邊界 + LRU stale 標記（不刪除）
"""

import json, logging, re, requests
from datetime import datetime, timedelta
from pymongo import MongoClient

logger       = logging.getLogger(__name__)
LLM_TIMEOUT  = 60
LLM_TEMP     = 0.3
STALE_DAYS   = 30
MAX_SKILL_LEN = 1200   # 放寬限制，gemma3:4b 生成的內容較長
MAX_BULLETS  = 5

REQUIRED_SECTIONS = [
    "## Behavior Patterns",
    "## Preferences",
    "## How to Handle Requests",
    "## What NOT to do",
]
FORBIDDEN_KEYWORDS = [
    "weather","天氣","politics","religion",
    "feelings about","personal belief","opinion on",
]

GENERATE_SYSTEM = """You are a skill profile generator for a home service robot.
STRICT CONTENT RULES:
1. ONLY record: physical needs, location/object preferences, time patterns, service feedback
2. NEVER include: weather, greetings, small talk, unobserved assumptions
3. Format: max 5 bullets/section, start with verb/noun, leave comment if no data
4. Do NOT add sections beyond the 4 required ones
5. Output ONLY the filled Markdown, no explanations"""

UPDATE_SYSTEM = """You are a skill profile updater for a home service robot.
Update ONLY parts with EXPLICIT new info from conversation.
Keep exact 4-section structure. Max 5 bullets/section.
Output ONLY the complete updated Markdown."""

GAP_SYSTEM = 'Gap detector for home robot. JSON only: {"has_gap":true/false,"missing":"description"}'
RELEVANCE_SYSTEM = 'Relevance judge for home robot. JSON only: {"should_update":true/false,"reason":"one line"}'
FILL_GAP_SYSTEM = """Add ONE new rule to appropriate section. Keep all existing content.
Max 5 bullets/section. Output complete Markdown only."""


def _call_llm(ollama_url, model, system, user, max_tokens=600):
    try:
        resp = requests.post(f"{ollama_url}/api/chat", json={
            "model": model,
            "messages": [{"role":"system","content":system},{"role":"user","content":user}],
            "stream": False,
            "options": {"temperature": LLM_TEMP, "num_predict": max_tokens},
        }, timeout=LLM_TIMEOUT)
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()
    except Exception as e:
        logger.error(f"LLM failed: {e}")
        return None

def _call_llm_json(ollama_url, model, system, user):
    raw = _call_llm(ollama_url, model, system, user, max_tokens=150)
    if not raw: return None
    try:
        clean = re.sub(r'```(?:json)?\s*','',raw).strip().rstrip('`')
        m = re.search(r'\{.*\}', clean, re.DOTALL)
        if m: return json.loads(m.group(0))
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse failed: {e}")
    return None

def validate_skill(skill_md):
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
                idx = skill_md.find(other, start+1)
                if 0 < idx < end: end = idx
        bullets = [l for l in skill_md[start:end].split('\n') if l.strip().startswith('-')]
        if len(bullets) > MAX_BULLETS:
            return False, f"{s} has {len(bullets)} bullets"
    return True, "OK"

def _skill_template(user_id, version):
    return (f"# {user_id} Skill Profile\n"
            f"*Version {version} | Updated: {datetime.now().strftime('%Y-%m-%d')}*\n\n"
            f"## Behavior Patterns\n<!-- Observed: action + location + frequency only -->\n\n"
            f"## Preferences\n<!-- Confirmed likes/dislikes from direct user feedback only -->\n\n"
            f"## How to Handle Requests\n<!-- Step-by-step rules for specific request types -->\n\n"
            f"## What NOT to do\n<!-- Explicit rejections or corrections from user -->\n")


class SkillManager:
    def __init__(self, db_client=None, ollama_url="http://localhost:11434",
                 model_name="gemma3:4b", db_name="robot_rag_db"):
        self.db         = db_client[db_name] if db_client else \
                          MongoClient("mongodb://localhost:27017")[db_name]
        self.ollama_url = ollama_url
        self.model_name = model_name

    def get_skill(self, user_id):
        doc = self.db.user_skills.find_one({"user_id": user_id})
        if not doc: return None
        skill = doc["skill_md"]
        if doc.get("is_stale", False):
            skill = f"> ⚠️ 此技能規範已超過 {STALE_DAYS} 天未使用。\n\n" + skill
        return skill

    def get_version(self, user_id):
        doc = self.db.user_skills.find_one({"user_id": user_id})
        return doc.get("version", 0) if doc else 0

    def generate(self, user_id):
        habits = list(self.db.observation_logs.find(
            {"user": user_id},
            {"action":1,"instance":1,"weight":1,"interacting_items":1}
        ).sort("weight",-1).limit(10))
        recent = list(self.db.semantic_memories.find(
            {"user": user_id},
            {"action":1,"bound_to":1,"timestamp":1}
        ).sort("timestamp",-1).limit(5))

        habit_text  = "\n".join(
            f"- {h['action']} near {h['instance']} ({h['weight']} times, items: {h.get('interacting_items',[])})"
            for h in habits) or "None recorded yet."
        recent_text = "\n".join(
            f"- {r['action']} near {r.get('bound_to','?')}"
            for r in recent) or "None recorded yet."

        user_prompt = (
            f"Fill in this SKILL.md for user: {user_id}\n\n"
            f"Observed habits:\n{habit_text}\n\n"
            f"Recent observations:\n{recent_text}\n\n"
            f"Template:\n{_skill_template(user_id,1)}\n\n"
            f"Rules: only fill from observations above, max 5 bullets/section, no new sections"
        )
        skill_md = _call_llm(self.ollama_url, self.model_name, GENERATE_SYSTEM, user_prompt)
        if skill_md:
            valid, reason = validate_skill(skill_md)
            if valid:
                self._save(user_id, skill_md)
                logger.info(f"[SkillManager] Generated v1 for {user_id}")
                return skill_md
            logger.warning(f"[SkillManager] Generate failed: {reason}")
        fallback = self._fallback(user_id, habits)
        self._save(user_id, fallback)
        return fallback

    def detect_gap(self, user_id, query):
        doc = self.db.user_skills.find_one({"user_id": user_id})
        if not doc: return True, "no skill profile"
        result = _call_llm_json(
            self.ollama_url, self.model_name, GAP_SYSTEM,
            f'User request: "{query}"\n\nCurrent SKILL.md:\n{doc["skill_md"]}\n\n'
            f'Does SKILL.md have a rule for this? If NOT → has_gap: true'
        )
        if result is None: return False, ""
        return result.get("has_gap", False), result.get("missing", "")

    def fill_gap(self, user_id, query, missing):
        # ── Evidence Gating：行為需出現 >= 2 次才寫入技能 ──────────────────
        # 避免單次「雜訊行為」污染技能庫
        habits = list(self.db.observation_logs.find(
            {"user": user_id}, {"action":1,"instance":1,"weight":1}
        ).sort("weight",-1).limit(5))

        # 計算和 missing 相關的行為是否有足夠 support
        SUPPORT_THRESHOLD = 2.0
        has_support = any(
            h.get("weight", 0) >= SUPPORT_THRESHOLD
            for h in habits
        )
        if not has_support and habits:
            logger.info(f"[EvidenceGate] Skipping fill_gap for '{missing}' — insufficient support")
            doc = self.db.user_skills.find_one({"user_id": user_id})
            return doc["skill_md"] if doc else self._fallback(user_id, habits)

        doc      = self.db.user_skills.find_one({"user_id": user_id})
        skill_md = doc["skill_md"] if doc else self._fallback(user_id, [])
        habit_text = "\n".join(
            f"- {h['action']} near {h['instance']} ({h['weight']} times)"
            for h in habits) or "No habits."
        user_prompt = (
            f"Current SKILL.md:\n{skill_md}\n\n"
            f"New request: \"{query}\"\nMissing: {missing}\n"
            f"Known behaviors (support >= {SUPPORT_THRESHOLD}):\n{habit_text}\n\n"
            f"Add a rule for this request. Keep all existing content."
        )
        updated = _call_llm(self.ollama_url, self.model_name, FILL_GAP_SYSTEM, user_prompt)
        if updated:
            valid, reason = validate_skill(updated)
            if valid:
                self._save(user_id, updated)
                return updated
            logger.warning(f"[SkillManager] fill_gap validation failed: {reason}")
        return skill_md

    def should_update(self, user_id, query, answer, trace):
        if len(query.strip()) < 3 or not trace: return False
        result = _call_llm_json(
            self.ollama_url, self.model_name, RELEVANCE_SYSTEM,
            f'User said: "{query}"\nRobot answered: "{answer}"\n'
            f'Tools used: {[t["tool"] for t in trace]}\n\n'
            f'Update skill profile? True ONLY if explicit correction/preference/rejection.'
        )
        if result is None: return False
        should = result.get("should_update", False)
        logger.info(f"[RelevanceGate] {should} — {result.get('reason','')}")
        return should

    def update(self, user_id, query, answer, trace):
        doc = self.db.user_skills.find_one({"user_id": user_id})
        if not doc: return self.generate(user_id)
        current    = doc["skill_md"]
        trace_text = "\n".join(
            f"Step {t['step']}: {t['tool']} → {str(t['result'])[:80]}"
            for t in trace)
        user_prompt = (
            f"Current SKILL.md:\n{current}\n\n"
            f"Conversation:\nUser: \"{query}\"\nRobot: \"{answer}\"\n"
            f"Trace:\n{trace_text}\n\nUpdate SKILL.md."
        )
        updated = _call_llm(self.ollama_url, self.model_name, UPDATE_SYSTEM, user_prompt)
        if updated:
            valid, reason = validate_skill(updated)
            if valid:
                self._save(user_id, updated)
                return updated
            logger.warning(f"[SkillManager] Update failed: {reason}")
        return current

    def check_stale(self, user_id):
        doc = self.db.user_skills.find_one({"user_id": user_id})
        if not doc: return
        last_used = doc.get("last_used")
        is_stale  = (last_used and (datetime.utcnow() - last_used).days > STALE_DAYS)
        self.db.user_skills.update_one(
            {"user_id": user_id},
            {"$set": {"is_stale": is_stale}}
        )
        if is_stale:
            logger.info(f"[SkillManager] Marked stale for {user_id}")

    def nightly_refactor(self, user_id):
        doc = self.db.user_skills.find_one({"user_id": user_id})
        if not doc: return ""
        current = doc["skill_md"]
        refactored = _call_llm(
            self.ollama_url, self.model_name, UPDATE_SYSTEM,
            f"Refactor this SKILL.md:\n{current}\n\n"
            f"Merge duplicate/similar rules. Remove contradictions (keep newer).\n"
            f"Max 5 bullets/section. Same 4-section structure. Output Markdown only."
        )
        if refactored:
            valid, _ = validate_skill(refactored)
            if valid:
                self._save(user_id, refactored)
                return refactored
        return current

    def _save(self, user_id, skill_md):
        version = self.get_version(user_id) + 1
        self.db.user_skills.update_one(
            {"user_id": user_id},
            {"$set": {
                "skill_md":   skill_md,
                "version":    version,
                "updated_at": datetime.utcnow(),
                "last_used":  datetime.utcnow(),
                "is_stale":   False,
            }},
            upsert=True
        )

    def _fallback(self, user_id, habits):
        lines = "\n".join(
            f"- {h['action']} near {h['instance']} ({h['weight']} times)"
            for h in habits[:5]) or "- No habits recorded yet"
        return (
            f"# {user_id} Skill Profile\n"
            f"*Version 1 | Updated: {datetime.now().strftime('%Y-%m-%d')}*\n\n"
            f"## Behavior Patterns\n{lines}\n\n"
            f"## Preferences\n<!-- No confirmed preferences yet -->\n\n"
            f"## How to Handle Requests\n"
            f"- Search user habits first using search_habit tool\n"
            f"- Check object availability using search_object tool\n"
            f"- Prioritize highest-frequency behavior location\n"
            f"- Confirm with user before navigating\n\n"
            f"## What NOT to do\n"
            f"- Do not repeat proposals within 10 minutes\n"
            f"- Do not assume object locations without checking\n"
        )