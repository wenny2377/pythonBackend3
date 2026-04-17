import json
import logging
import re
import requests
from datetime import datetime, timedelta
from pymongo import MongoClient
from config import Config

logger = logging.getLogger(__name__)

LLM_TIMEOUT = Config.LLM_TIMEOUT
LLM_TEMP    = Config.LLM_TEMPERATURE
STALE_DAYS  = getattr(Config, 'SKILL_STALE_DAYS', 30)

MAX_SKILL_LEN         = 2500
MAX_BULLETS           = 8
SKILL_CHUNK_TOP_K     = 2
SBERT_DEDUP_THRESHOLD = 0.85
SUPPORT_THRESHOLD     = 2

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

GENERATE_SYSTEM = """You are a skill profile generator for a home service robot.
RULES:
1. ONLY record: physical needs, location/object preferences, time patterns, service feedback.
2. NEVER include: weather, greetings, small talk, assumptions not backed by observations.
3. Each section: max 8 bullets. Start each bullet with "- " (hyphen space). Do NOT use * or **.
4. If no data for a section, leave the HTML comment line only.
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
    'correction, or rejection about FOOD, DRINK, or HOME OBJECTS '
    '(e.g. juice, cola, milk, remote, sofa). '
    'Ignore preferences about people, emotions, relationships, or topics '
    'unrelated to home objects and services.'
)


def _call_llm(ollama_url, model, system, user, max_tokens=600):
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
        logger.error(f"LLM failed: {e}")
        return None


def _call_llm_json(ollama_url, model, system, user):
    raw = _call_llm(ollama_url, model, system, user, max_tokens=150)
    if not raw:
        return None
    try:
        clean = re.sub(r'```(?:json)?\s*', '', raw).strip().rstrip('`')
        m = re.search(r'\{.*\}', clean, re.DOTALL)
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
                idx = skill_md.find(other, start + 1)
                if 0 < idx < end:
                    end = idx
        bullets = [
            l for l in skill_md[start:end].split('\n')
            if l.strip().startswith('-')
        ]
        if len(bullets) > MAX_BULLETS:
            return False, f"{s} has {len(bullets)} bullets (max {MAX_BULLETS})"
    return True, "OK"


def _skill_template(user_id, version):
    return (
        f"# {user_id} Skill Profile\n"
        f"*Version {version} | Updated: {datetime.now().strftime('%Y-%m-%d')}*\n\n"
        f"## Behavior Patterns\n"
        f"<!-- Observed: action + location + frequency only -->\n\n"
        f"## Preferences\n"
        f"<!-- Confirmed likes/dislikes from direct user feedback only -->\n\n"
        f"## How to Handle Requests\n"
        f"<!-- Step-by-step rules for specific request types -->\n\n"
        f"## What NOT to do\n"
        f"<!-- Explicit rejections or corrections from user -->\n"
    )


def _insert_bullet(skill_md: str, section: str, bullet: str) -> str:
    if section not in skill_md:
        return skill_md

    idx   = skill_md.find(section)
    after = skill_md[idx:]

    next_section_idx = len(after)
    for s in REQUIRED_SECTIONS:
        if s == section:
            continue
        i = after.find(s, len(section))
        if i != -1 and i < next_section_idx:
            next_section_idx = i

    block  = after[:next_section_idx]
    rest   = after[next_section_idx:]
    bullet = bullet.strip()
    if not bullet.startswith('-'):
        bullet = f"- {bullet}"

    existing_bullets = [l.strip() for l in block.split('\n') if l.strip().startswith('-')]
    if len(existing_bullets) >= MAX_BULLETS:
        logger.info(f"[SkillManager] Section '{section}' at max bullets, skipping insert")
        return skill_md

    bullet_lower = bullet.lower()
    if any(bullet_lower == b.lower() for b in existing_bullets):
        print(f"[insert] Duplicate bullet skipped: {bullet}", flush=True)
        return skill_md

    updated_block = block.rstrip() + f"\n{bullet}\n"
    return skill_md[:idx] + updated_block + rest


class SkillManager:

    def __init__(self, db_client=None, ollama_url="http://localhost:11434",
                 model_name="llama3.1:8b-instruct-q4_K_M", db_name="robot_rag_db"):
        self.db         = db_client[db_name] if db_client else \
                          MongoClient("mongodb://localhost:27017")[db_name]
        self.ollama_url = ollama_url
        self.model_name = model_name

        self._sbert             = None
        self._skill_chunk_index = None
        self._skill_chunk_meta  = []
        self._init_skill_faiss()

    def _init_skill_faiss(self):
        try:
            import faiss
            import numpy as np
            from sentence_transformers import SentenceTransformer

            self._sbert             = SentenceTransformer('paraphrase-MiniLM-L6-v2', device='cuda')
            self._skill_chunk_index = faiss.IndexFlatIP(384)

            chunks = list(self.db.skill_chunks.find({}))
            if chunks:
                for chunk in chunks:
                    if "vector" in chunk:
                        vec = np.array(chunk["vector"], dtype=np.float32).reshape(1, -1)
                        faiss.normalize_L2(vec)
                        self._skill_chunk_index.add(vec)
                        self._skill_chunk_meta.append(chunk)
                logger.info(f"[SkillChunk] Loaded {len(chunks)} chunks from MongoDB")
        except Exception as e:
            logger.warning(f"[SkillChunk] FAISS init failed: {e}")

    def _chunk_skill_md(self, skill_md: str, user_id: str) -> list:
        if not self._sbert or not skill_md:
            return []

        import faiss
        import numpy as np

        chunks   = []
        sections = re.split(r'\n(?=## )', skill_md)

        for section in sections:
            section = section.strip()
            if not section or len(section) < 10:
                continue

            title_match = re.match(r'## (.+)', section)
            title       = title_match.group(1).strip() if title_match else "general"
            import numpy as np
            vec = self._sbert.encode(section, normalize_embeddings=True).astype(np.float32)

            chunk_doc = {
                "user_id":    user_id,
                "title":      title,
                "content":    section,
                "vector":     vec.tolist(),
                "updated_at": datetime.utcnow(),
            }

            existing = self._find_similar_chunk(user_id, vec)
            if existing:
                self.db.skill_chunks.update_one(
                    {"_id": existing["_id"]},
                    {
                        "$inc": {"support": 1},
                        "$set": {"content": section, "updated_at": datetime.utcnow()},
                    },
                )
            else:
                chunk_doc["support"] = 1
                result               = self.db.skill_chunks.insert_one(chunk_doc)
                chunk_doc["_id"]     = result.inserted_id

                vec_norm = vec.reshape(1, -1).copy()
                faiss.normalize_L2(vec_norm)
                self._skill_chunk_index.add(vec_norm)
                self._skill_chunk_meta.append(chunk_doc)

            chunks.append(chunk_doc)

        return chunks

    def _find_similar_chunk(self, user_id: str, vec) -> dict | None:
        if not self._skill_chunk_index or self._skill_chunk_index.ntotal == 0:
            return None
        try:
            import numpy as np
            import faiss
            q = vec.reshape(1, -1).copy().astype(np.float32)
            faiss.normalize_L2(q)
            scores, indices = self._skill_chunk_index.search(q, 1)
            if scores[0][0] >= SBERT_DEDUP_THRESHOLD:
                idx = indices[0][0]
                if idx < len(self._skill_chunk_meta):
                    candidate = self._skill_chunk_meta[idx]
                    if candidate.get("user_id") == user_id:
                        return candidate
        except Exception as e:
            logger.warning(f"[SkillChunk] similarity search failed: {e}")
        return None

    def get_skill_chunks(self, user_id: str, query: str) -> str | None:
        if not self._sbert or not self._skill_chunk_index:
            return None
        if self._skill_chunk_index.ntotal == 0:
            return None

        user_chunks = [
            (i, m) for i, m in enumerate(self._skill_chunk_meta)
            if m.get("user_id") == user_id and m.get("support", 1) >= SUPPORT_THRESHOLD
        ]
        if not user_chunks:
            user_chunks = [
                (i, m) for i, m in enumerate(self._skill_chunk_meta)
                if m.get("user_id") == user_id
            ]
        if not user_chunks:
            return None

        try:
            import numpy as np
            import faiss

            q_vec = self._sbert.encode(
                query, normalize_embeddings=True
            ).astype(np.float32).reshape(1, -1)
            faiss.normalize_L2(q_vec)

            scored = []
            for _, meta in user_chunks:
                chunk_vec = np.array(
                    meta["vector"], dtype=np.float32
                ).reshape(1, -1)
                faiss.normalize_L2(chunk_vec)
                sim = float(np.dot(q_vec[0], chunk_vec[0]))
                scored.append((sim, meta))

            scored.sort(key=lambda x: x[0], reverse=True)
            top = scored[:SKILL_CHUNK_TOP_K]

            if not top:
                return None

            result         = "\n\n".join(m["content"] for _, m in top)
            token_estimate = len(result.split()) * 1.3
            logger.info(f"[SkillChunk] Top-{SKILL_CHUNK_TOP_K}: ~{token_estimate:.0f} tokens")
            return result

        except Exception as e:
            logger.warning(f"[SkillChunk] retrieval failed: {e}")
            return None

    def get_skill(self, user_id):
        doc = self.db.user_skills.find_one({"user_id": user_id})
        if not doc:
            return None
        skill = doc["skill_md"]
        if doc.get("is_stale", False):
            skill = f"> Warning: skill profile not used for over {STALE_DAYS} days.\n\n" + skill
        return skill

    def get_version(self, user_id):
        doc = self.db.user_skills.find_one({"user_id": user_id})
        return doc.get("version", 0) if doc else 0

    def generate(self, user_id):
        habits = list(self.db.observation_logs.find(
            {"user": user_id},
            {"action": 1, "instance": 1, "weight": 1, "interacting_items": 1},
        ).sort("weight", -1).limit(10))

        recent = list(self.db.semantic_memories.find(
            {"user": user_id},
            {"action": 1, "bound_to": 1, "timestamp": 1},
        ).sort("timestamp", -1).limit(5))

        habit_text = "\n".join(
            f"- {h['action']} near {h['instance']} "
            f"({h['weight']} times, items: {h.get('interacting_items', [])})"
            for h in habits
        ) or "None recorded yet."

        recent_text = "\n".join(
            f"- {r['action']} near {r.get('bound_to', '?')}"
            for r in recent
        ) or "None recorded yet."

        user_prompt = (
            f"Fill in this SKILL.md for user: {user_id}\n\n"
            f"Observed habits:\n{habit_text}\n\n"
            f"Recent observations:\n{recent_text}\n\n"
            f"Template:\n{_skill_template(user_id, 1)}\n\n"
            f"Rules: only fill from observations above, "
            f"max {MAX_BULLETS} bullets per section, no new sections."
        )

        skill_md = _call_llm(self.ollama_url, self.model_name, GENERATE_SYSTEM, user_prompt)
        if skill_md:
            skill_md = _normalize_bullets(skill_md)
            valid, reason = validate_skill(skill_md)
            if valid:
                self._save(user_id, skill_md)
                self._chunk_skill_md(skill_md, user_id)
                logger.info(f"[SkillManager] Generated v1 for {user_id}")
                return skill_md
            logger.warning(f"[SkillManager] Generate failed: {reason}")

        fallback = self._fallback(user_id, habits)
        self._save(user_id, fallback)
        self._chunk_skill_md(fallback, user_id)
        return fallback

    def should_update(self, user_id, query, answer, trace):
        if len(query.strip()) < 3:
            return False
        result = _call_llm_json(
            self.ollama_url, self.model_name, RELEVANCE_SYSTEM,
            f'User said: "{query}"\n'
            f'Robot answered: "{answer}"\n\n'
            f'Update skill profile? True ONLY if explicit correction, preference, or rejection.',
        )
        if result is None:
            return False
        should = result.get("should_update", False)
        print(f"[RelevanceGate] {should} — {result.get('reason', '')}", flush=True)
        return should

    def update(self, user_id, query, answer, trace):
        doc = self.db.user_skills.find_one({"user_id": user_id})
        if not doc:
            return self.generate(user_id)

        current = doc["skill_md"]

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

        new_bullet = _call_llm(
            self.ollama_url, self.model_name,
            PREF_EXTRACT_SYSTEM,
            f'User said: "{query}"\nRobot answered: "{answer}"',
            max_tokens=60,
        )

        if not new_bullet or new_bullet.strip().upper() == "NONE":
            print("[update] No explicit preference found, skipping", flush=True)
            return current

        new_bullet = new_bullet.strip()
        if not new_bullet.startswith('-'):
            new_bullet = f"- {new_bullet}"

        q_lower    = query.lower()
        has_dislike = any(w in q_lower for w in (
            "not like", "dislike", "don't like", "hate", "never", "stop", "no more"
        ))
        has_prefer  = any(w in q_lower for w in (
            "prefer", "love", "enjoy", "want", "like"
        ))

        updated = current
        if has_dislike and has_prefer:
            dislike_bullet = f"- Do not recommend cola" if "cola" in q_lower else new_bullet
            prefer_bullet  = new_bullet
            updated = _insert_bullet(updated, "## What NOT to do", dislike_bullet)
            updated = _insert_bullet(updated, "## Preferences",    prefer_bullet)
        elif has_dislike:
            updated = _insert_bullet(updated, "## What NOT to do", new_bullet)
            section = "## What NOT to do"
        else:
            updated = _insert_bullet(updated, "## Preferences", new_bullet)
            section = "## Preferences"

        updated = _normalize_bullets(updated)

        valid, reason = validate_skill(updated)
        if valid:
            self._save(user_id, updated)
            self._chunk_skill_md(updated, user_id)
            if has_dislike and has_prefer:
                print(f"[update] Bullets added to both Preferences and What NOT to do: {new_bullet}", flush=True)
            else:
                print(f"[update] Bullet added to {section}: {new_bullet}", flush=True)
            return updated

        print(f"[update] validate failed: {reason}", flush=True)
        return current

    def check_stale(self, user_id):
        doc = self.db.user_skills.find_one({"user_id": user_id})
        if not doc:
            return
        last_used = doc.get("last_used")
        is_stale  = bool(last_used and (datetime.utcnow() - last_used).days > STALE_DAYS)
        self.db.user_skills.update_one(
            {"user_id": user_id},
            {"$set": {"is_stale": is_stale}},
        )
        if is_stale:
            logger.info(f"[SkillManager] Marked stale for {user_id}")

    def nightly_refactor(self, user_id):
        doc = self.db.user_skills.find_one({"user_id": user_id})
        if not doc:
            return ""
        current = doc["skill_md"]
        refactored = _call_llm(
            self.ollama_url, self.model_name, UPDATE_SYSTEM,
            f"Refactor this SKILL.md:\n{current}\n\n"
            f"Merge duplicate or similar rules. Remove contradictions (keep newer).\n"
            f"Max {MAX_BULLETS} bullets per section. Keep exact 4-section structure. "
            f"Output Markdown only.",
        )
        if refactored:
            refactored = _normalize_bullets(refactored)
            valid, _ = validate_skill(refactored)
            if valid:
                self._save(user_id, refactored)
                self._chunk_skill_md(refactored, user_id)
                return refactored
        return current

    def _save(self, user_id, skill_md):
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

    def _fallback(self, user_id, habits):
        lines = "\n".join(
            f"- {h['action']} near {h['instance']} ({h['weight']} times)"
            for h in habits[:5]
        ) or "- No habits recorded yet"
        return (
            f"# {user_id} Skill Profile\n"
            f"*Version 1 | Updated: {datetime.now().strftime('%Y-%m-%d')}*\n\n"
            f"## Behavior Patterns\n{lines}\n\n"
            f"## Preferences\n<!-- No confirmed preferences yet -->\n\n"
            f"## How to Handle Requests\n"
            f"- Check object availability before recommending\n"
            f"- If requested item is unavailable, suggest nearest alternative\n\n"
            f"## What NOT to do\n"
            f"- Do not invent object locations\n"
            f"- Do not recommend items not in the environment snapshot\n"
        )