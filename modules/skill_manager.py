import json, logging, re, requests
from datetime import datetime, timedelta
from pymongo import MongoClient
from config import Config
logger       = logging.getLogger(__name__)

LLM_TIMEOUT = Config.LLM_TIMEOUT
LLM_TEMP    = Config.LLM_TEMPERATURE
STALE_DAYS  = getattr(Config, 'SKILL_STALE_DAYS', 30)

MAX_SKILL_LEN = 1200
MAX_BULLETS  = 5

# ── FAISS 技能切片設定（論文 §4.2 Context-Aware Dynamic Pruning）────────────
SKILL_CHUNK_TOP_K  = 2        # ReAct 時只注入 Top-2 相關技能塊
SKILL_CHUNK_TARGET = 400      # 目標 token 數（壓縮後）
SBERT_DEDUP_THRESHOLD = 0.85  # 語義去重門檻（論文 §4.1）
SUPPORT_THRESHOLD = 2         # 升層最低支持次數（論文 §4.1）

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

# fill_gap 系統提示（強調從 Episodic Summary 生成替代方案，論文 §4.3）
FILL_GAP_SYSTEM = """You are a skill rule generator for a home service robot.
A service task FAILED because an item is unavailable or missing.
Your job:
1. Look at the episodic summary to find what the user chose LAST TIME in a similar situation
2. Generate ONE new rule for the appropriate section
3. Keep all existing content — only ADD, never remove

Rule format (JSON embedded in Markdown):
- IF [trigger condition] AND [availability check] THEN [action/recommendation]
  {"trigger":"...","alternative":"...","condition":"...","confidence":0.85,"source":"episodic_summary"}

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

        # SBERT（重用 VectorMemory 的模型，不重複佔 VRAM）
        self._sbert = None
        self._skill_chunk_index = None   # FAISS index for skill chunks
        self._skill_chunk_meta  = []     # 對應 metadata
        self._init_skill_faiss()

    # ── FAISS 技能切片初始化（論文 §4.2）────────────────────────────────────
    def _init_skill_faiss(self):
        """建立或載入技能塊 FAISS Index"""
        try:
            import faiss
            import numpy as np
            from sentence_transformers import SentenceTransformer

            # 嘗試重用已有的 SBERT（從 VectorMemory）
            self._sbert = SentenceTransformer('paraphrase-MiniLM-L6-v2', device='cuda')
            self._skill_chunk_index = faiss.IndexFlatIP(384)

            # 從 MongoDB 載入現有技能塊
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
        """
        將 SKILL.md 按 ## 標題切割為獨立塊（論文 §4.2）
        每塊存入 MongoDB + FAISS Index
        """
        if not self._sbert or not skill_md:
            return []

        import faiss
        import numpy as np

        chunks = []
        sections = re.split(r'\n(?=## )', skill_md)
        for section in sections:
            section = section.strip()
            if not section or len(section) < 10:
                continue

            # 取標題行作為 chunk key
            title_match = re.match(r'## (.+)', section)
            title = title_match.group(1).strip() if title_match else "general"

            # 計算向量
            vec = self._sbert.encode(section, normalize_embeddings=True).astype(np.float32)

            chunk_doc = {
                "user_id":   user_id,
                "title":     title,
                "content":   section,
                "vector":    vec.tolist(),
                "updated_at": datetime.utcnow(),
            }

            # 語義去重（SBERT 相似度 > 0.85 則合併，論文 §4.1）
            existing = self._find_similar_chunk(user_id, vec)
            if existing:
                # 更新 support 計數
                self.db.skill_chunks.update_one(
                    {"_id": existing["_id"]},
                    {"$inc": {"support": 1}, "$set": {"content": section, "updated_at": datetime.utcnow()}}
                )
                logger.debug(f"[SkillChunk] Merged similar chunk: {title}")
            else:
                chunk_doc["support"] = 1
                result = self.db.skill_chunks.insert_one(chunk_doc)
                chunk_doc["_id"] = result.inserted_id

                # 加入 FAISS
                vec_norm = vec.reshape(1, -1).copy()
                faiss.normalize_L2(vec_norm)
                self._skill_chunk_index.add(vec_norm)
                self._skill_chunk_meta.append(chunk_doc)

            chunks.append(chunk_doc)

        logger.info(f"[SkillChunk] {len(chunks)} chunks indexed for {user_id}")
        return chunks

    def _find_similar_chunk(self, user_id: str, vec) -> dict | None:
        """找出相似度 > 0.85 的已有技能塊"""
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
        """
        FAISS 技能切片：根據 query 檢索 Top-2 相關技能塊（論文 §4.2）
        目標：把 1300+ tokens 壓縮到 < 400 tokens
        """
        if not self._sbert or not self._skill_chunk_index:
            return None
        if self._skill_chunk_index.ntotal == 0:
            return None

        # 只取該 user 的 chunks，且 support >= SUPPORT_THRESHOLD
        user_chunks = [
            (i, m) for i, m in enumerate(self._skill_chunk_meta)
            if m.get("user_id") == user_id and m.get("support", 1) >= SUPPORT_THRESHOLD
        ]
        if not user_chunks:
            # support 未達門檻，用全部
            user_chunks = [
                (i, m) for i, m in enumerate(self._skill_chunk_meta)
                if m.get("user_id") == user_id
            ]
        if not user_chunks:
            return None

        try:
            import numpy as np
            import faiss

            q_vec = self._sbert.encode(query, normalize_embeddings=True).astype(np.float32)
            q_vec = q_vec.reshape(1, -1)
            faiss.normalize_L2(q_vec)

            # 在 user 的 chunks 裡找最相關的
            scored = []
            for faiss_idx, meta in user_chunks:
                if faiss_idx < len(self._skill_chunk_meta):
                    chunk_vec = np.array(meta["vector"], dtype=np.float32).reshape(1, -1)
                    faiss.normalize_L2(chunk_vec)
                    sim = float(np.dot(q_vec[0], chunk_vec[0]))
                    scored.append((sim, meta))

            scored.sort(key=lambda x: x[0], reverse=True)
            top_chunks = scored[:SKILL_CHUNK_TOP_K]

            if not top_chunks:
                return None

            result = "\n\n".join(m["content"] for _, m in top_chunks)
            token_estimate = len(result.split()) * 1.3  # 粗估 token 數
            logger.info(f"[SkillChunk] Top-{SKILL_CHUNK_TOP_K} chunks: ~{token_estimate:.0f} tokens "
                        f"(full SKILL.md would be 1300+)")
            return result

        except Exception as e:
            logger.warning(f"[SkillChunk] retrieval failed: {e}")
            return None

    # ── 基本 CRUD ─────────────────────────────────────────────────────────────
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
                # 同步到 FAISS 技能切片（論文 §4.2）
                self._chunk_skill_md(skill_md, user_id)
                logger.info(f"[SkillManager] Generated v1 for {user_id}")
                return skill_md
            logger.warning(f"[SkillManager] Generate failed: {reason}")
        fallback = self._fallback(user_id, habits)
        self._save(user_id, fallback)
        self._chunk_skill_md(fallback, user_id)
        return fallback

    # ── Gap 偵測 ──────────────────────────────────────────────────────────────
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

    # ── fill_gap：核心自演化機制（論文 §4.3 Generative Skill Synthesis）───────
    def fill_gap(self, user_id: str, query: str, missing: str) -> str:
        """
        任務失敗時自動生成替代方案規則。

        流程（論文 §4.3 咖啡→茶場景）：
        1. Evidence Gating：行為需出現 >= 2 次
        2. 查詢 Episodic Summary（Layer 2）找歷史替代選擇
        3. 確認替代品目前在家（動態物件）
        4. AI 生成新規則 → 寫入 SKILL.md + FAISS Index
        """
        # ── Step 1：Evidence Gating ──
        habits = list(self.db.observation_logs.find(
            {"user": user_id}, {"action":1,"instance":1,"weight":1,"interacting_items":1}
        ).sort("weight",-1).limit(5))

        has_support = any(h.get("weight", 0) >= SUPPORT_THRESHOLD for h in habits)
        if not has_support and habits:
            logger.info(f"[EvidenceGate] Skipping fill_gap for '{missing}' — insufficient support")
            doc = self.db.user_skills.find_one({"user_id": user_id})
            return doc["skill_md"] if doc else self._fallback(user_id, habits)

        doc      = self.db.user_skills.find_one({"user_id": user_id})
        skill_md = doc["skill_md"] if doc else self._fallback(user_id, [])

        # ── Step 2：查詢 Episodic Summary（Layer 2）──
        # 找使用者上次遇到相同缺少情況時的選擇
        episodic_summary = self._get_episodic_alternative(user_id, missing)

        # ── Step 3：確認替代品目前在家 ──
        available_alternative = self._check_alternative_available(
            episodic_summary.get("alternative", "")
        )

        # ── Step 4：AI 生成規則 ──
        habit_text = "\n".join(
            f"- {h['action']} near {h['instance']} ({h['weight']} times)"
            for h in habits) or "No habits."

        episodic_text = ""
        if episodic_summary:
            episodic_text = (
                f"\nEpisodic Summary (user's past behavior when '{missing}' was unavailable):\n"
                f"- Last time user chose: {episodic_summary.get('alternative','unknown')}\n"
                f"- Frequency: {episodic_summary.get('count', 0)} times\n"
                f"- Currently available: {available_alternative or 'unknown'}\n"
            )

        user_prompt = (
            f"Current SKILL.md:\n{skill_md}\n\n"
            f"Service task FAILED. Details:\n"
            f"- User request: \"{query}\"\n"
            f"- Missing item/capability: {missing}\n"
            f"{episodic_text}\n"
            f"Known behaviors (support >= {SUPPORT_THRESHOLD}):\n{habit_text}\n\n"
            f"Generate ONE new rule that handles this failure case.\n"
            f"If episodic summary shows a preferred alternative, encode that preference.\n"
            f"Example rule format:\n"
            f"- IF 咖啡缺失 AND 茶存在 THEN 推薦茶\n"
            f'  {{"trigger":"咖啡缺失","alternative":"茶","condition":"茶存在","confidence":0.85,"source":"episodic_summary"}}\n\n'
            f"Add the rule to the appropriate section. Keep all existing content."
        )
        updated = _call_llm(self.ollama_url, self.model_name, FILL_GAP_SYSTEM, user_prompt)
        if updated:
            valid, reason = validate_skill(updated)
            if valid:
                self._save(user_id, updated)
                # 同步到 FAISS 技能切片
                self._chunk_skill_md(updated, user_id)
                logger.info(f"[fill_gap] New rule generated for '{missing}' → "
                            f"alternative='{episodic_summary.get('alternative','')}' "
                            f"({user_id})")
                return updated
            logger.warning(f"[SkillManager] fill_gap validation failed: {reason}")
        return skill_md

    def _get_episodic_alternative(self, user_id: str, missing: str) -> dict:
        """
        查詢 Episodic Summary（Layer 2），找出使用者上次遇到缺少情況時的替代選擇
        這是「咖啡→茶」場景的核心查詢（論文 §4.3）
        """
        # 查 observation_logs：找 interacting_items 裡有替代品的紀錄
        # 例：missing="coffee" → 找使用者要咖啡但選了茶的紀錄
        missing_lower = missing.lower()

        # 方法一：查 episodic_summaries collection（如果有）
        episodic_docs = list(self.db.episodic_summaries.find(
            {"user": user_id},
            {"original_request":1,"chosen_alternative":1,"count":1,"timestamp":1}
        ).sort("count", -1).limit(5))

        for doc in episodic_docs:
            if missing_lower in doc.get("original_request", "").lower():
                return {
                    "alternative": doc.get("chosen_alternative", ""),
                    "count":       doc.get("count", 1),
                    "source":      "episodic_summaries",
                }

        # 方法二：從 observation_logs 推斷（fallback）
        # 找 interacting_items 含有 missing 關鍵字附近的其他物品
        logs = list(self.db.observation_logs.find(
            {"user": user_id, "weight": {"$gte": SUPPORT_THRESHOLD}},
            {"action":1,"instance":1,"weight":1,"interacting_items":1}
        ).sort("weight", -1).limit(20))

        # 找最常出現但不是 missing 的物品（可能的替代品）
        item_counts = {}
        for log in logs:
            for item in log.get("interacting_items", []):
                item_l = item.lower()
                if missing_lower not in item_l:
                    item_counts[item_l] = item_counts.get(item_l, 0) + log.get("weight", 1)

        if item_counts:
            best_alt = max(item_counts, key=item_counts.get)
            return {
                "alternative": best_alt,
                "count":       item_counts[best_alt],
                "source":      "observation_logs_inference",
            }

        return {}

    def _check_alternative_available(self, alternative: str) -> str:
        """確認替代品目前在家（動態物件）"""
        if not alternative:
            return ""
        from datetime import timedelta
        cutoff = datetime.utcnow() - timedelta(hours=2)
        doc = self.db.dynamic_objects.find_one({
            "label": {"$regex": alternative, "$options": "i"},
            "last_seen": {"$gte": cutoff},
        })
        if doc:
            return f"{doc['label']} (on {doc.get('last_seen_on','?')} in {doc.get('room','?')})"
        # 放寬：不限時間
        doc = self.db.dynamic_objects.find_one({
            "label": {"$regex": alternative, "$options": "i"}
        })
        return doc["label"] if doc else ""

    # ── RelevanceGate ─────────────────────────────────────────────────────────
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
                self._chunk_skill_md(updated, user_id)  # 同步切片
                return updated
            logger.warning(f"[SkillManager] Update failed: {reason}")
        return current

    # ── Stale 檢查 ────────────────────────────────────────────────────────────
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

    # ── 夜間重構 ──────────────────────────────────────────────────────────────
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
                self._chunk_skill_md(refactored, user_id)  # 同步切片
                return refactored
        return current

    # ── 儲存 ──────────────────────────────────────────────────────────────────
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